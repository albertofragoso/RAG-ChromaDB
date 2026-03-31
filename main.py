from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime, timedelta

load_dotenv()

app = FastAPI(
  title="RAG API - Café Aurora",
  description="API de Q&A sobre documentos con ChromaDB y LangChain",
  version="1.0.0"
)

CHROMA_PATH = "./chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  model_kwargs={"device": "cpu"},
  encode_kwargs={"normalize_embeddings": True}
)

llm = ChatOpenAI(
  model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
  temperature=0,
  api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
  base_url=os.getenv("OPENAI_BASE_URL") or None
)

# === MODELOS PYDANTIC ===

class CollectionCreate(BaseModel):
  name: str = Field(..., min_length=3, max_length=50, description="Nombre de la colección")
  description: Optional[str] = Field(None, description="Descripción de la colección")

class DocumentAdd(BaseModel):
  text: str = Field(..., min_length=10, description="Texto del documento")
  metadata: Optional[Dict]= Field(default_factory=dict, description="Metada del documento")

class QueryRequest(BaseModel):
  query: str = Field(..., min_length=3, description="Pregunta sobre los documentos")
  k: int = Field(default=4, ge=1, le=20, description="Número de chunks a recuperar")
  filter: Optional[Dict] = Field(None, description="Filtro de metadata")

class QueryResponse(BaseModel):
  answer: str
  sources: List[Dict]
  collection: str
  query: str

class ChatRequest(BaseModel):
  session_id: str = Field(..., min_length=1, description="ID de la sesion de chat")
  message: str = Field(..., min_length=3, description="Mensaje del usuario")
  k: int = Field(default=4, ge=1, le=20, description="Número de chunks a recuperar")
  filter: Optional[Dict] = Field(None, description="Filtro de metadata")

class ChatResponse(BaseModel):
  session_id: str
  message: str
  response: str
  sources: List[Dict]

# === MEMORY MANAGER ===

class MemoryManager:
  """gestiona memoria conversacional para múltiples sesiones."""

  def __init__(self, max_age_minutes: int = 60):
    self._histories: Dict[str, InMemoryChatMessageHistory] = {}
    self._last_access: Dict[str, datetime] = {}
    self.max_age = timedelta(minutes=max_age_minutes)

  def get_history(self, session_id: str) -> InMemoryChatMessageHistory:
    """Obtiene o crea el historial para una sesión."""
    self._cleanup_old()

    if session_id not in self._histories:
      self._histories[session_id] = InMemoryChatMessageHistory()

    self._last_access[session_id] = datetime.now()
    return self._histories[session_id]

  def clear_session(self, session_id: str):
    """Limpia la memoria de una sesión"""
    self._histories.pop(session_id, None)
    self._last_access.pop(session_id, None)

  def active_sessions(self) -> int:
    """Número de sesioesn activas."""
    return len(self._histories)

  def _cleanup_old(self):
    """Elimina sesiones inactivas."""
    now = datetime.now()
    expired = [
      sid for sid, last in self._last_access.items()
      if now - last > self.max_age
    ]
    for sid in expired:
      self.clear_session(sid)

memory_manager = MemoryManager(max_age_minutes=60)
    

# === ENDPOINTS: HEALTH ===

@app.get("/", tags=["Health"])
def root():
  """Healt check - muestra estado de la API."""

  collections = chroma_client.list_collections()
  return {
    "status": "healthy",
    "service": "RAG API - Café Aurora",
    "collections": len(collections),
    "active_chat_sessions": memory_manager.active_sessions(),
    "chroma_path": CHROMA_PATH
  }

# === ENDPOINTS: COLECCIONES ===

@app.post("/collections", tags=["Collections"])
def create_collection(data: CollectionCreate):
  """Crea una nueva colección."""
  try:
    chroma_client.create_collection(
      name=data.name,
      metadata={"description": data.description or ""}
    )
    return {"message": f"Colección '{data.name}' creada", "name": data.name}
  except Exception as e:
    if 'already exists' in str(e).lower():
      raise HTTPException(409, f"La collección '{data.name}' ya existe")
    raise HTTPException(500, f"Error: {str(e)}")

@app.get("/collections", tags=["Collections"])
def list_collections():
  """Lista todas las colecciones."""
  collections = chroma_client.list_collections()
  return {
    "total": len(collections),
    "collections": [
      { 
        "name": collection.name, 
        "count": collection.count(), 
        "metadata": collection.metadata 
      }
      for collection in collections
    ]
  }

@app.delete("/collections/{name}", tags=["Collections"])
def delete_collection(name: str):
  """Elimina una colección y todos sus documentos."""
  try:
    chroma_client.delete_collection(name)
    return {"message": f"Colección '{name}' eliminada"}
  except Exception as e:
    raise HTTPException(404, f"Colección '{name}' no encontrada")

# === ENDPOINTS: DOCUMENTOS ===

@app.post("/collections/{name}/documents", tags=["Documents"])
def add_documents(name: str, documents: List[DocumentAdd]):
  """Agrega documentos a una colección con chunking automático."""
  try:
    chroma_client.get_collection(name)
  except Exception:
    raise HTTPException(404, f"Colección '{name}' no encontrada")

  splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
  )

  all_chunks = []
  all_metadatas = []
  all_ids = []

  for doc_idx, doc in enumerate(documents):
    chunks = splitter.split_text(doc.text)

    for chunk_idx, chunk_text in enumerate(chunks):
      chunk_id = f"doc{doc_idx}_chunk{chunk_idx}"
      chunk_metadata = doc.metadata.copy()
      chunk_metadata["doc_index"] = doc_idx
      chunk_metadata["chunk_index"] = chunk_idx

      all_chunks.append(chunk_text)
      all_metadatas.append(chunk_metadata)
      all_ids.append(chunk_id)

  vector_store = Chroma(
    client=chroma_client,
    collection_name=name,
    embedding_function=embeddings
  )

  from langchain_core.documents import Document

  langchain_docs = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(all_chunks, all_metadatas)
  ]
  vector_store.add_documents(langchain_docs, ids=all_ids)

  return {
    "message": f"{len(all_chunks)} chunks agregados a '{name}'",
    "documents_received": len(documents),
    "chunks_created": len(all_chunks)
  }

# === ENDPOINTS: QUERY ===

@app.post("/collections/{name}/query", response_model=QueryResponse, tags=["Query"])
def query_collection(name: str, request: QueryRequest):
  """Hace una pregunta sobre los documenos de una colección. (stateless)."""
  try:
    collection = chroma_client.get_collection(name)
    if collection.count() == 0:
      raise HTTPException(404, f"La colección '{name}' está vacía..")
  except HTTPException:
    raise
  except Exception:
    raise HTTPException(404, f"Colección '{name}, no encontrada'")

  vector_store = Chroma(
    client=chroma_client,
    collection_name=name,
    embedding_function=embeddings
  )

  search_kwargs = {"k": request.k}

  if request.filter:
    search_kwargs["filter"] = request.filter

  retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

  system_prompt = (
    "Usa SOLAMENTE el siguiente contexto para respondera pregunta."
    "Si la información no está en el contexto, responde:"
    "'No tengo esa información en los documentos proporcionados.'"
    "NO inventes información. NO uses tu conocimiento general."
    "Cuando cites información, mencioa de qué parte del conexto proviene.\n\n"
    "Contexto:\n{context}"
  )

  prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
  ])

  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  rag_chain = create_retrieval_chain(retriever, question_answer_chain)

  result = rag_chain.invoke({"input": request.query})

  sources = []
  seen = set()
  for doc in result.get("context", []):
    source = doc.metadata.get("source", "Desconocida")
    if source not in seen:
      seen.add(source)
      sources.append({
        "source": source,
        "preview": doc.page_content[:150] + '...'
      })

  return QueryResponse(
    answer=result["answer"],
    sources=sources,
    collection=name,
    query=request.query
  )

# === ENDPOINTS: STATS ===

@app.get("/collections/{name}/stats", tags=["Collections"])
def collection_stats(name: str):
  try:
    collection = chroma_client.get_collection(name)
  except Exception:
    raise HTTPException(404, f"Colección '{name}' no encontrada")

  all_docs = collection.get()
  sources = set()
  categories = {}

  for metadata in all_docs.get("metadatas", []):
    source = metadata.get("source", "unknown")
    sources.add(source)

    category = metadata.get("category", "uncategorized")
    categories[category] = categories.get(category, 0) + 1

  return {
    "name": name,
    "total_chunks": collection.count(),
    "unique_sources": len(sources),
    "sources": list(sources),
    "categories": categories
  }

# === ENDPOINTS: CHAT ===

@app.post("/collections/{name}/chat", response_model=ChatResponse, tags=["Chat"])
def chat_with_docs(name: str, request: ChatRequest):
  """
    Conversación con memoria sobre documentos.
    Usa session_id para mantener contexto entre mensajes.
  """
  try:
    collection = chroma_client.get_collection(name)
    if collection.count() == 0:
      raise HTTPException(404, f"La colección '{name}' está vacía")
  except HTTPException:
    raise
  except Exception:
    raise HTTPException(404, f"Colección '{name} no encontrada'")

  vector_store = Chroma(
    client=chroma_client,
    collection_name=name,
    embedding_function=embeddings
  )

  search_kwargs = {"k": request.k}

  if request.filter:
    search_kwargs["filter"] = request.filter

  retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
  relevant_docs = retriever.invoke(request.message)
  context = "\n\n".join([doc.page_content for doc in relevant_docs])

  history = memory_manager.get_history(request.session_id)

  prompt = ChatPromptTemplate.from_messages([
    ("system",
    "Eres un asistente que responde preguntas basándose en documentos."
    "Usa SOLO la siguiente información para responder."
    "Si la respuesta no está en los documentos, dilo.\n\n"
    "Documentos relevantes:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
  ])

  parser = StrOutputParser()
  chain = prompt | llm | parser

  response = chain.invoke({
    "context": context,
    "input": request.message,
    "history": history.messages
  })

  history.add_user_message(request.message)
  history.add_ai_message(response)

  seen = set()
  sources = []

  for doc in relevant_docs:
    source = doc.metadata.get("source", "Desconocida")  
    if source not in seen:
      seen.add(source)
      sources.append({
        "source": source,
        "preview": doc.page_content[:150] + '...'
      })

  return ChatResponse(
    session_id=request.session_id,
    message=request.message,
    response=response,
    sources=sources
  )

@app.delete("/collections/{name}/chat/{session_id}", tags=["Chat"])
def clear_chat(name: str, session_id: str):
  """Limpia la memoria de una sesión de chat."""
  memory_manager.clear_session(session_id)
  return { "message": f"Sesión '{session_id}' limpiada" }
