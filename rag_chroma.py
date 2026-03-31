import os
import warnings
from dotenv import load_dotenv
from pathlib import Path

# Silenciar avisos sobre futuras versiones o procesos internos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

CHROMA_PATH = './chroma_db'
COLLECTION_NAME = 'cafe_aurora'

def load_documents(directory="./docs"):
  """Carga todos los documentos TXT de un directorio"""
  print(f"📄 Cargando documentos de {directory}...")
  loader = DirectoryLoader(
    directory,
    glob="**/*txt",
    loader_cls=TextLoader,
  )
  documents = loader.load()
  print(f"✅ Se cargaron {len(documents)} documentos.")
  return documents

def split_documents(documents):
  """Divide documentos en chunks manejables."""
  print("✂️ Dividiendo documentos en chunks...")
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
  )
  chunks = splitter.split_documents(documents)
  print(f"✅ Se dividieron los documentos en {len(chunks)} chunks.")
  return  chunks

def get_embeddings():
  """Obtiene el LLM."""
  return ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
    base_url=os.getenv("OPENAI_BASE_URL") or None
  )

def load_or_create_vector_store(chunks):
  """Carga vector store de ChromaDB o lo crea desde cero."""
  embeddings = get_embeddings()

  if Path(CHROMA_PATH).exists() and any(Path(CHROMA_PATH).iterdir()):
    print(f"📂 Cargando vector store existente de ChromaDB...")
    vector_store = Chroma(
      persist_directory=CHROMA_PATH,
      embedding_function=embeddings,
      collection_name=COLLECTION_NAME
    )
    print(f"✅ Vector store cargado ({vector_store._collection.count()})")
  else:
    print(f"🔢 Creando embeddings para {len(chunks)} chunks...")
    vector_store = Chroma.from_documents(
      documents=chunks,
      embedding=embeddings,
      persist_directory=CHROMA_PATH,
      collection_name=COLLECTION_NAME
    )
    print(f"✅ ChromaDB Vector store creado y guardado en {CHROMA_PATH}")

  return vector_store

def create_qa_chain(vector_store):
  """Crea QA Chain con RAG y prompt anti-hallucination."""
  print(f"🔗 Creando QA chain...")
  llm = get_llm()
  retriever = vector_store.as_retriever(search_kwargs={"k": 4})

  system_prompt = (
    "Usa SOLAMENTE el siguiente contexto para responder la pregunta."
    "Si la información no está en el contexto, responde: "
    "'No tengo esa información en los documentos proporcionados.'"
    "NO inventes información. NO uses tu conocimiento general."
    "Cuando cites información, menciona de qué parte del contexto proviene.\n\n"
    "Contexto:\n{context}"
  )

  prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
  ])

  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  rag_chain = create_retrieval_chain(retriever, question_answer_chain)

  print(f"✅ Sistema RAG listo\n.")
  return rag_chain

def ask_question(rag_chain, question, show_chunks=False):
  """Hace una pregunta al sistema RAG."""
  print(f"\n❓ Pregunta: {question}\n")
  result = rag_chain.invoke({ "input": question })
  print(f"💬 Respuesta:\n{result["answer"]}\n")

  if result.get("context"):
    print(f"📚 Fuentes utilizadas:")
    seen_sources = set()
    for doc in result["context"]:
      source = doc.metadata.get("source", "Desconocida")
      if source not in seen_sources:
        seen_sources.add(source)
        print(f" - {source}")

      if show_chunks:
        print(f"\n📄 Chunks recuperados (debbuging):")
        for i, doc in enumerate(result["context"], 1):
          print(f"\n  --- Chunk {i} ---")
          print(f"  Fuente: {doc.metadata.get('source', 'N/A')}")
          print(f"  Texto: {doc.page_content[:150]}...")

def main():
  """Flujo principal del script RAG con ChromaDB."""
  print("🚀 Iniciando sistema RAG con ChromaDB...\n")

  documents = load_documents("./docs")
  if not documents:
    print("❌ No se encontraron documentos en ./docs")
    return
    
  chunks = split_documents(documents)
  vector_store = load_or_create_vector_store(chunks)
  rag_chain = create_qa_chain(vector_store)

  print("Escribe 'exit' para terminar.\n")
  while True:
    try:
      question = input("🔍 Tu pregunta: ").strip()
    except (EOFError, KeyboardInterrupt):
      print("\n👋 ¡Bye!")
      break

    if question.lower() in ["salir", "exit", "quit", ""]:
      print("\n👋 ¡Hasta luego!")
      break

    ask_question(rag_chain, question)  
  
if __name__ == "__main__":
  main()
