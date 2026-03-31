import chromadb

print("Probando ChromaDB...")

client = chromadb.Client()

collection = client.get_or_create_collection("test")

collection.add(
  documents=[
    "Café Aurora fue fundada en 2018",
    "El latte de mazapán cuesta $72 MN",
    "Python es un lenguaje de programación"
  ],
  ids=["doc1", "doc2", "doc3"]
)

results = collection.query(
  query_texts=["¿Cuánto cuesta el café?"],
  n_results=2
)

print(f"✅ ChromaDB funciona correctamente")
print(f"Total de documentos en colección: {collection.count()}")
print(f"Resultado 1: {results['documents'][0][0]}")
print(f"Distancia 1: {results['distances'][0][0]:.4f}")
print(f"Resultado 2: {results['documents'][0][1]}")
print(f"Distancia 2: {results['distances'][0][1]:.4f}")
