import chromadb

client = chromadb.PersistentClient(path="./chroma_test")

print(f"✅ Cliente ChromaDB creado (persistente en ./chroma_test)")
print(f"📊 Colecciones existentes: {client.list_collections()}")

collection = client.get_or_create_collection(
  name="cafe_docs",
  metadata={"description": "Documentos de Café Aurora"}
)

print(f"\n📁 Colección creada: {collection.name}")
print(f"📊 Documentos en colección: {collection.count()}")

collection.add(
  documents=[
    "Café Aurora fue fundado en 2018 en la Ciudad de México por los hermanos Espinoza.",
    "El latte de mazapán cuesta $72 MXN y es el más popular del menú.",
    "Los empleados del primer año tienen 12 días de vacaciones.",
    "El servicio a domicilio cubre un radio de 5 km desde cada sucursal.",
    "El turno matutino es de 6:30 AM a 2:30 PM."
  ],
  metadatas=[
    {"category": "historia", "sucursal": "todas"},
    {"category": "menu", "sucursal": "todas"},
    {"category": "politicas", "sucursal": "todas"},
    {"category": "pedidos", "sucursal": "todas"},
    {"category": "politicas", "sucursal": "todas"},
  ],
  ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

print(f"✅ {collection.count()} documentos agregados")

print("\n🔍 --- BÚSQUEDAS ---")

query = "¿Cuánto cuesta el café?"
results = collection.query(
  query_texts=[query],
  n_results=2
)

print(f"\nBúsqueda: '{query}'")
for i, (doc, distance, metadata) in enumerate(
  zip (
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
  )
):
  print(f". {i+1}. [{distance:.4f}] {doc}")
  print(f"     Metadata: {metadata}")

print("\n🔍 --- BÚSQUEDA CON FILTRO ---")

query = "horarios"
results_filtered = collection.query(
  query_texts=[query],
  n_results=3,
  where={"category": "politicas"}
)

print(f"\nBúsqueda: '{query} (solo categoría 'políticas')'")
for i, (doc, distance) in enumerate(zip(
  results_filtered['documents'][0],
  results_filtered['distances'][0]
)):
  print(f". {i+1}. [{distance:.4f}] {doc}")

print("\n🔄 --- CRUD: UPDATE ---")

collection.update(
  ids=["doc2"],
  documents=["El latte de mazapán ahora cuesta $85 MXN (precio actualizado 2026)."],
  metadatas=[{"category": "menu", "sucursal": "todas", "updated": "2026-03"}]
)

updated_doc = collection.get(ids=["doc2"])
print(f"Documento actualizado: {updated_doc['documents'][0]}")
print(f"Metadata: {updated_doc['metadatas'][0]}")

print("\n🗑️ --- CRUD: DELETE ---")

print(f"Antes de eliminar: {collection.count()} documentos")
collection.delete(ids=["doc5"])
print(f"Después de eliminar: {collection.count()} documentos")

print("\n📊 --- ESTADÍSTICAS ---")

print(f"Total documentos: {collection.count()}")

peek = collection.peek(limit=2)
print(f"\nPrimeros 2 documentos (peek):")
for i, (doc_id, doc_text) in enumerate(zip(
  peek['ids'],
  peek['documents']
)):
  print(f"  {i+1}. [{doc_id}] {doc_text[:60]}...")

print("\n📁 --- MÚLTIPLES COLECCIONES ---")

collection_menu = client.get_or_create_collection("menu")
collection_politicas = client.get_or_create_collection("politicas")

collection_menu.add(
  documents=["Latte $72", "Cold brew $60", "Matcha $78"],
  ids=["m1", "m2", "m3"]
)

collection_politicas.add(
  documents=["12 días de vacaciones", "Turno matutino 6:30 AM"],
  ids=["p1", "p2"]
)

print(f"Colección menú: {collection_menu.count()} documentos")
print(f"Colección políticas: {collection_politicas.count()} documentos")

menu_results = collection_menu.query(query_texts=["bebidas"], n_results=2)
print(f"\nBúsqueda 'bebidas' en menu: {menu_results['documents'][0]}")

politicas_results = collection_politicas.query(query_texts=['bebidas'], n_results=2)
print(f"\nBúsqueda 'bebidas' en políticas: {politicas_results['documents'][0]}")
