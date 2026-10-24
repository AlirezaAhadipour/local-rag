import chromadb

client = chromadb.PersistentClient(path="vector_store")
collection = client.get_collection(name="pdf_chunks")

print(f"Collections: {collection.name}")
print(f"Number of stored records: {collection.count()}")

result = collection.get(limit=1, include=["documents", "metadatas"])

print(f"Document: {result['documents'][0][:200]}")
print(f"Metadata: {result['metadatas'][0]}")