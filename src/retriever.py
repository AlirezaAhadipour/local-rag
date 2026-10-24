import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="vector_store")
collection = client.get_collection(name="pdf_chunks")

question = "How do internal waves dissipate energy?"
query_embedding = model.encode(question, normalize_embeddings=True)

results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=10, include=["documents", "metadatas", "distances"])

for text, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"Source: {metadata['source']} | Pages: {metadata['pages']} | Cosine distance: {distance:.4f}")
    print(f"Text: {text[:500]}\n")