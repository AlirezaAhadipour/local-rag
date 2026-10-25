import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# ------- Retrieval -------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="vector_store")
collection = client.get_collection(name="pdf_chunks")

question = "How do internal waves dissipate energy?"
query_embedding = model.encode(question, normalize_embeddings=True)

results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=10, include=["documents", "metadatas", "distances"])

for text, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"Source: {metadata['source']} | Pages: {metadata['pages']} | Cosine distance: {distance:.4f}")
    print(f"Text: {text[:500]}\n")



# ------- Reranker -------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

candidate_texts = results["documents"][0]
pairs = [(question, text) for text in candidate_texts]
scores = reranker.predict(pairs)
ranked_indices = scores.argsort()[::-1]

for rank, idx in enumerate(ranked_indices[:3], start=1):
    metadata = results["metadatas"][0][idx]
    text = results["documents"][0][idx]
    print(f"Rank: {rank} | Reranker score: {scores[idx]:.4f}")
    print(f"Source: {metadata['source']} | Pages: {metadata['pages']}")
    print(f"Text: {text[:500]}\n")