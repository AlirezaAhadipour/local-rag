from sentence_transformers import SentenceTransformer
from src.chunker import chunk_all_pdfs

chunks = chunk_all_pdfs("data/pdfs", chunk_size=200, overlap=40)
texts = [chunk["text"] for chunk in chunks]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)

print(f"Number of chunks: {len(chunks)}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"First chunk source: {chunks[0]['source']}")
print(f"First embedding (first 10 dimensions): {embeddings[0][:10]}")