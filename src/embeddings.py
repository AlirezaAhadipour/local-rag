from sentence_transformers import SentenceTransformer
from src.chunker import chunk_all_pdfs

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunks = chunk_all_pdfs("data/pdfs", chunk_size=200, overlap=40)
texts = [chunk["text"] for chunk in chunks]

tokenizer = model.tokenizer
token_lengths = [len(tokenizer.encode(text, add_special_token=True, truncation=False)) for text in texts]

print(f"Minimum tokens: {min(token_lengths)}")
print(f"Average tokens: {sum(token_lengths) / len(token_lengths):.1f}")
print(f"Maximum tokens: {max(token_lengths)}")
print(f"Chunks exceeding {model.max_seq_length} tokens: {sum(n > model.max_seq_length for n in token_lengths)}")


embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)

print(f"Number of chunks: {len(chunks)}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"First chunk source: {chunks[0]['source']}")
print(f"First embedding (first 10 dimensions): {embeddings[0][:10]}")