from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# text = "Internal waves transport energy through the ocean."
# embedding = model.encode(text)
# print(f"Embedding shape: {embedding.shape}")
# print(f"First few values: {embedding[:10]}")



sentences = ["Internal waves transport energy through the ocean.", 
             "Oceanic internal waves transfer energy across the water column.", 
             "Machine learning models predict customer purchases."]

embeddings = model.encode(sentences, normalize_embeddings=True)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Similarity A-B: {embeddings[0] @ embeddings[1]:.4f}")
print(f"Similarity A-C: {embeddings[0] @ embeddings[2]:.4f}")