import chromadb
from sentence_transformers import SentenceTransformer
from src.chunker import chunk_all_pdfs_tokens



model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunks = chunk_all_pdfs_tokens(folder_path="data/pdfs", tokenizer=model.tokenizer, chunk_size=200, overlap=40)

ids = [f"{chunk['source']}_chunk_{chunk['chunk_id']}" for chunk in chunks]
documents = [chunk["text"] for chunk in chunks]
metadatas = [{"source": chunk["source"], "pages": ",".join(map(str, chunk["pages"]))} for chunk in chunks]
embeddings = model.encode(documents, batch_size=16, normalize_embeddings=True, show_progress_bar=True)


client = chromadb.PersistentClient(path="vector_store")
collection = client.get_or_create_collection(name="pdf_chunks", metadata={"hnsw:space": "cosine"})
collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

print(f"Total records stored: {collection.count()}")