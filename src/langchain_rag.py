from langchain_ollama import ChatOllama



# load PDF documents
from pathlib import Path
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

pdf_folder = Path("data/pdfs")
documents = []

for pdf_file in pdf_folder.glob("*.pdf"):
    loader = PyMuPDF4LLMLoader(str(pdf_file), mode='page')
    documents.extend(loader.load())

# print(f"Total pages loaded: {len(documents)}")
# print(f"First page metadata: {documents[0].metadata}")
# print(f"First 300 characters:\n{documents[0].page_content[:300]}")



# split into chunks
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=200, chunk_overlap=40)
chunks = text_splitter.split_documents(documents)

# print(f"Total chunks created: {len(chunks)}")
# print(f"First chunk text:\n{chunks[0].page_content}")
# print(f"First chunk metadata: {chunks[0].metadata}")



# generate embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})



# store in vector db
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model, collection_name="langchain_pdf_chunks", persist_directory="langchain_vector_store")
