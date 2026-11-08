from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# load PDF documents
from pathlib import Path

pdf_folder = Path("data/pdfs")
documents = []

for pdf_file in pdf_folder.glob("*.pdf"):
    loader = PyMuPDF4LLMLoader(str(pdf_file), mode='page')
    documents.extend(loader.load())

print(f"Total pages loaded: {len(documents)}")
print(f"First page metadata: {documents[0].metadata}")
print(f"First 300 characters:\n{documents[0].page_content[:300]}")