from pathlib import Path
from src.pdf_loader import load_pdf


def prepare_document(pages):
    words_with_pages = []

    for page in pages:
        for word in page["text"].split():
            words_with_pages.append({"word": word, "page": page["page"]})

    return words_with_pages



def chunk_document(words_with_pages, chunk_size=200, overlap=40):
    if (chunk_size <= 0) or not (0 <= overlap < chunk_size):
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words_with_pages), step):
        window = words_with_pages[start: start + chunk_size]

        if not window:
            break

        chunks.append({
            "text": " ".join(item["word"] for item in window),
            "pages": sorted(set(item["page"] for item in window))
        })

        if start + chunk_size >= len(words_with_pages):
            break

    return chunks



# if __name__ == "__main__":
#     pages = [
#         {"page": 1, "text": "The Navier-Stokes equations are a set of mathematical formulas that describe"},
#         {"page": 2, "text": "exactly how fluids behave and travel through space."}
#     ]

#     words = prepare_document(pages)
#     chunks = chunk_document(words, chunk_size=6, overlap=2)

#     for i, chunk in enumerate(chunks, start=1):
#         print(f"Chunk: {i}, {chunk['text']}")
#         print(f"Pages: {chunk['pages']}\n")



def chunk_all_pdfs(folder_path, chunk_size=200, overlap=40):
    all_chunks = []

    for pdf_path in sorted(Path(folder_path).glob("*.pdf")):
        pages = load_pdf(pdf_path)
        words = prepare_document(pages)
        chunks = chunk_document(words, chunk_size, overlap)

        for i, chunk in enumerate(chunks, start=1):
            all_chunks.append({"source": pdf_path.name, "chunk_id": i, **chunk})

    return all_chunks



if __name__ == "__main__":
    chunks = chunk_all_pdfs("data/pdfs", chunk_size=200, overlap=40)
    print(f"Total chunks created: {len(chunks)}")

    for chunk in chunks[:3]:
        print(f"Source: {chunk['source']} | Chunk ID: {chunk['chunk_id']} | Pages: {chunk['pages']}")
        print(f"Text preview: {chunk['text'][:200]}\n")


    cross_page_chunks = [chunk for chunk in chunks if len(chunk["pages"]) > 1]
    print(f"Chunks spanning multiple pages: {len(cross_page_chunks)}")
    for chunk in cross_page_chunks[:3]: print(f"Source: {chunk['source']} | Chunk ID: {chunk['chunk_id']} | Pages: {chunk['pages']}")
