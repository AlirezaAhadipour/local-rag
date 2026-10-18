from pathlib import Path
import pymupdf



def load_pdf(pdf_path):
    pages = []

    with pymupdf.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            pages.append({
                "source": Path(pdf_path).name,
                "page": page_num,
                "text": page.get_text()
            })

    return pages



def load_all_pdfs(folder_path):
    all_pages = []

    pdf_files = sorted(Path(folder_path).glob("*.pdf"))

    for pdf_path in pdf_files:
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)

    return all_pages



if __name__ == "__main__":
    
    pages = load_all_pdfs("data/pdfs")
    print(f"Total pages extracted: {len(pages)}")

    for page in pages:
        print(f"Source: {page['source']} | Page: {page['page']} | Characters: {len(page['text'])}")