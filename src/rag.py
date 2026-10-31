from src.retriever import retrieve_chunks
from src.generator import generate_answer


def rag(question):
    chunks = retrieve_chunks(question, n_candidates=10, top_k=3)
    answer = generate_answer(question, chunks)

    return answer


if __name__ == "__main__":
    question = "How do internal waves dissapitate energy?"
    answer = rag(question)
    print(f"Question: {question}\n")
    print(f"Answer: \n{answer}")
    