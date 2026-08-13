from src.loader import load_pdf
from src.splitter import split_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, save_vector_store
from src.retriever import retrieve_documents
from src.llm import generate_answer


def build_rag_pipeline(pdf_path):

    text = load_pdf(pdf_path)

    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    index = create_vector_store(embeddings)

    save_vector_store(index, chunks)

    print("RAG data pipeline built successfully!")
    print("Number of chunks:", len(chunks))


def ask_question(question):

    results = retrieve_documents(question, k=3)

    context = "\n\n".join(results)

    answer = generate_answer(question, context)

    return answer


if __name__ == "__main__":

    question = input("Ask a question about your PDF: ")

    answer = ask_question(question)

    print("\nAnswer:")
    print(answer)