import faiss
import pickle
from src.embeddings import model


def retrieve_documents(query, k=3):

    # Load FAISS index
    index = faiss.read_index(
        "vector_db/faiss.index"
    )

    # Load chunks
    with open(
        "vector_db/chunks.pkl",
        "rb"
    ) as file:

        chunks = pickle.load(file)

    # Create embedding for user's question
    query_embedding = model.encode(
        [query]
    )

    # Search FAISS
    distances, indices = index.search(
        query_embedding.astype("float32"),
        k
    )

    # Get relevant chunks
    results = []

    for i in indices[0]:

        if i < len(chunks):
            results.append(chunks[i])

    return results


if __name__ == "__main__":

    query = "What is a variable in Python?"

    results = retrieve_documents(
        query,
        k=3
    )

    for i, result in enumerate(results):

        print(
            f"\n--- Result {i + 1} ---"
        )

        print(result)