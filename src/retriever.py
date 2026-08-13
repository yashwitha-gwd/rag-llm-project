import faiss
import pickle
from src.embeddings import model


index = faiss.read_index("vector_db/faiss.index")


with open("vector_db/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


def retrieve_documents(query, k=3):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        query_embedding.astype("float32"),
        k
    )

    results = [chunks[i] for i in indices[0]]

    return results
if __name__ == "__main__":

    query = "What is a variable in Python?"

    results = retrieve_documents(query, k=3)

    for i, result in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(result)