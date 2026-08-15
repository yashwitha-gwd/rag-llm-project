import faiss
import numpy as np
import pickle
import os


def create_vector_store(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    return index


def save_vector_store(index, chunks):

    # Create vector_db folder if it doesn't exist
    os.makedirs("vector_db", exist_ok=True)

    # Save FAISS index
    faiss.write_index(
        index,
        "vector_db/faiss.index"
    )

    # Save chunks
    with open(
        "vector_db/chunks.pkl",
        "wb"
    ) as file:

        pickle.dump(chunks, file)