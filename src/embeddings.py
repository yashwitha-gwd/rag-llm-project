from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings
if __name__ == "__main__":

    sample_chunks = [
        "Python is a programming language.",
        "Lists store multiple values.",
        "Functions are reusable blocks of code."
    ]

    embeddings = create_embeddings(sample_chunks)

    print("Number of chunks:", len(embeddings))
    print("Vector dimensions:", len(embeddings[0]))