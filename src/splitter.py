from langchain_text_splitters import RecursiveCharacterTextSplitter
def split_text(text):
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50 )
        chunks = splitter.split_text(text)
        return chunks
if __name__ == "__main__":

    sample_text = """
    Python is a programming language.
    Python is easy to learn.
    Variables store data.
    Lists store multiple values.
    Functions are reusable blocks of code.
    Classes are used in object oriented programming.
    """

    chunks = split_text(sample_text)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)