import streamlit as st
import os

from src.loader import load_pdf
from src.splitter import split_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, save_vector_store
from src.retriever import retrieve_documents
from src.llm import generate_answer


st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚"
)

st.title("📚 RAG PDF Chatbot")
st.write("Upload any PDF and ask questions about it.")


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    pdf_path = os.path.join(
        "data",
        uploaded_file.name
    )

    os.makedirs("data", exist_ok=True)

    with open(pdf_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success(f"PDF uploaded: {uploaded_file.name}")

    with st.spinner("Processing your PDF..."):

        text = load_pdf(pdf_path)

        chunks = split_text(text)

        embeddings = create_embeddings(chunks)

        index = create_vector_store(embeddings)

        save_vector_store(index, chunks)

    st.success(
        f"PDF processed successfully! {len(chunks)} chunks created."
    )

    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if question:

        with st.spinner("Searching the document..."):

            results = retrieve_documents(
                question,
                k=3
            )

            context = "\n\n".join(results)

            answer = generate_answer(
                question,
                context
            )

        st.subheader("💬 Answer")
        st.write(answer)