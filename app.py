import streamlit as st
import os

from src.loader import load_pdf
from src.splitter import split_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, save_vector_store
from src.retriever import retrieve_documents
from src.llm import generate_answer
import streamlit as st

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f5f3ff, #eef2ff, #f0fdfa);
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #7c3aed, #2563eb, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 19px;
    color: #64748b;
    margin-bottom: 35px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.85);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Answer */
.answer-card {
    background: linear-gradient(135deg, #ede9fe, #dbeafe);
    padding: 25px;
    border-radius: 20px;
    border-left: 6px solid #7c3aed;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
}

/* Section headings */
.section-title {
    color: #4c1d95;
    font-size: 24px;
    font-weight: 700;
}

/* Button */
.stButton > button {
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    font-weight: 600;
    padding: 10px 25px;
}

.stButton > button:hover {
    transform: scale(1.03);
    transition: 0.2s;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.8);
    border: 2px dashed #8b5cf6;
    border-radius: 18px;
    padding: 15px;
}

/* Text input */
[data-testid="stTextInput"] input {
    border-radius: 12px;
    border: 2px solid #c4b5fd;
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚"
)


st.markdown(
    '<div class="main-title">📚 RAG PDF Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload any PDF and chat with your document using AI ✨</div>',
    unsafe_allow_html=True
)



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
            st.markdown(
    '<div class="section-title">💡 AI Answer</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="answer-card">
        {answer}
    </div>
    """,
    unsafe_allow_html=True
)

       