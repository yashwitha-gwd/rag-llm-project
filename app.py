import streamlit as st
import os
from src.loader import load_pdf
from src.splitter import split_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, save_vector_store
from src.retriever import retrieve_documents
from src.llm import generate_answer


st.set_page_config(
    page_title="RAG LLM Project",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Intelligent Document Chatbot")
st.write("Welcome! This project uses Retrieval-Augmented Generation (RAG) with an LLM to answer questions from your documents.")

st.success("Project setup is complete!")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    st.write("PDF uploaded successfully!")
    st.write(uploaded_file.name)