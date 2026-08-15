import streamlit as st
import os

from src.loader import load_pdf
from src.splitter import split_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, save_vector_store
from src.retriever import retrieve_documents
from src.llm import generate_answer


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide"
)


# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

/* General text */
.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div {
    color: #2E1A47;
}
[data-testid="stFileUploader"] {
    background: #FCFAFF;
    border: 2px dashed #A78BFA;
    border-radius: 18px;
    padding: 15px;
}

[data-testid="stFileUploader"] * {
    color: #3B2754 !important;
}

[data-testid="stTextInput"] input {
    border-radius: 12px;
    border: 2px solid #C4B5FD;
    background: #FFFFFF;
    color: #2E1A47 !important;
    padding: 12px;
}

[data-testid="stTextInput"] input::placeholder {
    color: #8B7A9E !important;
}




.stApp {
    background: linear-gradient(
        135deg,
        #F8F5FF,
        #F0E8FF,
        #EDE9FE
    );
    color : #2E1A47;
}


/* Main title */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #8B5CF6,
        #A78BFA,
        #C4B5FD
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 5px;
}


/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 19px;
    color: #6B5B7A;
    margin-bottom: 35px;
}


/* Cards */
.card {
    background: rgba(255, 255, 255, 0.85);
    padding: 25px;
    border-radius: 20px;

    box-shadow:
        0px 8px 25px rgba(124, 58, 237, 0.08);

    margin-bottom: 20px;
}

.answer-card {
    background: #F3E8FF;
    padding: 25px;
    border-radius: 20px;
    border-left: 6px solid #8B5CF6;
    box-shadow: 0px 8px 25px rgba(124, 58, 237, 0.10);
    color: #241333 !important;
    font-size: 17px;
    line-height: 1.7;
}

   
    


/* Section headings */
.section-title {

    color: #6D28D9;

    font-size: 24px;

    font-weight: 700;

    margin-top: 20px;

    margin-bottom: 10px;
}


/* Button */
.stButton > button {

    border-radius: 12px;

    border: none;

    background: linear-gradient(
        90deg,
        #8B5CF6,
        #A78BFA
    );

    color: white;

    font-weight: 600;

    padding: 10px 25px;
}


.stButton > button:hover {

    background: #7C3AED;

    transform: scale(1.03);

    transition: 0.2s;
}


/* File uploader */
[data-testid="stFileUploader"] {

    background: rgba(
        255,
        255,
        255,
        0.8
    );

    border: 2px dashed #A78BFA;

    border-radius: 18px;

    padding: 15px;
}


/* Text input */
[data-testid="stTextInput"] input {

    border-radius: 12px;

    border: 2px solid #DDD6FE;

    background: #FCFAFF;

    padding: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# TITLE
# ==============================

st.markdown(
    '<div class="main-title">📚 RAG PDF Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload any PDF and chat with your document using AI ✨'
    '</div>',
    unsafe_allow_html=True
)


# ==============================
# PDF UPLOAD
# ==============================

st.markdown(
    '<div class="section-title">📤 Upload Your PDF</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)


# ==============================
# PROCESS PDF
# ==============================

if uploaded_file is not None:

    os.makedirs("data", exist_ok=True)

    pdf_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as file:

        file.write(
            uploaded_file.getbuffer()
        )

    st.success(
        f"📄 PDF uploaded: {uploaded_file.name}"
    )


    with st.spinner(
        "🔄 Processing your PDF..."
    ):

        text = load_pdf(pdf_path)

        chunks = split_text(text)

        embeddings = create_embeddings(
            chunks
        )

        index = create_vector_store(
            embeddings
        )

        save_vector_store(
            index,
            chunks
        )


    st.success(
        f"✅ PDF processed successfully! "
        f"{len(chunks)} chunks created."
    )


    # ==============================
    # QUESTION
    # ==============================

    st.markdown(
        '<div class="section-title">'
        '💬 Ask Your Question'
        '</div>',
        unsafe_allow_html=True
    )

    question = st.text_input(
        "Ask something about your PDF:",
        placeholder="Example: What is the main topic of this document?"
    )


    # ==============================
    # RAG QUESTION ANSWERING
    # ==============================

    if question:

        with st.spinner(
            "🔍 Searching your document..."
        ):

            results = retrieve_documents(
                question,
                k=3
            )

            context = "\n\n".join(
                results
            )

            answer = generate_answer(
                question,
                context
            )


        # ==============================
        # DISPLAY ANSWER
        # ==============================

        st.markdown(
            '<div class="section-title">'
            '💡 AI Answer'
            '</div>',
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