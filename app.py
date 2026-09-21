from __future__ import annotations

import streamlit as st

from src.file_utils import read_uploaded_file
from src.rag_engine import (
    create_documents,
    split_documents,
    build_vectorstore,
    run_career_coach,
    evaluate_answer,
)

st.set_page_config(
    page_title="AI Career Coach - RAG Project",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 42px; font-weight: 800; margin-bottom: 0px;}
    .subtitle {font-size: 18px; color: #666; margin-bottom: 25px;}
    .stage-box {padding: 14px; border-radius: 12px; background: #f7f7f7; border: 1px solid #e8e8e8; margin-bottom: 10px;}
    .success-box {padding: 12px; border-radius: 10px; background: #eaffea; border: 1px solid #8de28d;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎯 AI Career Coach using Traditional RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Resume + Job Description → RAG Pipeline → Skill Gap Analysis, Resume Suggestions & Interview Prep</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ RAG Settings")
    chunk_size = st.slider("Chunk size", 300, 1500, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 50)
    st.divider()
    st.markdown("### RAG Stages")
    st.markdown("1. Load Resume & JD")
    st.markdown("2. Split into Chunks")
    st.markdown("3. Convert to Embeddings")
    st.markdown("4. Store in ChromaDB")
    st.markdown("5. Retrieve Relevant Context")
    st.markdown("6. Generate Career Advice")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Resume")
    resume_file = st.file_uploader(
        "Upload resume (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        key="resume",
    )

with col2:
    st.subheader("💼 Upload Job Description")
    jd_file = st.file_uploader(
        "Upload JD (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        key="jd",
    )

resume_text = read_uploaded_file(resume_file) if resume_file else ""
jd_text = read_uploaded_file(jd_file) if jd_file else ""

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

st.divider()

if st.button("🚀 Build Career Coach RAG Index", type="primary"):
    if not resume_text.strip() or not jd_text.strip():
        st.error("Please upload or paste both Resume and Job Description.")
    else:
        with st.spinner("Running RAG stages: loading → chunking → embeddings → vector database..."):
            docs = create_documents(resume_text, jd_text)
            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            vectorstore = build_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = chunks
            st.session_state.chat_history = []

        st.success("RAG index created successfully!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", "2")
        c2.metric("Chunks", len(st.session_state.chunks))
        c3.metric("Vector DB", "ChromaDB")

if st.session_state.vectorstore:
    st.markdown("### 💬 Career Coach Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("accuracy_score") is not None:
                st.caption(
                    f"Accuracy: {message['accuracy_score']}% - "
                    f"{message['accuracy_reason']}"
                )

    question = st.chat_input("Ask a question about your documents")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                answer, _ = run_career_coach(
                    st.session_state.vectorstore,
                    resume_text,
                    jd_text,
                    question,
                )
                try:
                    accuracy_score, accuracy_reason = evaluate_answer(
                        st.session_state.vectorstore,
                        question,
                        answer,
                    )
                except Exception:
                    accuracy_score, accuracy_reason = None, "Accuracy check unavailable."

            st.write(answer)
            if accuracy_score is not None:
                st.caption(f"Accuracy: {accuracy_score}% - {accuracy_reason}")

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "accuracy_score": accuracy_score,
                "accuracy_reason": accuracy_reason,
            }
        )

else:
    st.info("Upload both documents, then click 'Build Career Coach RAG Index' to start chatting.")

st.divider()
st.markdown("### 🧠 How this Traditional RAG Project Works")
st.markdown(
    """
1. **Load documents:** Resume and Job Description are converted into LangChain `Document` objects.  
2. **Chunking:** Large text is split using `RecursiveCharacterTextSplitter`.  
3. **Embeddings:** Each chunk is converted into a numerical vector using HuggingFace embeddings.  
4. **Vector Database:** Chunks and embeddings are stored inside ChromaDB.  
5. **Query Embedding:** User question is also converted into an embedding.  
6. **Similarity Search:** ChromaDB finds the most relevant resume/JD chunks.  
7. **LLM Generation:** Groq Llama model generates the final career guidance using only retrieved context.  
"""
)
