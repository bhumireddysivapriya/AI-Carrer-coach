import os
import re
import shutil
import uuid
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DB_DIR = "chroma_db"


# loading of model
def get_llm(model: str | None = None, temperature: float = 0):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(model=model, temperature=temperature, api_key=api_key)


# loading of embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# loading of text files
def load_text_file(file_path: str, source_name: str, doc_type: str):
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": source_name, "doc_type": doc_type})]


def create_documents(resume_text: str, jd_text: str):
    return [
        Document(page_content=resume_text, metadata={"source": "uploaded_resume", "doc_type": "resume"}),
        Document(page_content=jd_text, metadata={"source": "uploaded_resume", "doc_type": "job_description"}),
    ]


def split_documents(docs: List[Document], chunk_size=800, chunk_overlap=150, separators=None):
    if separators is None:
        separators = ["\n\n", "\n", ".", " "]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks: List[Document], persist_directory: str = DB_DIR):
    directory = Path(persist_directory)
    if directory.exists():
        try:
            shutil.rmtree(directory)
        except PermissionError:
            directory = directory.with_name(f"{directory.name}_{uuid.uuid4().hex[:8]}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(directory),
        collection_name="career_coach_rag",
    )
    return vectorstore


# retrieval pipeline
def retrieve_context(vectorstore, query: str, k: int = 3):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context, docs


def run_career_coach(vectorstore, resume_text: str, jd_text: str, question: str):
    llm = get_llm()
    retrieval_query = (
        "Use the resume and job description content to answer this career question accurately. "
        f"Question: {question}"
    )
    context, source_docs = retrieve_context(vectorstore, retrieval_query, k=5)

    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert AI career coach for students, freshers, and working professionals.
        Use only the given context from the resume and job description.
        Do not invent skills, experience, or job requirements.

        CONTEXT:
        {context}

        USER QUESTION:
        {question}

        Give a clear, practical answer with these sections when relevant:
        1. Current Match Summary
        2. Strengths
        3. Missing skills / Gaps
        4. Recommended Improvements
        5. Suggested Projects
        6. Interview Preparation Tips

        Keep the answer simple, actionable, and beginner-friendly.
        """
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, source_docs


def evaluate_answer(vectorstore, question: str, answer: str):
    context, _ = retrieve_context(vectorstore, question, k=5)
    prompt = ChatPromptTemplate.from_template(
        """
        Evaluate whether the answer is accurate and supported by the provided resume and job description context.
        Do not reward information that is not present in the context. Consider factual support, relevance,
        and whether the answer avoids inventing skills, experience, or requirements.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        {answer}

        Return exactly this format:
        SCORE: <integer from 0 to 100>
        REASON: <one short sentence>
        """
    )
    evaluation = (prompt | get_llm() | StrOutputParser()).invoke(
        {"context": context, "question": question, "answer": answer}
    )
    score_match = re.search(r"SCORE\s*:\s*(\d{1,3})", evaluation, re.IGNORECASE)
    score = max(0, min(100, int(score_match.group(1)))) if score_match else None
    reason_match = re.search(r"REASON\s*:\s*(.+)", evaluation, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else evaluation.strip()
    return score, reason


def generate_complete_report(vectorstore, resume_text: str, jd_text: str):
    question = """
    Analyze this resume against this job description. Provide an ATS-style score,
    skill match summary, missing skills, resume improvement suggestions,
    project suggestions, and interview questions.
    """
    return run_career_coach(vectorstore, resume_text, jd_text, question)
