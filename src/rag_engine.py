import os
import re
import shutil
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
COLLECTION_NAME = "work_visa_consultancy"


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


def create_documents(uploaded_documents):
    return [
        Document(
            page_content=text,
            metadata={"source": file_name, "doc_type": "work_visa_reference"},
        )
        for file_name, text in uploaded_documents
        if text.strip()
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
        shutil.rmtree(directory)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(directory),
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def load_vectorstore(persist_directory: str = DB_DIR):
    directory = Path(persist_directory)
    if not (directory / "chroma.sqlite3").exists():
        return None
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(directory),
    )


# retrieval pipeline
def retrieve_context(vectorstore, query: str, k: int = 3):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context, docs


def run_career_coach(vectorstore, question: str):
    llm = get_llm()
    retrieval_query = (
        "Use the work visa consultancy reference documents to answer this question accurately. "
        f"Question: {question}"
    )
    context, source_docs = retrieve_context(vectorstore, retrieval_query, k=5)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a careful work visa consultancy assistant.
        Use only the provided reference documents. Do not invent visa rules, eligibility,
        timelines, fees, documents, legal advice, or government requirements.
        If the documents do not contain the answer, say that clearly and recommend
        checking the relevant official immigration authority or a qualified attorney.
        Explain which country, visa category, and date matter when relevant.

        CONTEXT:
        {context}

        USER QUESTION:
        {question}

        Give a clear, concise answer. Separate documented facts from uncertainty.
        Never present the response as a substitute for professional legal advice.
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

__all__ = [
    "build_vectorstore",
    "create_documents",
    "evaluate_answer",
    "load_vectorstore",
    "run_career_coach",
    "split_documents",
]






