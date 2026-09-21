from __future__ import annotations

import tempfile
from pathlib import Path

import docx2txt
from pypdf import PdfReader


def read_uploaded_file(uploaded_file) -> str:
    """Read txt, pdf, or docx uploaded from Streamlit."""
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            reader = PdfReader(tmp_path)
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text)

        if suffix == ".docx":
            return docx2txt.process(tmp_path) or ""

        raise ValueError("Unsupported file type. Please upload .txt, .pdf, or .docx")
    finally:
        path = Path(tmp_path)
        if path.exists():
            path.unlink()
