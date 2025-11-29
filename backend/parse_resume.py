# backend/parse_resume.py
import re
from io import BytesIO
from pdfminer.high_level import extract_text
from docx import Document

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    with BytesIO(file_bytes) as f:
        text = extract_text(f)
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    with BytesIO(file_bytes) as f:
        doc = Document(f)
        content = [p.text for p in doc.paragraphs]
    return "\n".join(content)

def parse_resume(file_name: str, file_bytes: bytes) -> dict:
    """Return a dict with extracted text and contact info."""
    if file_name.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_docx(file_bytes)

    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)

    return {
        "name": file_name,
        "file_name": file_name,
        "text": text,
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
    }
