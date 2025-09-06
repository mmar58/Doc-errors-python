from typing import List, Tuple
from PyPDF2 import PdfReader
import hashlib
import io

def extract_text_per_page(file_bytes: bytes) -> Tuple[List[str], str]:
    # Build PdfReader from a BytesIO buffer (PdfReader.from_bytes does not exist)
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text: List[str] = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages_text.append(t)

    # Title: first non-empty line of page 1
    doc_title = ""
    if pages_text:
        for line in pages_text[0].splitlines():
            if line.strip():
                doc_title = line.strip()
                break

    return pages_text, doc_title

def content_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]