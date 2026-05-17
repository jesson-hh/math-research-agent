"""PDF -> plain text via PyMuPDF (fitz).

Best-effort extraction. Returns concatenated page text. Page boundaries
preserved with double newlines. Caller can detect empty result and fall
back to abstract.
"""

from pathlib import Path

import fitz  # PyMuPDF


def extract_text(pdf_path: Path | str) -> str:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    chunks = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            chunks.append(page.get_text("text"))
    return "\n\n".join(chunks).strip()
