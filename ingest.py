"""
ingest.py
Load documents from data/docs and extract plain text.
Supports: .pdf, .txt, .docx, .csv
Returns list of dicts: {"id": filepath, "source": filename, "text": extracted_text}
"""
import os
from typing import List, Dict

# PDF
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
# DOCX
try:
    import docx
except Exception:
    docx = None
# CSV / table
import pandas as pd
# fallback pdfminer
from pdfminer.high_level import extract_text as pdfminer_extract

SUPPORTED = (".pdf", ".txt", ".docx", ".csv")


def extract_text_from_pdf(path: str) -> str:
    if fitz:
        try:
            doc = fitz.open(path)
            return "\n".join([p.get_text() for p in doc])
        except Exception:
            pass
    # fallback
    try:
        return pdfminer_extract(path)
    except Exception:
        return ""


def extract_text_from_docx(path: str) -> str:
    if docx:
        try:
            d = docx.Document(path)
            return "\n".join([p.text for p in d.paragraphs])
        except Exception:
            return ""
    return ""


def extract_text_from_csv(path: str) -> str:
    try:
        df = pd.read_csv(path, encoding="utf-8", errors="ignore")
        return df.to_csv(index=False)
    except Exception:
        return ""


def load_documents(folder: str = "data/docs") -> List[Dict]:
    docs = []
    if not os.path.exists(folder):
        print(f"[ingest] Create folder: {folder} and add files.")
        return docs
    for root, _, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in SUPPORTED:
                continue
            path = os.path.join(root, f)
            text = ""
            if ext == ".pdf":
                text = extract_text_from_pdf(path)
            elif ext == ".txt":
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                except Exception:
                    text = ""
            elif ext == ".docx":
                text = extract_text_from_docx(path)
            elif ext == ".csv":
                text = extract_text_from_csv(path)
            docs.append({"id": path, "source": f, "text": text})
    return docs


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")
    for d in docs[:5]:
        print(d["source"], "chars:", len(d["text"]))
