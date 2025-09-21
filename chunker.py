"""
chunker.py
Chunk a long text into overlapping character chunks.
Provides chunk_documents to convert list of doc dicts into chunk metadata.
"""
import re
from typing import List, Dict

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    L = len(text)
    chunks = []
    start = 0
    while start < L:
        end = min(start + max_chars, L)
        chunks.append(text[start:end].strip())
        if end >= L:
            break
        start = end - overlap
    return chunks

def chunk_documents(docs: List[Dict], max_chars: int = 1000, overlap: int = 200) -> List[Dict]:
    out = []
    for d in docs:
        chunks = chunk_text(d.get("text", ""), max_chars=max_chars, overlap=overlap)
        for i, c in enumerate(chunks):
            out.append({
                "doc_id": d["id"],
                "source": d["source"],
                "chunk_id": i,
                "text": c
            })
    return out

if __name__ == "__main__":
    s = "This is a sample. " * 200
    print(len(chunk_text(s, max_chars=200, overlap=40)))
