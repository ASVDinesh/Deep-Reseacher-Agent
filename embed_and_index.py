"""
embed_and_index.py

Builds embeddings for chunks and stores a FAISS index and a metadata JSON file.
Run: python -m embed_and_index
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from ingest import load_documents
from chunker import chunk_documents

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss.index"
META_PATH = "meta.json"
CHUNK_MAX = 900
CHUNK_OVERLAP = 200

def build_index(docs_folder: str = "data/docs"):
    docs = load_documents(docs_folder)
    if not docs:
        print("[build_index] no docs found. Put files into data/docs/")
        return
    chunks = chunk_documents(docs, max_chars=CHUNK_MAX, overlap=CHUNK_OVERLAP)
    texts = [c["text"] for c in chunks]
    print(f"[build_index] {len(chunks)} chunks to embed using {EMBED_MODEL}")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh)
    print(f"[build_index] Wrote index ({index.ntotal} vectors) -> {INDEX_PATH}")
    print(f"[build_index] Wrote metadata -> {META_PATH}")

if __name__ == "__main__":
    build_index("data/docs")
