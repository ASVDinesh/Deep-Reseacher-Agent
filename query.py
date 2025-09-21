"""
query.py
Load FAISS index + meta and retrieve top-k chunks for a query using the same embedding model.
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss.index"
META_PATH = "meta.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_path=INDEX_PATH, meta_path=META_PATH, embed_model=EMBED_MODEL):
        try:
            self.index = faiss.read_index(index_path)
        except Exception as e:
            raise RuntimeError(f"[Retriever] Could not read index: {e} (run embed_and_index.py first)")
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.metas = json.load(fh)
        self.model = SentenceTransformer(embed_model)

    def retrieve(self, query: str, k: int = 5):
        q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        D, I = self.index.search(q_emb, k)
        hits = []
        for dist, idx in zip(D[0], I[0]):
            meta = self.metas[idx].copy()
            meta.update({"distance": float(dist)})
            hits.append(meta)
        return hits

if __name__ == "__main__":
    r = Retriever()
    q = "What is latent dirichlet allocation?"
    hits = r.retrieve(q, k=5)
    for h in hits:
        print(h["source"], "chunk", h["chunk_id"], "dist", h["distance"])
        print(h["text"][:300].replace("\n"," "), "...")
        print("----")
