"""
utils.py
small utilities: save/export markdown, safe filename, load meta.
"""
import os
import json
import datetime
import re
import markdown2

META_PATH = "meta.json"

def safe_filename(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', s)[:200]

def load_meta(path: str = META_PATH):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def export_markdown(title: str, question: str, answer: str, hits: list, out_dir: str = "exports"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = safe_filename(f"{title}_{now}.md")
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write(f"**Question:** {question}\n\n")
        fh.write(f"**Answer:**\n\n{answer}\n\n")
        fh.write("## Sources\n\n")
        for h in hits:
            fh.write(f"- {h.get('source')} (chunk {h.get('chunk_id')})\n")
            snippet = h.get("text", "")[:400].replace("\n", " ")
            fh.write(f"  > {snippet}...\n\n")
    return path

def md_to_html(md_path: str, html_path: str):
    with open(md_path, "r", encoding="utf-8") as fh:
        md = fh.read()
    html = markdown2.markdown(md)
    with open(html_path, "w", encoding="utf-8") as out:
        out.write(html)
    return html_path
