"""
app.py
Gradio UI to interact with the Deep Researcher Agent.
- Make sure to run `python -m embed_and_index` first to build faiss.index + meta.json
- Launch: python app.py
"""
import gradio as gr
from query import Retriever
from synthesize import decompose_question, synthesize_answer
from utils import export_markdown

# initialize retriever
try:
    retr = Retriever()
except Exception as e:
    retr = None
    print("[app] Retriever init failed:", e)

def ask(question: str, k: int = 5, use_decompose: bool = True):
    if retr is None:
        return "Index not built. Run: python -m embed_and_index", "", ""
    if not question or not question.strip():
        return "Type a question", "", ""

    subs = decompose_question(question) if use_decompose else [question]
    agg_answers = []
    agg_hits = []
    for sub in subs:
        hits = retr.retrieve(sub, k=k)
        agg_hits.extend(hits)
        contexts = [h["text"] for h in hits]
        ans = synthesize_answer(sub, contexts)
        agg_answers.append(f"### Sub-answer for: {sub}\n{ans}\n")

    final_answer = "\n".join(agg_answers)
    # unique source list
    unique = []
    for h in agg_hits:
        s = f"{h.get('source')} (chunk {h.get('chunk_id')})"
        if s not in unique:
            unique.append(s)
    sources_text = "\n".join([f"- {s}" for s in unique])
    return final_answer, sources_text, str(len(agg_hits))

def export_result(question: str, answer: str):
    # regenerate hits to include up-to-date retrieval
    hits = retr.retrieve(question, k=10) if retr else []
    md_path = export_markdown("DeepResearcherAgent", question, answer, hits)
    return f"Saved: {md_path}"

with gr.Blocks(title="Deep Researcher Agent (Local RAG)") as demo:
    gr.Markdown("# Deep Researcher Agent")
    q = gr.Textbox(label="Ask a research question", placeholder="e.g., Compare LDA and NMF")
    with gr.Row():
        k = gr.Slider(1, 10, value=5, step=1, label="Top-k per subquestion")
        decomp = gr.Checkbox(value=True, label="Auto-decompose complex questions")
    ask_btn = gr.Button("Ask")
    answer_md = gr.Markdown("", label="Answer")
    sources = gr.Textbox(label="Sources", lines=6)
    hits_count = gr.Textbox(label="Total hits retrieved")
    export_btn = gr.Button("Export to Markdown")
    status = gr.Textbox(label="Status / Messages")

    def on_ask(question, k_val, decomp_flag):
        ans, srcs, hits = ask(question, k=int(k_val), use_decompose=decomp_flag)
        return ans, srcs, hits, ""

    ask_btn.click(on_ask, inputs=[q, k, decomp], outputs=[answer_md, sources, hits_count, status])
    export_btn.click(lambda question, ans: export_result(question, ans), inputs=[q, answer_md], outputs=status)

if __name__ == "__main__":
    demo.launch(share=True)
