# synthesize.py
from transformers import pipeline

# Create a text-generation pipeline (you can choose a local model or HuggingFace Hub model)
generator = pipeline(
    "text-generation",
    model="sshleifer/tiny-gpt2",
    device=-1  # CPU
)


def decompose_question(question: str):
    """
    Split complex questions into sub-questions.
    Here we do a simple placeholder split by commas.
    """
    subs = [q.strip() for q in question.split(",") if q.strip()]
    return subs if subs else [question]

def synthesize_answer(sub_question: str, contexts: list[str], max_tokens: int = 150):
    """
    Generate an answer using the LLM based on the retrieved context.
    """
    # Combine retrieved chunks into a single context
    context_text = "\n".join(contexts)
    
    prompt = (
        f"Answer the question based on the context below.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {sub_question}\nAnswer:"
    )

    response = generator(prompt, max_length=max_tokens, do_sample=True, temperature=0.7)
    # generator returns a list of dicts: [{"generated_text": "..."}]
    return response[0]["generated_text"].strip()
