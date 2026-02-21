import sys
import os
from openai import OpenAI

from embed import query_vectorstore

# ── config ────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-opus-4-5"
N_CHUNKS = 5

# ── system prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a document analyst answering questions strictly from the provided context excerpts.

Rules:
1. Answer ONLY using information from the context below. Do not use any prior knowledge.
2. After every claim or fact, cite the source as: [Section: <section name>, Page: <page number>]
3. If the answer cannot be found in the context, respond with exactly:
   "The answer to this question is not available in the provided document excerpts."
4. Be concise and precise. Do not speculate or infer beyond what is explicitly stated."""

# ── core function ─────────────────────────────────────────────────────────────

def generate_answer(question: str) -> str:
    """
    Retrieve top-N chunks for the question, send to Claude via OpenRouter,
    return a grounded answer with section + page citations.
    """
    # step 1: retrieve relevant chunks
    print(f"Retrieving top {N_CHUNKS} chunks for: '{question}'")
    chunks = query_vectorstore(question, n_results=N_CHUNKS)

    # step 2: format context block
    context_parts = []
    for chunk in chunks:
        header = f"[Section: {chunk['section']} | Page: {chunk['page_number']}]"
        context_parts.append(f"{header}\n{chunk['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    # step 3: build user message
    user_message = f"Context:\n\n{context_block}\n\nQuestion: {question}"

    # step 4: call OpenRouter
    print("Calling Claude via OpenRouter...\n")
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    return response.choices[0].message.content


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # accept question from command line, or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is the total fresh issue size and who are the promoters of the company?"

    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60 + "\n")

    answer = generate_answer(question)

    print("Answer:")
    print("-" * 60)
    print(answer)
    print("-" * 60)
