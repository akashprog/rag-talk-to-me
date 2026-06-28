# pipeline.py
# Phase 3 orchestration layer.
# Coordinates rewriting, retrieval, confidence checking, and answering.
# This is the only file that knows the full decision flow.
# retriever.py, rewriter.py, and utils.py each do one thing.

import requests
import config
from retriever import retrieve
from rewriter import rewrite


def check_confidence(results: list[dict]) -> bool:
    """
    Returns True if the top retrieved chunk clears the confidence threshold.
    Returns False if retrieval should be considered unreliable.
    Only checks the top result — if the best match isn't good enough,
    the rest won't be either.
    """
    if not results:
        return False
    return results[0]["score"] >= config.CONFIDENCE_THRESHOLD


def answer(query: str, retrieved_docs: list[dict]) -> str:
    """
    Passes the query and retrieved chunks to the LLM.
    The LLM answers strictly from retrieved context.
    Only called after confidence check passes.
    """
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])

    prompt = f"""You are a professional AI avatar representing Akash Jain.
Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say exactly:
"I don't have enough information on that."
Do not use any knowledge outside of the provided context.
Speak in third person — refer to Akash by name, not as "I".

Context:
{context}

Question:
{query}

Answer:"""

    if config.LLM_PROVIDER == "ollama":
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]

    raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")


def run(query: str) -> dict:
    """
    Full Phase 3 pipeline with fallback rewrite strategy.

    Flow:
    1. Retrieve with raw query
    2. Check confidence
       - if above threshold: answer directly
       - if below threshold: rewrite query and retrieve again
    3. Check confidence on rewritten results
       - if above threshold: answer from rewritten results
       - if still below: short circuit, return graceful failure

    Returns a dict with:
      - answer: the final response string
      - rewritten: bool, whether query rewriting was triggered
      - rewritten_query: the rewritten query if rewriting occurred, else None
      - retrieved: the chunks used for answering
    """
    # --- First attempt: raw query ---
    print(f"  [1] Retrieving with raw query...")
    results = retrieve(query)
    top_score = results[0]["score"] if results else 0.0
    print(f"  [1] Top score: {top_score:.4f}")

    if check_confidence(results):
        print(f"  [1] Confidence cleared — answering directly")
        return {
            "answer": answer(query, results),
            "rewritten": False,
            "rewritten_query": None,
            "retrieved": results
        }

    # --- Second attempt: rewrite and retry ---
    print(f"  [1] Below threshold ({config.CONFIDENCE_THRESHOLD}) — rewriting query...")
    rewritten_query = rewrite(query)
    print(f"  [2] Rewritten: {rewritten_query}")

    results = retrieve(rewritten_query)
    top_score = results[0]["score"] if results else 0.0
    print(f"  [2] Top score after rewrite: {top_score:.4f}")

    if check_confidence(results):
        print(f"  [2] Confidence cleared after rewrite — answering")
        return {
            "answer": answer(query, results),
            "rewritten": True,
            "rewritten_query": rewritten_query,
            "retrieved": results
        }

    # --- Short circuit: both attempts failed ---
    print(f"  [2] Still below threshold — short circuiting")
    return {
        "answer": "I don't have enough information on that.",
        "rewritten": True,
        "rewritten_query": rewritten_query,
        "retrieved": results
    }


if __name__ == "__main__":
    test_queries = [
        "Where did Akash study?",
        "What technologies does Akash work with?",
        "What is Perspica?",
        "What is Akash's favourite food?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = run(query)

        print(f"Rewritten: {result['rewritten']}")
        if result["rewritten_query"]:
            print(f"Rewritten query: {result['rewritten_query']}")

        print("Retrieved chunks:")
        for r in result["retrieved"]:
            print(f"  [{r['score']:.4f}] {r['section']}")

        print(f"Answer: {result['answer']}")
        print("-" * 60)