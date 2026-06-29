# pipeline.py
# Phase 4 orchestration layer.
# Extends Phase 3 with session management, topic shift detection,
# and history-aware query rewriting.
# Coordinates: session → rewrite → retrieve → confidence → answer → session update

import requests
import config
from retriever import retrieve
from rewriter import cold_rewrite, history_rewrite
from session import (
    create_session,
    detect_topic_shift,
    get_recent_turns,
    add_turn,
    clear_turns,
    update_user_context
)
from utils import get_embedding


def check_confidence(results: list[dict]) -> bool:
    """
    Returns True if the top retrieved chunk clears the confidence threshold.
    """
    if not results:
        return False
    return results[0]["score"] >= config.CONFIDENCE_THRESHOLD


def answer(query: str, retrieved_docs: list[dict], user_context: dict) -> str:
    """
    Passes the query and retrieved chunks to the LLM.
    Incorporates persistent user context to tailor the response.
    Only called after confidence check passes.
    """
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])

    # Build user context string if we have any
    user_context_text = ""
    if user_context:
        user_context_text = "\n\nUser context (use this to tailor your response):\n"
        for key, value in user_context.items():
            user_context_text += f"- {key}: {value}\n"

    prompt = f"""You are a professional AI avatar representing Akash Jain.
Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say exactly:
"I don't have enough information on that."
Do not use any knowledge outside of the provided context.
Speak in third person — refer to Akash by name, not as "I".
{user_context_text}

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


def run(query: str, session: dict) -> dict:
    """
    Full Phase 4 pipeline with session management.

    Flow:
    1. Embed the query
    2. Detect topic shift — clear turns if shifted
    3. Rewrite query
       - history_rewrite if turns exist
       - cold_rewrite if no turns (first turn or post-shift)
    4. Retrieve with rewritten query
    5. Check confidence
       - if below threshold: cold_rewrite and retry
       - if still below: short circuit
    6. Generate answer with user context
    7. Update session — add turn, trim history

    Returns dict with answer, rewrite info, retrieved chunks,
    and whether a topic shift was detected.
    """
    print(f"\n  [pipeline] Query: {query}")

    # --- Step 1: Embed the query ---
    query_embedding = get_embedding(query)

    # --- Step 2: Topic shift detection ---
    topic_shifted = detect_topic_shift(query_embedding, session)
    if topic_shifted:
        clear_turns(session)

    # --- Step 3: Rewrite query ---
    recent_turns = get_recent_turns(session)

    if recent_turns and not topic_shifted:
        print(f"  [rewriter] History rewrite — {len(recent_turns)} turn(s) in context")
        rewritten_query = history_rewrite(query, recent_turns)
    else:
        print(f"  [rewriter] Cold rewrite — no history")
        rewritten_query = cold_rewrite(query)

    print(f"  [rewriter] Rewritten: {rewritten_query}")

    # --- Step 4: Retrieve with rewritten query ---
    results = retrieve(rewritten_query)
    top_score = results[0]["score"] if results else 0.0
    print(f"  [retriever] Top score: {top_score:.4f}")

    # --- Step 5: Confidence check with fallback ---
    if not check_confidence(results):
        print(f"  [retriever] Below threshold — cold rewrite fallback")
        fallback_query = cold_rewrite(query)
        results = retrieve(fallback_query)
        top_score = results[0]["score"] if results else 0.0
        print(f"  [retriever] Fallback top score: {top_score:.4f}")

        if not check_confidence(results):
            print(f"  [pipeline] Short circuit — returning graceful failure")
            final_answer = "I don't have enough information on that."
            add_turn(
                session, query, query_embedding,
                rewritten_query,
                [r["section"] for r in results],
                final_answer
            )
            return {
                "answer": final_answer,
                "rewritten_query": rewritten_query,
                "topic_shifted": topic_shifted,
                "retrieved": results
            }

    # --- Step 6: Generate answer ---
    final_answer = answer(query, results, session["user_context"])

    # --- Step 7: Update session ---
    add_turn(
        session, query, query_embedding,
        rewritten_query,
        [r["section"] for r in results],
        final_answer
    )

    return {
        "answer": final_answer,
        "rewritten_query": rewritten_query,
        "topic_shifted": topic_shifted,
        "retrieved": results
    }


if __name__ == "__main__":
    session = create_session()
    # Simulate known user context — in production this would be
    # extracted automatically from early conversation turns.
    from session import update_user_context

    # update_user_context(session, "role", "CTO evaluating Akash for a strategic data and AI consulting engagement")
    # update_user_context(session, "expertise_level", "senior technical executive — comfortable with architecture concepts, avoid basic explanations")

    conversation = [
        "What kind of enterprise engagements has Akash led?",
        "Has he worked with telecommunications companies?",
        "What does he offer for strategic consulting?",
    ]
    # Simulated multi-turn conversation with a topic shift
    # conversation = [
    #     # Topic 1 — education
    #     "Where did Akash study?",
    #     "What did he study there?",
    #     "How did that shape his career?",

    #     # Topic shift — Perspica
    #     "What is Perspica?",
    #     "What series is he currently writing?",
    # ]

    for query in conversation:
        result = run(query, session)
        print(f"\n  Query: {query}")
        print(f"  Topic shifted: {result['topic_shifted']}")
        print(f"  Rewritten: {result['rewritten_query']}")
        print(f"  Retrieved: {[r['section'] for r in result['retrieved']]}")
        print(f"  Answer: {result['answer']}")
        print("-" * 60)