# rewriter.py
# Rewrites a semantically thin query into a richer one before retrieval.
# Called only when the first retrieval attempt falls below the confidence threshold.
# Uses the LLM to expand the query — not to answer it.

import requests
import config


def rewrite(query: str) -> str:
    """
    Takes a short or ambiguous query and rewrites it into a semantically
    richer version that retrieves better from a vector database.

    The rewriter does not answer the question — it only rephrases it.
    The rewritten query is passed back to the retriever for a second attempt.
    """
    # Giving the rewriter the section names prevents it from guessing
    # blindly when it encounters proper nouns like "Perspica" that it
    # has no training knowledge of.
    knowledge_base_sections = """
    - Early life and education
    - Academic highlights and leadership
    - How I got into data and technology
    - Career timeline
    - Key engagements and projects
    - Areas of expertise
    - Technologies I work with
    - Recognition and milestones
    - My writing — Perspica
    - What I offer
    - How I work
    """

    prompt = f"""You are a search query rewriter for a vector database about a professional named Akash Jain.
The database contains the following sections:
{knowledge_base_sections}

Your job is to rewrite the user's question into a longer, semantically richer version 
that will retrieve the most relevant section from the database.

Rules:
- Do not answer the question
- Do not add information that wasn't implied by the original question
- Use the section names above to guide your expansion
- Make the query self-contained and descriptive
- Return only the rewritten query, nothing else

Original query: {query}

Rewritten query:"""

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
        return response.json()["response"].strip()

    raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")