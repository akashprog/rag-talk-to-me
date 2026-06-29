# rewriter.py
# Extends Phase 3 rewriter with conversation history awareness.
# Two rewriting modes:
#   cold_rewrite — no history, identical to Phase 3 (used on topic shift or first turn)
#   history_rewrite — uses recent turns to resolve pronouns and references

import requests
import config

# Knowledge base section names provided to guide expansion.
# Update this list when new sections are added to the knowledge base.
KNOWLEDGE_BASE_SECTIONS = """
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


def cold_rewrite(query: str) -> str:
    """
    Rewrites a query with no conversation history.
    Used on the first turn of a session or after a topic shift.
    Identical to Phase 3 rewriter.
    """
    prompt = f"""You are a search query rewriter for a vector database about a professional named Akash Jain.
The database contains the following sections:
{KNOWLEDGE_BASE_SECTIONS}

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

    return _call_llm(prompt)


def history_rewrite(query: str, recent_turns: list[dict]) -> str:
    """
    Rewrites an ambiguous query using recent conversation history.
    Resolves pronouns and references like 'he', 'there', 'that role',
    'tell me more about that' into explicit, self-contained queries.

    Only receives the last MAX_TURNS turns — not the full history.
    """
    # Format recent turns as readable context for the rewriter
    history_text = ""
    for i, turn in enumerate(recent_turns):
        history_text += f"Q{i+1}: {turn['raw_query']}\n"
        history_text += f"A{i+1}: {turn['answer']}\n\n"

    prompt = f"""You are a search query rewriter for a vector database about a professional named Akash Jain.
The database contains the following sections:
{KNOWLEDGE_BASE_SECTIONS}

Recent conversation history:
{history_text}

The user has now asked: "{query}"

This query may contain pronouns or references to previous answers.
Rewrite it into a fully self-contained, semantically rich query that
can be understood without the conversation history.

Rules:
- Do not answer the question
- Resolve all pronouns and references using the conversation history
- Use the section names above to guide your expansion
- Return only the rewritten query, nothing else

Rewritten query:"""

    return _call_llm(prompt)


def _call_llm(prompt: str) -> str:
    """
    Internal function — sends a prompt to the configured LLM and returns
    the response text. Not called directly by the pipeline.
    """
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