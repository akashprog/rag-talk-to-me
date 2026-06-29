# session.py
# Owns all session state for a conversation.
# Manages episodic turn history, persistent user context,
# and topic shift detection via query embedding centroid.
# The pipeline never directly modifies session state —
# it calls functions here instead.

import numpy as np
import config
from utils import get_embedding


def create_session() -> dict:
    """
    Creates a fresh session with empty turn history and user context.
    Call this once at the start of a conversation.
    """
    return {
        "turns": [],
        "user_context": {}
    }


def get_recent_turns(session: dict) -> list[dict]:
    """
    Returns the last MAX_TURNS turns from the session.
    These are the only turns passed to the rewriter and
    used for centroid computation.
    """
    return session["turns"][-config.MAX_TURNS:]


def compute_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Computes the average vector across a list of embeddings.
    Returns the centroid — the centre of gravity of the conversation
    in vector space.
    """
    stacked = np.stack(embeddings, axis=0)
    return np.mean(stacked, axis=0)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Measures directional similarity between two vectors.
    Returns a float between -1 and 1.
    """
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    return float(dot_product / (magnitude_a * magnitude_b))


def detect_topic_shift(query_embedding: np.ndarray, session: dict) -> bool:
    """
    Detects whether the current query represents a topic shift
    relative to recent conversation history.

    Returns True if a topic shift is detected — caller should clear turns.
    Returns False if the query is consistent with recent history.

    If there are no previous turns, always returns False —
    the first query in a session is never a topic shift.
    """
    recent_turns = get_recent_turns(session)

    if not recent_turns:
        return False

    # Build centroid from recent query embeddings
    recent_embeddings = [
        np.array(turn["query_embedding"], dtype=np.float32)
        for turn in recent_turns
    ]
    centroid = compute_centroid(recent_embeddings)

    # Compare current query to centroid
    similarity = cosine_similarity(query_embedding, centroid)

    print(f"  [session] Centroid similarity: {similarity:.4f} "
          f"(threshold: {config.TOPIC_SHIFT_THRESHOLD})")

    return similarity < config.TOPIC_SHIFT_THRESHOLD


def add_turn(session: dict, query: str, query_embedding: np.ndarray,
             rewritten_query: str | None, retrieved_sections: list[str],
             answer: str):
    """
    Adds a completed turn to session history.
    Automatically trims to MAX_TURNS — oldest turns are dropped.
    """
    session["turns"].append({
        "raw_query": query,
        "query_embedding": query_embedding.tolist(),
        "rewritten_query": rewritten_query,
        "retrieved_sections": retrieved_sections,
        "answer": answer
    })

    # Trim to window size — drop oldest if over limit
    if len(session["turns"]) > config.MAX_TURNS:
        session["turns"] = session["turns"][-config.MAX_TURNS:]


def update_user_context(session: dict, key: str, value: str):
    """
    Stores a persistent piece of user context.
    This survives topic shifts — it is never cleared during a session.
    Call this when the pipeline detects something worth remembering
    about the person asking — their role, expertise level, intent.
    """
    session["user_context"][key] = value


def clear_turns(session: dict):
    """
    Clears episodic turn history on topic shift.
    Does NOT clear user_context — that persists for the full session.
    """
    print(f"  [session] Topic shift detected — clearing turn history")
    session["turns"] = []