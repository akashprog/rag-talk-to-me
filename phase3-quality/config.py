# config.py
# Phase 3 adds one new config value — the confidence threshold.
# Everything else is carried forward from Phase 2.

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Provider ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# --- Embedding Provider ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")

# --- Ollama Settings ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# --- Claude Settings ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# --- OpenAI Embedding Settings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --- Pinecone Settings ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "talk-to-me")

# --- Embedding Dimensions ---
EMBEDDING_DIMENSIONS = {
    "nomic-embed-text": 768,
    "text-embedding-3-small": 1536
}

def get_embedding_dimension() -> int:
    if EMBEDDING_PROVIDER == "ollama":
        return EMBEDDING_DIMENSIONS.get(OLLAMA_EMBEDDING_MODEL, 768)
    if EMBEDDING_PROVIDER == "openai":
        return EMBEDDING_DIMENSIONS.get(OPENAI_EMBEDDING_MODEL, 1536)
    raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")

# --- Retrieval Settings ---
TOP_K = int(os.getenv("TOP_K", "3"))

# --- Confidence Threshold ---
# Minimum cosine similarity score for the top retrieved chunk.
# Below this, the pipeline triggers a rewrite and retries.
# If the retry still fails to clear this threshold, the pipeline
# short circuits and returns a graceful failure without calling the LLM.
# Derived empirically from Phase 2 and Phase 3 diagnostic results:
#   correct retrievals clustered above 0.65
#   failed retrievals (after rewrite) clustered below 0.62
#   threshold set at 0.63 — clear margin on both sides
#   rationale: bounded professional knowledge base means weak post-rewrite
#   scores reliably indicate out-of-scope questions, not retrieval failure
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.63"))