# config.py
# Central configuration for the Phase 2 RAG pipeline.
# All provider choices, model names, and index settings live here.
# Nothing else in the codebase hardcodes a model name, URL, or index name.

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
# Must match the output size of the active embedding model.
# nomic-embed-text produces 768-dimensional vectors.
# text-embedding-3-small produces 1536-dimensional vectors.
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