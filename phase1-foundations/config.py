# config.py
# Central configuration for the RAG pipeline.
# All provider choices and model names live here.
# Nothing else in the codebase hardcodes a model name or URL.

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Provider ---
# Controls which provider handles chat completion (question answering).
# Supported values: "ollama", "claude"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# --- Embedding Provider ---
# Controls which provider generates vector embeddings.
# Supported values: "ollama", "openai"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")

# --- Ollama Settings ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# --- Claude Settings (used in production) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# --- OpenAI Embedding Settings (used in production) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --- Retrieval Settings ---
# How many chunks to retrieve before passing to the LLM.
TOP_K = int(os.getenv("TOP_K", "2"))