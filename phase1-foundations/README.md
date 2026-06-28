# Phase 1 — Foundations

A minimal **Retrieval-Augmented Generation (RAG)** pipeline. This phase strips RAG down to its core loop so every step is visible and easy to reason about.

**What it does:** embed → retrieve → answer.

**What it deliberately omits:** vector database, document chunking, ingestion pipeline, and UI. Those come in later phases.

The demo uses a hardcoded knowledge base of five short documents about Akash Jain and Perspica. When you run the script, it embeds a query, finds the most similar documents via cosine similarity, and passes only those documents to an LLM for a grounded answer.

---

## Project structure

```
phase1-foundations/
├── config.py        # Central configuration (providers, models, URLs)
├── pipeline.py      # RAG pipeline: embed, retrieve, answer
├── requirements.txt # Python dependencies
├── .env.example     # Optional environment variable template
└── README.md
```

---

## Prerequisites

- **Python 3.12+**
- **[Ollama](https://ollama.com/)** running locally (default: `http://localhost:11434`)
- Ollama models pulled:
  - `llama3` — used for question answering
  - `nomic-embed-text` — used for embeddings

```bash
ollama pull llama3
ollama pull nomic-embed-text
ollama serve   # skip if already running
```

---

## Setup

```bash
cd phase1-foundations

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and override defaults (see [Configuration](#configuration)).

---

## Run

```bash
source .venv/bin/activate
python pipeline.py
```

Example output:

```
Query: What does Akash like to eat?

Retrieved chunks:
  [0.5234] Akash Jain is a Senior Solutions Architect with 13 years of experience...
  [0.4891] Perspica is Akash's blog covering data architecture, AI engineering...

Generating answer...

Answer: I don't have enough information on that.
```

The LLM is instructed to answer **only** from retrieved context. If nothing relevant is found, it should refuse rather than hallucinate.

---

## How it works

```
Query
  │
  ▼
get_embedding()          ← turn query into a vector
  │
  ▼
retrieve()               ← embed each document, score with cosine_similarity(), return top_k
  │
  ▼
answer()                 ← send query + retrieved docs to LLM, return response
```

---

## Configuration (`config.py`)

All provider choices and model names live in `config.py`. Nothing else in the codebase hardcodes a model or URL. Values are read from environment variables (via `.env`) with sensible defaults.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Chat completion provider (`ollama`, `claude` planned) |
| `EMBEDDING_PROVIDER` | `ollama` | Embedding provider (`ollama`, `openai` planned) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_LLM_MODEL` | `llama3` | Model used for answering questions |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Model used for generating embeddings |
| `ANTHROPIC_API_KEY` | `""` | API key for Claude (future use) |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model name (future use) |
| `OPENAI_API_KEY` | `""` | API key for OpenAI embeddings (future use) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model (future use) |
| `TOP_K` | `2` | Number of documents to retrieve before answering |

---

## Functions (`pipeline.py`)

### `get_embedding(text: str) -> np.ndarray`

Converts a string into a dense vector (embedding) using the configured embedding provider.

- **Input:** any text string (a query or a document).
- **Output:** a `numpy` float32 array representing the text in embedding space.
- **Current behavior:** calls Ollama's `/api/embeddings` endpoint with `OLLAMA_EMBEDDING_MODEL`.
- **Why it matters:** embeddings capture semantic meaning. Similar texts produce vectors that point in similar directions, which is what retrieval is built on.

---

### `cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float`

Measures how similar two embedding vectors are by computing the cosine of the angle between them.

- **Input:** two embedding vectors (`vec_a`, `vec_b`).
- **Output:** a float between `-1` and `1`.
  - `1.0` — identical direction (very similar meaning)
  - `0.0` — perpendicular (unrelated)
  - `-1.0` — opposite direction (opposing meaning)
- **Implementation:** dot product divided by the product of each vector's magnitude.
- **Why it matters:** this is the scoring function for retrieval. Higher scores mean the document is more relevant to the query.

---

### `retrieve(query: str, documents: list[str], top_k: int = config.TOP_K) -> list[dict]`

Finds the most relevant documents from a list given a query.

- **Input:**
  - `query` — the user's question
  - `documents` — a list of text strings to search over
  - `top_k` — how many results to return (defaults to `TOP_K` from config)
- **Output:** a list of dicts, each with:
  - `"text"` — the document string
  - `"score"` — its cosine similarity to the query
- **Steps:**
  1. Embed the query
  2. Embed each document
  3. Compute cosine similarity between the query and each document
  4. Sort by score, highest first
  5. Return the top `top_k` results
- **Why it matters:** this is the **retrieval** step in RAG. The LLM never sees the full knowledge base — only what `retrieve()` selects.

---

### `answer(query: str, retrieved_docs: list[dict]) -> str`

Generates a natural-language answer using only the retrieved documents as context.

- **Input:**
  - `query` — the user's question
  - `retrieved_docs` — output from `retrieve()` (list of `{"text", "score"}` dicts)
- **Output:** the LLM's response as a string.
- **Steps:**
  1. Join retrieved document texts into a context block
  2. Build a prompt that instructs the LLM to answer **only** from that context
  3. Call Ollama's `/api/generate` endpoint with `OLLAMA_LLM_MODEL`
  4. Return the generated text
- **Grounding constraint:** the prompt explicitly forbids using outside knowledge. If the context is insufficient, the model is told to say *"I don't have enough information on that."*
- **Why it matters:** this is the **generation** step in RAG. Retrieval quality directly limits answer quality — garbage in, garbage out.

---

## What's next

Phase 2 replaces the hardcoded document list with real ingested content and introduces proper chunking. Later phases add a vector database, quality improvements, context management, a UI, and a full ingestion pipeline.
