# Phase 2 — Core

A production-shaped **Retrieval-Augmented Generation (RAG)** pipeline. Phase 2 builds on Phase 1 by replacing the hardcoded in-memory document list with a real knowledge base, proper chunking, and a vector database.

**What it adds over Phase 1:**

- Markdown ingestion from `knowledge_base/profile.md`
- Section-based chunking (one chunk per `##` heading)
- [Pinecone](https://www.pinecone.io/) vector storage for retrieval
- Separate ingest and query scripts

**What it still omits:** quality tuning, context management, UI, and automated ingestion from external sources. Those come in later phases.

The demo answers questions about Akash Jain as a professional AI avatar — grounded strictly on retrieved profile content.

---

## Project structure

```
phase2-core/
├── config.py                  # Central configuration (providers, models, Pinecone)
├── ingest.py                  # Chunk markdown → embed → upsert to Pinecone
├── pipeline.py                # Query Pinecone → retrieve → answer
├── knowledge_base/
│   └── profile.md             # Source knowledge base (11 sections)
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
└── README.md
```

---

## Prerequisites

- **Python 3.12+**
- **[Ollama](https://ollama.com/)** running locally (default: `http://localhost:11434`)
- Ollama models pulled:
  - `llama3` — used for question answering
  - `nomic-embed-text` — used for embeddings (768-dimensional vectors)
- **[Pinecone](https://app.pinecone.io/)** account with an API key

```bash
ollama pull llama3
ollama pull nomic-embed-text
ollama serve   # skip if already running
```

---

## Setup

```bash
cd phase2-core

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and set PINECONE_API_KEY
```

---

## Run

Ingestion and querying are separate steps. Run ingestion first (or re-run whenever `profile.md` changes), then run the pipeline.

### 1. Ingest the knowledge base

```bash
source .venv/bin/activate
python ingest.py
```

This reads `knowledge_base/profile.md`, splits it into 11 section-based chunks, embeds each one, and upserts them into Pinecone. On first run it creates the index if it doesn't exist.

### 2. Query the pipeline

```bash
python pipeline.py
```

This runs four test queries against the ingested knowledge base and prints retrieved sections plus generated answers.

Example output:

```
Query: Where did Akash study?

Retrieved chunks:
  [0.8123] Early life and education
  [0.6541] Academic highlights and leadership
Answer: Akash studied Bachelor of Technology in Information Technology at Bharati Vidyapeeth College of Engineering in Pune, Maharashtra, India, from 2009 to 2013.
------------------------------------------------------------
```

The LLM is instructed to answer **only** from retrieved context and speak in third person. If nothing relevant is found, it should refuse rather than hallucinate.

---

## How it works

### Ingestion (`ingest.py`)

```
profile.md
  │
  ▼
chunk_markdown()         ← split on ## headings, one chunk per section
  │
  ▼
get_embedding()          ← embed each chunk via Ollama
  │
  ▼
setup_pinecone_index()   ← create index if missing
  │
  ▼
ingest()                 ← upsert vectors + metadata to Pinecone
```

### Querying (`pipeline.py`)

```
Query
  │
  ▼
get_embedding()          ← turn query into a vector
  │
  ▼
retrieve()               ← query Pinecone for top_k similar chunks
  │
  ▼
answer()                 ← send query + retrieved chunks to LLM
```

Unlike Phase 1, retrieval no longer embeds every document at query time. Embeddings are precomputed at ingest time and stored in Pinecone. At query time, only the query is embedded and matched against the index.

---

## Configuration (`config.py`)

All provider choices, model names, and index settings live in `config.py`. Values are read from environment variables (via `.env`) with sensible defaults.

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
| `PINECONE_API_KEY` | `""` | Pinecone API key (required) |
| `PINECONE_INDEX_NAME` | `talk-to-me` | Pinecone index name |
| `TOP_K` | `3` | Number of chunks to retrieve before answering |

### `get_embedding_dimension() -> int`

Returns the vector dimension for the active embedding model. Pinecone index dimension must match:

- `nomic-embed-text` → 768
- `text-embedding-3-small` → 1536

If you change embedding models, delete and recreate the Pinecone index (or use a new index name) so dimensions stay aligned.

---

## Functions

### `ingest.py`

#### `chunk_markdown(filepath: str) -> list[dict]`

Splits a markdown file into chunks based on `##` headings. Each section becomes one chunk.

- **Input:** path to a markdown file.
- **Output:** a list of dicts, each with:
  - `"id"` — sequential ID (`chunk_0`, `chunk_1`, …)
  - `"section"` — the heading text (without `##`)
  - `"text"` — the full chunk (heading + body)
- **Behavior:** uses regex to split on `##` headings. Content before the first heading (e.g. the `#` title) is skipped. Only sections with a non-empty body are included.
- **Why it matters:** chunking strategy directly affects retrieval quality. Section-based splitting keeps each chunk semantically coherent — a career timeline chunk won't get mixed with an education chunk.

#### `get_embedding(text: str) -> np.ndarray`

Converts a string into a dense vector using the configured embedding provider.

- **Input:** any text string (a chunk during ingest, or a query during retrieval).
- **Output:** a `numpy` float32 array.
- **Current behavior:** calls Ollama's `/api/embeddings` endpoint with `OLLAMA_EMBEDDING_MODEL`.
- **Note:** duplicated in both `ingest.py` and `pipeline.py` so each file is self-contained.

#### `setup_pinecone_index()`

Connects to Pinecone and creates the index if it doesn't already exist.

- **Input:** none (reads from `config`).
- **Output:** a Pinecone index object ready for upsert or query.
- **Behavior:**
  - Lists existing indexes and creates `PINECONE_INDEX_NAME` if missing
  - Sets dimension via `get_embedding_dimension()`
  - Uses cosine similarity metric
  - Creates a serverless index on AWS `us-east-1`
- **Why it matters:** idempotent setup — safe to run on every ingest without manual index management.

#### `ingest(filepath: str)`

Runs the full ingestion pipeline end to end.

- **Input:** path to the markdown file to ingest.
- **Steps:**
  1. Chunk the markdown file via `chunk_markdown()`
  2. Embed each chunk via `get_embedding()`
  3. Upsert vectors into Pinecone with metadata (`section`, `text`)
- **When to re-run:** whenever `profile.md` content changes, or after switching embedding models.

---

### `pipeline.py`

#### `get_embedding(text: str) -> np.ndarray`

Same as in `ingest.py`. Converts text into a vector using the configured embedding provider. Must use the same model as ingest so query vectors live in the same embedding space as stored chunks.

#### `retrieve(query: str) -> list[dict]`

Embeds the query and retrieves the most similar chunks from Pinecone.

- **Input:** the user's question.
- **Output:** a list of dicts, each with:
  - `"section"` — the chunk's section name (from metadata)
  - `"text"` — the full chunk text (from metadata)
  - `"score"` — Pinecone similarity score
- **Steps:**
  1. Embed the query
  2. Connect to Pinecone and query the index with `top_k=config.TOP_K`
  3. Extract section, text, and score from each match
- **Why it matters:** this replaces Phase 1's in-memory cosine similarity loop. Retrieval is now fast and scalable — Pinecone handles similarity search across pre-stored vectors.

#### `answer(query: str, retrieved_docs: list[dict]) -> str`

Generates a natural-language answer using only the retrieved chunks as context.

- **Input:**
  - `query` — the user's question
  - `retrieved_docs` — output from `retrieve()`
- **Output:** the LLM's response as a string.
- **Steps:**
  1. Join retrieved chunk texts into a context block
  2. Build a prompt instructing the LLM to act as Akash's AI avatar, answer only from context, and speak in third person
  3. Call Ollama's `/api/generate` endpoint with `OLLAMA_LLM_MODEL`
  4. Return the generated text
- **Grounding constraint:** same as Phase 1 — no outside knowledge. If context is insufficient, the model should say *"I don't have enough information on that."*

---

## Knowledge base

`knowledge_base/profile.md` contains 11 sections:

| Section | Topic |
|---|---|
| Early life and education | Background, schooling, university |
| Academic highlights and leadership | Student leadership roles |
| How I got into data and technology | Career origin story |
| Career timeline | Work history |
| Key engagements and projects | Notable client work |
| Areas of expertise | Domain strengths |
| Technologies I work with | Tech stack |
| Recognition and milestones | Awards and achievements |
| My writing — Perspica | Blog and publications |
| What I offer | Services and value proposition |
| How I work | Working style and approach |

Each `##` section becomes one chunk in Pinecone. Edit this file and re-run `ingest.py` to update the knowledge base.

---

## Known limitation — named entity queries

A short query built around a proper noun that the embedding model has never 
seen in training retrieves poorly, even when the answer clearly exists in 
the knowledge base.

**Example:**

| Query | Top chunk retrieved | Score |
|---|---|---|
| "What is Perspica?" | How I work | 0.508 |
| "What does Akash write about?" | My writing — Perspica | 0.654 |
| "Perspica blog data AI" | My writing — Perspica | 0.684 |

Both the first and second queries have identical intent. The first fails 
retrieval. The second succeeds. The difference is semantic richness.

**Why this happens:**

"What is Perspica?" is a named entity lookup. The word "Perspica" is not 
in the embedding model's training vocabulary — it carries almost no vector 
signal. The query has nothing else to anchor it semantically, so the query 
vector drifts away from the correct chunk.

"What does Akash write about?" works because "write," "about," and "Akash" 
are all semantically rich words that map directly onto content in the 
Perspica chunk. The embedding model understands these words and places the 
query vector close to the right chunk.

**The fix — Phase 3:**

Query rewriting. Before retrieval, a rewriter expands short or 
semantically thin queries into richer ones. "What is Perspica?" becomes 
"What does Akash write about and what is his blog called?" That rewritten 
query retrieves correctly.

This is a known, documented limitation of Phase 2. It is not a bug — 
it is the design boundary that motivates Phase 3.

---

## What's next

Phase 3 focuses on retrieval and answer quality — better chunking, reranking, and evaluation. Later phases add context management, a UI, and automated ingestion from external sources.
