# Phase 4 — Context Management

## What this phase builds

- Session management — episodic turn history and persistent user context
- Topic shift detection — centroid-based comparison of query embeddings
- History-aware query rewriting — resolves pronouns and references using recent turns
- Cold rewrite fallback — used on first turn and after topic shifts
- User context persistence — survives topic shifts, shapes answer tone and depth

## How to run

### First time setup
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
# Fill in PINECONE_API_KEY in .env
# Uses the same Pinecone index as Phase 2 and Phase 3 — no re-ingestion needed
```

### Run the pipeline
```bash
python3 pipeline.py
```

## Architecture
session = create_session()
run(query, session)

→ embed query

→ detect_topic_shift()

→ compute centroid of recent query embeddings

→ cosine similarity: current query vs centroid

→ if below TOPIC_SHIFT_THRESHOLD: clear turns, cold start

→ rewrite query

→ history_rewrite() if turns exist and no topic shift

→ cold_rewrite() if no turns or topic shift

→ retrieve(rewritten_query)

→ check_confidence()

→ if below threshold: cold_rewrite fallback, retry

→ if still below: short circuit, graceful failure

→ answer(query, retrieved_docs, user_context)

→ add_turn() — update session, trim to MAX_TURNS

## File responsibilities

| File | Responsibility |
|---|---|
| session.py | Session state, topic shift detection, turn management |
| rewriter.py | Cold rewrite and history-aware rewrite — two distinct modes |
| retriever.py | Embed query, query Pinecone, return ranked chunks |
| utils.py | Shared get_embedding() — single definition, imported everywhere |
| pipeline.py | Orchestration only — coordinates all of the above |

## Key design decisions

### Centroid-based topic shift detection
Topic shift is detected by comparing the current query embedding against
the centroid of recent query embeddings — the average vector representing
what the conversation has been about so far.

Pairwise comparison (current vs previous query only) misses gradual drift.
The centroid absorbs the full conversational direction across MAX_TURNS,
making it a more reliable reference point.

### History rewrite vs cold rewrite
On topic shift, turns are cleared before rewriting. The rewriter then
sees an empty history and falls back to cold rewrite — identical to
Phase 3. This prevents educational context from contaminating a Perspica
query, or platform engineering context from contaminating an early life query.

Using history to rewrite across a topic shift would pull the rewrite
in the wrong semantic direction.

### user_context persists, turns reset
Two types of memory serve different purposes:

- turns: episodic memory for the current topic — what has been asked
  and answered in this thread of conversation. Resets on topic shift.
- user_context: semantic memory about the person asking — their role,
  expertise level, intent. Persists for the full session regardless
  of topic shifts, and shapes every answer.

### MAX_TURNS = 3
The meaningful pronoun resolution window for a professional avatar
is rarely more than 2-3 turns. Beyond that, references become
explicit again. Limiting to 3 turns keeps the rewriter prompt
concise and prevents stale context from influencing rewrites.

## Key insight — knowledge base shape determines topic shift sensitivity

Topic shift detection via centroid similarity is directly affected
by how the knowledge base content is chunked and how focused each
chunk is semantically.

When chunks are broad — covering multiple sub-topics in one section —
the query embeddings for questions about that section carry mixed
signals. The centroid of several broad-chunk queries sits in a diffuse
region of vector space. A query about a genuinely different topic may
not land far enough from that centroid to trigger a shift detection.

**Observed in Phase 4:**
Three questions about platform engineering (Kafka, Kubernetes, OpenShift)
produced a centroid that still shared enough semantic overlap with
"childhood in Delhi" to not trigger a shift at the 0.55 threshold.
The cause: the "Technologies I work with" chunk is broad enough that
its embedding carries general Akash Jain signals alongside the
technical ones.

**The fix:**
Finer-grained chunking — one focused topic per chunk — produces
tighter query embeddings, a more specific centroid, and more reliable
topic shift detection. This will be addressed by rewriting profile.md
with more granular sections before Phase 5.

## Key insight — knowledge base shape is not always under your control

In enterprise RAG deployments, the source documents are rarely
structured for retrieval. You get policy documents written for legal
compliance, data dictionaries written by committees, runbooks organised
by system component — none written with semantic chunking in mind.

Three approaches practitioners use:

**Reshape at ingestion** — pre-process documents through an LLM that
restructures content into retrieval-optimised sections before chunking.
Works for structured documents. Breaks for unstructured content.
Adds cost and a pre-processing step that can introduce errors.

**Chunk more granularly** — chunk at paragraph level rather than
section level. Finer chunks, sharper embeddings, lower individual
scores. Compensate with higher TOP_K and reranking (Phase 5).

**Hybrid search** — combine vector search with BM25 keyword search.
BM25 matches exact terms regardless of chunk shape. Vector search
handles semantics. Together they are more robust to poorly shaped
knowledge bases than either alone. This is the primary motivation
for Phase 5.

In practice, all three are used in combination. The shape of source
content is an architectural input — audit it before designing the
pipeline, not after.

## Configuration

| Parameter | Default | Purpose |
|---|---|---|
| MAX_TURNS | 3 | Episodic memory window size |
| TOPIC_SHIFT_THRESHOLD | 0.55 | Centroid similarity below which a topic shift is detected |
| CONFIDENCE_THRESHOLD | 0.65 | Minimum retrieval score to proceed to LLM |
| TOP_K | 3 | Number of chunks retrieved per query |

## What Phase 5 will add

- Hybrid search — vector search combined with BM25 keyword search
- Reranking — a second scoring pass over retrieved chunks for precision
- Chunk size tuning — empirical testing of chunk granularity impact
  on retrieval scores and topic shift detection sensitivity