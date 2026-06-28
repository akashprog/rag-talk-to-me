# Phase 3 — RAG Quality Layer

## What this phase builds

- Fallback rewrite strategy — retrieve with raw query first, rewrite only if confidence fails
- Confidence thresholding — pipeline short circuits before calling the LLM if retrieval is weak
- Context-aware query rewriter — section names from the knowledge base guide expansion
- Strict grounding enforced at pipeline level, not just prompt level
- Utility layer — shared functions extracted to utils.py, no duplication across files

## How to run

### First time setup
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
# Fill in PINECONE_API_KEY in .env
# Uses the same Pinecone index as Phase 2 — no re-ingestion needed
```

### Run the pipeline
```bash
python3 pipeline.py
```

## Architecture
query

→ retrieve(raw query)

→ check_confidence()

→ above threshold (0.63): answer directly

→ below threshold:

→ rewrite(query)          # LLM expands query using section names

→ retrieve(rewritten)

→ check_confidence()

→ above threshold: answer from rewritten results

→ still below: short circuit, return graceful failure

→ answer(query, retrieved_docs)

## Key design decisions

### Fallback rewrite, not always-rewrite
The rewriter is only called when the first retrieval attempt fails confidence.
Well-formed queries pay zero rewrite cost. Only thin or ambiguous queries
trigger the second LLM call. This keeps latency low for the majority of queries.

### Context-aware rewriter
The rewriter receives the knowledge base section names as context.
Without this, it guesses blindly when it encounters proper nouns like
"Perspica" that are not in its training vocabulary — expanding in the
wrong semantic direction and retrieving the wrong chunks.

With section names provided, it correctly identifies "My writing — Perspica"
as the target and expands toward it.

| Query | Without section context | With section context |
|---|---|---|
| "What is Perspica?" | Retrieved: Areas of expertise (0.711) | Retrieved: My writing — Perspica (0.757) |

### Confidence threshold — 0.63
Derived empirically from diagnostic data across Phase 2 and Phase 3:

| Query | Post-rewrite score | Outcome |
|---|---|---|
| "Where did Akash study?" | 0.657 (raw) | Correct retrieval |
| "What technologies does Akash work with?" | 0.698 (raw) | Correct retrieval |
| "What is Perspica?" | 0.757 (after rewrite) | Correct retrieval |
| "What is Akash's favourite food?" | 0.621 (after rewrite) | Wrong chunks, short circuit |

Correct retrievals cluster above 0.65. Failed retrievals after rewrite
cluster below 0.63. Threshold set at 0.63 with clear margin on both sides.

Rationale for 0.63 over 0.60: this is a bounded professional knowledge base.
Weak post-rewrite scores reliably indicate out-of-scope questions, not
retrieval struggling with a legitimate query. Raising the threshold
eliminates unnecessary LLM calls without blocking valid questions.

## Insight — why retrieval scores are not higher for fact-based questions

A retrieval score of 0.65 for "Where did Akash study?" is correct behaviour,
not a weakness. Three factors explain why scores are not higher:

### Factor 1 — chunk topic density
A chunk covering multiple sub-topics produces a diffuse embedding.
"Early life and education" covers school, college, location, sports,
debates, and graduation. The embedding is an average of all those signals.
A query about education aligns with only part of it — hence 0.65, not 0.90.

A dedicated "Education" chunk covering only degrees and institutions
would score significantly higher for education queries. This is the
fundamental chunking trade-off:

- Broader chunks → more context for the LLM, weaker retrieval precision
- Narrower chunks → sharper retrieval, less context per retrieved unit

### Factor 2 — query length and semantic surface area
Short queries have sparse embeddings. "Where did Akash study?" shares
limited semantic surface area with the chunk. A longer, richer query
like "Where did Akash Jain complete his schooling and university
education and what degree did he study?" would score higher against
the same chunk — more overlap, tighter vector alignment.

This is why the rewriter helps even for legitimate queries — it increases
semantic surface area, which increases alignment with the chunk.

### Factor 3 — embedding model generalisation
nomic-embed-text is a strong general-purpose model but not fine-tuned
on professional biography content. Domain-specific embedding models
(Voyage AI, OpenAI text-embedding-3-large) produce tighter alignment
for this content type. This is one reason the production stack switches
to a better embedding model.

### Implication for Phase 7
As the knowledge base grows with Perspica articles, chunk granularity
should be revisited. Splitting "Early life and education" into
"Schooling" and "University education" would immediately improve
retrieval precision for education queries without changing any
pipeline code.

## Known limitations

### Rewriter prompt injection risk
The rewriter receives the raw user query inside a prompt. A malicious
query could attempt to override the rewriter's instructions. Phase 3
does not sanitise input. This is acceptable for a personal avatar with
a trusted audience — production deployments should add input validation.

### Section names hardcoded in rewriter
The rewriter prompt contains the knowledge base section names as a
static list. When new sections are added in Phase 7, the rewriter
prompt must be updated manually. A future improvement would read
section names dynamically from the ingested index metadata.

## What Phase 4 will add

- Multi-turn conversation — maintain session history across turns
- Topic shift detection — reset retrieval context when the user changes subject
- Query rewriting using recent history to resolve references like "tell me more about that"