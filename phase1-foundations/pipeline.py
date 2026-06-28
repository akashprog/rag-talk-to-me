# pipeline.py
# Phase 1 — minimal RAG pipeline.
# No vector database. No chunking. No UI.
# Just: embed → retrieve → answer. Every step visible.

import requests
import numpy as np
import config

def get_embedding(text: str) -> np.ndarray:
    """
    Converts a string into a vector (embedding) using the configured provider.
    Returns a numpy array of floats.
    """
    if config.EMBEDDING_PROVIDER == "ollama":
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/embeddings",
            json={
                "model": config.OLLAMA_EMBEDDING_MODEL,
                "prompt": text
            }
        )
        response.raise_for_status()
        embedding = response.json()["embedding"]
        return np.array(embedding, dtype=np.float32)

    raise ValueError(f"Unsupported embedding provider: {config.EMBEDDING_PROVIDER}")

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Measures how similar two vectors are by computing the cosine of the angle between them.
    Returns a float between -1 and 1.
    1.0  = identical direction (very similar meaning)
    0.0  = perpendicular (unrelated)
    -1.0 = opposite direction (opposing meaning)
    """
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    return float(dot_product / (magnitude_a * magnitude_b))

def retrieve(query: str, documents: list[str], top_k: int = config.TOP_K) -> list[dict]:
    """
    Given a query and a list of documents, returns the top_k most relevant documents.
    
    Steps:
    1. Embed the query
    2. Embed each document
    3. Compute cosine similarity between query and each document
    4. Sort by similarity score, highest first
    5. Return top_k results with their scores
    """
    query_embedding = get_embedding(query)

    scored_documents = []
    for doc in documents:
        doc_embedding = get_embedding(doc)
        score = cosine_similarity(query_embedding, doc_embedding)
        scored_documents.append({
            "text": doc,
            "score": score
        })

    scored_documents.sort(key=lambda x: x["score"], reverse=True)

    return scored_documents[:top_k]
    

def answer(query: str, retrieved_docs: list[dict]) -> str:
    """
    Passes the query and retrieved documents to the LLM and returns its response.
    The LLM only sees what was retrieved — nothing else.
    This is the grounding constraint: no retrieved context = no answer.
    """
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])

    prompt = f"""You are a professional assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say "I don't have enough information on that."
Do not use any knowledge outside of the provided context.

Context:
{context}

Question:
{query}

Answer:"""

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
        return response.json()["response"]

    raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")

if __name__ == "__main__":
    # Hardcoded knowledge base — 5 documents about Akash / Perspica
    # In Phase 2 these will be replaced by real ingested content
    documents = [
        "Akash Jain is a Senior Solutions Architect with 13 years of experience in data and AI, based in Abu Dhabi.",
        "Perspica is Akash's blog covering data architecture, AI engineering, and enterprise transformation.",
        "Akash has worked extensively with Cloudera, Confluent, Redis, IBM Watsonx.data, and other technologies on large enterprise engagements.",
        "The Data Mesh Diaries is a 10-part series on Perspica exploring the philosophy and implementation of data mesh.",
        "Akash has worked at Knowesis for 6 years from 2016 to 2022. He then joined GBM. He currently works at GBM in Abu Dhabi."
    ]

    query = "What does Akash like to eat?"

    print(f"\nQuery: {query}\n")

    results = retrieve(query, documents)

    print("Retrieved chunks:")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['text']}")

    print("\nGenerating answer...\n")
    response = answer(query, results)
    print(f"Answer: {response}")