# pipeline.py
# Phase 2 RAG pipeline.
# Retrieval is now backed by Pinecone — no in-memory cosine similarity.
# Ingestion is separate (ingest.py) — this file only handles querying.

import requests
import numpy as np
from pinecone import Pinecone
import config

def get_embedding(text: str) -> np.ndarray:
    """
    Converts text into a vector using the configured embedding provider.
    Identical to ingest.py — kept separate so each file is self-contained.
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


def retrieve(query: str) -> list[dict]:
    """
    Embeds the query and retrieves the top_k most similar chunks from Pinecone.
    Returns a list of dicts with 'section', 'text', and 'score' keys.
    """
    query_embedding = get_embedding(query)

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=config.TOP_K,
        include_metadata=True
    )

    retrieved = []
    for match in results["matches"]:
        retrieved.append({
            "section": match["metadata"]["section"],
            "text": match["metadata"]["text"],
            "score": match["score"]
        })

    return retrieved


def answer(query: str, retrieved_docs: list[dict]) -> str:
    """
    Passes the query and retrieved chunks to the LLM.
    The LLM answers strictly from retrieved context.
    """
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])

    prompt = f"""You are a professional AI avatar representing Akash Jain.
Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say "I don't have enough information on that."
Do not use any knowledge outside of the provided context.
Speak in third person — refer to Akash by name, not as "I".

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
    test_queries = [
        "Where did Akash study?",
        "What technologies does Akash work with?",
        "What is Perspica?",
        "What is Akash's favourite food?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retrieve(query)

        print("Retrieved chunks:")
        for r in results:
            print(f"  [{r['score']:.4f}] {r['section']}")

        response = answer(query, results)
        print(f"Answer: {response}")
        print("-" * 60)