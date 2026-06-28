# ingest.py
# Reads the profile markdown, splits it into chunks by section,
# embeds each chunk, and stores it in Pinecone.
# Run this once to build the knowledge base.
# Re-run whenever the profile content changes.

import re
import requests
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import config

def chunk_markdown(filepath: str) -> list[dict]:
    """
    Splits a markdown file into chunks based on ## headings.
    Each ## section becomes one chunk.
    Returns a list of dicts with 'id', 'section', and 'text' keys.
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Split on ## headings — each match starts a new chunk
    # re.split keeps the delimiter when wrapped in a capture group
    parts = re.split(r"(^## .+$)", content, flags=re.MULTILINE)

    chunks = []
    chunk_id = 0

    # parts looks like: [preamble, ## heading, body, ## heading, body, ...]
    # We step through in pairs of heading + body
    i = 1
    while i < len(parts) - 1:
        heading = parts[i].strip()
        body = parts[i + 1].strip()

        if body:
            section_name = heading.replace("## ", "").strip()
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "section": section_name,
                "text": f"{heading}\n\n{body}"
            })
            chunk_id += 1

        i += 2

    return chunks

def get_embedding(text: str) -> np.ndarray:
    """
    Converts text into a vector using the configured embedding provider.
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

def setup_pinecone_index():
    """
    Connects to Pinecone and creates the index if it doesn't exist.
    Returns the index object ready for upsert or query operations.
    """
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if config.PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating index: {config.PINECONE_INDEX_NAME}")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.get_embedding_dimension(),
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    else:
        print(f"Index already exists: {config.PINECONE_INDEX_NAME}")

    return pc.Index(config.PINECONE_INDEX_NAME)

def ingest(filepath: str):
    """
    Full ingestion pipeline:
    1. Chunk the markdown file
    2. Embed each chunk
    3. Upsert into Pinecone with metadata
    """
    print(f"Reading and chunking: {filepath}")
    chunks = chunk_markdown(filepath)
    print(f"Found {len(chunks)} chunks\n")

    index = setup_pinecone_index()

    vectors = []
    for chunk in chunks:
        print(f"Embedding: {chunk['section']}")
        embedding = get_embedding(chunk["text"])

        vectors.append({
            "id": chunk["id"],
            "values": embedding.tolist(),
            "metadata": {
                "section": chunk["section"],
                "text": chunk["text"]
            }
        })

    print(f"\nUpserting {len(vectors)} vectors to Pinecone...")
    index.upsert(vectors=vectors)
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest("knowledge_base/profile.md")