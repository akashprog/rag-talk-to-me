# retriever.py
# Handles embedding the query and retrieving chunks from Pinecone.
# Single responsibility: given a query string, return ranked chunks.
# Does not rewrite queries, does not call the LLM, does not check confidence.

from pinecone import Pinecone
import config
from utils import get_embedding


def retrieve(query: str) -> list[dict]:
    """
    Embeds the query and retrieves the top_k most similar chunks from Pinecone.
    Returns a list of dicts with 'section', 'text', and 'score' keys.
    Sorted by score descending — highest confidence first.
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