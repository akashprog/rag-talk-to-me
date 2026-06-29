# utils.py
# Shared utility functions used across the Phase 3 pipeline.
# Import from here — never duplicate these in other files.

import numpy as np
import requests
import config


def get_embedding(text: str) -> np.ndarray:
    """
    Converts text into a vector using the configured embedding provider.
    Returns a numpy array of float32 values.
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