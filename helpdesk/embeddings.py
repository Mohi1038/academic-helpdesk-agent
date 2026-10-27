"""Thin client for Ollama's /api/embed endpoint."""

import requests

from helpdesk import config


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = requests.post(
            f"{config.OLLAMA_API_BASE}/api/embed",
            json={"model": config.EMBED_MODEL, "input": texts[i:i + batch_size]},
            timeout=300,
        )
        resp.raise_for_status()
        vectors.extend(resp.json()["embeddings"])
    return vectors
