"""
Local embeddings using chromadb's built-in DefaultEmbeddingFunction.

This uses the same all-MiniLM-L6-v2 model as sentence-transformers but via
onnxruntime (already a chromadb dependency) — no PyTorch required.

Embeddings are always local regardless of LLM provider choice,
so the Chroma index never needs rebuilding on a provider switch.
"""
from __future__ import annotations

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from backend.logging_config import get_logger

logger = get_logger(__name__)


class LocalEmbeddings:
    """
    Wrapper around chromadb's DefaultEmbeddingFunction (all-MiniLM-L6-v2 via onnxruntime).
    API is identical to the previous sentence-transformers version.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("Loading local embedding model (onnxruntime)", model=model_name)
        # DefaultEmbeddingFunction ignores model_name arg (always uses all-MiniLM-L6-v2)
        # but we keep the parameter for API compatibility / future swap.
        self._fn = DefaultEmbeddingFunction()
        logger.info("Embedding model ready (onnxruntime)")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings. Returns float vectors."""
        if not texts:
            return []
        return list(self._fn(texts))

    def embed_query(self, query: str) -> list[float]:
        """Single-query embedding for retrieval."""
        return self.embed([query])[0]
