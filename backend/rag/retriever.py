"""
Retriever: queries the Chroma vector store and returns top-k chunks.
"""
from __future__ import annotations

import chromadb
from backend.config import settings
from backend.rag.embeddings import LocalEmbeddings
from backend.logging_config import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.chroma_path)
        self._embedder = LocalEmbeddings(settings.embeddings_model)
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            try:
                self._collection = self._client.get_collection(settings.collection_name)
                logger.info(
                    "Opened Chroma collection",
                    collection=settings.collection_name,
                    count=self._collection.count(),
                )
            except Exception as exc:
                logger.warning(
                    "Chroma collection not found — run ingest first.",
                    collection=settings.collection_name,
                    error=str(exc),
                )
                raise RuntimeError(
                    "RAG index not found. Run: python -m backend.rag.ingest"
                ) from exc
        return self._collection

    def query(self, text: str, k: int | None = None, where: dict | None = None) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            text:  The query string.
            k:     Number of results to return (defaults to settings.rag_top_k).
            where: Optional ChromaDB metadata filter, e.g.
                   {"source": {"$in": ["Davinci_Resolve_Scripting_API_DOC.txt"]}}

        Returns a list of dicts:
            {
                "text":   str,   # chunk content
                "source": str,   # source filename
                "score":  float, # distance (lower = more similar)
                "idx":    int,   # chunk index
            }
        """
        top_k = k or settings.rag_top_k
        collection = self._get_collection()
        query_vec = self._embedder.embed_query(text)

        query_kwargs: dict = {
            "query_embeddings": [query_vec],
            "n_results": min(top_k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":   doc,
                "source": meta.get("source", "unknown"),
                "score":  round(dist, 4),
                "idx":    meta.get("chunk_idx", -1),
            })

        logger.debug("Retrieval complete", query=text[:80], results=len(chunks))
        return chunks

