"""
src/embeddings/adapter.py — EmbeddingAdapter Protocol.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Interface for embedding model adapters."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns list of embedding vectors."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string. Returns one embedding vector."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...
