"""
src/embeddings/mock_embedder.py
────────────────────────────────────────────────────────────────
Deterministic mock embedder for tests.

Returns reproducible pseudo-random embeddings based on text hash.
Ensures tests are fast and require no model downloads.
"""

import hashlib


class MockEmbedder:
    """Deterministic mock embedding adapter for testing."""

    DIMENSION = 384

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_embed(query)

    def _hash_embed(self, text: str) -> list[float]:
        """Create a deterministic pseudo-embedding from text hash."""
        digest = hashlib.sha256(text.encode()).digest()
        # Repeat digest to fill 384 dimensions, then normalise
        raw = list(digest * (self.DIMENSION // len(digest) + 1))[:self.DIMENSION]
        # Convert bytes to floats in [-1, 1]
        floats = [(b / 127.5) - 1.0 for b in raw]
        # L2-normalise
        norm = sum(x ** 2 for x in floats) ** 0.5
        return [x / norm for x in floats] if norm > 0 else floats
