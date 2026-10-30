"""
src/embeddings/minilm_embedder.py
────────────────────────────────────────────────────────────────
all-MiniLM-L6-v2 sentence-transformer embedding adapter.

- Model: sentence-transformers/all-MiniLM-L6-v2
- Dimension: 384
- Runs locally on CPU (no API key needed)
- ~22 MB model size
"""

from functools import lru_cache
from src.utils.errors import EmbeddingModelLoadError
from src.utils.logger import get_logger

log = get_logger(__name__)
_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the MiniLM model (called once at startup)."""
    try:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedding model: {_MODEL_NAME}")
        model = SentenceTransformer(_MODEL_NAME)
        log.info("Embedding model loaded successfully.")
        return model
    except Exception as e:
        raise EmbeddingModelLoadError(
            f"Failed to load embedding model '{_MODEL_NAME}': {e}"
        ) from e


class MiniLMEmbedder:
    """Sentence-transformer embedding adapter using all-MiniLM-L6-v2."""

    @property
    def dimension(self) -> int:
        return 384

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
