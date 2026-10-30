"""
src/retrieval/engine.py
────────────────────────────────────────────────────────────────
ChromaDB-based retrieval engine with a FAISS fallback interface.

Responsibilities:
  1. Index KB records using an EmbeddingAdapter.
  2. Accept a query string and return top-k RetrievedEvidence.
  3. Handle empty KB and no-match scenarios gracefully.
  4. Persist the ChromaDB index to disk for reuse.
"""

import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.knowledge_base.schema import KBRecord
from src.retrieval.schemas import RetrievedEvidence, RetrievalResult
from src.utils.errors import EmptyKnowledgeBaseError, NoEvidenceFoundError, VectorStoreError
from src.utils.logger import get_logger

log = get_logger(__name__)

COLLECTION_NAME = "tamil_cultural_kb"
MIN_RELEVANCE_SCORE = 0.0  # Accept all results; let callers filter


class ChromaRetrievalEngine:
    """
    ChromaDB-backed semantic retrieval engine.

    Usage:
        engine = ChromaRetrievalEngine(embedder, persist_path)
        engine.build_index(records)
        result = engine.retrieve("Tamil harvest festival clay pot", top_k=5)
    """

    def __init__(self, embedder, persist_path: Optional[Path] = None):
        """
        Args:
            embedder: Any EmbeddingAdapter (MiniLM or Mock).
            persist_path: Directory to persist the ChromaDB index.
                          If None, runs in-memory (for tests).
        """
        self.embedder = embedder
        self.persist_path = persist_path
        self._collection = None
        self._client = None
        self._num_records = 0
        self._setup_client()

    def _setup_client(self):
        """Initialise ChromaDB client (persistent or in-memory)."""
        try:
            if self.persist_path:
                self.persist_path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self.persist_path))
                log.info(f"ChromaDB persistent client at: {self.persist_path}")
            else:
                self._client = chromadb.Client()
                log.info("ChromaDB in-memory client initialised.")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialise ChromaDB: {e}") from e

    def build_index(self, records: list[KBRecord], force_rebuild: bool = False) -> int:
        """
        Index a list of KB records.

        Args:
            records: Validated KBRecord list.
            force_rebuild: If True, delete and rebuild the collection.

        Returns:
            Number of records indexed.
        """
        if not records:
            raise EmptyKnowledgeBaseError("Cannot build index from empty record list.")

        # Delete existing collection if rebuild requested
        if force_rebuild:
            try:
                self._client.delete_collection(COLLECTION_NAME)
                log.info(f"Deleted existing collection '{COLLECTION_NAME}' for rebuild.")
            except Exception:
                pass

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Check if already indexed
        existing_count = self._collection.count()
        if existing_count > 0 and not force_rebuild:
            log.info(f"Collection '{COLLECTION_NAME}' already has {existing_count} records. Skipping re-index.")
            self._num_records = existing_count
            return existing_count

        # Embed and index
        texts = [r.evidence_text for r in records]
        ids = [r.record_id for r in records]
        metadatas = [
            {
                "topic": r.topic,
                "cultural_region": r.cultural_region,
                "source_title": r.source_title,
                "source_url": r.source_url,
                "confidence": r.confidence if isinstance(r.confidence, str) else r.confidence.value,
                "known_limitations": r.known_limitations,
                "tags": ",".join(r.tags),
                "is_synthetic": str(r.is_synthetic()),
            }
            for r in records
        ]
        documents = texts

        log.info(f"Embedding {len(records)} records...")
        embeddings = self.embedder.embed_texts(texts)

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            self._num_records = len(records)
            log.info(f"Indexed {self._num_records} records into '{COLLECTION_NAME}'.")
            return self._num_records
        except Exception as e:
            raise VectorStoreError(f"Failed to add records to ChromaDB: {e}") from e

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve top-k culturally relevant records for a query.

        Args:
            query: Semantic search query string.
            top_k: Number of results to return.

        Returns:
            RetrievalResult with evidence records and metadata.
        """
        if not query.strip():
            return RetrievalResult(
                query_used=query,
                evidence_found=False,
                retrieval_message="Empty query — no retrieval performed.",
                num_records_in_index=self._num_records,
            )

        if self._collection is None or self._collection.count() == 0:
            return RetrievalResult(
                query_used=query,
                evidence_found=False,
                retrieval_message="Knowledge base index is empty. Run build_index() first.",
                num_records_in_index=0,
            )

        try:
            query_embedding = self.embedder.embed_query(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise VectorStoreError(f"ChromaDB query failed: {e}") from e

        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        if not ids:
            return RetrievalResult(
                query_used=query,
                evidence_found=False,
                retrieval_message="No matching records found.",
                num_records_in_index=self._num_records,
            )

        evidence_list = []
        for rank, (rec_id, dist, meta, doc) in enumerate(
            zip(ids, distances, metadatas, documents), start=1
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score [0, 1]
            similarity = max(0.0, 1.0 - (dist / 2.0))

            evidence_list.append(RetrievedEvidence(
                record_id=rec_id,
                topic=meta.get("topic", ""),
                cultural_region=meta.get("cultural_region", ""),
                evidence_text=doc,
                source_title=meta.get("source_title", ""),
                source_url=meta.get("source_url", "NOT_AVAILABLE"),
                confidence=meta.get("confidence", "plausible"),
                known_limitations=meta.get("known_limitations", ""),
                tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                relevance_score=round(similarity, 4),
                retrieval_rank=rank,
                is_synthetic=meta.get("is_synthetic", "False") == "True",
            ))

        return RetrievalResult(
            query_used=query,
            evidence_found=True,
            retrieved_records=evidence_list,
            num_records_in_index=self._num_records,
            retrieval_message=f"Retrieved {len(evidence_list)} records (top-{top_k}).",
        )

    def get_index_size(self) -> int:
        """Return number of records currently indexed."""
        if self._collection is None:
            return 0
        return self._collection.count()
