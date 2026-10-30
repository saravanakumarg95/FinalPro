"""
src/retrieval/schemas.py — Pydantic schemas for the retrieval stage.
"""

from typing import Optional
from pydantic import BaseModel, Field


class RetrievedEvidence(BaseModel):
    """A single retrieved cultural knowledge base record with retrieval metadata."""
    record_id: str
    topic: str
    cultural_region: str
    evidence_text: str
    source_title: str
    source_url: str
    confidence: str
    known_limitations: str
    tags: list[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0,
                                   description="Semantic similarity score [0, 1].")
    retrieval_rank: int = Field(default=0, description="1-indexed rank in retrieved results.")
    is_synthetic: bool = Field(default=False)

    def to_evidence_block(self) -> str:
        """Format for inclusion in LLM prompts."""
        synthetic_note = "\n⚠️ SYNTHETIC — NOT A VERIFIED FACT" if self.is_synthetic else ""
        return (
            f"[Evidence {self.retrieval_rank}] {self.topic} "
            f"(relevance: {self.relevance_score:.2f})\n"
            f"Region: {self.cultural_region}\n"
            f"Evidence: {self.evidence_text}\n"
            f"Source: {self.source_title} | {self.source_url}\n"
            f"Confidence: {self.confidence}\n"
            f"Limitations: {self.known_limitations}"
            f"{synthetic_note}"
        )


class RetrievalResult(BaseModel):
    """Complete output of one retrieval operation."""
    query_used: str
    evidence_found: bool
    retrieved_records: list[RetrievedEvidence] = Field(default_factory=list)
    num_records_in_index: int = Field(default=0)
    retrieval_message: str = Field(default="")

    @property
    def has_synthetic(self) -> bool:
        return any(e.is_synthetic for e in self.retrieved_records)

    @property
    def top_evidence(self) -> Optional[RetrievedEvidence]:
        return self.retrieved_records[0] if self.retrieved_records else None
