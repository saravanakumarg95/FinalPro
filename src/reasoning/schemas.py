"""
src/reasoning/schemas.py — Pydantic schema for the ReasoningTrace.
The ReasoningTrace is the core inspectable intermediate representation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class EvidenceAlignment(BaseModel):
    """How well a retrieved evidence record aligns with visual observations."""
    evidence_id: str
    evidence_topic: str
    alignment_score: float = Field(ge=0.0, le=1.0)
    aligned_observations: list[str] = Field(default_factory=list)
    misaligned_aspects: list[str] = Field(default_factory=list)
    use_in_story: bool = True


class StoryOutline(BaseModel):
    """Structured story plan produced before generation."""
    setting: str
    main_character: str
    cultural_elements_to_use: list[str]
    evidence_ids_supporting: list[str]
    narrative_arc: str
    tone: str = "respectful and engaging"
    cultural_claims_to_avoid: list[str] = Field(default_factory=list)
    approximate_word_count: int = 400


class ReasoningTrace(BaseModel):
    """
    Complete inspectable reasoning trace for one image processing run.

    This is the research contribution's core artefact — every decision
    made by the pipeline is documented here for audit and evaluation.
    """
    # Step 1 — Observation summary
    observation_summary: str = ""
    key_cultural_clues: list[str] = Field(default_factory=list)

    # Step 2 — Retrieved evidence (IDs only; full records in retrieval_result)
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    retrieval_query_used: str = ""

    # Step 3 — Evidence-observation alignment
    evidence_alignments: list[EvidenceAlignment] = Field(default_factory=list)
    evidence_used_ids: list[str] = Field(default_factory=list)  # After filtering

    # Step 4 — Uncertainty analysis
    uncertainties_identified: list[str] = Field(default_factory=list)
    contradictions_found: list[str] = Field(default_factory=list)
    evidence_coverage: str = "sufficient"  # sufficient | partial | insufficient

    # Step 5 — Story outline
    story_outline: Optional[StoryOutline] = None

    # Step 6 — Pre-generation consistency check
    pre_generation_warnings: list[str] = Field(default_factory=list)

    # Metadata
    pipeline_version: str = "1.0"
    total_evidence_retrieved: int = 0
    total_evidence_used: int = 0
