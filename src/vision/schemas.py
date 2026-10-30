"""
src/vision/schemas.py
────────────────────────────────────────────────────────────────
Pydantic schema for the Visual Understanding stage output.

The VisualObservation model deliberately separates:
  1. Directly visible observations (objects, scene, clothing)
  2. Plausible interpretations (what activities might be occurring)
  3. Cultural clues (observable features that *may* indicate culture)
  4. Claims that require external evidence (never asserted as facts)

This separation is the research contribution's core principle:
visual appearance alone does NOT determine cultural identity.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ObservationConfidence(str, Enum):
    HIGH = "high"       # Clearly visible, unambiguous
    MEDIUM = "medium"   # Visible but requires interpretation
    LOW = "low"         # Inferred, may be wrong


class VisualElement(BaseModel):
    """A single observed visual element with confidence and type."""
    description: str = Field(..., description="What is observed.")
    confidence: ObservationConfidence = Field(default=ObservationConfidence.MEDIUM)
    observation_type: str = Field(
        default="direct",
        description="direct | interpreted | requires_evidence"
    )


class VisualObservation(BaseModel):
    """
    Structured output of the Visual Understanding stage.

    Every field is separated by certainty level. The pipeline must
    never treat 'cultural_clues' as confirmed cultural identity.
    """
    # Direct visual observations
    objects_present: list[str] = Field(
        default_factory=list,
        description="List of objects clearly visible in the image."
    )
    people_description: list[str] = Field(
        default_factory=list,
        description="Physical descriptions of people — posture, activity, clothing visible."
    )
    setting: str = Field(
        default="",
        description="Description of the physical environment/scene."
    )
    activities: list[str] = Field(
        default_factory=list,
        description="Observable activities or actions taking place."
    )
    clothing_and_objects: list[str] = Field(
        default_factory=list,
        description="Clothing items, accessories, and significant objects."
    )
    colours_prominent: list[str] = Field(
        default_factory=list,
        description="Dominant colours present in the image."
    )

    # Plausible interpretations (not confirmed facts)
    possible_activities: list[str] = Field(
        default_factory=list,
        description="Activities that could be occurring — inferred, not certain."
    )
    architectural_features: list[str] = Field(
        default_factory=list,
        description="Architectural elements visible (towers, columns, decorations, etc.)."
    )

    # Cultural clues (observable, not conclusive cultural identification)
    cultural_clues: list[str] = Field(
        default_factory=list,
        description=(
            "Observable features that may suggest cultural context — "
            "e.g. script on signs, specific garments, festival decorations. "
            "NEVER used to assert a person's identity."
        )
    )
    potential_cultural_context: Optional[str] = Field(
        default=None,
        description=(
            "A cautious, high-level cultural context suggestion "
            "based only on visible evidence. May be None if unclear."
        )
    )

    # Uncertainty
    uncertainty_notes: list[str] = Field(
        default_factory=list,
        description="What is ambiguous, unclear, or could be misidentified."
    )
    claims_requiring_evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Statements that would require external cultural knowledge "
            "to verify — never asserted as fact by the VLM alone."
        )
    )

    # Metadata
    image_quality: str = Field(default="acceptable", description="good | acceptable | poor")
    raw_vlm_response: Optional[str] = Field(
        default=None,
        description="Raw VLM text response (for debugging). Not shown to users."
    )

    def build_retrieval_query(self) -> str:
        """Build a semantic search query from cultural clues and context."""
        parts = []
        if self.potential_cultural_context:
            parts.append(self.potential_cultural_context)
        parts.extend(self.cultural_clues[:5])
        parts.extend(self.clothing_and_objects[:3])
        parts.extend(self.architectural_features[:3])
        return " ".join(parts) if parts else " ".join(self.objects_present[:5])
