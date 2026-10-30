"""
src/knowledge_base/schema.py
────────────────────────────────────────────────────────────────
Pydantic schema for Cultural Knowledge Base records.

Design notes:
  - Every field is validated at ingestion time.
  - 'confidence' and 'source_type' use enums for safety.
  - Synthetic / demonstration records are explicitly labelled.
  - The 'known_limitations' field is required to prevent
    records from being presented as unconditionally authoritative.
  - 'source_url' accepts "NOT_AVAILABLE" as a fallback but
    real sources should always provide a URL or bibliographic ref.
"""

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class SourceType(str, Enum):
    """Classifies the origin of the knowledge record."""

    ACADEMIC = "academic"
    GOVERNMENT = "government"
    CULTURAL_ORG = "cultural_org"
    ENCYCLOPEDIC = "encyclopedic"
    NEWS = "news"
    SYNTHETIC_DEMO = "synthetic_demo"


class Confidence(str, Enum):
    """
    Confidence level of the record's claims.

    - verified: traceable to a named, stable, primary source.
    - plausible: from a secondary or summarised source; likely correct
                 but not independently verified.
    - synthetic_demo: a demonstration record created for development
                      purposes; NOT a verified fact.
    """

    VERIFIED = "verified"
    PLAUSIBLE = "plausible"
    SYNTHETIC_DEMO = "synthetic_demo"


class KBRecord(BaseModel):
    """
    A single cultural knowledge base record.

    Each record represents one retrievable cultural fact or
    description that the retrieval pipeline can return as evidence
    for story generation.
    """

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the record.",
    )
    topic: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Short label for the cultural topic or practice.",
    )
    cultural_region: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description=(
            "Geographic or community context. "
            "Use documented scope, not assumptions about individuals."
        ),
    )
    description: str = Field(
        ...,
        min_length=20,
        max_length=5000,
        description="Paragraph description of the cultural topic.",
    )
    evidence_text: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description=(
            "Key text chunk used for embedding and retrieval. "
            "Should be the most informative excerpt from the description."
        ),
    )
    source_title: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Publication, website, or document name.",
    )
    source_url: str = Field(
        ...,
        description=(
            "URL or bibliographic reference. "
            "Use 'NOT_AVAILABLE' only when no URL exists."
        ),
    )
    source_type: SourceType = Field(
        ...,
        description="Category of the source.",
    )
    confidence: Confidence = Field(
        ...,
        description="Confidence level of the record's claims.",
    )
    known_limitations: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description=(
            "Known limitations, caveats, or scope restrictions for this record. "
            "Required — no record may claim unconditional accuracy."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Keywords for filtering and display.",
    )

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, v: str) -> str:
        v = v.strip()
        if v == "NOT_AVAILABLE":
            return v
        if not (v.startswith("http://") or v.startswith("https://")):
            # Accept bibliographic references (not starting with http)
            # as long as they are non-empty and reasonably long
            if len(v) < 10:
                raise ValueError(
                    "source_url must be a valid URL, 'NOT_AVAILABLE', "
                    "or a bibliographic reference of at least 10 characters."
                )
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v) -> list[str]:
        if isinstance(v, str):
            # Accept comma-separated string as tags
            return [t.strip() for t in v.split(",") if t.strip()]
        return v

    @model_validator(mode="after")
    def synthetic_records_must_declare_limitations(self) -> "KBRecord":
        """Synthetic records must explicitly state they are not verified facts."""
        if self.confidence == Confidence.SYNTHETIC_DEMO:
            if "synthetic" not in self.known_limitations.lower():
                raise ValueError(
                    "Records with confidence='synthetic_demo' must state "
                    "'synthetic' in known_limitations to prevent misrepresentation."
                )
        return self

    model_config = {"use_enum_values": True}

    def is_synthetic(self) -> bool:
        """Return True if this is a synthetic demonstration record."""
        return self.confidence == Confidence.SYNTHETIC_DEMO.value or self.confidence == Confidence.SYNTHETIC_DEMO

    def citation_string(self) -> str:
        """Return a human-readable citation string."""
        url_part = f" ({self.source_url})" if self.source_url != "NOT_AVAILABLE" else ""
        return f"{self.source_title}{url_part}"
