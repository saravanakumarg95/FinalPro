"""
src/generation/schemas.py — Pydantic schema for GeneratedStory.
"""

from typing import Optional
from pydantic import BaseModel, Field


class GeneratedStory(BaseModel):
    """Complete output of the story generation stage."""
    story_text: str = Field(..., description="The generated story text.")
    word_count: int = Field(default=0)
    evidence_ids_used: list[str] = Field(
        default_factory=list,
        description="Record IDs of KB records used in generation."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the story not supported by retrieved evidence."
    )
    cultural_elements_included: list[str] = Field(
        default_factory=list,
        description="Cultural elements explicitly included in the story."
    )
    generation_method: str = Field(
        default="proposed",
        description="proposed | baseline_a | baseline_b"
    )
    model_used: str = Field(default="")
    reading_level: str = Field(default="general")
    raw_llm_response: Optional[str] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if self.word_count == 0 and self.story_text:
            self.word_count = len(self.story_text.split())
