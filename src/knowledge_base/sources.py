"""
src/knowledge_base/sources.py
────────────────────────────────────────────────────────────────
Source citation utilities for the knowledge base.

These helpers format citations for display in the UI and
for inclusion in the generated story's evidence section.
"""

from src.knowledge_base.schema import Confidence, KBRecord


def format_citation(record: KBRecord) -> str:
    """
    Return a formatted citation string for display.

    Args:
        record: A validated KBRecord.

    Returns:
        Human-readable citation string.
    """
    parts = [record.source_title]
    if record.source_url != "NOT_AVAILABLE":
        parts.append(f"<{record.source_url}>")
    confidence_label = f"[{record.confidence}]"
    if record.is_synthetic():
        confidence_label = "[SYNTHETIC DEMO — NOT A VERIFIED FACT]"
    return f"{' '.join(parts)} {confidence_label}"


def format_evidence_block(record: KBRecord, rank: int = 0) -> str:
    """
    Return a formatted evidence block for inclusion in prompts.

    Args:
        record: A validated KBRecord.
        rank: Retrieval rank (1-indexed, optional).

    Returns:
        Formatted evidence string for LLM prompts.
    """
    rank_label = f"[Evidence {rank}] " if rank > 0 else ""
    synthetic_warning = ""
    if record.is_synthetic():
        synthetic_warning = "\n⚠️  SYNTHETIC RECORD — NOT A VERIFIED FACT. Do not treat as real cultural evidence."

    return (
        f"{rank_label}Topic: {record.topic}\n"
        f"Region: {record.cultural_region}\n"
        f"Evidence: {record.evidence_text}\n"
        f"Source: {record.source_title}\n"
        f"Confidence: {record.confidence}\n"
        f"Limitations: {record.known_limitations}"
        f"{synthetic_warning}"
    )


def get_synthetic_warning() -> str:
    """Return a standard warning string to display alongside synthetic records."""
    return (
        "⚠️ WARNING: One or more retrieved records are synthetic demonstration records "
        "created for system testing only. They are NOT verified cultural facts. "
        "Any story content derived from these records should be treated as fictional."
    )
