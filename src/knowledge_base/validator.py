"""
src/knowledge_base/validator.py
────────────────────────────────────────────────────────────────
Per-record validation logic for the cultural knowledge base.

Validation rules beyond Pydantic schema:
  1. evidence_text must be a substring or paraphrase of description
     (relaxed check: at least 3 words in evidence appear in description)
  2. Tags must not exceed 20 items
  3. Record IDs must be unique (checked at collection level)
  4. Synthetic records must have explicit disclaimer in evidence_text
"""

from src.knowledge_base.schema import Confidence, KBRecord
from src.utils.errors import KBRecordValidationError
from src.utils.logger import get_logger

log = get_logger(__name__)

MAX_TAGS = 20
MIN_EVIDENCE_WORD_OVERLAP = 3


def validate_record(record: KBRecord) -> list[str]:
    """
    Run additional validation checks on a KBRecord beyond Pydantic.

    Args:
        record: A validated KBRecord instance.

    Returns:
        List of warning strings (non-fatal issues).

    Raises:
        KBRecordValidationError: For fatal validation failures.
    """
    warnings: list[str] = []

    # Check tag count
    if len(record.tags) > MAX_TAGS:
        raise KBRecordValidationError(
            f"Record '{record.record_id}' has {len(record.tags)} tags; "
            f"maximum is {MAX_TAGS}."
        )

    # Check evidence_text has meaningful overlap with description
    evidence_words = set(record.evidence_text.lower().split())
    description_words = set(record.description.lower().split())
    overlap = len(evidence_words & description_words)
    if overlap < MIN_EVIDENCE_WORD_OVERLAP:
        warnings.append(
            f"Record '{record.record_id}': evidence_text has low word overlap "
            f"with description ({overlap} shared words). "
            "Ensure evidence_text is derived from or related to description."
        )

    # Synthetic records must have disclaimer visible in evidence_text
    if record.is_synthetic():
        if "synthetic" not in record.evidence_text.lower() and \
           "demonstration" not in record.evidence_text.lower() and \
           "demo" not in record.evidence_text.lower():
            warnings.append(
                f"Record '{record.record_id}' is synthetic_demo but "
                "evidence_text does not mention this. "
                "Consider adding a disclaimer to evidence_text."
            )

    # Warn on NOT_AVAILABLE source URLs for non-synthetic records
    if record.source_url == "NOT_AVAILABLE" and not record.is_synthetic():
        warnings.append(
            f"Record '{record.record_id}' has no source URL. "
            "This reduces traceability. Provide a URL or bibliographic reference."
        )

    return warnings


def validate_record_ids_unique(records: list[KBRecord]) -> None:
    """
    Raise KBRecordValidationError if any record IDs are duplicated.

    Args:
        records: List of KBRecord instances.

    Raises:
        KBRecordValidationError: If duplicate record IDs are found.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for r in records:
        if r.record_id in seen:
            duplicates.append(r.record_id)
        seen.add(r.record_id)
    if duplicates:
        raise KBRecordValidationError(
            f"Duplicate record IDs found: {duplicates}. "
            "Each record must have a unique record_id."
        )
