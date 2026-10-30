"""
src/knowledge_base/ingestion.py
────────────────────────────────────────────────────────────────
Knowledge base ingestion pipeline.

Responsibilities:
  1. Load raw JSONL records from disk.
  2. Validate each record against KBRecord schema.
  3. Run additional validation checks (validator.py).
  4. Deduplicate by record_id.
  5. Return a list of validated KBRecord objects.

This module does NOT embed or index records.
Embedding and indexing happen in src/embeddings/ and src/retrieval/.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.knowledge_base.schema import KBRecord
from src.knowledge_base.validator import validate_record, validate_record_ids_unique
from src.utils.errors import (
    EmptyKnowledgeBaseError,
    KBIngestionError,
    KBRecordValidationError,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_kb_from_jsonl(
    path: Path,
    skip_invalid: bool = False,
    skip_synthetic: bool = False,
) -> list[KBRecord]:
    """
    Load and validate knowledge base records from a JSONL file.

    Each line must be a JSON object matching the KBRecord schema.
    Empty lines and comment lines starting with '#' are skipped.

    Args:
        path: Path to the JSONL file.
        skip_invalid: If True, log and skip invalid records instead
                      of raising an error. Useful for data exploration.
        skip_synthetic: If True, exclude synthetic_demo records from output.

    Returns:
        List of validated KBRecord objects.

    Raises:
        KBIngestionError: If the file cannot be opened.
        KBRecordValidationError: If a record fails validation and skip_invalid=False.
        EmptyKnowledgeBaseError: If the file is empty or all records are invalid.
    """
    path = Path(path)
    if not path.exists():
        raise KBIngestionError(f"Knowledge base file not found: {path}")

    records: list[KBRecord] = []
    errors: list[str] = []
    line_count = 0

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                line_count += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    msg = f"Line {line_num}: JSON parse error — {e}"
                    if skip_invalid:
                        log.warning(msg)
                        errors.append(msg)
                        continue
                    raise KBIngestionError(msg) from e

                try:
                    record = KBRecord(**data)
                except ValidationError as e:
                    msg = f"Line {line_num}: Schema validation failed — {e.error_count()} error(s): {e}"
                    if skip_invalid:
                        log.warning(msg)
                        errors.append(msg)
                        continue
                    raise KBRecordValidationError(msg) from e

                # Additional validation
                try:
                    warnings = validate_record(record)
                    for w in warnings:
                        log.warning(w)
                except KBRecordValidationError as e:
                    msg = f"Line {line_num}: Additional validation failed — {e}"
                    if skip_invalid:
                        log.warning(msg)
                        errors.append(msg)
                        continue
                    raise

                if skip_synthetic and record.is_synthetic():
                    log.debug(f"Skipping synthetic record: {record.record_id}")
                    continue

                records.append(record)

    except OSError as e:
        raise KBIngestionError(f"Cannot read file {path}: {e}") from e

    log.info(
        f"Ingestion complete: {len(records)} valid records loaded, "
        f"{len(errors)} skipped, from {line_count} non-empty lines."
    )

    if not records:
        raise EmptyKnowledgeBaseError(
            f"No valid records found in {path}. "
            f"Processed {line_count} lines; {len(errors)} were invalid."
        )

    # Check for duplicate IDs across the loaded set
    validate_record_ids_unique(records)

    return records


def load_kb_from_dict_list(
    data: list[dict],
    skip_invalid: bool = False,
) -> list[KBRecord]:
    """
    Load KB records from a list of dictionaries (e.g., for testing).

    Args:
        data: List of raw dictionaries.
        skip_invalid: If True, skip invalid records.

    Returns:
        List of validated KBRecord objects.

    Raises:
        EmptyKnowledgeBaseError: If no valid records are found.
    """
    records: list[KBRecord] = []
    errors: list[str] = []

    for i, raw in enumerate(data):
        try:
            record = KBRecord(**raw)
            warnings = validate_record(record)
            for w in warnings:
                log.warning(w)
            records.append(record)
        except (ValidationError, KBRecordValidationError) as e:
            msg = f"Record {i}: validation failed — {e}"
            if skip_invalid:
                log.warning(msg)
                errors.append(msg)
            else:
                raise KBRecordValidationError(msg) from e

    if not records:
        raise EmptyKnowledgeBaseError("No valid records found in provided data.")

    validate_record_ids_unique(records)
    return records
