"""
tests/unit/test_kb_ingestion.py
────────────────────────────────────────────────────────────────
Unit tests for knowledge base schema, ingestion, and validation.

Tests cover:
  - Valid record creation
  - Invalid schema fields
  - Synthetic record requirements
  - Ingestion from JSONL (valid, invalid, empty, duplicates)
  - Ingestion from dict list
  - Duplicate record ID detection
  - skip_invalid mode
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.knowledge_base.ingestion import load_kb_from_dict_list, load_kb_from_jsonl
from src.knowledge_base.schema import Confidence, KBRecord, SourceType
from src.knowledge_base.validator import validate_record, validate_record_ids_unique
from src.utils.errors import (
    EmptyKnowledgeBaseError,
    KBIngestionError,
    KBRecordValidationError,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_record_data():
    return {
        "record_id": "test-001",
        "topic": "Test Festival",
        "cultural_region": "Test Region, Country",
        "description": "A test cultural festival for unit testing purposes. It involves gathering, celebration, and traditional rituals.",
        "evidence_text": "The Test Festival involves gathering and celebration with traditional rituals.",
        "source_title": "Test Source Title",
        "source_url": "https://example.com/test-festival",
        "source_type": "encyclopedic",
        "confidence": "verified",
        "known_limitations": "This is a test record with limited scope.",
        "tags": ["test", "festival"],
    }


@pytest.fixture
def synthetic_record_data():
    return {
        "record_id": "syn-test-001",
        "topic": "Synthetic Test Topic",
        "cultural_region": "Fictional Region",
        "description": "A synthetic demonstration record for testing. This is not a verified cultural fact and exists solely for system testing.",
        "evidence_text": "Synthetic demonstration record for testing only. This record is fictional.",
        "source_title": "Synthetic Demonstration Source",
        "source_url": "NOT_AVAILABLE",
        "source_type": "synthetic_demo",
        "confidence": "synthetic_demo",
        "known_limitations": "This is a synthetic record. It is not a real cultural fact.",
        "tags": ["synthetic", "test"],
    }


# ── KBRecord Schema Tests ──────────────────────────────────────────────────────

class TestKBRecordSchema:
    def test_valid_record_creates_successfully(self, valid_record_data):
        record = KBRecord(**valid_record_data)
        assert record.record_id == "test-001"
        assert record.topic == "Test Festival"

    def test_missing_required_field_raises(self, valid_record_data):
        del valid_record_data["topic"]
        with pytest.raises(ValidationError):
            KBRecord(**valid_record_data)

    def test_topic_too_short_raises(self, valid_record_data):
        valid_record_data["topic"] = "X"
        with pytest.raises(ValidationError):
            KBRecord(**valid_record_data)

    def test_description_too_short_raises(self, valid_record_data):
        valid_record_data["description"] = "Short"
        with pytest.raises(ValidationError):
            KBRecord(**valid_record_data)

    def test_evidence_text_too_short_raises(self, valid_record_data):
        valid_record_data["evidence_text"] = "Short"
        with pytest.raises(ValidationError):
            KBRecord(**valid_record_data)

    def test_invalid_source_type_raises(self, valid_record_data):
        valid_record_data["source_type"] = "invalid_type"
        with pytest.raises(ValidationError):
            KBRecord(**valid_record_data)

    def test_invalid_confidence_raises(self, valid_record_data):
        valid_record_data["confidence"] = "uncertain"
        with pytest.raises(ValidationError):
            KBRecord(**valid_record_data)

    def test_not_available_url_accepted(self, valid_record_data):
        valid_record_data["source_url"] = "NOT_AVAILABLE"
        record = KBRecord(**valid_record_data)
        assert record.source_url == "NOT_AVAILABLE"

    def test_valid_https_url_accepted(self, valid_record_data):
        record = KBRecord(**valid_record_data)
        assert record.source_url.startswith("https://")

    def test_tags_as_comma_string_converted_to_list(self, valid_record_data):
        valid_record_data["tags"] = "one,two, three"
        record = KBRecord(**valid_record_data)
        assert "one" in record.tags
        assert "two" in record.tags
        assert "three" in record.tags

    def test_is_synthetic_false_for_verified(self, valid_record_data):
        record = KBRecord(**valid_record_data)
        assert record.is_synthetic() is False

    def test_citation_string_includes_source_title(self, valid_record_data):
        record = KBRecord(**valid_record_data)
        assert "Test Source Title" in record.citation_string()


class TestSyntheticRecordValidation:
    def test_valid_synthetic_record_accepted(self, synthetic_record_data):
        record = KBRecord(**synthetic_record_data)
        assert record.is_synthetic() is True

    def test_synthetic_record_without_synthetic_in_limitations_raises(self, synthetic_record_data):
        synthetic_record_data["known_limitations"] = "No disclaimer here."
        with pytest.raises(ValidationError):
            KBRecord(**synthetic_record_data)

    def test_is_synthetic_true_for_synthetic_demo(self, synthetic_record_data):
        record = KBRecord(**synthetic_record_data)
        assert record.is_synthetic() is True


# ── Validator Tests ────────────────────────────────────────────────────────────

class TestValidator:
    def test_valid_record_returns_no_fatal_errors(self, valid_record_data):
        record = KBRecord(**valid_record_data)
        warnings = validate_record(record)
        assert isinstance(warnings, list)

    def test_not_available_url_generates_warning(self, valid_record_data):
        valid_record_data["source_url"] = "NOT_AVAILABLE"
        record = KBRecord(**valid_record_data)
        warnings = validate_record(record)
        assert any("source URL" in w for w in warnings)

    def test_duplicate_ids_raises_error(self, valid_record_data):
        record1 = KBRecord(**valid_record_data)
        record2 = KBRecord(**valid_record_data)
        with pytest.raises(KBRecordValidationError) as exc_info:
            validate_record_ids_unique([record1, record2])
        assert "test-001" in str(exc_info.value)

    def test_unique_ids_passes(self, valid_record_data):
        record1 = KBRecord(**valid_record_data)
        record2_data = dict(valid_record_data)
        record2_data["record_id"] = "test-002"
        record2 = KBRecord(**record2_data)
        # Should not raise
        validate_record_ids_unique([record1, record2])


# ── JSONL Ingestion Tests ──────────────────────────────────────────────────────

class TestJSONLIngestion:
    def test_load_valid_sample_kb(self):
        """Load the actual sample KB file — all records should be valid."""
        path = Path("data/raw/sample_kb.jsonl")
        if not path.exists():
            pytest.skip("Sample KB file not found; run from project root.")
        records = load_kb_from_jsonl(path)
        assert len(records) > 0
        assert all(isinstance(r, KBRecord) for r in records)

    def test_load_single_valid_record(self, valid_record_data, tmp_path):
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(json.dumps(valid_record_data) + "\n", encoding="utf-8")
        records = load_kb_from_jsonl(jsonl_file)
        assert len(records) == 1
        assert records[0].record_id == "test-001"

    def test_load_multiple_records(self, valid_record_data, tmp_path):
        data2 = dict(valid_record_data)
        data2["record_id"] = "test-002"
        lines = json.dumps(valid_record_data) + "\n" + json.dumps(data2) + "\n"
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(lines, encoding="utf-8")
        records = load_kb_from_jsonl(jsonl_file)
        assert len(records) == 2

    def test_empty_file_raises_empty_error(self, tmp_path):
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(EmptyKnowledgeBaseError):
            load_kb_from_jsonl(empty_file)

    def test_comment_lines_skipped(self, valid_record_data, tmp_path):
        jsonl_file = tmp_path / "test.jsonl"
        content = "# This is a comment\n" + json.dumps(valid_record_data) + "\n"
        jsonl_file.write_text(content, encoding="utf-8")
        records = load_kb_from_jsonl(jsonl_file)
        assert len(records) == 1

    def test_missing_file_raises_ingestion_error(self, tmp_path):
        with pytest.raises(KBIngestionError):
            load_kb_from_jsonl(tmp_path / "nonexistent.jsonl")

    def test_invalid_json_line_raises_ingestion_error(self, tmp_path):
        jsonl_file = tmp_path / "bad.jsonl"
        jsonl_file.write_text("not valid json\n", encoding="utf-8")
        with pytest.raises(KBIngestionError):
            load_kb_from_jsonl(jsonl_file)

    def test_invalid_json_skipped_with_skip_invalid(self, valid_record_data, tmp_path):
        jsonl_file = tmp_path / "mixed.jsonl"
        content = "not valid json\n" + json.dumps(valid_record_data) + "\n"
        jsonl_file.write_text(content, encoding="utf-8")
        records = load_kb_from_jsonl(jsonl_file, skip_invalid=True)
        assert len(records) == 1

    def test_duplicate_ids_raises_error(self, valid_record_data, tmp_path):
        jsonl_file = tmp_path / "dups.jsonl"
        content = json.dumps(valid_record_data) + "\n" + json.dumps(valid_record_data) + "\n"
        jsonl_file.write_text(content, encoding="utf-8")
        with pytest.raises(KBRecordValidationError):
            load_kb_from_jsonl(jsonl_file)

    def test_skip_synthetic_filters_out_synthetic_records(self, valid_record_data, synthetic_record_data, tmp_path):
        jsonl_file = tmp_path / "mixed.jsonl"
        content = json.dumps(valid_record_data) + "\n" + json.dumps(synthetic_record_data) + "\n"
        jsonl_file.write_text(content, encoding="utf-8")
        records = load_kb_from_jsonl(jsonl_file, skip_synthetic=True)
        assert len(records) == 1
        assert not records[0].is_synthetic()


# ── Dict List Ingestion ────────────────────────────────────────────────────────

class TestDictListIngestion:
    def test_valid_single_record(self, valid_record_data):
        records = load_kb_from_dict_list([valid_record_data])
        assert len(records) == 1

    def test_empty_list_raises_empty_error(self):
        with pytest.raises(EmptyKnowledgeBaseError):
            load_kb_from_dict_list([])

    def test_invalid_record_raises_by_default(self, valid_record_data):
        valid_record_data["confidence"] = "invalid"
        with pytest.raises(KBRecordValidationError):
            load_kb_from_dict_list([valid_record_data])

    def test_invalid_record_skipped_with_flag(self, valid_record_data, sample_kb_record_dict):
        bad_data = {"confidence": "invalid"}  # Invalid — missing required fields
        records = load_kb_from_dict_list([bad_data, sample_kb_record_dict], skip_invalid=True)
        assert len(records) == 1
