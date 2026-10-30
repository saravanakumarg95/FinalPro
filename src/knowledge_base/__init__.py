"""
src/knowledge_base/__init__.py — KB package exports.
"""
from src.knowledge_base.schema import Confidence, KBRecord, SourceType
from src.knowledge_base.ingestion import load_kb_from_jsonl, load_kb_from_dict_list
from src.knowledge_base.validator import validate_record, validate_record_ids_unique

__all__ = [
    "KBRecord",
    "Confidence",
    "SourceType",
    "load_kb_from_jsonl",
    "load_kb_from_dict_list",
    "validate_record",
    "validate_record_ids_unique",
]
