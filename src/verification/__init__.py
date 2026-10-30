"""src/verification/__init__.py — Verification package exports."""
from src.verification.schemas import ConsistencyReport, ConsistencyWarning, WarningSeverity
from src.verification.checker import ConsistencyChecker

__all__ = ["ConsistencyReport", "ConsistencyWarning", "WarningSeverity", "ConsistencyChecker"]
