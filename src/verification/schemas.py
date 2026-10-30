"""
src/verification/schemas.py — Pydantic schema for ConsistencyReport.
"""
from enum import Enum
from pydantic import BaseModel, Field


class WarningSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConsistencyWarning(BaseModel):
    """A single cultural consistency warning."""
    category: str  # unsupported_claim | stereotype | overgeneralisation | contradiction | uncertainty
    severity: WarningSeverity
    message: str
    suggested_revision: str = ""
    evidence_id: str = ""  # Which evidence record (if relevant)


class ConsistencyReport(BaseModel):
    """Complete cultural consistency and inclusivity check report."""
    warnings: list[ConsistencyWarning] = Field(default_factory=list)
    unsupported_claim_count: int = 0
    stereotype_flags: list[str] = Field(default_factory=list)
    overall_status: str = "pass"  # pass | caution | fail
    summary: str = ""

    def has_high_severity(self) -> bool:
        return any(w.severity == WarningSeverity.HIGH for w in self.warnings)

    def add_warning(self, category: str, severity: str, message: str,
                    suggested_revision: str = "", evidence_id: str = "") -> None:
        self.warnings.append(ConsistencyWarning(
            category=category, severity=WarningSeverity(severity),
            message=message, suggested_revision=suggested_revision,
            evidence_id=evidence_id,
        ))
        if severity == "high":
            self.overall_status = "fail"
        elif severity == "medium" and self.overall_status == "pass":
            self.overall_status = "caution"
