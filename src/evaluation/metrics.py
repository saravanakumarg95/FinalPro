"""
src/evaluation/metrics.py
────────────────────────────────────────────────────────────────
Evaluation metrics for the Cultural Story Generation research project.

Metrics implemented:
  1. Unsupported Claim Rate (UCR) — rule-based
  2. Evidence Usage Rate (EUR) — % retrieved records referenced
  3. Story Length Ratio — actual vs. target words
  4. Retrieval Precision@k (mock mode only; real mode uses MiniLM)
  5. Cultural Element Coverage — elements from outline present in story
  6. Consistency Check Score — from ConsistencyReport

These metrics are computed for all three methods (proposed, A, B)
and stored in ExperimentResult for comparison.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.generation.schemas import GeneratedStory
from src.reasoning.schemas import ReasoningTrace
from src.retrieval.schemas import RetrievalResult
from src.verification.schemas import ConsistencyReport


@dataclass
class MetricResult:
    """A single metric value with metadata."""
    name: str
    value: float
    unit: str = ""
    notes: str = ""
    lower_is_better: bool = False  # True for UCR, latency


@dataclass
class EvaluationResult:
    """Complete evaluation output for one generation run."""
    method: str  # proposed | baseline_a | baseline_b
    image_id: str = ""
    metrics: list[MetricResult] = field(default_factory=list)
    story_text: str = ""
    evidence_ids_used: list[str] = field(default_factory=list)
    consistency_status: str = "pass"
    latency_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "image_id": self.image_id,
            "metrics": {m.name: m.value for m in self.metrics},
            "consistency_status": self.consistency_status,
            "latency_seconds": self.latency_seconds,
            "evidence_count": len(self.evidence_ids_used),
            "story_word_count": len(self.story_text.split()),
            "notes": self.notes,
        }


def compute_unsupported_claim_rate(
    story: GeneratedStory,
    retrieval: RetrievalResult,
) -> MetricResult:
    """
    Compute Unsupported Claim Rate (UCR).

    UCR = (sentences with no supporting evidence term) / (total sentences)

    Simple approximation: a sentence is 'supported' if it shares at least
    one non-trivial content word with any retrieved evidence text.
    Real evaluation requires human annotation; this gives an automated
    lower-bound estimate.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', story.story_text) if len(s.strip()) > 15]
    if not sentences:
        return MetricResult("unsupported_claim_rate", 0.0, "%", "No sentences to evaluate.",
                            lower_is_better=True)

    # Build evidence word set
    evidence_words: set[str] = set()
    for e in retrieval.retrieved_records:
        evidence_words.update(
            w.lower() for w in e.evidence_text.split()
            if len(w) > 4  # skip short words
        )

    unsupported_count = 0
    for sentence in sentences:
        sentence_words = set(w.lower().strip(".,;:\"'") for w in sentence.split() if len(w) > 4)
        if not sentence_words & evidence_words:
            unsupported_count += 1

    ucr = round(100.0 * unsupported_count / len(sentences), 2)
    return MetricResult(
        name="unsupported_claim_rate",
        value=ucr,
        unit="%",
        notes=f"{unsupported_count}/{len(sentences)} sentences have no evidence overlap.",
        lower_is_better=True,
    )


def compute_evidence_usage_rate(
    story: GeneratedStory,
    retrieval: RetrievalResult,
) -> MetricResult:
    """EUR = evidence records referenced in story / total retrieved records."""
    if not retrieval.retrieved_records:
        return MetricResult("evidence_usage_rate", 0.0, "%", "No evidence retrieved.")
    used = len(story.evidence_ids_used)
    total = len(retrieval.retrieved_records)
    eur = round(100.0 * used / total, 2)
    return MetricResult(
        name="evidence_usage_rate",
        value=eur,
        unit="%",
        notes=f"{used}/{total} retrieved records used.",
    )


def compute_story_length_ratio(story: GeneratedStory, target_words: int) -> MetricResult:
    """How close the story is to the target length."""
    ratio = round(story.word_count / max(1, target_words), 3)
    return MetricResult(
        name="story_length_ratio",
        value=ratio,
        notes=f"{story.word_count} words vs target {target_words}.",
    )


def compute_cultural_element_coverage(
    story: GeneratedStory,
    trace: ReasoningTrace,
) -> MetricResult:
    """
    Fraction of planned cultural elements mentioned in the story.
    """
    if not trace.story_outline or not trace.story_outline.cultural_elements_to_use:
        return MetricResult("cultural_element_coverage", 0.0, "%", "No planned elements.")

    planned = trace.story_outline.cultural_elements_to_use
    story_lower = story.story_text.lower()
    covered = sum(
        1 for elem in planned
        if any(word.lower() in story_lower for word in elem.split()[:2])
    )
    coverage = round(100.0 * covered / len(planned), 2)
    return MetricResult(
        name="cultural_element_coverage",
        value=coverage,
        unit="%",
        notes=f"{covered}/{len(planned)} planned elements mentioned.",
    )


def compute_consistency_score(report: ConsistencyReport) -> MetricResult:
    """
    Convert ConsistencyReport to a numeric score.
    pass=1.0, caution=0.5, fail=0.0
    """
    scores = {"pass": 1.0, "caution": 0.5, "fail": 0.0}
    score = scores.get(report.overall_status, 0.5)
    warning_penalty = min(0.3, len(report.warnings) * 0.05)
    final = max(0.0, score - warning_penalty)
    return MetricResult(
        name="consistency_score",
        value=round(final, 3),
        notes=f"Status: {report.overall_status}. Warnings: {len(report.warnings)}.",
    )


def evaluate_story(
    story: GeneratedStory,
    trace: ReasoningTrace,
    retrieval: RetrievalResult,
    report: ConsistencyReport,
    target_words: int = 400,
    image_id: str = "",
    latency: float = 0.0,
) -> EvaluationResult:
    """
    Compute all metrics for one story generation run.

    Args:
        story: Generated story.
        trace: Reasoning trace.
        retrieval: Retrieved evidence result.
        report: Consistency check report.
        target_words: Target story length.
        image_id: Image identifier for experiment logging.
        latency: Wall-clock time in seconds.

    Returns:
        EvaluationResult with all metrics.
    """
    metrics = [
        compute_unsupported_claim_rate(story, retrieval),
        compute_evidence_usage_rate(story, retrieval),
        compute_story_length_ratio(story, target_words),
        compute_cultural_element_coverage(story, trace),
        compute_consistency_score(report),
    ]

    return EvaluationResult(
        method=story.generation_method,
        image_id=image_id,
        metrics=metrics,
        story_text=story.story_text,
        evidence_ids_used=story.evidence_ids_used,
        consistency_status=report.overall_status,
        latency_seconds=latency,
    )
