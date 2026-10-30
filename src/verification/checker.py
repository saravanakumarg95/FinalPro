"""
src/verification/checker.py
────────────────────────────────────────────────────────────────
Cultural Consistency and Inclusivity Checker.

Performs deterministic rule-based checks on the generated story
against the retrieved evidence and reasoning trace.

Checks implemented:
  1. Unsupported cultural assertions (rule-based keyword match)
  2. Synthetic evidence in story
  3. Insufficient evidence warning propagation
  4. Common Tamil cultural stereotype detection
  5. Overgeneralisation detection (e.g. "all Tamils...")
  6. Contradiction between story and evidence limitations
"""

import re
from src.generation.schemas import GeneratedStory
from src.reasoning.schemas import ReasoningTrace
from src.retrieval.schemas import RetrievalResult
from src.verification.schemas import ConsistencyReport
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Stereotype patterns ────────────────────────────────────────────────────────
STEREOTYPE_PATTERNS = [
    (r"\ball (tamil|south indian)\b.*?\balways\b", "Overgeneralisation: 'all Tamils always...'"),
    (r"\btamils? (never|always|only)\b", "Potential overgeneralisation about Tamil people"),
    (r"\bprimitive\b|\bbackward\b|\buncivilised\b", "Potentially offensive term used"),
    (r"\bexotic\b.*?\btamil\b|\btamil\b.*?\bexotic\b", "Exoticising Tamil culture"),
    (r"\bthey (all|always|never)\b", "Potential stereotyping with 'they all/always/never'"),
]

# ── Unsupported claim markers ─────────────────────────────────────────────────
# Cultural claims in Tamil context that require specific evidence
CULTURAL_CLAIM_TERMS = [
    "always celebrated", "every Tamil", "all Tamils", "in their culture they",
    "it is their tradition to", "Tamils believe that", "Tamil people never",
    "ancient tradition of", "sacred ritual of", "forbidden by"
]


class ConsistencyChecker:
    """
    Checks generated stories for cultural consistency and inclusivity.
    Uses deterministic rules against retrieved evidence.
    """

    def check(
        self,
        story: GeneratedStory,
        trace: ReasoningTrace,
        retrieval_result: RetrievalResult,
    ) -> ConsistencyReport:
        """
        Run all consistency checks on a generated story.

        Args:
            story: The generated story to check.
            trace: ReasoningTrace from the pipeline.
            retrieval_result: Retrieved evidence used.

        Returns:
            ConsistencyReport with all warnings and overall status.
        """
        report = ConsistencyReport()
        text_lower = story.story_text.lower()

        # Check 1: Pre-generation warnings propagated
        self._check_pregeneration_warnings(report, trace)

        # Check 2: Synthetic evidence used
        self._check_synthetic_evidence(report, retrieval_result)

        # Check 3: Stereotype patterns
        self._check_stereotypes(report, story.story_text)

        # Check 4: Overgeneralisation detection
        self._check_overgeneralisations(report, text_lower)

        # Check 5: Insufficient evidence with confident claims
        self._check_evidence_confidence(report, story, trace)

        # Set summary
        report.unsupported_claim_count = sum(
            1 for w in report.warnings if w.category == "unsupported_claim"
        )
        if not report.warnings:
            report.summary = "✅ No significant cultural consistency issues detected."
        elif report.overall_status == "fail":
            report.summary = f"❌ {len(report.warnings)} issue(s) found. Review before use."
        else:
            report.summary = f"⚠️ {len(report.warnings)} caution(s). Review recommended."

        log.info(
            f"Consistency check complete. Status: {report.overall_status}. "
            f"Warnings: {len(report.warnings)}."
        )
        return report

    def _check_pregeneration_warnings(self, report: ConsistencyReport, trace: ReasoningTrace):
        for warning in trace.pre_generation_warnings:
            if "INSUFFICIENT" in warning:
                report.add_warning(
                    "unsupported_claim", "high",
                    "Story generated with insufficient cultural evidence. "
                    "Cultural claims in this story cannot be verified.",
                    suggested_revision="Qualify all cultural claims with uncertainty markers.",
                )
            elif "SYNTHETIC" in warning:
                report.add_warning(
                    "unsupported_claim", "high",
                    "Story may be based on synthetic (non-verified) cultural records.",
                    suggested_revision="Remove or clearly label culturally specific claims.",
                )

    def _check_synthetic_evidence(self, report: ConsistencyReport, retrieval: RetrievalResult):
        if retrieval.has_synthetic:
            for e in retrieval.retrieved_records:
                if e.is_synthetic:
                    report.add_warning(
                        "unsupported_claim", "high",
                        f"Retrieved evidence '{e.topic}' is a synthetic demonstration record "
                        "and should not be treated as verified cultural knowledge.",
                        suggested_revision="Replace synthetic records with verified sources before use.",
                        evidence_id=e.record_id,
                    )

    def _check_stereotypes(self, report: ConsistencyReport, text: str):
        for pattern, message in STEREOTYPE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                report.add_warning(
                    "stereotype", "medium",
                    f"Possible stereotype detected: {message}",
                    suggested_revision="Rephrase to describe specific, documented practices rather than universal claims.",
                )

    def _check_overgeneralisations(self, report: ConsistencyReport, text_lower: str):
        for phrase in CULTURAL_CLAIM_TERMS:
            if phrase.lower() in text_lower:
                report.add_warning(
                    "overgeneralisation", "medium",
                    f"Potential overgeneralisation: phrase '{phrase}' found. "
                    "Cultural practices vary; avoid universal claims.",
                    suggested_revision=f"Replace '{phrase}' with specific, evidence-supported language.",
                )

    def _check_evidence_confidence(
        self,
        report: ConsistencyReport,
        story: GeneratedStory,
        trace: ReasoningTrace,
    ):
        if trace.evidence_coverage == "insufficient" and len(story.story_text) > 200:
            report.add_warning(
                "unsupported_claim", "medium",
                "Story contains substantial cultural content despite insufficient evidence retrieval.",
                suggested_revision="Reduce specific cultural claims or add uncertainty qualifiers.",
            )
