"""
src/reasoning/pipeline.py
────────────────────────────────────────────────────────────────
The 7-step Evidence-Grounded Cultural Reasoning Pipeline.

This is the core research contribution — an inspectable, structured
reasoning process that:
  1. Summarises visual observations
  2. Records retrieval results
  3. Aligns evidence to observations
  4. Identifies uncertainty and contradictions
  5. Plans the story
  6. Performs pre-generation cultural consistency check
  7. Hands off to story generation
"""

from src.reasoning.schemas import (
    EvidenceAlignment, ReasoningTrace, StoryOutline
)
from src.retrieval.schemas import RetrievalResult
from src.vision.schemas import VisualObservation
from src.utils.logger import get_logger

log = get_logger(__name__)

# Minimum relevance score for evidence to be used in story generation
MIN_EVIDENCE_SCORE_FOR_USE = 0.25


class ReasoningPipeline:
    """
    Executes the 7-step evidence-grounded reasoning pipeline.

    All steps are logged and stored in a ReasoningTrace for inspection.
    """

    def run(
        self,
        observation: VisualObservation,
        retrieval_result: RetrievalResult,
        target_words: int = 400,
    ) -> ReasoningTrace:
        """
        Run the full reasoning pipeline.

        Args:
            observation: VisualObservation from the VLM stage.
            retrieval_result: RetrievalResult from the retrieval stage.
            target_words: Target word count for story outline.

        Returns:
            ReasoningTrace with all intermediate outputs.
        """
        trace = ReasoningTrace()

        # ── Step 1: Observation Summary ────────────────────────────────────
        log.info("Reasoning Step 1: Observation summary")
        trace.observation_summary = self._summarise_observations(observation)
        trace.key_cultural_clues = observation.cultural_clues[:8]

        # ── Step 2: Record retrieval ───────────────────────────────────────
        log.info("Reasoning Step 2: Recording retrieval results")
        trace.retrieved_evidence_ids = [e.record_id for e in retrieval_result.retrieved_records]
        trace.retrieval_query_used = retrieval_result.query_used
        trace.total_evidence_retrieved = len(retrieval_result.retrieved_records)

        # ── Step 3: Evidence-observation alignment ─────────────────────────
        log.info("Reasoning Step 3: Evidence-observation alignment")
        trace.evidence_alignments = self._align_evidence(
            observation, retrieval_result
        )

        # Filter to evidence that should be used
        trace.evidence_used_ids = [
            a.evidence_id for a in trace.evidence_alignments if a.use_in_story
        ]
        trace.total_evidence_used = len(trace.evidence_used_ids)

        # ── Step 4: Uncertainty and contradiction analysis ─────────────────
        log.info("Reasoning Step 4: Uncertainty analysis")
        trace.uncertainties_identified = list(observation.uncertainty_notes)
        trace.uncertainties_identified += list(observation.claims_requiring_evidence)
        trace.contradictions_found = self._find_contradictions(
            observation, retrieval_result
        )

        if not retrieval_result.evidence_found or trace.total_evidence_used == 0:
            trace.evidence_coverage = "insufficient"
        elif trace.total_evidence_used < 2:
            trace.evidence_coverage = "partial"
        else:
            trace.evidence_coverage = "sufficient"

        # ── Step 5: Story outline ──────────────────────────────────────────
        log.info("Reasoning Step 5: Story outline")
        trace.story_outline = self._create_story_outline(
            observation, retrieval_result, trace, target_words
        )

        # ── Step 6: Pre-generation consistency check ───────────────────────
        log.info("Reasoning Step 6: Pre-generation consistency check")
        trace.pre_generation_warnings = self._pre_generation_check(
            trace, retrieval_result
        )

        log.info(
            f"Reasoning complete. Evidence used: {trace.total_evidence_used}/"
            f"{trace.total_evidence_retrieved}. "
            f"Warnings: {len(trace.pre_generation_warnings)}."
        )
        return trace

    # ── Private helpers ────────────────────────────────────────────────────

    def _summarise_observations(self, obs: VisualObservation) -> str:
        parts = []
        if obs.setting:
            parts.append(f"Setting: {obs.setting}")
        if obs.activities:
            parts.append(f"Activities: {'; '.join(obs.activities)}")
        if obs.clothing_and_objects:
            parts.append(f"Objects and clothing: {'; '.join(obs.clothing_and_objects)}")
        if obs.cultural_clues:
            parts.append(f"Cultural clues (not confirmed): {'; '.join(obs.cultural_clues)}")
        if obs.potential_cultural_context:
            parts.append(f"Possible cultural context: {obs.potential_cultural_context}")
        if obs.uncertainty_notes:
            parts.append(f"Uncertainties: {'; '.join(obs.uncertainty_notes)}")
        return "\n".join(parts)

    def _align_evidence(
        self,
        obs: VisualObservation,
        retrieval: RetrievalResult,
    ) -> list[EvidenceAlignment]:
        """Score how well each retrieved record aligns with what was seen."""
        alignments = []
        visible_terms = set(
            " ".join(
                obs.cultural_clues + obs.clothing_and_objects
                + obs.activities + obs.objects_present
            ).lower().split()
        )

        for evidence in retrieval.retrieved_records:
            evidence_terms = set(evidence.evidence_text.lower().split())
            overlap = visible_terms & evidence_terms
            score = min(1.0, len(overlap) / max(1, len(evidence_terms) * 0.1))
            score = round(max(score, evidence.relevance_score * 0.7), 4)

            aligned = [
                term for term in overlap
                if len(term) > 3 and term not in {"with", "from", "that", "this", "have", "they"}
            ][:5]

            use = (
                evidence.relevance_score >= MIN_EVIDENCE_SCORE_FOR_USE
                and not evidence.is_synthetic
            )

            alignments.append(EvidenceAlignment(
                evidence_id=evidence.record_id,
                evidence_topic=evidence.topic,
                alignment_score=score,
                aligned_observations=aligned,
                misaligned_aspects=[],
                use_in_story=use,
            ))

        return alignments

    def _find_contradictions(
        self,
        obs: VisualObservation,
        retrieval: RetrievalResult,
    ) -> list[str]:
        """Identify potential contradictions between observations and evidence."""
        contradictions = []
        if retrieval.has_synthetic:
            contradictions.append(
                "One or more retrieved records are synthetic. "
                "Do not present synthetic content as verified cultural fact."
            )
        if not retrieval.evidence_found:
            contradictions.append(
                "No cultural evidence was retrieved. "
                "Story must not assert unverified cultural claims."
            )
        return contradictions

    def _create_story_outline(
        self,
        obs: VisualObservation,
        retrieval: RetrievalResult,
        trace: ReasoningTrace,
        target_words: int,
    ) -> StoryOutline:
        """Generate a story outline from available evidence."""
        # Gather cultural elements from evidence
        cultural_elements = []
        for e in retrieval.retrieved_records:
            if e.record_id in trace.evidence_used_ids:
                cultural_elements.append(e.topic)

        # Claims to avoid (from uncertainty)
        avoid = list(obs.claims_requiring_evidence[:3])

        return StoryOutline(
            setting=obs.setting or "A Tamil cultural setting",
            main_character="A Tamil family member or community member",
            cultural_elements_to_use=cultural_elements[:5],
            evidence_ids_supporting=trace.evidence_used_ids,
            narrative_arc=(
                "A personal moment during a Tamil festival or cultural event, "
                "showing lived experience rather than encyclopedic description."
            ),
            tone="warm, respectful, and culturally grounded",
            cultural_claims_to_avoid=avoid,
            approximate_word_count=target_words,
        )

    def _pre_generation_check(
        self,
        trace: ReasoningTrace,
        retrieval: RetrievalResult,
    ) -> list[str]:
        """Generate warnings that the story generator should respect."""
        warnings = []
        if trace.evidence_coverage == "insufficient":
            warnings.append(
                "INSUFFICIENT EVIDENCE: No reliable cultural evidence was retrieved. "
                "The story must not assert specific cultural facts."
            )
        if trace.evidence_coverage == "partial":
            warnings.append(
                "PARTIAL EVIDENCE: Limited evidence retrieved. "
                "Limit cultural claims to what is directly supported."
            )
        if retrieval.has_synthetic:
            warnings.append(
                "SYNTHETIC EVIDENCE: Some retrieved records are synthetic demos. "
                "Do not present these as verified cultural knowledge."
            )
        if trace.contradictions_found:
            warnings.extend(trace.contradictions_found)
        return warnings
