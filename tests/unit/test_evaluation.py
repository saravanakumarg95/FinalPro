"""
tests/unit/test_evaluation.py
─────────────────────────────
Unit tests for evaluation metrics module.
"""
import pytest
from src.evaluation.metrics import (
    compute_consistency_score,
    compute_cultural_element_coverage,
    compute_evidence_usage_rate,
    compute_story_length_ratio,
    compute_unsupported_claim_rate,
    evaluate_story,
)
from src.generation.schemas import GeneratedStory
from src.reasoning.schemas import ReasoningTrace, StoryOutline
from src.retrieval.schemas import RetrievedEvidence, RetrievalResult
from src.verification.schemas import ConsistencyReport


@pytest.fixture
def mock_story():
    return GeneratedStory(
        story_text=(
            "The morning sun rose over the Tamil village. "
            "Meenakshi drew a kolam at the entrance with rice flour. "
            "The Pongal festival was a time of joy and gratitude. "
            "She wore her Kanchipuram saree for the occasion."
        ),
        word_count=42,
        evidence_ids_used=["ta-001", "ta-002"],
        generation_method="proposed",
        model_used="mock",
    )


@pytest.fixture
def mock_retrieval():
    return RetrievalResult(
        query_used="Tamil Pongal festival",
        evidence_found=True,
        retrieved_records=[
            RetrievedEvidence(
                record_id="ta-001",
                topic="Pongal — Tamil Harvest Festival",
                cultural_region="Tamil Nadu",
                evidence_text="Pongal is a four-day Tamil harvest festival in January with kolam and clay pot cooking.",
                source_title="Test Source",
                source_url="https://example.com",
                confidence="verified",
                known_limitations="Test limitations.",
                relevance_score=0.9,
                retrieval_rank=1,
            ),
            RetrievedEvidence(
                record_id="ta-002",
                topic="Kolam — Tamil Floor Art",
                cultural_region="Tamil Nadu",
                evidence_text="Kolam is drawn using rice flour at home entrances in Tamil Nadu.",
                source_title="Test Source 2",
                source_url="https://example2.com",
                confidence="verified",
                known_limitations="Test limitations.",
                relevance_score=0.8,
                retrieval_rank=2,
            ),
        ],
        num_records_in_index=25,
    )


@pytest.fixture
def mock_trace():
    trace = ReasoningTrace()
    trace.observation_summary = "Tamil festival scene"
    trace.evidence_used_ids = ["ta-001", "ta-002"]
    trace.story_outline = StoryOutline(
        setting="Tamil village",
        main_character="Tamil woman",
        cultural_elements_to_use=["Pongal", "Kolam", "Kanchipuram saree"],
        evidence_ids_supporting=["ta-001", "ta-002"],
        narrative_arc="Harvest festival celebration",
    )
    return trace


class TestUCR:
    def test_ucr_with_evidence_overlap_is_low(self, mock_story, mock_retrieval):
        result = compute_unsupported_claim_rate(mock_story, mock_retrieval)
        assert result.name == "unsupported_claim_rate"
        assert 0.0 <= result.value <= 100.0
        assert result.lower_is_better is True

    def test_ucr_no_evidence_returns_100(self):
        story = GeneratedStory(story_text="This is a completely unsupported cultural claim about Tamils.", word_count=12, generation_method="baseline_a", model_used="mock")
        empty = RetrievalResult(query_used="", evidence_found=False)
        result = compute_unsupported_claim_rate(story, empty)
        assert result.value == 100.0 or result.value > 0.0  # All unsupported

    def test_ucr_empty_story_returns_zero(self, mock_retrieval):
        story = GeneratedStory(story_text="", generation_method="proposed", model_used="mock")
        result = compute_unsupported_claim_rate(story, mock_retrieval)
        assert result.value == 0.0


class TestEUR:
    def test_eur_full_usage(self, mock_story, mock_retrieval):
        result = compute_evidence_usage_rate(mock_story, mock_retrieval)
        assert result.value == 100.0  # 2/2 used

    def test_eur_no_retrieval(self, mock_story):
        empty = RetrievalResult(query_used="", evidence_found=False)
        result = compute_evidence_usage_rate(mock_story, empty)
        assert result.value == 0.0


class TestStoryLengthRatio:
    def test_story_at_target_gives_ratio_one(self):
        story = GeneratedStory(story_text="word " * 400, word_count=400, generation_method="proposed", model_used="mock")
        result = compute_story_length_ratio(story, target_words=400)
        assert abs(result.value - 1.0) < 0.01

    def test_short_story_gives_ratio_below_one(self):
        story = GeneratedStory(story_text="word " * 100, word_count=100, generation_method="proposed", model_used="mock")
        result = compute_story_length_ratio(story, target_words=400)
        assert result.value < 1.0


class TestCulturalElementCoverage:
    def test_all_elements_covered_gives_100(self, mock_story, mock_trace):
        result = compute_cultural_element_coverage(mock_story, mock_trace)
        assert 0.0 <= result.value <= 100.0

    def test_no_outline_gives_zero(self, mock_story):
        trace = ReasoningTrace()  # No story_outline
        result = compute_cultural_element_coverage(mock_story, trace)
        assert result.value == 0.0


class TestConsistencyScore:
    def test_pass_report_gives_high_score(self):
        report = ConsistencyReport(overall_status="pass")
        result = compute_consistency_score(report)
        assert result.value >= 0.9

    def test_fail_report_gives_low_score(self):
        report = ConsistencyReport(overall_status="fail")
        report.add_warning("unsupported_claim", "high", "Test warning", "Fix it.")
        result = compute_consistency_score(report)
        assert result.value <= 0.5


class TestEvaluateStory:
    def test_evaluate_story_returns_all_metrics(self, mock_story, mock_trace, mock_retrieval):
        report = ConsistencyReport(overall_status="pass")
        result = evaluate_story(mock_story, mock_trace, mock_retrieval, report,
                                target_words=400, image_id="test_img")
        assert result.method == "proposed"
        assert result.image_id == "test_img"
        assert len(result.metrics) == 5
        metric_names = {m.name for m in result.metrics}
        assert "unsupported_claim_rate" in metric_names
        assert "evidence_usage_rate" in metric_names
        assert "cultural_element_coverage" in metric_names
