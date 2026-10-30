"""
tests/integration/test_pipeline_integration.py
────────────────────────────────────────────────────────────────
Integration test for the complete Tamil cultural story pipeline
in mock mode — no API calls, no model downloads required.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import AppConfig
from src.embeddings.mock_embedder import MockEmbedder
from src.generation.mock_llm import MockLLM
from src.knowledge_base.ingestion import load_kb_from_jsonl
from src.reasoning.pipeline import ReasoningPipeline
from src.retrieval.engine import ChromaRetrievalEngine
from src.verification.checker import ConsistencyChecker
from src.vision.mock_vlm import MockVLM
from src.vision.schemas import VisualObservation


class TestFullPipelineIntegration:
    """Integration test: mock image → observation → retrieval → reasoning → story → consistency."""

    @pytest.fixture(scope="class")
    def engine(self):
        """Build a retrieval engine with real KB records and mock embedder."""
        config = AppConfig()
        records = load_kb_from_jsonl(config.kb_raw_path, skip_invalid=True)
        embedder = MockEmbedder()
        eng = ChromaRetrievalEngine(embedder, persist_path=None)  # in-memory
        eng.build_index(records)
        return eng

    @pytest.fixture
    def mock_pil_image(self, valid_pil_image):
        return valid_pil_image

    def test_vlm_returns_visual_observation(self, valid_pil_image):
        vlm = MockVLM()
        obs = vlm.analyze_image(valid_pil_image)
        assert isinstance(obs, VisualObservation)
        assert obs.setting != ""
        assert len(obs.cultural_clues) > 0

    def test_retrieval_returns_results(self, engine):
        query = "Tamil Pongal festival clay pot kolam"
        result = engine.retrieve(query, top_k=5)
        assert result.evidence_found
        assert len(result.retrieved_records) > 0
        assert result.retrieved_records[0].relevance_score >= 0.0

    def test_reasoning_pipeline_produces_trace(self, valid_pil_image, engine):
        vlm = MockVLM()
        obs = vlm.analyze_image(valid_pil_image)
        query = obs.build_retrieval_query()
        retrieval = engine.retrieve(query, top_k=5)
        pipeline = ReasoningPipeline()
        trace = pipeline.run(obs, retrieval, target_words=400)

        assert trace.observation_summary != ""
        assert trace.total_evidence_retrieved >= 0
        assert trace.story_outline is not None
        assert trace.pipeline_version == "1.0"

    def test_story_generation_produces_story(self, valid_pil_image, engine):
        vlm = MockVLM()
        obs = vlm.analyze_image(valid_pil_image)
        retrieval = engine.retrieve(obs.build_retrieval_query(), top_k=5)
        trace = ReasoningPipeline().run(obs, retrieval)
        llm = MockLLM()
        story = llm.generate_story(trace, target_words=400)

        assert len(story.story_text) > 100
        assert story.word_count > 0
        assert story.generation_method == "proposed"

    def test_consistency_check_runs_successfully(self, valid_pil_image, engine):
        vlm = MockVLM()
        obs = vlm.analyze_image(valid_pil_image)
        retrieval = engine.retrieve(obs.build_retrieval_query(), top_k=5)
        trace = ReasoningPipeline().run(obs, retrieval)
        story = MockLLM().generate_story(trace)
        checker = ConsistencyChecker()
        report = checker.check(story, trace, retrieval)

        assert report.overall_status in {"pass", "caution", "fail"}
        assert report.summary != ""

    def test_full_pipeline_end_to_end(self, valid_pil_image, engine):
        """Single end-to-end run through all stages."""
        # VLM
        vlm = MockVLM()
        obs = vlm.analyze_image(valid_pil_image)
        # Retrieval
        retrieval = engine.retrieve(obs.build_retrieval_query(), top_k=5)
        # Reasoning
        trace = ReasoningPipeline().run(obs, retrieval, target_words=400)
        # Generation
        story = MockLLM().generate_story(trace, target_words=400)
        # Baselines
        llm = MockLLM()
        baseline_a = llm.generate_baseline_a(trace.observation_summary)
        baseline_b = llm.generate_baseline_b(trace.observation_summary, "")
        # Consistency
        report = ConsistencyChecker().check(story, trace, retrieval)

        # Verify outputs
        assert obs.potential_cultural_context is not None
        assert retrieval.evidence_found
        assert trace.total_evidence_retrieved > 0
        assert "Tamil" in story.story_text or "Pongal" in story.story_text
        assert baseline_a.generation_method == "baseline_a"
        assert baseline_b.generation_method == "baseline_b"
        assert report.overall_status in {"pass", "caution", "fail"}

    def test_retrieval_precision_at_k(self, engine):
        """
        Retrieval test: verify that top-k results contain Tamil records.

        Note: MockEmbedder uses SHA-256 hashes (not semantic vectors), so
        semantic ordering is not guaranteed. This test verifies retrieval
        returns results from the Tamil KB index — semantic Precision@k is
        evaluated in Phase 6 with the real MiniLM embedder.
        """
        result = engine.retrieve("Pongal Tamil harvest festival clay pot overflow", top_k=10)
        assert result.evidence_found, "Expected retrieval to find records"
        assert len(result.retrieved_records) > 0, "Expected at least one result"
        # All results must come from the Tamil KB
        for e in result.retrieved_records:
            assert e.record_id.startswith("ta-") or e.record_id.startswith("syn-"), (
                f"Unexpected record_id: {e.record_id}"
            )

