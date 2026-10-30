"""
app/components/pipeline_runner.py
────────────────────────────────────────────────────────────────
Orchestrates the complete evidence-grounded pipeline for the UI.
Handles both mock mode and live mode seamlessly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL.Image import Image

from src.config.settings import AppConfig
from src.knowledge_base.ingestion import load_kb_from_jsonl
from src.retrieval.engine import ChromaRetrievalEngine
from src.reasoning.pipeline import ReasoningPipeline
from src.verification.checker import ConsistencyChecker
from src.utils.logger import get_logger

log = get_logger(__name__)

# Cached index engine (rebuilt only once per session)
_engine_cache: dict = {}


def _get_engine(config: AppConfig, top_k: int):
    """Get or create the retrieval engine (cached)."""
    cache_key = str(config.chroma_db_path) + str(config.is_mock_mode)
    if cache_key not in _engine_cache:
        # Choose embedder
        if config.is_mock_mode:
            from src.embeddings.mock_embedder import MockEmbedder
            embedder = MockEmbedder()
            persist_path = None  # In-memory for mock
        else:
            from src.embeddings.minilm_embedder import MiniLMEmbedder
            embedder = MiniLMEmbedder()
            persist_path = config.chroma_db_path

        engine = ChromaRetrievalEngine(embedder, persist_path=persist_path)

        # Load and index KB
        records = load_kb_from_jsonl(config.kb_raw_path, skip_invalid=True)
        engine.build_index(records)
        _engine_cache[cache_key] = engine
        log.info(f"Engine cached with {engine.get_index_size()} records.")
    return _engine_cache[cache_key]


def run_pipeline(
    pil_image: Image,
    config: AppConfig,
    top_k: int = 5,
    target_words: int = 400,
    reading_level: str = "general",
    run_baselines: bool = True,
) -> dict:
    """
    Run the complete Tamil cultural story generation pipeline.

    Returns a dict with keys:
      observation, retrieval, trace, story, consistency, baselines
    """
    # ── Step 1: Visual understanding ──────────────────────────────────────
    if config.is_mock_mode:
        from src.vision.mock_vlm import MockVLM
        vlm = MockVLM()
    else:
        from src.vision.gemini_vlm import GeminiVLM
        vlm = GeminiVLM(api_key=config.require_gemini_key(), model_name=config.vlm_model)

    observation = vlm.analyze_image(pil_image)
    log.info(f"VLM observation complete. Cultural clues: {observation.cultural_clues[:3]}")

    # ── Step 2: Retrieval ─────────────────────────────────────────────────
    engine = _get_engine(config, top_k)
    query = observation.build_retrieval_query()
    retrieval_result = engine.retrieve(query, top_k=top_k)
    log.info(f"Retrieval complete. Found: {retrieval_result.evidence_found}, Records: {len(retrieval_result.retrieved_records)}")

    # ── Step 3: Reasoning ─────────────────────────────────────────────────
    reasoning = ReasoningPipeline()
    trace = reasoning.run(observation, retrieval_result, target_words=target_words)
    log.info(f"Reasoning complete. Evidence used: {trace.total_evidence_used}")

    # ── Step 4: Story Generation ──────────────────────────────────────────
    if config.is_mock_mode:
        from src.generation.mock_llm import MockLLM
        llm = MockLLM()
        story = llm.generate_story(trace, target_words=target_words, reading_level=reading_level)
        baselines = {
            "baseline_a": llm.generate_baseline_a(trace.observation_summary, target_words),
            "baseline_b": llm.generate_baseline_b(trace.observation_summary, "", target_words),
        } if run_baselines else {}
    else:
        from src.generation.gemini_llm import GeminiLLM
        llm = GeminiLLM(api_key=config.require_gemini_key(), model_name=config.llm_model)
        story = llm.generate_story(trace, retrieval_result, target_words=target_words, reading_level=reading_level)
        baselines = {
            "baseline_a": llm.generate_baseline_a(trace.observation_summary, target_words),
            "baseline_b": llm.generate_baseline_b(trace.observation_summary, target_words=target_words),
        } if run_baselines else {}

    # ── Step 5: Consistency check ─────────────────────────────────────────
    checker = ConsistencyChecker()
    consistency = checker.check(story, trace, retrieval_result)

    return {
        "observation": observation,
        "retrieval": retrieval_result,
        "trace": trace,
        "story": story,
        "consistency": consistency,
        "baselines": baselines,
    }
