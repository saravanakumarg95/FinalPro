"""
scripts/run_evaluation.py
────────────────────────────────────────────────────────────────
Automated evaluation script — runs all three methods (proposed,
baseline_a, baseline_b) on a set of test images and logs results.

Usage (from project root):
    python scripts/run_evaluation.py --images data/sample/
    python scripts/run_evaluation.py --images data/sample/ --live

This script:
  1. Loads test images from a directory
  2. Runs all three methods for each image
  3. Computes evaluation metrics for each
  4. Logs results to experiments/results/
  5. Prints a summary comparison table
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import AppConfig, AppMode
from src.embeddings.minilm_embedder import MiniLMEmbedder
from src.embeddings.mock_embedder import MockEmbedder
from src.evaluation.logger import ExperimentLogger
from src.evaluation.metrics import evaluate_story
from src.generation.mock_llm import MockLLM
from src.knowledge_base.ingestion import load_kb_from_jsonl
from src.reasoning.pipeline import ReasoningPipeline
from src.retrieval.engine import ChromaRetrievalEngine
from src.utils.errors import ImageValidationError
from src.utils.image_utils import validate_image_path
from src.utils.logger import get_logger, setup_logger
from src.verification.checker import ConsistencyChecker
from src.vision.mock_vlm import MockVLM

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_test_images(images_path: Path) -> list[Path]:
    """Get all image files from a directory or single file."""
    images_path = Path(images_path)
    if images_path.is_file():
        return [images_path]
    return sorted([
        p for p in images_path.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])


def run_evaluation(
    image_paths: list[Path],
    config: AppConfig,
    exp_logger: ExperimentLogger,
    top_k: int = 5,
    target_words: int = 400,
) -> None:
    """Run all three methods on all images and log results."""
    # Setup shared components
    embedder = MockEmbedder() if config.is_mock_mode else MiniLMEmbedder()
    engine = ChromaRetrievalEngine(
        embedder,
        persist_path=None if config.is_mock_mode else config.chroma_db_path
    )
    records = load_kb_from_jsonl(config.kb_raw_path, skip_invalid=True)
    engine.build_index(records)

    vlm = MockVLM()
    llm = MockLLM()
    reasoning = ReasoningPipeline()
    checker = ConsistencyChecker()

    for i, img_path in enumerate(image_paths):
        log.info(f"\n{'='*60}\nProcessing image {i+1}/{len(image_paths)}: {img_path.name}")
        image_id = img_path.stem

        try:
            pil_image = validate_image_path(img_path)
        except ImageValidationError as e:
            log.warning(f"Skipping {img_path.name}: {e}")
            continue

        # --- VLM (shared across all methods) ---
        t0 = time.time()
        observation = vlm.analyze_image(pil_image)
        query = observation.build_retrieval_query()
        retrieval = engine.retrieve(query, top_k=top_k)
        trace = reasoning.run(observation, retrieval, target_words=target_words)
        vlm_retrieval_time = time.time() - t0

        results_this_image = []

        # --- Proposed Method ---
        t_start = time.time()
        story_proposed = llm.generate_story(trace, target_words=target_words)
        report_proposed = checker.check(story_proposed, trace, retrieval)
        latency_proposed = vlm_retrieval_time + (time.time() - t_start)
        eval_proposed = evaluate_story(
            story_proposed, trace, retrieval, report_proposed,
            target_words=target_words, image_id=image_id, latency=latency_proposed
        )
        results_this_image.append(eval_proposed)

        # --- Baseline A ---
        t_start = time.time()
        story_a = llm.generate_baseline_a(trace.observation_summary, target_words)
        # For baseline A, retrieval result is empty
        from src.retrieval.schemas import RetrievalResult
        empty_retrieval = RetrievalResult(query_used="", evidence_found=False)
        report_a = checker.check(story_a, trace, empty_retrieval)
        latency_a = time.time() - t_start + (time.time() - t0 - vlm_retrieval_time)
        eval_a = evaluate_story(
            story_a, trace, empty_retrieval, report_a,
            target_words=target_words, image_id=image_id, latency=latency_a
        )
        results_this_image.append(eval_a)

        # --- Baseline B ---
        t_start = time.time()
        story_b = llm.generate_baseline_b(trace.observation_summary, "", target_words)
        report_b = checker.check(story_b, trace, empty_retrieval)
        latency_b = time.time() - t_start
        eval_b = evaluate_story(
            story_b, trace, empty_retrieval, report_b,
            target_words=target_words, image_id=image_id, latency=latency_b
        )
        results_this_image.append(eval_b)

        exp_logger.log_comparison(results_this_image, image_id=image_id)

        # Per-image summary
        for r in results_this_image:
            metrics_str = " | ".join(
                f"{m.name}={m.value}{m.unit}"
                for m in r.metrics
            )
            log.info(f"  [{r.method:20s}] {metrics_str}")

    # Final summary
    summary = exp_logger.summarise()
    log.info(f"\n{'='*60}\nEXPERIMENT SUMMARY\n{'='*60}")
    for method, avg_metrics in summary.items():
        log.info(f"\n  Method: {method}")
        for metric, value in avg_metrics.items():
            log.info(f"    {metric}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Run Tamil Cultural Story Generation evaluation.")
    parser.add_argument("--images", type=str, default="data/sample",
                        help="Path to image or directory of images.")
    parser.add_argument("--live", action="store_true",
                        help="Use live Gemini API instead of mock mode.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--target-words", type=int, default=400)
    args = parser.parse_args()

    config = AppConfig()
    if args.live:
        import os
        os.environ["APP_MODE"] = "live"
        config = AppConfig()

    setup_logger(config.log_level.value)
    log.info(f"Mode: {config.app_mode.value}")

    images_path = Path(args.images)
    if not images_path.exists():
        log.warning(f"Images path does not exist: {images_path}. Using synthetic image.")
        from PIL import Image
        images_path.mkdir(parents=True, exist_ok=True)
        dummy = Image.new("RGB", (300, 300), color=(180, 80, 40))
        dummy_path = images_path / "demo_image.jpg"
        dummy.save(dummy_path)

    image_paths = get_test_images(images_path)
    if not image_paths:
        log.error(f"No images found in {images_path}")
        sys.exit(1)

    log.info(f"Found {len(image_paths)} image(s) to evaluate.")

    exp_logger = ExperimentLogger(config.experiments_path)
    run_evaluation(image_paths, config, exp_logger, top_k=args.top_k, target_words=args.target_words)


if __name__ == "__main__":
    main()
