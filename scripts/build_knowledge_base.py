"""
scripts/build_knowledge_base.py
────────────────────────────────────────────────────────────────
Script to ingest the raw JSONL KB, embed all records,
and build the ChromaDB vector index.

Run from project root:
    python scripts/build_knowledge_base.py

Supports:
  --force-rebuild   Delete and rebuild the index
  --skip-synthetic  Exclude synthetic demo records
  --kb-path         Override default KB JSONL path
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import AppConfig, AppMode
from src.embeddings.minilm_embedder import MiniLMEmbedder
from src.embeddings.mock_embedder import MockEmbedder
from src.knowledge_base.ingestion import load_kb_from_jsonl
from src.retrieval.engine import ChromaRetrievalEngine
from src.utils.logger import get_logger, setup_logger

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build the Tamil Cultural Knowledge Base index.")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Delete and rebuild the ChromaDB index.")
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="Exclude synthetic demonstration records.")
    parser.add_argument("--kb-path", type=str, default=None,
                        help="Override the default KB JSONL file path.")
    parser.add_argument("--mock-embedder", action="store_true",
                        help="Use mock embedder (for testing without model downloads).")
    args = parser.parse_args()

    config = AppConfig()
    setup_logger(config.log_level.value)

    kb_path = Path(args.kb_path) if args.kb_path else config.kb_raw_path
    chroma_path = config.chroma_db_path

    log.info(f"Loading KB from: {kb_path}")
    records = load_kb_from_jsonl(
        kb_path,
        skip_invalid=True,
        skip_synthetic=args.skip_synthetic,
    )
    log.info(f"Loaded {len(records)} valid records.")

    # Choose embedder
    if args.mock_embedder or config.is_mock_mode:
        log.info("Using MockEmbedder (mock mode).")
        embedder = MockEmbedder()
    else:
        log.info("Using MiniLM embedder (will download model on first run ~22 MB).")
        embedder = MiniLMEmbedder()

    log.info(f"Building ChromaDB index at: {chroma_path}")
    engine = ChromaRetrievalEngine(embedder, persist_path=chroma_path)
    count = engine.build_index(records, force_rebuild=args.force_rebuild)

    log.info(f"✅ Index built successfully: {count} records indexed.")
    log.info(f"ChromaDB path: {chroma_path}")

    # Quick test retrieval
    log.info("Running test retrieval: 'Tamil harvest festival Pongal clay pot'")
    result = engine.retrieve("Tamil harvest festival Pongal clay pot", top_k=3)
    if result.evidence_found:
        for e in result.retrieved_records:
            log.info(f"  [{e.retrieval_rank}] {e.topic} (score={e.relevance_score:.3f})")
    else:
        log.warning("Test retrieval returned no results.")

    log.info("Knowledge base build complete.")


if __name__ == "__main__":
    main()
