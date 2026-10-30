"""
src/evaluation/logger.py
────────────────────────────────────────────────────────────────
Experiment result logger — saves evaluation results to JSON for
later analysis in notebooks.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.metrics import EvaluationResult
from src.utils.errors import ExperimentLogError
from src.utils.logger import get_logger

log = get_logger(__name__)


class ExperimentLogger:
    """Saves experiment results to a JSONL file for analysis."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"experiment_{self.run_id}.jsonl"
        log.info(f"ExperimentLogger: writing to {self.results_file}")

    def log_result(self, result: EvaluationResult) -> None:
        """Append one evaluation result to the experiment log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            **result.to_dict(),
        }
        try:
            with self.results_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.debug(f"Logged result for method={result.method}, image={result.image_id}")
        except OSError as e:
            raise ExperimentLogError(f"Failed to write experiment log: {e}") from e

    def log_comparison(
        self,
        results: list[EvaluationResult],
        image_id: str = "",
    ) -> None:
        """Log a side-by-side comparison of methods for one image."""
        for r in results:
            r.image_id = r.image_id or image_id
            self.log_result(r)

    def summarise(self) -> dict:
        """Read back all logged results and compute per-method averages."""
        if not self.results_file.exists():
            return {}

        results_by_method: dict[str, list[dict]] = {}
        with self.results_file.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                method = entry.get("method", "unknown")
                results_by_method.setdefault(method, []).append(entry)

        summary = {}
        for method, entries in results_by_method.items():
            all_metrics: dict[str, list[float]] = {}
            for entry in entries:
                for metric, value in entry.get("metrics", {}).items():
                    all_metrics.setdefault(metric, []).append(value)
            summary[method] = {
                metric: round(sum(vals) / len(vals), 4)
                for metric, vals in all_metrics.items()
            }
            summary[method]["n_runs"] = len(entries)
        return summary
