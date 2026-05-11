"""Result serialization for benchmark output."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from llm_grammar_bench.types import BenchmarkResult

logger = logging.getLogger(__name__)


def serialize_results(results: BenchmarkResult, output_path: str | Path) -> str:
    """Write benchmark results to a JSON file.

    Args:
        results: The benchmark result to serialize.
        output_path: Path to write the JSON file.

    Returns:
        The actual file path written.
    """
    output_path = Path(output_path)

    if output_path.is_dir() or output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
        safe_model = results.model_id.replace(":", "_").replace("/", "_")
        filename = f"{safe_model}_{results.dataset_name}.json"
        output_path = output_path / filename
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, handling dataclass nesting
    data = _result_to_dict(results)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Results written to %s", output_path)
    return str(output_path)


def _result_to_dict(results: BenchmarkResult) -> dict:
    """Convert a BenchmarkResult to a JSON-serializable dict."""
    # Use dataclass asdict for the main structure, then handle CorpusScores nesting
    base = asdict(results)

    # Convert corpus_scores values (CorpusScores) to dicts
    base["corpus_scores"] = {k: asdict(v) for k, v in results.corpus_scores.items()}

    # Convert by_cefr nesting
    base["by_cefr"] = {
        cefr: {k: asdict(v) for k, v in metrics.items()}
        for cefr, metrics in results.by_cefr.items()
    }

    return base
