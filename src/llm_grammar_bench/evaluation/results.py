"""Result serialization for benchmark output."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from llm_grammar_bench.types import BenchmarkResult


def serialize_results(results: BenchmarkResult, output_path: str | Path) -> None:
    """Write benchmark results to a JSON file.

    Args:
        results: The benchmark result to serialize.
        output_path: Path to write the JSON file.
    """
    output_path = Path(output_path)

    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"{results.model_id.replace(':', '_')}_{results.dataset_name}.json"
        output_path = output_path / filename
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, handling dataclass nesting
    data = _result_to_dict(results)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    import logging

    logger = logging.getLogger(__name__)
    logger.info("Results written to %s", output_path)


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
