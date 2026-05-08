"""Tests for result serialization."""

import json
import tempfile
from pathlib import Path

from llm_grammar_bench.evaluation.results import serialize_results
from llm_grammar_bench.types import BenchmarkResult, CorpusScores


def test_serialize_results_to_directory() -> None:
    """Test serializing results to a directory."""
    result = BenchmarkResult(
        model_id="hf:test-model",
        dataset_name="TestDataset",
        strategy_name="Seq2SeqStrategy",
        corpus_scores={
            "errant": CorpusScores(metric="errant", precision=0.8, recall=0.7, f_score=0.75),
        },
        per_sentence=[],
        by_cefr={},
        runtime_seconds=1.5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        serialize_results(result, tmpdir)
        output_dir = Path(tmpdir)
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["model_id"] == "hf:test-model"
        assert "errant" in data["corpus_scores"]
        assert data["corpus_scores"]["errant"]["f_score"] == 0.75


def test_serialize_results_to_file() -> None:
    """Test serializing results to a specific file path."""
    result = BenchmarkResult(
        model_id="hf:test-model",
        dataset_name="TestDataset",
        strategy_name="Seq2SeqStrategy",
        corpus_scores={},
        per_sentence=[],
        by_cefr={},
        runtime_seconds=0.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "output.json"
        serialize_results(result, str(filepath))
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data["model_id"] == "hf:test-model"


def test_serialize_results_with_cefr_breakdown() -> None:
    """Test serialization includes CEFR breakdown."""
    result = BenchmarkResult(
        model_id="hf:test-model",
        dataset_name="TestDataset",
        strategy_name="Seq2SeqStrategy",
        corpus_scores={
            "errant": CorpusScores(metric="errant", precision=0.0, recall=0.0, f_score=0.5),
        },
        per_sentence=[],
        by_cefr={
            "A": {
                "errant": CorpusScores(metric="errant", precision=0.0, recall=0.0, f_score=0.3),
            },
            "B": {
                "errant": CorpusScores(metric="errant", precision=0.0, recall=0.0, f_score=0.7),
            },
        },
        runtime_seconds=2.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        serialize_results(result, tmpdir)
        output_dir = Path(tmpdir)
        json_files = list(output_dir.glob("*.json"))
        data = json.loads(json_files[0].read_text())

        assert "A" in data["by_cefr"]
        assert "B" in data["by_cefr"]
        assert data["by_cefr"]["A"]["errant"]["f_score"] == 0.3
        assert data["by_cefr"]["B"]["errant"]["f_score"] == 0.7
