"""Tests for MetricsRunner."""


def test_metrics_runner_creation() -> None:
    from llm_grammar_bench.evaluation.metrics import MetricsRunner

    runner = MetricsRunner()
    assert runner is not None


def test_metrics_runner_compute_errant_and_gleu() -> None:
    """Test that ERRANT and GLEU metrics compute without errors."""
    from llm_grammar_bench.evaluation.metrics import MetricsRunner

    runner = MetricsRunner(beta=0.5)
    sources = ["I goes to school .", "She walk to the park ."]
    hypotheses = ["I go to school .", "She walks to the park ."]
    references = [["I go to school ."], ["She walks to the park ."]]

    scores = runner.compute(sources, hypotheses, references, metric_names=["errant", "gleu"])

    assert "errant" in scores
    assert "gleu" in scores
    assert scores["errant"].metric == "errant"
    assert scores["gleu"].metric == "gleu"
    assert scores["errant"].f_score is not None
    assert scores["gleu"].f_score is not None


def test_metrics_runner_rejects_unknown_metric() -> None:
    """Test that unknown metric names raise ValueError."""
    import pytest

    from llm_grammar_bench.evaluation.metrics import MetricsRunner

    runner = MetricsRunner()
    with pytest.raises(ValueError, match="Unknown metrics"):
        runner.compute([], [], [], metric_names=["nonexistent"])


def test_transpose_references_single_ref() -> None:
    """Test reference transposition with single reference per sentence."""
    from llm_grammar_bench.evaluation.metrics import _transpose_references

    refs = [["ref A1"], ["ref B1"], ["ref C1"]]
    transposed = _transpose_references(refs)

    # (3 sents, 1 ref) -> (1 ref, 3 sents)
    assert len(transposed) == 1
    assert len(transposed[0]) == 3
    assert transposed[0] == ["ref A1", "ref B1", "ref C1"]


def test_transpose_references_multiple_refs() -> None:
    """Test reference transposition with multiple references per sentence."""
    from llm_grammar_bench.evaluation.metrics import _transpose_references

    refs = [["ref A1", "ref A2"], ["ref B1", "ref B2"]]
    transposed = _transpose_references(refs)

    # (2 sents, 2 refs) -> (2 refs, 2 sents)
    assert len(transposed) == 2
    assert transposed[0] == ["ref A1", "ref B1"]
    assert transposed[1] == ["ref A2", "ref B2"]


def test_transpose_references_uneven() -> None:
    """Test transposition with uneven number of references."""
    from llm_grammar_bench.evaluation.metrics import _transpose_references

    refs = [["ref A1", "ref A2"], ["ref B1"]]
    transposed = _transpose_references(refs)

    assert len(transposed) == 2
    assert transposed[0] == ["ref A1", "ref B1"]
    assert transposed[1] == ["ref A2", ""]


def test_transpose_references_empty() -> None:
    """Test transposition with empty input."""
    from llm_grammar_bench.evaluation.metrics import _transpose_references

    assert _transpose_references([]) == []
