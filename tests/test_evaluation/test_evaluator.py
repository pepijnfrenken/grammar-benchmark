"""Tests for Evaluator."""

import unittest.mock as mock

import pytest

from llm_grammar_bench.types import Example


def test_evaluator_run_with_mock_backend() -> None:
    """Test evaluator.run() with mocked backend and dataset, verify BenchmarkResult structure."""
    from llm_grammar_bench.evaluation.evaluator import Evaluator

    # Create mock backend
    mock_backend = mock.Mock()
    mock_backend.model_id = "test-model"

    # Create mock dataset
    mock_dataset = mock.Mock()
    mock_dataset.load.return_value = [
        Example(
            id="test-1",
            source="This are wrong .",
            references=["This is wrong ."],
            cefr="A",
            metadata={"source": "test"},
        ),
        Example(
            id="test-2",
            source="She go to school .",
            references=["She goes to school ."],
            cefr="B",
            metadata={"source": "test"},
        ),
    ]

    # Create mock strategy
    mock_strategy = mock.Mock()
    mock_strategy.correct.side_effect = [
        "This is wrong .",
        "She goes to school .",
    ]

    # Create evaluator and run
    evaluator = Evaluator(
        backend=mock_backend,
        dataset=mock_dataset,
        strategy=mock_strategy,
        split="validation",
        max_sentences=None,
        beta=0.5,
    )

    result = evaluator.run(metrics=["errant"])

    # Verify BenchmarkResult structure
    assert result.model_id == "test-model"
    assert result.dataset_name == "Mock"  # Mock class name
    assert result.strategy_name == "Mock"  # Mock class name
    assert isinstance(result.corpus_scores, dict)
    assert isinstance(result.per_sentence, list)
    assert len(result.per_sentence) == 2
    assert result.per_sentence[0]["id"] == "test-1"
    assert result.per_sentence[0]["source"] == "This are wrong ."
    assert result.per_sentence[0]["hypothesis"] == "This is wrong ."
    assert result.runtime_seconds >= 0


def test_evaluator_respects_max_sentences() -> None:
    """Test evaluator respects max_sentences, only processes limited examples."""
    from llm_grammar_bench.evaluation.evaluator import Evaluator

    # Create mock backend
    mock_backend = mock.Mock()
    mock_backend.model_id = "test-model"

    # Create mock dataset with 10 examples
    examples = [
        Example(
            id=f"test-{i}",
            source=f"Error {i} .",
            references=[f"Correct {i} ."],
            cefr="A",
            metadata={"index": i},
        )
        for i in range(10)
    ]

    mock_dataset = mock.Mock()
    mock_dataset.load.return_value = examples

    # Create mock strategy
    mock_strategy = mock.Mock()
    mock_strategy.correct.side_effect = [f"Correct {i} ." for i in range(10)]

    # Create evaluator with max_sentences=3
    evaluator = Evaluator(
        backend=mock_backend,
        dataset=mock_dataset,
        strategy=mock_strategy,
        split="validation",
        max_sentences=3,
        beta=0.5,
    )

    result = evaluator.run(metrics=[])

    # Verify only 3 examples were processed
    assert len(result.per_sentence) == 3
    assert mock_strategy.correct.call_count == 3


def test_evaluator_cefr_breakdown() -> None:
    """Test evaluator with examples having different CEFR levels, verify by_cefr grouping."""
    from llm_grammar_bench.evaluation.evaluator import Evaluator

    # Create mock backend
    mock_backend = mock.Mock()
    mock_backend.model_id = "test-model"

    # Create mock dataset with mixed CEFR levels
    # Note: CEFR must be in metadata dict for evaluator to find it
    mock_dataset = mock.Mock()
    mock_dataset.load.return_value = [
        Example(
            id="a1",
            source="Error A1 .",
            references=["Correct A1 ."],
            cefr=None,
            metadata={"cefr": "A1"},
        ),
        Example(
            id="a2",
            source="Error A2 .",
            references=["Correct A2 ."],
            cefr=None,
            metadata={"cefr": "A2"},
        ),
        Example(
            id="b1",
            source="Error B1 .",
            references=["Correct B1 ."],
            cefr=None,
            metadata={"cefr": "B1"},
        ),
        Example(
            id="b2",
            source="Error B2 .",
            references=["Correct B2 ."],
            cefr=None,
            metadata={"cefr": "B2"},
        ),
    ]

    # Create mock strategy
    mock_strategy = mock.Mock()
    mock_strategy.correct.side_effect = [
        "Correct A1 .",
        "Correct A2 .",
        "Correct B1 .",
        "Correct B2 .",
    ]

    # Create evaluator
    evaluator = Evaluator(
        backend=mock_backend,
        dataset=mock_dataset,
        strategy=mock_strategy,
        split="validation",
        max_sentences=None,
        beta=0.5,
    )

    result = evaluator.run(metrics=[])

    # Verify by_cefr has correct groups
    assert "A1" in result.by_cefr
    assert "A2" in result.by_cefr
    assert "B1" in result.by_cefr
    assert "B2" in result.by_cefr
    assert len(result.by_cefr) == 4


def test_evaluator_no_cefr_metadata() -> None:
    """Test evaluator handles examples without CEFR metadata."""
    from llm_grammar_bench.evaluation.evaluator import Evaluator

    # Create mock backend
    mock_backend = mock.Mock()
    mock_backend.model_id = "test-model"

    # Create mock dataset with no CEFR
    mock_dataset = mock.Mock()
    mock_dataset.load.return_value = [
        Example(
            id="test-1",
            source="Error .",
            references=["Correct ."],
            cefr=None,
            metadata={},
        ),
    ]

    # Create mock strategy
    mock_strategy = mock.Mock()
    mock_strategy.correct.return_value = "Correct ."

    # Create evaluator
    evaluator = Evaluator(
        backend=mock_backend,
        dataset=mock_dataset,
        strategy=mock_strategy,
        split="validation",
        max_sentences=None,
        beta=0.5,
    )

    result = evaluator.run(metrics=[])

    # Verify result is valid even without CEFR grouping
    assert len(result.per_sentence) == 1
    assert result.by_cefr == {}


def test_stratified_sample_preserves_group_proportions() -> None:
    """Test stratified sampling keeps the CEFR distribution close to the full set."""
    from llm_grammar_bench.evaluation.evaluator import _stratified_sample

    examples = [
        Example(
            id=f"a-{i}",
            source=f"A error {i}.",
            references=[f"A correction {i}."],
            metadata={"cefr": "A"},
        )
        for i in range(6)
    ]
    examples.extend(
        Example(
            id=f"b-{i}",
            source=f"B error {i}.",
            references=[f"B correction {i}."],
            metadata={"cefr": "B"},
        )
        for i in range(4)
    )

    sampled = _stratified_sample(examples, sample_size=5, metadata_key="cefr", seed=7)
    sampled_ids = [example.id for example in sampled]

    assert len(sampled) == 5
    assert sum(example.metadata["cefr"] == "A" for example in sampled) == 3
    assert sum(example.metadata["cefr"] == "B" for example in sampled) == 2
    assert sampled_ids == [example.id for example in _stratified_sample(examples, 5, "cefr", 7)]


def test_stratified_sample_rejects_empty_sample() -> None:
    """Test stratified sampling rejects non-positive sample sizes."""
    from llm_grammar_bench.evaluation.evaluator import _stratified_sample

    examples = [
        Example(
            id="a-1",
            source="Error.",
            references=["Correction."],
            metadata={"cefr": "A"},
        )
    ]

    with pytest.raises(ValueError, match="greater than 0"):
        _stratified_sample(examples, sample_size=0, metadata_key="cefr", seed=0)


def test_evaluator_uses_stratified_sample_size() -> None:
    """Test evaluator processes a deterministic stratified sample."""
    from llm_grammar_bench.evaluation.evaluator import Evaluator

    mock_backend = mock.Mock()
    mock_backend.model_id = "test-model"
    mock_dataset = mock.Mock()
    mock_dataset.load.return_value = [
        Example(
            id=f"a-{i}",
            source=f"A error {i}.",
            references=[f"A correction {i}."],
            metadata={"cefr": "A"},
        )
        for i in range(6)
    ]
    mock_dataset.load.return_value.extend(
        Example(
            id=f"b-{i}",
            source=f"B error {i}.",
            references=[f"B correction {i}."],
            metadata={"cefr": "B"},
        )
        for i in range(4)
    )
    mock_strategy = mock.Mock()
    mock_strategy.correct.side_effect = lambda _backend, source: source.replace(
        "error", "correction"
    )

    evaluator = Evaluator(
        backend=mock_backend,
        dataset=mock_dataset,
        strategy=mock_strategy,
        sample_size=5,
        sample_seed=7,
    )

    result = evaluator.run(metrics=[])

    assert len(result.per_sentence) == 5
    assert mock_strategy.correct.call_count == 5
    assert len(result.by_cefr["A"]) == 0
    assert len(result.by_cefr["B"]) == 0
    assert sum(item["metadata"]["cefr"] == "A" for item in result.per_sentence) == 3
    assert sum(item["metadata"]["cefr"] == "B" for item in result.per_sentence) == 2
