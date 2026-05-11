"""Tests for the dataset factory function."""

import pytest


def test_load_dataset_unknown() -> None:
    """Test that load_dataset raises ValueError for unknown dataset name."""
    from llm_grammar_bench.datasets import load_dataset

    try:
        load_dataset("unknown_dataset")
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        assert "Unknown dataset" in str(e)
        assert "unknown_dataset" in str(e)


def test_load_dataset_bea2019() -> None:
    """Test that load_dataset returns BEA2019Dataset for 'bea2019'."""
    from llm_grammar_bench.datasets import load_dataset
    from llm_grammar_bench.datasets.bea2019 import BEA2019Dataset

    dataset = load_dataset("bea2019")
    assert isinstance(dataset, BEA2019Dataset)
