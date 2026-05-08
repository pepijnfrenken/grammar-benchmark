"""Tests for BEA-2019 dataset loading."""


def test_bea2019_dataset_creation() -> None:
    from llm_grammar_bench.datasets.bea2019 import BEA2019Dataset

    dataset = BEA2019Dataset()
    assert dataset._cefr_filter is None


def test_bea2019_dataset_cefr_filter() -> None:
    from llm_grammar_bench.datasets.bea2019 import BEA2019Dataset

    dataset = BEA2019Dataset(cefr_filter=["A", "B"])
    assert dataset._cefr_filter == {"A", "B"}


def test_bea2019_dataset_split_validation() -> None:
    """Test that unknown splits raise ValueError."""
    import pytest

    from llm_grammar_bench.datasets.bea2019 import BEA2019Dataset

    dataset = BEA2019Dataset()
    with pytest.raises(ValueError, match="Unknown split"):
        dataset.load("unknown")
