"""Tests for the strategy factory function."""

from unittest.mock import Mock

import pytest

from llm_grammar_bench.types import Example


def test_load_strategy_unknown() -> None:
    """Test that load_strategy raises ValueError for unknown strategy name."""
    from llm_grammar_bench.strategies import load_strategy

    try:
        load_strategy("unknown_strategy")
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        assert "Unknown strategy" in str(e)
        assert "unknown_strategy" in str(e)


def test_load_strategy_seq2seq() -> None:
    """Test that load_strategy returns Seq2SeqStrategy for 'seq2seq'."""
    from llm_grammar_bench.strategies import load_strategy
    from llm_grammar_bench.strategies.seq2seq import Seq2SeqStrategy

    strategy = load_strategy("seq2seq")
    assert isinstance(strategy, Seq2SeqStrategy)


def test_load_strategy_edit_based() -> None:
    """Test that load_strategy returns EditBasedStrategy for 'edit_based'."""
    from llm_grammar_bench.strategies import load_strategy
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = load_strategy("edit_based")
    assert isinstance(strategy, EditBasedStrategy)


def test_load_strategy_few_shot() -> None:
    """Test that load_strategy returns FewShotStrategy for 'few_shot'."""
    from llm_grammar_bench.strategies import load_strategy
    from llm_grammar_bench.strategies.few_shot import FewShotStrategy

    # Create a mock base strategy
    mock_base = Mock()

    # Create example instances
    examples = [
        Example(id="ex1", source="test source", references=["test reference"]),
    ]

    strategy = load_strategy("few_shot", base=mock_base, examples=examples)
    assert isinstance(strategy, FewShotStrategy)
