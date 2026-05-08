"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_examples() -> list:
    """Small set of synthetic examples for testing."""
    from llm_grammar_bench.types import Example

    return [
        Example(
            id="test-1",
            source="This are a sentence .",
            references=["This is a sentence ."],
            cefr="A",
        ),
        Example(
            id="test-2",
            source="I goes to school yesterday .",
            references=["I went to school yesterday ."],
            cefr="B",
        ),
        Example(
            id="test-3",
            source="She dont like the food .",
            references=["She doesn't like the food ."],
            cefr="A",
        ),
    ]
