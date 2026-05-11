"""Tests for the FewShot strategy."""

from unittest.mock import Mock

from llm_grammar_bench.strategies.few_shot import FewShotStrategy
from llm_grammar_bench.types import Example


def test_few_shot_with_examples() -> None:
    """Test that FewShotStrategy prepends examples to the prompt."""
    # Create mock base strategy
    base_strategy = Mock()
    base_strategy.correct.return_value = "corrected result"

    # Create examples
    examples = [
        Example(
            id="ex1",
            source="He are going",
            references=["He is going"],
            cefr="A1",
        ),
        Example(
            id="ex2",
            source="She don't know",
            references=["She doesn't know"],
            cefr="A2",
        ),
    ]

    # Create FewShotStrategy with 2 examples
    strategy = FewShotStrategy(base=base_strategy, examples=examples, n=2)

    # Call correct
    result = strategy.correct(base_strategy, "I are happy")

    # Verify the base strategy received a prompt with examples
    base_strategy.correct.assert_called_once()
    call_args = base_strategy.correct.call_args
    prompt = call_args[0][1]  # Second argument is the augmented prompt

    # Check that examples are in the prompt
    assert "Source: He are going" in prompt
    assert "Corrected: He is going" in prompt
    assert "Source: She don't know" in prompt
    assert "Corrected: She doesn't know" in prompt
    # Check that target sentence is in the prompt
    assert "Source: I are happy" in prompt
    # Result should be what the base strategy returned
    assert result == "corrected result"


def test_few_shot_no_examples() -> None:
    """Test that when examples list is empty, original source is passed through."""
    base_strategy = Mock()
    base_strategy.correct.return_value = "corrected result"

    # Empty examples list
    examples: list[Example] = []

    strategy = FewShotStrategy(base=base_strategy, examples=examples, n=3)

    result = strategy.correct(base_strategy, "I are happy")

    # With no examples, the original source should be passed unchanged
    base_strategy.correct.assert_called_once_with(base_strategy, "I are happy")
    assert result == "corrected result"


def test_few_shot_with_n_limit() -> None:
    """Test that only n examples are used when there are more examples available."""
    base_strategy = Mock()
    base_strategy.correct.return_value = "corrected result"

    # Create 5 examples
    examples = [
        Example(
            id=f"ex{i}",
            source=f"Error {i}",
            references=[f"Correct {i}"],
            cefr="A1",
        )
        for i in range(5)
    ]

    # Create FewShotStrategy with n=2 but 5 examples available
    strategy = FewShotStrategy(base=base_strategy, examples=examples, n=2)

    strategy.correct(base_strategy, "I are happy")

    # Check that only 2 examples are in the prompt
    call_args = base_strategy.correct.call_args
    prompt = call_args[0][1]

    # Only ex0 and ex1 should be in the prompt
    assert "Source: Error 0" in prompt
    assert "Source: Error 1" in prompt
    # ex2, ex3, ex4 should NOT be in the prompt
    assert "Source: Error 2" not in prompt
    assert "Source: Error 3" not in prompt
    assert "Source: Error 4" not in prompt


def test_few_shot_uses_first_reference() -> None:
    """Test that the first reference is used as the corrected text."""
    base_strategy = Mock()
    base_strategy.correct.return_value = "final result"

    # Example with multiple references
    examples = [
        Example(
            id="ex1",
            source="He go to school",
            references=["He goes to school", "He went to school", "He is going to school"],
            cefr="A1",
        ),
    ]

    strategy = FewShotStrategy(base=base_strategy, examples=examples, n=1)
    strategy.correct(base_strategy, "I go to school")

    call_args = base_strategy.correct.call_args
    prompt = call_args[0][1]

    # Should use the first reference only
    assert "Corrected: He goes to school" in prompt
    # Other references should not appear
    assert "He went to school" not in prompt
    assert "He is going to school" not in prompt


def test_few_shot_example_with_no_references() -> None:
    """Test that example with empty references list falls back to using source."""
    base_strategy = Mock()
    base_strategy.correct.return_value = "result"

    # Example with empty references list
    examples = [
        Example(
            id="ex1",
            source="He are going",
            references=[],  # Empty references list
            cefr="A1",
        ),
    ]

    strategy = FewShotStrategy(base=base_strategy, examples=examples, n=1)
    strategy.correct(base_strategy, "She are happy")

    call_args = base_strategy.correct.call_args
    prompt = call_args[0][1]

    # Should fall back to using source as the corrected text
    assert "Source: He are going" in prompt
    assert "Corrected: He are going" in prompt
