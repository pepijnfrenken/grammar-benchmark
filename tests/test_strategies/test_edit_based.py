"""Tests for the EditBasedStrategy."""

from unittest.mock import Mock

import pytest


def test_edit_based_uses_default_system_prompt() -> None:
    """Verify EditBasedStrategy uses default system prompt when not provided."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    assert strategy._system_prompt is not None
    assert "grammatical error correction" in strategy._system_prompt.lower()


def test_edit_based_custom_system_prompt() -> None:
    """Verify custom system prompt is stored correctly."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    custom_prompt = "My custom system prompt for testing"
    strategy = EditBasedStrategy(system_prompt=custom_prompt)
    assert strategy._system_prompt == custom_prompt


def test_extract_json_direct() -> None:
    """Test _extract_json with clean JSON array input."""
    from llm_grammar_bench.strategies.edit_based import _extract_json

    json_input = '[{"start": 0, "end": 3, "text": "The"}]'
    result = _extract_json(json_input)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["start"] == 0
    assert result[0]["end"] == 3
    assert result[0]["text"] == "The"


def test_extract_json_with_markdown_fence() -> None:
    """Test _extract_json with markdown code fence wrapping."""
    from llm_grammar_bench.strategies.edit_based import _extract_json

    markdown_input = """```json
[{"start": 5, "end": 8, "text": "cat"}]
```"""
    result = _extract_json(markdown_input)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["start"] == 5
    assert result[0]["end"] == 8
    assert result[0]["text"] == "cat"


def test_extract_json_with_extra_text() -> None:
    """Test _extract_json with text before and after JSON array."""
    from llm_grammar_bench.strategies.edit_based import _extract_json

    text_with_json = """Here are the corrections:
[{"start": 0, "end": 1, "text": "A"}]
That's all!"""
    result = _extract_json(text_with_json)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["start"] == 0
    assert result[0]["end"] == 1
    assert result[0]["text"] == "A"


def test_extract_json_invalid() -> None:
    """Test _extract_json raises ValueError on non-JSON input."""
    from llm_grammar_bench.strategies.edit_based import _extract_json

    invalid_input = "This is not JSON at all"
    with pytest.raises(ValueError, match="Could not extract valid JSON edits"):
        _extract_json(invalid_input)


def test_correct_applies_edits() -> None:
    """Test correct() applies edits from backend response."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    backend = Mock()
    backend.correct.return_value = '[{"start": 0, "end": 5, "text": "Hello"}]'

    source = "wrong sentence"
    result = strategy.correct(backend, source)

    assert result == "Hello sentence"
    backend.correct.assert_called_once()
    call_args = backend.correct.call_args
    assert "wrong sentence" in call_args[0][0]


def test_correct_parsing_failure_returns_source() -> None:
    """Test correct() returns original source when parsing fails."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    backend = Mock()
    backend.correct.return_value = "garbage that is not json"

    source = "original sentence"
    result = strategy.correct(backend, source)

    assert result == source


def test_correct_applies_reverse_order() -> None:
    """Test correct() applies edits in reverse order to preserve indices."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    backend = Mock()
    # Edits provided in forward order: replace "is" with "are", then replace "The" with "This"
    backend.correct.return_value = (
        '[{"start": 0, "end": 3, "text": "This"}, {"start": 4, "end": 6, "text": "are"}]'
    )

    source = "The is wrong"
    result = strategy.correct(backend, source)

    assert result == "This are wrong"


def test_correct_empty_edits_list() -> None:
    """Test correct() with empty edits list returns source unchanged."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    backend = Mock()
    backend.correct.return_value = "[]"

    source = "some sentence"
    result = strategy.correct(backend, source)

    assert result == source


def test_correct_edit_at_start() -> None:
    """Test correct() applies edit at the very start of string."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    backend = Mock()
    backend.correct.return_value = '[{"start": 0, "end": 2, "text": "The"}]'

    source = "an apple"
    result = strategy.correct(backend, source)

    assert result == "The apple"


def test_correct_edit_at_end() -> None:
    """Test correct() applies edit at the very end of string."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    backend = Mock()
    backend.correct.return_value = '[{"start": 10, "end": 16, "text": "apple."}]'

    source = "I like an orange"
    result = strategy.correct(backend, source)

    assert result == "I like an apple."


def test_correct_multiple_edits_various_positions() -> None:
    """Test correct() with multiple edits at different positions."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    backend = Mock()
    # Replace "The" (0-3) with "This", replace "is" (4-6) with "are", replace "." (13-14) with "!"
    backend.correct.return_value = (
        '[{"start": 0, "end": 3, "text": "This"},'
        ' {"start": 4, "end": 6, "text": "are"},'
        ' {"start": 13, "end": 14, "text": "!"}]'
    )

    source = "The is a test."
    result = strategy.correct(backend, source)

    assert result == "This are a test!"


def test_correct_out_of_bounds_edits_ignored() -> None:
    """Test correct() ignores edits that are out of bounds."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    backend = Mock()
    # Valid edit + out of bounds edit
    backend.correct.return_value = (
        '[{"start": 0, "end": 3, "text": "The"}, {"start": 100, "end": 200, "text": "ignored"}]'
    )

    source = "bad sentence"
    result = strategy.correct(backend, source)

    assert result == "The sentence"


def test_correct_system_prompt_passed_to_backend() -> None:
    """Test correct() passes the system prompt to backend.correct()."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    custom_prompt = "Custom system prompt for testing"
    strategy = EditBasedStrategy(system_prompt=custom_prompt)
    backend = Mock()
    backend.correct.return_value = "[]"

    strategy.correct(backend, "source text")

    backend.correct.assert_called_once()
    call_kwargs = backend.correct.call_args[1]
    assert call_kwargs["system_prompt"] == custom_prompt


def test_extract_json_single_valid_array() -> None:
    """Test _extract_json extracts valid array even with extra text."""
    from llm_grammar_bench.strategies.edit_based import _extract_json

    text = 'Here is the JSON: [{"start": 0, "end": 1, "text": "A"}]'
    result = _extract_json(text)
    assert len(result) == 1
    assert result[0]["text"] == "A"


def test_extract_json_with_code_fence_no_lang() -> None:
    """Test _extract_json with code fence without language specification."""
    from llm_grammar_bench.strategies.edit_based import _extract_json

    markdown_input = """```
[{"start": 1, "end": 2, "text": "x"}]
```"""
    result = _extract_json(markdown_input)
    assert len(result) == 1
    assert result[0]["text"] == "x"


def test_extract_json_nested_objects() -> None:
    """Test _extract_json with JSON containing nested/complex objects."""
    from llm_grammar_bench.strategies.edit_based import _extract_json

    json_input = '[{"start": 0, "end": 5, "text": "Hello", "metadata": {"type": "replacement"}}]'
    result = _extract_json(json_input)
    assert len(result) == 1
    assert result[0]["text"] == "Hello"
    assert result[0]["metadata"]["type"] == "replacement"


def test_correct_replacement_with_special_characters() -> None:
    """Test correct() with replacements containing special characters."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    backend = Mock()
    backend.correct.return_value = '[{"start": 0, "end": 3, "text": "Café"}]'

    source = "bad café"
    result = strategy.correct(backend, source)

    assert result == "Café café"


def test_correct_insertion_by_overlapping_span() -> None:
    """Test correct() handles insertion (start == end scenario)."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    backend = Mock()
    # Insertion: start == end means insert at position 1
    backend.correct.return_value = '[{"start": 1, "end": 1, "text": "X"}]'

    source = "ab"
    result = strategy.correct(backend, source)

    assert result == "aXb"


def test_correct_deletion_by_empty_text() -> None:
    """Test correct() handles deletion (empty replacement text)."""
    from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

    strategy = EditBasedStrategy()
    backend = Mock()
    # Deletion: replace span with empty string
    backend.correct.return_value = '[{"start": 0, "end": 2, "text": ""}]'

    source = "abcd"
    result = strategy.correct(backend, source)

    assert result == "cd"
