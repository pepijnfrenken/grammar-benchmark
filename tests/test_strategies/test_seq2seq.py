"""Tests for the Seq2Seq strategy."""


def test_seq2seq_uses_default_prefix() -> None:
    from llm_grammar_bench.strategies.seq2seq import Seq2SeqStrategy

    strategy = Seq2SeqStrategy()
    assert strategy.task_prefix == "grammar: "


def test_seq2seq_custom_prefix() -> None:
    from llm_grammar_bench.strategies.seq2seq import Seq2SeqStrategy

    strategy = Seq2SeqStrategy(task_prefix="correct: ")
    assert strategy.task_prefix == "correct: "


def test_seq2seq_correct_calls_backend() -> None:
    """Test that correct() calls backend.correct() with the expected prompt."""
    from unittest.mock import Mock

    from llm_grammar_bench.strategies.seq2seq import Seq2SeqStrategy

    strategy = Seq2SeqStrategy()
    mock_backend = Mock()
    mock_backend.correct.return_value = "corrected text"

    result = strategy.correct(mock_backend, "test sentence")

    # Verify backend.correct was called with the right prompt
    mock_backend.correct.assert_called_once()
    call_args = mock_backend.correct.call_args
    assert call_args[0][0] == "grammar: test sentence"
    assert result == "corrected text"


def test_seq2seq_strips_echoed_prefix() -> None:
    """Test that correct() strips the echoed task prefix from output."""
    from unittest.mock import Mock

    from llm_grammar_bench.strategies.seq2seq import Seq2SeqStrategy

    strategy = Seq2SeqStrategy()
    mock_backend = Mock()
    mock_backend.correct.return_value = "grammar: corrected text"

    result = strategy.correct(mock_backend, "test sentence")

    # Should strip the prefix if the model echoed it back
    assert result == "corrected text"


def test_seq2seq_preserves_no_prefix_output() -> None:
    """Test that correct() preserves output when there's no prefix to strip."""
    from unittest.mock import Mock

    from llm_grammar_bench.strategies.seq2seq import Seq2SeqStrategy

    strategy = Seq2SeqStrategy()
    mock_backend = Mock()
    mock_backend.correct.return_value = "corrected text"

    result = strategy.correct(mock_backend, "test sentence")

    # Should preserve the output as-is since no prefix was echoed
    assert result == "corrected text"
