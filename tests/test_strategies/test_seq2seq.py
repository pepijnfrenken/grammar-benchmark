"""Tests for the Seq2Seq strategy."""


def test_seq2seq_uses_default_prefix() -> None:
    from llm_grammar_bench.strategies.seq2seq import Seq2SeqStrategy

    strategy = Seq2SeqStrategy()
    assert strategy.task_prefix == "grammar: "


def test_seq2seq_custom_prefix() -> None:
    from llm_grammar_bench.strategies.seq2seq import Seq2SeqStrategy

    strategy = Seq2SeqStrategy(task_prefix="correct: ")
    assert strategy.task_prefix == "correct: "
