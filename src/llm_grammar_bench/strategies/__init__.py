"""Correction strategies that format prompts and parse model output."""

from llm_grammar_bench.strategies.base import BaseStrategy


def load_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
    """Factory to load a correction strategy by name.

    Args:
        strategy_name: One of "seq2seq", "edit_based", "few_shot".
        **kwargs: Strategy-specific configuration.

    Returns:
        A configured BaseStrategy instance.

    Raises:
        ValueError: If the strategy name is unknown.
    """
    match strategy_name:
        case "seq2seq":
            from llm_grammar_bench.strategies.seq2seq import Seq2SeqStrategy

            return Seq2SeqStrategy(**kwargs)
        case "edit_based":
            from llm_grammar_bench.strategies.edit_based import EditBasedStrategy

            return EditBasedStrategy(**kwargs)
        case "few_shot":
            from llm_grammar_bench.strategies.few_shot import FewShotStrategy

            return FewShotStrategy(**kwargs)
        case _:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. Valid options: seq2seq, edit_based, few_shot"
            )


__all__ = ["BaseStrategy", "load_strategy"]
