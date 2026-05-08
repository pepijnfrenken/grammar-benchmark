"""Few-shot strategy that prepends example corrections before the target sentence."""

from __future__ import annotations

from llm_grammar_bench.backends.base import BaseBackend
from llm_grammar_bench.strategies.base import BaseStrategy
from llm_grammar_bench.types import Example


class FewShotStrategy(BaseStrategy):
    """Wraps another strategy with N example corrections prepended to the prompt.

    Examples are drawn from the dataset's training/dev split for in-domain few-shot.
    """

    def __init__(
        self,
        base: BaseStrategy,
        examples: list[Example],
        n: int = 3,
    ) -> None:
        self._base = base
        self._examples = examples
        self._n = n

    def correct(self, backend: BaseBackend, source: str) -> str:
        raise NotImplementedError("FewShotStrategy.correct not yet implemented")
