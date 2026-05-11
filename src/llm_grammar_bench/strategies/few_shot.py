"""Few-shot strategy that prepends example corrections before the target sentence."""

from __future__ import annotations

from llm_grammar_bench.backends.base import BaseBackend
from llm_grammar_bench.strategies.base import BaseStrategy
from llm_grammar_bench.types import Example


class FewShotStrategy(BaseStrategy):
    """Wraps another strategy with N example corrections prepended to the prompt.

    Examples are drawn from the dataset for in-domain few-shot prompting.
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
        """Correct a single sentence with few-shot examples prepended.

        Args:
            backend: The model backend to use for inference.
            source: The erroneous source sentence.

        Returns:
            The corrected sentence.
        """
        shots = self._examples[: self._n]
        if not shots:
            return self._base.correct(backend, source)

        prefix_parts: list[str] = []
        for ex in shots:
            ref = ex.references[0] if ex.references else ex.source
            prefix_parts.append(f"Source: {ex.source}\nCorrected: {ref}")

        prefix = "\n\n".join(prefix_parts)
        augmented = f"{prefix}\n\nSource: {source}"
        return self._base.correct(backend, augmented)
