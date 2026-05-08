"""Edit-based correction strategy where the model outputs edit spans as JSON."""

from __future__ import annotations

from llm_grammar_bench.backends.base import BaseBackend
from llm_grammar_bench.strategies.base import BaseStrategy


class EditBasedStrategy(BaseStrategy):
    """Model outputs edits as a JSON array of {start, end, text} objects,
    which are then applied to the source to produce the corrected text."""

    def correct(self, backend: BaseBackend, source: str) -> str:
        raise NotImplementedError("EditBasedStrategy.correct not yet implemented")
