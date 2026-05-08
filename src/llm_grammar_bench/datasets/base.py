"""Abstract base class for benchmark datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_grammar_bench.types import Example


class BaseDataset(ABC):
    """Yields benchmark examples from a GEC corpus."""

    @abstractmethod
    def load(self, split: str) -> list[Example]:
        """Load examples for the given split.

        Args:
            split: Dataset split name (e.g. "train", "validation").

        Returns:
            A list of Example objects.
        """
        ...
