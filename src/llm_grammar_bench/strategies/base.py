"""Abstract base class for correction strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_grammar_bench.backends.base import BaseBackend


class BaseStrategy(ABC):
    """A strategy formats a source sentence into a prompt, sends it to a backend,
    and parses the model output into a corrected sentence."""

    @abstractmethod
    def correct(self, backend: BaseBackend, source: str) -> str:
        """Correct one sentence using the given backend.

        Args:
            backend: The model backend to use for inference.
            source: The erroneous source sentence.

        Returns:
            The corrected sentence.
        """
        ...
