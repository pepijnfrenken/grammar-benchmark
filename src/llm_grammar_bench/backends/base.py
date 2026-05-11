"""Abstract base class for all model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BackendError(Exception):
    """Raised when a backend fails to produce a correction."""


class BaseBackend(ABC):
    """Abstract interface for a model backend.

    Each backend handles the transport, authentication, retry logic, and
    output caching for a specific model provider (OpenAI, Anthropic, etc.).
    """

    @abstractmethod
    def correct(self, text: str, **kwargs: Any) -> str:
        """Return the corrected version of the input text.

        Args:
            text: The erroneous source text to correct.
            **kwargs: Backend-specific overrides (temperature, max_tokens, etc.).

        Returns:
            The corrected text.

        Raises:
            BackendError: If inference fails after all retries.
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """A human-readable identifier for this model (e.g. 'openai:gpt-4o')."""
        ...

    @property
    @abstractmethod
    def metadata(self) -> dict[str, str]:
        """Provider metadata: provider name, model name, parameter count, etc."""
        ...

    def release(self) -> None:
        """Release backend resources (e.g. GPU memory).

        Override in subclasses that hold heavyweight resources.
        Default implementation is a no-op.
        """
        return
