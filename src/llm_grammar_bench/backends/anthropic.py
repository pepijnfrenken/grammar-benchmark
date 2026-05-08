"""Anthropic API backend."""

from __future__ import annotations

import logging
import os
from typing import Any

from llm_grammar_bench.backends.base import BackendError, BaseBackend
from llm_grammar_bench.utils.cache import CacheStore
from llm_grammar_bench.utils.retry import retry

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.0


class AnthropicBackend(BaseBackend):
    """Backend for Anthropic message models (Claude 3 Opus, Claude 3.5 Sonnet, etc.).

    Requires: pip install llmgrammarbench[anthropic]
    Sets ANTHROPIC_API_KEY from environment.
    """

    def __init__(
        self,
        model: str,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: float = _DEFAULT_TIMEOUT,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._cache = CacheStore(cache_dir)

        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            logger.warning("ANTHROPIC_API_KEY not set. Anthropic backend will fail at runtime.")

        self._client: Any | None = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic backend requires the anthropic package. "
                "Install with: pip install llmgrammarbench[anthropic]"
            ) from None
        self._client = Anthropic(api_key=self._api_key, timeout=self._timeout)

    @retry(max_attempts=_MAX_RETRIES, base_delay=_RETRY_BASE_DELAY)
    def _call_api(self, system_prompt: str, user_text: str, **overrides: Any) -> str:
        from anthropic import Anthropic

        assert self._client is not None
        client: Anthropic = self._client
        temperature = float(overrides.get("temperature", self._temperature))
        max_tokens = int(overrides.get("max_tokens", self._max_tokens))

        response = client.messages.create(
            model=self._model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_text}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.content[0].text
        return content.strip() if content else ""

    def correct(self, text: str, **kwargs: Any) -> str:
        """Run grammatical error correction via Anthropic Messages API.

        Args:
            text: The user message text (erroneous sentence).
            **kwargs: system_prompt, temperature, max_tokens overrides.

        Returns:
            The corrected text.

        Raises:
            BackendError: If the API call fails after all retries.
        """
        cached = self._cache.get(self.model_id, text)
        if cached is not None:
            logger.debug("Cache hit for '%s'", self.model_id)
            return cached

        self._ensure_client()

        system_prompt = str(kwargs.get("system_prompt", "Correct the grammar errors."))

        try:
            correction = self._call_api(system_prompt, text, **kwargs)
        except Exception as exc:
            raise BackendError(f"Anthropic request failed for '{self._model_name}': {exc}") from exc

        self._cache.set(self.model_id, text, correction)
        return correction

    @property
    def model_id(self) -> str:
        return f"anthropic:{self._model_name}"

    @property
    def metadata(self) -> dict:
        return {"provider": "anthropic", "model": self._model_name}
