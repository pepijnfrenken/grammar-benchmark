"""Generic backend for any Anthropic-compatible Messages API endpoint."""

from __future__ import annotations

import logging
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


class AnthropicCompatibleBackend(BaseBackend):
    """Backend for any Anthropic-compatible Messages API endpoint.

    Works with self-hosted Claude proxies, AWS Bedrock, GCP Vertex,
    or any service exposing an Anthropic-compatible /v1/messages endpoint.

    Requires: pip install llmgrammarbench[anthropic]
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: float = _DEFAULT_TIMEOUT,
        reasoning: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._reasoning = reasoning
        self._cache = CacheStore(cache_dir)
        self._client: Any | None = None

        if not api_key and "localhost" not in base_url:
            logger.warning("No API key provided for '%s'. Backend may fail at runtime.", base_url)

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic-compatible backend requires the anthropic package. "
                "Install with: pip install llmgrammarbench[anthropic]"
            ) from None
        self._client = Anthropic(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    @retry(max_attempts=_MAX_RETRIES, base_delay=_RETRY_BASE_DELAY)
    def _call_api(self, system_prompt: str, user_text: str, **overrides: Any) -> str:
        from anthropic import Anthropic

        assert self._client is not None
        client: Anthropic = self._client
        temperature = float(overrides.get("temperature", self._temperature))
        max_tokens = int(overrides.get("max_tokens", self._max_tokens))

        extra_kwargs: dict[str, Any] = {}
        if self._reasoning:
            extra_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1024}

        response = client.messages.create(
            model=self._model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_text}],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_kwargs,
        )

        # Extract text content, skipping any thinking blocks
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                return block.text.strip()
        return ""

    def correct(self, text: str, **kwargs: Any) -> str:
        """Run grammatical error correction via the Anthropic-compatible endpoint.

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
            raise BackendError(
                f"API request failed for '{self._model_name}' at '{self._base_url}': {exc}"
            ) from exc

        self._cache.set(self.model_id, text, correction)
        return correction

    @property
    def model_id(self) -> str:
        return f"anthropic_compatible:{self._model_name}"

    @property
    def metadata(self) -> dict:
        return {
            "provider": "anthropic_compatible",
            "model": self._model_name,
            "base_url": self._base_url,
            "reasoning": self._reasoning,
        }
