"""Generic backend for any OpenAI-compatible API endpoint."""

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


class GenericOpenAICompatibleBackend(BaseBackend):
    """Backend for any OpenAI-compatible chat completion endpoint.

    Works with OpenRouter, Groq, Together, local vLLM, Ollama, etc.
    Configure via the config file's providers section.

    Requires: pip install llmgrammarbench[openai]
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

        if not api_key and "openai.com" not in base_url and "localhost" not in base_url:
            logger.warning("No API key provided for '%s'. Backend may fail at runtime.", base_url)

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI-compatible backend requires the openai package. "
                "Install with: pip install llmgrammarbench[openai]"
            ) from None
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    @retry(max_attempts=_MAX_RETRIES, base_delay=_RETRY_BASE_DELAY)
    def _call_api(self, system_prompt: str, user_text: str, **overrides: Any) -> str:
        from openai import OpenAI

        assert self._client is not None
        client: OpenAI = self._client
        temperature = float(overrides.get("temperature", self._temperature))
        max_tokens = int(overrides.get("max_tokens", self._max_tokens))

        extra_body: dict[str, Any] = {}
        if self._reasoning:
            extra_body["reasoning"] = {"effort": "medium"}

        response = client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body if extra_body else None,
        )
        content = response.choices[0].message.content
        if not content:
            logger.warning(
                "Empty response from '%s': finish_reason=%s, model=%s, reasoning=%s",
                self._base_url,
                response.choices[0].finish_reason,
                response.model,
                self._reasoning,
            )
        return content.strip() if content else ""

    def correct(self, text: str, **kwargs: Any) -> str:
        """Run grammatical error correction via the OpenAI-compatible endpoint.

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

        system_prompt = str(kwargs.pop("system_prompt", "Correct the grammar errors."))

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
        return f"openai_compatible:{self._model_name}"

    @property
    def metadata(self) -> dict:
        return {
            "provider": "openai_compatible",
            "model": self._model_name,
            "base_url": self._base_url,
            "reasoning": self._reasoning,
        }
