"""OpenRouter API backend (OpenAI-compatible endpoint)."""

from __future__ import annotations

import logging
import os
from typing import Any

from llm_grammar_bench.backends.base import BackendError, BaseBackend
from llm_grammar_bench.utils.cache import CacheStore, make_request_cache_text
from llm_grammar_bench.utils.retry import retry

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.0
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(BaseBackend):
    """Backend for OpenRouter models using the OpenAI-compatible endpoint.

    Requires: pip install llmgrammarbench[openrouter]
    Sets OPENROUTER_API_KEY from environment.
    """

    def __init__(
        self,
        model: str,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: float = _DEFAULT_TIMEOUT,
        base_url: str = _OPENROUTER_BASE_URL,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._base_url = base_url
        self._cache = CacheStore(cache_dir)

        self._api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self._api_key:
            logger.warning("OPENROUTER_API_KEY not set. OpenRouter backend will fail at runtime.")

        self._client: Any | None = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenRouter backend requires the openai package. "
                "Install with: pip install llmgrammarbench[openrouter]"
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

        response = client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def correct(self, text: str, **kwargs: Any) -> str:
        """Run grammatical error correction via OpenRouter.

        Args:
            text: The user message text (erroneous sentence).
            **kwargs: system_prompt, temperature, max_tokens overrides.

        Returns:
            The corrected text.

        Raises:
            BackendError: If the API call fails after all retries.
        """
        cache_text = make_request_cache_text(text, **kwargs)

        def _compute() -> str:
            self._ensure_client()
            system_prompt = str(kwargs.pop("system_prompt", "Correct the grammar errors."))
            try:
                return self._call_api(system_prompt, text, **kwargs)
            except Exception as exc:
                raise BackendError(
                    f"OpenRouter request failed for '{self._model_name}': {exc}"
                ) from exc

        return self._cache.get_or_compute(self.model_id, cache_text, _compute)

    @property
    def model_id(self) -> str:
        return f"openrouter:{self._model_name}"

    @property
    def metadata(self) -> dict:
        return {"provider": "openrouter", "model": self._model_name}
