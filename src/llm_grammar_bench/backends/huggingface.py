"""HuggingFace local model backend."""

from __future__ import annotations

import logging
from typing import Any

import torch

from llm_grammar_bench.backends.base import BackendError, BaseBackend
from llm_grammar_bench.utils.cache import CacheStore

logger = logging.getLogger(__name__)

_MAX_INPUT_TOKENS = 512
_MAX_NEW_TOKENS = 128
_DEFAULT_CACHE_DIR = None  # Use CacheStore default


class HuggingFaceBackend(BaseBackend):
    """Backend for local HuggingFace transformer models.

    Supports encoder-decoder models (T5, BART) and decoder-only models (GPT-2, Llama).
    Lazy-loads the model on first use.

    Requires: pip install llmgrammarbench[huggingface]
    """

    def __init__(
        self,
        model: str,
        device: str | None = None,
        max_input_tokens: int = _MAX_INPUT_TOKENS,
        max_new_tokens: int = _MAX_NEW_TOKENS,
        cache_dir: str | None = _DEFAULT_CACHE_DIR,
    ) -> None:
        self._model_name = model
        self._max_input_tokens = max_input_tokens
        self._max_new_tokens = max_new_tokens
        self._cache = CacheStore(cache_dir)

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_encoder_decoder: bool | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return

        logger.info("Loading model '%s' on %s...", self._model_name, self._device)

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        except ImportError:
            raise ImportError(
                "HuggingFace backend requires transformers and torch. "
                "Install with: pip install llmgrammarbench[huggingface]"
            ) from None

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        self._is_encoder_decoder = getattr(self._model.config, "is_encoder_decoder", False)

        logger.info("Model '%s' loaded successfully.", self._model_name)

    def correct(self, text: str, **kwargs: Any) -> str:
        """Run grammatical error correction on the input text.

        Args:
            text: Pre-formatted input text (strategy handles task prefix).
            **kwargs: Overrides for max_new_tokens, temperature, etc.

        Returns:
            The corrected text.

        Raises:
            BackendError: If model inference fails.
        """
        cached = self._cache.get(self.model_id, text)
        if cached is not None:
            logger.debug("Cache hit for '%s'", self.model_id)
            return cached

        self._ensure_loaded()

        max_new_tokens = int(kwargs.get("max_new_tokens", self._max_new_tokens))
        temperature = float(kwargs.get("temperature", 0.0))

        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_input_tokens,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    num_beams=4,
                    early_stopping=True,
                )

            correction = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            self._cache.set(self.model_id, text, correction)
            return correction

        except Exception as exc:
            raise BackendError(
                f"HuggingFace inference failed for '{self._model_name}': {exc}"
            ) from exc

    @property
    def model_id(self) -> str:
        return f"hf:{self._model_name}"

    @property
    def metadata(self) -> dict:
        return {
            "provider": "huggingface",
            "model": self._model_name,
            "device": self._device,
        }
