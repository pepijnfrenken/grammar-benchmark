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
        api_key: str = "",
    ) -> None:
        self._model_name = model
        self._max_input_tokens = max_input_tokens
        self._max_new_tokens = max_new_tokens
        self._cache = CacheStore(cache_dir)
        self._api_key = api_key

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_encoder_decoder: bool | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return

        logger.info("Loading model '%s' on %s...", self._model_name, self._device)

        # fmt: off
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore  # noqa: E501, I001
        # fmt: on
        except ImportError:
            raise ImportError(
                "HuggingFace backend requires transformers and torch. "
                "Install with: pip install llmgrammarbench[huggingface]"
            ) from None

        from_pretrained_kwargs = self._from_pretrained_kwargs()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **from_pretrained_kwargs
        )

        # Detect model architecture from config before loading weights
        config = AutoConfig.from_pretrained(self._model_name, **from_pretrained_kwargs)
        self._is_encoder_decoder = getattr(config, "is_encoder_decoder", True)

        if self._is_encoder_decoder:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self._model_name, **from_pretrained_kwargs
            ).to(self._device)
        else:
            causal_model: Any = AutoModelForCausalLM.from_pretrained(
                self._model_name, **from_pretrained_kwargs
            )
            self._model = causal_model.to(self._device)
            tok: Any = self._tokenizer
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

        self._model.eval()
        logger.info("Model '%s' loaded successfully.", self._model_name)

    def _from_pretrained_kwargs(self) -> dict[str, str]:
        """Build HuggingFace Hub download options."""
        if not self._api_key:
            return {}
        return {"token": self._api_key}

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

            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature if temperature > 0 else None,
                "do_sample": temperature > 0,
                "num_beams": 4,
                "early_stopping": True,
            }

            if not self._is_encoder_decoder:
                generate_kwargs["pad_token_id"] = self._tokenizer.pad_token_id

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **generate_kwargs)

            if self._is_encoder_decoder:
                correction = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            else:
                # Decoder-only: strip the input prompt from the output
                input_len = inputs.input_ids.shape[1]
                correction = self._tokenizer.decode(
                    outputs[0][input_len:], skip_special_tokens=True
                ).strip()

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

    def release(self) -> None:
        """Move model to CPU and free GPU memory."""
        if self._model is not None:
            self._model.to("cpu")
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        import gc

        gc.collect()
        torch.cuda.empty_cache()
