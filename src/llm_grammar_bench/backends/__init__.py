"""Model backends for inference (local and API-based)."""

from __future__ import annotations

from typing import Any

from llm_grammar_bench.backends.base import BackendError, BaseBackend
from llm_grammar_bench.config import BenchmarkConfig, ModelEntry, ProviderConfig


def load_backend(
    backend_type: str,
    model: str,
    provider_cfg: ProviderConfig | None = None,
    model_entry: ModelEntry | None = None,
    **kwargs: Any,
) -> BaseBackend:
    """Factory to load a backend by type.

    Args:
        backend_type: One of "openai", "anthropic", "huggingface",
                      "openrouter", or "openai_compatible".
        model: Model identifier string (e.g. "gpt-4o", "claude-3-opus-20240229").
        provider_cfg: Optional ProviderConfig with api_key, base_url, etc.
        model_entry: Optional ModelEntry with temperature, max_tokens, reasoning.
        **kwargs: Additional backend-specific overrides.

    Returns:
        A configured BaseBackend instance.

    Raises:
        ValueError: If the backend type is unknown.
        ImportError: If optional dependencies are not installed.
    """
    match backend_type:
        case "openai":
            from llm_grammar_bench.backends.openai import OpenAIBackend

            return OpenAIBackend(model=model, **kwargs)
        case "anthropic":
            from llm_grammar_bench.backends.anthropic import AnthropicBackend

            return AnthropicBackend(model=model, **kwargs)
        case "huggingface":
            from llm_grammar_bench.backends.huggingface import HuggingFaceBackend

            return HuggingFaceBackend(model=model, **kwargs)
        case "openrouter":
            from llm_grammar_bench.backends.openrouter import OpenRouterBackend

            return OpenRouterBackend(model=model, **kwargs)
        case "openai_compatible":
            from llm_grammar_bench.backends.generic_openai import (
                GenericOpenAICompatibleBackend,
            )

            if provider_cfg is None:
                raise ValueError("openai_compatible backend requires a provider configuration")
            return GenericOpenAICompatibleBackend(
                model=model,
                api_key=provider_cfg.api_key,
                base_url=provider_cfg.base_url or "",
                temperature=model_entry.temperature if model_entry else 0.0,
                max_tokens=model_entry.max_tokens if model_entry else 512,
                reasoning=model_entry.reasoning if model_entry else False,
                **kwargs,
            )

        case "anthropic_compatible":
            from llm_grammar_bench.backends.anthropic_compatible import (
                AnthropicCompatibleBackend,
            )

            if provider_cfg is None:
                raise ValueError("anthropic_compatible backend requires a provider configuration")
            return AnthropicCompatibleBackend(
                model=model,
                api_key=provider_cfg.api_key,
                base_url=provider_cfg.base_url or "",
                temperature=model_entry.temperature if model_entry else 0.0,
                max_tokens=model_entry.max_tokens if model_entry else 512,
                reasoning=model_entry.reasoning if model_entry else False,
                **kwargs,
            )
        case _:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Valid options: openai, anthropic, huggingface, openrouter, "
                f"openai_compatible, anthropic_compatible"
            )


def load_backend_from_config(
    config: BenchmarkConfig,
    model_ref: str,
    **kwargs: Any,
) -> BaseBackend:
    """Load a backend using a config file for provider and model resolution.

    Args:
        config: The loaded BenchmarkConfig.
        model_ref: Model key from config, or shorthand spec.
        **kwargs: Additional overrides passed to load_backend.

    Returns:
        A configured BaseBackend instance.
    """
    from llm_grammar_bench.config import resolve_model

    provider_cfg, model_entry = resolve_model(config, model_ref)

    # Build kwargs for the backend — only pass params the backend accepts.
    # HuggingFace backend uses max_new_tokens, not max_tokens/temperature.
    if provider_cfg.provider_type == "huggingface":
        backend_kwargs: dict[str, Any] = {}
    else:
        backend_kwargs = {
            "temperature": model_entry.temperature,
            "max_tokens": model_entry.max_tokens,
        }
    backend_kwargs.update(kwargs)

    return load_backend(
        backend_type=provider_cfg.provider_type,
        model=model_entry.model,
        provider_cfg=provider_cfg,
        model_entry=model_entry,
        **backend_kwargs,
    )


__all__ = ["BackendError", "BaseBackend", "load_backend", "load_backend_from_config"]
