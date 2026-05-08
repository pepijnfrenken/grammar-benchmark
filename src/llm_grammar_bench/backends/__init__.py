"""Model backends for inference (local and API-based)."""

from llm_grammar_bench.backends.base import BackendError, BaseBackend


def load_backend(backend_type: str, model: str, **kwargs) -> BaseBackend:
    """Factory to load a backend by type.

    Args:
        backend_type: One of "openai", "anthropic", "huggingface", "openrouter".
        model: Model identifier string (e.g. "gpt-4o", "claude-3-opus-20240229").
        **kwargs: Backend-specific configuration.

    Returns:
        A configured BaseBackend instance.

    Raises:
        ValueError: If the backend type is unknown.
        ImportError: If optional dependencies for the backend are not installed.
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
        case _:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Valid options: openai, anthropic, huggingface, openrouter"
            )


__all__ = ["BackendError", "BaseBackend", "load_backend"]
