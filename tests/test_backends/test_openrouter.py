"""Tests for the OpenRouter backend."""


def test_openrouter_backend_creation() -> None:
    import os

    from llm_grammar_bench.backends.openrouter import OpenRouterBackend

    os.environ["OPENROUTER_API_KEY"] = "sk-or-dummy"
    backend = OpenRouterBackend(model="openai/gpt-4o")
    assert backend.model_id == "openrouter:openai/gpt-4o"
    assert backend.metadata["provider"] == "openrouter"
