"""Tests for the Anthropic backend."""


def test_anthropic_backend_creation() -> None:
    import os

    from llm_grammar_bench.backends.anthropic import AnthropicBackend

    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-dummy"
    backend = AnthropicBackend(model="claude-3-opus-20240229")
    assert backend.model_id == "anthropic:claude-3-opus-20240229"
    assert backend.metadata["provider"] == "anthropic"
