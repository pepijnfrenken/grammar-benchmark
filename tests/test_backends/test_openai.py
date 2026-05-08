"""Tests for the OpenAI backend."""


def test_openai_backend_creation() -> None:
    import os

    from llm_grammar_bench.backends.openai import OpenAIBackend

    os.environ["OPENAI_API_KEY"] = "sk-test-dummy"
    backend = OpenAIBackend(model="gpt-4o")
    assert backend.model_id == "openai:gpt-4o"
    assert backend.metadata["provider"] == "openai"


def test_openai_backend_metadata() -> None:
    import os

    from llm_grammar_bench.backends.openai import OpenAIBackend

    os.environ["OPENAI_API_KEY"] = "sk-test-dummy"
    backend = OpenAIBackend(model="gpt-3.5-turbo")
    assert backend.metadata["model"] == "gpt-3.5-turbo"
