"""Tests for the OpenAI backend."""

import tempfile
from typing import Any


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


def test_openai_backend_caches_by_prompt_kwargs() -> None:
    """Test backend cache includes prompt kwargs and skips repeated API calls."""
    import os

    from llm_grammar_bench.backends.openai import OpenAIBackend

    os.environ["OPENAI_API_KEY"] = "sk-test-dummy"
    calls: list[tuple[str, str]] = []

    class FakeOpenAIBackend(OpenAIBackend):
        def _ensure_client(self) -> None:
            return

        def _call_api(self, system_prompt: str, user_text: str, **_kwargs: Any) -> str:
            calls.append((system_prompt, user_text))
            return f"{system_prompt}: {user_text}"

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FakeOpenAIBackend(model="gpt-4o", cache_dir=tmpdir)

        assert backend.correct("text", system_prompt="A") == "A: text"
        assert backend.correct("text", system_prompt="A") == "A: text"
        assert backend.correct("text", system_prompt="B") == "B: text"
        assert calls == [("A", "text"), ("B", "text")]
