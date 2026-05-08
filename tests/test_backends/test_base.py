"""Tests for the BaseBackend ABC."""


def test_base_backend_cannot_be_instantiated() -> None:
    import contextlib

    from llm_grammar_bench.backends.base import BaseBackend

    with contextlib.suppress(TypeError):
        BaseBackend()  # type: ignore[abstract]
