"""Tests for the backend factory function."""

import os

import pytest


def test_load_backend_unknown() -> None:
    """Test that load_backend raises ValueError for unknown backend type."""
    from llm_grammar_bench.backends import load_backend

    try:
        load_backend("unknown_backend", "test-model")
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        assert "Unknown backend type" in str(e)
        assert "unknown_backend" in str(e)


def test_load_backend_huggingface() -> None:
    """Test that load_backend returns HuggingFaceBackend for 'huggingface' type."""
    from llm_grammar_bench.backends import load_backend
    from llm_grammar_bench.backends.huggingface import HuggingFaceBackend

    backend = load_backend("huggingface", "test-model")
    assert isinstance(backend, HuggingFaceBackend)


def test_load_backend_openai_with_env() -> None:
    """Test that load_backend returns OpenAIBackend when OPENAI_API_KEY is set."""
    from llm_grammar_bench.backends import load_backend
    from llm_grammar_bench.backends.openai import OpenAIBackend

    # Set the API key environment variable
    os.environ["OPENAI_API_KEY"] = "test-key-12345"
    try:
        backend = load_backend("openai", "gpt-4o")
        assert isinstance(backend, OpenAIBackend)
    finally:
        # Clean up
        os.environ.pop("OPENAI_API_KEY", None)
