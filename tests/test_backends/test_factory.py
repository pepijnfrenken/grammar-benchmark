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


def test_load_backend_from_config_passes_huggingface_api_key() -> None:
    """Test HuggingFace provider api_key is passed as a Hub download token."""
    from llm_grammar_bench.backends import load_backend_from_config
    from llm_grammar_bench.backends.huggingface import HuggingFaceBackend
    from llm_grammar_bench.config import BenchmarkConfig, ModelEntry, ProviderConfig

    config = BenchmarkConfig(
        providers={
            "huggingface": ProviderConfig(
                provider_type="huggingface",
                api_key="hf_test_token",
            )
        },
        models={
            "local-model": ModelEntry(
                provider="huggingface",
                model="google/flan-t5-small",
            )
        },
    )

    backend = load_backend_from_config(config, "local-model")

    assert isinstance(backend, HuggingFaceBackend)
    assert backend._from_pretrained_kwargs() == {"token": "hf_test_token"}


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
