"""Tests for the HuggingFace backend."""


def test_huggingface_backend_creation() -> None:
    from llm_grammar_bench.backends.huggingface import HuggingFaceBackend

    backend = HuggingFaceBackend(model="t5-small", device="cpu")
    assert backend.model_id == "hf:t5-small"
    assert backend.metadata["provider"] == "huggingface"


def test_huggingface_backend_metadata() -> None:
    from llm_grammar_bench.backends.huggingface import HuggingFaceBackend

    backend = HuggingFaceBackend(model="google/flan-t5-base")
    assert backend.metadata["model"] == "google/flan-t5-base"
    assert "device" in backend.metadata


def test_huggingface_backend_uses_download_token() -> None:
    from llm_grammar_bench.backends.huggingface import HuggingFaceBackend

    backend = HuggingFaceBackend(model="google/flan-t5-small", api_key="hf_test_token")

    assert backend._from_pretrained_kwargs() == {"token": "hf_test_token"}


def test_huggingface_backend_omits_empty_download_token() -> None:
    from llm_grammar_bench.backends.huggingface import HuggingFaceBackend

    backend = HuggingFaceBackend(model="google/flan-t5-small")

    assert backend._from_pretrained_kwargs() == {}
