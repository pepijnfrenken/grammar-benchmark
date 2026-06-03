"""Tests for configuration module."""

import os
import tempfile
from pathlib import Path

import pytest


def test_provider_config_defaults() -> None:
    """Test ProviderConfig with minimal fields."""
    from llm_grammar_bench.config import ProviderConfig

    config = ProviderConfig(provider_type="openai")
    assert config.provider_type == "openai"
    assert config.api_key == ""
    assert config.base_url is None


def test_provider_config_full() -> None:
    """Test ProviderConfig with all fields."""
    from llm_grammar_bench.config import ProviderConfig

    config = ProviderConfig(
        provider_type="openai_compatible",
        api_key="sk-test",
        base_url="https://api.example.com/v1",
    )
    assert config.provider_type == "openai_compatible"
    assert config.api_key == "sk-test"
    assert config.base_url == "https://api.example.com/v1"


def test_model_entry_defaults() -> None:
    """Test ModelEntry with minimal fields, verify defaults."""
    from llm_grammar_bench.config import ModelEntry

    entry = ModelEntry(provider="openai", model="gpt-4")
    assert entry.provider == "openai"
    assert entry.model == "gpt-4"
    assert entry.temperature == 0.0
    assert entry.max_tokens == 512
    assert entry.reasoning is False
    assert entry.nickname is None


def test_model_entry_full() -> None:
    """Test ModelEntry with all fields."""
    from llm_grammar_bench.config import ModelEntry

    entry = ModelEntry(
        provider="openai",
        model="gpt-4o",
        nickname="GPT-4o",
        reasoning=True,
        temperature=0.3,
        max_tokens=1024,
    )
    assert entry.nickname == "GPT-4o"
    assert entry.reasoning is True
    assert entry.temperature == 0.3
    assert entry.max_tokens == 1024


def test_evaluation_config_defaults() -> None:
    """Test EvaluationConfig, verify default metrics list."""
    from llm_grammar_bench.config import EvaluationConfig

    config = EvaluationConfig()
    assert config.metrics == ["errant", "gleu", "bertscore"]
    assert config.beta == 0.5
    assert config.output_dir == "results/"
    assert config.max_workers is None


def test_sampling_config_defaults() -> None:
    """Test SamplingConfig defaults disable sampling."""
    from llm_grammar_bench.config import SamplingConfig

    config = SamplingConfig()
    assert config.sample_size is None
    assert config.stratify_by == "cefr"
    assert config.seed == 0


def test_benchmark_config_empty() -> None:
    """Test BenchmarkConfig defaults."""
    from llm_grammar_bench.config import BenchmarkConfig

    config = BenchmarkConfig()
    assert config.models == {}
    assert config.datasets == {}
    assert config.providers == {}
    assert isinstance(config.evaluation, object)


def test_env_interpolation() -> None:
    """Test _interpolate_env replaces ${VAR} with env var value."""
    from llm_grammar_bench.config import _interpolate_env

    os.environ["TEST_VAR"] = "test_value"
    result = _interpolate_env("prefix_${TEST_VAR}_suffix")
    assert result == "prefix_test_value_suffix"


def test_env_interpolation_no_match() -> None:
    """Test _interpolate_env leaves non-matching patterns unchanged."""
    from llm_grammar_bench.config import _interpolate_env

    result = _interpolate_env("no interpolation here")
    assert result == "no interpolation here"


def test_env_interpolation_missing_var() -> None:
    """Test _interpolate_env raises ValueError for missing env var."""
    from llm_grammar_bench.config import _interpolate_env

    if "NONEXISTENT_VAR_12345" in os.environ:
        del os.environ["NONEXISTENT_VAR_12345"]

    with pytest.raises(ValueError, match="Environment variable"):
        _interpolate_env("${NONEXISTENT_VAR_12345}")


def test_env_interpolation_default_for_missing_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ${VAR:-default} interpolation uses a default for missing variables."""
    from llm_grammar_bench.config import _interpolate_env

    monkeypatch.delenv("OPTIONAL_VAR_12345", raising=False)

    result = _interpolate_env("prefix_${OPTIONAL_VAR_12345:-fallback}_suffix")

    assert result == "prefix_fallback_suffix"


def test_load_env_file_sets_missing_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test dotenv loading supports comments, export, and quoted values."""
    from llm_grammar_bench.config import load_env_file

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        """
# local secrets
HF_TOKEN="hf_test_token"
export OPENCODE_API_KEY='opencode_test_key'
"""
    )

    load_env_file(env_path)

    assert os.environ["HF_TOKEN"] == "hf_test_token"
    assert os.environ["OPENCODE_API_KEY"] == "opencode_test_key"


def test_load_env_file_preserves_existing_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test real environment variables take precedence over dotenv values."""
    from llm_grammar_bench.config import load_env_file

    monkeypatch.setenv("HF_TOKEN", "existing_token")
    env_path = tmp_path / ".env"
    env_path.write_text("HF_TOKEN=dotenv_token\n")

    load_env_file(env_path)

    assert os.environ["HF_TOKEN"] == "existing_token"


def test_load_config_reads_env_file_next_to_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test config interpolation can use a colocated .env file."""
    from llm_grammar_bench.config import load_config

    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_test_token\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
providers:
  huggingface:
    provider_type: huggingface
    api_key: "${HF_TOKEN}"
"""
    )

    config = load_config(config_path)

    assert config.providers["huggingface"].api_key == "hf_test_token"


def test_default_config_loads_without_optional_opencode_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test bundled config does not require optional OpenCode credentials."""
    from llm_grammar_bench.config import load_config

    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)

    config = load_config("configs/default.yaml")

    assert config.providers["opencode-go"].api_key == ""
    assert config.models["qwen37-plus"].model == "qwen3.7-plus"


def test_walk_interpolate_nested() -> None:
    """Test _walk_interpolate recursively interpolates nested dicts/lists."""
    from llm_grammar_bench.config import _walk_interpolate

    os.environ["KEY1"] = "value1"
    os.environ["KEY2"] = "value2"

    obj = {
        "string": "prefix_${KEY1}",
        "nested": {"inner": "prefix_${KEY2}"},
        "list": ["item_${KEY1}", "item_${KEY2}"],
    }
    result = _walk_interpolate(obj)
    assert result["string"] == "prefix_value1"
    assert result["nested"]["inner"] == "prefix_value2"
    assert result["list"] == ["item_value1", "item_value2"]


def test_load_config_basic() -> None:
    """Test loading a basic YAML config file."""
    from llm_grammar_bench.config import load_config

    yaml_content = """
models:
  gpt4:
    provider: openai
    model: gpt-4
    temperature: 0.3
    max_tokens: 1024
datasets:
  bea:
    type: bea2019
    split: test
evaluation:
  metrics: ["errant", "gleu"]
  beta: 0.7
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        try:
            config = load_config(f.name)
            assert "gpt4" in config.models
            assert config.models["gpt4"].provider == "openai"
            assert config.models["gpt4"].temperature == 0.3
            assert "bea" in config.datasets
            assert config.evaluation.beta == 0.7
        finally:
            Path(f.name).unlink()


def test_load_config_file_not_found() -> None:
    """Test load_config raises FileNotFoundError for missing file."""
    from llm_grammar_bench.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_load_config_with_env_interpolation() -> None:
    """Test load_config interpolates environment variables."""
    from llm_grammar_bench.config import load_config

    os.environ["OPENAI_KEY"] = "sk-test-key-123"

    yaml_content = """
models:
  gpt4:
    provider: openai
    model: gpt-4
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        try:
            config = load_config(f.name)
            assert config is not None
        finally:
            Path(f.name).unlink()


def test_resolve_model_by_key() -> None:
    """Test resolve_model finds a model by its config key."""
    from llm_grammar_bench.config import BenchmarkConfig, ModelEntry, resolve_model

    config = BenchmarkConfig(
        models={"my-gpt": ModelEntry(provider="openai", model="gpt-4o", nickname="My GPT")}
    )
    provider, entry = resolve_model(config, "my-gpt")
    assert entry.model == "gpt-4o"
    assert entry.nickname == "My GPT"
    assert provider.provider_type == "openai"


def test_resolve_model_by_shorthand() -> None:
    """Test resolve_model parses 'openai:gpt-4o' shorthand."""
    from llm_grammar_bench.config import BenchmarkConfig, resolve_model

    config = BenchmarkConfig()
    provider, entry = resolve_model(config, "openai:gpt-4o")
    assert entry.model == "gpt-4o"
    assert provider.provider_type == "openai"


def test_resolve_model_unknown_provider() -> None:
    """Test resolve_model raises on unknown provider reference."""
    from llm_grammar_bench.config import BenchmarkConfig, ModelEntry, resolve_model

    config = BenchmarkConfig(
        models={
            "bad-model": ModelEntry(provider="nonexistent", model="test"),
        }
    )
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_model(config, "bad-model")


def test_benchmark_config_merges_default_providers() -> None:
    """Test load_config always has the four built-in providers."""
    from llm_grammar_bench.config import load_config

    yaml_content = """
models:
  gpt4:
    provider: openai
    model: gpt-4
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        try:
            config = load_config(f.name)
            assert "openai" in config.providers
            assert "anthropic" in config.providers
            assert "openrouter" in config.providers
            assert "huggingface" in config.providers
        finally:
            Path(f.name).unlink()
