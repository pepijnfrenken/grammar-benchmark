"""Configuration loading from YAML files with environment variable interpolation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Definition of an API provider.

    Supports built-in types (openai, anthropic, huggingface) and
    OpenAI-compatible custom endpoints via ``openai_compatible``.
    """

    provider_type: str
    api_key: str = ""
    base_url: str | None = None


class ModelEntry(BaseModel):
    """A named model entry that references a provider."""

    provider: str
    model: str
    nickname: str | None = None
    reasoning: bool = False
    temperature: float = 0.0
    max_tokens: int = 512


class DatasetConfig(BaseModel):
    type: str
    split: str = "validation"
    cefr_filter: list[str] | None = None


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["errant", "gleu", "bertscore"])
    beta: float = 0.5
    output_dir: str = "results/"


class BenchmarkConfig(BaseModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelEntry] = Field(default_factory=dict)
    datasets: dict[str, DatasetConfig] = Field(default_factory=dict)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# Pre-built default providers so the config works without a providers section.
_DEFAULT_PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(provider_type="openai", api_key="${OPENAI_API_KEY}"),
    "anthropic": ProviderConfig(provider_type="anthropic", api_key="${ANTHROPIC_API_KEY}"),
    "openrouter": ProviderConfig(
        provider_type="openai_compatible",
        api_key="${OPENROUTER_API_KEY}",
        base_url="https://openrouter.ai/api/v1",
    ),
    "huggingface": ProviderConfig(provider_type="huggingface"),
}


def _interpolate_env(value: str) -> str:
    """Replace ``${VAR}`` patterns with environment variable values."""
    import re

    def _replace(match: re.Match[str]) -> str:
        var = match.group(1)
        result = os.environ.get(var, "")
        if not result:
            raise ValueError(f"Environment variable ${var} is not set")
        return result

    return re.sub(r"\$\{(\w+)\}", _replace, value)


def _walk_interpolate(obj: Any) -> Any:
    """Recursively apply env interpolation to all string values in a dict/list."""
    if isinstance(obj, str):
        return _interpolate_env(obj)
    if isinstance(obj, dict):
        return {k: _walk_interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_interpolate(v) for v in obj]
    return obj


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a benchmark configuration from a YAML file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A validated BenchmarkConfig instance with default providers merged in.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is invalid or required env vars are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    raw = _walk_interpolate(raw)

    # Coerce None providers (from empty YAML section) to empty dict
    if isinstance(raw, dict) and raw.get("providers") is None:
        raw["providers"] = {}

    config = BenchmarkConfig.model_validate(raw)

    # Merge default providers for any not explicitly configured
    for name, provider in _DEFAULT_PROVIDERS.items():
        if name not in config.providers:
            config.providers[name] = provider

    # Validate that every model references a known provider
    for model_name, entry in config.models.items():
        if entry.provider not in config.providers:
            raise ValueError(
                f"Model '{model_name}' references unknown provider "
                f"'{entry.provider}'. Known providers: {sorted(config.providers)}"
            )

    return config


def resolve_model(config: BenchmarkConfig, model_ref: str) -> tuple[ProviderConfig, ModelEntry]:
    """Resolve a model reference to its provider and model entry.

    Args:
        config: The loaded benchmark configuration.
        model_ref: Either a model nickname/key from config, or a shorthand
                   spec like ``openai:gpt-4o``, ``hf:google/flan-t5-small``.

    Returns:
        A tuple of (ProviderConfig, ModelEntry).

    Raises:
        ValueError: If the model reference cannot be resolved.
    """
    # Ensure default providers are available
    for name, provider in _DEFAULT_PROVIDERS.items():
        if name not in config.providers:
            config.providers[name] = provider

    # Try exact match in configured models
    if model_ref in config.models:
        entry = config.models[model_ref]
        if entry.provider not in config.providers:
            raise ValueError(
                f"Model '{model_ref}' references unknown provider "
                f"'{entry.provider}'. Known providers: {sorted(config.providers)}"
            )
        provider = config.providers[entry.provider]
        return provider, entry

    # Try shorthand spec: "openai:gpt-4o" or "hf:google/flan-t5-small"
    mapping = {
        "hf": "huggingface",
        "openai": "openai",
        "anthropic": "anthropic",
        "openrouter": "openrouter",
        "opencode-go": "opencode-go",
        "huggingface": "huggingface",
    }
    if ":" in model_ref:
        prefix, rest = model_ref.split(":", 1)
        provider_name = mapping.get(prefix.lower(), prefix.lower())
        provider = config.providers.get(provider_name)
        if provider is None:
            raise ValueError(
                f"Unknown provider '{provider_name}' from model spec '{model_ref}'. "
                f"Known providers: {sorted(config.providers)}"
            )
        # Build a synthetic ModelEntry
        entry = ModelEntry(provider=provider_name, model=rest)
        return provider, entry

    # Default: assume huggingface if no match
    provider = config.providers.get("huggingface")
    if provider is None:
        raise ValueError("No huggingface provider configured")
    entry = ModelEntry(provider="huggingface", model=model_ref)
    return provider, entry
