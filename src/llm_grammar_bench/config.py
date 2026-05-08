"""Configuration loading from YAML files with environment variable interpolation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    backend: str
    model: str
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
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    datasets: dict[str, DatasetConfig] = Field(default_factory=dict)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def _interpolate_env(value: str) -> str:
    """Replace `${VAR}` patterns with environment variable values."""
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
        A validated BenchmarkConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is invalid or required env vars are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    raw = _walk_interpolate(raw)

    return BenchmarkConfig.model_validate(raw)
