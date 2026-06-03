"""Click CLI for the LLM Grammar Bench framework."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_grammar_bench.backends.base import BaseBackend
    from llm_grammar_bench.config import BenchmarkConfig
import click

logger = logging.getLogger(__name__)
_DEFAULT_API_MAX_WORKERS = 8


@click.group()
@click.version_option(version="0.1.0", prog_name="llm-grammar-bench")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to YAML config file with providers, models, datasets, and evaluation.",
)
@click.pass_context
def main(ctx: click.Context, config_path: Path | None) -> None:
    """LLM Grammar Bench — evaluate AI models on grammatical error correction."""
    ctx.ensure_object(dict)
    if config_path is not None:
        from llm_grammar_bench.config import load_config

        ctx.obj["config"] = load_config(config_path)
    else:
        ctx.obj["config"] = None


def _load_backend_for_model(
    config: object | None,
    model_ref: str,
) -> BaseBackend:
    """Resolve and load a backend for a model reference."""
    from llm_grammar_bench.config import BenchmarkConfig

    if config is not None and isinstance(config, BenchmarkConfig) and config.models:
        from llm_grammar_bench.backends import load_backend_from_config

        return load_backend_from_config(config, model_ref)
    else:
        from llm_grammar_bench.backends import load_backend

        backend_type, model_name = _parse_model_spec(model_ref)
        return load_backend(backend_type, model_name)


def _run_single_benchmark(
    config: object | None,
    model_ref: str,
    dataset: str,
    split: str,
    strategy: str,
    metrics: str | list[str],
    output: str,
    max_sentences: int | None,
    beta: float,
    max_workers: int | None = None,
    rate_limit: float | None = None,
    sample_size: int | None = None,
    stratify_by: str = "cefr",
    sample_seed: int = 0,
    sample_apis_only: bool = False,
) -> None:
    backend = _load_backend_for_model(config, model_ref)
    click.echo(f"Backend: {backend.model_id}")  # type: ignore[union-attr]

    # Load strategy
    from llm_grammar_bench.strategies import load_strategy

    strategy_instance = load_strategy(strategy)
    click.echo(f"Strategy: {type(strategy_instance).__name__}")

    # Load dataset
    from llm_grammar_bench.datasets import load_dataset as load_ds

    dataset_instance = load_ds(dataset)
    click.echo(f"Dataset: {type(dataset_instance).__name__}")

    # Run benchmark
    from llm_grammar_bench.evaluation.evaluator import Evaluator
    from llm_grammar_bench.evaluation.results import serialize_results

    effective_sample_size = sample_size
    if sample_apis_only and backend.metadata.get("provider") == "huggingface":
        effective_sample_size = None
    effective_max_workers = max_workers
    if backend.metadata.get("provider") == "huggingface":
        effective_max_workers = 1
    elif effective_max_workers is None:
        effective_max_workers = _DEFAULT_API_MAX_WORKERS

    evaluator = Evaluator(
        backend=backend,  # type: ignore[arg-type]
        dataset=dataset_instance,  # type: ignore[arg-type]
        strategy=strategy_instance,  # type: ignore[arg-type]
        split=split,
        max_sentences=max_sentences,
        beta=beta,
        max_workers=effective_max_workers,
        rate_limit=rate_limit,
        sample_size=effective_sample_size,
        stratify_by=stratify_by,
        sample_seed=sample_seed,
    )
    metric_list = [m.strip() for m in metrics.split(",")] if isinstance(metrics, str) else metrics

    result = evaluator.run(metrics=metric_list)

    # Print summary
    click.echo()
    click.echo(f"Model: {result.model_id}")
    click.echo(f"Dataset: {result.dataset_name} ({split})")
    click.echo(f"Sentences: {len(result.per_sentence)}")
    click.echo(f"Runtime: {result.runtime_seconds:.1f}s")
    click.echo()
    click.echo("Corpus-level scores:")
    for metric_name, scores in result.corpus_scores.items():
        click.echo(f"  {metric_name}: F={scores.f_score:.4f}")

    if result.by_cefr:
        click.echo()
        click.echo("CEFR breakdown:")
        for cefr in sorted(result.by_cefr):
            cefr_scores = result.by_cefr[cefr]
            parts = []
            for metric_name in sorted(cefr_scores):
                parts.append(f"{metric_name}={cefr_scores[metric_name].f_score:.4f}")
            click.echo(f"  {cefr}: {', '.join(parts)}")

    # Serialize results
    filepath = serialize_results(result, output)
    click.echo(f"Full results saved to: {filepath}")
    click.echo()


@main.command()
@click.option(
    "--model",
    default=None,
    help="Single model: config key or shorthand spec (e.g. 'openai:gpt-4o').",
)
@click.option(
    "--models",
    default=None,
    help="Comma-separated model keys from config for batch benchmarking.",
)
@click.option("--dataset", default="bea2019", help="Dataset key from config or name.")
@click.option("--split", default="validation", help="Dataset split.")
@click.option("--strategy", default="seq2seq", help="Correction strategy.")
@click.option("--metrics", default="errant,gleu,bertscore", help="Comma-separated metrics.")
@click.option("--output", default="results/", help="Output directory or file path.")
@click.option("--max-sentences", type=int, default=None, help="Limit sentences for quick testing.")
@click.option("--beta", type=float, default=0.5, help="Beta value for ERRANT F-score.")
@click.option("--max-workers", type=int, default=None, help="Concurrent API requests.")
@click.option("--rate-limit", type=float, default=None, help="Max API calls per second.")
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Stratified sample size; overrides configured API sampling.",
)
@click.option("--stratify-by", default=None, help="Example metadata field for stratified sampling.")
@click.option("--sample-seed", type=int, default=None, help="Seed for deterministic sampling.")
@click.pass_context
def run(
    ctx: click.Context,
    model: str | None,
    models: str | None,
    dataset: str,
    split: str,
    strategy: str,
    metrics: str,
    output: str,
    max_sentences: int | None,
    beta: float,
    max_workers: int | None,
    rate_limit: float | None,
    sample_size: int | None,
    stratify_by: str | None,
    sample_seed: int | None,
) -> None:
    """Run a benchmark with one or more models.

    Single model (no config needed):
        llm-grammar-bench run --model hf:google/flan-t5-small

    Single model from config:
        llm-grammar-bench --config configs/default.yaml run --model t5-small

    Batch multiple models from config:
        llm-grammar-bench --config configs/default.yaml run --models t5-small,coedit-large
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if not model and not models:
        raise click.UsageError("Either --model or --models must be specified.")
    if model and models:
        raise click.UsageError("Cannot use both --model and --models together.")

    raw_config = ctx.obj.get("config") if ctx.obj else None
    from llm_grammar_bench.config import BenchmarkConfig

    config: BenchmarkConfig | None = raw_config if isinstance(raw_config, BenchmarkConfig) else None

    # Resolve runtime options from config if not specified via CLI.
    sample_apis_only = False
    if max_workers is None and config is not None:
        max_workers = config.evaluation.max_workers
    if rate_limit is None and config is not None:
        rate_limit = config.evaluation.rate_limit
    if config is not None:
        sampling_config = config.evaluation.api_sampling
        if sample_size is None:
            sample_size = sampling_config.sample_size
            sample_apis_only = sample_size is not None
        if stratify_by is None:
            stratify_by = sampling_config.stratify_by
        if sample_seed is None:
            sample_seed = sampling_config.seed
    if stratify_by is None:
        stratify_by = "cefr"
    if sample_seed is None:
        sample_seed = 0

    model_list: list[str]
    if models:
        if config is None:
            raise click.UsageError("--models requires --config with configured models.")
        model_list = [m.strip() for m in models.split(",")]
        if not model_list:
            raise click.UsageError("--models must contain at least one model key.")
    else:
        assert model is not None
        model_list = [model]

    for i, model_ref in enumerate(model_list):
        if len(model_list) > 1:
            click.secho(f"\n{'=' * 60}", fg="cyan")
            click.secho(f"  Model {i + 1}/{len(model_list)}: {model_ref}", fg="cyan")
            click.secho(f"{'=' * 60}\n", fg="cyan")

        _run_single_benchmark(
            config=config,
            model_ref=model_ref,
            dataset=dataset,
            split=split,
            strategy=strategy,
            metrics=metrics,
            output=output,
            max_sentences=max_sentences,
            beta=beta,
            max_workers=max_workers,
            rate_limit=rate_limit,
            sample_size=sample_size,
            stratify_by=stratify_by,
            sample_seed=sample_seed,
            sample_apis_only=sample_apis_only,
        )


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse a model spec string into backend type and model name."""
    mapping = {
        "hf": "huggingface",
        "openai": "openai",
        "anthropic": "anthropic",
        "openrouter": "openrouter",
        "huggingface": "huggingface",
    }
    if ":" in spec:
        prefix, rest = spec.split(":", 1)
        backend = mapping.get(prefix.lower(), prefix.lower())
        return backend, rest
    return "huggingface", spec


@main.command()
@click.pass_context
def list_models(ctx: click.Context) -> None:
    """List available model backends and configured models."""
    config = ctx.obj.get("config") if ctx.obj else None

    if config is not None and config.models:
        click.echo("Configured models:")
        for key, entry in config.models.items():
            nickname = f" ({entry.nickname})" if entry.nickname else ""
            reasoning = " [reasoning]" if entry.reasoning else ""
            click.echo(f"  {key}: {entry.provider}/{entry.model}{nickname}{reasoning}")
        click.echo()
        click.echo("Configured providers:")
        for name, provider in config.providers.items():
            base = f" ({provider.base_url})" if provider.base_url else ""
            click.echo(f"  {name}: {provider.provider_type}{base}")
    else:
        click.echo("Available backends:")
        click.echo("  openai                   — OpenAI (GPT-4o, GPT-4, etc.)")
        click.echo("  anthropic                — Anthropic (Claude 3, Claude 3.5, etc.)")
        click.echo("  huggingface              — Local models via transformers")
        click.echo("  openrouter               — OpenRouter multi-provider gateway")
        click.echo("  openai_compatible        — Any OpenAI-compatible endpoint")
        click.echo("  anthropic_compatible     — Any Anthropic-compatible endpoint")
        click.echo()
        click.echo("Use --config <file> to see configured models from a YAML file.")
        click.echo("Model spec format: '<backend>:<model>' (e.g. 'openai:gpt-4o')")


@main.command()
def list_datasets() -> None:
    """List available datasets."""
    click.echo("Available datasets: bea2019")


@main.command()
def clear_cache() -> None:
    """Clear the correction cache to force fresh API calls."""
    from llm_grammar_bench.utils.cache import CacheStore

    cache = CacheStore()
    cache.clear()
    click.echo("Correction cache cleared.")
