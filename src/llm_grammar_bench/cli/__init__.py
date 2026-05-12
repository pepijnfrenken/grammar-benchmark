"""Click CLI for the LLM Grammar Bench framework."""

from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="llm-grammar-bench")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to YAML config file with providers, models, datasets, and evaluation settings.",
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


@main.command()
@click.option(
    "--model",
    required=True,
    help="Model nickname from config, or shorthand: 'hf:google/flan-t5-small', 'openai:gpt-4o'.",
)
@click.option("--dataset", default="bea2019", help="Dataset key from config or name.")
@click.option("--split", default="validation", help="Dataset split.")
@click.option("--strategy", default="seq2seq", help="Correction strategy.")
@click.option("--metrics", default="errant,gleu,bertscore", help="Comma-separated metrics.")
@click.option("--output", default="results/", help="Output directory or file path.")
@click.option("--max-sentences", type=int, default=None, help="Limit sentences for quick testing.")
@click.option("--beta", type=float, default=0.5, help="Beta value for ERRANT F-score.")
@click.pass_context
def run(
    ctx: click.Context,
    model: str,
    dataset: str,
    split: str,
    strategy: str,
    metrics: str,
    output: str,
    max_sentences: int | None,
    beta: float,
) -> None:
    """Run a benchmark with a single model."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    config = ctx.obj.get("config") if ctx.obj else None

    # Load backend — prefer config-based resolution, fall back to spec parsing
    if config is not None and config.models:
        from llm_grammar_bench.backends import load_backend_from_config

        backend = load_backend_from_config(config, model)
    else:
        from llm_grammar_bench.backends import load_backend

        backend_type, model_name = _parse_model_spec(model)
        backend = load_backend(backend_type, model_name)

    click.echo(f"Backend: {backend.model_id}")

    # Load strategy
    from llm_grammar_bench.strategies import load_strategy

    strategy_instance = load_strategy(strategy)
    click.echo(f"Strategy: {type(strategy_instance).__name__}")

    # Load dataset
    from llm_grammar_bench.datasets import load_dataset as load_ds

    dataset_instance = load_ds(dataset)
    click.echo(f"Dataset: {type(dataset_instance).__name__}")

    # Determine evaluation settings from config or defaults
    eval_kwargs: dict = {}
    if config is not None:
        eval_kwargs["output"] = config.evaluation.output_dir
        eval_kwargs["beta"] = config.evaluation.beta

    # Run benchmark
    from llm_grammar_bench.evaluation.evaluator import Evaluator
    from llm_grammar_bench.evaluation.results import serialize_results

    evaluator = Evaluator(
        backend=backend,
        dataset=dataset_instance,
        strategy=strategy_instance,
        split=split,
        max_sentences=max_sentences,
        beta=eval_kwargs.get("beta", beta),
    )

    # Determine metrics from config or CLI default
    if config is not None and config.evaluation.metrics:
        metric_list = config.evaluation.metrics
    else:
        metric_list = [m.strip() for m in metrics.split(",")]

    result = evaluator.run(metrics=metric_list)

    # Determine output directory from config or CLI
    output_path = eval_kwargs.get("output", output)

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
    filepath = serialize_results(result, output_path)
    click.echo(f"Full results saved to: {filepath}")


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse a model spec string into backend type and model name.

    Accepts shorthand prefixes:
        hf:google/flan-t5-small  -> ("huggingface", "google/flan-t5-small")
        openai:gpt-4o            -> ("openai", "gpt-4o")
        anthropic:claude-3       -> ("anthropic", "claude-3")
        openrouter:meta-llama    -> ("openrouter", "meta-llama")
        huggingface:google/t5    -> ("huggingface", "google/t5")
    """
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
        click.echo("  openai       — OpenAI (GPT-4o, GPT-4, etc.)")
        click.echo("  anthropic    — Anthropic (Claude 3, Claude 3.5, etc.)")
        click.echo("  huggingface  — Local models via transformers")
        click.echo("  openrouter   — OpenRouter multi-provider gateway")
        click.echo("  openai_compatible — Any OpenAI-compatible endpoint")
        click.echo()
        click.echo("Use --config <file> to see configured models from a YAML file.")
        click.echo("Model spec format: '<backend>:<model>' (e.g. 'openai:gpt-4o')")


@main.command()
def list_datasets() -> None:
    """List available datasets."""
    click.echo("Available datasets: bea2019")
