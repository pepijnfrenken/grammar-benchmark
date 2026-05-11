"""Click CLI for the LLM Grammar Bench framework."""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="llm-grammar-bench")
def main() -> None:
    """LLM Grammar Bench — evaluate AI models on grammatical error correction."""


@main.command()
@click.option(
    "--model",
    required=True,
    help="Model spec: 'hf:google/flan-t5-small', 'openai:gpt-4o', etc.",
)
@click.option("--dataset", default="bea2019", help="Dataset name.")
@click.option("--split", default="validation", help="Dataset split.")
@click.option("--strategy", default="seq2seq", help="Correction strategy.")
@click.option("--metrics", default="errant,gleu,bertscore", help="Comma-separated metrics.")
@click.option("--output", default="results/", help="Output directory.")
@click.option("--max-sentences", type=int, default=None, help="Limit sentences for quick testing.")
@click.option("--beta", type=float, default=0.5, help="Beta value for ERRANT F-score.")
def run(
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
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Parse model spec: "hf:google/flan-t5-small" -> ("huggingface", "google/flan-t5-small")
    backend_type, model_name = _parse_model_spec(model)

    # Load backend
    from llm_grammar_bench.backends import load_backend

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

    # Run benchmark
    from llm_grammar_bench.evaluation.evaluator import Evaluator
    from llm_grammar_bench.evaluation.results import serialize_results

    evaluator = Evaluator(
        backend=backend,
        dataset=dataset_instance,
        strategy=strategy_instance,
        split=split,
        max_sentences=max_sentences,
        beta=beta,
    )

    metric_list = [m.strip() for m in metrics.split(",")]
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
    # Default: assume huggingface if no prefix
    return "huggingface", spec


@main.command()
def list_models() -> None:
    """List available model backends."""
    click.echo("Available backends: openai, anthropic, huggingface, openrouter")


@main.command()
def list_datasets() -> None:
    """List available datasets."""
    click.echo("Available datasets: bea2019")
