"""Benchmark evaluator: orchestrates dataset, strategy, backend, and metrics."""

from __future__ import annotations

import logging
import time

from llm_grammar_bench.backends.base import BaseBackend
from llm_grammar_bench.datasets.base import BaseDataset
from llm_grammar_bench.evaluation.metrics import MetricsRunner
from llm_grammar_bench.strategies.base import BaseStrategy
from llm_grammar_bench.types import BenchmarkResult, CorpusScores, Example
from llm_grammar_bench.utils.concurrency import BatchExecutor
from llm_grammar_bench.utils.retry import RateLimiter

logger = logging.getLogger(__name__)



def _log_correction_sanity_check(examples: list[Example], hypotheses: list[str]) -> None:
    """Log a summary of correction quality to catch silent failures early.

    Prints sample outputs and warns on high rates of empty/unchanged results,
    so the user can abort before expensive metric computation.
    """
    total = len(hypotheses)
    empty_count = 0
    unchanged_count = 0

    for example, hypothesis in zip(examples, hypotheses, strict=True):
        if not hypothesis.strip():
            empty_count += 1
        elif hypothesis.strip() == example.source.strip():
            unchanged_count += 1

    # Show first 3 non-trivial examples (source → hypothesis)
    shown = 0
    for example, hypothesis in zip(examples, hypotheses, strict=True):
        if shown >= 3:
            break
        if hypothesis.strip() and hypothesis.strip() != example.source.strip():
            logger.info("  [%s] %r → %r", example.id, example.source, hypothesis)
            shown += 1

    # If no non-trivial examples, show the raw first 3
    if shown == 0:
        for i in range(min(3, total)):
            logger.info(
                "  [%s] src=%r → hyp=%r",
                examples[i].id,
                examples[i].source,
                hypotheses[i],
            )
    # Summary stats
    empty_pct = (empty_count / total) * 100 if total else 0
    unchanged_pct = (unchanged_count / total) * 100 if total else 0

    logger.info(
        "Correction summary: %d/%d empty (%.1f%%), %d/%d unchanged (%.1f%%)",
        empty_count, total, empty_pct,
        unchanged_count, total, unchanged_pct,
    )

    if empty_pct > 30:
        logger.warning(
            "%.0f%% of corrections are empty — model may not understand the prompt. "
            "Try a different strategy (--strategy edit_based) or check the model.",
            empty_pct,
        )
    elif unchanged_pct > 50:
        logger.warning(
            "%.0f%% of corrections are unchanged — model may be copying input. "
            "Check the system prompt or try a different strategy.",
            unchanged_pct,
        )

class Evaluator:
    """Orchestrates a benchmark run: load data, run corrections, compute metrics."""

    def __init__(
        self,
        backend: BaseBackend,
        dataset: BaseDataset,
        strategy: BaseStrategy,
        split: str = "validation",
        max_sentences: int | None = None,
        beta: float = 0.5,
        max_workers: int = 1,
        rate_limit: float | None = None,
    ) -> None:
        self._backend = backend
        self._dataset = dataset
        self._strategy = strategy
        self._split = split
        self._max_sentences = max_sentences
        self._beta = beta
        self._max_workers = max_workers
        self._rate_limit = rate_limit

    def run(self, metrics: list[str] | None = None) -> BenchmarkResult:
        """Execute the full benchmark pipeline.

        Args:
            metrics: List of metric names to compute (default: all available).

        Returns:
            A BenchmarkResult with corpus-level scores, per-sentence detail,
            CEFR breakdown, and runtime information.
        """
        start_time = time.monotonic()

        # Load dataset
        examples = self._dataset.load(self._split)
        if self._max_sentences is not None and self._max_sentences < len(examples):
            examples = examples[: self._max_sentences]
            logger.info("Limited to %d examples.", self._max_sentences)

        logger.info("Running corrections on %d examples...", len(examples))

        # Build rate limiter if configured
        rate_limiter: RateLimiter | None = None
        if self._rate_limit is not None and self._rate_limit > 0:
            rate_limiter = RateLimiter(calls_per_second=self._rate_limit)

        if self._max_workers > 1:
            logger.info(
                "Using %d concurrent workers%s.",
                self._max_workers,
                (f" with rate limit {self._rate_limit:.1f} calls/s" if rate_limiter else ""),
            )

        # Run corrections (concurrently if max_workers > 1)
        backend = self._backend
        strategy = self._strategy

        import threading

        total = len(examples)
        progress_lock = threading.Lock()
        completed_count = 0

        def _correct_one(
            example: Example,
            _backend: BaseBackend = backend,
            _strategy: BaseStrategy = strategy,
        ) -> str:
            nonlocal completed_count
            result = _strategy.correct(_backend, example.source)
            with progress_lock:
                completed_count += 1
                if completed_count % 100 == 0 or completed_count == total:
                    logger.info(
                        "Progress: %d/%d corrections complete", completed_count, total
                    )
            return result

        executor = BatchExecutor(max_workers=self._max_workers, rate_limiter=rate_limiter)
        hypotheses = executor.map(_correct_one, list(examples))


        # ── Sanity-check corrections before computing metrics ──────────────
        _log_correction_sanity_check(examples, hypotheses)

        # Build result collections
        sources: list[str] = []
        references: list[list[str]] = []
        per_sentence: list[dict] = []
        cefr_groups: dict[str, dict[str, list]] = {}

        for example, hypothesis in zip(examples, hypotheses, strict=True):
            sources.append(example.source)
            references.append(example.references)

            # Per-sentence record
            per_sentence.append(
                {
                    "id": example.id,
                    "source": example.source,
                    "hypothesis": hypothesis,
                    "references": example.references,
                    "metadata": example.metadata,
                }
            )

            # Group by CEFR for breakdown
            cefr = example.metadata.get("cefr")
            if cefr:
                if cefr not in cefr_groups:
                    cefr_groups[cefr] = {
                        "sources": [],
                        "hypotheses": [],
                        "references": [],
                    }
                cefr_groups[cefr]["sources"].append(example.source)
                cefr_groups[cefr]["hypotheses"].append(hypothesis)
                cefr_groups[cefr]["references"].append(example.references)

        # Capture model_id before releasing backend GPU memory
        model_id = backend.model_id

        # Release backend to free GPU memory before metric computation
        backend.release()
        del backend

        # Compute corpus-level metrics
        metrics_runner = MetricsRunner(beta=self._beta)
        corpus_scores = metrics_runner.compute(
            sources, hypotheses, references, metric_names=metrics
        )

        # Compute per-CEFR scores
        by_cefr: dict[str, dict[str, CorpusScores]] = {}
        for cefr, groups in cefr_groups.items():
            if len(groups["sources"]) > 0:
                cefr_scores = metrics_runner.compute(
                    groups["sources"],
                    groups["hypotheses"],
                    groups["references"],
                    metric_names=metrics,
                )
                by_cefr[cefr] = cefr_scores

        runtime = time.monotonic() - start_time
        logger.info("Benchmark completed in %.2f seconds.", runtime)

        # Infer dataset and strategy names from classes
        dataset_name = type(self._dataset).__name__
        strategy_name = type(self._strategy).__name__

        return BenchmarkResult(
            model_id=model_id,
            dataset_name=dataset_name,
            strategy_name=strategy_name,
            corpus_scores=corpus_scores,
            per_sentence=per_sentence,
            by_cefr=by_cefr,
            runtime_seconds=runtime,
        )
