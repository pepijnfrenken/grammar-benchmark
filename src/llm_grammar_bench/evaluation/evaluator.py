"""Benchmark evaluator: orchestrates dataset, strategy, backend, and metrics."""

from __future__ import annotations

import logging
import time

from llm_grammar_bench.backends.base import BaseBackend
from llm_grammar_bench.datasets.base import BaseDataset
from llm_grammar_bench.evaluation.metrics import MetricsRunner
from llm_grammar_bench.strategies.base import BaseStrategy
from llm_grammar_bench.types import BenchmarkResult, CorpusScores

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._backend = backend
        self._dataset = dataset
        self._strategy = strategy
        self._split = split
        self._max_sentences = max_sentences
        self._beta = beta

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

        # Collect predictions
        sources: list[str] = []
        hypotheses: list[str] = []
        references: list[list[str]] = []
        per_sentence: list[dict] = []
        cefr_groups: dict[str, dict[str, list]] = {}

        backend = self._backend  # Local ref for the inference loop
        for i, example in enumerate(examples):
            logger.debug("Correcting example %d/%d", i + 1, len(examples))
            hypothesis = self._strategy.correct(backend, example.source)

            sources.append(example.source)
            hypotheses.append(hypothesis)
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
                    cefr_groups[cefr] = {"sources": [], "hypotheses": [], "references": []}
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
