"""Metrics computation wrapping gec-metrics for ERRANT, GLEU, and BERTScore."""

from __future__ import annotations

import logging

from llm_grammar_bench.types import CorpusScores

logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = frozenset({"errant", "gleu", "bertscore"})


class MetricsRunner:
    """Computes GEC metrics using the gec-metrics library.

    Supported metrics: errant, gleu, bertscore.
    """

    def __init__(self, beta: float = 0.5) -> None:
        self._beta = beta

    def compute(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]],
        metric_names: list[str] | None = None,
    ) -> dict[str, CorpusScores]:
        """Compute configured metrics on the given predictions.

        Args:
            sources: Original erroneous sentences.
            hypotheses: Model-corrected sentences.
            references: Gold-standard corrections (may have multiple per sentence).
            metric_names: Metrics to compute. Defaults to all supported metrics.

        Returns:
            Dictionary mapping metric names to CorpusScores.

        Raises:
            ValueError: If a requested metric is not supported.
        """
        names = metric_names if metric_names is not None else list(_SUPPORTED_METRICS)
        unknown = [n for n in names if n not in _SUPPORTED_METRICS]
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}. Supported: {sorted(_SUPPORTED_METRICS)}")

        scores: dict[str, CorpusScores] = {}

        for name in names:
            logger.info("Computing %s metric...", name)
            score = self._compute_single(name, sources, hypotheses, references)
            scores[name] = score
            logger.info("%s: %.4f", name, score.f_score)

        return scores

    def _compute_single(
        self,
        name: str,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]],
    ) -> CorpusScores:
        """Compute a single metric."""
        from typing import Any

        import gec_metrics

        metric_cls: Any = gec_metrics.get_metric(name)

        if name == "errant":
            config = metric_cls.Config(beta=self._beta, language="en")
        elif name == "gleu" or name == "bertscore":
            config = metric_cls.Config()
        else:
            raise ValueError(f"Unsupported metric: {name}")

        metric = metric_cls(config)

        # gec-metrics expects references as (num_references, num_sentences).
        # Our input is (num_sentences, num_references) — transpose it.
        refs_transposed = _transpose_references(references)

        if name == "bertscore":
            # BERTScore is source-free: only needs hypotheses and references
            value = float(metric.score_corpus(hypotheses, refs_transposed))
        else:
            # ERRANT and GLEU require sources
            value = float(metric.score_corpus(sources, hypotheses, refs_transposed))

        return CorpusScores(
            metric=name,
            precision=0.0,
            recall=0.0,
            f_score=value,
            extra={"beta": self._beta} if name == "errant" else {},
        )


def _transpose_references(references: list[list[str]]) -> list[list[str]]:
    """Transpose references from (sentences, refs) to (refs, sentences).

    gec-metrics expects references shaped as (num_references, num_sentences).
    Our internal format is (num_sentences, num_references).
    """
    if not references:
        return []
    num_refs = max(len(refs) for refs in references)
    num_sents = len(references)
    transposed: list[list[str]] = [[] for _ in range(num_refs)]
    for sent_idx in range(num_sents):
        for ref_idx in range(num_refs):
            if ref_idx < len(references[sent_idx]):
                transposed[ref_idx].append(references[sent_idx][ref_idx])
            else:
                # Pad missing references with empty string
                transposed[ref_idx].append("")
    return transposed
