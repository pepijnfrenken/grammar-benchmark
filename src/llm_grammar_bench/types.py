"""Shared types used across the framework."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Example:
    """A single benchmark example: erroneous source + one or more gold references."""

    id: str
    source: str
    references: list[str]
    cefr: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class CorpusScores:
    """Aggregate scores for a single metric across the full dataset."""

    metric: str
    precision: float
    recall: float
    f_score: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete result of a benchmark run."""

    model_id: str
    dataset_name: str
    strategy_name: str
    corpus_scores: dict[str, CorpusScores]
    per_sentence: list[dict] = field(default_factory=list)
    by_cefr: dict[str, dict[str, CorpusScores]] = field(default_factory=dict)
    runtime_seconds: float = 0.0
