"""BEA-2019 W&I+LOCNESS dataset loader via HuggingFace."""

from __future__ import annotations

import logging
from typing import ClassVar

from llm_grammar_bench.datasets.base import BaseDataset
from llm_grammar_bench.types import Example

logger = logging.getLogger(__name__)

_DATASET_ID = "jvamvas/peer_wi-locness"


# ── Detokenization ─────────────────────────────────────────────────────
# The dataset stores text as space-separated tokens (e.g. "It 's").
# Simple join(" ") leaves artifacts like "teacher ," instead of "teacher,".

_DETOKENIZE_RULES: list[tuple[str, str]] = [
    # Remove space before closing punctuation / clitics
    (r"\s+(\.)", r"\1"),
    (r"\s+(,)", r"\1"),
    (r"\s+(!)", r"\1"),
    (r"\s+(\?)", r"\1"),
    (r"\s+(;)", r"\1"),
    (r"\s+(:)", r"\1"),
    (r"\s+(\))", r"\1"),
    (r"\s+(\])", r"\1"),
    (r"\s+(\})", r"\1"),
    (r"\s+(%)", r"\1"),
    # Contractions: "do n't" → "don't", "I 'm" → "I'm"
    (r"\s+('s)\b", r"\1"),
    (r"\s+('ll)\b", r"\1"),
    (r"\s+('ve)\b", r"\1"),
    (r"\s+('re)\b", r"\1"),
    (r"\s+('d)\b", r"\1"),
    (r"\s+('m)\b", r"\1"),
    (r"\s+(n't)\b", r"\1"),
    (r"\s+('t)\b", r"\1"),
    # Quotes — handled in _detokenize for contextual open/close detection
    # Remove space after opening punctuation
    (r"(\()\s+", r"\1"),
    (r"(\[)\s+", r"\1"),
    (r"(\{)\s+", r"\1"),
    # Normalize multiple spaces
    (r"\s{2,}", " "),
]


def _detokenize(tokens: list[str]) -> str:
    """Join tokens and apply orthographic detokenization rules."""
    import re

    text = " ".join(tokens)
    for pattern, replacement in _DETOKENIZE_RULES:
        text = re.sub(pattern, replacement, text)

    # Handle quotes contextually: opening " removes trailing space;
    # closing " removes leading space (preceded by word or sentence-end punct).
    text = re.sub(r'(?:^|\s)"\s+', lambda m: m.group().rstrip(), text)
    text = re.sub(r'([\w.?!,;:])\s+"', r'\1"', text)

    return text.strip()


class BEA2019Dataset(BaseDataset):
    """Loads the BEA-2019 W&I+LOCNESS dataset from HuggingFace.

    Source: jvamvas/peer_wi-locness (parquet-converted mirror of the original
    bea2019st/wi_locness dataset).

    Splits: train (22,737 examples), dev, test
    CEFR levels: A (beginner), B (intermediate), C (advanced), N (native)

    Each example has a single reference correction (tgt). The original dataset
    provides multiple annotator references; this mirror uses one per example.
    """

    _SPLIT_MAP: ClassVar[dict[str, str]] = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    def __init__(self, cefr_filter: list[str] | None = None) -> None:
        self._cefr_filter = set(cefr_filter) if cefr_filter else None

    def load(self, split: str) -> list[Example]:
        """Load examples for the given split.

        Args:
            split: One of "train", "validation", "test".

        Returns:
            A list of Example objects with source, single-reference corrections,
            and CEFR level metadata.

        Raises:
            ValueError: If the split is unknown.
        """
        hf_split = self._SPLIT_MAP.get(split)
        if hf_split is None:
            raise ValueError(f"Unknown split '{split}'. Valid options: {list(self._SPLIT_MAP)}")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "BEA-2019 dataset requires the datasets package. Install with: pip install datasets"
            ) from None

        logger.info("Loading %s split '%s'...", _DATASET_ID, hf_split)
        ds = load_dataset(_DATASET_ID, split=hf_split)

        examples: list[Example] = []
        skipped_filter = 0

        for item in ds:
            # Determine CEFR level from tgt_class
            tgt_class: dict = item["tgt_class"]
            cefr_levels = [lvl for lvl, present in tgt_class.items() if present]
            cefr_level = cefr_levels[0] if cefr_levels else None

            # Apply CEFR filter if configured
            if self._cefr_filter is not None and cefr_level not in self._cefr_filter:
                skipped_filter += 1
                continue

            # Detokenize: join tokens then fix orthographic spacing
            source = _detokenize(item["src"])
            reference = _detokenize(item["tgt"])
            example_id = str(item.get("id", ""))
            examples.append(
                Example(
                    id=example_id,
                    source=source,
                    references=[reference],
                    metadata={"cefr": cefr_level},
                )
            )
        if skipped_filter > 0:
            logger.info(
                "Skipped %d examples due to CEFR filter (%s).",
                skipped_filter,
                self._cefr_filter,
            )

        logger.info("Loaded %d examples from %s/%s.", len(examples), _DATASET_ID, hf_split)
        return examples
