"""BEA-2019 W&I+LOCNESS dataset loader via HuggingFace."""

from __future__ import annotations

import logging
from typing import ClassVar

from llm_grammar_bench.datasets.base import BaseDataset
from llm_grammar_bench.types import Example

logger = logging.getLogger(__name__)

_DATASET_ID = "jvamvas/peer_wi-locness"


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

            # Detokenize by joining tokens with spaces
            # The dataset uses a subword-like tokenization where spaces are
            # represented explicitly. Joining with space reconstructs the text.
            source = " ".join(item["src"])
            reference = " ".join(item["tgt"])

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
