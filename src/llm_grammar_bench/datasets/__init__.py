"""Dataset loaders for benchmark corpora."""

from llm_grammar_bench.datasets.base import BaseDataset, Example


def load_dataset(dataset_name: str, **kwargs) -> BaseDataset:
    """Factory to load a dataset by name.

    Args:
        dataset_name: Currently only "bea2019" is supported.
        **kwargs: Dataset-specific configuration (split, cefr_filter, etc.).

    Returns:
        A configured BaseDataset instance.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    match dataset_name:
        case "bea2019":
            from llm_grammar_bench.datasets.bea2019 import BEA2019Dataset

            return BEA2019Dataset(**kwargs)
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}. Valid options: bea2019")


__all__ = ["BaseDataset", "Example", "load_dataset"]
