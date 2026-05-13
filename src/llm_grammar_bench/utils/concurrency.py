"""Concurrent execution utilities for batch processing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, TypeVar, cast

from llm_grammar_bench.utils.retry import RateLimiter

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchExecutor:
    """Processes items concurrently with configurable concurrency and rate limiting.

    Usage:
        executor = BatchExecutor(max_workers=5, rate_limiter=RateLimiter(10))
        results = executor.map(correct_fn, sentences)
    """

    def __init__(
        self,
        max_workers: int = 5,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._max_workers = max_workers
        self._rate_limiter = rate_limiter

    def map(self, func: Callable[[T], R], items: list[T]) -> list[R]:
        """Apply func to each item concurrently, returning results in original order.

        Args:
            func: Callable that takes one item and returns a result.
            items: List of items to process.

        Returns:
            List of results in the same order as items.

        Raises:
            The first exception raised by any call to func.
        """
        if not items:
            return []

        if self._max_workers == 1 and self._rate_limiter is None:
            # Fast path: sequential execution without overhead
            return [func(item) for item in items]

        # Build an index-keyed result buffer (pre-allocated, filled by futures)
        results = cast("list[R]", [None] * len(items))

        def _wrap(item: T, index: int) -> tuple[int, R]:
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            return index, func(item)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures: dict[Future[Any], int] = {}
            for idx, item in enumerate(items):
                futures[pool.submit(_wrap, item, idx)] = idx

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results
