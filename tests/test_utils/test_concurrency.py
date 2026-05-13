"""Tests for BatchExecutor concurrent execution utility."""

import time

import pytest

from llm_grammar_bench.utils.concurrency import BatchExecutor
from llm_grammar_bench.utils.retry import RateLimiter


def _identity(x: int) -> int:
    return x


def _slow_identity(x: int, delay: float = 0.05) -> int:
    time.sleep(delay)
    return x


def test_batch_executor_sequential_empty() -> None:
    """BatchExecutor.map with empty list returns empty list."""
    executor = BatchExecutor(max_workers=1)
    assert executor.map(_identity, []) == []


def test_batch_executor_sequential_single() -> None:
    """BatchExecutor.map with one item returns that result."""
    executor = BatchExecutor(max_workers=1)
    assert executor.map(_identity, [42]) == [42]


def test_batch_executor_sequential_multiple() -> None:
    """BatchExecutor.map with max_workers=1 processes sequentially in order."""
    executor = BatchExecutor(max_workers=1)
    items = [1, 2, 3, 4, 5]
    assert executor.map(_identity, items) == items


def test_batch_executor_concurrent_order_preserved() -> None:
    """Concurrent execution preserves input order in output."""
    executor = BatchExecutor(max_workers=4)
    items = [10, 20, 30, 40, 50, 60, 70, 80]
    results = executor.map(_identity, items)
    assert results == items


def test_batch_executor_concurrent_faster_than_sequential() -> None:
    """Concurrent execution of slow tasks is faster than sequential."""
    items = list(range(8))
    delay = 0.05

    def _slow(x: int) -> int:
        time.sleep(delay)
        return x

    # Sequential
    start = time.monotonic()
    seq_executor = BatchExecutor(max_workers=1)
    seq_executor.map(_slow, items)
    seq_time = time.monotonic() - start

    # Concurrent
    start = time.monotonic()
    conc_executor = BatchExecutor(max_workers=4)
    conc_executor.map(_slow, items)
    conc_time = time.monotonic() - start

    # Sequential: 8 * 0.05 = 0.4s; Concurrent with 4 workers: ~2 * 0.05 = 0.1s
    # Allow generous tolerance
    assert conc_time < seq_time * 0.7, (
        f"Concurrent ({conc_time:.3f}s) should be faster than sequential ({seq_time:.3f}s)"
    )


def test_batch_executor_with_rate_limiter() -> None:
    """BatchExecutor respects rate limiting."""
    rate_limiter = RateLimiter(calls_per_second=20)  # 0.05s between calls
    executor = BatchExecutor(max_workers=4, rate_limiter=rate_limiter)

    items = list(range(12))  # 12 calls at 20/s = at least 0.55s

    start = time.monotonic()
    executor.map(_identity, items)
    elapsed = time.monotonic() - start

    # With 12 calls at 20 calls/s, min time = 11 * 0.05 = 0.55s
    # But with 4 workers, actual time could be less due to concurrent dispatching.
    # The rate limiter gates before each call, so minimum is still ~0.55s.
    assert elapsed >= 0.45, f"Rate-limited execution took {elapsed:.3f}s, expected >= 0.45s"


def test_batch_executor_no_rate_limiter_fast() -> None:
    """Without rate limiter, concurrent execution is fast."""
    executor = BatchExecutor(max_workers=8)
    items = list(range(16))

    start = time.monotonic()
    executor.map(_identity, items)
    elapsed = time.monotonic() - start

    # Should be nearly instant (just thread overhead)
    assert elapsed < 0.5, f"Unlimited execution took {elapsed:.3f}s"


def test_batch_executor_exception_propagates() -> None:
    """Exceptions in the mapped function propagate to the caller."""

    def _fail(x: int) -> int:
        if x == 3:
            raise ValueError(f"Failed on {x}")
        return x

    executor = BatchExecutor(max_workers=2)
    with pytest.raises(ValueError, match="Failed on 3"):
        executor.map(_fail, [1, 2, 3, 4])


def test_batch_executor_invalid_max_workers() -> None:
    """max_workers < 1 raises ValueError."""
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        BatchExecutor(max_workers=0)
