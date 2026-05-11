"""Tests for retry and rate limiting utilities."""

import time

import pytest


def test_retry_success_first_attempt() -> None:
    """Test decorated function succeeds on first attempt."""
    from llm_grammar_bench.utils.retry import retry

    @retry(max_attempts=3)
    def succeed() -> str:
        return "success"

    result = succeed()
    assert result == "success"


def test_retry_on_exception() -> None:
    """Test decorated function fails twice then succeeds, verify it was called 3 times."""
    from llm_grammar_bench.utils.retry import retry

    call_count = 0

    @retry(max_attempts=3, base_delay=0.01, backoff_factor=1.0)
    def fail_twice_then_succeed() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TimeoutError("Temporary failure")
        return "success"

    result = fail_twice_then_succeed()
    assert result == "success"
    assert call_count == 3


def test_retry_exhausted() -> None:
    """Test decorated function always fails, verify it raises after max_attempts."""
    from llm_grammar_bench.utils.retry import retry

    call_count = 0

    @retry(max_attempts=3, base_delay=0.01, backoff_factor=1.0)
    def always_fail() -> str:
        nonlocal call_count
        call_count += 1
        raise TimeoutError("Persistent failure")

    try:
        always_fail()
        pytest.fail("Should have raised TimeoutError")
    except TimeoutError as e:
        assert "Persistent failure" in str(e)
        assert call_count == 3


def test_retry_non_retryable_exception() -> None:
    """Test that non-retryable exceptions are not retried."""
    from llm_grammar_bench.utils.retry import retry

    call_count = 0

    @retry(max_attempts=3, base_delay=0.01, retryable_exceptions=(TimeoutError,))
    def raise_non_retryable() -> str:
        nonlocal call_count
        call_count += 1
        raise ValueError("Not retryable")

    try:
        raise_non_retryable()
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        assert "Not retryable" in str(e)
        assert call_count == 1


def test_rate_limiter_initial_call() -> None:
    """Test RateLimiter allows first call immediately."""
    from llm_grammar_bench.utils.retry import RateLimiter

    limiter = RateLimiter(calls_per_second=10)

    start = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - start

    # First call should be immediate (no delay)
    assert elapsed < 0.1


def test_rate_limiter_waits() -> None:
    """Test call twice in rapid succession, verify second call waits."""
    from llm_grammar_bench.utils.retry import RateLimiter

    limiter = RateLimiter(calls_per_second=2)  # 0.5 sec between calls

    start = time.monotonic()
    limiter.acquire()
    first_elapsed = time.monotonic() - start

    call1_time = time.monotonic()
    limiter.acquire()
    call2_time = time.monotonic()

    # First call should be immediate
    assert first_elapsed < 0.1

    # Second call should have waited at least ~0.5 seconds
    wait_time = call2_time - call1_time
    assert wait_time >= 0.4  # Allow some tolerance


def test_rate_limiter_multiple_calls() -> None:
    """Test rate limiter with multiple sequential calls."""
    from llm_grammar_bench.utils.retry import RateLimiter

    limiter = RateLimiter(calls_per_second=5)  # 0.2 sec between calls

    times = []
    start = time.monotonic()
    for _ in range(3):
        limiter.acquire()
        times.append(time.monotonic() - start)

    # Check intervals between calls (approximately 0.2 sec each)
    interval_1 = times[1] - times[0]
    interval_2 = times[2] - times[1]

    # Intervals should be around 0.2 seconds (with tolerance)
    assert 0.1 < interval_1 < 0.4
    assert 0.1 < interval_2 < 0.4
