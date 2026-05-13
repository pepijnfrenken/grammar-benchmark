"""Retry and rate limiting utilities for API backends."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (
        TimeoutError,
        ConnectionError,
    ),
) -> Callable[[F], F]:
    """Decorator: retry a function with exponential backoff.

    Args:
        max_attempts: Maximum number of total attempts (including the first).
        base_delay: Initial delay in seconds before first retry.
        backoff_factor: Multiplier for each subsequent delay.
        retryable_exceptions: Exception types that trigger a retry.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (backoff_factor**attempt)
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                        attempt + 1,
                        max_attempts,
                        func.__name__,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            # Unreachable: the loop either returns or raises.
            # If all attempts are exhausted without a return, raise the last exception.
            assert last_exception is not None
            raise last_exception

        return cast(F, wrapper)

    return decorator


class RateLimiter:
    """Simple token-bucket rate limiter for API calls.

    Thread-safe: can be shared across multiple threads.

    Usage:
        limiter = RateLimiter(calls_per_second=5)
        for item in items:
            limiter.acquire()
            api_call(item)
    """

    def __init__(self, calls_per_second: float) -> None:
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until the next call is permitted."""
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_call
            sleep_time = self._min_interval - elapsed if elapsed < self._min_interval else 0.0
            self._last_call = now + sleep_time

        if sleep_time > 0:
            time.sleep(sleep_time)
