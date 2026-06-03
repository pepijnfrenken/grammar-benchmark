"""Disk-based output cache to avoid re-running expensive model calls."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any


def make_request_cache_text(text: str, **kwargs: Any) -> str:
    """Return stable cache text that includes request-shaping parameters."""
    if not kwargs:
        return text

    parts = [text]
    for key in sorted(kwargs):
        value = kwargs[key]
        parts.append(f"{key}={value!r}")
    return "\n\n__cache_request__\n" + "\n".join(parts)


class CacheStore:
    """Simple file-based cache keyed on model + input text hash.

    Cache entries are stored as plain text files in the cache directory.
    Thread-safe: can be shared across multiple threads.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "llm-grammar-bench"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def _key(self, model_id: str, text: str) -> str:
        payload = f"{model_id}:{text}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _entry_path(self, model_id: str, text: str) -> Path:
        return self._cache_dir / self._key(model_id, text)

    def _get_key_lock(self, cache_key: str) -> threading.Lock:
        with self._lock:
            lock = self._key_locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[cache_key] = lock
            return lock

    def get(self, model_id: str, text: str) -> str | None:
        """Return cached correction, or None if not cached."""
        entry_path = self._entry_path(model_id, text)
        with self._lock:
            if entry_path.exists():
                return entry_path.read_text(encoding="utf-8")
        return None

    def set(self, model_id: str, text: str, correction: str) -> None:
        """Store a correction in the cache."""
        entry_path = self._entry_path(model_id, text)
        with self._lock:
            entry_path.write_text(correction, encoding="utf-8")

    def get_or_compute(self, model_id: str, text: str, compute: Callable[[], str]) -> str:
        """Return a cached correction, computing it once on cache miss."""
        cached = self.get(model_id, text)
        if cached is not None:
            return cached

        cache_key = self._key(model_id, text)
        with self._get_key_lock(cache_key):
            cached = self.get(model_id, text)
            if cached is not None:
                return cached
            correction = compute()
            self.set(model_id, text, correction)
            return correction

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            for entry in self._cache_dir.iterdir():
                entry.unlink()
