"""Disk-based output cache to avoid re-running expensive model calls."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path


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

    def _key(self, model_id: str, text: str) -> str:
        payload = f"{model_id}:{text}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, model_id: str, text: str) -> str | None:
        """Return cached correction, or None if not cached."""
        entry_path = self._cache_dir / self._key(model_id, text)
        with self._lock:
            if entry_path.exists():
                return entry_path.read_text(encoding="utf-8")
        return None

    def set(self, model_id: str, text: str, correction: str) -> None:
        """Store a correction in the cache."""
        entry_path = self._cache_dir / self._key(model_id, text)
        with self._lock:
            entry_path.write_text(correction, encoding="utf-8")

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            for entry in self._cache_dir.iterdir():
                entry.unlink()
