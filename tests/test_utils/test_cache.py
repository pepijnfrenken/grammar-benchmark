"""Tests for cache module."""

import tempfile


def test_cache_set_and_get() -> None:
    """Test setting a value and retrieving it."""
    from llm_grammar_bench.utils.cache import CacheStore

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheStore(cache_dir=tmpdir)
        cache.set("model-1", "input text", "output correction")
        result = cache.get("model-1", "input text")
        assert result == "output correction"


def test_cache_miss() -> None:
    """Test getting from empty cache returns None."""
    from llm_grammar_bench.utils.cache import CacheStore

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheStore(cache_dir=tmpdir)
        result = cache.get("model-1", "nonexistent input")
        assert result is None


def test_cache_different_keys() -> None:
    """Test different model_id or input_text produce different cache entries."""
    from llm_grammar_bench.utils.cache import CacheStore

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheStore(cache_dir=tmpdir)

        cache.set("model-1", "input A", "output A")
        cache.set("model-2", "input A", "output B")
        cache.set("model-1", "input B", "output C")

        assert cache.get("model-1", "input A") == "output A"
        assert cache.get("model-2", "input A") == "output B"
        assert cache.get("model-1", "input B") == "output C"
        assert cache.get("model-1", "input X") is None


def test_cache_overwrite() -> None:
    """Test setting same key twice, second value wins."""
    from llm_grammar_bench.utils.cache import CacheStore

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheStore(cache_dir=tmpdir)

        cache.set("model-1", "input", "first value")
        assert cache.get("model-1", "input") == "first value"

        cache.set("model-1", "input", "second value")
        assert cache.get("model-1", "input") == "second value"


def test_cache_clear() -> None:
    """Test clear removes all entries."""
    from llm_grammar_bench.utils.cache import CacheStore

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheStore(cache_dir=tmpdir)

        cache.set("model-1", "input-1", "output-1")
        cache.set("model-1", "input-2", "output-2")
        cache.set("model-2", "input-1", "output-3")

        # Verify all are cached
        assert cache.get("model-1", "input-1") == "output-1"
        assert cache.get("model-1", "input-2") == "output-2"
        assert cache.get("model-2", "input-1") == "output-3"

        # Clear and verify all are gone
        cache.clear()
        assert cache.get("model-1", "input-1") is None
        assert cache.get("model-1", "input-2") is None
        assert cache.get("model-2", "input-1") is None


def test_cache_thread_safety() -> None:
    """CacheStore works correctly when shared across multiple threads."""
    import tempfile
    import threading

    from llm_grammar_bench.utils.cache import CacheStore

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheStore(cache_dir=tmpdir)
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(20):
                    cache.set(f"model-{thread_id}", f"input-{i}", f"output-{thread_id}-{i}")
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(100):
                    cache.get("model-0", "input-0")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-safety errors: {errors}"

        # Verify data integrity: each writer's entries should be readable
        for thread_id in range(4):
            for i in range(20):
                result = cache.get(f"model-{thread_id}", f"input-{i}")
                assert result == f"output-{thread_id}-{i}"
