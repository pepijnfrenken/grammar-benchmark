"""Tests for cache module."""

import tempfile
import threading


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


def test_make_request_cache_text_includes_kwargs() -> None:
    """Test request cache keys include prompt-shaping kwargs."""
    from llm_grammar_bench.utils.cache import make_request_cache_text

    base = make_request_cache_text("input", system_prompt="A", temperature=0)
    same = make_request_cache_text("input", temperature=0, system_prompt="A")
    different = make_request_cache_text("input", system_prompt="B", temperature=0)

    assert base == same
    assert base != different
    assert make_request_cache_text("input") == "input"


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


def test_cache_get_or_compute_reuses_cached_value() -> None:
    """Test get_or_compute skips compute after a value is cached."""
    from llm_grammar_bench.utils.cache import CacheStore

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheStore(cache_dir=tmpdir)
        calls = 0

        def compute() -> str:
            nonlocal calls
            calls += 1
            return "computed"

        assert cache.get_or_compute("model-1", "input", compute) == "computed"
        assert cache.get_or_compute("model-1", "input", compute) == "computed"
        assert calls == 1


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


def test_cache_get_or_compute_deduplicates_concurrent_misses() -> None:
    """Test concurrent misses for one key call compute once."""
    from llm_grammar_bench.utils.cache import CacheStore

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheStore(cache_dir=tmpdir)
        calls = 0
        compute_started = threading.Event()
        release_compute = threading.Event()
        errors: list[Exception] = []
        results: list[str] = []

        def compute() -> str:
            nonlocal calls
            calls += 1
            compute_started.set()
            release_compute.wait(timeout=5)
            return "computed"

        def worker() -> None:
            try:
                results.append(cache.get_or_compute("model-1", "input", compute))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        assert compute_started.wait(timeout=5)
        release_compute.set()
        for thread in threads:
            thread.join()

        assert not errors
        assert results == ["computed"] * 4
        assert calls == 1


def test_cache_thread_safety() -> None:
    """CacheStore works correctly when shared across multiple threads."""
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
