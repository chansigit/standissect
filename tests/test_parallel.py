import pathlib
import sys
import time

import pytest

_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_DIR))

import parallel  # noqa: E402


def _double(x):
    return x * 2


def test_thread_map_preserves_order():
    assert parallel.thread_map(_double, [1, 2, 3, 4], max_workers=3) == [2, 4, 6, 8]


def test_thread_map_serial_fallback():
    assert parallel.thread_map(_double, [5], max_workers=8) == [10]
    assert parallel.thread_map(_double, [1, 2], max_workers=1) == [2, 4]


def test_thread_map_is_concurrent():
    def slow(x):
        time.sleep(0.2)
        return x
    t0 = time.perf_counter()
    parallel.thread_map(slow, list(range(8)), max_workers=8)
    assert time.perf_counter() - t0 < 1.0      # 8x0.2s serial=1.6s; concurrent<1s


def test_process_map_preserves_order():
    # _double is a top-level fn so the spawn pool can pickle it
    assert parallel.process_map(_double, [1, 2, 3], max_workers=2) == [2, 4, 6]


_calls = {"n": 0}


def _flaky():
    _calls["n"] += 1
    if _calls["n"] < 3:
        raise ValueError("transient")
    return "ok"


def test_with_retry_succeeds_on_third_attempt():
    _calls["n"] = 0
    assert parallel.with_retry(_flaky, retries=3, backoff=0.0, jitter=0.0,
                               exceptions=(ValueError,)) == "ok"
    assert _calls["n"] == 3


def test_with_retry_reraises_after_exhaustion():
    def always_fail():
        raise ValueError("nope")
    with pytest.raises(ValueError):
        parallel.with_retry(always_fail, retries=2, backoff=0.0, jitter=0.0,
                            exceptions=(ValueError,))


def test_with_retry_does_not_catch_other_exceptions():
    def boom():
        raise KeyError("k")
    with pytest.raises(KeyError):
        parallel.with_retry(boom, retries=3, backoff=0.0, jitter=0.0,
                            exceptions=(ValueError,))
