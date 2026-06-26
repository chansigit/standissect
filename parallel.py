"""Small stdlib-only concurrency helpers shared across standissect.

- thread_map  : bounded ThreadPoolExecutor for I/O-bound work (LLM calls).
- process_map : bounded spawn ProcessPoolExecutor for CPU-bound work (DEG),
                with single-thread BLAS in the workers.
- with_retry  : retry a call on transient exceptions with exponential backoff + jitter.

Imports nothing from standissect (no import cycle).
"""
from __future__ import annotations

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

_BLAS_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')


def thread_map(fn, items, *, max_workers):
    """Map ``fn`` over ``items`` on a bounded thread pool, results in input order.

    Serial (no pool) when ``max_workers<=1`` or there is at most one item.
    """
    items = list(items)
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, items))


def process_map(fn, items, *, max_workers):
    """Map ``fn`` over ``items`` on a bounded spawn process pool, in input order.

    Workers run single-thread BLAS: the relevant env vars are set in the parent
    *before* the pool is built (spawned children inherit them at import time — a
    worker initializer would run after numpy/BLAS already loaded), then restored.
    Serial (no pool, no pickling) when ``max_workers<=1`` or <=1 item.
    """
    items = list(items)
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    saved = {v: os.environ.get(v) for v in _BLAS_VARS}
    for v in _BLAS_VARS:
        os.environ[v] = '1'
    try:
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            return list(ex.map(fn, items))
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def with_retry(fn, *, retries=3, backoff=0.5, jitter=0.25, exceptions=(Exception,)):
    """Call ``fn()``; on a listed exception retry up to ``retries`` times with
    exponential backoff + uniform jitter. Re-raise the last exception if all fail."""
    last = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except exceptions as exc:
            last = exc
            if attempt == retries:
                break
            time.sleep(backoff * (2 ** attempt) + random.uniform(0, jitter))
    if last is None:
        raise RuntimeError("with_retry made no attempt (retries<0 or empty exceptions?)")
    raise last
