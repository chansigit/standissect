# DEG + LLM concurrency & fast Wilcoxon — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make standissect's DEG fast + correct (sparse, tie-corrected, rank-once Wilcoxon) and parallelize the slow stages — per-cluster DEG across processes, the LLM stages across threads with retry/backoff.

**Architecture:** A new stdlib-only `parallel.py` supplies `thread_map` (I/O-bound LLM), `process_map` (CPU-bound DEG, spawn + BLAS-pinned), and `with_retry`. The Wilcoxon kernel in `cluster.py` is rewritten to rank CSC-sparse columns without densifying, with tie-corrected variance, computing all groups via `Gᵀ@ranks` sparse matmuls. The pipeline's dissect loop fans out over `process_map`; the LLM engines gain retry; the naming/narrative/diagnosis stages fan out over `thread_map`. All stages still reassemble global outputs from per-cluster artifacts, so parallel output is byte-identical to serial.

**Tech Stack:** Python 3.12, numpy, scipy (`scipy.sparse`, `scipy.stats`), pandas, `statsmodels.multipletests` (already used), stdlib `concurrent.futures`/`multiprocessing`/`time`/`random`. The vendored stdlib `llm_client` (ARK). No new pip dependencies.

## Global Constraints

- **Branch:** `feat/deg-llm-concurrency` (already checked out, clean off `main`). Per-task commits on this branch. **Do NOT merge to `main`** without the user.
- **Run tests LOCALLY on this compute node** — the session runs inside a Slurm allocation (`sh02-01n51`, ~5 cores / 105 GB). Do **NOT** `srun`/`sbatch`/request other nodes. Test runner, from the repo root `/scratch/users/chensj16/projects/standissect`:
  `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest <target> -v`
- **No new pip dependencies** (numpy/scipy/pandas/statsmodels/stdlib only). Do **NOT** modify `llm_client.py` (vendored, byte-identical with stanmetacols).
- **Determinism:** every parallel stage writes per-cluster artifacts and reassembles global TSVs **after** the pool — output must be identical regardless of completion order.
- **Public DEG signatures + columns unchanged:** `wilcoxon_one_vs_rest` returns long columns `group, gene, scores, pvals, pvals_adj, logfoldchanges, mean_in, mean_out`; `wilcoxon_vs_reference` returns `names, logfoldchanges, pvals, pvals_adj, scores, direction` (top `n_genes` by `scores` desc).
- **Tie correction changes DEG scores by design** (current code omits it → conservative). Pin the new values with a scipy reference.
- **Defaults (verbatim):** `llm_concurrency=8`, `llm_retries=3`, `with_retry(backoff=0.5, jitter=0.25)`, shared ARK client `timeout` default `60→120`, DEG `n_jobs` default `min(n_clusters, os.cpu_count() or 1, 8)`.
- **Spec:** `docs/superpowers/specs/2026-06-25-deg-llm-concurrency-design.md`.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `parallel.py` | **create** | `thread_map`, `process_map` (spawn + BLAS-pin), `with_retry`. stdlib only; imports nothing from standissect. |
| `cluster.py` | modify | rewrite `wilcoxon_one_vs_rest` + `wilcoxon_vs_reference` onto a shared `_wilcoxon_sparse_stats` core; delete fork-pool. |
| `diagnosis.py` | modify | `LLMDiagnosisEngine` retry; bump `timeout` defaults 60→120; `llm_retries` param on engine + factories. |
| `annotate.py` | modify | `LLMNamingEngine`/`NarrativeEngine` retry; `run_naming_stage`/`run_narrative_stage` thread-pool; `llm_concurrency`/`llm_retries` plumbing. |
| `pipeline.py` | modify | dissect `process_map` (refactor `_dissect_one`); diagnosis-stage thread-pool; new params + params.json. |
| `cli.py` | modify | `--llm-concurrency`, `--llm-retries`, `--ark-timeout`; `--n-jobs` help. |
| `tests/test_parallel.py` | **create** | thread_map/process_map/with_retry. |
| `tests/test_wilcoxon.py` | **create** | kernel vs scipy reference (ties/sparse/edge). |
| `tests/test_annotate.py` | modify | retry + thread-pool naming/narrative. |
| `tests/test_diagnosis_llm.py` | modify | diagnosis retry. |

Dependency graph (no cycles): `parallel.py` ← {`cluster`, `diagnosis`, `annotate`, `pipeline`}; `llm_client` ← {`diagnosis`, `annotate`}; everything ← `pipeline`.

---

## Task 0: Branch baseline

**Files:** none.

- [ ] **Step 1: Confirm branch + clean tree**
```bash
cd /scratch/users/chensj16/projects/standissect
git branch --show-current      # expect: feat/deg-llm-concurrency
git status --short             # expect: empty
```
- [ ] **Step 2: Green baseline (run locally — we are on a compute node)**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/ -q
```
Expected: `28 passed`.

---

## Task 1: `parallel.py` — thread_map, process_map, with_retry

**Files:**
- Create: `parallel.py`
- Test: create `tests/test_parallel.py`

**Interfaces — Produces:**
- `thread_map(fn, items, *, max_workers) -> list` — results in input order; `max_workers<=1` or `len(items)<=1` → serial.
- `process_map(fn, items, *, max_workers) -> list` — spawn pool, BLAS pinned via parent env, results in input order; serial for `max_workers<=1`/`len<=1`.
- `with_retry(fn, *, retries=3, backoff=0.5, jitter=0.25, exceptions=(Exception,)) -> result` — retry on listed exceptions with exp backoff + jitter; re-raise last after exhaustion.

- [ ] **Step 1: Write the failing tests** — create `tests/test_parallel.py`
```python
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
```

- [ ] **Step 2: Run to verify it fails**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_parallel.py -v
```
Expected: `ModuleNotFoundError: No module named 'parallel'`.

- [ ] **Step 3: Implement** — create `parallel.py`
```python
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
    raise last
```

- [ ] **Step 4: Run to verify it passes**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_parallel.py -v
```
Expected: `7 passed`.

- [ ] **Step 5: Commit**
```bash
git add parallel.py tests/test_parallel.py
git commit -m "feat(parallel): add thread_map/process_map/with_retry helpers"
```

---

## Task 2: Fast sparse + tie-corrected Wilcoxon kernel (`cluster.py`)

**Files:**
- Modify: `cluster.py` — add `_wilcoxon_sparse_stats`; rewrite `wilcoxon_one_vs_rest` (924–1011) and `_wilcoxon_chunked` (1014–1100) and `wilcoxon_vs_reference` (195–268); delete `_WILCOXON_WORKER`/`_wilcoxon_worker_call`.
- Test: create `tests/test_wilcoxon.py`

**Interfaces:**
- Produces `_wilcoxon_sparse_stats(X, group_labels, groups) -> dict[group -> (z, mean_in, mean_out)]`, each a `(n_genes,)` `float32` array. `z` is the tie-corrected normal-approx Mann-Whitney one-vs-rest statistic; `NaN` where a group has `<2` cells or `>n-2` cells.
- `wilcoxon_one_vs_rest(X, group_labels, *, gene_names, n_jobs=1, chunk_size=None, apply_logfoldchanges_expm1=True)` and `wilcoxon_vs_reference(X, group_labels, *, group, reference, gene_names, n_genes=50, chunk_size=2000, apply_logfoldchanges_expm1=True)` — signatures + output columns unchanged; `n_jobs`/`chunk_size` kept for compatibility but the kernel no longer densifies or forks.

- [ ] **Step 1: Write the failing tests** — create `tests/test_wilcoxon.py`
```python
import pathlib
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_DIR))

import cluster  # noqa: E402


def _ref_z_p(x, y):
    """Reference tie-corrected two-sided MWU on group x vs group y (one gene)."""
    res = mannwhitneyu(x, y, alternative='two-sided', use_continuity=False,
                       method='asymptotic')
    # recover z from p (two-sided): |z| = isf(p/2)
    from scipy.stats import norm
    return res.statistic, res.pvalue


def test_kernel_matches_scipy_with_ties_sparse_eq_dense():
    rng = np.random.default_rng(0)
    # 60 cells x 5 genes, integer counts with many zeros (heavy ties at 0)
    dense = rng.poisson(0.4, size=(60, 5)).astype(np.float32)
    labels = np.array(['a'] * 25 + ['b'] * 35)
    out_dense = cluster._wilcoxon_sparse_stats(dense, labels, ['a', 'b'])
    out_sparse = cluster._wilcoxon_sparse_stats(sp.csr_matrix(dense), labels, ['a', 'b'])
    # sparse == dense
    for k in ('a', 'b'):
        np.testing.assert_allclose(out_dense[k][0], out_sparse[k][0], rtol=1e-4, atol=1e-4,
                                   equal_nan=True)
    # vs scipy reference (group 'a' one-vs-rest = a vs b), per gene, via p-value
    from scipy.stats import norm
    za = out_dense['a'][0]
    for g in range(5):
        x = dense[labels == 'a', g]
        y = dense[labels == 'b', g]
        _, p_ref = _ref_z_p(x, y)
        p_ours = 2.0 * norm.sf(np.abs(za[g]))
        np.testing.assert_allclose(p_ours, p_ref, rtol=1e-3, atol=1e-3)


def test_kernel_all_zero_gene_is_nonsignificant():
    dense = np.zeros((20, 3), dtype=np.float32)
    dense[:, 1] = 1.0  # constant nonzero gene also a full tie
    labels = np.array(['a'] * 10 + ['b'] * 10)
    out = cluster._wilcoxon_sparse_stats(dense, labels, ['a', 'b'])
    z = out['a'][0]
    # all-tie genes -> U at the mean -> z ~ 0 (or nan); p must be ~1
    from scipy.stats import norm
    p = 2.0 * norm.sf(np.abs(np.nan_to_num(z, nan=0.0)))
    assert np.all(p > 0.99)


def test_kernel_small_group_is_nan():
    dense = np.random.default_rng(1).poisson(0.5, size=(10, 4)).astype(np.float32)
    labels = np.array(['a'] + ['b'] * 9)   # group 'a' has 1 cell
    out = cluster._wilcoxon_sparse_stats(dense, labels, ['a', 'b'])
    assert np.all(np.isnan(out['a'][0]))


def test_one_vs_rest_columns_and_no_densify_path():
    rng = np.random.default_rng(2)
    X = sp.csr_matrix(rng.poisson(0.5, size=(40, 6)).astype(np.float32))
    labels = np.array(['g0'] * 13 + ['g1'] * 13 + ['g2'] * 14)
    df = cluster.wilcoxon_one_vs_rest(X, labels, gene_names=[f"G{i}" for i in range(6)])
    assert set(df.columns) == {'group', 'gene', 'scores', 'pvals', 'pvals_adj',
                               'logfoldchanges', 'mean_in', 'mean_out'}
    assert sorted(df['group'].unique()) == ['g0', 'g1', 'g2']
    assert len(df) == 18


def test_vs_reference_columns_and_topn():
    rng = np.random.default_rng(3)
    X = sp.csr_matrix(rng.poisson(0.5, size=(50, 8)).astype(np.float32))
    labels = np.array(['minor'] * 20 + ['core'] * 20 + ['other'] * 10)
    df = cluster.wilcoxon_vs_reference(X, labels, group='minor', reference='core',
                                       gene_names=[f"G{i}" for i in range(8)], n_genes=5)
    assert list(df.columns) == ['names', 'logfoldchanges', 'pvals', 'pvals_adj',
                                'scores', 'direction']
    assert len(df) == 5
    assert df['scores'].is_monotonic_decreasing
```

- [ ] **Step 2: Run to verify it fails**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_wilcoxon.py -v
```
Expected: `AttributeError: module 'cluster' has no attribute '_wilcoxon_sparse_stats'`.

- [ ] **Step 3a: Add the shared kernel** — insert `_wilcoxon_sparse_stats` in `cluster.py` just above `def wilcoxon_one_vs_rest` (line 924):
```python
def _wilcoxon_sparse_stats(X, group_labels, groups):
    """Tie-corrected one-vs-rest Mann-Whitney across all genes for ``groups``,
    on sparse (or dense) X WITHOUT densifying. Returns {group: (z, mean_in, mean_out)}
    of float32 (n_genes,) arrays. presto-style: rank each gene once (only its
    nonzeros), share rank sums across groups via Gᵀ@ranks sparse matmuls.

    Assumes X is non-negative (log-normalized counts) so all zeros are the
    smallest values. z is NaN where a group has <2 cells or n-2<n1.
    """
    import scipy.sparse as sp
    from scipy.stats import rankdata

    Xcsc = sp.csc_matrix(X).astype(np.float64)
    Xcsc.eliminate_zeros()
    N, n_genes = Xcsc.shape
    indptr, indices, data = Xcsc.indptr, Xcsc.indices, Xcsc.data

    # Per-gene: global ranks of the nonzeros, zero-block rank r0, tie term T.
    r0 = np.empty(n_genes, dtype=np.float64)
    Tcorr = np.empty(n_genes, dtype=np.float64)
    rnz_data = np.empty_like(data)
    for g in range(n_genes):
        s, e = indptr[g], indptr[g + 1]
        nz = e - s
        nzeros = N - nz
        r0[g] = (nzeros + 1) / 2.0
        Tg = float(nzeros) ** 3 - float(nzeros)          # zero-block tie group
        if nz:
            vals = data[s:e]
            rr = rankdata(vals, method='average')        # ranks 1..nz among nonzeros
            rnz_data[s:e] = rr + nzeros                  # global ranks
            _, counts = np.unique(vals, return_counts=True)
            tt = counts[counts > 1].astype(np.float64)
            Tg += float(np.sum(tt ** 3 - tt))
        Tcorr[g] = Tg

    Rnz = sp.csc_matrix((rnz_data, indices, indptr), shape=(N, n_genes))
    Bnz = sp.csc_matrix((np.ones_like(data), indices, indptr), shape=(N, n_genes))

    groups = list(groups)
    col_of = {k: i for i, k in enumerate(groups)}
    rows = np.arange(N)
    cols = np.fromiter((col_of[l] for l in group_labels), dtype=np.int64, count=N)
    G = sp.csr_matrix((np.ones(N), (rows, cols)), shape=(N, len(groups)))
    GT = G.T.tocsr()                                     # (K x N)
    n1 = np.asarray(G.sum(axis=0)).ravel()               # (K,)

    S_nz = np.asarray((GT @ Rnz).todense())              # (K x n_genes) rank sums of nonzeros
    C_nz = np.asarray((GT @ Bnz).todense())              # (K x n_genes) nonzero counts
    gsum = np.asarray((GT @ Xcsc).todense())             # (K x n_genes) expression sums
    total = np.asarray(Xcsc.sum(axis=0)).ravel()         # (n_genes,)

    out = {}
    denom = N * (N - 1) if N > 1 else 1
    for i, k in enumerate(groups):
        n1k = float(n1[i]); n2k = float(N - n1[i])
        if n1k < 2 or n2k < 2:
            z = np.full(n_genes, np.nan, dtype=np.float32)
        else:
            R1 = S_nz[i] + r0 * (n1k - C_nz[i])
            U1 = R1 - n1k * (n1k + 1) / 2.0
            var = (n1k * n2k / 12.0) * ((N + 1) - Tcorr / denom)
            with np.errstate(invalid='ignore', divide='ignore'):
                z = (U1 - n1k * n2k / 2.0) / np.sqrt(var)
            z = np.where(var > 0, z, np.nan).astype(np.float32)
        mean_in = (gsum[i] / n1k if n1k else np.zeros(n_genes)).astype(np.float32)
        mean_out = ((total - gsum[i]) / n2k if n2k else np.zeros(n_genes)).astype(np.float32)
        out[k] = (z, mean_in, mean_out)
    return out
```

- [ ] **Step 3b: Rewrite `wilcoxon_one_vs_rest`** — replace its body (924–1011) with:
```python
def wilcoxon_one_vs_rest(
    X,
    group_labels,
    *,
    gene_names,
    n_jobs: int = 1,
    chunk_size: int | None = None,
    apply_logfoldchanges_expm1: bool = True,
) -> pd.DataFrame:
    """One-vs-rest tie-corrected Mann-Whitney U, sparse and vectorised across all
    groups (presto-style, no densify). ``n_jobs``/``chunk_size`` accepted for
    backward compatibility but unused — the kernel is single-pass over sparse X.

    Returns long: group, gene, scores (z), pvals, pvals_adj, logfoldchanges,
    mean_in, mean_out.
    """
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    group_labels = np.asarray(group_labels)
    groups = sorted(set(group_labels.tolist()))
    stats = _wilcoxon_sparse_stats(X, group_labels, groups)
    frames = []
    for k in groups:
        z, mi, mo = stats[k]
        with np.errstate(invalid='ignore'):
            p = 2.0 * norm.sf(np.abs(z))
        p = np.where(np.isnan(z), 1.0, p)
        padj = multipletests(p, method='fdr_bh')[1]
        if apply_logfoldchanges_expm1:
            lfc = np.log2((np.expm1(mi) + 1e-9) / (np.expm1(mo) + 1e-9))
        else:
            lfc = mi - mo
        frames.append(pd.DataFrame({
            'group': k, 'gene': gene_names,
            'scores': np.nan_to_num(z, nan=0.0).astype(np.float32),
            'pvals': p, 'pvals_adj': padj,
            'logfoldchanges': lfc, 'mean_in': mi, 'mean_out': mo,
        }))
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 3c: Delete the dead fork-pool** — remove `_wilcoxon_chunked` (the old 1014–1094 body) and the `_WILCOXON_WORKER` slot + `_wilcoxon_worker_call` (1097–1100). (The new `wilcoxon_one_vs_rest` no longer calls them.)

- [ ] **Step 3d: Rewrite `wilcoxon_vs_reference`** — replace its body (195–268) with the kernel-backed 2-group version (same output columns):
```python
def wilcoxon_vs_reference(
    X,
    group_labels,
    *,
    group: str,
    reference: str,
    gene_names,
    n_genes: int = 50,
    chunk_size: int = 2000,
    apply_logfoldchanges_expm1: bool = True,
) -> pd.DataFrame:
    """Tie-corrected 2-group Mann-Whitney: ``group`` vs ``reference`` (sparse, no
    densify). ``chunk_size`` accepted for compatibility but unused. Returns the
    top ``n_genes`` by score desc: names / logfoldchanges / pvals / pvals_adj /
    scores / direction.
    """
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    group_labels = np.asarray(group_labels)
    sel = (group_labels == group) | (group_labels == reference)
    stats = _wilcoxon_sparse_stats(X[sel], group_labels[sel], [group, reference])
    z, mean_in, mean_out = stats[group]
    with np.errstate(invalid='ignore'):
        p = 2.0 * norm.sf(np.abs(z))
    p = np.where(np.isnan(z), 1.0, p)
    padj = multipletests(p, method='fdr_bh')[1]
    if apply_logfoldchanges_expm1:
        lfc = np.log2((np.expm1(mean_in) + 1e-9) / (np.expm1(mean_out) + 1e-9))
    else:
        lfc = mean_in - mean_out
    df = pd.DataFrame({
        'names':          list(gene_names),
        'logfoldchanges': lfc,
        'pvals':          p,
        'pvals_adj':      padj,
        'scores':         np.nan_to_num(z, nan=0.0),
    })
    df['direction'] = np.where(df['logfoldchanges'] > 0, 'up', 'down')
    return (df.sort_values('scores', ascending=False)
              .head(n_genes).reset_index(drop=True))
```

- [ ] **Step 4: Run kernel tests + the existing suite**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_wilcoxon.py -v
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/ -q
```
Expected: `test_wilcoxon.py` 5 passed; full suite still `28 passed` plus the new files.

- [ ] **Step 5: Commit**
```bash
git add cluster.py tests/test_wilcoxon.py
git commit -m "perf(cluster): sparse tie-corrected rank-once Wilcoxon kernel; drop densify+fork-pool"
```

---

## Task 3: LLM retry/backoff + timeout bump (`diagnosis.py`, `annotate.py`)

**Files:**
- Modify: `diagnosis.py` (`LLMDiagnosisEngine`, `ArkChatClient`/`from_env`/`make_chat_client`/`make_diagnosis_engine` timeout defaults), `annotate.py` (`LLMNamingEngine`, `NarrativeEngine`, factories)
- Test: append to `tests/test_diagnosis_llm.py` and `tests/test_annotate.py`

**Interfaces:**
- Consumes `parallel.with_retry`.
- Produces: `LLMDiagnosisEngine(client, *, mode='llm', fallback_to_rule=True, diagnosis_roles=None, llm_retries=3)`; `LLMNamingEngine(client, *, local=None, fallback_to_local=True, llm_retries=3)`; `NarrativeEngine(client, *, llm_retries=3)`; `make_naming_engine(..., llm_retries=3)`. Shared client `timeout` default `120`.

- [ ] **Step 1: Write failing tests** — append to `tests/test_diagnosis_llm.py`:
```python
def test_diagnosis_retries_then_succeeds():
    calls = {"n": 0}
    def flaky(s, u):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("read timeout")
        return json.dumps({"likely_cause": ALLOWED_CAUSES[0], "confidence": 0.7,
                           "rationale": "r"})
    from diagnosis import LLMDiagnosisEngine, MinorEvidence
    eng = LLMDiagnosisEngine(CallableChatClient(flaky, model="m"), mode="llm",
                             llm_retries=3)
    r = eng.diagnose(MinorEvidence(parent_cluster="0", subcluster="c0_1",
                     reference_subcluster="c0_0", minor_umap_label="u1",
                     main_umap_label="u0", n_cells=10, frac_of_parent=0.1))
    assert r.diagnosis_source == "llm"
    assert calls["n"] == 3


def test_diagnosis_retry_exhaustion_falls_back_to_rule():
    def always_timeout(s, u):
        raise TimeoutError("read timeout")
    from diagnosis import LLMDiagnosisEngine, MinorEvidence
    eng = LLMDiagnosisEngine(CallableChatClient(always_timeout, model="m"), mode="llm",
                             llm_retries=2)
    r = eng.diagnose(MinorEvidence(parent_cluster="0", subcluster="c0_1",
                     reference_subcluster="c0_0", minor_umap_label="u1",
                     main_umap_label="u0", n_cells=10, frac_of_parent=0.1))
    assert r.diagnosis_source == "rule-fallback"
    assert r.error is not None
```
Append to `tests/test_annotate.py`:
```python
def test_naming_retries_then_succeeds():
    calls = {"n": 0}
    def flaky(s, u):
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("read timeout")
        return json.dumps({"cell_type": "T cell", "confidence": 0.9, "rationale": "r",
                           "markers_used": ["CD3D"]})
    eng = annotate.LLMNamingEngine(CallableChatClient(flaky, model="m"), llm_retries=3)
    r = eng.name(_evi(["CD3D", "CD3E"]))
    assert r.source == "llm" and r.cell_type == "T cell"
    assert calls["n"] == 2
```

- [ ] **Step 2: Run to verify it fails**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_diagnosis_llm.py tests/test_annotate.py -k "retry or retries" -v
```
Expected: failures — `LLMDiagnosisEngine`/`LLMNamingEngine` don't accept `llm_retries` yet (TypeError) and don't retry.

- [ ] **Step 3a: `diagnosis.py` — import + engine retry.** Add near the top imports (after the llm_client dual-import):
```python
try:
    from .parallel import with_retry
except ImportError:
    from parallel import with_retry
```
In `LLMDiagnosisEngine.__init__`, add `llm_retries=3`:
```python
    def __init__(self, client, *, mode='llm', fallback_to_rule=True,
                 diagnosis_roles=None, llm_retries=3):
        ...
        self.llm_retries = llm_retries
```
In `LLMDiagnosisEngine.diagnose`, wrap the `call_structured` in the existing `try`:
```python
        try:
            return with_retry(
                lambda: call_structured(
                    self.client, system_prompt, user_prompt,
                    lambda data: _diagnosis_from_dict(
                        data, rule_baseline=evidence.rule_baseline,
                        mode=self.mode, model=self.model)),
                retries=self.llm_retries, backoff=0.5, jitter=0.25,
                exceptions=(Exception,))
        except Exception as e:
            ... (unchanged fallback)
```
Bump timeout defaults `60→120` in `ArkChatClient.__init__`, `ArkChatClient.from_env`, `make_chat_client`, and `make_diagnosis_engine` (each `timeout=60` → `timeout=120`); thread `llm_retries=3` through `make_diagnosis_engine` into `LLMDiagnosisEngine`.

- [ ] **Step 3b: `annotate.py` — import + engine retry.** Add the dual-import for `with_retry` (mirroring the `llm_client` one). In `LLMNamingEngine.__init__` add `llm_retries=3` (store it); wrap its `call_structured` in `with_retry(..., retries=self.llm_retries, backoff=0.5, jitter=0.25, exceptions=(Exception,))`. Same for `NarrativeEngine.__init__`/`narrate`. Add `llm_retries=3` to `make_naming_engine` and pass it to `LLMNamingEngine`.

- [ ] **Step 4: Run to verify pass**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_diagnosis_llm.py tests/test_annotate.py -v
```
Expected: all pass (prior + 3 new).

- [ ] **Step 5: Commit**
```bash
git add diagnosis.py annotate.py tests/test_diagnosis_llm.py tests/test_annotate.py
git commit -m "feat(llm): retry/backoff before fallback; default client timeout 120s"
```

---

## Task 4: LLM thread concurrency in the stages (`annotate.py`, `pipeline.py`)

**Files:**
- Modify: `annotate.py` (`run_naming_stage`, `run_narrative_stage` add `max_workers`), `pipeline.py` (diagnosis stage threads minors; thread `llm_concurrency`)
- Test: append to `tests/test_annotate.py`

**Interfaces:**
- `run_naming_stage(..., max_workers=1)` / `run_narrative_stage(..., max_workers=1)` — the per-cluster compute loop runs through `parallel.thread_map(..., max_workers=max_workers)`; reassembly stays serial after.

- [ ] **Step 1: Write failing test** — append to `tests/test_annotate.py`:
```python
def test_naming_stage_parallel_matches_serial(tmp_path):
    import pandas as pd
    def _setup(root):
        clusters = root / "clusters"; canonical = root / "canonical_markers"
        canonical.mkdir(parents=True)
        for p, gene in [("0", "CD3D"), ("1", "LYZ")]:
            (clusters / f"c{p}").mkdir(parents=True)
            pd.DataFrame({"group": [f"c{p}_0"], "rank": [0], "gene": [gene],
                          "logfoldchanges": [3.0], "pvals": [1e-9], "pvals_adj": [1e-8],
                          "scores": [20.0]}).to_csv(canonical / f"markers_c{p}_0.tsv",
                                                    sep="\t", index=False)
        return clusters, canonical
    payload = json.dumps({"cell_type": "T cell", "confidence": 0.9, "rationale": "r",
                          "markers_used": ["CD3D"]})
    def eng():
        return annotate.make_naming_engine(client=CallableChatClient(lambda s, u: payload, model="m"))
    a = tmp_path / "a"; b = tmp_path / "b"
    ca, cana = _setup(a); cb, canb = _setup(b)
    annotate.run_naming_stage(clusters_dir=ca, canonical_dir=cana,
        core_names_path=a / "core_names.tsv", parents=["0", "1"], engine=eng(), max_workers=1)
    annotate.run_naming_stage(clusters_dir=cb, canonical_dir=canb,
        core_names_path=b / "core_names.tsv", parents=["0", "1"], engine=eng(), max_workers=4)
    assert (a / "core_names.tsv").read_text() == (b / "core_names.tsv").read_text()
```

- [ ] **Step 2: Run to verify it fails**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_annotate.py -k parallel_matches_serial -v
```
Expected: `TypeError: run_naming_stage() got an unexpected keyword argument 'max_workers'`.

- [ ] **Step 3a: `annotate.py`** — add the `with_retry`/`thread_map` dual-import for `parallel`. Add `max_workers=1` to `run_naming_stage` and `run_narrative_stage`; replace their per-cluster `for parent in parents:` compute loop with:
```python
    def _do(parent):
        if not forced and _naming_current(clusters_dir / f"c{parent}", model=model):
            return f'naming:c{parent}'        # skip label
        evidence = build_core_evidence(parent, canonical_dir / f"markers_c{parent}_0.tsv",
                                       n_cells=core_sizes.get(str(parent), 0), hint=hint)
        write_naming_artifacts(clusters_dir / f"c{parent}", evidence, engine.name(evidence))
        return None
    skipped = [s for s in thread_map(_do, parents, max_workers=max_workers) if s]
    # ...then the existing reassembly of core_names.tsv (unchanged)
```
(Analogous for `run_narrative_stage` with `narrative_current`/`build`/`engine.narrate` and `'narrative:c{parent}'`.)

- [ ] **Step 3b: `pipeline.py`** — add `llm_concurrency=8` to `run_dissect_pipeline`; pass `max_workers=llm_concurrency` to `run_naming_stage`/`run_narrative_stage`. In `_apply_diagnosis_to_cluster_panel`, thread the per-minor diagnosis: collect `rows` via `thread_map(_diagnose_row, panel.iterrows()-as-list, max_workers=llm_concurrency)` (each `_diagnose_row` builds evidence + `engine.diagnose` + `write_diagnosis_artifacts`, returns the row dict), then write the panel once after. Thread `llm_concurrency` from `run_dissect_pipeline` into `_run_diagnosis_stage` → `_apply_diagnosis_to_cluster_panel`. Import `thread_map` from `.parallel`.

- [ ] **Step 4: Run to verify pass**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_annotate.py -v
```
Expected: all pass incl. the new parallel-equivalence test.

- [ ] **Step 5: Commit**
```bash
git add annotate.py pipeline.py tests/test_annotate.py
git commit -m "perf(llm): thread-pool the naming/narrative/diagnosis stages (llm_concurrency)"
```

---

## Task 5: DEG cross-cluster process parallelism (`pipeline.py`)

**Files:**
- Modify: `pipeline.py` — refactor `_dissect_one` off `_DISSECT_CTX`; `process_map` the dissect loop; repurpose `n_jobs`.
- Test: append to `tests/test_pipeline_parallel.py` (create) — a minimal real `dissect_one_cluster` path is scanpy-heavy; instead test the new pure helper `_dissect_subset_task` in isolation.

**Interfaces:**
- `_dissect_one_subset(parent, subset, ctx)` — pure function: runs `dissect_one_cluster(subset, ...)` + `_persist_cluster(...)` using only its args (no module global), returns `str(parent)` or raises.

- [ ] **Step 1: Write failing test** — create `tests/test_pipeline_parallel.py`:
```python
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].parent))
from standissect import pipeline  # noqa: E402   (scanpy import -> runs on this compute node)


def test_dissect_helper_signature_is_global_free():
    # the refactored helper must accept explicit subset+ctx (no _DISSECT_CTX)
    import inspect
    params = inspect.signature(pipeline._dissect_one_subset).parameters
    assert 'subset' in params and 'ctx' in params
```

- [ ] **Step 2: Run to verify it fails**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_pipeline_parallel.py -v
```
Expected: `AttributeError: module ... has no attribute '_dissect_one_subset'`.

- [ ] **Step 3: Implement** — in `pipeline.py`:
  - Replace `_DISSECT_CTX` + `_dissect_one(parent)` with `_dissect_one_subset(parent, subset, ctx)` that calls `dissect_one_cluster(subset, cluster_col=ctx['cluster_col'], parent=str(parent), umap_label_col=ctx['umap_label_col'], crosstab_row=ctx['crosstab'].loc[parent], size_rank_name=ctx['srn_by_parent'][parent], cat_cols=ctx['cat_cols'], qc_cols=ctx['qc_cols'], top_n_deg=ctx['top_n_deg'], deg_layer=ctx['deg_layer'], min_subcluster_size=ctx['min_subcluster_size'])` then `_persist_cluster(subset, res, cdir=ctx['lay'].cluster_dir(parent), umap_key=ctx['umap_key'], cluster_col=ctx['cluster_col'], size_rank_name=ctx['srn_by_parent'][parent])`; returns `str(parent)`.
  - Add a top-level picklable task `_dissect_task(args)` unpacking `(parent, subset, ctx)` → `_dissect_one_subset(...)` so the spawn pool can call it.
  - In the dissect stage of `run_dissect_pipeline`: build `ctx` (the small scalars/cols, **not** the full adata), build `subsets = {p: adata[adata.obs[cluster_col].astype(str)==str(p)] for p in todo}`, then:
    ```python
    deg_jobs = max(1, min(len(todo), os.cpu_count() or 1, n_jobs))
    try:
        process_map(_dissect_task, [(p, subsets[p], ctx) for p in todo], max_workers=deg_jobs)
    except Exception as exc:
        print(f"[pipeline] dissect process pool failed ({exc}); recomputing serially", flush=True)
        for p in todo:
            _dissect_one_subset(p, subsets[p], ctx)
    ```
    Per-cluster worker failures are caught and that cluster recomputed serially (robust). Import `process_map` from `.parallel`, `os` at top.

- [ ] **Step 4: Run to verify pass + import smoke**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_pipeline_parallel.py -v
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -c "from standissect.pipeline import _dissect_one_subset, _dissect_task; print('ok')"
```
Expected: pass; `ok`.

- [ ] **Step 5: Commit**
```bash
git add pipeline.py tests/test_pipeline_parallel.py
git commit -m "perf(dissect): cross-cluster process_map over per-cluster subsets (n_jobs)"
```

---

## Task 6: CLI knobs, params.json, end-to-end equivalence + full suite

**Files:**
- Modify: `cli.py`, `pipeline.py` (params.json + pass-throughs)
- Test: append to `tests/test_cli.py`; run the Marrow e2e equivalence check manually (documented).

- [ ] **Step 1: Write failing CLI tests** — append to `tests/test_cli.py`:
```python
def test_cli_concurrency_defaults():
    a = build_parser().parse_args(["run", "x.h5ad", "--cluster-col", "leiden",
                                   "--output-dir", "o"])
    assert a.llm_concurrency == 8
    assert a.llm_retries == 3
    assert a.ark_timeout == 120


def test_cli_concurrency_overrides():
    a = build_parser().parse_args(["run", "x.h5ad", "--cluster-col", "leiden",
        "--output-dir", "o", "--llm-concurrency", "4", "--llm-retries", "1",
        "--ark-timeout", "90", "--n-jobs", "2"])
    assert (a.llm_concurrency, a.llm_retries, a.ark_timeout, a.n_jobs) == (4, 1, 90, 2)
```

- [ ] **Step 2: Run to verify it fails**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_cli.py -k concurrency -v
```
Expected: `AttributeError: 'Namespace' object has no attribute 'llm_concurrency'`.

- [ ] **Step 3a: `cli.py`** — in `_add_common_run_args`'s `diag` group add:
```python
    diag.add_argument('--llm-concurrency', type=int, default=8,
                      help='Concurrent ARK calls for diagnosis/naming/narrative. Default: 8.')
    diag.add_argument('--llm-retries', type=int, default=3,
                      help='Retries (exp backoff + jitter) before fallback. Default: 3.')
    diag.add_argument('--ark-timeout', type=int, default=120,
                      help='Per-call ARK timeout (seconds). Default: 120.')
```
Update the `--n-jobs` help to "DEG process-pool size (cross-cluster parallelism)." In `run_cmd`, pass `llm_concurrency=args.llm_concurrency, llm_retries=args.llm_retries, diagnosis_timeout=args.ark_timeout, n_jobs=args.n_jobs` to `run_dissect_pipeline`.

- [ ] **Step 3b: `pipeline.py`** — ensure `run_dissect_pipeline` accepts `llm_concurrency=8, llm_retries=3` (and existing `diagnosis_timeout` now default 120, `n_jobs` repurposed); pass `llm_retries` into `make_diagnosis_engine`/`make_naming_engine`/`NarrativeEngine`. Add to the `params.json` dict: `'llm_concurrency': llm_concurrency, 'llm_retries': llm_retries, 'diagnosis_timeout': diagnosis_timeout, 'n_jobs': n_jobs` (effective).

- [ ] **Step 4: Run CLI tests + full suite**
```bash
/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/ -q
```
Expected: all pass (28 original + new: parallel 7, wilcoxon 5, retry 3, naming-parallel 1, pipeline_parallel 1, cli 2).

- [ ] **Step 5: End-to-end equivalence smoke (real, local)** — confirm parallel == serial on the preprocessed Marrow file:
```bash
cd /scratch/users/chensj16/projects
PY=/scratch/users/chensj16/venvs/dl2025/.venv/bin/python
export MPLCONFIGDIR=/scratch/users/chensj16/.cache/mpl NUMBA_CACHE_DIR=/scratch/users/chensj16/.cache/numba
# serial
$PY -m standissect run /scratch/users/chensj16/standissect_test/marrow_pp.h5ad \
  --cluster-col cell_ontology_class --output-dir /scratch/users/chensj16/standissect_test/marrow_serial \
  --umap-key X_umap --diagnosis-mode rule --n-jobs 1 --llm-concurrency 1 --no-report
# parallel (rule mode keeps it LLM-free + deterministic for the diff)
$PY -m standissect run /scratch/users/chensj16/standissect_test/marrow_pp.h5ad \
  --cluster-col cell_ontology_class --output-dir /scratch/users/chensj16/standissect_test/marrow_par \
  --umap-key X_umap --diagnosis-mode rule --n-jobs 4 --llm-concurrency 8 --no-report
for f in cell_ontology_class/panel.tsv cell_ontology_class/canonical_markers/deg_long.tsv; do
  diff <(sort /scratch/users/chensj16/standissect_test/marrow_serial/$f) \
       <(sort /scratch/users/chensj16/standissect_test/marrow_par/$f) && echo "IDENTICAL: $f"
done
```
Expected: `IDENTICAL: ...` for both (DEG/panel identical serial vs parallel). Note: with the new tie-corrected kernel these differ from the *pre-Task-2* run by design.

- [ ] **Step 6: Commit**
```bash
git add cli.py pipeline.py tests/test_cli.py
git commit -m "feat(cli): --llm-concurrency/--llm-retries/--ark-timeout; params.json + wiring"
```

---

## Self-Review

**Spec coverage:** §3 parallel.py → Task 1. §4 fast kernel (sparse, tie-correct, rank-once, drop fork-pool) → Task 2. §5 DEG process parallelism (refactor `_dissect_one`, `process_map`, subset marshaling, serial-fallback, `n_jobs`) → Task 5. §6 LLM threads → Task 4. §7 retry + timeout 120 → Task 3. §8 knobs/params → Task 6. §9 determinism → Task 4/5 (reassembly after pools) + Task 6 e2e diff. §10 testing → Tasks 1–6. ✓

**Placeholder scan:** none — every code step shows complete code or an exact anchored change; every run step has a command + expected result. The Task 3b/4a/5 "analogous"/prose edits name the exact functions, params, and the pattern with full reference code in the sibling step.

**Type consistency:** `_wilcoxon_sparse_stats(X, group_labels, groups) -> {group:(z,mean_in,mean_out)}` used identically by both public fns (Task 2); `with_retry(retries,backoff,jitter,exceptions)` signature matches its calls in Tasks 3/parallel; `thread_map(fn, items, *, max_workers)` / `process_map(...)` calls in Tasks 4/5 match Task 1; `llm_concurrency`/`llm_retries`/`n_jobs`/`diagnosis_timeout` names consistent across pipeline/cli/engines.

**Note (scope):** Task 5's unit test only asserts the global-free signature (a real `dissect_one_cluster` run needs scanpy + an AnnData); the **behavioral** parallel==serial guarantee for DEG is covered by Task 6 Step 5's real Marrow diff (`deg_long.tsv`/`panel.tsv` identical for `n_jobs=1` vs `4`).
