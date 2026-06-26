# DEG + LLM concurrency & fast Wilcoxon — design spec

**Date:** 2026-06-25
**Branch:** `feat/deg-llm-concurrency` (committed here; `main` untouched until the user merges)
**Scope:** Three coupled efficiency/correctness improvements to standissect — (1) a faster, sparse, **tie-corrected** Wilcoxon kernel; (2) **cross-cluster process parallelism** for the dissect (per-cluster DEG) stage; (3) **thread-pool concurrency + retry/backoff** for the LLM stages (diagnosis, naming, narrative). A new stdlib-only `parallel.py` holds the shared primitives.

## 1. Goal & background

standissect's DEG is **already scanpy-free** — `cluster.py` implements its own Wilcoxon (`wilcoxon_one_vs_rest` for K-group canonical markers, `wilcoxon_vs_reference` for 2-group minor-vs-core), with a "faster scanpy replacement" comment. So the dependency to shed is not scanpy; the real problems, confirmed by reading the code and a real 1.39M-cell attempt, are:

- **Densification:** both kernels do `X.toarray()` (full or gene-chunked) before `scipy.stats.rankdata`. For sparse scRNA this blows up memory and wastes time ranking millions of zeros (the 1.39M-cell kidney atlas was impractical on a 5-core/105 GB node largely for this reason).
- **No tie-correction in the variance:** `sigma = sqrt(n1·n2·(N+1)/12)`. On zero-inflated data (huge ties at 0) the true variance is smaller, so the current `z` is biased **conservative** — a correctness gap, not just speed.
- **Serial everything:** dissect loops clusters serially (`for p in todo: _dissect_one(p)`, `n_jobs` reserved/unused); the LLM stages (diagnosis/naming/narrative) loop serially over clusters/minors, each a blocking ARK HTTP call. On the Marrow test some diagnosis calls hit ARK **read timeouts and fell back to rule** — serial latency + no retry.
- **Latent deadlock:** `_wilcoxon_chunked`'s parallel path uses `mp.get_context('fork')`, the fork-pool the pipeline comment says deadlocks after threaded BLAS (hence it runs `n_jobs=1`).

The efficient method is [presto](https://github.com/immunogenomics/presto) (Korsunsky et al., bioRxiv 653253): rank **once** across all cells (standissect already does this), **without densifying** (zeros share one tie-rank; rank only nonzeros → O(nnz log nnz)), with a **tie-corrected** variance, computing all groups via shared rank sums. presto is R, but the algorithm is ~tens of lines of scipy/numpy on a CSC matrix — **no new dependency**.

## 2. Design decisions (user-approved)

1. **Fast Wilcoxon kernel** rewrite (sparse + tie-corrected + rank-once-all-groups) for both `wilcoxon_one_vs_rest` and `wilcoxon_vs_reference`; delete the fork-pool path.
2. **Keep cross-cluster process parallelism** for dissect (the kernel speedup alone might suffice, but the user wants both): `process_map` over clusters.
3. **LLM stages → thread pool** (`thread_map`) with bounded `llm_concurrency` (default 8).
4. **LLM retry/backoff** (`with_retry`, default **3 retries**, exp backoff **+ jitter**) before the existing rule/local fallback, plus a bump of the shared ARK client default **timeout 60→120 s** (Marrow showed 60 s timing out).
5. **Shared primitives in a new stdlib-only `parallel.py`**.
6. **No new pip dependencies** (scipy/numpy/stdlib only; `statsmodels.multipletests` already used — kept).

## 3. `parallel.py` (new, stdlib only)

`concurrent.futures`, `multiprocessing`, `time`, `os`. Imports nothing from standissect → no cycle; imported by `cluster.py`/`diagnosis.py`/`annotate.py`/`pipeline.py`.

- `thread_map(fn, items, *, max_workers) -> list` — run `fn(item)` over `items` on a bounded `ThreadPoolExecutor`, results in **input order**. `max_workers<=1` or `len(items)<=1` → plain serial loop (no pool). Exceptions propagate per item (callers pass never-raising fns).
- `process_map(fn, items, *, max_workers) -> list` — bounded `ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'))`, results in input order. **BLAS pinning:** `process_map` sets `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1` in `os.environ` **before constructing the pool** (so spawned children inherit single-thread BLAS at their import time — a worker `initializer` would run too late, after numpy/BLAS are already loaded), then restores the parent's prior env after the pool closes. `max_workers<=1` or `len(items)<=1` → serial loop (no pool, no pickling). A worker exception is re-raised to the caller tagged with the item index so the stage can fall back.
- `with_retry(fn, *, retries=3, backoff=0.5, jitter=0.25, exceptions=(Exception,)) -> result` — call `fn()`; on a listed exception, sleep `backoff * 2**attempt + random.uniform(0, jitter)` and retry up to `retries` times (**up to 4 total attempts**); re-raise the last exception if all fail. The jitter desynchronizes concurrent retriers so `llm_concurrency` threads don't back off in lockstep and re-trigger 429s. (`time.sleep` + stdlib `random`.)

## 4. Fast Wilcoxon kernel (`cluster.py`)

Rewrite the shared ranking core so both entry points operate on **CSC sparse without densifying**, with tie correction. Public signatures and the returned long-DataFrame columns (`group, gene, scores, pvals, pvals_adj, logfoldchanges, mean_in, mean_out`) are **unchanged** — only the internals.

**Per-gene sparse tie-corrected average ranks** (log-normalized X ≥ 0, so all zeros are the smallest values):
- For column g with `nz` nonzeros and `nzeros = N - nz`: the zero entries occupy ranks `1..nzeros` and share the average rank `r0_g = (nzeros + 1) / 2`. Nonzero values are average-tie-ranked among themselves and offset by `nzeros`. Only the nonzeros are sorted → O(nnz log nnz), no dense column.
- **Tie-correction accumulator** per gene: `T_g = Σ(t³ − t)` over all tie groups = the zero block (`t = nzeros`) plus tie groups among the nonzeros.

**All-groups rank sums via sparse algebra** (no per-group / per-cluster loop). With group-indicator `G` (N × K, sparse 0/1):
- `R_nz = ` sparse matrix of nonzero average ranks (same sparsity pattern as X).
- per-group nonzero-rank sums: `Gᵀ @ R_nz` → (K × n_genes).
- per-group nonzero **counts**: `Gᵀ @ (X != 0)` → (K × n_genes).
- per-group rank sum: `R1[k,g] = (Gᵀ@R_nz)[k,g] + r0_g · (n_k − count[k,g])` (the zero cells in group k each contribute `r0_g`).
- `U1 = R1 − n_k(n_k+1)/2`; **tie-corrected** `sigma_g² = (n1·n2/12)·[(N+1) − T_g/(N(N−1))]`; `z = (U1 − n1·n2/2)/sigma_g`.
- `mean_in`/`mean_out` from `Gᵀ@X` group sums (sparse) → logFC as today (`apply_logfoldchanges_expm1`).

`wilcoxon_vs_reference` is the K=2 case on the parent-cluster subset (minor vs reference core) — same core, two columns of `G`.

**Removed:** the `toarray()` densify paths and the `mp.get_context('fork')` pool + `_WILCOXON_WORKER` slot. The kernel is now fast enough serial; gene-chunking is retained only as an optional memory cap (operating on sparse column blocks, no densify).

**Correctness reference (tests):** on a small dense matrix, the new `z`/`pvals` match `scipy.stats.mannwhitneyu(x, y, alternative='two-sided', use_continuity=False)` with tie correction, including all-zero genes, heavy-tie columns, and a group with <2 cells (→ NaN/p=1). Sparse and dense inputs give identical results.

## 5. DEG cross-cluster parallelism (`pipeline.py`)

- Refactor `_dissect_one(parent)` to take explicit args instead of reading the module global `_DISSECT_CTX`: `_dissect_one(parent, *, adata_subset, ctx)` where `adata_subset = adata[adata.obs[cluster_col]==parent]` (carries the in-memory partition labels + `obsm[umap_key]`), and `ctx` holds the small scalars/cols. `_persist_cluster` already works on a per-parent view.
- The dissect loop becomes: build the per-cluster subsets in the parent, then `process_map(_dissect_one_task, todo, max_workers=n_jobs)`. Each worker computes DEG (fast sparse kernel) + persists its own `clusters/cN/*` files (independent paths, no contention) + returns a status. `n_jobs<=1` → serial (unchanged behavior).
- **Marshaling:** per-cluster **sparse** subset is pickled to the worker (small with the new kernel). Peak memory ≈ resident adata + `n_jobs × subset`. Default `n_jobs = min(n_clusters, os.cpu_count() or 1, 8)`.
- A worker failure for a cluster is caught, logged, and that cluster is recomputed **serially** in the parent (robust degrade), so a single bad worker never aborts the run.
- The global panel/qc reassembly (`_concat_tsvs`) stays after the pool — output identical to serial.

## 6. LLM thread concurrency

- `annotate.run_naming_stage` / `run_narrative_stage`: the per-cluster compute loop → `thread_map(_do_one, todo, max_workers=llm_concurrency)`; the global `core_names.tsv`/`narratives.tsv` reassembly stays serial after the pool (order-independent output).
- diagnosis stage: `_apply_diagnosis_to_cluster_panel` runs its per-minor `engine.diagnose` calls via `thread_map(..., max_workers=llm_concurrency)`, then writes the panel once (after). Clusters themselves may stay serial (the minors are the parallel axis); the panel rewrite per cluster is unaffected.
- Safe because engines are never-raise and the ARK/`OpenAICompatClient` is stateless (each call its own request) → thread-safe. `llm_concurrency<=1` → serial.

## 7. LLM retry/backoff (`diagnosis.py`, `annotate.py`)

Each engine wraps its `call_structured(...)` invocation:
`with_retry(lambda: call_structured(client, system, user, parse), retries=llm_retries, backoff=0.5, jitter=0.25, exceptions=(LLMUnavailable,))`,
inside the existing `try/except` so that after retries are exhausted the current rule/local fallback fires. `llm_retries` default **3** (up to 4 attempts), threaded from the pipeline; `0` = no retry (current behavior). Retries cover read-timeout / connection-reset / 429 / 5xx (all surface as `LLMUnavailable`); a malformed-JSON reply is also re-asked (often self-heals). The shared client's per-call `timeout` default is bumped 60→120 s (see §8).

## 8. Knobs / CLI

- `run_dissect_pipeline` new params: `llm_concurrency=8`, `llm_retries=3`; `n_jobs` is **repurposed** from reserved → DEG process-pool size (default `min(n_clusters, cpu, 8)`). The shared-LLM-client **`diagnosis_timeout` default 60→120 s**, matched by bumping the `timeout=60→120` defaults in `ArkChatClient`/`from_env`/`make_chat_client`/`make_diagnosis_engine` (`diagnosis.py`).
- `cli.py`: `--llm-concurrency` (default 8), `--llm-retries` (default 3), `--ark-timeout` (default 120, → `diagnosis_timeout`); `--n-jobs` help updated (now DEG process parallelism, not reserved).
- `params.json` records `llm_concurrency`, `llm_retries`, `diagnosis_timeout`, effective `n_jobs`.

## 9. Determinism & idempotency

All parallel stages write per-cluster artifacts and reassemble the global TSVs **after** the pool, so output is byte-identical regardless of thread/process completion order. File-existence idempotency, `prompt_version`+`model` checks, and force cascades are unchanged.

## 10. Testing

- `tests/test_parallel.py`: `thread_map`/`process_map` preserve order + run concurrently; `process_map` worker pins BLAS; `with_retry` succeeds on the Nth attempt for a flaky fn and re-raises after exhaustion. (`process_map` test uses a top-level module fn so spawn can pickle it.)
- `tests/test_wilcoxon.py`: new kernel `z`/`pvals`/`logfoldchanges` vs a `scipy.stats.mannwhitneyu` reference (tie-corrected, `use_continuity=False`) on small fixtures — dense vs sparse identical; all-zero gene; heavy-tie column; group <2 cells → NaN/p=1; K-group one-vs-rest and 2-group vs-reference.
- Parallel-vs-serial equivalence: a tiny end-to-end run (stub LLM client) with `n_jobs=2, llm_concurrency=4` produces identical `deg_long.tsv` / `panel.tsv` / `core_names.tsv` / `narratives.tsv` to `n_jobs=1, llm_concurrency=1`.
- Retry-to-fallback: a client that times out N+1 times → engine returns the rule/local fallback (never raises).
- Run on a dev-style allocation via the venv pytest; the existing 28 tests stay green.

## 11. Out of scope (YAGNI)

- asyncio rewrite of the client (threads suffice for the sync urllib client).
- canonical-stage internal gene-chunk parallelism beyond the kernel speedup (the fast kernel already accelerates it; `wilcoxon_n_jobs` left as-is).
- GPU / approximate DE (FastDE).
- No new pip dependencies.

## 12. Risks / notes

- **Correctness-sensitive kernel:** tie-corrected variance changes `z`/p-values vs the current (conservative) output — this is a fix, but golden tests against scipy pin it down; flag in the PR that DEG scores shift (more sensitive) by design.
- **spawn marshaling cost:** bounded by `n_jobs`; defaults conservative. On a tiny dataset (Marrow) `process_map` falls back to ~serial (few clusters) — fine.
- **fork-pool removal** eliminates the latent deadlock; the new `process_map` uses spawn + BLAS-pinned workers.
- **Feature branch only:** `feat/deg-llm-concurrency`; do not merge to `main` without the user. Per-task commits on the branch (same flow as the prior feature).
- `llm_client.py` (vendored) is **not** modified.
