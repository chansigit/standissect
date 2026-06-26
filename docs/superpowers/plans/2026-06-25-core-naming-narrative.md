# Core cell-type naming + per-cluster narrative — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two LLM annotation layers to standissect — canonical-core **cell-type naming** (LLM primary, local marker-overlap backup) and per-cluster **narrative** (LLM only) — in a new `annotate.py`, wire them into the pipeline/CLI/report, and flip the default `diagnosis_mode` to `llm` with graceful keyless degrade.

**Architecture:** Mirror the existing diagnosis layer (evidence dataclass → engine over a chat client → result dataclass → pipeline stage → report). A new `annotate.py` owns the dataclasses, engines, prompts, parsers, marker table, artifact writers, and two stage runners; it depends only on `llm_client` + pandas (no scanpy, no `diagnosis` import). The ARK chat client is built **once** in the pipeline via a new `make_chat_client` (added to `diagnosis.py`, next to the client classes) and shared across diagnosis + naming + narrative. Naming always runs (LLM when a client exists, else local). Narrative runs only when a client exists.

**Tech Stack:** Python 3.12, stdlib-only LLM client (`llm_client.OpenAICompatClient` over Volcengine ARK), pandas/numpy. No new third-party dependencies. No pydantic (standissect's stdlib-only stance).

## Global Constraints

- **Target repo:** `/scratch/users/chensj16/projects/standissect` (the canonical git repo). **NOT** the stale May snapshot at `/scratch/users/chensj16/data/shanghaid/synovial.prep-231031/standissect` — do not touch that copy.
- **Branch, no auto-commit.** First create branch `feat/annotate-naming-narrative` off `main`. The repo has pre-existing uncommitted WIP (`README.md`/`__init__.py`/`cluster.py`/`pipeline.py` modified; `cli.py`/`__main__.py`/`METHODS.md`/`docs/`/`process_visualization.html` untracked). **Do NOT `git commit` and do NOT `git add` the pre-existing modified files** — the user commits deliberately (standing instruction: "do not commit unless asked"). Each task ends with a **passing-test gate**, not a commit.
- **Do NOT modify `llm_client.py`** — it is the vendored shared module (byte-identical with stanmetacols). Reuse it as-is.
- **No new dependencies.** stdlib + pandas/numpy only.
- **Sherlock policy: never run Python/pytest on the login node.** Offload every test run to a `dev` node. Canonical test command (run from `/scratch/users/chensj16/projects/standissect`):
  ```bash
  srun -p dev -c 2 --mem=4G -t 00:15:00 \
    /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest <target> -v
  ```
  If `srun` queueing is slow/unavailable, grab `sh_dev -t 00:20:00` and run the `python -m pytest <target> -v` line interactively inside that shell. The venv (`/scratch/users/chensj16/venvs/dl2025/.venv`) has pytest 9.0.2, numpy 1.26.3, pandas 2.3.3, and scanpy.
- **Constant names (use verbatim):** `NAMING_PROMPT_VERSION = 'standissect-naming-v1'`, `NARRATIVE_PROMPT_VERSION = 'standissect-narrative-v1'`. ARK defaults reused from `diagnosis.py`: `DEFAULT_ARK_MODEL = 'ep-20260412124039-zjq7v'`, `DEFAULT_ARK_ENDPOINT = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'`, `ARK_API_KEY`.
- **Marker file schema (consumed by naming):** `canonical_markers/markers_c{N}_0.tsv` has columns `group, rank, gene, logfoldchanges, pvals, pvals_adj, scores` (gene column is `gene`; positive `logfoldchanges` = up in the core; rank by `scores` desc).
- **Spec:** `docs/superpowers/specs/2026-06-25-core-naming-narrative-design.md`.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `annotate.py` | **create** | naming + narrative dataclasses, engines, prompts, parsers, marker table, artifact writers, stage runners. Imports only `llm_client` + pandas. |
| `diagnosis.py` | modify | add `make_chat_client(...) -> client \| None`; refactor `make_diagnosis_engine` to use it (preserve its raise-on-no-key contract). |
| `pipeline.py` | modify | build chat client once + degrade; insert `naming` + `narrative` stages; `_STAGES`, `_Layout`, `params.json`, return dict, docstring. |
| `cli.py` | modify | default `--diagnosis-mode llm`; `--force` adds `naming`/`narrative`; new `--annotation-hint`, `--naming-markers`. |
| `report.py` | modify | per-cluster header `cluster N — {cell_type}` + narrative paragraph; overview core-names table; tolerant of missing files. |
| `__init__.py` | modify | export the new annotate symbols + `make_chat_client`. |
| `tests/test_annotate.py` | **create** | engines/parsers/marker-overlap/stage runners (no scanpy). |
| `tests/test_diagnosis_llm.py` | modify | append `make_chat_client` + degrade-contract tests. |
| `tests/test_report.py` | **create** | `build_report` shows name + narrative (no scanpy). |
| `tests/test_cli.py` | **create** | parser defaults + new options (imports scanpy → dev node). |

Dependency graph (no cycles): `llm_client ← diagnosis ← pipeline`; `llm_client ← annotate ← pipeline`. `diagnosis` and `annotate` are siblings; `pipeline` depends on both.

---

## Task 0: Branch + green baseline

**Files:** none (git only)

- [ ] **Step 1: Create the feature branch**

```bash
cd /scratch/users/chensj16/projects/standissect
git switch -c feat/annotate-naming-narrative
git branch --show-current      # expect: feat/annotate-naming-narrative
```
(The uncommitted WIP follows onto the new branch; leave it untracked/unstaged.)

- [ ] **Step 2: Confirm the existing test suite is green (baseline)**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_diagnosis_llm.py -v
```
Expected: 4 passed.

---

## Task 1: `make_chat_client` + degrade-friendly `make_diagnosis_engine` (diagnosis.py)

**Files:**
- Modify: `diagnosis.py` (replace `make_diagnosis_engine`, add `make_chat_client` before it)
- Test: `tests/test_diagnosis_llm.py` (extend imports + append 5 tests)

**Interfaces:**
- Produces: `make_chat_client(*, mode='llm', llm_client=None, ark_api_key=None, ark_api_key_env='ARK_API_KEY', ark_model=DEFAULT_ARK_MODEL, ark_endpoint=DEFAULT_ARK_ENDPOINT, timeout=60) -> client | None`. Returns `None` for `mode == 'rule'` or when no client/key is available; wraps a bare callable as `CallableChatClient`; else builds `ArkChatClient`. **Never raises** on missing key.
- `make_diagnosis_engine(...)` signature unchanged; still raises `ValueError` for `mode in {'llm','hybrid'}` with no client/key (backward-compatible).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_diagnosis_llm.py`

Extend the existing import on line 13–14 to add the new names. Replace:
```python
from diagnosis import (parse_llm_result, _diagnosis_from_dict,  # noqa: E402
                       CallableChatClient, ALLOWED_CAUSES)
```
with:
```python
from diagnosis import (parse_llm_result, _diagnosis_from_dict,  # noqa: E402
                       CallableChatClient, ALLOWED_CAUSES,
                       make_chat_client, make_diagnosis_engine, ArkChatClient)
```

Append at end of file:
```python
def test_make_chat_client_returns_none_for_rule():
    assert make_chat_client(mode="rule") is None


def test_make_chat_client_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    assert make_chat_client(mode="llm", ark_api_key_env="ARK_API_KEY") is None


def test_make_chat_client_wraps_callable():
    c = make_chat_client(mode="llm", llm_client=lambda s, u: "{}")
    assert isinstance(c, CallableChatClient)
    assert hasattr(c, "complete")


def test_make_chat_client_builds_ark_with_key():
    c = make_chat_client(mode="llm", ark_api_key="secret", ark_model="mymodel")
    assert isinstance(c, ArkChatClient)
    assert c.model == "mymodel"


def test_make_diagnosis_engine_still_raises_on_llm_without_key(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    with pytest.raises(ValueError):
        make_diagnosis_engine(mode="llm")
```

- [ ] **Step 2: Run to verify they fail**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_diagnosis_llm.py -v
```
Expected: ImportError / `cannot import name 'make_chat_client'` → collection error or failures on the 5 new tests.

- [ ] **Step 3: Implement** — in `diagnosis.py`, **replace the entire `make_diagnosis_engine` function** (currently lines 534–574) with the following two functions:

```python
def make_chat_client(
    *,
    mode: str = 'llm',
    llm_client=None,
    ark_api_key: str | None = None,
    ark_api_key_env: str = 'ARK_API_KEY',
    ark_model: str = DEFAULT_ARK_MODEL,
    ark_endpoint: str = DEFAULT_ARK_ENDPOINT,
    timeout: int = 60,
):
    """Build a chat client for LLM modes, or return ``None``.

    Returns ``None`` when ``mode == 'rule'`` or when no client/key is available,
    so callers can degrade gracefully instead of crashing. A provided
    ``llm_client`` is returned as-is when it has ``.complete``, else wrapped in a
    ``CallableChatClient``. Never raises on a missing key.
    """
    if str(mode) == 'rule':
        return None
    if llm_client is not None:
        if hasattr(llm_client, 'complete'):
            return llm_client
        return CallableChatClient(llm_client, model=ark_model)
    api_key = ark_api_key if ark_api_key is not None else os.environ.get(ark_api_key_env)
    if not api_key:
        return None
    return ArkChatClient(
        api_key=api_key, model=ark_model, endpoint=ark_endpoint, timeout=timeout)


def make_diagnosis_engine(
    *,
    mode: str = 'rule',
    llm_client=None,
    ark_api_key: str | None = None,
    ark_api_key_env: str = 'ARK_API_KEY',
    ark_model: str = DEFAULT_ARK_MODEL,
    ark_endpoint: str = DEFAULT_ARK_ENDPOINT,
    timeout: int = 60,
    fallback_to_rule: bool = True,
    diagnosis_roles=None,
):
    """Create the requested diagnosis engine.

    ``llm_client`` may be an object with ``complete(system, user)`` or a callable
    with that same signature. Without ``llm_client``, LLM modes create an Ark
    client from ``ARK_API_KEY``. Raises ``ValueError`` for an LLM mode with no
    client and no key (use ``make_chat_client`` directly for graceful degrade).
    """
    mode = str(mode)
    if mode not in {'rule', 'llm', 'hybrid'}:
        raise ValueError("diagnosis_mode must be 'rule', 'llm', or 'hybrid'")
    if mode == 'rule':
        return RuleDiagnosisEngine(diagnosis_roles)
    client = make_chat_client(
        mode=mode, llm_client=llm_client, ark_api_key=ark_api_key,
        ark_api_key_env=ark_api_key_env, ark_model=ark_model,
        ark_endpoint=ark_endpoint, timeout=timeout)
    if client is None:
        raise ValueError(
            f"diagnosis_mode={mode!r} requires diagnosis_llm_client, "
            f"diagnosis_ark_api_key, or environment variable {ark_api_key_env!r}"
        )
    return LLMDiagnosisEngine(
        client, mode=mode, fallback_to_rule=fallback_to_rule,
        diagnosis_roles=diagnosis_roles)
```

- [ ] **Step 4: Run to verify they pass**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_diagnosis_llm.py -v
```
Expected: 9 passed.

---

## Task 2: annotate.py — module skeleton, marker table, local naming

**Files:**
- Create: `annotate.py`
- Test: create `tests/test_annotate.py`

**Interfaces:**
- Produces:
  - `NAMING_PROMPT_VERSION`, `NARRATIVE_PROMPT_VERSION`, `DEFAULT_MARKER_SETS`
  - `load_marker_sets(markers=None) -> dict[str, list[str]]`
  - `CoreEvidence(parent_cluster, core_subcluster, n_cells=0, top_markers=[], hint='')` with `.to_dict()` and `.marker_genes() -> list[str]`
  - `CoreNaming(cell_type=None, confidence=0.0, rationale='', markers_used=[], alternatives=[], source='unnamed', model=None, prompt_version=NAMING_PROMPT_VERSION, error=None)` with `.to_dict()` and `.to_core_name_row(evidence)`
  - `build_core_evidence(parent_cluster, markers_path, *, n_cells=0, hint='', top_n=20) -> CoreEvidence`
  - `LocalNamingEngine(markers=None, *, min_overlap=1)` with `.name(evidence) -> CoreNaming`, attrs `source='local'`, `model=None`

- [ ] **Step 1: Write the failing tests** — create `tests/test_annotate.py`

```python
import json
import pathlib
import sys

import pandas as pd
import pytest

# Import annotate as a top-level module, bypassing standissect/__init__.py
# (which imports scanpy via .cluster). annotate.py uses dual-import.
_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]   # .../standissect
sys.path.insert(0, str(_PKG_DIR))

import annotate  # noqa: E402
from diagnosis import CallableChatClient  # noqa: E402


def _evi(genes, parent="0"):
    return annotate.CoreEvidence(
        parent_cluster=parent, core_subcluster=f"c{parent}_0",
        top_markers=[{"gene": g, "logfoldchanges": 2.0, "scores": 10.0}
                     for g in genes])


def test_load_marker_sets_default_dict_and_tsv(tmp_path):
    d = annotate.load_marker_sets(None)
    assert "T cell" in d and "Fibroblast" in d
    assert annotate.load_marker_sets({"Foo": ["A", "B"]}) == {"Foo": ["A", "B"]}
    p = tmp_path / "m.tsv"
    p.write_text("Bar\tX,Y,Z\nBaz\tP,Q\n", encoding="utf-8")
    assert annotate.load_marker_sets(str(p)) == {"Bar": ["X", "Y", "Z"], "Baz": ["P", "Q"]}


def test_local_naming_picks_t_cell():
    r = annotate.LocalNamingEngine().name(_evi(["CD3D", "CD3E", "TRAC", "CD2", "IL7R"]))
    assert r.cell_type == "T cell"
    assert r.source == "local"
    assert r.confidence > 0
    assert set(r.markers_used) <= {"CD3D", "CD3E", "TRAC", "CD2", "IL7R"}


def test_local_naming_unnamed_on_no_overlap():
    r = annotate.LocalNamingEngine().name(_evi(["FAKE1", "FAKE2", "FAKE3"]))
    assert r.cell_type is None
    assert r.source == "unnamed"


def test_build_core_evidence_reads_top_up_markers(tmp_path):
    p = tmp_path / "markers_c0_0.tsv"
    pd.DataFrame({
        "group": ["c0_0"] * 4, "rank": [0, 1, 2, 3],
        "gene": ["CD3D", "CD3E", "NEG1", "TRAC"],
        "logfoldchanges": [3.0, 2.5, -1.0, 2.0],
        "pvals": [1e-9] * 4, "pvals_adj": [1e-8] * 4,
        "scores": [20.0, 18.0, 15.0, 9.0],
    }).to_csv(p, sep="\t", index=False)
    evi = annotate.build_core_evidence("0", p, n_cells=123, hint="synovium", top_n=10)
    genes = evi.marker_genes()
    assert "NEG1" not in genes          # negative LFC dropped
    assert genes[0] == "CD3D"           # highest score first
    assert evi.n_cells == 123
    assert evi.hint == "synovium"
    assert evi.core_subcluster == "c0_0"
```

- [ ] **Step 2: Run to verify they fail**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_annotate.py -v
```
Expected: collection error `ModuleNotFoundError: No module named 'annotate'`.

- [ ] **Step 3: Implement** — create `annotate.py` with this content:

```python
"""standissect.annotate — LLM cell-type naming + per-cluster narrative.

Two annotation layers on top of diagnosis, mirroring its shape (evidence
dataclass -> engine over a chat client -> result dataclass):

  * core cell-type NAMING — map a canonical core's ranked markers to a cell type
    (LLM primary, local marker-overlap backup; always produces a result);
  * per-cluster NARRATIVE — one evidence-grounded paragraph (LLM only).

Stdlib + pandas only. Reuses the shared OpenAI-compatible client via
``llm_client.call_structured``; the client itself is built/owned by the caller
(see ``diagnosis.make_chat_client``) and passed in, so this module imports
neither scanpy nor diagnosis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

try:                                    # package use (standissect.annotate)
    from .llm_client import call_structured, LLMUnavailable
except ImportError:                     # standalone use (tests import top-level)
    from llm_client import call_structured, LLMUnavailable

import pandas as pd


NAMING_PROMPT_VERSION = 'standissect-naming-v1'
NARRATIVE_PROMPT_VERSION = 'standissect-narrative-v1'

# Broad, well-established lineage markers for the local naming backup. Tuned for
# synovial tissue but generally useful; override via ``naming_markers``.
DEFAULT_MARKER_SETS = {
    'Fibroblast':        ['PRG4', 'THY1', 'PDPN', 'FAP', 'COL1A1', 'COL1A2', 'DCN', 'LUM', 'PDGFRA'],
    'Macrophage/Myeloid': ['CD68', 'CD14', 'LYZ', 'AIF1', 'CD163', 'C1QA', 'C1QB', 'FCGR3A'],
    'T cell':            ['CD3D', 'CD3E', 'CD3G', 'CD2', 'TRAC', 'CD8A', 'CD4', 'IL7R'],
    'NK cell':           ['NKG7', 'GNLY', 'KLRD1', 'NCAM1', 'KLRF1'],
    'B cell':            ['CD79A', 'CD79B', 'MS4A1', 'CD19', 'BANK1'],
    'Plasma cell':       ['MZB1', 'IGHG1', 'JCHAIN', 'XBP1', 'SDC1', 'DERL3'],
    'Endothelial':       ['PECAM1', 'VWF', 'CLDN5', 'CDH5', 'EGFL7'],
    'Mural/Pericyte':    ['ACTA2', 'RGS5', 'MYH11', 'PDGFRB', 'NOTCH3'],
    'Dendritic cell':    ['FCER1A', 'CLEC10A', 'CD1C', 'LILRA4'],
    'Mast cell':         ['TPSAB1', 'TPSB2', 'CPA3', 'MS4A2'],
}


def _num(value):
    """JSON-safe float, or None for NaN/missing/non-numeric."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:                          # NaN
        return None
    return f


def _read_tsv(path):
    try:
        return pd.read_csv(path, sep='\t')
    except Exception:
        return pd.DataFrame()


def load_marker_sets(markers=None) -> dict:
    """Resolve the naming marker table.

    ``None`` -> a copy of ``DEFAULT_MARKER_SETS``; a ``dict[cell_type -> genes]``
    -> normalized copy; a path/str -> a 2-column TSV ``cell_type<TAB>gene,gene,...``
    (no header).
    """
    if markers is None:
        return {k: list(v) for k, v in DEFAULT_MARKER_SETS.items()}
    if isinstance(markers, dict):
        return {str(k): [str(g) for g in v] for k, v in markers.items()}
    out: dict = {}
    df = pd.read_csv(markers, sep='\t', header=None)
    for _, row in df.iterrows():
        cell_type = str(row.iloc[0]).strip()
        genes = [g.strip() for g in str(row.iloc[1]).split(',') if g.strip()]
        if cell_type and genes:
            out[cell_type] = genes
    return out


@dataclass
class CoreEvidence:
    """Compact, serializable evidence for one canonical core (``c{N}_0``)."""

    parent_cluster: str
    core_subcluster: str
    n_cells: int = 0
    top_markers: list[dict] = field(default_factory=list)   # {gene, logfoldchanges, scores}
    hint: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    def marker_genes(self) -> list[str]:
        return [str(m.get('gene')) for m in self.top_markers if m.get('gene')]


@dataclass
class CoreNaming:
    """Stable naming output written to core_names.tsv and naming.output.json."""

    cell_type: str | None = None
    confidence: float = 0.0
    rationale: str = ''
    markers_used: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    source: str = 'unnamed'             # 'llm' | 'local' | 'unnamed'
    model: str | None = None
    prompt_version: str = NAMING_PROMPT_VERSION
    error: str | None = None

    def __post_init__(self):
        self.confidence = float(min(1.0, max(0.0, self.confidence)))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_core_name_row(self, evidence: CoreEvidence) -> dict:
        return {
            'parent_cluster': evidence.parent_cluster,
            'core_subcluster': evidence.core_subcluster,
            'cell_type': self.cell_type,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'source': self.source,
            'model': self.model,
        }


def build_core_evidence(parent_cluster, markers_path, *, n_cells=0, hint='',
                        top_n=20) -> CoreEvidence:
    """Build core evidence from ``markers_c{N}_0.tsv`` (top up-regulated by score)."""
    top_markers: list[dict] = []
    df = _read_tsv(markers_path)
    gene_col = 'gene' if 'gene' in df.columns else ('names' if 'names' in df.columns else None)
    if len(df) and gene_col and 'scores' in df.columns:
        up = df.copy()
        if 'logfoldchanges' in up.columns:
            up = up[up['logfoldchanges'] > 0]
        up = up.sort_values('scores', ascending=False).head(top_n)
        for _, r in up.iterrows():
            top_markers.append({
                'gene': str(r[gene_col]),
                'logfoldchanges': _num(r.get('logfoldchanges')),
                'scores': _num(r.get('scores')),
            })
    return CoreEvidence(
        parent_cluster=str(parent_cluster),
        core_subcluster=f"c{parent_cluster}_0",
        n_cells=int(n_cells or 0),
        top_markers=top_markers,
        hint=str(hint or ''),
    )


class LocalNamingEngine:
    """Backup namer: overlap of the core's top markers against a marker table.

    Score = Szymkiewicz-Simpson overlap coefficient
    ``|core ∩ type| / min(|core|, |type|)``; the highest-scoring type (then
    highest raw overlap count) wins. No network, no new dependency.
    """

    source = 'local'
    model = None

    def __init__(self, markers=None, *, min_overlap=1):
        self.markers = load_marker_sets(markers)
        self.min_overlap = min_overlap

    def name(self, evidence: CoreEvidence) -> CoreNaming:
        genes = {g.upper() for g in evidence.marker_genes()}
        if not genes or not self.markers:
            return CoreNaming(cell_type=None, source='unnamed',
                              rationale='no markers available for local overlap')
        best = None                     # (coef, n, cell_type, sorted_overlap)
        for cell_type, mset in self.markers.items():
            ref = {g.upper() for g in mset}
            if not ref:
                continue
            inter = genes & ref
            n = len(inter)
            if n == 0:
                continue
            coef = n / min(len(genes), len(ref))
            cand = (coef, n, cell_type, sorted(inter))
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best is None or best[1] < self.min_overlap:
            return CoreNaming(cell_type=None, source='unnamed',
                              rationale='no marker set overlapped the core markers')
        coef, n, cell_type, inter = best
        return CoreNaming(
            cell_type=cell_type, confidence=coef,
            rationale=(f"local marker overlap: {n} of the core top markers match "
                       f"{cell_type} ({', '.join(inter)})"),
            markers_used=inter, source='local', model=None)
```

- [ ] **Step 4: Run to verify they pass**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_annotate.py -v
```
Expected: 4 passed.

---

## Task 3: annotate.py — LLM naming (prompt, parser, engine, factory)

**Files:**
- Modify: `annotate.py` (append)
- Test: `tests/test_annotate.py` (append)

**Interfaces:**
- Consumes: `CoreEvidence`, `CoreNaming`, `LocalNamingEngine`, `call_structured`, `LLMUnavailable`.
- Produces:
  - `build_core_naming_prompt(evidence) -> (system, user)`
  - `_core_naming_from_dict(data, evidence, *, model) -> CoreNaming` (hallucination guard; `uncertain`/empty → `cell_type=None`, `source='llm'`)
  - `LLMNamingEngine(client, *, local=None, fallback_to_local=True)` with `.name(evidence) -> CoreNaming`, attr `model = getattr(client, 'model', None)`; never raises (falls back to `local` on `LLMUnavailable`, else `unnamed`)
  - `make_naming_engine(*, client=None, markers=None, fallback_to_local=True) -> LocalNamingEngine | LLMNamingEngine`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_annotate.py`

```python
def _raise(system, user):
    raise RuntimeError("no network")


def test_llm_naming_happy_and_marker_guard():
    payload = json.dumps({
        "cell_type": "T cell", "confidence": 0.9, "rationale": "CD3D/CD3E present",
        "markers_used": ["CD3D", "CD3E", "HALLUCINATED"], "alternatives": ["NK cell"]})
    eng = annotate.LLMNamingEngine(CallableChatClient(lambda s, u: payload, model="m"))
    r = eng.name(_evi(["CD3D", "CD3E", "TRAC"]))
    assert r.cell_type == "T cell"
    assert r.source == "llm"
    assert r.model == "m"
    assert "HALLUCINATED" not in r.markers_used      # not in supplied list -> dropped
    assert set(r.markers_used) <= {"CD3D", "CD3E", "TRAC"}


def test_llm_naming_uncertain_to_none():
    payload = json.dumps({"cell_type": "uncertain", "confidence": 0.1, "rationale": "ambiguous"})
    r = annotate.LLMNamingEngine(CallableChatClient(lambda s, u: payload)).name(_evi(["CD3D"]))
    assert r.cell_type is None
    assert r.source == "llm"


def test_llm_naming_falls_back_to_local():
    eng = annotate.make_naming_engine(client=CallableChatClient(_raise, model="m"))
    r = eng.name(_evi(["CD3D", "CD3E", "TRAC", "CD2"]))
    assert r.source == "local"
    assert r.cell_type == "T cell"
    assert r.model == "m"               # engine model preserved through fallback
    assert r.error is not None


def test_llm_naming_no_fallback_unnamed():
    eng = annotate.LLMNamingEngine(CallableChatClient(_raise, model="m"),
                                   local=None, fallback_to_local=False)
    r = eng.name(_evi(["CD3D"]))
    assert r.cell_type is None
    assert r.source == "unnamed"
    assert r.error is not None


def test_make_naming_engine_selects_local_or_llm():
    assert isinstance(annotate.make_naming_engine(client=None), annotate.LocalNamingEngine)
    eng = annotate.make_naming_engine(client=CallableChatClient(lambda s, u: "{}"))
    assert isinstance(eng, annotate.LLMNamingEngine)
```

- [ ] **Step 2: Run to verify they fail**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_annotate.py -k naming -v
```
Expected: `AttributeError: module 'annotate' has no attribute 'LLMNamingEngine'` (and `make_naming_engine`).

- [ ] **Step 3: Implement** — append to `annotate.py`:

```python
_UNCERTAIN = {'uncertain', 'unknown', 'unclear', 'ambiguous', 'na', 'n/a', 'none', 'null', ''}


def build_core_naming_prompt(evidence: CoreEvidence) -> tuple[str, str]:
    schema = {
        'cell_type': 'cell type/state name, or "uncertain"',
        'confidence': 'number from 0 to 1',
        'rationale': 'one concise sentence citing supplied markers',
        'markers_used': ['subset of the supplied marker genes'],
        'alternatives': ['other plausible cell types'],
    }
    system = (
        "You are a single-cell biologist. Name the most likely cell type or state "
        "for a cluster from its ranked canonical marker genes, using established "
        "marker-to-cell-type knowledge. If the markers are ambiguous, return "
        '"uncertain" with low confidence. Cite only markers from the supplied '
        "list; do not introduce markers that are not listed. Return strict JSON only."
    )
    user = json.dumps({
        'task': 'name_one_canonical_core',
        'tissue_hint': evidence.hint,
        'output_schema': schema,
        'evidence': evidence.to_dict(),
    }, ensure_ascii=False, indent=2)
    return system, user


def _core_naming_from_dict(data, evidence: CoreEvidence, *, model) -> CoreNaming:
    supplied = set(evidence.marker_genes())
    raw = data.get('cell_type')
    cell_type = str(raw).strip() if raw is not None else None
    if cell_type is None or cell_type.lower() in _UNCERTAIN:
        cell_type = None
    used = [str(g) for g in (data.get('markers_used') or []) if str(g) in supplied]
    return CoreNaming(
        cell_type=cell_type,
        confidence=float(data.get('confidence', 0.0) or 0.0),
        rationale=str(data.get('rationale', '')),
        markers_used=used,
        alternatives=[str(a) for a in (data.get('alternatives') or [])],
        source='llm',
        model=model,
    )


class LLMNamingEngine:
    """Primary namer over a chat client; falls back to ``local`` on failure."""

    source = 'llm'

    def __init__(self, client, *, local: 'LocalNamingEngine | None' = None,
                 fallback_to_local: bool = True):
        self.client = client
        self.local = local
        self.fallback_to_local = fallback_to_local
        self.model = getattr(client, 'model', None)

    def name(self, evidence: CoreEvidence) -> CoreNaming:
        system, user = build_core_naming_prompt(evidence)
        try:
            return call_structured(
                self.client, system, user,
                lambda data: _core_naming_from_dict(data, evidence, model=self.model))
        except Exception as e:
            if self.local is not None and self.fallback_to_local:
                result = self.local.name(evidence)
                result.model = self.model
                result.error = str(e)
                return result
            return CoreNaming(cell_type=None, source='unnamed',
                              model=self.model, error=str(e))


def make_naming_engine(*, client=None, markers=None, fallback_to_local=True):
    """LLM primary + local backup when a client exists, else local-only.

    Naming therefore always produces a result.
    """
    local = LocalNamingEngine(markers)
    if client is None:
        return local
    return LLMNamingEngine(client, local=local, fallback_to_local=fallback_to_local)
```

- [ ] **Step 4: Run to verify they pass**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_annotate.py -v
```
Expected: 9 passed.

---

## Task 4: annotate.py — narrative (evidence, result, prompt, parser, engine)

**Files:**
- Modify: `annotate.py` (append)
- Test: `tests/test_annotate.py` (append)

**Interfaces:**
- Produces:
  - `ClusterNarrativeEvidence(parent_cluster, cell_type=None, minors=[], hint='')` with `.to_dict()`
  - `ClusterNarrative(narrative='', source='skipped', model=None, prompt_version=NARRATIVE_PROMPT_VERSION, error=None)` with `.to_dict()` and `.to_narrative_row(evidence)`
  - `build_narrative_prompt(evidence) -> (system, user)`
  - `_narrative_from_dict(data, *, model) -> ClusterNarrative` (raises `ValueError` on empty → caller maps to skipped)
  - `NarrativeEngine(client)` with `.narrate(evidence) -> ClusterNarrative`, attr `model`; never raises (empty/error → `source='skipped'`)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_annotate.py`

```python
def test_narrative_happy():
    payload = json.dumps({"narrative": "Cluster 0 is a T cell population with a doublet fragment."})
    eng = annotate.NarrativeEngine(CallableChatClient(lambda s, u: payload, model="m"))
    ev = annotate.ClusterNarrativeEvidence(
        parent_cluster="0", cell_type="T cell",
        minors=[{"subcluster": "c0_1", "likely_cause": "doublet-driven",
                 "cause_detail": "x", "diagnosis_rationale": "y"}])
    r = eng.narrate(ev)
    assert r.source == "llm"
    assert "T cell" in r.narrative
    assert r.model == "m"


def test_narrative_empty_to_skipped():
    eng = annotate.NarrativeEngine(CallableChatClient(lambda s, u: json.dumps({"narrative": "  "})))
    r = eng.narrate(annotate.ClusterNarrativeEvidence(parent_cluster="0"))
    assert r.source == "skipped"
    assert r.narrative == ""
    assert r.error is not None
```

- [ ] **Step 2: Run to verify they fail**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_annotate.py -k narrative -v
```
Expected: `AttributeError: module 'annotate' has no attribute 'NarrativeEngine'`.

- [ ] **Step 3: Implement** — append to `annotate.py`:

```python
@dataclass
class ClusterNarrativeEvidence:
    """Facts for one cluster's narrative — its core identity + minor diagnoses."""

    parent_cluster: str
    cell_type: str | None = None
    minors: list[dict] = field(default_factory=list)   # {subcluster, likely_cause, cause_detail, diagnosis_rationale}
    hint: str = ''

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClusterNarrative:
    """Stable narrative output written to narratives.tsv and narrative.output.json."""

    narrative: str = ''
    source: str = 'skipped'             # 'llm' | 'skipped'
    model: str | None = None
    prompt_version: str = NARRATIVE_PROMPT_VERSION
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_narrative_row(self, evidence: ClusterNarrativeEvidence) -> dict:
        return {
            'parent_cluster': evidence.parent_cluster,
            'cell_type': evidence.cell_type,
            'narrative': self.narrative,
        }


def build_narrative_prompt(evidence: ClusterNarrativeEvidence) -> tuple[str, str]:
    schema = {'narrative': 'one concise paragraph of plain prose'}
    system = (
        "Summarize this single-cell cluster for a report using only the supplied "
        "facts: its core cell-type identity and each minor fragment's diagnosis. "
        "Write one concise paragraph of plain prose. Do not introduce new cell "
        "types or causes beyond those supplied. Return strict JSON only."
    )
    user = json.dumps({
        'task': 'narrate_one_cluster',
        'tissue_hint': evidence.hint,
        'output_schema': schema,
        'evidence': evidence.to_dict(),
    }, ensure_ascii=False, indent=2)
    return system, user


def _narrative_from_dict(data, *, model) -> ClusterNarrative:
    text = data.get('narrative')
    if text is None or not str(text).strip():
        raise ValueError("narrative missing or empty")
    return ClusterNarrative(narrative=str(text).strip(), source='llm', model=model)


class NarrativeEngine:
    """Evidence-grounded one-paragraph narrative over a chat client. LLM only."""

    def __init__(self, client):
        self.client = client
        self.model = getattr(client, 'model', None)

    def narrate(self, evidence: ClusterNarrativeEvidence) -> ClusterNarrative:
        system, user = build_narrative_prompt(evidence)
        try:
            return call_structured(
                self.client, system, user,
                lambda data: _narrative_from_dict(data, model=self.model))
        except Exception as e:
            return ClusterNarrative(narrative='', source='skipped',
                                    model=self.model, error=str(e))
```

- [ ] **Step 4: Run to verify they pass**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_annotate.py -v
```
Expected: 11 passed.

---

## Task 5: annotate.py — artifact writers, idempotency, stage runners

**Files:**
- Modify: `annotate.py` (append)
- Test: `tests/test_annotate.py` (append)

**Interfaces:**
- Produces:
  - `CORE_NAME_COLS`, `NARRATIVE_COLS` (column order constants)
  - `write_naming_artifacts(cdir, evidence, naming)`, `write_narrative_artifacts(cdir, evidence, narrative)`
  - `_naming_current(cdir, *, model) -> bool`, `_narrative_current(cdir, *, model) -> bool` (match `prompt_version` + `model` in the saved `.output.json`)
  - `run_naming_stage(*, clusters_dir, canonical_dir, core_names_path, parents, engine, hint='', forced=False, core_sizes=None) -> list[str]` (skipped labels `naming:c{p}`); writes `clusters/c{p}/naming.{input,output}.json` and `core_names.tsv`
  - `run_narrative_stage(*, clusters_dir, core_names_path, narratives_path, parents, engine, hint='', forced=False) -> list[str]` (skipped labels `narrative:c{p}`); writes `clusters/c{p}/narrative.{input,output}.json` and `narratives.tsv`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_annotate.py`

```python
def _write_markers(canonical, parent, genes):
    canonical.mkdir(parents=True, exist_ok=True)
    n = len(genes)
    pd.DataFrame({
        "group": [f"c{parent}_0"] * n, "rank": list(range(n)), "gene": genes,
        "logfoldchanges": [3.0] * n, "pvals": [1e-9] * n, "pvals_adj": [1e-8] * n,
        "scores": [float(20 - i) for i in range(n)],
    }).to_csv(canonical / f"markers_c{parent}_0.tsv", sep="\t", index=False)


def test_run_naming_stage_writes_and_is_idempotent(tmp_path):
    clusters = tmp_path / "clusters"
    canonical = tmp_path / "canonical_markers"
    (clusters / "c0").mkdir(parents=True)
    _write_markers(canonical, "0", ["CD3D", "CD3E", "TRAC"])
    core_names = tmp_path / "core_names.tsv"
    eng = annotate.make_naming_engine(client=None)      # local
    sk1 = annotate.run_naming_stage(
        clusters_dir=clusters, canonical_dir=canonical, core_names_path=core_names,
        parents=["0"], engine=eng, forced=False)
    assert sk1 == []
    df = pd.read_csv(core_names, sep="\t")
    assert df.loc[0, "cell_type"] == "T cell"
    assert (clusters / "c0" / "naming.output.json").exists()
    sk2 = annotate.run_naming_stage(
        clusters_dir=clusters, canonical_dir=canonical, core_names_path=core_names,
        parents=["0"], engine=eng, forced=False)
    assert sk2 == ["naming:c0"]         # local model None matches -> skipped


def test_run_naming_stage_recomputes_on_model_change(tmp_path):
    clusters = tmp_path / "clusters"
    canonical = tmp_path / "canonical_markers"
    (clusters / "c0").mkdir(parents=True)
    _write_markers(canonical, "0", ["CD3D"])
    core_names = tmp_path / "core_names.tsv"
    annotate.run_naming_stage(
        clusters_dir=clusters, canonical_dir=canonical, core_names_path=core_names,
        parents=["0"], engine=annotate.make_naming_engine(client=None), forced=False)
    llm = annotate.make_naming_engine(client=CallableChatClient(
        lambda s, u: json.dumps({"cell_type": "T cell", "confidence": 0.9,
                                 "rationale": "r", "markers_used": ["CD3D"]}), model="m"))
    sk = annotate.run_naming_stage(
        clusters_dir=clusters, canonical_dir=canonical, core_names_path=core_names,
        parents=["0"], engine=llm, forced=False)
    assert sk == []                     # model None -> 'm' => recomputed


def test_run_narrative_stage_writes_and_is_idempotent(tmp_path):
    clusters = tmp_path / "clusters"
    (clusters / "c0").mkdir(parents=True)
    pd.DataFrame({"subcluster": ["c0_1"], "likely_cause": ["doublet-driven"],
                  "cause_detail": ["x"], "diagnosis_rationale": ["y"]}
                 ).to_csv(clusters / "c0" / "panel.tsv", sep="\t", index=False)
    core_names = tmp_path / "core_names.tsv"
    pd.DataFrame({"parent_cluster": ["0"], "core_subcluster": ["c0_0"],
                  "cell_type": ["T cell"], "confidence": [0.9], "rationale": ["r"],
                  "source": ["llm"], "model": ["m"]}).to_csv(core_names, sep="\t", index=False)
    narr = tmp_path / "narratives.tsv"
    eng = annotate.NarrativeEngine(CallableChatClient(
        lambda s, u: json.dumps({"narrative": "A T cell cluster with a doublet fragment."}),
        model="m"))
    sk = annotate.run_narrative_stage(
        clusters_dir=clusters, core_names_path=core_names, narratives_path=narr,
        parents=["0"], engine=eng, forced=False)
    assert sk == []
    df = pd.read_csv(narr, sep="\t")
    assert "doublet" in df.loc[0, "narrative"]
    assert df.loc[0, "cell_type"] == "T cell"
    sk2 = annotate.run_narrative_stage(
        clusters_dir=clusters, core_names_path=core_names, narratives_path=narr,
        parents=["0"], engine=eng, forced=False)
    assert sk2 == ["narrative:c0"]
```

- [ ] **Step 2: Run to verify they fail**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_annotate.py -k stage -v
```
Expected: `AttributeError: module 'annotate' has no attribute 'run_naming_stage'`.

- [ ] **Step 3: Implement** — append to `annotate.py`:

```python
CORE_NAME_COLS = ['parent_cluster', 'core_subcluster', 'cell_type', 'confidence',
                  'rationale', 'source', 'model']
NARRATIVE_COLS = ['parent_cluster', 'cell_type', 'narrative']


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None


def _safe_str(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _load_core_names(path) -> dict:
    """core_names.tsv -> {parent_cluster: cell_type|None}."""
    df = _read_tsv(path)
    out: dict = {}
    if len(df) and 'parent_cluster' in df.columns and 'cell_type' in df.columns:
        for _, r in df.iterrows():
            out[str(r['parent_cluster'])] = _safe_str(r.get('cell_type'))
    return out


def _read_minor_diagnoses(panel_path) -> list[dict]:
    """A cluster's panel.tsv -> [{subcluster, likely_cause, cause_detail, diagnosis_rationale}]."""
    df = _read_tsv(panel_path)
    if not len(df) or 'subcluster' not in df.columns:
        return []
    cols = ['subcluster', 'likely_cause', 'cause_detail', 'diagnosis_rationale']
    return [{c: _safe_str(r.get(c)) for c in cols} for _, r in df.iterrows()]


def write_naming_artifacts(cdir, evidence: CoreEvidence, naming: CoreNaming):
    cdir = Path(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / 'naming.input.json').write_text(
        json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    (cdir / 'naming.output.json').write_text(
        json.dumps(naming.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')


def write_narrative_artifacts(cdir, evidence: ClusterNarrativeEvidence,
                              narrative: ClusterNarrative):
    cdir = Path(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / 'narrative.input.json').write_text(
        json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    (cdir / 'narrative.output.json').write_text(
        json.dumps(narrative.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')


def _naming_current(cdir, *, model) -> bool:
    data = _read_json(Path(cdir) / 'naming.output.json')
    if not data:
        return False
    if data.get('prompt_version') != NAMING_PROMPT_VERSION:
        return False
    return data.get('model') == model


def _narrative_current(cdir, *, model) -> bool:
    data = _read_json(Path(cdir) / 'narrative.output.json')
    if not data:
        return False
    if data.get('prompt_version') != NARRATIVE_PROMPT_VERSION:
        return False
    return data.get('model') == model


def run_naming_stage(*, clusters_dir, canonical_dir, core_names_path, parents,
                     engine, hint='', forced=False, core_sizes=None) -> list:
    """Name each cluster's canonical core where missing/stale/forced; rewrite
    ``core_names.tsv`` from all ``naming.output.json``. Always runs (LLM or local)."""
    clusters_dir = Path(clusters_dir)
    canonical_dir = Path(canonical_dir)
    core_sizes = core_sizes or {}
    model = getattr(engine, 'model', None)
    skipped: list = []
    for parent in parents:
        cdir = clusters_dir / f"c{parent}"
        if not forced and _naming_current(cdir, model=model):
            skipped.append(f'naming:c{parent}')
            continue
        evidence = build_core_evidence(
            parent, canonical_dir / f"markers_c{parent}_0.tsv",
            n_cells=core_sizes.get(str(parent), 0), hint=hint)
        naming = engine.name(evidence)
        write_naming_artifacts(cdir, evidence, naming)
    rows = []
    for parent in parents:
        data = _read_json(clusters_dir / f"c{parent}" / 'naming.output.json')
        if not data:
            continue
        rows.append({
            'parent_cluster': str(parent),
            'core_subcluster': f"c{parent}_0",
            'cell_type': data.get('cell_type'),
            'confidence': data.get('confidence'),
            'rationale': data.get('rationale'),
            'source': data.get('source'),
            'model': data.get('model'),
        })
    Path(core_names_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CORE_NAME_COLS).to_csv(core_names_path, sep='\t', index=False)
    return skipped


def run_narrative_stage(*, clusters_dir, core_names_path, narratives_path, parents,
                        engine, hint='', forced=False) -> list:
    """Write a per-cluster narrative where missing/stale/forced; rewrite
    ``narratives.tsv``. Caller runs this only when a chat client exists."""
    clusters_dir = Path(clusters_dir)
    core_names = _load_core_names(core_names_path)
    model = getattr(engine, 'model', None)
    skipped: list = []
    for parent in parents:
        cdir = clusters_dir / f"c{parent}"
        if not forced and _narrative_current(cdir, model=model):
            skipped.append(f'narrative:c{parent}')
            continue
        evidence = ClusterNarrativeEvidence(
            parent_cluster=str(parent),
            cell_type=core_names.get(str(parent)),
            minors=_read_minor_diagnoses(cdir / 'panel.tsv'),
            hint=hint)
        narrative = engine.narrate(evidence)
        write_narrative_artifacts(cdir, evidence, narrative)
    rows = []
    for parent in parents:
        data = _read_json(clusters_dir / f"c{parent}" / 'narrative.output.json')
        if not data:
            continue
        rows.append({
            'parent_cluster': str(parent),
            'cell_type': core_names.get(str(parent)),
            'narrative': data.get('narrative'),
        })
    Path(narratives_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=NARRATIVE_COLS).to_csv(narratives_path, sep='\t', index=False)
    return skipped
```

- [ ] **Step 4: Run to verify they pass**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_annotate.py -v
```
Expected: 15 passed.

---

## Task 6: pipeline.py integration

**Files:**
- Modify: `pipeline.py` (imports, `_STAGES`, `_Layout`, `run_dissect_pipeline` body, `params.json`, return dict, docstring)
- Verify: import-graph + stages smoke on a dev node (scanpy ok there) + full annotate/diagnosis suites

**Interfaces:**
- Consumes: `make_chat_client` (diagnosis); `make_naming_engine`, `NarrativeEngine`, `load_marker_sets`, `run_naming_stage`, `run_narrative_stage`, `NAMING_PROMPT_VERSION`, `NARRATIVE_PROMPT_VERSION` (annotate).
- Produces: `run_dissect_pipeline(..., diagnosis_mode='llm', annotation_hint='', naming_markers=None, ...)`; return dict adds `'core_names'`, `'narratives'`.

All edits are anchor-based (find the shown text, replace as directed).

- [ ] **Step 1: Extend the diagnosis import** — replace the import block (lines 23–32):

Find:
```python
from .diagnosis import (
    DEFAULT_ARK_ENDPOINT,
    DEFAULT_ARK_MODEL,
    PROMPT_VERSION,
    build_minor_evidence,
    make_diagnosis_engine,
    normalize_diagnosis_roles,
    safe_subcluster_name,
    write_diagnosis_artifacts,
)
```
Replace with:
```python
from .diagnosis import (
    DEFAULT_ARK_ENDPOINT,
    DEFAULT_ARK_MODEL,
    PROMPT_VERSION,
    build_minor_evidence,
    make_chat_client,
    make_diagnosis_engine,
    normalize_diagnosis_roles,
    safe_subcluster_name,
    write_diagnosis_artifacts,
)
from .annotate import (
    NAMING_PROMPT_VERSION,
    NARRATIVE_PROMPT_VERSION,
    NarrativeEngine,
    load_marker_sets,
    make_naming_engine,
    run_naming_stage,
    run_narrative_stage,
)
```

- [ ] **Step 2: Add the two new stages** — replace line 34:

Find:
```python
_STAGES = ('partition', 'dissect', 'diagnosis', 'canonical', 'profile')
```
Replace with:
```python
_STAGES = ('partition', 'dissect', 'diagnosis', 'canonical', 'naming', 'narrative', 'profile')
```

- [ ] **Step 3: Add `_Layout` paths** — find the `canonical_deg` property (lines 72–73):

Find:
```python
    @property
    def canonical_deg(self): return self.canonical / 'deg_long.tsv'
```
Replace with:
```python
    @property
    def canonical_deg(self): return self.canonical / 'deg_long.tsv'
    @property
    def core_names(self):    return self.root / 'core_names.tsv'
    @property
    def narratives(self):    return self.root / 'narratives.tsv'
```

- [ ] **Step 4: Add the two new pipeline parameters** — find (lines 614–616):

Find:
```python
    diagnosis_timeout=60,
    diagnosis_fallback_to_rule=True,
    force=(),
```
Replace with:
```python
    diagnosis_timeout=60,
    diagnosis_fallback_to_rule=True,
    annotation_hint='',
    naming_markers=None,
    force=(),
```

- [ ] **Step 5: Flip the default + build the shared client and engines** — replace the engine-construction block (lines 670–680):

Find:
```python
    diagnosis_engine = make_diagnosis_engine(
        mode=diagnosis_mode,
        llm_client=diagnosis_llm_client,
        ark_api_key=diagnosis_ark_api_key,
        ark_api_key_env=diagnosis_ark_api_key_env,
        ark_model=diagnosis_ark_model,
        ark_endpoint=diagnosis_ark_endpoint,
        timeout=diagnosis_timeout,
        fallback_to_rule=diagnosis_fallback_to_rule,
        diagnosis_roles=resolved_roles,
    )
```
Replace with:
```python
    # Build the chat client once and share it across diagnosis + naming +
    # narrative. A missing key (or rule mode) yields None -> graceful degrade.
    chat_client = make_chat_client(
        mode=diagnosis_mode,
        llm_client=diagnosis_llm_client,
        ark_api_key=diagnosis_ark_api_key,
        ark_api_key_env=diagnosis_ark_api_key_env,
        ark_model=diagnosis_ark_model,
        ark_endpoint=diagnosis_ark_endpoint,
        timeout=diagnosis_timeout,
    )
    effective_diagnosis_mode = diagnosis_mode
    if diagnosis_mode != 'rule' and chat_client is None:
        print(f"[pipeline] WARNING: diagnosis_mode={diagnosis_mode!r} requested but "
              f"no LLM client is available (set {diagnosis_ark_api_key_env} or pass "
              f"diagnosis_llm_client). Falling back to rule diagnosis; naming uses "
              f"local marker overlap; narrative is skipped.", flush=True)
        effective_diagnosis_mode = 'rule'
    diagnosis_engine = make_diagnosis_engine(
        mode=effective_diagnosis_mode,
        llm_client=chat_client,
        ark_api_key=diagnosis_ark_api_key,
        ark_api_key_env=diagnosis_ark_api_key_env,
        ark_model=diagnosis_ark_model,
        ark_endpoint=diagnosis_ark_endpoint,
        timeout=diagnosis_timeout,
        fallback_to_rule=diagnosis_fallback_to_rule,
        diagnosis_roles=resolved_roles,
    )
    resolved_markers = load_marker_sets(naming_markers)
    naming_engine = make_naming_engine(
        client=chat_client, markers=resolved_markers, fallback_to_local=True)
    narrative_engine = NarrativeEngine(chat_client) if chat_client is not None else None
```

- [ ] **Step 6: Insert the naming + narrative stages after canonical, before profile** — find the STAGE 5 header (lines 812–813):

Find:
```python
    # ---- STAGE 5: per-cluster minor-profile heatmaps -------------------
    profile_force = part_force or ('dissect' in force) or ('profile' in force)
```
Replace with:
```python
    # ---- STAGE: core cell-type naming (always runs) -------------------
    parents = [str(p) for p in crosstab.index]
    core_sizes = {str(p): int(crosstab.loc[p].max()) for p in crosstab.index}
    naming_force = part_force or ('canonical' in force) or ('naming' in force)
    print(f"[pipeline] naming canonical cores "
          f"(source={'llm' if chat_client is not None else 'local'}) ...", flush=True)
    skipped += run_naming_stage(
        clusters_dir=lay.clusters, canonical_dir=lay.canonical,
        core_names_path=lay.core_names, parents=parents, engine=naming_engine,
        hint=annotation_hint, forced=naming_force, core_sizes=core_sizes)
    core_names_df = _read_tsv(lay.core_names)

    # ---- STAGE: per-cluster narrative (LLM only) ----------------------
    narrative_force = naming_force or diagnosis_force or ('narrative' in force)
    if narrative_engine is not None:
        print("[pipeline] writing per-cluster narratives ...", flush=True)
        skipped += run_narrative_stage(
            clusters_dir=lay.clusters, core_names_path=lay.core_names,
            narratives_path=lay.narratives, parents=parents, engine=narrative_engine,
            hint=annotation_hint, forced=narrative_force)
    else:
        skipped += [f'narrative:c{p}' for p in parents]
        print("[pipeline] narrative skipped (no LLM client)", flush=True)
    narratives_df = _read_tsv(lay.narratives)

    # ---- STAGE 5: per-cluster minor-profile heatmaps -------------------
    profile_force = part_force or ('dissect' in force) or ('profile' in force)
```
(`diagnosis_force` is already defined just above STAGE 3 at line 771 and is in scope here.)

- [ ] **Step 7: Record annotation fields in params.json** — find (lines 851–855):

Find:
```python
        'diagnosis_mode': diagnosis_mode,
        'diagnosis_model': getattr(diagnosis_engine, 'model', None),
        'diagnosis_prompt_version': PROMPT_VERSION,
        'diagnosis_ark_endpoint': diagnosis_ark_endpoint,
        'diagnosis_fallback_to_rule': diagnosis_fallback_to_rule,
```
Replace with:
```python
        'diagnosis_mode': diagnosis_mode,
        'effective_diagnosis_mode': effective_diagnosis_mode,
        'diagnosis_model': getattr(diagnosis_engine, 'model', None),
        'diagnosis_prompt_version': PROMPT_VERSION,
        'diagnosis_ark_endpoint': diagnosis_ark_endpoint,
        'diagnosis_fallback_to_rule': diagnosis_fallback_to_rule,
        'annotation_hint': annotation_hint,
        'annotation_model': getattr(chat_client, 'model', None),
        'naming_prompt_version': NAMING_PROMPT_VERSION,
        'narrative_prompt_version': NARRATIVE_PROMPT_VERSION,
        'naming_marker_types': sorted(resolved_markers),
```

- [ ] **Step 8: Add the new outputs to the return dict** — find (lines 864–866):

Find:
```python
    return {'root': str(lay.root), 'crosstab': crosstab, 'panel': panel,
            'partition_info': partition_info, 'canonical_deg': canonical_deg,
            'skipped': skipped}
```
Replace with:
```python
    return {'root': str(lay.root), 'crosstab': crosstab, 'panel': panel,
            'partition_info': partition_info, 'canonical_deg': canonical_deg,
            'core_names': core_names_df, 'narratives': narratives_df,
            'skipped': skipped}
```

- [ ] **Step 9: Update the docstring stage list** — find (lines 622–624):

Find:
```python
    the unit is not named in ``force`` (a subset of {'partition','dissect',
    'diagnosis','canonical','profile'}, or 'all'). Existing
```
Replace with:
```python
    the unit is not named in ``force`` (a subset of {'partition','dissect',
    'diagnosis','canonical','naming','narrative','profile'}, or 'all'). Existing
```

- [ ] **Step 10: Byte-compile and smoke-test the import graph + stages** (dev node)

```bash
cd /scratch/users/chensj16/projects/standissect
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python - <<'PY'
import standissect
from standissect.pipeline import _STAGES, _normalize_force
assert 'naming' in _STAGES and 'narrative' in _STAGES, _STAGES
assert _normalize_force('naming') == {'naming'}
assert _normalize_force('narrative') == {'narrative'}
assert _normalize_force('all') == set(_STAGES)
print("pipeline import + stages OK:", _STAGES)
PY
```
Expected: `pipeline import + stages OK: (...'naming', 'narrative', 'profile')` with no ImportError (confirms no circular import).

- [ ] **Step 11: Re-run the full lightweight suites**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_annotate.py tests/test_diagnosis_llm.py -v
```
Expected: 24 passed (15 + 9).

---

## Task 7: cli.py — default llm, force choices, annotation flags

**Files:**
- Modify: `cli.py`
- Test: create `tests/test_cli.py`

**Interfaces:**
- Produces: `--diagnosis-mode` default `llm`; `--force` adds `naming`/`narrative`; `--annotation-hint` (default `''`); `--naming-markers` (default `None`); both passed through to `run_dissect_pipeline`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_cli.py`

```python
import pathlib
import sys

# cli.py uses relative imports, so import it as standissect.cli (this pulls
# standissect/__init__.py -> scanpy; run on a dev node).
_PKG_PARENT = pathlib.Path(__file__).resolve().parents[2]   # .../projects
sys.path.insert(0, str(_PKG_PARENT))

from standissect.cli import build_parser  # noqa: E402


def test_cli_diagnosis_mode_defaults_to_llm():
    a = build_parser().parse_args(
        ["run", "x.h5ad", "--cluster-col", "leiden", "--output-dir", "o"])
    assert a.diagnosis_mode == "llm"
    assert a.annotation_hint == ""
    assert a.naming_markers is None


def test_cli_accepts_naming_force_and_annotation_flags():
    a = build_parser().parse_args(
        ["run", "x.h5ad", "--cluster-col", "leiden", "--output-dir", "o",
         "--force", "naming", "--force", "narrative",
         "--annotation-hint", "synovial tissue", "--naming-markers", "m.tsv"])
    assert "naming" in a.force and "narrative" in a.force
    assert a.annotation_hint == "synovial tissue"
    assert a.naming_markers == "m.tsv"
```

- [ ] **Step 2: Run to verify they fail**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_cli.py -v
```
Expected: `test_cli_diagnosis_mode_defaults_to_llm` fails (`'rule' != 'llm'`); the second fails on unrecognized `--annotation-hint`/`--naming-markers` and invalid `--force` choice `naming`.

- [ ] **Step 3a: Implement — default to llm** — in `cli.py`, find (lines 70–71):

Find:
```python
    diag.add_argument('--diagnosis-mode', choices=('rule', 'llm', 'hybrid'),
                      default='rule', help='Diagnosis engine. Default: rule.')
```
Replace with:
```python
    diag.add_argument('--diagnosis-mode', choices=('rule', 'llm', 'hybrid'),
                      default='llm', help='Diagnosis engine. Default: llm '
                      '(falls back to rule when no ARK key is available).')
    diag.add_argument('--annotation-hint', default='',
                      help='Optional tissue/context hint for naming + narrative, '
                           'e.g. "synovial tissue, OA vs RA".')
    diag.add_argument('--naming-markers',
                      help='Optional TSV (cell_type<TAB>gene,gene,...) for the '
                           'local naming backup. Defaults to a bundled marker set.')
```

- [ ] **Step 3b: Implement — force choices** — find (lines 82–84):

Find:
```python
    rerun.add_argument('--force', action='append', default=(),
                       choices=('partition', 'dissect', 'diagnosis', 'canonical', 'profile', 'all'),
                       help='Stage to recompute. Repeatable; use all for every stage.')
```
Replace with:
```python
    rerun.add_argument('--force', action='append', default=(),
                       choices=('partition', 'dissect', 'diagnosis', 'canonical',
                                'naming', 'narrative', 'profile', 'all'),
                       help='Stage to recompute. Repeatable; use all for every stage.')
```

- [ ] **Step 3c: Implement — pass-through** — find (lines 136–137):

Find:
```python
        diagnosis_ark_api_key_env=args.ark_api_key_env,
        diagnosis_fallback_to_rule=not args.no_diagnosis_fallback,
```
Replace with:
```python
        diagnosis_ark_api_key_env=args.ark_api_key_env,
        diagnosis_fallback_to_rule=not args.no_diagnosis_fallback,
        annotation_hint=args.annotation_hint,
        naming_markers=_none_if_empty(args.naming_markers),
```

- [ ] **Step 4: Run to verify they pass**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_cli.py -v
```
Expected: 2 passed.

---

## Task 8: report.py — cell-type header + narrative paragraph

**Files:**
- Modify: `report.py`
- Test: create `tests/test_report.py`

**Interfaces:**
- Produces: `build_report` renders `cluster N — {cell_type}` headers (sidebar + main), a narrative paragraph per cluster, and a core-names table in the overview; tolerant of missing `core_names.tsv` / `narratives.tsv`.

- [ ] **Step 1: Write the failing test** — create `tests/test_report.py`

```python
import pathlib
import sys

import pandas as pd

# report.py has no relative imports (stdlib + pandas only) -> import top-level,
# avoiding standissect/__init__.py's scanpy import.
_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]   # .../standissect
sys.path.insert(0, str(_PKG_DIR))

import report  # noqa: E402


def test_build_report_includes_name_and_narrative(tmp_path):
    root = tmp_path / "out" / "leiden"
    (root / "clusters" / "c0").mkdir(parents=True)
    pd.DataFrame({"parent_cluster": ["c0_1"], "subcluster": ["c0_1"],
                  "likely_cause": ["doublet-driven"]}
                 ).to_csv(root / "clusters" / "c0" / "panel.tsv", sep="\t", index=False)
    pd.DataFrame({"parent_cluster": ["0"], "core_subcluster": ["c0_0"],
                  "cell_type": ["T cell"], "confidence": [0.9], "rationale": ["r"],
                  "source": ["llm"], "model": ["m"]}
                 ).to_csv(root / "core_names.tsv", sep="\t", index=False)
    pd.DataFrame({"parent_cluster": ["0"], "cell_type": ["T cell"],
                  "narrative": ["A clean T cell cluster."]}
                 ).to_csv(root / "narratives.tsv", sep="\t", index=False)
    out = report.build_report(str(root))
    html = pathlib.Path(out).read_text(encoding="utf-8")
    assert "cluster 0 — T cell" in html
    assert "A clean T cell cluster." in html


def test_build_report_tolerates_missing_annotation(tmp_path):
    root = tmp_path / "out" / "leiden"
    (root / "clusters" / "c0").mkdir(parents=True)
    pd.DataFrame({"parent_cluster": ["c0_1"], "subcluster": ["c0_1"]}
                 ).to_csv(root / "clusters" / "c0" / "panel.tsv", sep="\t", index=False)
    out = report.build_report(str(root))          # no core_names/narratives
    html = pathlib.Path(out).read_text(encoding="utf-8")
    assert 'id="c0"' in html                       # still renders the cluster
```

- [ ] **Step 2: Run to verify it fails**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_report.py -v
```
Expected: `test_build_report_includes_name_and_narrative` fails (`'cluster 0 — T cell' not in html`); the tolerance test passes.

- [ ] **Step 3a: Add `import html`** — find (lines 10–13):

Find:
```python
from __future__ import annotations
import base64
import sys
from pathlib import Path
```
Replace with:
```python
from __future__ import annotations
import base64
import html as _html
import sys
from pathlib import Path
```

- [ ] **Step 3b: Add loaders** — find the `_table` function end (lines 44–45):

Find:
```python
    return note + df.to_html(index=False, border=0, classes='deg',
                             float_format=lambda x: f'{x:.3g}')
```
Replace with:
```python
    return note + df.to_html(index=False, border=0, classes='deg',
                             float_format=lambda x: f'{x:.3g}')


def _read_tsv_safe(path):
    try:
        return pd.read_csv(path, sep='\t')
    except Exception:
        return pd.DataFrame()


def _load_core_names_map(path):
    """core_names.tsv -> {parent_cluster: cell_type} (only named clusters)."""
    df = _read_tsv_safe(path)
    out = {}
    if len(df) and 'parent_cluster' in df.columns and 'cell_type' in df.columns:
        for _, r in df.iterrows():
            ct = r.get('cell_type')
            if ct is not None and not (isinstance(ct, float) and pd.isna(ct)) and str(ct) != 'nan':
                out[str(r['parent_cluster'])] = str(ct)
    return out


def _load_narratives_map(path):
    """narratives.tsv -> {parent_cluster: narrative}."""
    df = _read_tsv_safe(path)
    out = {}
    if len(df) and 'parent_cluster' in df.columns and 'narrative' in df.columns:
        for _, r in df.iterrows():
            nv = r.get('narrative')
            if nv is not None and not (isinstance(nv, float) and pd.isna(nv)) and str(nv) != 'nan':
                out[str(r['parent_cluster'])] = str(nv)
    return out
```

- [ ] **Step 3c: Add the `.narrative` CSS** — find the `.muted` rule (line 68) at the end of `_CSS`:

Find:
```python
.muted{color:#889;font-size:11px;margin:2px 0;}
"""
```
Replace with:
```python
.muted{color:#889;font-size:11px;margin:2px 0;}
.narrative{font-size:13px;color:#333;background:#f6f8fc;border-left:3px solid #4a6da7;
  padding:8px 12px;margin:6px 0 12px;line-height:1.5;}
"""
```

- [ ] **Step 3d: Load the maps + use names in the sidebar** — find (lines 77–80):

Find:
```python
    cluster_ids = sorted(
        (d.name[1:] for d in clusters_dir.glob('c*') if d.is_dir()),
        key=lambda x: int(x) if x.isdigit() else 10**9,
    )
```
Replace with:
```python
    cluster_ids = sorted(
        (d.name[1:] for d in clusters_dir.glob('c*') if d.is_dir()),
        key=lambda x: int(x) if x.isdigit() else 10**9,
    )
    core_names = _load_core_names_map(root / 'core_names.tsv')
    narratives = _load_narratives_map(root / 'narratives.tsv')
```

- [ ] **Step 3e: Sidebar entries show the name** — find (lines 91–92):

Find:
```python
    for cid in cluster_ids:
        h.append(f'<a href="#c{cid}">cluster {cid}</a>')
```
Replace with:
```python
    for cid in cluster_ids:
        nm = core_names.get(str(cid))
        label = f'cluster {cid}' + (f' — {_html.escape(nm)}' if nm else '')
        h.append(f'<a href="#c{cid}">{label}</a>')
```

- [ ] **Step 3f: Overview core-names table** — find (lines 107–108):

Find:
```python
    h.append('<div class="cap">minor sub-population panel — all clusters</div>')
    h.append(_table(root / 'panel.tsv'))
```
Replace with:
```python
    h.append('<div class="cap">minor sub-population panel — all clusters</div>')
    h.append(_table(root / 'panel.tsv'))
    core_names_html = _table(root / 'core_names.tsv')
    if core_names_html:
        h.append('<div class="cap">canonical-core cell-type names</div>')
        h.append(core_names_html)
```

- [ ] **Step 3g: Per-cluster header + narrative** — find (lines 111–114):

Find:
```python
    for cid in cluster_ids:
        cdir = clusters_dir / f'c{cid}'
        h.append(f'<h2 id="c{cid}">cluster {cid}</h2>')
        h.append('<div class="imgrow">')
```
Replace with:
```python
    for cid in cluster_ids:
        cdir = clusters_dir / f'c{cid}'
        nm = core_names.get(str(cid))
        title = f'cluster {cid}' + (f' — {_html.escape(nm)}' if nm else '')
        h.append(f'<h2 id="c{cid}">{title}</h2>')
        narr = narratives.get(str(cid))
        if narr:
            h.append(f'<p class="narrative">{_html.escape(narr)}</p>')
        h.append('<div class="imgrow">')
```

- [ ] **Step 4: Run to verify it passes**

```bash
srun -p dev -c 2 --mem=4G -t 00:15:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_report.py -v
```
Expected: 2 passed.

---

## Task 9: __init__.py exports + full-suite gate

**Files:**
- Modify: `__init__.py`
- Verify: full suite + import smoke

**Interfaces:**
- Produces: top-level exports `CoreNaming`, `ClusterNarrative`, `LocalNamingEngine`, `NarrativeEngine`, `make_naming_engine`, `make_chat_client`.

- [ ] **Step 1: Add the diagnosis + annotate exports** — find (lines 28–36):

Find:
```python
from .diagnosis import (
    ArkChatClient,
    DiagnosisResult,
    MinorEvidence,
    RuleDiagnosisEngine,
    build_minor_evidence,
    make_diagnosis_engine,
    normalize_diagnosis_roles,
)
```
Replace with:
```python
from .diagnosis import (
    ArkChatClient,
    DiagnosisResult,
    MinorEvidence,
    RuleDiagnosisEngine,
    build_minor_evidence,
    make_chat_client,
    make_diagnosis_engine,
    normalize_diagnosis_roles,
)
from .annotate import (
    ClusterNarrative,
    CoreNaming,
    LocalNamingEngine,
    NarrativeEngine,
    make_naming_engine,
)
```

- [ ] **Step 2: Extend `__all__`** — find (lines 39–48):

Find:
```python
__all__ = [
    "run_dissect_pipeline",
    "build_report",
    "ArkChatClient",
    "DiagnosisResult",
    "MinorEvidence",
    "RuleDiagnosisEngine",
    "build_minor_evidence",
    "make_diagnosis_engine",
    "normalize_diagnosis_roles",
```
Replace with:
```python
__all__ = [
    "run_dissect_pipeline",
    "build_report",
    "ArkChatClient",
    "DiagnosisResult",
    "MinorEvidence",
    "RuleDiagnosisEngine",
    "build_minor_evidence",
    "make_chat_client",
    "make_diagnosis_engine",
    "normalize_diagnosis_roles",
    "ClusterNarrative",
    "CoreNaming",
    "LocalNamingEngine",
    "NarrativeEngine",
    "make_naming_engine",
```

- [ ] **Step 3: Import smoke + full suite** (dev node)

```bash
cd /scratch/users/chensj16/projects/standissect
srun -p dev -c 2 --mem=4G -t 00:20:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python - <<'PY'
import standissect
for name in ["CoreNaming", "ClusterNarrative", "LocalNamingEngine",
             "NarrativeEngine", "make_naming_engine", "make_chat_client"]:
    assert hasattr(standissect, name), name
print("exports OK")
PY

srun -p dev -c 2 --mem=4G -t 00:20:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/ -v
```
Expected: `exports OK`, then all tests pass (`test_annotate` 15 + `test_diagnosis_llm` 9 + `test_cli` 2 + `test_report` 2 = 28 passed).

- [ ] **Step 4: Report final state to the user (no commit)**

Summarize: branch `feat/annotate-naming-narrative`, files created/modified, test counts. Leave all changes uncommitted/unstaged for the user to review (do **not** `git add`/`git commit`).

---

## Manual end-to-end validation (optional, after user review)

A real ARK-keyed run is not a unit test (needs a GPU-free dev node, an `.h5ad`, network, and `ARK_API_KEY`). Suggested smoke once the user wants it, on a dev node with the key exported:

```bash
srun -p dev -c 4 --mem=16G -t 00:30:00 \
  /scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m standissect.cli run \
    <input.h5ad> --cluster-col <col> --output-dir <out> \
    --sample-col <s> --mito-col <m> \
    --annotation-hint "synovial tissue"
```
Then confirm `<out>/<col>/core_names.tsv`, `narratives.tsv`, and `report.html` (per-cluster `cluster N — <type>` + narrative). Without `ARK_API_KEY`: the run still completes — diagnosis degrades to rule, naming emits local marker-overlap names, narrative is skipped (the degrade warning prints).

---

## Self-Review

**Spec coverage** (`2026-06-25-core-naming-narrative-design.md`):
- §2.1 default `rule`→`llm` + construction-time graceful degrade → Task 6 (pipeline default + warn) + Task 7 (cli default) + Task 1 (`make_chat_client` never raises). ✓
- §2.2 new `annotate.py` → Tasks 2–5. ✓
- §2.3 naming knowledge-allowed prompt; narrative evidence-grounded → Task 3 `build_core_naming_prompt` / Task 4 `build_narrative_prompt`. ✓
- §2.4 naming LLM-primary + local backup, always runs; narrative LLM-only → Task 3 `LLMNamingEngine`/`make_naming_engine`, Task 5 `run_naming_stage` (always) / `run_narrative_stage` (gated by client in Task 6). ✓
- §2.5 `naming_markers` (dict or TSV) + bundled `DEFAULT_MARKER_SETS` → Task 2 `load_marker_sets`/`DEFAULT_MARKER_SETS`; Task 6/7 plumbing. ✓
- §3 shared `make_chat_client` → Task 1 (placed in `diagnosis.py` to avoid an `annotate↔diagnosis` import cycle — a deliberate refinement of the spec's "annotate.py" wording; documented in the Architecture note). ✓
- §3.1/§3.2 dataclasses + engines (`source∈{llm,local,unnamed}` / `{llm,skipped}`) → Tasks 2–4. ✓
- §3.3 prompts → Tasks 3–4. ✓
- §4 `_STAGES`, `_Layout`, stages, idempotency (`_naming_current`/`_narrative_current` on `prompt_version`+`model`), force cascade, `params.json`, return dict → Tasks 5–6. ✓
- §5 cli flags → Task 7. ✓
- §6 report → Task 8. ✓
- §7 testing → Tasks 2–8 (engines, parsers, overlap, stages, degrade, idempotency, report, cli). ✓
- §8 YAGNI (no SingleR/CellTypist, no new deps) / §9 risks (no-key degrade tested; marker change needs `--force naming`; `llm_client.py` untouched) → honored across tasks + Global Constraints. ✓

**Hint placement note:** the spec listed `hint` on both `CoreEvidence` and `make_naming_engine`. This plan carries `hint` on the evidence only (mirrors diagnosis's "config travels in the evidence packet" pattern and makes `naming.input.json` self-contained); `make_naming_engine` has no `hint` param. The stage injects `annotation_hint` when building evidence.

**Type consistency:** `CoreEvidence`/`CoreNaming`/`ClusterNarrativeEvidence`/`ClusterNarrative` field names and the `run_naming_stage`/`run_narrative_stage`/`make_naming_engine`/`make_chat_client` signatures are identical wherever referenced across tasks. `core_names.tsv` columns (`CORE_NAME_COLS`) match the report loader keys (`parent_cluster`, `cell_type`); `narratives.tsv` columns (`NARRATIVE_COLS`) match `_load_narratives_map`. The naming idempotency model = `engine.model` (engine config, preserved through local fallback), consistent between `LLMNamingEngine.name`, `_naming_current`, and `run_naming_stage`.

**Placeholder scan:** none — every code/test step contains complete content; every Run step has an exact command + expected result.
