# Interactive Review Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `standissect serve` — a FastAPI web app that projects a dissect run-output tree into an interactive dashboard for recording per-minor KEEP/DISCARD/UNCERTAIN verdicts, plus an interactive Plotly UMAP with lasso selection (inspect / export barcodes / manual keep-discard sets), fed by a new `standissect export-coords` tool.

**Architecture:** Three new flat modules (`review_store.py`, `export_coords.py`, `webreview.py`) + a `webreview_static/` SPA + two CLI subcommands. The server only *reads* the run tree (TSV/PNG/JSON/`cell_coords.tsv.gz`) and only *writes* decision/selection files; it never imports anndata or opens the 7 GB `.h5ad`. Coordinates come from `export-coords`, run separately in `sh_dev`.

**Tech Stack:** Python stdlib + pandas (stores), FastAPI + uvicorn (server), Plotly.js scattergl (browser UMAP), pytest + FastAPI TestClient (tests).

## Global Constraints

- **Sherlock policy:** never run Python / pip / pytest on the login node. All Python execution (tests, `export-coords`) goes through a compute node: prefix with `srun -p dev -t 00:20:00 --mem=8G -c 2 …` or inside `sh_dev`. Job I/O on `$SCRATCH`.
- **Import discipline (matches the repo):** modules cross-referencing siblings use the dual-import shim so tests can import them top-level without triggering `__init__.py`'s scanpy import:
  ```python
  try:                       # package use (standissect.webreview)
      from .review_store import ReviewStore, ManualStore
  except ImportError:        # standalone use (tests import top-level)
      from review_store import ReviewStore, ManualStore
  ```
- **Optional deps:** `fastapi`/`uvicorn`/`httpx` are NOT required to import `standissect` or run the pipeline. `serve` raises a clear `pip install` hint if missing; web tests are guarded with `pytest.importorskip`.
- **Decisions-file-only:** no `.h5ad` writes, no apply-discard wiring. Files are *shaped* for a future apply step but not consumed by one here.
- **Atomic writes:** every file write goes temp-in-same-dir → `os.replace`.
- **Package layout:** repo dir IS the package `standissect`; parent is `/home/users/chensj16/s/projects`. Test files insert `parents[1]` (the package dir) on `sys.path` for top-level imports, or `parents[2]` for `from standissect.x import`.

## File Structure

| File | Responsibility |
|---|---|
| `review_store.py` (create) | `ReviewStore` (human_review.tsv) + `ManualStore` (manual_cells.tsv + selections/). Pure stdlib+pandas, no sibling imports. |
| `export_coords.py` (create) | `export_cell_coords()`: backed-mode h5ad → `cell_coords.tsv.gz`. Lazy anndata import. |
| `webreview.py` (create) | `build_app(root,…) -> FastAPI`, `serve(…)`, run-tree readers. Dual-imports review_store + report helpers. |
| `webreview_static/index.html`, `app.js`, `style.css`, `plotly.min.js` (create) | The SPA + vendored Plotly. |
| `cli.py` (modify) | add `serve` + `export-coords` subparsers and their `_cmd` funcs. |
| `__init__.py` (modify) | export `build_app`, `serve` (lazy, no fastapi at import time). |
| `tests/test_review_store.py` (create) | store unit tests (no web deps). |
| `tests/test_webreview.py` (create) | API tests via TestClient + synthetic run tree. |
| `tests/test_export_coords.py` (create) | exporter test via synthetic AnnData. |
| `README.md` (modify) | document `serve` + `export-coords`. |

Reuses from `report.py`: `_read_tsv_safe`, `_load_core_names_map`, `_load_narratives_map`.

---

## Task 1: ReviewStore + ManualStore

**Files:** Create `review_store.py`; Test `tests/test_review_store.py`.

**Interfaces — Produces:**
- `ReviewStore(path, reviewer="")` · `.get(subcluster)->dict|None` · `.get_all()->dict` · `.set(subcluster, parent_cluster, llm_disposition, human_disposition, note="", timestamp=None)->dict|None` (empty `human_disposition` clears; invalid→`ValueError`) · `.progress()->{"decided":int}`
- `ManualStore(root, reviewer="")` · `.write_selection(label, barcodes)->{"path","n"}` · `.add_manual(label, barcodes, disposition, timestamp=None)->{"n","total"}` (invalid disposition→`ValueError`)
- module consts `REVIEW_COLUMNS`, `MANUAL_COLUMNS`, helper `_slugify(label)`

- [ ] **Step 1: Write the failing test** — `tests/test_review_store.py`:

```python
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest
from review_store import ReviewStore, ManualStore, _slugify


def test_set_get_roundtrip_and_resume(tmp_path):
    p = tmp_path / "hr.tsv"
    s = ReviewStore(p, reviewer="alice")
    s.set("c14_1", "14", "KEEP", "discard", note="junk", timestamp="t0")
    assert s.get("c14_1")["human_disposition"] == "DISCARD"
    assert p.exists()
    s2 = ReviewStore(p)                       # resume from disk
    assert s2.get("c14_1")["human_disposition"] == "DISCARD"
    assert s2.get("c14_1")["reviewer"] == "alice"


def test_clear_decision(tmp_path):
    p = tmp_path / "hr.tsv"
    s = ReviewStore(p)
    s.set("c1_1", "1", "KEEP", "KEEP", timestamp="t")
    s.set("c1_1", "1", "KEEP", "", timestamp="t")     # empty clears
    assert s.get("c1_1") is None
    assert s.progress()["decided"] == 0


def test_invalid_disposition(tmp_path):
    s = ReviewStore(tmp_path / "hr.tsv")
    with pytest.raises(ValueError):
        s.set("c1_1", "1", "", "BOGUS")


def test_atomic_no_tmp_left(tmp_path):
    p = tmp_path / "hr.tsv"
    ReviewStore(p).set("c1_1", "1", "", "KEEP", timestamp="t")
    assert not list(tmp_path.glob("*.tmp"))


def test_manual_store(tmp_path):
    m = ManualStore(tmp_path, reviewer="bob")
    r = m.write_selection("my sel!", ["A", "B"])
    assert (tmp_path / "selections" / "selection_my_sel.tsv").exists()
    assert r["n"] == 2
    assert m.add_manual("set 1", ["A", "B", "C"], "discard", timestamp="t")["n"] == 3
    m.add_manual("set 2", ["D"], "KEEP", timestamp="t")            # appends
    df = pd.read_csv(tmp_path / "manual_cells.tsv", sep="\t")
    assert len(df) == 4 and list(df["disposition"])[:3] == ["DISCARD"] * 3


def test_slugify():
    assert _slugify("a b/c!") == "a_b_c"
    assert _slugify("   ") == "selection"
```

- [ ] **Step 2: Run, verify it fails** — `srun -p dev -t 00:10:00 --mem=4G -c 1 python -m pytest tests/test_review_store.py -q` → FAIL (`No module named review_store`).

- [ ] **Step 3: Implement `review_store.py`:**

```python
"""standissect.review_store — flat-file stores for human review decisions.

No web/anndata dependency; importable top-level (tests) or as a package
module. ``ReviewStore`` owns ``human_review.tsv`` (per-minor verdicts);
``ManualStore`` owns ``manual_cells.tsv`` + ``selections/`` (per-cell
hand-picked sets). All writes are atomic (temp + os.replace).
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re

import pandas as pd

_VALID = {"KEEP", "DISCARD", "UNCERTAIN"}
REVIEW_COLUMNS = ["subcluster", "parent_cluster", "llm_disposition",
                  "human_disposition", "note", "reviewer", "updated_at"]
MANUAL_COLUMNS = ["barcode", "disposition", "label", "reviewer", "updated_at"]


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_tsv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def _slugify(label):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label).strip())
    return s.strip("_") or "selection"


class ReviewStore:
    def __init__(self, path, reviewer=""):
        self.path = Path(path)
        self.reviewer = reviewer
        self._rows = {}
        if self.path.exists():
            self._load()

    def _load(self):
        try:
            df = pd.read_csv(self.path, sep="\t", dtype=str).fillna("")
        except Exception:
            return
        for _, r in df.iterrows():
            sc = str(r.get("subcluster", "")).strip()
            if sc:
                self._rows[sc] = {c: str(r.get(c, "")) for c in REVIEW_COLUMNS}

    def get(self, subcluster):
        return self._rows.get(str(subcluster))

    def get_all(self):
        return dict(self._rows)

    def set(self, subcluster, parent_cluster, llm_disposition,
            human_disposition, note="", timestamp=None):
        sc = str(subcluster).strip()
        hd = (human_disposition or "").strip().upper()
        if hd and hd not in _VALID:
            raise ValueError(f"invalid disposition: {human_disposition!r}")
        if not hd:
            self._rows.pop(sc, None)
        else:
            self._rows[sc] = {
                "subcluster": sc,
                "parent_cluster": str(parent_cluster),
                "llm_disposition": str(llm_disposition or ""),
                "human_disposition": hd,
                "note": str(note or ""),
                "reviewer": self.reviewer,
                "updated_at": str(timestamp if timestamp is not None else _now()),
            }
        self._flush()
        return self._rows.get(sc)

    def _flush(self):
        df = pd.DataFrame(list(self._rows.values()), columns=REVIEW_COLUMNS)
        _atomic_write_tsv(df, self.path)

    def progress(self):
        return {"decided": len(self._rows)}


class ManualStore:
    def __init__(self, root, reviewer=""):
        self.root = Path(root)
        self.manual_path = self.root / "manual_cells.tsv"
        self.sel_dir = self.root / "selections"
        self.reviewer = reviewer

    def write_selection(self, label, barcodes):
        path = self.sel_dir / f"selection_{_slugify(label)}.tsv"
        df = pd.DataFrame({"barcode": [str(b) for b in barcodes]})
        _atomic_write_tsv(df, path)
        return {"path": str(path), "n": int(len(df))}

    def add_manual(self, label, barcodes, disposition, timestamp=None):
        d = (disposition or "").strip().upper()
        if d not in _VALID:
            raise ValueError(f"invalid disposition: {disposition!r}")
        ts = str(timestamp if timestamp is not None else _now())
        new = pd.DataFrame({
            "barcode": [str(b) for b in barcodes],
            "disposition": d, "label": _slugify(label),
            "reviewer": self.reviewer, "updated_at": ts,
        }, columns=MANUAL_COLUMNS)
        if self.manual_path.exists():
            old = pd.read_csv(self.manual_path, sep="\t", dtype=str).fillna("")
            new = pd.concat([old, new], ignore_index=True)
        _atomic_write_tsv(new, self.manual_path)
        return {"n": int(len(barcodes)), "total": int(len(new))}
```

- [ ] **Step 4: Run, verify pass** — `srun -p dev -t 00:10:00 --mem=4G -c 1 python -m pytest tests/test_review_store.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add review_store.py tests/test_review_store.py && git commit -m "feat(webreview): ReviewStore + ManualStore flat-file decision stores"`.

---

## Task 2: webreview run-tree readers + read endpoints

**Files:** Create `webreview.py` (readers + `build_app` with GET `/api/run`, `/api/cluster/{cid}`, `/api/image`, `/api/table`, `/`, static mount); add to `tests/test_webreview.py`.

**Interfaces — Consumes:** `ReviewStore` from Task 1. **Produces:** `build_app(root, decisions_file=None, reviewer="") -> FastAPI`; module-level readers `_list_clusters(root)`, `_run_payload(root, store)`, `_cluster_payload(root, cid, store)`, `_parent_of(subcluster)`.

- [ ] **Step 1: Write failing tests** — create `tests/test_webreview.py` with the synthetic-run helper and the read-endpoint tests:

```python
import base64, pathlib, sys
import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from fastapi.testclient import TestClient
import webreview

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def _make_run(root):
    c14 = root / "clusters" / "c14"
    c14.mkdir(parents=True)
    panel = pd.DataFrame({
        "parent_cluster": ["14", "14"], "subcluster": ["c14_1", "c14_2"],
        "n_cells": [100, 50], "frac_of_parent": [0.3, 0.15],
        "top5_up_genes": ["A,B", "C"], "top5_down_genes": ["", ""],
        "likely_cause": ["biology-candidate", "sample-driven"],
        "diagnosis_confidence": [0.85, 0.9],
        "diagnosis_rationale": ["r1", "r2"],
        "recommended_disposition": ["KEEP", "UNCERTAIN"],
        "proposed_cell_type": ["neutrophil", ""], "disposition_reason": ["", ""]})
    panel.to_csv(c14 / "panel.tsv", sep="\t", index=False)
    panel.to_csv(root / "panel.tsv", sep="\t", index=False)
    pd.DataFrame({"subcluster": ["c14_0"] * 200 + ["c14_1"] * 100
                  + ["c14_2"] * 50 + ["c14_5"] * 10}).to_csv(
        c14 / "subcluster_labels.tsv", sep="\t", index=False)
    pd.DataFrame({"parent_cluster": ["14"], "cell_type": ["Neutrophil"]}).to_csv(
        root / "core_names.tsv", sep="\t", index=False)
    pd.DataFrame({"parent_cluster": ["14"], "narrative": ["a story"]}).to_csv(
        root / "narratives.tsv", sep="\t", index=False)
    pd.DataFrame({"gene": ["A", "B"], "score": [3.1, 2.0]}).to_csv(
        c14 / "deg_c14_1.tsv", sep="\t", index=False)
    (c14 / "minor_profile.png").write_bytes(_PNG)
    bcs = [f"cell{i}" for i in range(8)]
    pd.DataFrame({"": bcs,
                  "umap_cluster": ["c14_0", "c14_0", "c14_1", "c14_1",
                                   "c14_2", "c14_2", "c14_5", "c14_0"],
                  "recommended_disposition": ["", "", "KEEP", "KEEP",
                                              "UNCERTAIN", "UNCERTAIN", "", ""],
                  "proposed_cell_type": [""] * 8}).to_csv(
        root / "cell_labels.tsv", sep="\t", index=False)
    pd.DataFrame({"barcode": bcs, "umap_x": range(8), "umap_y": range(8),
                  "pct_counts_mt": [0.1] * 8}).to_csv(
        root / "cell_coords.tsv.gz", sep="\t", index=False, compression="gzip")


def _client(root, **kw):
    return TestClient(webreview.build_app(str(root), **kw))


def test_api_run(tmp_path):
    _make_run(tmp_path)
    j = _client(tmp_path).get("/api/run").json()
    assert j["has_coords"] is True
    assert j["totals"] == {"minors": 2, "decided": 0}
    assert j["clusters"][0]["cid"] == "14"
    assert j["clusters"][0]["core_name"] == "Neutrophil"
    assert j["clusters"][0]["n_minors"] == 2


def test_api_cluster(tmp_path):
    _make_run(tmp_path)
    j = _client(tmp_path).get("/api/cluster/14").json()
    assert j["core_name"] == "Neutrophil" and j["narrative"] == "a story"
    assert [m["subcluster"] for m in j["minors"]] == ["c14_1", "c14_2"]
    assert j["minors"][0]["recommended_disposition"] == "KEEP"
    assert j["minors"][0]["deg_table"] == "deg_c14_1.tsv"
    assert j["minors"][0]["human_disposition"] == ""
    kinds = {o["subcluster"]: o["kind"] for o in j["others"]}
    assert kinds["c14_0"] == "core" and kinds["c14_5"] == "below_threshold"


def test_api_cluster_404(tmp_path):
    _make_run(tmp_path)
    assert _client(tmp_path).get("/api/cluster/99").status_code == 404


def test_image_and_table(tmp_path):
    _make_run(tmp_path)
    c = _client(tmp_path)
    assert c.get("/api/image/14/minor_profile").status_code == 200
    assert c.get("/api/image/14/umap_subcluster").status_code == 404
    t = c.get("/api/table/14/deg_c14_1.tsv").json()
    assert "gene" in t["columns"] and len(t["rows"]) == 2
    assert c.get("/api/table/14/bogus").status_code == 400


def test_index_and_static(tmp_path):
    _make_run(tmp_path)
    c = _client(tmp_path)
    assert c.get("/").status_code == 200
    assert "standissect" in c.get("/").text.lower()
    assert c.get("/static/app.js").status_code == 200
```

- [ ] **Step 2: Run, verify it fails** — `srun -p dev -t 00:15:00 --mem=8G -c 2 python -m pytest tests/test_webreview.py -q` → FAIL (no `webreview`). (If it instead SKIPs, fastapi/httpx are missing — install once: `srun -p dev -t 00:15:00 --mem=8G -c 2 pip install --user fastapi uvicorn httpx`.)

- [ ] **Step 3: Implement `webreview.py`** (readers + app shell; the `/api/decision`, `/api/cells`, selection routes are added in Tasks 3 & 5 but include them now if writing the file in one pass — this task's tests only exercise the read routes + static). Minimum to pass this task:

```python
"""standissect.webreview — interactive review server for a dissect run tree.

Reads the run output (TSV/PNG/JSON/cell_coords) and writes only decision /
selection files via review_store. Never imports anndata / opens the h5ad.
"""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

try:                       # package use (standissect.webreview)
    from .review_store import ReviewStore, ManualStore
    from .report import _read_tsv_safe, _load_core_names_map, _load_narratives_map
except ImportError:        # standalone use (tests import top-level)
    from review_store import ReviewStore, ManualStore
    from report import _read_tsv_safe, _load_core_names_map, _load_narratives_map

_STATIC = Path(__file__).resolve().parent / "webreview_static"


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        f = float(v)
        return None if f != f else f       # drop NaN
    except (TypeError, ValueError):
        return None


def _parent_of(subcluster):
    m = re.match(r"c?(\d+)_", str(subcluster))
    return m.group(1) if m else str(subcluster)


def _list_clusters(root):
    d = Path(root) / "clusters"
    if not d.exists():
        return []
    return sorted((p.name[1:] for p in d.glob("c*") if p.is_dir()),
                  key=lambda x: int(x) if x.isdigit() else 10 ** 9)


def _minors_of(root, cid):
    panel = _read_tsv_safe(Path(root) / "clusters" / f"c{cid}" / "panel.tsv")
    if "subcluster" not in panel.columns:
        return panel.iloc[0:0], []
    return panel, [str(s) for s in panel["subcluster"]]


def _run_payload(root, store):
    root = Path(root)
    core_names = _load_core_names_map(root / "core_names.tsv")
    clusters, tot_m, tot_d = [], 0, 0
    for cid in _list_clusters(root):
        _, minors = _minors_of(root, cid)
        decided = sum(1 for s in minors if store.get(s))
        tot_m += len(minors)
        tot_d += decided
        clusters.append({"cid": str(cid), "core_name": core_names.get(str(cid), ""),
                         "n_minors": len(minors), "n_decided": decided})
    return {"root": str(root), "clusters": clusters,
            "totals": {"minors": tot_m, "decided": tot_d},
            "has_coords": (root / "cell_coords.tsv.gz").exists()}


def _cluster_payload(root, cid, store):
    root = Path(root)
    cdir = root / "clusters" / f"c{cid}"
    core_names = _load_core_names_map(root / "core_names.tsv")
    narratives = _load_narratives_map(root / "narratives.tsv")
    panel, minor_ids = _minors_of(root, cid)
    labels = _read_tsv_safe(cdir / "subcluster_labels.tsv")
    sizes = (labels["subcluster"].value_counts().to_dict()
             if "subcluster" in labels.columns else {})
    minors = []
    for _, r in panel.iterrows():
        sc = str(r["subcluster"])
        dec = store.get(sc) or {}
        minors.append({
            "subcluster": sc, "parent_cluster": str(r.get("parent_cluster", cid)),
            "n_cells": _to_int(r.get("n_cells")),
            "frac_of_parent": _to_float(r.get("frac_of_parent")),
            "top5_up_genes": str(r.get("top5_up_genes", "") or ""),
            "top5_down_genes": str(r.get("top5_down_genes", "") or ""),
            "likely_cause": str(r.get("likely_cause", "") or ""),
            "diagnosis_confidence": _to_float(r.get("diagnosis_confidence")),
            "diagnosis_rationale": str(r.get("diagnosis_rationale", "") or ""),
            "recommended_disposition": str(r.get("recommended_disposition", "") or ""),
            "proposed_cell_type": str(r.get("proposed_cell_type", "") or ""),
            "disposition_reason": str(r.get("disposition_reason", "") or ""),
            "human_disposition": dec.get("human_disposition", ""),
            "note": dec.get("note", ""), "reviewable": True,
            "deg_table": (f"deg_{sc}.tsv" if (cdir / f"deg_{sc}.tsv").exists() else None),
            "qc_table": (f"qc_drift_{sc}.tsv" if (cdir / f"qc_drift_{sc}.tsv").exists() else None),
        })
    reviewable = set(minor_ids)
    others = []
    for sc, n in sorted(sizes.items(), key=lambda kv: str(kv[0])):
        if str(sc) in reviewable:
            continue
        others.append({"subcluster": str(sc), "n_cells": int(n),
                       "reviewable": False,
                       "kind": "core" if str(sc).endswith("_0") else "below_threshold"})
    return {"cid": str(cid), "core_name": core_names.get(str(cid), ""),
            "narrative": narratives.get(str(cid), ""),
            "images": {"minor_profile": (cdir / "minor_profile.png").exists(),
                       "umap_subcluster": (cdir / "umap_subcluster.png").exists()},
            "minors": minors, "others": others}


def build_app(root, decisions_file=None, reviewer=""):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    root = Path(root)
    dec_path = Path(decisions_file) if decisions_file else root / "human_review.tsv"
    store = ReviewStore(dec_path, reviewer=reviewer)
    manual = ManualStore(root, reviewer=reviewer)

    app = FastAPI(title="standissect review")
    app.state.cells = None
    app.state.barcodes = None
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/run")
    def api_run():
        return _run_payload(root, store)

    @app.get("/api/cluster/{cid}")
    def api_cluster(cid: str):
        if not (root / "clusters" / f"c{cid}").exists():
            raise HTTPException(404, f"no cluster {cid}")
        return _cluster_payload(root, cid, store)

    @app.get("/api/image/{cid}/{name}")
    def api_image(cid: str, name: str):
        if name not in ("minor_profile", "umap_subcluster"):
            raise HTTPException(404, "unknown image")
        p = root / "clusters" / f"c{cid}" / f"{name}.png"
        if not p.exists():
            raise HTTPException(404, "missing image")
        return FileResponse(str(p), media_type="image/png")

    @app.get("/api/table/{cid}/{name}")
    def api_table(cid: str, name: str):
        if not re.fullmatch(r"(deg|qc_drift)_c\d+_\d+\.tsv", name):
            raise HTTPException(400, "bad table name")
        df = _read_tsv_safe(root / "clusters" / f"c{cid}" / name)
        return {"columns": list(df.columns),
                "rows": df.head(200).to_dict("records")}

    # decision + cells + selection routes are added in Tasks 3 and 5.
    _attach_write_routes(app, root, store, manual, BaseModel, HTTPException)
    return app


def _attach_write_routes(app, root, store, manual, BaseModel, HTTPException):
    """Filled in Task 3 (decision) and Task 5 (cells + selection)."""
    return
```

Then create the static dir so the static mount + index route work for `test_index_and_static`:
`mkdir -p webreview_static` and write minimal `index.html` (`<!doctype html><title>standissect review</title>`) and `app.js` (`// standissect review`). They are replaced fully in Task 4/6.

- [ ] **Step 4: Run, verify pass** — `srun -p dev -t 00:15:00 --mem=8G -c 2 python -m pytest tests/test_webreview.py -q` → the read-endpoint + static tests PASS (decision/cells tests not added yet).
- [ ] **Step 5: Commit** — `git add webreview.py webreview_static tests/test_webreview.py && git commit -m "feat(webreview): run-tree readers + read endpoints (run/cluster/image/table)"`.

---

## Task 3: POST /api/decision

**Files:** Modify `webreview.py` (`_attach_write_routes`); add tests to `tests/test_webreview.py`.

**Interfaces — Consumes:** `_run_payload`, `_parent_of`, `ReviewStore`. **Produces:** `POST /api/decision {subcluster, disposition, note}` → `{"progress": {minors, decided}}`; 400 on invalid disposition or unknown minor.

- [ ] **Step 1: Add failing tests** to `tests/test_webreview.py`:

```python
def test_decision_persist_and_validate(tmp_path):
    _make_run(tmp_path)
    c = _client(tmp_path)
    r = c.post("/api/decision",
               json={"subcluster": "c14_1", "disposition": "DISCARD", "note": "junk"})
    assert r.status_code == 200 and r.json()["progress"]["decided"] == 1
    saved = pd.read_csv(tmp_path / "human_review.tsv", sep="\t")
    assert saved.iloc[0]["human_disposition"] == "DISCARD"
    assert saved.iloc[0]["note"] == "junk"
    assert c.post("/api/decision",
                  json={"subcluster": "c14_1", "disposition": "BOGUS"}).status_code == 400
    assert c.post("/api/decision",
                  json={"subcluster": "c99_9", "disposition": "KEEP"}).status_code == 400


def test_decision_clear(tmp_path):
    _make_run(tmp_path)
    c = _client(tmp_path)
    c.post("/api/decision", json={"subcluster": "c14_1", "disposition": "KEEP"})
    r = c.post("/api/decision", json={"subcluster": "c14_1", "disposition": ""})
    assert r.json()["progress"]["decided"] == 0
```

- [ ] **Step 2: Run, verify fail** — `srun -p dev -t 00:15:00 --mem=8G -c 2 python -m pytest tests/test_webreview.py -k decision -q` → FAIL (404, route missing).

- [ ] **Step 3: Implement** — replace `_attach_write_routes` body with the decision route:

```python
def _attach_write_routes(app, root, store, manual, BaseModel, HTTPException):
    class Decision(BaseModel):
        subcluster: str
        disposition: str = ""
        note: str = ""

    @app.post("/api/decision")
    def api_decision(d: Decision):
        cid = _parent_of(d.subcluster)
        panel, ids = _minors_of(root, cid)
        if d.subcluster not in set(ids):
            raise HTTPException(400, f"unknown minor {d.subcluster}")
        row = panel[panel["subcluster"].astype(str) == d.subcluster]
        llm = str(row.iloc[0].get("recommended_disposition", "")) if len(row) else ""
        try:
            store.set(d.subcluster, cid, llm, d.disposition, d.note)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"progress": _run_payload(root, store)["totals"]}

    _attach_cells_routes(app, root, manual, BaseModel, HTTPException)  # Task 5
```

Add a stub so the file imports until Task 5: `def _attach_cells_routes(*a, **k): return`.

- [ ] **Step 4: Run, verify pass** — `srun -p dev -t 00:15:00 --mem=8G -c 2 python -m pytest tests/test_webreview.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add webreview.py tests/test_webreview.py && git commit -m "feat(webreview): POST /api/decision with validation + persistence"`.

---

## Task 4: SPA dashboard (index.html, app.js, style.css)

**Files:** Replace `webreview_static/index.html`, `app.js`, `style.css`.

**Interfaces — Consumes:** GET `/api/run`, `/api/cluster/{cid}`, `/api/image/{cid}/{name}`, `/api/table/{cid}/{name}`; POST `/api/decision`. **Produces:** a single-page dashboard. No unit test beyond the Task 2 smoke test (`/` 200 + `/static/app.js` 200); validated manually.

Required behavior (the contract the JS must honor):
- Top bar: title `standissect review`, a global progress chip (`decided/minors`), and a Dashboard / UMAP toggle (UMAP button hidden when `/api/run` `has_coords` is false; wired in Task 6).
- Left sidebar: one entry per cluster `cid — core_name` with a `n_decided/n_minors` badge; click loads that cluster.
- Cluster panel: core name + narrative; `<img src="/api/image/{cid}/minor_profile">` and `…/umap_subcluster` (hide on natural error); a table of `minors[]` — each row shows size, frac, top-up/down genes, likely_cause, confidence, the LLM `recommended_disposition`, three buttons **KEEP / DISCARD / UNCERTAIN** (highlight the active `human_disposition`), an "adopt LLM" link (sets disposition = `recommended_disposition`), and a note `<input>` (POST on blur). Each row expands to show `diagnosis_rationale` and lazily fetches `deg_table`/`qc_table` via `/api/table`.
- `others[]` rendered greyed/read-only with size + `kind` label.
- Every decision POSTs to `/api/decision`, then updates the row state + the sidebar badge + global chip from the returned `progress`.

- [ ] **Step 1: Write `index.html`** — shell with `<div id="topbar">`, `<div id="sidebar">`, `<div id="main">`, `<link rel="stylesheet" href="/static/style.css">`, `<script src="/static/app.js"></script>`, and (for Task 6) `<script src="/static/plotly.min.js"></script>` + a hidden `<div id="umap">`. Title text must contain "standissect".
- [ ] **Step 2: Write `style.css`** — the dashboard styling (reuse `report.py`'s palette: sidebar `#1e2330`, accent `#4a6da7`; buttons; greyed read-only rows; toast).
- [ ] **Step 3: Write `app.js`** — `fetch`-based renderers implementing the contract above (`loadRun()`, `loadCluster(cid)`, `renderMinor(m)`, `postDecision(sc, disp, note)`, `toast(msg)`).
- [ ] **Step 4: Smoke test** — `srun -p dev -t 00:10:00 --mem=8G -c 2 python -m pytest tests/test_webreview.py::test_index_and_static -q` → PASS. Then a manual launch check in Task 9.
- [ ] **Step 5: Commit** — `git add webreview_static && git commit -m "feat(webreview): dashboard SPA (cluster sidebar + inline keep/discard controls)"`.

---

## Task 5: serve CLI + export-coords CLI + __init__ exports

**Files:** Modify `cli.py`, `__init__.py`; add `tests/test_cli` cases (extend existing `tests/test_cli.py`).

**Interfaces — Produces:** `serve(root, host, port, decisions_file, reviewer)` and CLI subcommands `serve`, `export-coords`. `serve` raises `SystemExit` with a pip hint if uvicorn missing.

- [ ] **Step 1: Add failing CLI parser tests** to `tests/test_cli.py`:

```python
def test_cli_serve_defaults():
    a = build_parser().parse_args(["serve", "/run"])
    assert a.output_root == "/run" and a.host == "127.0.0.1" and a.port == 8050


def test_cli_export_coords():
    a = build_parser().parse_args(
        ["export-coords", "x.h5ad", "--output-dir", "/run",
         "--mito-col", "pct_mt", "--extra-qc-col", "foo"])
    assert a.output_dir == "/run" and a.umap_key == "X_umap"
    assert a.mito_col == "pct_mt" and a.extra_qc_col == ["foo"]
```

- [ ] **Step 2: Run, verify fail** — `srun -p dev -t 00:10:00 --mem=8G -c 2 python -m pytest tests/test_cli.py -k "serve or export_coords" -q` → FAIL.
- [ ] **Step 3: Implement** — in `cli.py` add `import os`, the two `_cmd` funcs and subparsers:

```python
def serve_cmd(args):
    from .webreview import serve
    serve(args.output_root, host=args.host, port=args.port,
          decisions_file=args.decisions_file, reviewer=args.reviewer)
    return 0


def export_coords_cmd(args):
    from .export_coords import export_cell_coords
    qc = [c for c in (args.doublet_score_col, args.mito_col,
                      args.feature_count_col, args.umi_count_col) if c]
    qc += list(args.extra_qc_col or [])
    print(f"[standissect] wrote "
          f"{export_cell_coords(args.h5ad, args.output_dir, umap_key=args.umap_key, qc_cols=tuple(qc))}")
    return 0
```

In `build_parser()`:

```python
    srv = sub.add_parser('serve', help='Interactive review server for a run.')
    srv.add_argument('output_root')
    srv.add_argument('--host', default='127.0.0.1')
    srv.add_argument('--port', type=int, default=8050)
    srv.add_argument('--decisions-file')
    srv.add_argument('--reviewer', default=os.environ.get('USER', ''))
    srv.set_defaults(func=serve_cmd)

    ec = sub.add_parser('export-coords', help='Export per-cell UMAP coords for serve.')
    ec.add_argument('h5ad')
    ec.add_argument('--output-dir', required=True)
    ec.add_argument('--umap-key', default='X_umap')
    ec.add_argument('--doublet-score-col')
    ec.add_argument('--mito-col')
    ec.add_argument('--feature-count-col')
    ec.add_argument('--umi-count-col')
    ec.add_argument('--extra-qc-col', action='append', default=[])
    ec.set_defaults(func=export_coords_cmd)
```

In `webreview.py` add:

```python
def serve(root, host="127.0.0.1", port=8050, decisions_file=None, reviewer=""):
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit("`standissect serve` needs fastapi + uvicorn:\n"
                         "  pip install fastapi uvicorn "
                         "(run in sh_dev, not the login node)") from e
    app = build_app(root, decisions_file=decisions_file, reviewer=reviewer)
    print(f"[standissect] review server for {Path(root).name} "
          f"-> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
```

In `__init__.py` add lazy exports (must not import fastapi at package import):

```python
def serve(*args, **kwargs):
    """Lazy wrapper so importing standissect never requires fastapi."""
    from .webreview import serve as _serve
    return _serve(*args, **kwargs)


def build_app(*args, **kwargs):
    from .webreview import build_app as _b
    return _b(*args, **kwargs)
```

and add `"serve"`, `"build_app"` to `__all__`.

- [ ] **Step 4: Run, verify pass** — `srun -p dev -t 00:15:00 --mem=8G -c 2 python -m pytest tests/test_cli.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add cli.py __init__.py webreview.py tests/test_cli.py && git commit -m "feat(cli): serve + export-coords subcommands; lazy package exports"`.

---

## Task 6: export_coords.py + export-coords logic

**Files:** Create `export_coords.py`; create `tests/test_export_coords.py`.

**Interfaces — Produces:** `export_cell_coords(h5ad_path, output_dir, umap_key="X_umap", qc_cols=()) -> str` (path to `cell_coords.tsv.gz`); `KeyError` if `umap_key` absent.

- [ ] **Step 1: Write failing test** — `tests/test_export_coords.py`:

```python
import pathlib, sys
import pytest
pytest.importorskip("anndata")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from export_coords import export_cell_coords


def test_export(tmp_path):
    import anndata as ad
    obs = pd.DataFrame({"mt": [0.1, 0.2, 0.3, 0.4, 0.5]},
                       index=[f"b{i}" for i in range(5)])
    a = ad.AnnData(X=np.zeros((5, 3)), obs=obs)
    a.obsm["X_umap"] = np.arange(10).reshape(5, 2).astype(float)
    h = tmp_path / "a.h5ad"
    a.write_h5ad(h)
    out = export_cell_coords(str(h), str(tmp_path), qc_cols=("mt", "missing"))
    df = pd.read_csv(out, sep="\t")
    assert list(df["barcode"]) == [f"b{i}" for i in range(5)]
    assert df["umap_x"].iloc[1] == 2.0 and df["umap_y"].iloc[1] == 3.0
    assert "mt" in df.columns and "missing" not in df.columns


def test_export_missing_key(tmp_path):
    import anndata as ad
    a = ad.AnnData(X=np.zeros((3, 2)))
    h = tmp_path / "b.h5ad"
    a.write_h5ad(h)
    with pytest.raises(KeyError):
        export_cell_coords(str(h), str(tmp_path), umap_key="X_umap")
```

- [ ] **Step 2: Run, verify fail** — `srun -p dev -t 00:15:00 --mem=8G -c 2 python -m pytest tests/test_export_coords.py -q` → FAIL.
- [ ] **Step 3: Implement `export_coords.py`:**

```python
"""standissect.export_coords — dump per-cell UMAP coords (+QC) for `serve`.

Reads the h5ad in backed mode (never loads X) and writes a lightweight
``cell_coords.tsv.gz`` (barcode, umap_x, umap_y, <qc...>) into the run dir.
No web dependency; anndata imported lazily.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def export_cell_coords(h5ad_path, output_dir, umap_key="X_umap", qc_cols=()):
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if umap_key not in adata.obsm:
            raise KeyError(f"{umap_key!r} not in obsm; have {list(adata.obsm)}")
        coords = np.asarray(adata.obsm[umap_key])[:, :2]
        df = pd.DataFrame({"barcode": adata.obs_names.astype(str),
                           "umap_x": coords[:, 0], "umap_y": coords[:, 1]})
        for c in qc_cols:
            if c and c in adata.obs.columns:
                df[c] = pd.to_numeric(np.asarray(adata.obs[c]), errors="coerce")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    out = Path(output_dir) / "cell_coords.tsv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False, compression="gzip")
    return str(out)
```

- [ ] **Step 4: Run, verify pass** — `srun -p dev -t 00:15:00 --mem=8G -c 2 python -m pytest tests/test_export_coords.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add export_coords.py tests/test_export_coords.py && git commit -m "feat(export-coords): backed-mode h5ad -> cell_coords.tsv.gz"`.

---

## Task 7: GET /api/cells + selection endpoints

**Files:** Modify `webreview.py` (`_attach_cells_routes` + `_load_cells`, `_cells_payload`); add tests to `tests/test_webreview.py`.

**Interfaces — Consumes:** `ManualStore` from Task 1. **Produces:** `GET /api/cells` (404 if no coords) → column-arrays payload; `POST /api/selection/export {label, indices}` → `{path, n}`; `POST /api/selection/manual {label, indices, disposition}` → `{n, total}`; 400 on out-of-range index / invalid disposition.

- [ ] **Step 1: Add failing tests** to `tests/test_webreview.py`:

```python
def test_cells_and_selection(tmp_path):
    _make_run(tmp_path)
    c = _client(tmp_path)
    j = c.get("/api/cells").json()
    assert j["n"] == 8 and len(j["x"]) == 8 and len(j["y"]) == 8
    assert "c14_1" in j["subcluster_categories"]
    assert j["disposition_categories"] == ["", "KEEP", "DISCARD", "UNCERTAIN"]
    assert "pct_counts_mt" in j["qc"]
    e = c.post("/api/selection/export", json={"label": "foo", "indices": [2, 3]})
    assert e.status_code == 200
    sel = pd.read_csv(tmp_path / "selections" / "selection_foo.tsv", sep="\t")
    assert list(sel["barcode"]) == ["cell2", "cell3"]
    m = c.post("/api/selection/manual",
               json={"label": "bar", "indices": [0, 1], "disposition": "DISCARD"})
    assert m.status_code == 200 and m.json()["n"] == 2
    man = pd.read_csv(tmp_path / "manual_cells.tsv", sep="\t")
    assert list(man["barcode"]) == ["cell0", "cell1"]
    assert c.post("/api/selection/manual",
                  json={"label": "x", "indices": [999], "disposition": "KEEP"}
                  ).status_code == 400


def test_no_coords(tmp_path):
    _make_run(tmp_path)
    (tmp_path / "cell_coords.tsv.gz").unlink()
    c = _client(tmp_path)
    assert c.get("/api/run").json()["has_coords"] is False
    assert c.get("/api/cells").status_code == 404
```

- [ ] **Step 2: Run, verify fail** — `srun -p dev -t 00:15:00 --mem=8G -c 2 python -m pytest tests/test_webreview.py -k "cells or coords" -q` → FAIL.
- [ ] **Step 3: Implement** — add module-level `_load_cells`/`_cells_payload` and replace the `_attach_cells_routes` stub:

```python
def _load_cells(root):
    root = Path(root)
    cp = root / "cell_coords.tsv.gz"
    if not cp.exists():
        return None
    coords = pd.read_csv(cp, sep="\t")
    if "barcode" not in coords.columns:
        coords = coords.rename(columns={coords.columns[0]: "barcode"})
    coords["barcode"] = coords["barcode"].astype(str)
    labels = _read_tsv_safe(root / "cell_labels.tsv")
    if len(labels):
        labels = labels.rename(columns={labels.columns[0]: "barcode"})
        labels["barcode"] = labels["barcode"].astype(str)
        keep = ["barcode"] + [c for c in ("umap_cluster", "recommended_disposition")
                              if c in labels.columns]
        df = coords.merge(labels[keep], on="barcode", how="inner")
    else:
        df = coords
    df["subcluster"] = df.get("umap_cluster", "").astype(str)
    df["disposition"] = df.get("recommended_disposition", "").fillna("").astype(str)
    return df


def _cells_payload(df):
    import pandas as _pd
    parent = df["subcluster"].astype(str).str.extract(r"^c?(\d+)_")[0].fillna("?")
    sub_cat = _pd.Categorical(df["subcluster"].astype(str))
    par_cat = _pd.Categorical(parent)
    disp_cat = _pd.Categorical(df["disposition"].fillna("").astype(str),
                               categories=["", "KEEP", "DISCARD", "UNCERTAIN"])
    skip = {"barcode", "umap_x", "umap_y", "subcluster", "disposition", "umap_cluster",
            "recommended_disposition"}
    qc = {c: [round(float(v), 4) for v in df[c]] for c in df.columns
          if c not in skip and _pd.api.types.is_numeric_dtype(df[c])}
    return {
        "n": int(len(df)),
        "x": [round(float(v), 4) for v in df["umap_x"]],
        "y": [round(float(v), 4) for v in df["umap_y"]],
        "parent_cluster": par_cat.codes.tolist(),
        "parent_categories": list(par_cat.categories),
        "subcluster": sub_cat.codes.tolist(),
        "subcluster_categories": list(sub_cat.categories),
        "disposition": disp_cat.codes.tolist(),
        "disposition_categories": ["", "KEEP", "DISCARD", "UNCERTAIN"],
        "qc": qc,
    }


def _attach_cells_routes(app, root, manual, BaseModel, HTTPException):
    def _ensure():
        if app.state.barcodes is None:
            df = _load_cells(root)
            if df is None:
                return False
            app.state.cells = _cells_payload(df)
            app.state.barcodes = df["barcode"].astype(str).tolist()
        return True

    def _resolve(indices):
        bcs = app.state.barcodes
        out = []
        for i in indices:
            if not isinstance(i, int) or i < 0 or i >= len(bcs):
                raise HTTPException(400, f"index out of range: {i}")
            out.append(bcs[i])
        return out

    @app.get("/api/cells")
    def api_cells():
        if not _ensure():
            raise HTTPException(404, "no cell_coords.tsv.gz")
        return app.state.cells

    class Selection(BaseModel):
        label: str = "selection"
        indices: list[int]
        disposition: str = "DISCARD"

    @app.post("/api/selection/export")
    def api_sel_export(s: Selection):
        if not _ensure():
            raise HTTPException(404, "no cell_coords.tsv.gz")
        return manual.write_selection(s.label, _resolve(s.indices))

    @app.post("/api/selection/manual")
    def api_sel_manual(s: Selection):
        if not _ensure():
            raise HTTPException(404, "no cell_coords.tsv.gz")
        try:
            return manual.add_manual(s.label, _resolve(s.indices), s.disposition)
        except ValueError as e:
            raise HTTPException(400, str(e))
```

(Remove the temporary `_attach_cells_routes` stub from Task 3.)

- [ ] **Step 4: Run, verify pass** — `srun -p dev -t 00:15:00 --mem=8G -c 2 python -m pytest tests/test_webreview.py -q` → PASS (all webreview tests).
- [ ] **Step 5: Commit** — `git add webreview.py tests/test_webreview.py && git commit -m "feat(webreview): /api/cells + lasso selection export/manual endpoints"`.

---

## Task 8: Interactive UMAP + lasso in the SPA + vendor Plotly

**Files:** Add `webreview_static/plotly.min.js`; extend `app.js`, `index.html`, `style.css`.

**Interfaces — Consumes:** `/api/cells`, `/api/selection/export`, `/api/selection/manual`; `has_coords` from `/api/run`. **Produces:** an interactive UMAP view + selection panel.

Required behavior:
- Vendor Plotly: `curl -L -o webreview_static/plotly.min.js https://cdn.plot.ly/plotly-2.35.2.min.js` (a one-time ~3.5 MB download; if the login node has no egress, run on a DTN or note the CDN `<script>` fallback in README). Verify it is a JS file (`head -c 40`), not an HTML error page.
- The UMAP toggle shows the view only when `has_coords`. On first show, `fetch('/api/cells')` once; build a `scattergl` trace. Global mode colors by `parent_categories`; cluster-focus (entered by clicking a minor row, or a cluster picker) filters to one parent and color-splits the chosen minor vs the core `_0`.
- Enable Plotly `dragmode: 'lasso'` (+ box). On `plotly_selected`, read `pts.points[].pointIndex` → the integer indices into the served arrays. Show a selection panel: count, breakdown by `subcluster_categories`, disposition mix, and per-`qc` min/median/max — all computed in JS from the arrays. Buttons: **Export barcodes** (prompt label → POST `/api/selection/export`), **Manual KEEP/DISCARD** (prompt label → POST `/api/selection/manual` with disposition). Toast the result.

- [ ] **Step 1: Vendor Plotly** — run the curl above; `head -c 40 webreview_static/plotly.min.js` should look like JS (not `<!DOCTYPE`).
- [ ] **Step 2: Extend `index.html`** — add `<script src="/static/plotly.min.js"></script>`, the `#umap` container, mode controls, and `#selpanel`.
- [ ] **Step 3: Extend `app.js`** — `loadCells()`, `drawUmap(mode, cid, minor)`, `onSelected(pts)`, `exportSel()`, `manualSel(disp)`; wire the Dashboard/UMAP toggle and the minor-row → cluster-focus click.
- [ ] **Step 4: Manual launch verification** — on a dev node: `srun -p dev -t 00:30:00 --mem=16G -c 4 --pty bash`, then `cd <repo parent>; PYTHONPATH=$PWD python -m standissect serve <run> --host 0.0.0.0 --port 8050`; tunnel/ngrok and confirm dashboard + UMAP + a lasso → export writes `selections/selection_*.tsv`. (No automated browser test.)
- [ ] **Step 5: Commit** — `git add webreview_static && git commit -m "feat(webreview): interactive Plotly UMAP + lasso selection (inspect/export/manual)"`.

---

## Task 9: README + full suite

**Files:** Modify `README.md`; final verification.

- [ ] **Step 1:** Add a "Reviewing a run interactively (`serve`)" section to `README.md`: the `export-coords` step (run in `sh_dev`), `serve` flags, the ngrok/tunnel note (recommend ngrok basic-auth since it's externally exposed), and the output files (`human_review.tsv`, `manual_cells.tsv`, `selections/`). Mention `pip install fastapi uvicorn` as an optional extra.
- [ ] **Step 2: Run full suite** — `srun -p dev -t 00:25:00 --mem=8G -c 2 python -m pytest tests/ -q` → all PASS (web tests skip only if deps absent).
- [ ] **Step 3: Commit** — `git add README.md && git commit -m "docs: interactive review server (serve + export-coords) usage"`.

---

## Self-Review Notes

- **Spec coverage:** decisions-file-only (Tasks 1,3) ✓; dashboard + inline controls (Task 4) ✓; minor-vs-major compare (Task 8 cluster-focus + existing deg/qc tables via Task 2 `/api/table`) ✓; interactive UMAP + lasso → inspect/export/manual (Tasks 7,8) ✓; export-coords (Tasks 5,6) ✓; graceful no-coords (Task 7) ✓; atomic writes + resume (Task 1) ✓; optional deps / login-node safety (Global Constraints, Task 5) ✓; tests via TestClient + importorskip (Tasks 1,2,3,6,7) ✓.
- **Type consistency:** `set(subcluster, parent_cluster, llm_disposition, human_disposition, note, timestamp)` used identically in Tasks 1 & 3; `write_selection(label, barcodes)` / `add_manual(label, barcodes, disposition, timestamp)` identical in Tasks 1 & 7; `_minors_of` returns `(panel_df, ids_list)` used in Tasks 2 & 3; cells payload keys identical in Tasks 7 & 8.
- **No placeholders:** every code step is complete; the only deferred bodies (`_attach_write_routes`, `_attach_cells_routes`) are explicitly stubbed then filled in named later tasks.
