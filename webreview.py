"""standissect.webreview — interactive review server for a dissect run tree.

Reads the run output (TSV/PNG/JSON/cell_coords) and writes only decision /
selection files via :mod:`review_store`. Never imports anndata or opens the
source ``.h5ad`` — safe to run on a login node behind an SSH tunnel / ngrok.

    from standissect.webreview import serve
    serve("/path/to/dissect/run", host="127.0.0.1", port=8050)

NB: this module intentionally does NOT use ``from __future__ import
annotations`` — FastAPI must see the real Pydantic request-model classes
(``Decision``/``Selection``) as annotation objects, not PEP 563 strings, to
treat them as request bodies.
"""
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


# --------------------------------------------------------------------------- #
# small coercions
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# run-tree readers
# --------------------------------------------------------------------------- #
def _list_clusters(root):
    d = Path(root) / "clusters"
    if not d.exists():
        return []
    return sorted((p.name[1:] for p in d.glob("c*") if p.is_dir()),
                  key=lambda x: int(x) if x.isdigit() else 10 ** 9)


def _minors_of(root, cid):
    """(panel_df, [subcluster ids]) for one cluster; empty df if absent."""
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


# --------------------------------------------------------------------------- #
# interactive-UMAP cell data
# --------------------------------------------------------------------------- #
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
    df["subcluster"] = (df["umap_cluster"].astype(str)
                        if "umap_cluster" in df.columns else "")
    df["disposition"] = (df["recommended_disposition"].fillna("").astype(str)
                         if "recommended_disposition" in df.columns else "")
    return df


def _cells_payload(df):
    parent = df["subcluster"].astype(str).str.extract(r"^c?(\d+)_")[0].fillna("?")
    sub_cat = pd.Categorical(df["subcluster"].astype(str))
    par_cat = pd.Categorical(parent)
    disp_cat = pd.Categorical(df["disposition"].fillna("").astype(str),
                              categories=["", "KEEP", "DISCARD", "UNCERTAIN"])
    skip = {"barcode", "umap_x", "umap_y", "subcluster", "disposition",
            "umap_cluster", "recommended_disposition"}
    qc = {c: [round(float(v), 4) for v in df[c]] for c in df.columns
          if c not in skip and pd.api.types.is_numeric_dtype(df[c])}
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


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
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

    # ---- writes: decision ---- #
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

    # ---- interactive UMAP + selection ---- #
    def _ensure_cells():
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
        if not _ensure_cells():
            raise HTTPException(404, "no cell_coords.tsv.gz")
        return app.state.cells

    class Selection(BaseModel):
        label: str = "selection"
        indices: list[int]
        disposition: str = "DISCARD"

    @app.post("/api/selection/export")
    def api_sel_export(s: Selection):
        if not _ensure_cells():
            raise HTTPException(404, "no cell_coords.tsv.gz")
        return manual.write_selection(s.label, _resolve(s.indices))

    @app.post("/api/selection/manual")
    def api_sel_manual(s: Selection):
        if not _ensure_cells():
            raise HTTPException(404, "no cell_coords.tsv.gz")
        try:
            return manual.add_manual(s.label, _resolve(s.indices), s.disposition)
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


def serve(root, host="127.0.0.1", port=8050, decisions_file=None, reviewer=""):
    try:
        import uvicorn
    except ImportError as e:                       # pragma: no cover
        raise SystemExit("`standissect serve` needs fastapi + uvicorn:\n"
                         "  pip install fastapi uvicorn "
                         "(run in sh_dev, not the login node)") from e
    app = build_app(root, decisions_file=decisions_file, reviewer=reviewer)
    print(f"[standissect] review server for {Path(root).name} "
          f"-> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":                         # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python -m standissect.webreview <run_root> [port]")
    serve(sys.argv[1], port=int(sys.argv[2]) if len(sys.argv) > 2 else 8050)
