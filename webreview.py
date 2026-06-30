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
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import tempfile
import time

import numpy as np
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
# minor-profile heatmap — rebuilt from the persisted TSVs for an interactive
# Plotly heatmap that matches minor_profile.png exactly: same z-score, same
# optimal-leaf ordering, same matplotlib colormaps. No anndata needed.
# --------------------------------------------------------------------------- #
def _leaf_order(arr):
    """Optimal-leaf-ordering of the rows of a 2-D array; rows with any NaN go
    last in original order (mirrors cluster._cluster_rows/_cluster_columns)."""
    finite = ~np.isnan(arr).any(axis=1)
    if int(finite.sum()) < 2:
        return list(range(arr.shape[0]))
    from scipy.cluster.hierarchy import linkage, leaves_list
    idx = np.where(finite)[0]
    Z = linkage(arr[idx], method="average", metric="euclidean", optimal_ordering=True)
    order = [int(i) for i in idx[leaves_list(Z)]]
    order += [i for i in range(arr.shape[0]) if not finite[i]]
    return order


def _cmap_to_plotly(name, n=33):
    """Sample a matplotlib colormap into a Plotly colorscale (identical colours)."""
    import matplotlib
    matplotlib.use("Agg")
    cmap = matplotlib.colormaps[name]
    out = []
    for i in range(n):
        f = i / (n - 1)
        r, g, b, _ = cmap(f)
        out.append([f, f"rgb({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))})"])
    return out


def _grid(df):
    """2-D DataFrame -> list-of-lists of floats with NaN -> None (JSON-safe)."""
    return [[None if (v is None or (isinstance(v, float) and v != v)) else float(v)
             for v in row] for row in np.asarray(df.values).tolist()]


def _heat_indexed(path):
    """Read a heatmap TSV (first column is the row index) or return None."""
    df = _read_tsv_safe(path)
    if not len(df) or df.shape[1] < 2:
        return None
    df = df.set_index(df.columns[0])
    df.columns = [str(c) for c in df.columns]
    return df


def _heatmap_payload(root, cid):
    """Rebuild cluster ``cid``'s minor-profile heatmap (gene / QC / sample
    blocks) from persisted TSVs, ordered + z-scored exactly like the PNG."""
    cdir = Path(root) / "clusters" / f"c{cid}"
    mat = _heat_indexed(cdir / "heatmap_data.tsv")
    if mat is None:
        return None
    cols = list(mat.columns)
    minor_pref = f"c{cid}_"
    core_names = [c for c in cols if c.endswith("_0")]
    minor_names = [c for c in cols if c.startswith(minor_pref) and not c.endswith("_0")]

    z = (mat.sub(mat.mean(axis=1), axis=0)
            .div(mat.std(axis=1).replace(0, np.nan), axis=0).clip(-2, 2))
    z_fc = z.fillna(0)
    gene_order = [str(z.index[i]) for i in _leaf_order(z_fc.values)]

    def _ord(names):
        if len(names) >= 2:
            return [names[i] for i in _leaf_order(z_fc[names].values.T)]
        return list(names)
    ordered_core, ordered_minor = _ord(core_names), _ord(minor_names)
    col_order = ordered_core + ordered_minor
    heat = z.reindex(index=gene_order, columns=col_order)

    qc_rows, qc_grid = [], []
    qc = _heat_indexed(cdir / "qc_tracks.tsv")
    if qc is not None:
        qcz = qc.sub(qc.mean(axis=1), axis=0).div(
            qc.std(axis=1).replace(0, np.nan), axis=0).reindex(columns=col_order)
        qc_rows, qc_grid = [str(r) for r in qc.index], _grid(qcz)

    sm_rows, sm_grid = [], []
    sm = _heat_indexed(cdir / "sample_composition.tsv")
    if sm is not None:
        sm = sm.reindex(columns=col_order)
        sm_rows, sm_grid = [str(r) for r in sm.index], _grid(sm)

    return {
        "cid": str(cid), "cols": col_order,
        "home_core": f"c{cid}_0", "minor_cols": ordered_minor,
        "n_core": len(ordered_core), "n_minor": len(ordered_minor),
        "genes": gene_order, "gene_z": _grid(heat),
        "qc_rows": qc_rows, "qc_z": qc_grid,
        "sample_rows": sm_rows, "sample": sm_grid,
        "colorscales": {"gene": _cmap_to_plotly("RdBu_r"),
                        "qc": _cmap_to_plotly("coolwarm"),
                        "sample": _cmap_to_plotly("magma")},
        "ranges": {"gene": [-2, 2], "qc": [-2, 2], "sample": [0, 1]},
    }


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
    # the c{parent}_{minor} subcluster label lives in `original_cluster_split`
    # (cell_labels.tsv's `umap_cluster` is the raw u-fragment, not what we want).
    sub_col = None
    if len(labels):
        labels = labels.rename(columns={labels.columns[0]: "barcode"})
        labels["barcode"] = labels["barcode"].astype(str)
        sub_col = next((c for c in ("original_cluster_split", "subcluster",
                                    "umap_cluster") if c in labels.columns), None)
        keep = ["barcode"] + [c for c in (sub_col, "recommended_disposition")
                              if c and c in labels.columns]
        df = coords.merge(labels[keep], on="barcode", how="inner")
    else:
        df = coords
    df["subcluster"] = (df[sub_col].astype(str)
                        if sub_col and sub_col in df.columns else "")
    df["disposition"] = (df["recommended_disposition"].fillna("").astype(str)
                         if "recommended_disposition" in df.columns else "")
    # rows with NaN coordinates cannot be plotted; drop them so indices stay
    # contiguous and the served arrays are JSON-clean.
    df = df[df["umap_x"].notna() & df["umap_y"].notna()].reset_index(drop=True)
    return df


def _jnum(v):
    """Round to 4 dp, mapping NaN -> None (stdlib/Starlette JSON rejects NaN)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 4)


def _cells_payload(df):
    parent = df["subcluster"].astype(str).str.extract(r"^c?(\d+)_")[0].fillna("?")
    sub_cat = pd.Categorical(df["subcluster"].astype(str))
    par_cat = pd.Categorical(parent)
    disp_cat = pd.Categorical(df["disposition"].fillna("").astype(str),
                              categories=["", "KEEP", "DISCARD", "UNCERTAIN"])
    skip = {"barcode", "umap_x", "umap_y", "subcluster", "disposition",
            "umap_cluster", "original_cluster_split", "recommended_disposition"}
    qc = {c: [_jnum(v) for v in df[c]] for c in df.columns
          if c not in skip and pd.api.types.is_numeric_dtype(df[c])}
    return {
        "n": int(len(df)),
        "x": [_jnum(v) for v in df["umap_x"]],
        "y": [_jnum(v) for v in df["umap_y"]],
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
def _compute_deg(expr, bcs_a, bcs_b, *, layer=None, top_n=25, cap=1500):
    """A-vs-B Mann-Whitney DEG on log-normalised expression for two barcode sets.

    ``expr`` is ``(backed AnnData, {barcode: row}, var_names)``. Cells in both
    groups (overlapping lassos) and barcodes absent from the h5ad are dropped;
    each group is capped to ``cap`` (random, seeded) to bound latency. Reuses the
    pipeline's ``wilcoxon_vs_reference`` so results match the tree's deg_*.tsv.
    Returns top up-in-A / up-in-B genes. Raises ValueError on degenerate input.
    """
    import scipy.sparse as sp
    try:
        from .cluster import wilcoxon_vs_reference
    except ImportError:
        from cluster import wilcoxon_vs_reference
    adata, bc2row, var_names = expr
    a = {x for x in bcs_a if x in bc2row}
    b = {x for x in bcs_b if x in bc2row}
    common = a & b
    a -= common
    b -= common
    n_unknown = len([x for x in (set(bcs_a) | set(bcs_b)) if x not in bc2row])
    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            "need >=2 cells in each group after removing overlap/unknown barcodes")
    rng = np.random.default_rng(0)

    def _rows(s):
        arr = np.array(sorted(bc2row[x] for x in s), dtype=np.int64)
        if len(arr) > cap:
            arr = np.sort(rng.choice(arr, cap, replace=False))
        return arr
    ra, rb = _rows(a), _rows(b)
    rows = np.concatenate([ra, rb])
    labels = np.array(["A"] * len(ra) + ["B"] * len(rb))
    order = np.argsort(rows, kind="stable")          # backed reads want sorted rows
    src = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
    X = sp.csr_matrix(src[rows[order]])
    # Drop genes expressed in too few of the selected cells: ranking all ~60k
    # genes is the entire cost of the test, and a gene in <1% of cells can't be a
    # real marker. Cuts the Wilcoxon ~2-3x with no effect on the top hits.
    thr = max(3, int(0.01 * X.shape[0]))
    keep = np.asarray((X != 0).sum(axis=0)).ravel() >= thr
    if not keep.any():
        raise ValueError("no genes pass the expression filter for these groups")
    X = X[:, keep].tocsr()
    names = [g for g, k in zip(var_names, keep) if k]
    df = wilcoxon_vs_reference(X, labels[order], group="A", reference="B",
                               gene_names=names, n_genes=10 ** 9)

    def _fmt(d):
        return [{"gene": str(r["names"]), "log2fc": _to_float(r["logfoldchanges"]),
                 "pval": _to_float(r["pvals"]), "padj": _to_float(r["pvals_adj"]),
                 "score": _to_float(r["scores"])} for _, r in d.iterrows()]
    up = df[df["logfoldchanges"] > 0].head(top_n)
    dn = df[df["logfoldchanges"] < 0].sort_values("scores").head(top_n)
    return {"n_a": int(len(ra)), "n_b": int(len(rb)),
            "dropped_overlap": int(len(common)), "dropped_unknown": int(n_unknown),
            "layer": layer or "X", "n_genes": int(len(names)),
            "up_in_a": _fmt(up), "up_in_b": _fmt(dn)}


def build_app(root, decisions_file=None, reviewer="", h5ad=None, deg_layer=None):
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
    app.state.expr = None
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (_STATIC / "index.html").read_text(encoding="utf-8")
        # cache-bust: version app.js/style.css by mtime so browsers always fetch
        # the current build, and never cache index.html itself.
        ver = 0
        for f in ("app.js", "style.css"):
            p = _STATIC / f
            if p.exists():
                ver = max(ver, int(p.stat().st_mtime))
        html = (html.replace("/static/app.js", f"/static/app.js?v={ver}")
                    .replace("/static/style.css", f"/static/style.css?v={ver}"))
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/api/run")
    def api_run():
        return {**_run_payload(root, store), "deg_enabled": bool(h5ad)}

    @app.get("/api/cluster/{cid}")
    def api_cluster(cid: str):
        if not (root / "clusters" / f"c{cid}").exists():
            raise HTTPException(404, f"no cluster {cid}")
        return _cluster_payload(root, cid, store)

    @app.get("/api/heatmap/{cid}")
    def api_heatmap(cid: str):
        if not (root / "clusters" / f"c{cid}").exists():
            raise HTTPException(404, f"no cluster {cid}")
        data = _heatmap_payload(root, cid)
        if data is None:
            raise HTTPException(404, "no heatmap data")
        return data

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

    # ---- dynamic DEG between two lassoed groups (opt-in via --h5ad) ---- #
    def _ensure_expr():
        """Lazily open the DEG h5ad (backed) + build a barcode->row map. Only
        touched when /api/deg is hit, so the default server never reads an h5ad."""
        if app.state.expr is None:
            if not h5ad:
                return None
            try:
                import anndata as ad
                a = ad.read_h5ad(h5ad, backed="r")
            except Exception:
                return None
            bc2row = {str(bc): i for i, bc in enumerate(a.obs_names)}
            app.state.expr = (a, bc2row, [str(v) for v in a.var_names])
        return app.state.expr

    class DegReq(BaseModel):
        a: list[int]
        b: list[int]
        top_n: int = 25

    @app.post("/api/deg")
    def api_deg(req: DegReq):
        if not h5ad:
            raise HTTPException(400, "DEG disabled: restart serve with --h5ad PATH")
        if not _ensure_cells():
            raise HTTPException(404, "no cell_coords.tsv.gz")
        expr = _ensure_expr()
        if expr is None:
            raise HTTPException(400, f"could not open DEG h5ad: {h5ad}")
        try:
            return _compute_deg(expr, _resolve(req.a), _resolve(req.b),
                                layer=deg_layer, top_n=max(1, min(100, req.top_n)))
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


def _pidfile(port):
    return Path(tempfile.gettempdir()) / f"standissect-serve-{port}.pid"


def _port_in_use(host, port):
    target = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            return s.connect_ex((target, int(port))) == 0
        except OSError:
            return False


def _pids_on_port(port):
    """Best-effort PIDs listening on ``port`` via lsof/fuser (empty if neither)."""
    pids = set()
    for cmd in (["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                ["fuser", f"{port}/tcp"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except Exception:
            continue
        for tok in (out.stdout + " " + out.stderr).replace("\n", " ").split():
            tok = tok.strip().split("/")[0]
            if tok.isdigit():
                pids.add(int(tok))
        if pids:
            break
    return pids


def _looks_like_server(pid):
    try:
        cmd = (Path(f"/proc/{pid}/cmdline").read_bytes()
               .replace(b"\x00", b" ").decode("utf-8", "replace").lower())
    except Exception:
        return False
    return "uvicorn" in cmd or ("standissect" in cmd and "serve" in cmd)


def _ensure_port_free(host, port, *, wait=8.0):
    """Stop a previous standissect server on this port (idempotent restart).

    Targets only our own prior instance: a PID from our pidfile (trusted) or a
    process listening on the port whose cmdline looks like a standissect/uvicorn
    server. Returns True once the port is free.
    """
    me = os.getpid()
    victims = set()
    pf = _pidfile(port)
    try:
        if pf.exists():
            p = int(pf.read_text().split()[0])
            if p != me and _port_in_use(host, port):
                victims.add(p)
    except Exception:
        pass
    for p in _pids_on_port(port):
        if p != me and _looks_like_server(p):
            victims.add(p)
    if not victims:
        return not _port_in_use(host, port)
    for p in victims:
        try:
            os.kill(p, signal.SIGTERM)
            print(f"[standissect] stopped previous server (pid {p}) on port {port}")
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.time() + wait
    while _port_in_use(host, port) and time.time() < deadline:
        time.sleep(0.25)
    if _port_in_use(host, port):                       # escalate to SIGKILL
        for p in victims:
            try:
                os.kill(p, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        end = time.time() + 3
        while _port_in_use(host, port) and time.time() < end:
            time.sleep(0.2)
    return not _port_in_use(host, port)


def serve(root, host="127.0.0.1", port=8050, decisions_file=None, reviewer="",
          replace=True, h5ad=None, deg_layer=None):
    try:
        import uvicorn
    except ImportError as e:                       # pragma: no cover
        raise SystemExit("`standissect serve` needs fastapi + uvicorn:\n"
                         "  pip install fastapi uvicorn "
                         "(run in sh_dev, not the login node)") from e
    if replace and not _ensure_port_free(host, port):
        raise SystemExit(
            f"[standissect] port {port} is still in use and could not be freed "
            f"(another user's process?). Choose a different --port.")
    app = build_app(root, decisions_file=decisions_file, reviewer=reviewer,
                    h5ad=h5ad, deg_layer=deg_layer)
    if h5ad:
        print(f"[standissect] DEG enabled — expression from {h5ad}"
              + (f" (layer={deg_layer})" if deg_layer else " (X)"))
    pf = _pidfile(port)
    try:
        pf.write_text(str(os.getpid()))
    except Exception:
        pf = None
    print(f"[standissect] review server for {Path(root).name} "
          f"-> http://{host}:{port}")
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        if pf is not None:
            try:
                pf.unlink()
            except OSError:
                pass


if __name__ == "__main__":                         # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python -m standissect.webreview <run_root> [port]")
    serve(sys.argv[1], port=int(sys.argv[2]) if len(sys.argv) > 2 else 8050)
