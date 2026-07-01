import base64
import pathlib
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import webreview  # noqa: E402

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
    bcs = [f"cell{i}" for i in range(9)]   # cell8 has NaN coord -> dropped
    # real cell_labels.tsv: `umap_cluster` is the raw u-fragment; the
    # c{parent}_{minor} subcluster is in `original_cluster_split`.
    pd.DataFrame({"": bcs,
                  "umap_cluster": ["u0", "u0", "u1", "u1", "u2", "u2", "u5", "u0", "u0"],
                  "original_cluster_split": ["c14_0", "c14_0", "c14_1", "c14_1",
                                             "c14_2", "c14_2", "c14_5", "c14_0", "c14_0"],
                  "recommended_disposition": ["", "", "KEEP", "KEEP",
                                              "UNCERTAIN", "UNCERTAIN", "", "", ""],
                  "proposed_cell_type": [""] * 9}).to_csv(
        root / "cell_labels.tsv", sep="\t", index=False)
    xs = list(range(8)) + [float("nan")]   # cell8 NaN coord
    mt = [0.1] * 9
    mt[6] = float("nan")                    # NaN QC -> JSON null
    pd.DataFrame({"barcode": bcs, "umap_x": xs, "umap_y": xs,
                  "pct_counts_mt": mt}).to_csv(
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


def _write_heatmap_tsvs(root):
    c14 = root / "clusters" / "c14"
    # cores end in _0; minors are c14_k (k>0). FLAT has zero std -> z = NaN -> null.
    pd.DataFrame({"": ["GENEA", "GENEB", "FLAT"],
                  "c0_0": [1.0, 4.0, 5.0], "c14_0": [2.0, 3.0, 5.0],
                  "c14_2": [3.0, 2.0, 5.0], "c14_1": [4.0, 1.0, 5.0]}).to_csv(
        c14 / "heatmap_data.tsv", sep="\t", index=False)
    pd.DataFrame({"": ["doublet", "mito"],
                  "c0_0": [0.1, 0.5], "c14_0": [0.2, 0.4],
                  "c14_2": [0.3, 0.3], "c14_1": [0.4, 0.2]}).to_csv(
        c14 / "qc_tracks.tsv", sep="\t", index=False)
    pd.DataFrame({"": ["sampleA", "sampleB"],
                  "c0_0": [0.5, 0.5], "c14_0": [0.6, 0.4],
                  "c14_2": [0.1, 0.9], "c14_1": [0.2, 0.8]}).to_csv(
        c14 / "sample_composition.tsv", sep="\t", index=False)


def test_heatmap_payload(tmp_path):
    _make_run(tmp_path)
    _write_heatmap_tsvs(tmp_path)
    j = _client(tmp_path).get("/api/heatmap/14").json()

    assert j["cid"] == "14" and j["home_core"] == "c14_0"
    assert (j["n_core"], j["n_minor"]) == (2, 2)
    assert set(j["cols"]) == {"c0_0", "c14_0", "c14_1", "c14_2"}
    assert set(j["cols"][:2]) == {"c0_0", "c14_0"}        # cores first
    assert set(j["cols"][2:]) == {"c14_1", "c14_2"}       # minors after
    assert set(j["minor_cols"]) == {"c14_1", "c14_2"}

    assert set(j["genes"]) == {"GENEA", "GENEB", "FLAT"}
    assert len(j["gene_z"]) == 3 and all(len(r) == 4 for r in j["gene_z"])
    flat = j["gene_z"][j["genes"].index("FLAT")]
    assert flat == [None, None, None, None]              # zero-std row -> null
    ga = j["gene_z"][j["genes"].index("GENEA")]
    assert all(v is not None and -2 <= v <= 2 for v in ga)

    assert set(j["qc_rows"]) == {"doublet", "mito"} and len(j["qc_z"]) == 2
    assert set(j["sample_rows"]) == {"sampleA", "sampleB"}
    assert all(0 <= v <= 1 for row in j["sample"] for v in row)   # raw fracs, not z

    for k in ("gene", "qc", "sample"):
        cs = j["colorscales"][k]
        assert cs[0][0] == 0.0 and cs[-1][0] == 1.0 and cs[0][1].startswith("rgb(")
    assert j["ranges"] == {"gene": [-2, 2], "qc": [-2, 2], "sample": [0, 1]}


def test_heatmap_404_when_no_data(tmp_path):
    _make_run(tmp_path)
    # cluster dir exists but no heatmap_data.tsv -> 404; unknown cluster -> 404
    (tmp_path / "clusters" / "c30").mkdir()
    assert _client(tmp_path).get("/api/heatmap/30").status_code == 404
    assert _client(tmp_path).get("/api/heatmap/99").status_code == 404


def _write_expr_h5ad(path):
    """Tiny log-norm expression h5ad sharing barcodes (cell0..cell8) with the run.
    G0 high in cell0-3 (group A), G1 high in cell4-7 (group B)."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    bcs = [f"cell{i}" for i in range(9)]
    genes = ["G0", "G1", "G2", "G3"]
    X = np.zeros((9, 4), dtype=np.float32)
    X[:4, 0] = 3.0       # G0 up in A (cell0-3)
    X[4:8, 1] = 3.0      # G1 up in B (cell4-7)
    X[:, 2] = 1.0        # G2 constant (no DE)
    adata = ad.AnnData(X=sp.csr_matrix(X),
                       obs=pd.DataFrame(index=bcs), var=pd.DataFrame(index=genes))
    adata.write_h5ad(path)


def test_deg_between_two_groups(tmp_path):
    pytest.importorskip("anndata")
    _make_run(tmp_path)
    p = tmp_path / "expr.h5ad"
    _write_expr_h5ad(p)
    c = _client(tmp_path, h5ad=str(p))

    assert c.get("/api/run").json()["deg_enabled"] is True
    r = c.post("/api/deg", json={"a": [0, 1, 2, 3], "b": [4, 5, 6, 7], "top_n": 5})
    assert r.status_code == 200
    j = r.json()
    assert (j["n_a"], j["n_b"], j["layer"]) == (4, 4, "X")
    a_genes = [g["gene"] for g in j["up_in_a"]]
    b_genes = [g["gene"] for g in j["up_in_b"]]
    assert "G0" in a_genes and "G1" in b_genes
    assert all(g["log2fc"] is None or g["log2fc"] > 0 for g in j["up_in_a"])
    assert all(g["log2fc"] is None or g["log2fc"] < 0 for g in j["up_in_b"])
    # <2 cells per group -> 400
    assert c.post("/api/deg", json={"a": [0], "b": [4]}).status_code == 400


def test_deg_disabled_without_h5ad(tmp_path):
    _make_run(tmp_path)
    c = _client(tmp_path)                       # no --h5ad
    assert c.get("/api/run").json()["deg_enabled"] is False
    assert c.post("/api/deg", json={"a": [0, 1], "b": [2, 3]}).status_code == 400


def test_expr_feature_plot(tmp_path):
    pytest.importorskip("anndata")
    _make_run(tmp_path)
    p = tmp_path / "expr.h5ad"
    _write_expr_h5ad(p)
    c = _client(tmp_path, h5ad=str(p))
    j = c.get("/api/expr/G0").json()
    assert j["gene"] == "G0"
    assert len(j["values"]) == 8                       # cell8 (NaN coord) dropped
    assert j["values"][:4] == [3.0, 3.0, 3.0, 3.0]     # G0 high in cell0-3
    assert all(v == 0.0 for v in j["values"][4:])
    assert j["vmax"] > 0
    assert c.get("/api/expr/NOPE").status_code == 404
    assert _client(tmp_path).get("/api/expr/G0").status_code == 400   # disabled w/o --h5ad


def test_feature_gene_numeric_and_categorical(tmp_path):
    pytest.importorskip("anndata")
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    _make_run(tmp_path)
    bcs = [f"cell{i}" for i in range(9)]
    X = np.zeros((9, 2), dtype=np.float32)
    X[:4, 0] = 3.0
    obs = pd.DataFrame({
        "score": [0.1 * i for i in range(9)],                     # float -> continuous
        "grp": ["x", "x", "x", "x", "y", "y", "y", "y", "z"],     # object -> categorical
    }, index=bcs)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs,
                       var=pd.DataFrame(index=["G0", "G1"]))
    p = tmp_path / "f.h5ad"
    adata.write_h5ad(p)
    c = _client(tmp_path, h5ad=str(p))

    g = c.get("/api/feature/G0").json()
    assert g["kind"] == "cont" and g["values"][:4] == [3.0, 3.0, 3.0, 3.0]
    n = c.get("/api/feature/score").json()
    assert n["kind"] == "cont" and len(n["values"]) == 8 and n["vmax"] > 0
    cat = c.get("/api/feature/grp").json()
    assert cat["kind"] == "cat"
    assert set(cat["categories"]) == {"x", "y", "z"}
    assert len(cat["codes"]) == 8
    assert c.get("/api/feature/NOPE").status_code == 404


def test_index_and_static(tmp_path):
    _make_run(tmp_path)
    c = _client(tmp_path)
    assert c.get("/").status_code == 200
    assert "standissect" in c.get("/").text.lower()
    assert c.get("/static/app.js").status_code == 200


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


def test_cells_and_selection(tmp_path):
    _make_run(tmp_path)
    c = _client(tmp_path)
    j = c.get("/api/cells").json()
    assert j["n"] == 8 and len(j["x"]) == 8 and len(j["y"]) == 8  # cell8 dropped
    assert "c14_1" in j["subcluster_categories"]
    assert "14" in j["parent_categories"]      # parent parsed from c14_* subcluster
    assert j["disposition_categories"] == ["", "KEEP", "DISCARD", "UNCERTAIN"]
    assert "pct_counts_mt" in j["qc"]
    assert j["qc"]["pct_counts_mt"][6] is None                    # NaN -> null
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


def test_port_in_use_detection():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        assert webreview._port_in_use("127.0.0.1", port) is True
    finally:
        s.close()


def test_port_free_detection():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert webreview._port_in_use("127.0.0.1", port) is False
