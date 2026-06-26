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
