import json
import pathlib
import sys

import pandas as pd
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


def test_naming_prompt_advertises_differs_and_cites_parent():
    system, user = annotate.build_core_naming_prompt(_evi(["IL3RA"], parent="myeloid"))
    assert "differs_from_original" in user
    assert "parent_cluster" in (system + user)


def test_core_naming_from_dict_parses_differs_and_original_label():
    evi = _evi(["IL3RA", "CLEC4C"], parent="myeloid")
    data = {"cell_type": "pDC", "confidence": 0.9, "rationale": "pDC markers",
            "differs_from_original": True, "markers_used": ["IL3RA"]}
    naming = annotate._core_naming_from_dict(data, evi, model="m")
    assert naming.differs_from_original is True
    row = naming.to_core_name_row(evi)
    assert row["original_label"] == "myeloid"
    assert row["differs_from_original"] is True
    assert "original_label" in annotate.CORE_NAME_COLS
    assert "differs_from_original" in annotate.CORE_NAME_COLS


def test_run_naming_stage_writes_relabel_columns(tmp_path):
    from diagnosis import CallableChatClient
    clusters = tmp_path / "clusters"
    canon = tmp_path / "canonical_markers"
    (clusters / "cmyeloid").mkdir(parents=True)
    canon.mkdir(parents=True)
    pd.DataFrame({"gene": ["IL3RA", "CLEC4C"], "logfoldchanges": [2.0, 2.0],
                  "scores": [10.0, 9.0]}
                 ).to_csv(canon / "markers_cmyeloid_0.tsv", sep="\t", index=False)
    client = CallableChatClient(lambda s, u: json.dumps(
        {"cell_type": "pDC", "confidence": 0.9, "rationale": "pDC markers",
         "differs_from_original": True, "markers_used": ["IL3RA"]}), model="m")
    engine = annotate.make_naming_engine(client=client)
    annotate.run_naming_stage(clusters_dir=clusters, canonical_dir=canon,
        core_names_path=tmp_path / "core_names.tsv", parents=["myeloid"],
        engine=engine, forced=True)
    df = pd.read_csv(tmp_path / "core_names.tsv", sep="\t")
    assert df.loc[0, "cell_type"] == "pDC"
    assert str(df.loc[0, "original_label"]) == "myeloid"
    assert bool(df.loc[0, "differs_from_original"]) is True
