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


def test_cli_accepts_extra_cols():
    a = build_parser().parse_args(
        ["run", "x.h5ad", "--cluster-col", "leiden", "--output-dir", "o",
         "--extra-cat-col", "foo", "--extra-qc-col", "bar"])
    assert a.extra_cat_col == ["foo"]
    assert a.extra_qc_col == ["bar"]


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


def test_cli_discard_threshold_default():
    a = build_parser().parse_args(["run", "x.h5ad", "--cluster-col", "leiden",
                                   "--output-dir", "o"])
    assert a.discard_confidence_threshold == 0.5


def test_cli_discard_threshold_override():
    a = build_parser().parse_args(["run", "x.h5ad", "--cluster-col", "leiden",
        "--output-dir", "o", "--discard-confidence-threshold", "0.75"])
    assert a.discard_confidence_threshold == 0.75


def test_cli_annotation_col_default_none():
    a = build_parser().parse_args(["run", "x.h5ad", "--cluster-col", "leiden",
                                   "--output-dir", "o"])
    assert a.annotation_col is None


def test_cli_annotation_col_override():
    a = build_parser().parse_args(["run", "x.h5ad", "--cluster-col", "leiden",
        "--output-dir", "o", "--annotation-col", "cell_ontology_class"])
    assert a.annotation_col == "cell_ontology_class"


def test_cli_serve_defaults():
    a = build_parser().parse_args(["serve", "/run"])
    assert a.output_root == "/run"
    assert a.host == "127.0.0.1" and a.port == 8050
    assert a.func.__name__ == "serve_cmd"


def test_cli_serve_overrides():
    a = build_parser().parse_args(
        ["serve", "/run", "--host", "0.0.0.0", "--port", "9000",
         "--decisions-file", "d.tsv", "--reviewer", "sijie"])
    assert a.host == "0.0.0.0" and a.port == 9000
    assert a.decisions_file == "d.tsv" and a.reviewer == "sijie"


def test_cli_export_coords():
    a = build_parser().parse_args(
        ["export-coords", "x.h5ad", "--output-dir", "/run",
         "--mito-col", "pct_mt", "--extra-qc-col", "foo", "--extra-qc-col", "bar"])
    assert a.output_dir == "/run" and a.umap_key == "X_umap"
    assert a.mito_col == "pct_mt" and a.extra_qc_col == ["foo", "bar"]
    assert a.func.__name__ == "export_coords_cmd"
