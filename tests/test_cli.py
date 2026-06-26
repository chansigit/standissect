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
