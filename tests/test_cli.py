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
