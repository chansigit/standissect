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
