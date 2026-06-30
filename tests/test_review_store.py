import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from review_store import ReviewStore, ManualStore, _slugify  # noqa: E402


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
