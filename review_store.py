"""standissect.review_store — flat-file stores for human review decisions.

No web/anndata dependency; importable top-level (tests) or as a package
module. ``ReviewStore`` owns ``human_review.tsv`` (per-minor verdicts);
``ManualStore`` owns ``manual_cells.tsv`` + ``selections/`` (per-cell
hand-picked sets). All writes are atomic (temp + os.replace).
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re

import pandas as pd

_VALID = {"KEEP", "DISCARD", "UNCERTAIN"}
REVIEW_COLUMNS = ["subcluster", "parent_cluster", "llm_disposition",
                  "human_disposition", "note", "reviewer", "updated_at"]
MANUAL_COLUMNS = ["barcode", "disposition", "label", "reviewer", "updated_at"]


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_tsv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def _slugify(label):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label).strip())
    return s.strip("_") or "selection"


class ReviewStore:
    """Per-minor human verdicts, persisted to ``human_review.tsv``."""

    def __init__(self, path, reviewer=""):
        self.path = Path(path)
        self.reviewer = reviewer
        self._rows = {}
        if self.path.exists():
            self._load()

    def _load(self):
        try:
            df = pd.read_csv(self.path, sep="\t", dtype=str).fillna("")
        except Exception:
            return
        for _, r in df.iterrows():
            sc = str(r.get("subcluster", "")).strip()
            if sc:
                self._rows[sc] = {c: str(r.get(c, "")) for c in REVIEW_COLUMNS}

    def get(self, subcluster):
        return self._rows.get(str(subcluster))

    def get_all(self):
        return dict(self._rows)

    def set(self, subcluster, parent_cluster, llm_disposition,
            human_disposition, note="", timestamp=None):
        sc = str(subcluster).strip()
        hd = (human_disposition or "").strip().upper()
        if hd and hd not in _VALID:
            raise ValueError(f"invalid disposition: {human_disposition!r}")
        if not hd:
            self._rows.pop(sc, None)
        else:
            self._rows[sc] = {
                "subcluster": sc,
                "parent_cluster": str(parent_cluster),
                "llm_disposition": str(llm_disposition or ""),
                "human_disposition": hd,
                "note": str(note or ""),
                "reviewer": self.reviewer,
                "updated_at": str(timestamp if timestamp is not None else _now()),
            }
        self._flush()
        return self._rows.get(sc)

    def _flush(self):
        df = pd.DataFrame(list(self._rows.values()), columns=REVIEW_COLUMNS)
        _atomic_write_tsv(df, self.path)

    def progress(self):
        return {"decided": len(self._rows)}


class ManualStore:
    """Hand-picked per-cell sets: ``manual_cells.tsv`` + ``selections/``."""

    def __init__(self, root, reviewer=""):
        self.root = Path(root)
        self.manual_path = self.root / "manual_cells.tsv"
        self.sel_dir = self.root / "selections"
        self.reviewer = reviewer

    def write_selection(self, label, barcodes):
        path = self.sel_dir / f"selection_{_slugify(label)}.tsv"
        df = pd.DataFrame({"barcode": [str(b) for b in barcodes]})
        _atomic_write_tsv(df, path)
        return {"path": str(path), "n": int(len(df))}

    def add_manual(self, label, barcodes, disposition, timestamp=None):
        d = (disposition or "").strip().upper()
        if d not in _VALID:
            raise ValueError(f"invalid disposition: {disposition!r}")
        ts = str(timestamp if timestamp is not None else _now())
        new = pd.DataFrame({
            "barcode": [str(b) for b in barcodes],
            "disposition": d, "label": _slugify(label),
            "reviewer": self.reviewer, "updated_at": ts,
        }, columns=MANUAL_COLUMNS)
        if self.manual_path.exists():
            old = pd.read_csv(self.manual_path, sep="\t", dtype=str).fillna("")
            new = pd.concat([old, new], ignore_index=True)
        _atomic_write_tsv(new, self.manual_path)
        return {"n": int(len(barcodes)), "total": int(len(new))}
