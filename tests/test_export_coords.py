import pathlib
import sys

import pytest

pytest.importorskip("anndata")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from export_coords import export_cell_coords  # noqa: E402


def test_export(tmp_path):
    import anndata as ad
    obs = pd.DataFrame({"mt": [0.1, 0.2, 0.3, 0.4, 0.5]},
                       index=[f"b{i}" for i in range(5)])
    a = ad.AnnData(X=np.zeros((5, 3)), obs=obs)
    a.obsm["X_umap"] = np.arange(10).reshape(5, 2).astype(float)
    h = tmp_path / "a.h5ad"
    a.write_h5ad(h)
    out = export_cell_coords(str(h), str(tmp_path), qc_cols=("mt", "missing"))
    df = pd.read_csv(out, sep="\t")
    assert list(df["barcode"]) == [f"b{i}" for i in range(5)]
    assert df["umap_x"].iloc[1] == 2.0 and df["umap_y"].iloc[1] == 3.0
    assert "mt" in df.columns and "missing" not in df.columns


def test_export_missing_key(tmp_path):
    import anndata as ad
    a = ad.AnnData(X=np.zeros((3, 2)))
    h = tmp_path / "b.h5ad"
    a.write_h5ad(h)
    with pytest.raises(KeyError):
        export_cell_coords(str(h), str(tmp_path), umap_key="X_umap")
