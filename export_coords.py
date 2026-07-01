"""standissect.export_coords — dump per-cell UMAP coords (+QC) for `serve`.

Reads the h5ad in backed mode (never loads X) and writes a lightweight
``cell_coords.tsv.gz`` (barcode, umap_x, umap_y, <qc...>) into the run dir,
which :mod:`standissect.webreview` joins with ``cell_labels.tsv`` to drive the
interactive UMAP. No web dependency; anndata imported lazily so importing this
module is cheap.

    from standissect.export_coords import export_cell_coords
    export_cell_coords("data.h5ad", "out/run", umap_key="X_umap",
                       qc_cols=("doublet_score", "pct_counts_mt"))
"""
from pathlib import Path

import numpy as np
import pandas as pd


def _coords_frame(adata, umap_key, qc_cols):
    """Build the barcode/umap_x/umap_y(+QC) frame from an AnnData (backed or not).
    Raises ``KeyError`` if ``umap_key`` is absent; missing QC cols are skipped."""
    if umap_key not in adata.obsm:
        raise KeyError(f"{umap_key!r} not in obsm; have {list(adata.obsm)}")
    coords = np.asarray(adata.obsm[umap_key])[:, :2]
    df = pd.DataFrame({"barcode": adata.obs_names.astype(str),
                       "umap_x": coords[:, 0], "umap_y": coords[:, 1]})
    for c in qc_cols:
        if c and c in adata.obs.columns:
            df[c] = pd.to_numeric(np.asarray(adata.obs[c]), errors="coerce")
    return df


def _write(df, output_dir):
    out = Path(output_dir) / "cell_coords.tsv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False, compression="gzip")
    return str(out)


def write_coords_from_adata(adata, output_dir, umap_key="X_umap", qc_cols=()):
    """Write ``cell_coords.tsv.gz`` from an already-loaded AnnData (so ``run``
    can emit it without re-opening the h5ad). Returns the output path."""
    return _write(_coords_frame(adata, umap_key, qc_cols), output_dir)


def export_cell_coords(h5ad_path, output_dir, umap_key="X_umap", qc_cols=()):
    """Write ``<output_dir>/cell_coords.tsv.gz`` and return its path.

    Raises ``KeyError`` if ``umap_key`` is not in ``adata.obsm``. QC columns
    not present in ``adata.obs`` are silently skipped.
    """
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        df = _coords_frame(adata, umap_key, qc_cols)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    return _write(df, output_dir)
