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


def export_cell_coords(h5ad_path, output_dir, umap_key="X_umap", qc_cols=()):
    """Write ``<output_dir>/cell_coords.tsv.gz`` and return its path.

    Raises ``KeyError`` if ``umap_key`` is not in ``adata.obsm``. QC columns
    not present in ``adata.obs`` are silently skipped.
    """
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if umap_key not in adata.obsm:
            raise KeyError(f"{umap_key!r} not in obsm; have {list(adata.obsm)}")
        coords = np.asarray(adata.obsm[umap_key])[:, :2]
        df = pd.DataFrame({"barcode": adata.obs_names.astype(str),
                           "umap_x": coords[:, 0], "umap_y": coords[:, 1]})
        for c in qc_cols:
            if c and c in adata.obs.columns:
                df[c] = pd.to_numeric(np.asarray(adata.obs[c]), errors="coerce")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    out = Path(output_dir) / "cell_coords.tsv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False, compression="gzip")
    return str(out)
