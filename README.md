# standissect

Cluster cleanup-diagnosis for single-cell data. Given an existing clustering,
`standissect` re-partitions the cells on their UMAP coordinates, cross-tabulates
the two clusterings, and surfaces — per original cluster — the off-main
fragments ("minors"), each with DEG vs the cluster core, composition drift, QC
drift, and a `likely_cause` verdict (sample-driven / doublet / low-quality /
shallow-depth / biology-candidate / unclear).

## Install

```
pip install -e .
```

or directly from GitHub:

```
pip install git+https://github.com/chansigit/standissect.git
```

## Use

```python
import anndata as ad
from standissect import run_dissect_pipeline, build_report

adata = ad.read_h5ad("my_data.h5ad")        # needs obsm['X_umap'] + a clustering in obs
result = run_dissect_pipeline(
    adata,
    cluster_col="leiden",
    output_dir="results/dissect",
    cat_cols=("orig.ident", "batch"),       # any project-specific obs columns
    qc_cols=("percent.mt", "nCount_RNA", "nFeature_RNA", "hybrid_score"),
    sample_col="orig.ident",
)
build_report(result["root"])                # -> results/dissect/leiden/report.html
```

The pipeline is **idempotent** — re-running skips stages whose output files
already exist. Pass `force=("partition", "dissect", "canonical", "anatomy")`
(or `"all"`) to recompute. No dataset-specific column names are hardcoded; the
caller supplies `cat_cols` / `qc_cols` / `sample_col`.

## Output tree

```
<output_dir>/<cluster_col>/
├── crosstab.tsv  panel.tsv  cell_labels.tsv  qc_drift_all.tsv  params.json
├── global_umap_compare.png
├── canonical_markers/    deg_long.tsv  markers_*.tsv  heatmap_top_markers.png
├── clusters/c0/ ... c{N}/   panel, DEG/QC/composition TSVs,
│                            umap_subcluster.png, minor_anatomy.png
└── report.html              self-contained HTML report (images embedded)
```

## Modules

| module | role |
|---|---|
| `standissect.cluster`  | analysis primitives — UMAP-Leiden partition, per-cluster dissection, vectorised Mann-Whitney DEG, canonical-core markers, minor-anatomy heatmaps |
| `standissect.pipeline` | `run_dissect_pipeline` — staged orchestrator, unified output tree, file-existence idempotency |
| `standissect.report`   | `build_report` — single-file HTML report |

## Tests

```
pytest tests/
```
