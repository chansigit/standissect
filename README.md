# standissect

> **Diagnose what's hiding inside your single-cell clusters.**

You already clustered your cells (Leiden, etc.) and have a UMAP. For each
cluster, `standissect` asks: *is this one clean population, or a clean core plus
a few stowaway fragments?* It re-clusters the cells on their UMAP coordinates,
isolates the fragments that don't belong to a cluster's main blob — its
**minors** — and for each one reports the marker genes, composition/QC drift,
and a one-word `likely_cause`. That lets you tell a doublet pocket or a
low-quality tail apart from a genuine rare subpopulation.

It runs *downstream* of your scanpy pipeline — it neither embeds nor clusters
from scratch; it cleans up and explains a clustering you already have.

## Install

This is a flat package with no build config — cloning gives you a `standissect/`
directory you import directly. Put its parent on the path:

```
git clone https://github.com/chansigit/standissect.git
export PYTHONPATH="$PWD:$PYTHONPATH"     # $PWD is the parent of standissect/
python -c "import standissect"           # smoke test
```

Requires `numpy`, `pandas`, `scipy`, `statsmodels`, `scanpy`/`anndata`,
`scikit-learn`, `python-igraph`, `leidenalg`, `matplotlib`, `seaborn`.

## Quickstart

```python
import anndata as ad
from standissect import run_dissect_pipeline, build_report

adata = ad.read_h5ad("my_data.h5ad")        # needs obsm['X_umap'] + a clustering in obs
result = run_dissect_pipeline(
    adata,
    cluster_col="leiden",
    output_dir="results/dissect",
    labeled_h5ad_path="my_data.with_dissect_labels.h5ad",
    cat_cols=("orig.ident", "batch"),       # categorical obs columns → composition drift
    qc_cols=("percent.mt", "nCount_RNA", "nFeature_RNA", "hybrid_score"),
    sample_col="orig.ident",
)
build_report(result["root"])                # -> results/dissect/leiden/report.html
```

Open `results/dissect/leiden/report.html` — a single self-contained file (images
embedded) with a global overview and a per-cluster breakdown.

## How it works

For each original cluster, the cells are re-partitioned on their UMAP coords.
The **largest** resulting fragment is that cluster's clean **main core**; every
other fragment above `min_subcluster_size` cells is a **minor** to diagnose:

```
original cluster "3"        UMAP-Leiden re-partition        standissect label
────────────────────        ────────────────────────        ──────────────────────
                        ┌──  u5  (8,021 cells, largest) ──▶  c3_0   main core   (kept clean)
  cluster 3   ──────────┼──  u9  (  412 cells)          ──▶  c3_1   minor  → diagnose
  (8,800 cells)         └──  u2  (  367 cells)          ──▶  c3_2   minor  → diagnose
```

Fragments are named `c{cluster}_{rank}`, rank 0 being the main core. Each cell
ends up with two new labels: `obs['umap_cluster']` (the global re-partition) and
`obs['original_cluster_split']` (the `c{cluster}_{rank}` name). The full
cluster × re-partition contingency table is written to `crosstab.tsv`.

The pipeline runs in four stages — each independently skippable via `force=`
(see [Re-running](#re-running)):

1. **partition** — kNN + Leiden on the 2-D UMAP → the re-partition above.
2. **dissect** — per cluster: find minors, run DEG vs the core, measure
   composition + QC drift, assign a `likely_cause`.
3. **canonical** — canonical-core markers per cluster: what each *clean* core is.
4. **anatomy** — a per-cluster heatmap placing each minor against the core's
   markers and QC.

## Verdicts (`likely_cause`)

Every minor gets exactly one verdict. Rules are checked top-to-bottom and the
first match wins; the QC rows are evaluated against the single most-strongly-
drifted *significant* QC column (all drift tests use BH-FDR `padj < 0.05`).

| verdict | what it means | fires when |
|---|---|---|
| `sample-driven` | mostly one sample/donor — a batch or donor artifact, not a cell state | a `cat_cols` column named `orig.ident` is enriched (`log2_OR ≥ 2`) |
| `doublet-driven` | enriched for doublets | `hybrid_score` drifts up (`relative_delta > 0.5`) |
| `low-quality (high mt)` | dying / stressed cells | `percent.mt` drifts up (`delta > 2`) |
| `shallow-depth` | under-sequenced cells | `nFeature_RNA` drifts down (`relative_delta < −0.3`) |
| `biology-candidate` | none of the above, yet clearly distinct — a real candidate | `≥ 20` significant DEGs vs the core |
| `unclear` | not enough signal to call | otherwise |

**The column names in the last column are literal.** The composition/QC
*machinery* works on any column you pass, but these four verdict branches only
fire for columns named exactly `orig.ident`, `hybrid_score`, `percent.mt`, and
`nFeature_RNA`. Rename your `obs` columns to match (or pass them under these
names); otherwise even a clear artifact only ever reaches `biology-candidate` or
`unclear`.

## Inputs

`standissect` reads the embedding and clustering straight from `adata`; the
QC/sample columns are optional but are what unlock the per-minor diagnosis.

| what | where | required? | how to specify |
|---|---|---|---|
| 2-D embedding | `adata.obsm[umap_key]` | **yes** — `KeyError` if absent | `umap_key="X_umap"` (default) |
| existing clustering | `adata.obs[cluster_col]` | **yes** — `KeyError` if absent | `cluster_col="leiden"` |
| categorical cols (composition drift) | `adata.obs[...]` | optional — missing cols silently skipped | `cat_cols=("orig.ident", "batch")` |
| continuous QC cols (QC drift) | `adata.obs[...]` | optional — missing cols silently skipped | `qc_cols=("percent.mt", "nCount_RNA", "nFeature_RNA", "hybrid_score")` |
| sample col (anatomy heatmap) | `adata.obs[sample_col]` | optional | `sample_col="orig.ident"` |

The expression for DEG comes from `adata.X` (or a layer via `deg_layer=`), not
from an `obs` column. For which column *names* make the verdict specific rather
than generic, see [Verdicts](#verdicts-likely_cause) above.

## Re-running

The pipeline is **idempotent**: re-running skips any stage whose output files
already exist, so an interrupted or extended run resumes cheaply. To force a
recompute, name the stages to redo:

```python
run_dissect_pipeline(adata, ..., force=("partition", "dissect"))  # or force="all"
```

Valid stage names are `partition`, `dissect`, `canonical`, `anatomy`. Each run
overwrites `adata.obs["umap_cluster"]` and `adata.obs["original_cluster_split"]`
**in memory**; pass `labeled_h5ad_path=` to also persist those labels to an
h5ad on disk.

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

`panel.tsv` is the headline table — one row per minor across all clusters, with
its top genes, top drift, and verdict. `report.html` is what you actually open.

## Modules

| module | role |
|---|---|
| `standissect.cluster`  | analysis primitives — UMAP-Leiden partition, per-cluster dissection, vectorised Mann-Whitney DEG, canonical-core markers, minor-anatomy heatmaps |
| `standissect.pipeline` | `run_dissect_pipeline` — staged orchestrator, unified output tree, file-existence idempotency |
| `standissect.report`   | `build_report` — single-file HTML report |
