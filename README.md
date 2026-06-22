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

**Contents** ·
[Install](#install) ·
[Quickstart](#quickstart) ·
[Inputs](#inputs) ·
[How it works](#how-it-works) ·
[Pipeline](#pipeline--end-to-end) ·
[Verdicts](#verdicts-likely_cause) ·
[Re-running](#re-running) ·
[Output](#output-tree) ·
[Modules](#modules)

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
from an `obs` column. Which column *names* make the verdict specific rather than
generic is spelled out under [Verdicts](#verdicts-likely_cause).

## How it works

For each original cluster, the cells are re-partitioned on their UMAP coords.
The **largest** resulting fragment is that cluster's clean **main core**; every
other fragment above `min_subcluster_size` cells is a **minor** to diagnose:

```
original cluster "3"        UMAP-Leiden re-partition        standissect label
────────────────────        ────────────────────────        ──────────────────
                        ┌──  u5  (8,021 cells, largest) ──▶  c3_0   main core  (kept clean)
  cluster 3   ──────────┼──  u9  (  412 cells)          ──▶  c3_1   minor → diagnose
  (8,800 cells)         └──  u2  (  367 cells)          ──▶  c3_2   minor → diagnose
```

Fragments are named `c{cluster}_{rank}`, rank 0 being the main core. Each cell
ends up with two new labels: `obs['umap_cluster']` (the global re-partition) and
`obs['original_cluster_split']` (the `c{cluster}_{rank}` name). The full
cluster × re-partition contingency table is written to `crosstab.tsv`.

The full run is formulated step by step in [Pipeline — end to end](#pipeline--end-to-end).

## Pipeline — end to end

```
AnnData (X_umap + clustering)
  └─▶ 1. re-partition  ─▶ 2. crosstab + c{cluster}_{rank} naming
        └─▶ 3. dissect minors  (DEG · composition · QC · verdict)
              └─▶ 4. canonical-core markers ─▶ 5. anatomy heatmaps
                    └─▶ 6. assemble panel / params ─▶ 7. report.html
```

Each numbered step writes its result to the [output tree](#output-tree) and is
**skipped when that output already exists** (see [Re-running](#re-running)); the
stage tags `partition` / `dissect` / `canonical` / `anatomy` are the names you
pass to `force=`.

**0 · Preconditions.** `obsm[umap_key]` and `obs[cluster_col]` must exist, or the
run aborts with `KeyError`. Nothing is embedded or clustered from scratch.

**1 · UMAP-Leiden partition** *(stage `partition`)* — build a kNN graph
(`n_neighbors`, default 30) on the 2-D UMAP and run Leiden. If `target_k` is set
(default = the number of original clusters), binary-search `resolution` until the
partition lands within `target_tol` (default 2) clusters of `target_k`, capped at
12 iterations. → per-cell `obs['umap_cluster']` (`u0, u1, …`).

**2 · Crosstab + Cartesian naming** *(always — cheap)* — cross-tabulate
`cluster_col × umap_cluster` → `crosstab.tsv`. Within each original cluster, rank
its UMAP fragments by size and name them `c{cluster}_{rank}` (rank 0 = largest =
**main core**). → `obs['original_cluster_split']`, `cell_labels.tsv`.

**3 · Per-cluster dissection** *(stage `dissect`, one cluster at a time)* — for
each original cluster, with **main** = `c{cluster}_0` and **minors** = the
off-main fragments holding ≥ `min_subcluster_size` (default 50) cells:

- **DEG** — minor vs main core, a vectorised Mann-Whitney (Wilcoxon rank-sum) on
  `adata.X` (or `deg_layer`); keep the top `top_n_deg` genes. A gene counts as
  *significant* when `pvals_adj < 0.05` **and** `|log2FC| > 0.5` (BH-FDR).
- **Composition drift** — for every `cat_cols` column, a 2×2 Fisher exact per
  category (minor vs main, this category vs the rest; Haldane–Anscombe `+0.5`,
  BH-FDR), reported as a log2 odds ratio.
- **QC drift** — for every `qc_cols` column, a Mann-Whitney of minor vs main
  (BH-FDR), recording the mean shift `Δ` and its relative size.
- **Verdict** — fold the strongest sample-enrichment and QC-drift signals
  together with the DEG count into one `likely_cause`
  (rules in [Verdicts](#verdicts-likely_cause)).

→ `clusters/c{N}/`: `panel.tsv`, `deg_*.tsv`, `qc_drift_*.tsv`,
`composition_*.tsv`, `umap_subcluster.png`.

**4 · Canonical-core markers** *(stage `canonical`)* — one-vs-rest Wilcoxon
markers for each cluster's clean core (its dominant fragment), gene-chunked
(`wilcoxon_chunk_size`, default 3000) to bound memory; keep the top
`top_n_canonical` genes. → `canonical_markers/`: `deg_long.tsv`, `markers_c*.tsv`,
`heatmap_top_markers.png`.

**5 · Minor-anatomy heatmaps** *(stage `anatomy`)* — per cluster, one heatmap
placing every minor against the core's canonical markers, the QC columns, and the
sample composition. → `clusters/c{N}/minor_anatomy.png`.

**6 · Assembly** *(always — cheap)* — concatenate the per-cluster panels →
`panel.tsv` (the headline table) and the QC drift → `qc_drift_all.tsv`; redraw
`global_umap_compare.png`; dump every resolved parameter → `params.json`. If
`labeled_h5ad_path` is given, the labelled AnnData is written via an atomic
temp-file swap.

**7 · Report** — `build_report(result["root"])` inlines every table and PNG into a
single self-contained `report.html`.

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
