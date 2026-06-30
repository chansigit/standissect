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
[Diagnoses](#diagnoses-likely_cause) ·
[Dispositions](#recommended-discards--dispositions) ·
[Re-running](#re-running) ·
[Output](#output-tree) ·
[Review UI](#reviewing-a-run-interactively-serve) ·
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
| existing cell-type annotation | `adata.obs[annotation_col]` | optional — but if given, **must exist** (`KeyError` otherwise) | `annotation_col="cell_ontology_class"` |
| categorical cols (composition drift) | `adata.obs[...]` | optional — missing cols silently skipped | `cat_cols=("orig.ident", "batch")` |
| continuous QC cols (QC drift) | `adata.obs[...]` | optional — missing cols silently skipped | `qc_cols=("percent.mt", "nCount_RNA", "nFeature_RNA", "hybrid_score")` |
| sample col (anatomy heatmap) | `adata.obs[sample_col]` | optional | `sample_col="orig.ident"` |

The expression for DEG comes from `adata.X` (or a layer via `deg_layer=`), not
from an `obs` column. Which column *names* make the diagnosis specific rather than
generic is spelled out under [Diagnoses](#diagnoses-likely_cause).

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
`obs['original_cluster_split']` (the `c{cluster}_{rank}` name). A table of how
many cells each original cluster shares with each UMAP fragment is written to
`cluster_overlap.tsv`.

The full run is formulated step by step in [Pipeline — end to end](#pipeline--end-to-end).

## Pipeline — end to end

```
AnnData (X_umap + clustering)
  └─▶ 1. re-partition  ─▶ 2. overlap table + c{cluster}_{rank} naming
        └─▶ 3. dissect minors  (DEG · composition · QC · diagnosis)
              └─▶ 4. canonical-core markers ─▶ 5. profile heatmaps
                    └─▶ 6. assemble panel / params ─▶ 7. report.html
```

Each numbered step writes its result to the [output tree](#output-tree) and is
**skipped when that output already exists** (see [Re-running](#re-running)); the
stage tags `partition` / `dissect` / `canonical` / `profile` are the names you
pass to `force=`.

**0 · Preconditions.** `obsm[umap_key]` and `obs[cluster_col]` must exist, or the
run aborts with `KeyError`. Nothing is embedded or clustered from scratch.

**1 · UMAP-Leiden partition** *(stage `partition`)* — build a kNN graph
(`n_neighbors`, default 30) on the 2-D UMAP and run Leiden. If `target_k` is set
(default = the number of original clusters), binary-search `resolution` until the
partition lands within `target_tol` (default 2) clusters of `target_k`, capped at
12 iterations. → per-cell `obs['umap_cluster']` (`u0, u1, …`).

**2 · Overlap table + ranked naming** *(always — cheap)* — count how many
cells each `cluster_col` value shares with each `umap_cluster` fragment →
`cluster_overlap.tsv`. Within each original cluster, rank
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
- **Diagnosis** — fold the strongest sample-enrichment and QC-drift signals
  together with the DEG count into one `likely_cause`
  (rules in [Diagnoses](#diagnoses-likely_cause)).

→ `clusters/c{N}/`: `panel.tsv`, `subcluster_labels.tsv`, `deg_*.tsv`,
`qc_drift_*.tsv`, `composition_*.tsv`, `umap_subcluster.png`.

**4 · Canonical-core markers** *(stage `canonical`)* — one-vs-rest Wilcoxon
markers for each cluster's clean core (its dominant fragment), gene-chunked
(3000 genes at a time) to bound memory; keep the top
`top_n_canonical` genes. → `canonical_markers/`: `deg_long.tsv`, `markers_c*.tsv`,
`heatmap_top_markers.png`.

**5 · Minor-profile heatmaps** *(stage `profile`)* — per cluster, one heatmap
placing every minor against the core's canonical markers, the QC columns, and the
sample composition. → `clusters/c{N}/minor_profile.png` (with `heatmap_data.tsv` /
`qc_tracks.tsv` / `sample_composition.tsv` / `genes_*.txt` sidecars).

**6 · Assembly** *(always — cheap)* — concatenate the per-cluster panels →
`panel.tsv` (the headline table) and the QC drift → `qc_drift_all.tsv`; redraw
`global_umap_compare.png`; dump every resolved parameter → `params.json`. If
`labeled_h5ad_path` is given, the labelled AnnData is written via an atomic
temp-file swap.

**7 · Report** — `build_report(result["root"])` inlines every table and PNG into a
single self-contained `report.html`.

## Diagnoses (`likely_cause`)

Every minor gets exactly one diagnosis. Rules are checked top-to-bottom and the
first match wins; the QC rows are evaluated against the single most-strongly-
drifted *significant* QC column (all drift tests use BH-FDR `padj < 0.05`).

| diagnosis | what it means | fires when |
|---|---|---|
| `sample-driven` | mostly one sample/donor — a batch or donor artifact, not a cell state | a `cat_cols` column named `orig.ident` is enriched (`log2_OR ≥ 2`) |
| `doublet-driven` | enriched for doublets | `hybrid_score` drifts up (`relative_delta > 0.5`) |
| `low-quality (high mt)` | dying / stressed cells | `percent.mt` drifts up (`delta > 2`) |
| `shallow-depth` | under-sequenced cells | `nFeature_RNA` drifts down (`relative_delta < −0.3`) |
| `biology-candidate` | none of the above, yet clearly distinct — a real candidate | `≥ 20` significant DEGs vs the core |
| `unclear` | not enough signal to call | otherwise |

**The column names in the last column are literal.** The composition/QC
*machinery* works on any column you pass, but these four diagnosis branches only
fire for columns named exactly `orig.ident`, `hybrid_score`, `percent.mt`, and
`nFeature_RNA`. Rename your `obs` columns to match (or pass them under these
names); otherwise even a clear artifact only ever reaches `biology-candidate` or
`unclear`.

### Existing annotation as a consistency-check prior (`annotation_col`)

`cluster_col` is the partition you want dissected — it may be numeric Leiden ids
with no biological meaning. If you *also* have a curated cell-type annotation in
`obs`, point `--annotation-col` at it. For each minor fragment (and its main /
reference fragment) standissect then computes the per-cell annotation
composition (`annotation: n_cells, frac`) and hands it to the **LLM** diagnosis
as a *consistency check*:

- when a minor fragment's dominant existing annotation differs from its parent
  or main fragment, that supports `ambient-contamination`, `doublet-driven`, or a
  genuinely distinct/finer cell type (→ `proposed_cell_type`);
- the LLM is explicitly told **not to blindly trust** the existing annotation —
  it is weighed against the DEG/QC/composition evidence and may itself be wrong,
  so the "catch a mislabel" capability is preserved.

This is **LLM-only** (the rule baseline ignores it). It is optional, but if you
pass a column name it must already exist in `obs` (otherwise `KeyError`). The
chosen column is recorded in `params.json` (`annotation_col`).

## Recommended discards & dispositions

After diagnosis every minor gets a `recommended_disposition` (written to
`panel.tsv`, `diagnosis_all.tsv`, and `cell_labels.tsv`):

| value | meaning |
|---|---|
| `DISCARD` | strong evidence of a technical artifact; cells are candidates for removal |
| `UNCERTAIN` | ambiguous — kept by default but flagged for manual review |
| `KEEP` | genuine biology or too weak a signal to act on |

**Conservative-only rule.** Automation never moves toward DISCARD compared with
the cause baseline. An LLM override is accepted only if it is at least as
keep-leaning as the rule baseline; a DISCARD call below `--discard-confidence-threshold`
(default 0.5) is automatically downgraded to UNCERTAIN.

### `likely_cause` → baseline disposition

| `likely_cause` | baseline | rationale |
|---|---|---|
| `doublet-driven` | DISCARD | doublets are artefacts |
| `low-quality (high mt)` | DISCARD | dying/stressed cells |
| `shallow-depth` | DISCARD | under-sequenced |
| `dissociation-effect` | DISCARD | transcriptional noise from dissociation |
| `ambient-contamination` | DISCARD | background RNA contamination |
| `sample-driven` | UNCERTAIN | may be biology; review per-sample |
| `unclear` | UNCERTAIN | insufficient signal to call |
| `cell-cycle` | KEEP | known biology |
| `sex-driven` | KEEP | known biology |
| `interferon-response` | KEEP | known biology |
| `biology-candidate` | KEEP | distinct DEG profile, likely real |

### New outputs

| file | description |
|---|---|
| `panel.tsv` | gains `disposition_baseline`, `recommended_disposition`, `disposition_overridden`, `disposition_reason`, `proposed_cell_type` columns |
| `diagnosis_all.tsv` | same disposition columns, diagnosis-focused subset |
| `discard_cells.tsv` | one row per DISCARD cell — `barcode` (obs_name, the stable cross-version key) + `input_row_index` (0-based row position in the input adata) + subcluster/cause/confidence/reason |
| `proposed_cell_types.tsv` | LLM-proposed cell-type relabels — `minor` (per-subcluster `proposed_cell_type`) and `major` (`differs_from_original` core renames) |

The HTML report gains a **Recommended discards** section (DISCARD table + collapsible UNCERTAIN list) and a **Proposed cell types** section.

### CLI flags

`--discard-confidence-threshold FLOAT` (default `0.5`) — DISCARD calls below this
confidence are downgraded to UNCERTAIN.

`--apply-discard PATH` — after the pipeline finishes, write a cleaned `.h5ad` to
exactly `PATH` with all `recommended_disposition==DISCARD` cells removed. KEEP and
UNCERTAIN cells are retained. Off when omitted.

After the naming stage, all three outputs carry per-cell annotation columns resolved
via a fallback chain (minor subcluster `proposed_cell_type` → major core `cell_type`
from `core_names.tsv` → original `cluster_col` label, so every cell always gets a
value):

| Output | New `obs` columns |
|---|---|
| cleaned h5ad (`--apply-discard`) | `recommended_disposition`, `proposed_cell_type` |
| labeled h5ad (`--labeled-h5ad-path`) | `recommended_disposition`, `proposed_cell_type` |
| `cell_labels.tsv` | `recommended_disposition` (from diagnosis), `proposed_cell_type` |

## Re-running

The pipeline is **idempotent**: re-running skips any stage whose output files
already exist, so an interrupted or extended run resumes cheaply. To force a
recompute, name the stages to redo:

```python
run_dissect_pipeline(adata, ..., force=("partition", "dissect"))  # or force="all"
```

Valid stage names are `partition`, `dissect`, `canonical`, `profile`. Each run
overwrites `adata.obs["umap_cluster"]` and `adata.obs["original_cluster_split"]`
**in memory**; pass `labeled_h5ad_path=` to also persist those labels to an
h5ad on disk.

## Output tree

```
<output_dir>/<cluster_col>/
├── cluster_overlap.tsv  panel.tsv  cell_labels.tsv  qc_drift_all.tsv  params.json
├── global_umap_compare.png
├── canonical_markers/    deg_long.tsv  markers_*.tsv  heatmap_top_markers.png
├── clusters/c0/ ... c{N}/   panel.tsv  subcluster_labels.tsv
│                            deg_*.tsv  qc_drift_*.tsv  composition_*.tsv
│                            heatmap_data.tsv  qc_tracks.tsv  sample_composition.tsv
│                            genes_canonical.txt  genes_minor.txt
│                            umap_subcluster.png  minor_profile.png
└── report.html              self-contained HTML report (images embedded)
```

`panel.tsv` is the headline table — one row per minor across all clusters, with
its top genes, top drift, and diagnosis. `report.html` is what you actually open.

## Reviewing a run interactively (`serve`)

`report.html` is read-only. To **record your own keep/discard verdicts** per
minor subcluster — and to lasso-select cells on the UMAP — run the review
server. It only *reads* the run tree and *writes* decision files; it never opens
the source `.h5ad`, so it is safe on a login node behind a tunnel.

It needs two optional dependencies (not required by the pipeline):

```bash
pip install fastapi uvicorn          # run in sh_dev, not on the login node
```

> The commands below use the **ts-blood** run on Sherlock as a concrete,
> copy-pasteable example — substitute your own `.h5ad` and output paths. They
> run `python -m standissect` from the parent of `standissect/`
> (`/scratch/users/chensj16/projects`) so the package resolves via the current
> directory; a `PYTHONPATH=…` prefix can break `h5py`'s bundled libraries in
> this venv, while `cd` + `-m` does not.

**1. (optional) Export per-cell coordinates** for the interactive UMAP. The
server runs without this — the UMAP tab is simply hidden — but to enable it,
dump `cell_coords.tsv.gz` once. This reads the `.h5ad` (use `sh_dev`/a job, and
pass the embedding key your run used — check `params.json` → `umap_key`, which is
`X_umap_harmony` for this run):

```bash
sh_dev
cd /scratch/users/chensj16/projects          # parent of standissect/
python -m standissect export-coords \
    /scratch/users/chensj16/sc-curation-output/ts-blood/Blood_TSP1_30_version2d_10X_smartseq_scvi_Nov122024_dissected.h5ad \
    --output-dir /scratch/users/chensj16/sc-curation-output/ts-blood/dissect/harmony_leiden \
    --umap-key X_umap_harmony \
    --doublet-score-col doublet_score --mito-col pct_counts_mt \
    --feature-count-col n_genes_by_counts --umi-count-col total_counts
```

The UMAP shows the cells present in *both* this `.h5ad` and `cell_labels.tsv`.
If you point `export-coords` at an `--apply-discard` *cleaned* `.h5ad`, the
already-discarded cells won't appear; pass the pre-discard input instead to see
them.

**2. Launch the server** — on a **compute node** (not the login node). Run it
from the *parent* of `standissect/` so `python -m` finds the package via the
current directory. Avoid a `PYTHONPATH=…` prefix — in some venvs it breaks
`h5py`'s bundled libraries; `cd`-ing to the parent and using `-m` does not:

```bash
sh_dev -t 02:00:00                            # a compute node for the review session
cd /scratch/users/chensj16/projects           # parent of standissect/
python -m standissect serve \
    /scratch/users/chensj16/sc-curation-output/ts-blood/dissect/harmony_leiden \
    --host 127.0.0.1 --port 8050
# extra flags: --decisions-file PATH (default <root>/human_review.tsv) · --reviewer NAME
```

You should see `review server for harmony_leiden -> http://127.0.0.1:8050`. Leave
it running while you review.

> `sh_dev` caps at 2 h. For a longer session use e.g.
> `salloc -p normal -t 08:00:00` (or your owner partition) and run the same
> command on the allocated node.

**3. Open it in your browser via ngrok** (public URL, reachable anywhere). ngrok
must run on the **same node** as the server. The one-shell recipe — background
the server, then start ngrok:

```bash
ngrok config add-authtoken <YOUR_TOKEN>       # once per machine (get it from dashboard.ngrok.com)

cd /scratch/users/chensj16/projects
python -m standissect serve \
    /scratch/users/chensj16/sc-curation-output/ts-blood/dissect/harmony_leiden \
    --port 8050 &                             # background (note the trailing &)
ngrok http --basic-auth "user:reviewpass1" 8050   # foreground; ALWAYS set auth — the app has none
#                          ^login  ^password (8–128 chars; pick your own)
```

ngrok prints a `Forwarding  https://<random>.ngrok-free.app -> http://localhost:8050`
line. Open that HTTPS URL, log in with the login/password you chose, and you're
in. When done: `kill %1` stops the backgrounded server, `Ctrl-C` stops ngrok.

> Recent ngrok prints `Flag --basic-auth has been deprecated` — it still works
> (just a warning). The password must be **8–128 characters**, or ngrok errors
> with `ERR_NGROK_360`.

The app has **no authentication of its own**, so never expose it without
`--basic-auth` (anyone with the URL could otherwise read your run and write
decision files).

**Alternative — SSH tunnel** (no third party). Bind the server to all
interfaces (`--host 0.0.0.0`), find your node with `squeue --me`, then from your
laptop:

```bash
# <compute-node> is the node sh_dev/salloc gave you (e.g. sh02-09n46), from `squeue --me`
ssh -N -L 8050:<compute-node>:8050 chensj16@login.sherlock.stanford.edu
# open http://localhost:8050
```

**What you can do:**

- per cluster, see each minor's heatmap / UMAP / DEG / QC drift / LLM diagnosis,
  and click **KEEP / DISCARD / UNCERTAIN** (+ a note); **adopt LLM** copies the
  model's call. Progress is tracked in the sidebar; decisions autosave and
  resume across restarts.
- **compare a minor vs the major (core) cluster** on the interactive UMAP
  (cluster-focus mode highlights the minor against `_0`).
- **lasso / box-select** cells to inspect their composition + QC live, **export
  their barcodes**, or record them as a **manual KEEP/DISCARD set**.

**Outputs (in the run dir):** `human_review.tsv` (per-minor verdicts),
`manual_cells.tsv` (hand-picked cells), `selections/selection_*.tsv` (exported
barcode lists). These are decisions only — nothing is deleted; feed them into
your own `--apply-discard` step if you want a cleaned `.h5ad`.

## Modules

| module | role |
|---|---|
| `standissect.cluster`  | analysis primitives — UMAP-Leiden partition, per-cluster dissection, vectorised Mann-Whitney DEG, canonical-core markers, minor-profile heatmaps |
| `standissect.pipeline` | `run_dissect_pipeline` — staged orchestrator, unified output tree, file-existence idempotency |
| `standissect.report`   | `build_report` — single-file HTML report |
| `standissect.webreview` | `serve` / `build_app` — interactive review server (dashboard + Plotly UMAP lasso) |
| `standissect.export_coords` | `export_cell_coords` — per-cell UMAP coords (+QC) → `cell_coords.tsv.gz` |
| `standissect.review_store` | `ReviewStore` / `ManualStore` — flat-file decision stores |
