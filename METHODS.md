# Methods

## Overview

`standissect` is a diagnostic procedure for finding and evaluation of 
off-core fragmented noisy cell populations in a scRNA-seq dataset.
It starts with a pre-existing clustering result and a 2D embedding such as UMAP,
splits the clusters into major cores and minor fragmented outliers, analyzes the
fragmented outliers' identities, and finally decide whether to retain the
outlier minor clusters or not.  The decisions are made in a comprehensive manner
that combines DEGs, QC metrics, and LLM judgements. 


The procedure takes as input an `AnnData` object with a two-dimensional embedding
in `adata.obsm[umap_key]` and an existing cluster assignment in
`adata.obs[cluster_col]`. Optional cell-level metadata can be supplied through
role-specific arguments. Source-like categorical columns such as sample, batch,
donor, or library preparation are used to test whether a minor fragment is
enriched for a particular acquisition source. Continuous quality-control columns
such as doublet score, mitochondrial fraction, detected features, and UMI count
are used to test QC drift. The main output is a per-minor panel containing marker
genes, composition and quality-control signals, a categorical `likely_cause`, and
audit files recording the evidence used for diagnosis.

## UMAP-based re-partitioning

Let \(x_i \in \mathbb{R}^2\) denote the supplied UMAP coordinates for cell
\(i\), and let \(g_i\) denote its original cluster label. A k-nearest-neighbor
graph is constructed in the two-dimensional UMAP space using \(k\) neighbors
(`n_neighbors`, default 30). The graph is symmetrized and partitioned with
Leiden community detection using the RBConfiguration objective.

The Leiden resolution controls the granularity of the UMAP fragments. If a
target number of UMAP fragments is specified, the resolution is iteratively
adjusted until the number of fragments lies within a tolerance of that target,
or until a maximum number of search iterations is reached. By default, the target
is the number of original clusters, which provides a comparable global
partitioning scale without constraining any individual cluster to split into a
fixed number of fragments. The resulting UMAP partition labels are size-ranked
globally and stored as `umap_cluster` labels.

## Mapping UMAP fragments to original clusters

For each original cluster \(c\), `standissect` computes the overlap between
cells in \(c\) and each UMAP fragment \(u\):

\[
n_{c,u} = |\{i: g_i = c,\; z_i = u\}|,
\]

where \(z_i\) is the UMAP-fragment label. Within each original cluster, UMAP
fragments are ranked by \(n_{c,u}\). The largest fragment is defined as the
major core of that original cluster and is assigned the split label
`c{parent}_0`. All remaining non-empty fragments are assigned split labels
`c{parent}_1`, `c{parent}_2`, and so on in descending size order.

Every cell therefore receives two additional labels: `umap_cluster`, the global
UMAP-fragment assignment, and `original_cluster_split`, the parent-cluster-aware
split label. The overlap matrix is written to `cluster_overlap.tsv`, and the
per-cell labels are written to `cell_labels.tsv`.

## Minor fragment definition

For a parent cluster \(c\), the reference subcluster is always its own major
core:

\[
\mathrm{reference}(c) = c_0.
\]

Any off-core fragment \(c_j\), \(j > 0\), containing at least
`min_subcluster_size` cells is treated as a minor fragment to diagnose. Smaller
fragments are labeled in the per-cell output but are not subjected to the full
minor diagnostic procedure. This threshold avoids over-interpreting very small
geometric fragments.

## Minor-versus-own-core evidence

For each minor fragment \(c_j\), all primary statistical evidence is computed
against its own parent major core \(c_0\). This reference is recorded explicitly
as `reference_subcluster` in the panel and diagnosis audit files. Comparisons to
other major cores are computed only as contextual evidence and do not replace
the primary minor-versus-own-core reference.

### Differential expression

Differential expression is computed between cells in the minor fragment \(c_j\)
and cells in the corresponding major core \(c_0\). Expression values are taken
from `adata.X` by default. If `deg_layer="counts_recovered"` is requested, the
specified count layer is normalized to a fixed library size and log-transformed
before testing.

For each gene, a vectorized two-sided Mann-Whitney U test is performed between
the minor and the core. The implementation computes a normal-approximation test
statistic and applies Benjamini-Hochberg false-discovery-rate correction across
genes. Log fold changes are computed on the expression scale used by the input
matrix, with log-normalized data transformed by `expm1` before fold-change
calculation. A gene is counted as significantly differentially expressed when
the adjusted p-value is below 0.05 and the absolute log2 fold change is greater
than 0.5. The top up-regulated and down-regulated genes are stored for each
minor.

### Composition drift

For each supplied categorical metadata column, `standissect` compares the
category distribution in the minor against the category distribution in its own
major core. These columns are supplied through role-specific arguments such as
sample, batch, donor, library, and condition, plus optional additional
categorical evidence columns. For each category, a 2-by-2 Fisher exact test is
performed: minor versus core by category versus all other categories. A
Haldane-Anscombe correction of 0.5 is added to all four cells of the table before
computing the log2 odds ratio. P-values are adjusted across categories within
the covariate using Benjamini-Hochberg correction.

### Quality-control drift

For each supplied continuous quality-control variable, the minor is compared
against its own major core using a two-sided Mann-Whitney U test. These variables
are supplied through role-specific arguments for doublet score, mitochondrial
fraction, detected feature count, UMI count, and optional additional QC evidence
columns. The method records the mean value in the minor, the mean value in the
core, the absolute difference, the relative difference, and the adjusted p-value
across tested QC variables.

## Cross-major-core context

In addition to the primary minor-versus-own-core evidence, `standissect`
computes a compact contextual comparison between the minor and all major cores
`c*_0`. The diagnostic genes used for this comparison are selected from the
minor's top up-regulated and down-regulated genes. For the minor and for each
major core, the method computes the mean expression vector over these genes.
It then reports the Pearson correlation and mean absolute expression difference
between the minor vector and each major-core vector.

The minor's own reference core is always included in this context block, followed
by the nearest other major cores ranked by correlation on the selected
diagnostic genes. This block helps distinguish a minor that is merely an
off-core tail from one that resembles another established major population. It
is intended for interpretation and language model-assisted diagnosis; it does
not change the primary reference used for differential expression, composition
testing, or QC drift.

## Diagnosis assignment

Each diagnosable minor receives exactly one `likely_cause` from a fixed
enumeration:

- `sample-driven`
- `doublet-driven`
- `low-quality (high mt)`
- `shallow-depth`
- `biology-candidate`
- `unclear`

The default diagnosis engine is deterministic. Rules are evaluated in priority
order and depend on semantic column roles rather than literal column names. A
minor is called `sample-driven` if any source-like role column, such as sample,
batch, donor, or library, has a significantly enriched category with log2 odds
ratio at least 2. It is called `doublet-driven` if the configured doublet-score
column is significantly increased with relative delta above 0.5. It is called
`low-quality (high mt)` if the configured mitochondrial column is significantly
increased by more than 2 percentage points, and `shallow-depth` if the
configured feature-count or UMI-count column is significantly decreased with
relative delta below -0.3. If no artifact rule fires but the minor has at least
20 significant differentially expressed genes, it is labeled
`biology-candidate`. Otherwise, it is labeled `unclear`.

When language model-assisted diagnosis is enabled, the model receives only a
compact JSON evidence packet. This packet contains the minor size, its own
reference subcluster, top differential-expression rows, composition enrichment
rows, QC drift rows, cross-major-core context, and, in hybrid mode, the rule
baseline. The model is not given the raw expression matrix. It must return
strict JSON containing one allowed `likely_cause`, a short `cause_detail`, a
confidence score, a rationale, evidence used, alternative causes, recommended
checks, and an indicator of whether it overrode the rule baseline. If the model
call fails or produces invalid output and fallback is enabled, the deterministic
rule baseline is used.

For auditability, each minor's compact diagnosis input and final diagnosis
output are written to `diagnosis_*.input.json` and `diagnosis_*.output.json`,
respectively. The global diagnosis summary is written to `diagnosis_all.tsv`.

## Canonical-core marker analysis

After the minor diagnosis stage, `standissect` computes canonical markers for
the major cores. For each original cluster, cells in the dominant UMAP fragment
are treated as the canonical core. One-versus-rest Mann-Whitney testing is then
performed across all canonical cores. This analysis is gene-chunked to bound
memory usage. The resulting marker table is written in long format, with
per-core marker files and a summary heatmap of top canonical markers.

## Minor-profile visualization

For each parent cluster, `standissect` builds a minor-profile heatmap comparing
the parent minor fragments against the collection of canonical cores. The heatmap
contains a gene-expression block, QC tracks, and sample-composition tracks.
Genes are selected from canonical-core markers and minor-specific markers, and
expression values are summarized as mean expression per split label before
row-wise z-scoring for display. These visualizations provide a compact view of
whether a minor appears to be a technical artifact, a tail of its own core, or a
fragment resembling another major population.

## Outputs and reproducibility

The pipeline is organized as idempotent stages. Stages whose primary output
already exists are skipped unless explicitly forced. This allows interrupted
runs to resume and allows diagnosis to be recomputed without rerunning
differential expression or visualization. The output directory contains the
cluster-overlap matrix, per-cell labels, per-cluster evidence files, diagnosis
audit JSON files, canonical-core markers, profile heatmaps, global summary
tables, resolved parameters, and a self-contained HTML report.

All resolved parameters, including the diagnosis mode, model identifier when
applicable, and prompt version, are written to `params.json`. API keys and other
secrets are never written to the output directory.
