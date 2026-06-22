"""standissect.pipeline — unified orchestrator for the cluster cleanup-diagnosis.

``run_dissect_pipeline`` runs the three stages (partition + dissect,
canonical-core markers, minor-anatomy heatmaps) into one output tree, with
file-existence idempotency. The analysis primitives live in
``standissect.cluster``.

The stages run serially and memory-bounded — the canonical-core Wilcoxon is
gene-chunked to stay within a modest memory budget; the per-minor DEG is a
vectorised in-process Mann-Whitney. Process-pool parallelism was tried and
dropped (a fork pool deadlocks after heavy threaded-BLAS use).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cluster import (umap_leiden_partition, dissect_one_cluster,
                      canonical_marker_deg, plot_minor_anatomy)

_STAGES = ('partition', 'dissect', 'canonical', 'anatomy')

_PANEL_COLS = ['parent_cluster', 'subcluster', 'minor_umap_label',
               'main_umap_label', 'n_cells', 'frac_of_parent',
               'top5_up_genes', 'top5_down_genes', 'n_sig_genes',
               'top_sample_enriched', 'top_qc_drift', 'likely_cause']


class _Layout:
    """Path scheme for the unified output tree ``<output_dir>/<cluster_col>/``."""

    def __init__(self, output_dir, cluster_col):
        self.root = Path(output_dir) / str(cluster_col)
        self.clusters = self.root / 'clusters'
        self.canonical = self.root / 'canonical_markers'

    @property
    def crosstab(self):      return self.root / 'cluster_overlap.tsv'
    @property
    def panel(self):         return self.root / 'panel.tsv'
    @property
    def cell_labels(self):   return self.root / 'cell_labels.tsv'
    @property
    def qc_drift_all(self):  return self.root / 'qc_drift_all.tsv'
    @property
    def params(self):        return self.root / 'params.json'
    @property
    def global_umap(self):   return self.root / 'global_umap_compare.png'
    @property
    def canonical_deg(self): return self.canonical / 'deg_long.tsv'

    def cluster_dir(self, parent):    return self.clusters / f"c{parent}"
    def cluster_panel(self, parent):  return self.cluster_dir(parent) / 'panel.tsv'
    def anatomy_png(self, parent):    return self.cluster_dir(parent) / 'minor_anatomy.png'


def _normalize_force(force):
    """Coerce the ``force`` argument to a set of stage names."""
    if force is True or force == 'all':
        return set(_STAGES)
    if not force:
        return set()
    if isinstance(force, str):
        force = (force,)
    force = set(force)
    bad = force - set(_STAGES)
    if bad:
        raise ValueError(f"unknown force stage(s) {sorted(bad)}; "
                         f"valid: {sorted(_STAGES)} or 'all'")
    return force


def _skip(path, forced):
    """A unit may be skipped iff its artifact exists and it is not forced."""
    return Path(path).exists() and not forced


def _labels_match(path, adata):
    """True if a cached cell_labels.tsv has exactly this adata's cell index."""
    try:
        idx = pd.read_csv(path, sep='\t', index_col=0).index
    except Exception:
        return False
    return (len(idx) == adata.n_obs and
            list(idx.astype(str)) == list(adata.obs_names.astype(str)))


def _concat_tsvs(paths):
    """Read and concatenate non-empty TSVs; missing/empty files are skipped."""
    frames = []
    for p in sorted(paths):
        try:
            df = pd.read_csv(p, sep='\t')
        except Exception:
            continue
        if len(df):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_h5ad_atomic(adata, path):
    """Write an AnnData file via a same-directory temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    adata.write_h5ad(tmp)
    tmp.replace(path)
    return path


def _persist_cluster(adata, res, *, cdir, umap_key, cluster_col, size_rank_name):
    """Write one cluster's dissect outputs into ``cdir`` (= clusters/cN/)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cdir.mkdir(parents=True, exist_ok=True)
    parent = res['parent']

    pd.DataFrame(res['panel_rows'], columns=_PANEL_COLS).to_csv(
        cdir / 'panel.tsv', sep='\t', index=False)
    res['subcluster_labels'].to_frame().to_csv(
        cdir / 'subcluster_labels.tsv', sep='\t')
    for minor, deg_df in res['deg'].items():
        deg_df.to_csv(cdir / f"deg_{size_rank_name[minor]}.tsv",
                      sep='\t', index=False)
    for minor, qdf in res['qc_drift'].items():
        qdf.assign(subcluster=size_rank_name[minor]).to_csv(
            cdir / f"qc_drift_{size_rank_name[minor]}.tsv", sep='\t', index=False)
    comp_by_cat: dict = {}
    for (minor, cat), cdf in res['composition'].items():
        comp_by_cat.setdefault(cat, []).append(
            cdf.assign(subcluster=size_rank_name[minor]))
    for cat, frames in comp_by_cat.items():
        pd.concat(frames, ignore_index=True).to_csv(
            cdir / f"composition_{cat.replace('/', '_')}.tsv",
            sep='\t', index=False)

    # zoom UMAP — this parent's cells coloured by their c{parent}_j label
    mask = (adata.obs[cluster_col].astype(str) == str(parent)).values
    xy = adata.obsm[umap_key][mask]
    labels = res['subcluster_labels'].values

    def _rank(x):
        try:
            return int(str(x).split('_')[1])
        except (IndexError, ValueError):
            return 99

    cats = sorted(pd.unique(labels), key=_rank)
    palette = plt.get_cmap('tab10', max(10, len(cats)))
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for i, cat in enumerate(cats):
        m = labels == cat
        ax.scatter(xy[m, 0], xy[m, 1], s=3, c=[palette(i)],
                   label=f"{cat} (n={int(m.sum())})", linewidths=0)
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2'); ax.set_aspect('equal')
    ax.set_title(f"{cluster_col} c{parent}")
    ax.legend(fontsize=7, markerscale=2, frameon=False, loc='best')
    fig.savefig(cdir / 'umap_subcluster.png', bbox_inches='tight')
    plt.close(fig)


def _plot_global_umap(adata, *, cluster_col, umap_label_col, umap_key, path):
    """2-panel UMAP coloured by the original clustering and the UMAP-Leiden one.
    Each cluster gets a centroid text label *and* a colour-legend entry below the
    panel — the legend is the fallback when the on-plot labels get crowded."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    xy = adata.obsm[umap_key]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), dpi=130)
    for ax, col, title in [(axes[0], cluster_col, f'original {cluster_col}'),
                           (axes[1], umap_label_col, f'global {umap_label_col}')]:
        col_str = adata.obs[col].astype(str)
        cats = list(col_str.unique())
        cats.sort(key=lambda x: -int((col_str == x).sum()))
        palette = plt.get_cmap('tab20', max(20, len(cats)))
        for i, cat in enumerate(cats):
            m = (col_str == cat).values
            ax.scatter(xy[m, 0], xy[m, 1], s=1, c=[palette(i)],
                       linewidths=0, alpha=0.6)
        for i, cat in enumerate(cats):
            m = (col_str == cat).values
            if not m.any():
                continue
            cx, cy = float(np.median(xy[m, 0])), float(np.median(xy[m, 1]))
            ax.text(cx, cy, str(cat), fontsize=8, fontweight='bold',
                    ha='center', va='center', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='gray', lw=0.5, alpha=0.85))
        ax.set_title(title); ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
        ax.set_aspect('equal')
        handles = [Line2D([0], [0], marker='o', linestyle='', markersize=5,
                          color=palette(i)) for i in range(len(cats))]
        ax.legend(handles, [str(c) for c in cats], title=col,
                  loc='upper center', bbox_to_anchor=(0.5, -0.09),
                  ncol=min(8, max(1, len(cats))), fontsize=7, frameon=False,
                  handletextpad=0.3, columnspacing=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# _dissect_one bundles "dissect one cluster + persist it"; the heavy shared
# inputs (adata, crosstab, ...) are passed through this module-level slot
# rather than threaded through every call.
_DISSECT_CTX: dict | None = None


def _dissect_one(parent):
    """Dissect + persist one cluster, reading shared inputs from ``_DISSECT_CTX``."""
    ctx = _DISSECT_CTX
    assert ctx is not None, "_DISSECT_CTX must be set before _dissect_one is called"
    res = dissect_one_cluster(
        ctx['adata'], cluster_col=ctx['cluster_col'], parent=str(parent),
        umap_label_col=ctx['umap_label_col'],
        crosstab_row=ctx['crosstab'].loc[parent],
        size_rank_name=ctx['srn_by_parent'][parent],
        cat_cols=ctx['cat_cols'], qc_cols=ctx['qc_cols'],
        top_n_deg=ctx['top_n_deg'], deg_layer=ctx['deg_layer'],
        min_subcluster_size=ctx['min_subcluster_size'],
    )
    _persist_cluster(ctx['adata'], res, cdir=ctx['lay'].cluster_dir(parent),
                     umap_key=ctx['umap_key'], cluster_col=ctx['cluster_col'],
                     size_rank_name=ctx['srn_by_parent'][parent])
    return str(parent)


def run_dissect_pipeline(
    adata,
    *,
    cluster_col,
    output_dir,
    labeled_h5ad_path=None,
    umap_key='X_umap',
    cat_cols=('orig.ident', 'batch'),
    qc_cols=('percent.mt', 'nCount_RNA', 'nFeature_RNA', 'hybrid_score'),
    sample_col='orig.ident',
    resolution=0.5,
    target_k=None,
    target_tol=2,
    n_neighbors=30,
    min_subcluster_size=50,
    top_n_deg=50,
    top_n_canonical=50,
    deg_layer=None,
    force=(),
    n_jobs=8,
    random_state=0,
):
    """Run the full cleanup-diagnosis pipeline into ``<output_dir>/<cluster_col>/``.

    Idempotent: a unit is skipped when its primary output file already exists and
    the unit is not named in ``force`` (a subset of {'partition','dissect',
    'canonical','anatomy'}, or 'all'). Existing ``adata.obs['umap_cluster']`` and
    ``adata.obs['original_cluster_split']`` are overwritten in memory. Recomputed
    labels always go to ``cell_labels.tsv``; pass ``labeled_h5ad_path`` to also
    persist those overwritten obs columns to an h5ad file.

    Stages run serially; stage 3's Wilcoxon is gene-chunked to bound memory.
    ``n_jobs`` is currently unused (reserved).

    Returns ``{'root', 'crosstab', 'panel', 'partition_info', 'canonical_deg',
    'skipped'}``.
    """
    if umap_key not in adata.obsm:
        raise KeyError(f"umap_key '{umap_key}' not in adata.obsm "
                       f"(have: {list(adata.obsm.keys())})")
    if cluster_col not in adata.obs.columns:
        raise KeyError(f"cluster_col '{cluster_col}' not in adata.obs")
    force = _normalize_force(force)
    lay = _Layout(output_dir, cluster_col)
    lay.root.mkdir(parents=True, exist_ok=True)
    lay.clusters.mkdir(exist_ok=True)
    umap_label_col = 'umap_cluster'
    skipped: list = []

    # ---- STAGE 1: UMAP-Leiden partition --------------------------------
    part_force = 'partition' in force
    if _skip(lay.cell_labels, part_force) and _labels_match(lay.cell_labels, adata):
        labels_df = pd.read_csv(lay.cell_labels, sep='\t', index_col=0)
        adata.obs[umap_label_col] = pd.Categorical(
            labels_df['umap_cluster'].astype(str).values)
        adata.obs['original_cluster_split'] = pd.Categorical(
            labels_df['original_cluster_split'].astype(str).values)
        partition_info = {'final_resolution': None,
                          'n_clusters': int(adata.obs[umap_label_col].nunique()),
                          'history': [], 'reused': True}
        partition_skipped = True
        skipped.append('partition')
        print(f"[pipeline] partition reused from {lay.cell_labels}", flush=True)
    else:
        tk = target_k if target_k is not None else \
            adata.obs[cluster_col].astype(str).nunique()
        print(f"[pipeline] computing UMAP-Leiden partition (target_k={tk}) ...",
              flush=True)
        labels, partition_info = umap_leiden_partition(
            adata.obsm[umap_key], target_k=tk, resolution=resolution,
            n_neighbors=n_neighbors, tol=target_tol, random_state=random_state)
        labels.index = adata.obs_names
        adata.obs[umap_label_col] = pd.Categorical([f"u{x}" for x in labels.values])
        partition_info['reused'] = False
        partition_skipped = False

    # ---- crosstab + Cartesian-product naming (always — cheap) ----------
    crosstab = pd.crosstab(adata.obs[cluster_col].astype(str),
                           adata.obs[umap_label_col].astype(str))
    rank_map: dict = {}
    srn_by_parent: dict = {}
    for parent in crosstab.index:
        row = crosstab.loc[parent].sort_values(ascending=False)
        srn: dict = {}
        rank = 0
        for u in row.index:
            if int(row[u]) == 0:
                continue
            name = f"c{parent}_{rank}"
            rank_map[(parent, u)] = name
            srn[u] = name
            rank += 1
        srn_by_parent[parent] = srn

    if not partition_skipped:
        parent_arr = adata.obs[cluster_col].astype(str).values
        umap_arr = adata.obs[umap_label_col].astype(str).values
        split = np.array([rank_map.get((p, u), f"c{p}_?")
                          for p, u in zip(parent_arr, umap_arr)], dtype=object)
        adata.obs['original_cluster_split'] = pd.Categorical(split)
        pd.DataFrame(
            {'umap_cluster': adata.obs[umap_label_col].astype(str).values,
             'original_cluster_split': adata.obs['original_cluster_split'].astype(str).values},
            index=adata.obs_names,
        ).to_csv(lay.cell_labels, sep='\t')

    crosstab.to_csv(lay.crosstab, sep='\t')

    # ---- STAGE 2: per-cluster dissect (serial) ------------------------
    dissect_force = part_force or ('dissect' in force)
    todo = [p for p in crosstab.index
            if not _skip(lay.cluster_panel(p), dissect_force)]
    todo_set = set(todo)
    skipped += [f'dissect:c{p}' for p in crosstab.index if p not in todo_set]
    if todo:
        global _DISSECT_CTX
        _DISSECT_CTX = {
            'adata': adata, 'cluster_col': cluster_col,
            'umap_label_col': umap_label_col, 'crosstab': crosstab,
            'srn_by_parent': srn_by_parent, 'cat_cols': cat_cols,
            'qc_cols': qc_cols, 'top_n_deg': top_n_deg, 'deg_layer': deg_layer,
            'min_subcluster_size': min_subcluster_size, 'umap_key': umap_key,
            'lay': lay,
        }
        try:
            for p in todo:
                _dissect_one(p)
        finally:
            _DISSECT_CTX = None
    print(f"[pipeline] dissect: {len(todo)} clusters computed, "
          f"{len(crosstab.index) - len(todo)} reused", flush=True)

    # ---- global panel + qc_drift_all (reassembled from disk) -----------
    panel = _concat_tsvs(lay.clusters.glob('c*/panel.tsv'))
    if not len(panel):
        panel = pd.DataFrame(columns=_PANEL_COLS)
    panel.to_csv(lay.panel, sep='\t', index=False)
    qc_all = _concat_tsvs(lay.clusters.glob('c*/qc_drift_*.tsv'))
    if len(qc_all):
        qc_all.to_csv(lay.qc_drift_all, sep='\t', index=False)

    # ---- global UMAP compare plot (always redrawn — cheap) -------------
    _plot_global_umap(adata, cluster_col=cluster_col,
                      umap_label_col=umap_label_col, umap_key=umap_key,
                      path=lay.global_umap)

    # ---- STAGE 3: canonical-core markers -------------------------------
    canon_force = part_force or ('canonical' in force)
    if _skip(lay.canonical_deg, canon_force):
        canonical_deg = pd.read_csv(lay.canonical_deg, sep='\t')
        skipped.append('canonical')
        print(f"[pipeline] canonical markers reused from {lay.canonical_deg}",
              flush=True)
    else:
        lay.canonical.mkdir(exist_ok=True)
        dominant = {p: crosstab.loc[p].idxmax() for p in crosstab.index}
        print("[pipeline] computing canonical-core markers ...", flush=True)
        canon_res = canonical_marker_deg(
            adata, cluster_col=cluster_col, umap_label_col=umap_label_col,
            top_n_genes=top_n_canonical, deg_layer=deg_layer,
            output_dir=str(lay.canonical), dominant=dominant,
            wilcoxon_chunk_size=3000, wilcoxon_n_jobs=1,
        )
        canonical_deg = canon_res['deg']

    # ---- STAGE 4: per-cluster minor-anatomy heatmaps -------------------
    anat_force = part_force or ('dissect' in force) or ('anatomy' in force)
    if anat_force:
        parents_todo = [str(p) for p in crosstab.index]
    else:
        parents_todo = [str(p) for p in crosstab.index
                        if not lay.anatomy_png(p).exists()]
        skipped += [f'anatomy:c{p}' for p in crosstab.index
                    if str(p) not in parents_todo]
    if parents_todo:
        print(f"[pipeline] drawing minor-anatomy heatmaps for "
              f"{len(parents_todo)} clusters ...", flush=True)
        plot_minor_anatomy(
            adata, subcluster_col='original_cluster_split',
            canonical_deg_df=canonical_deg, clusters_dir=str(lay.clusters),
            qc_cols=qc_cols, sample_col=sample_col,
            top_n_canonical=5, top_n_minor=5,
            min_subcluster_size=min_subcluster_size, parents=parents_todo,
        )

    # ---- params.json ---------------------------------------------------
    lay.params.write_text(json.dumps({
        'cluster_col': cluster_col, 'umap_key': umap_key,
        'cat_cols': list(cat_cols), 'qc_cols': list(qc_cols),
        'sample_col': sample_col, 'resolution': resolution,
        'target_k': target_k, 'target_tol': target_tol,
        'n_neighbors': n_neighbors,
        'min_subcluster_size': min_subcluster_size, 'top_n_deg': top_n_deg,
        'top_n_canonical': top_n_canonical, 'deg_layer': deg_layer,
        'random_state': random_state, 'n_jobs': n_jobs, 'force': sorted(force),
        'partition_info': partition_info,
    }, default=str, indent=2))

    if labeled_h5ad_path is not None:
        written = _write_h5ad_atomic(adata, labeled_h5ad_path)
        print(f"[pipeline] wrote labeled h5ad to {written}", flush=True)

    return {'root': str(lay.root), 'crosstab': crosstab, 'panel': panel,
            'partition_info': partition_info, 'canonical_deg': canonical_deg,
            'skipped': skipped}
