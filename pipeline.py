"""standissect.pipeline — unified orchestrator for the cluster cleanup-diagnosis.

``run_dissect_pipeline`` runs partitioning, per-cluster evidence extraction,
diagnosis, canonical-core markers, and minor-profile heatmaps into one output
tree, with file-existence idempotency. The analysis primitives live in
``standissect.cluster`` and the interpretation layer lives in
``standissect.diagnosis``.

The per-cluster DEG stage (Stage 2) is parallelised across clusters using a
spawn ProcessPoolExecutor (via ``parallel.process_map``).  Each worker receives
a materialised AnnData copy of one parent cluster's cells so no view reference
to the full adata is serialised.  On any pool-level failure the stage falls
back to serial execution.  ``n_jobs`` controls the worker count (capped by
``os.cpu_count()`` and the number of clusters to process).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .cluster import (umap_leiden_partition, dissect_one_cluster,
                      canonical_marker_deg, plot_minor_profile)
from .parallel import thread_map, process_map
from .diagnosis import (
    DEFAULT_ARK_ENDPOINT,
    DEFAULT_ARK_MODEL,
    PROMPT_VERSION,
    build_minor_evidence,
    make_chat_client,
    make_diagnosis_engine,
    normalize_diagnosis_roles,
    safe_subcluster_name,
    write_diagnosis_artifacts,
)
from .annotate import (
    NAMING_PROMPT_VERSION,
    NARRATIVE_PROMPT_VERSION,
    NarrativeEngine,
    load_marker_sets,
    make_naming_engine,
    run_naming_stage,
    run_narrative_stage,
)

_STAGES = ('partition', 'dissect', 'diagnosis', 'canonical', 'naming', 'narrative', 'profile')

_PANEL_COLS = ['parent_cluster', 'subcluster', 'reference_subcluster',
               'minor_umap_label', 'main_umap_label', 'n_cells', 'frac_of_parent',
               'top5_up_genes', 'top5_down_genes', 'n_sig_genes',
               'top_sample_enriched', 'top_qc_drift', 'rule_baseline',
               'likely_cause', 'cause_detail', 'diagnosis_confidence',
               'diagnosis_rationale', 'llm_overrode_rule',
               'disposition_baseline', 'recommended_disposition',
               'disposition_overridden', 'disposition_reason', 'proposed_cell_type']

_DIAGNOSIS_COLS = ['parent_cluster', 'subcluster', 'reference_subcluster',
                   'minor_umap_label', 'main_umap_label', 'n_cells', 'frac_of_parent',
                   'rule_baseline', 'likely_cause', 'cause_detail',
                   'diagnosis_confidence', 'diagnosis_rationale',
                   'llm_overrode_rule', 'disposition_baseline',
                   'recommended_disposition', 'disposition_overridden',
                   'disposition_reason', 'proposed_cell_type']


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
    def discard_cells(self): return self.root / 'discard_cells.tsv'
    @property
    def proposed_cell_types(self): return self.root / 'proposed_cell_types.tsv'
    @property
    def qc_drift_all(self):  return self.root / 'qc_drift_all.tsv'
    @property
    def diagnosis_all(self): return self.root / 'diagnosis_all.tsv'
    @property
    def params(self):        return self.root / 'params.json'
    @property
    def global_umap(self):   return self.root / 'global_umap_compare.png'
    @property
    def canonical_deg(self): return self.canonical / 'deg_long.tsv'
    @property
    def core_names(self):    return self.root / 'core_names.tsv'
    @property
    def narratives(self):    return self.root / 'narratives.tsv'

    def cluster_dir(self, parent):    return self.clusters / f"c{parent}"
    def cluster_panel(self, parent):  return self.cluster_dir(parent) / 'panel.tsv'
    def profile_png(self, parent):    return self.cluster_dir(parent) / 'minor_profile.png'


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


def _read_tsv(path):
    """Read a TSV, returning an empty DataFrame for missing/invalid files."""
    try:
        return pd.read_csv(path, sep='\t')
    except Exception:
        return pd.DataFrame()


def _ordered_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure headline panel columns remain in a stable order."""
    out = df.copy()
    for col in _PANEL_COLS:
        if col not in out.columns:
            out[col] = None
    extras = [c for c in out.columns if c not in _PANEL_COLS]
    return out[_PANEL_COLS + extras]


_DISCARD_CELL_COLS = ['barcode', 'input_row_index', 'subcluster', 'parent_cluster',
                      'likely_cause', 'diagnosis_confidence', 'disposition_reason']


def _write_cell_dispositions(lay, panel, obs_names):
    """Join recommended_disposition onto cell_labels.tsv (per cell), and write
    discard_cells.tsv (DISCARD cells only), keyed by barcode (obs_name) with the
    0-based row position in the standissect-input adata (from obs_names)."""
    if not lay.cell_labels.exists():
        return
    labels = pd.read_csv(lay.cell_labels, sep='\t', index_col=0)
    sub = labels['original_cluster_split'].astype(str)
    if len(panel) and 'subcluster' in panel.columns:
        p = panel.copy()
        p['subcluster'] = p['subcluster'].astype(str)
        p = p.drop_duplicates('subcluster').set_index('subcluster')
    else:
        p = pd.DataFrame()

    def col(name):
        if len(p) and name in p.columns:
            return sub.map(p[name])
        return pd.Series([None] * len(labels), index=labels.index)

    labels['recommended_disposition'] = col('recommended_disposition').fillna('')
    labels.to_csv(lay.cell_labels, sep='\t')

    # input_row_index assumes unique obs_names; last-wins on duplicates.
    pos = {str(b): i for i, b in enumerate(obs_names)}
    mask = (labels['recommended_disposition'] == 'DISCARD').values
    bc = labels.index[mask]
    discard = pd.DataFrame({
        'barcode': bc,
        'input_row_index': [pos.get(str(b)) for b in bc],
        'subcluster': sub[mask].values,
        'parent_cluster': col('parent_cluster')[mask].values,
        'likely_cause': col('likely_cause')[mask].values,
        'diagnosis_confidence': col('diagnosis_confidence')[mask].values,
        'disposition_reason': col('disposition_reason')[mask].values,
    }, columns=_DISCARD_CELL_COLS)
    discard.to_csv(lay.discard_cells, sep='\t', index=False)


def _apply_discard(adata, panel, path):
    """Write a cleaned .h5ad with DISCARD cells removed (KEEP + UNCERTAIN kept).
    Cleaned obs gains recommended_disposition for provenance. Does not mutate adata."""
    disp_map = {}
    if len(panel) and 'subcluster' in panel.columns and 'recommended_disposition' in panel.columns:
        p = panel.drop_duplicates('subcluster')
        disp_map = dict(zip(p['subcluster'].astype(str), p['recommended_disposition'].astype(str)))
    per_cell = adata.obs['original_cluster_split'].astype(str).map(disp_map).fillna('')
    keep_mask = (per_cell != 'DISCARD').values
    cleaned = adata[keep_mask].copy()
    cleaned.obs['recommended_disposition'] = per_cell.values[keep_mask]
    written = _write_h5ad_atomic(cleaned, path)
    n_disc = int((~keep_mask).sum())
    print(f"[pipeline] apply-discard: removed {n_disc} DISCARD cells; "
          f"wrote {int(cleaned.n_obs)} kept cells to {written}", flush=True)
    return n_disc, int(cleaned.n_obs)


_PROPOSED_COLS = ['level', 'parent_cluster', 'subcluster', 'proposed_cell_type',
                  'confidence', 'rationale']


def _write_proposed_cell_types(lay, panel, core_names_df):
    """Collect LLM-proposed cell types: minor (panel.proposed_cell_type) +
    major (core_names.differs_from_original) into proposed_cell_types.tsv."""
    rows = []
    if len(panel) and 'proposed_cell_type' in panel.columns:
        # The str.lower() != 'nan' filter intentionally excludes the literal string 'nan'.
        m = panel[panel['proposed_cell_type'].notna()
                  & (panel['proposed_cell_type'].astype(str).str.strip() != '')
                  & (panel['proposed_cell_type'].astype(str).str.lower() != 'nan')]
        for _, r in m.iterrows():
            rows.append({
                'level': 'minor',
                'parent_cluster': r.get('parent_cluster'),
                'subcluster': r.get('subcluster'),
                'proposed_cell_type': r.get('proposed_cell_type'),
                'confidence': r.get('diagnosis_confidence'),
                'rationale': r.get('diagnosis_rationale'),
            })
    if (len(core_names_df) and 'differs_from_original' in core_names_df.columns
            and 'cell_type' in core_names_df.columns):
        differs = core_names_df['differs_from_original'].apply(
            lambda v: str(v).strip().lower() in ('true', '1'))
        named = (core_names_df['cell_type'].notna()
                 & (core_names_df['cell_type'].astype(str).str.strip() != '')
                 & (core_names_df['cell_type'].astype(str).str.strip().str.lower() != 'nan'))
        d = core_names_df[differs & named]
        for _, r in d.iterrows():
            rows.append({
                'level': 'major',
                'parent_cluster': r.get('parent_cluster'),
                'subcluster': r.get('core_subcluster'),
                'proposed_cell_type': r.get('cell_type'),
                'confidence': r.get('confidence'),
                'rationale': r.get('rationale'),
            })
    pd.DataFrame(rows, columns=_PROPOSED_COLS).to_csv(
        lay.proposed_cell_types, sep='\t', index=False)


def _as_tuple(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(v for v in value if v)


def _unique_cols(*groups):
    cols = []
    for group in groups:
        for col in _as_tuple(group):
            if col and col not in cols:
                cols.append(col)
    return tuple(cols)


def _resolve_metadata_roles(
    *,
    cat_cols,
    qc_cols,
    sample_col,
    batch_col,
    donor_col,
    library_col,
    condition_col,
    doublet_score_col,
    mito_col,
    feature_count_col,
    umi_count_col,
    extra_cat_cols,
    extra_qc_cols,
    diagnosis_roles,
):
    """Resolve role-specific metadata columns into evidence columns and roles."""
    role_source_cols = _unique_cols(sample_col, batch_col, donor_col, library_col)
    role_map = normalize_diagnosis_roles({
        'source_cols': role_source_cols,
        'doublet_score_col': doublet_score_col,
        'mitochondrial_col': mito_col,
        'feature_count_col': feature_count_col,
        'umi_count_col': umi_count_col,
        **(diagnosis_roles or {}),
    }, use_defaults=False)
    resolved_cat = _unique_cols(
        cat_cols,
        role_map.get('source_cols'),
        condition_col,
        extra_cat_cols,
    )
    resolved_qc = _unique_cols(
        qc_cols,
        role_map.get('doublet_score_col'),
        role_map.get('mitochondrial_col'),
        role_map.get('feature_count_col'),
        role_map.get('umi_count_col'),
        extra_qc_cols,
    )
    return resolved_cat, resolved_qc, role_map


def _decat(frame):
    """Cast all Categorical columns in a DataFrame to object dtype (in-place).

    pandas Categorical raises ``NotImplementedError`` in
    ``NDArrayBacked.__setstate__`` when an AnnData is unpickled in a spawn
    worker.  The dissect/persist code only touches obs/var columns via
    ``.astype(str)``, ``.unique()``, and ``var_names`` — never a ``.cat``
    accessor — so object dtype is behaviour-preserving.
    """
    for _col in frame.columns:
        if isinstance(frame[_col].dtype, pd.CategoricalDtype):
            frame[_col] = frame[_col].astype(object)


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

    _ordered_panel(pd.DataFrame(res['panel_rows'])).to_csv(
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
        reference_subcluster = f"c{parent}_0"
        comp_by_cat.setdefault(cat, []).append(
            cdf.assign(cat_col=cat, subcluster=size_rank_name[minor],
                       reference_subcluster=reference_subcluster))
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


def _composition_frames_for_subcluster(cdir, subcluster):
    """Load all composition drift rows for one subcluster."""
    frames = []
    for path in sorted(Path(cdir).glob('composition_*.tsv')):
        df = _read_tsv(path)
        if not len(df):
            continue
        if 'subcluster' in df.columns:
            df = df[df['subcluster'].astype(str) == str(subcluster)]
        if not len(df):
            continue
        if 'cat_col' not in df.columns:
            df = df.copy()
            df['cat_col'] = path.stem.replace('composition_', '')
        frames.append(df)
    return frames


def _split_subcluster_label(label):
    """Parse c{parent}_{rank}; parent may itself contain underscores."""
    body = str(label)
    if body.startswith('c'):
        body = body[1:]
    parent, sep, rank = body.rpartition('_')
    if sep and rank.isdigit():
        return parent, int(rank)
    return body, None


def _diagnostic_genes_from_deg(deg_df, *, max_each=8):
    """Genes used to compare this minor against every major core."""
    if deg_df is None or not len(deg_df):
        return []
    name_col = 'names' if 'names' in deg_df.columns else 'gene'
    if name_col not in deg_df.columns or 'logfoldchanges' not in deg_df.columns:
        return []
    up = (deg_df[deg_df['logfoldchanges'] > 0]
          .sort_values('scores' if 'scores' in deg_df.columns else 'logfoldchanges',
                       ascending=False)[name_col].head(max_each).tolist())
    down = (deg_df[deg_df['logfoldchanges'] < 0]
            .sort_values('scores' if 'scores' in deg_df.columns else 'logfoldchanges',
                         ascending=True)[name_col].head(max_each).tolist())
    genes = []
    for g in up + down:
        if g not in genes:
            genes.append(g)
    return genes


def _mean_expression_for_label(adata, *, subcluster_col, label, genes):
    """Mean expression vector for one subcluster label over ``genes``."""
    if not genes or subcluster_col not in adata.obs.columns:
        return None
    var_idx = pd.Index(adata.var_names).get_indexer(genes)
    keep = var_idx >= 0
    if not keep.any():
        return None
    var_idx = var_idx[keep]
    labels = adata.obs[subcluster_col].astype(str).values
    mask = labels == str(label)
    if not mask.any():
        return None
    X = adata.X[mask][:, var_idx]
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64).mean(axis=0)


def _pearson(a, b):
    if a is None or b is None or len(a) != len(b) or len(a) < 2:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.nanstd(a) <= 1e-12 or np.nanstd(b) <= 1e-12:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else None


def _major_core_comparisons(
    adata,
    *,
    subcluster_col,
    subcluster,
    reference_subcluster,
    deg_df,
    max_other_cores=10,
):
    """Compare one minor to its own major core and the nearest other major cores."""
    if adata is None or subcluster_col not in adata.obs.columns:
        return []
    genes = _diagnostic_genes_from_deg(deg_df)
    minor_mean = _mean_expression_for_label(
        adata, subcluster_col=subcluster_col, label=subcluster, genes=genes)
    if minor_mean is None:
        return []

    labels = pd.Series(adata.obs[subcluster_col].astype(str).values)
    core_labels = []
    for label in labels.unique():
        _parent, rank = _split_subcluster_label(label)
        if rank == 0:
            core_labels.append(str(label))
    rows = []
    for core in sorted(core_labels):
        core_mean = _mean_expression_for_label(
            adata, subcluster_col=subcluster_col, label=core, genes=genes)
        if core_mean is None:
            continue
        corr = _pearson(minor_mean, core_mean)
        delta = float(np.mean(np.abs(minor_mean - core_mean)))
        parent, _rank = _split_subcluster_label(core)
        rows.append({
            'core_subcluster': core,
            'core_parent_cluster': parent,
            'is_reference_core': core == str(reference_subcluster),
            'n_core_cells': int((labels == core).sum()),
            'n_genes': len(genes),
            'pearson_on_minor_deg_genes': corr,
            'mean_abs_delta_on_minor_deg_genes': delta,
            'genes_used': genes,
        })
    reference = [r for r in rows if r['is_reference_core']]
    others = [r for r in rows if not r['is_reference_core']]
    def _core_sort_key(row):
        corr = row['pearson_on_minor_deg_genes']
        return (
            corr is None,
            -(corr if corr is not None else -np.inf),
            row['mean_abs_delta_on_minor_deg_genes'],
        )

    others.sort(key=_core_sort_key)
    return reference + others[:max_other_cores]


def _diagnosis_output_path(cdir, subcluster):
    return Path(cdir) / f"diagnosis_{safe_subcluster_name(subcluster)}.output.json"


def _diagnosis_current(cdir, row, *, mode, model):
    """True if the saved diagnosis matches the requested mode/model."""
    path = _diagnosis_output_path(cdir, row['subcluster'])
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return False
    if data.get('prompt_version') != PROMPT_VERSION:
        return False
    if data.get('diagnosis_mode') != mode:
        return False
    if mode != 'rule' and data.get('model') != model:
        return False
    return True


def _cluster_diagnoses_current(cdir, *, mode, model):
    panel = _read_tsv(Path(cdir) / 'panel.tsv')
    if not len(panel):
        return True
    if 'subcluster' not in panel.columns:
        return False
    return all(_diagnosis_current(cdir, row, mode=mode, model=model)
               for _, row in panel.iterrows())


def _apply_diagnosis_to_cluster_panel(
    cdir,
    engine,
    *,
    adata=None,
    diagnosis_roles=None,
    llm_concurrency=1,
):
    """Diagnose every minor row in one cluster panel and rewrite the panel."""
    cdir = Path(cdir)
    panel_path = cdir / 'panel.tsv'
    panel = _read_tsv(panel_path)
    if not len(panel):
        _ordered_panel(panel).to_csv(panel_path, sep='\t', index=False)
        return panel

    def _diagnose_row(idx_row):
        _, row = idx_row
        row_dict = row.to_dict()
        subcluster = str(row_dict['subcluster'])
        reference_subcluster = str(
            row_dict.get('reference_subcluster')
            or f"c{row_dict.get('parent_cluster')}_0"
        )
        row_dict['reference_subcluster'] = reference_subcluster
        deg_df = _read_tsv(cdir / f"deg_{subcluster}.tsv")
        qc_df = _read_tsv(cdir / f"qc_drift_{subcluster}.tsv")
        comp = _composition_frames_for_subcluster(cdir, subcluster)
        core_comparisons = _major_core_comparisons(
            adata,
            subcluster_col='original_cluster_split',
            subcluster=subcluster,
            reference_subcluster=reference_subcluster,
            deg_df=deg_df,
        )
        evidence = build_minor_evidence(
            row_dict, deg_df=deg_df, qc_df=qc_df, composition_frames=comp,
            major_core_comparisons=core_comparisons,
            diagnosis_roles=diagnosis_roles)
        result = engine.diagnose(evidence)
        row_dict.update(result.to_panel_fields())
        write_diagnosis_artifacts(cdir, evidence, result)
        return row_dict

    rows = thread_map(_diagnose_row, list(panel.iterrows()), max_workers=llm_concurrency)

    diagnosed = _ordered_panel(pd.DataFrame(rows))
    diagnosed.to_csv(panel_path, sep='\t', index=False)
    return diagnosed


def _run_diagnosis_stage(
    lay,
    crosstab,
    engine,
    *,
    forced,
    adata=None,
    diagnosis_roles=None,
    llm_concurrency=1,
):
    """Run diagnosis where missing, stale, or explicitly forced."""
    mode = getattr(engine, 'mode', 'rule')
    model = getattr(engine, 'model', None)
    todo = [
        p for p in crosstab.index
        if forced or not _cluster_diagnoses_current(
            lay.cluster_dir(p), mode=mode, model=model)
    ]
    if todo:
        print(f"[pipeline] diagnosing minor causes for {len(todo)} clusters "
              f"(mode={mode}) ...", flush=True)
        for p in todo:
            _apply_diagnosis_to_cluster_panel(lay.cluster_dir(p), engine,
                                              adata=adata,
                                              diagnosis_roles=diagnosis_roles,
                                              llm_concurrency=llm_concurrency)
    print(f"[pipeline] diagnosis: {len(todo)} clusters computed, "
          f"{len(crosstab.index) - len(todo)} reused", flush=True)
    return [f'diagnosis:c{p}' for p in crosstab.index if p not in set(todo)]


def _dissect_one_subset(parent, subset, ctx):
    """Dissect + persist one cluster; pure function — no module global.

    ``subset`` is a materialised AnnData copy (not a view) of this parent's
    cells.  ``ctx`` is a plain dict of scalars/DataFrames/Paths (no full adata)
    and is therefore safe to pickle for spawn workers.
    """
    res = dissect_one_cluster(
        subset, cluster_col=ctx['cluster_col'], parent=str(parent),
        umap_label_col=ctx['umap_label_col'],
        crosstab_row=ctx['crosstab'].loc[parent],
        size_rank_name=ctx['srn_by_parent'][parent],
        cat_cols=ctx['cat_cols'], qc_cols=ctx['qc_cols'],
        top_n_deg=ctx['top_n_deg'], deg_layer=ctx['deg_layer'],
        min_subcluster_size=ctx['min_subcluster_size'],
    )
    _persist_cluster(subset, res, cdir=ctx['lay'].cluster_dir(parent),
                     umap_key=ctx['umap_key'], cluster_col=ctx['cluster_col'],
                     size_rank_name=ctx['srn_by_parent'][parent])
    return str(parent)


def _dissect_task(args):
    """Top-level picklable task wrapper for the spawn ProcessPool.

    Must be a module-level function (not a closure) so ``pickle`` can locate it
    by qualified name when serialising the callable to the worker process.
    Unpacks ``(parent, subset, ctx)`` → ``_dissect_one_subset``.
    """
    parent, subset, ctx = args
    return _dissect_one_subset(parent, subset, ctx)


def run_dissect_pipeline(
    adata,
    *,
    cluster_col,
    output_dir,
    labeled_h5ad_path=None,
    apply_discard_path=None,
    umap_key='X_umap',
    cat_cols=None,
    qc_cols=None,
    sample_col=None,
    batch_col=None,
    donor_col=None,
    library_col=None,
    condition_col=None,
    doublet_score_col=None,
    mito_col=None,
    feature_count_col=None,
    umi_count_col=None,
    extra_cat_cols=(),
    extra_qc_cols=(),
    diagnosis_roles=None,
    resolution=0.5,
    target_k=None,
    target_tol=2,
    n_neighbors=30,
    min_subcluster_size=50,
    top_n_deg=50,
    top_n_canonical=50,
    deg_layer=None,
    diagnosis_mode='llm',
    diagnosis_llm_client=None,
    diagnosis_ark_model=DEFAULT_ARK_MODEL,
    diagnosis_ark_endpoint=DEFAULT_ARK_ENDPOINT,
    diagnosis_ark_api_key=None,
    diagnosis_ark_api_key_env='ARK_API_KEY',
    diagnosis_timeout=120,
    diagnosis_fallback_to_rule=True,
    annotation_hint='',
    naming_markers=None,
    force=(),
    n_jobs=8,
    llm_concurrency=8,
    llm_retries=3,
    discard_confidence_threshold=0.5,
    random_state=0,
):
    """Run the full cleanup-diagnosis pipeline into ``<output_dir>/<cluster_col>/``.

    Idempotent: a unit is skipped when its primary output file already exists and
    the unit is not named in ``force`` (a subset of {'partition','dissect',
    'diagnosis','canonical','naming','narrative','profile'}, or 'all'). Existing
    ``adata.obs['umap_cluster']`` and ``adata.obs['original_cluster_split']`` are
    overwritten in memory. Recomputed labels always go to ``cell_labels.tsv``;
    pass ``labeled_h5ad_path`` to also persist those overwritten obs columns to
    an h5ad file.

    ``diagnosis_mode`` is ``'llm'`` by default. ``'llm'`` and ``'hybrid'`` use a
    chat client over compact per-minor evidence packets. If no
    ``diagnosis_llm_client`` is supplied, the built-in Ark client reads
    ``ARK_API_KEY`` and uses the configured Ark model/endpoint.

    Metadata is role-specific. ``sample_col``, ``batch_col``, ``donor_col`` and
    ``library_col`` are source-like categorical columns for sample/batch-driven
    diagnosis. ``condition_col`` and ``extra_cat_cols`` are composition evidence
    only. QC roles are controlled by ``doublet_score_col``, ``mito_col``,
    ``feature_count_col`` and ``umi_count_col``. ``cat_cols`` and ``qc_cols`` are
    retained as compatibility escape hatches for extra evidence columns.

    Stages run serially except the DEG dissect stage, which fans out across
    spawn workers — one worker per parent cluster, bounded by ``n_jobs`` and
    ``os.cpu_count()``.  Canonical-core Wilcoxon is gene-chunked to bound memory.

    Returns ``{'root', 'crosstab', 'panel', 'partition_info', 'canonical_deg',
    'skipped'}``.
    """
    if umap_key not in adata.obsm:
        raise KeyError(f"umap_key '{umap_key}' not in adata.obsm "
                       f"(have: {list(adata.obsm.keys())})")
    if cluster_col not in adata.obs.columns:
        raise KeyError(f"cluster_col '{cluster_col}' not in adata.obs")
    force = _normalize_force(force)
    resolved_cat_cols, resolved_qc_cols, resolved_roles = _resolve_metadata_roles(
        cat_cols=cat_cols,
        qc_cols=qc_cols,
        sample_col=sample_col,
        batch_col=batch_col,
        donor_col=donor_col,
        library_col=library_col,
        condition_col=condition_col,
        doublet_score_col=doublet_score_col,
        mito_col=mito_col,
        feature_count_col=feature_count_col,
        umi_count_col=umi_count_col,
        extra_cat_cols=extra_cat_cols,
        extra_qc_cols=extra_qc_cols,
        diagnosis_roles=diagnosis_roles,
    )
    # Build the chat client once and share it across diagnosis + naming +
    # narrative. A missing key (or rule mode) yields None -> graceful degrade.
    chat_client = make_chat_client(
        mode=diagnosis_mode,
        llm_client=diagnosis_llm_client,
        ark_api_key=diagnosis_ark_api_key,
        ark_api_key_env=diagnosis_ark_api_key_env,
        ark_model=diagnosis_ark_model,
        ark_endpoint=diagnosis_ark_endpoint,
        timeout=diagnosis_timeout,
    )
    effective_diagnosis_mode = diagnosis_mode
    if diagnosis_mode != 'rule' and chat_client is None:
        print(f"[pipeline] WARNING: diagnosis_mode={diagnosis_mode!r} requested but "
              f"no LLM client is available (set {diagnosis_ark_api_key_env} or pass "
              f"diagnosis_llm_client). Falling back to rule diagnosis; naming uses "
              f"local marker overlap; narrative is skipped.", flush=True)
        effective_diagnosis_mode = 'rule'
    diagnosis_engine = make_diagnosis_engine(
        mode=effective_diagnosis_mode,
        llm_client=chat_client,
        ark_api_key=diagnosis_ark_api_key,
        ark_api_key_env=diagnosis_ark_api_key_env,
        ark_model=diagnosis_ark_model,
        ark_endpoint=diagnosis_ark_endpoint,
        timeout=diagnosis_timeout,
        fallback_to_rule=diagnosis_fallback_to_rule,
        diagnosis_roles=resolved_roles,
        llm_retries=llm_retries,
        discard_confidence_threshold=discard_confidence_threshold,
    )
    resolved_markers = load_marker_sets(naming_markers)
    naming_engine = make_naming_engine(
        client=chat_client, markers=resolved_markers, fallback_to_local=True,
        llm_retries=llm_retries)
    narrative_engine = NarrativeEngine(chat_client, llm_retries=llm_retries) if chat_client is not None else None
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

    # ---- STAGE 2: per-cluster dissect (parallel across clusters) -------
    dissect_force = part_force or ('dissect' in force)
    todo = [p for p in crosstab.index
            if not _skip(lay.cluster_panel(p), dissect_force)]
    todo_set = set(todo)
    skipped += [f'dissect:c{p}' for p in crosstab.index if p not in todo_set]
    if todo:
        # ctx holds only picklable scalars/DataFrames/Paths — no full adata.
        ctx = {
            'cluster_col': cluster_col,
            'umap_label_col': umap_label_col, 'crosstab': crosstab,
            'srn_by_parent': srn_by_parent, 'cat_cols': resolved_cat_cols,
            'qc_cols': resolved_qc_cols, 'top_n_deg': top_n_deg, 'deg_layer': deg_layer,
            'min_subcluster_size': min_subcluster_size, 'umap_key': umap_key,
            'lay': lay,
        }
        # Materialise each subset with .copy() so the spawn worker receives a
        # standalone AnnData (not a view) — a view retains a reference to the
        # full parent adata, which would be serialised in its entirety.
        # Cast obs/var Categorical columns to object: pandas Categorical raises
        # NotImplementedError in NDArrayBacked.__setstate__ when unpickled in a
        # spawn worker, which would silently force the serial fallback. The
        # dissect/persist code reads obs via .astype(str)/.unique() and only
        # touches var through var_names (the index) — never a .cat. accessor —
        # so object dtype is behavior-preserving.
        subsets = {}
        for p in todo:
            subset = adata[adata.obs[cluster_col].astype(str) == str(p)].copy()
            _decat(subset.obs)
            _decat(subset.var)
            subsets[p] = subset
        deg_jobs = max(1, min(len(todo), os.cpu_count() or 1, n_jobs))
        try:
            process_map(_dissect_task, [(p, subsets[p], ctx) for p in todo],
                        max_workers=deg_jobs)
        except Exception as exc:
            print(f"[pipeline] dissect process pool failed ({exc}); "
                  f"recomputing serially", flush=True)
            for p in todo:
                _dissect_one_subset(p, subsets[p], ctx)
    print(f"[pipeline] dissect: {len(todo)} clusters computed, "
          f"{len(crosstab.index) - len(todo)} reused", flush=True)

    # ---- STAGE 3: per-minor diagnosis --------------------------------
    diagnosis_force = part_force or ('dissect' in force) or ('diagnosis' in force)
    skipped += _run_diagnosis_stage(lay, crosstab, diagnosis_engine,
                                    forced=diagnosis_force, adata=adata,
                                    diagnosis_roles=resolved_roles,
                                    llm_concurrency=llm_concurrency)

    # ---- global panel + qc_drift_all (reassembled from disk) -----------
    panel = _concat_tsvs(lay.clusters.glob('c*/panel.tsv'))
    if not len(panel):
        panel = pd.DataFrame(columns=_PANEL_COLS)
    panel = _ordered_panel(panel)
    panel.to_csv(lay.panel, sep='\t', index=False)
    diag_cols = [c for c in _DIAGNOSIS_COLS if c in panel.columns]
    panel[diag_cols].to_csv(lay.diagnosis_all, sep='\t', index=False)
    _write_cell_dispositions(lay, panel, adata.obs_names)
    if apply_discard_path is not None:
        _apply_discard(adata, panel, apply_discard_path)
    qc_all = _concat_tsvs(lay.clusters.glob('c*/qc_drift_*.tsv'))
    if len(qc_all):
        qc_all.to_csv(lay.qc_drift_all, sep='\t', index=False)

    # ---- global UMAP compare plot (always redrawn — cheap) -------------
    _plot_global_umap(adata, cluster_col=cluster_col,
                      umap_label_col=umap_label_col, umap_key=umap_key,
                      path=lay.global_umap)

    # ---- STAGE 4: canonical-core markers -------------------------------
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

    # ---- STAGE: core cell-type naming (always runs) -------------------
    parents = [str(p) for p in crosstab.index]
    core_sizes = {str(p): int(crosstab.loc[p].max()) for p in crosstab.index}
    naming_force = part_force or ('canonical' in force) or ('naming' in force)
    print(f"[pipeline] naming canonical cores "
          f"(source={'llm' if chat_client is not None else 'local'}) ...", flush=True)
    skipped += run_naming_stage(
        clusters_dir=lay.clusters, canonical_dir=lay.canonical,
        core_names_path=lay.core_names, parents=parents, engine=naming_engine,
        hint=annotation_hint, forced=naming_force, core_sizes=core_sizes,
        max_workers=llm_concurrency)
    core_names_df = _read_tsv(lay.core_names)
    _write_proposed_cell_types(lay, panel, core_names_df)

    # ---- STAGE: per-cluster narrative (LLM only) ----------------------
    narrative_force = naming_force or diagnosis_force or ('narrative' in force)
    if narrative_engine is not None:
        print("[pipeline] writing per-cluster narratives ...", flush=True)
        skipped += run_narrative_stage(
            clusters_dir=lay.clusters, core_names_path=lay.core_names,
            narratives_path=lay.narratives, parents=parents, engine=narrative_engine,
            hint=annotation_hint, forced=narrative_force,
            max_workers=llm_concurrency)
    else:
        skipped += [f'narrative:c{p}' for p in parents]
        print("[pipeline] narrative skipped (no LLM client)", flush=True)
    narratives_df = _read_tsv(lay.narratives)

    # ---- STAGE 5: per-cluster minor-profile heatmaps -------------------
    profile_force = part_force or ('dissect' in force) or ('profile' in force)
    if profile_force:
        parents_todo = [str(p) for p in crosstab.index]
    else:
        parents_todo = [str(p) for p in crosstab.index
                        if not lay.profile_png(p).exists()]
        skipped += [f'profile:c{p}' for p in crosstab.index
                    if str(p) not in parents_todo]
    if parents_todo:
        print(f"[pipeline] drawing minor-profile heatmaps for "
              f"{len(parents_todo)} clusters ...", flush=True)
        plot_minor_profile(
            adata, subcluster_col='original_cluster_split',
            canonical_deg_df=canonical_deg, clusters_dir=str(lay.clusters),
            qc_cols=resolved_qc_cols, sample_col=sample_col,
            top_n_canonical=5, top_n_minor=5,
            min_subcluster_size=min_subcluster_size, parents=parents_todo,
        )

    # ---- params.json ---------------------------------------------------
    lay.params.write_text(json.dumps({
        'cluster_col': cluster_col, 'umap_key': umap_key,
        'cat_cols': list(resolved_cat_cols), 'qc_cols': list(resolved_qc_cols),
        'sample_col': sample_col, 'batch_col': batch_col,
        'donor_col': donor_col, 'library_col': library_col,
        'condition_col': condition_col,
        'doublet_score_col': doublet_score_col,
        'mito_col': mito_col,
        'feature_count_col': feature_count_col,
        'umi_count_col': umi_count_col,
        'extra_cat_cols': list(_as_tuple(extra_cat_cols)),
        'extra_qc_cols': list(_as_tuple(extra_qc_cols)),
        'diagnosis_roles': resolved_roles,
        'resolution': resolution,
        'target_k': target_k, 'target_tol': target_tol,
        'n_neighbors': n_neighbors,
        'min_subcluster_size': min_subcluster_size, 'top_n_deg': top_n_deg,
        'top_n_canonical': top_n_canonical, 'deg_layer': deg_layer,
        'diagnosis_mode': diagnosis_mode,
        'effective_diagnosis_mode': effective_diagnosis_mode,
        'diagnosis_model': getattr(diagnosis_engine, 'model', None),
        'diagnosis_prompt_version': PROMPT_VERSION,
        'diagnosis_ark_endpoint': diagnosis_ark_endpoint,
        'diagnosis_fallback_to_rule': diagnosis_fallback_to_rule,
        'annotation_hint': annotation_hint,
        'annotation_model': getattr(chat_client, 'model', None),
        'naming_prompt_version': NAMING_PROMPT_VERSION,
        'narrative_prompt_version': NARRATIVE_PROMPT_VERSION,
        'naming_marker_types': sorted(resolved_markers),
        'random_state': random_state, 'n_jobs': n_jobs, 'force': sorted(force),
        'llm_concurrency': llm_concurrency,
        'llm_retries': llm_retries,
        'diagnosis_timeout': diagnosis_timeout,
        'discard_confidence_threshold': discard_confidence_threshold,
        'apply_discard_path': apply_discard_path,
        'partition_info': partition_info,
    }, default=str, indent=2))

    if labeled_h5ad_path is not None:
        written = _write_h5ad_atomic(adata, labeled_h5ad_path)
        print(f"[pipeline] wrote labeled h5ad to {written}", flush=True)

    return {'root': str(lay.root), 'crosstab': crosstab, 'panel': panel,
            'partition_info': partition_info, 'canonical_deg': canonical_deg,
            'core_names': core_names_df, 'narratives': narratives_df,
            'skipped': skipped}
