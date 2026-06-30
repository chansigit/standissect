"""standissect.cluster — analysis primitives for the cluster cleanup-diagnosis.

UMAP-Leiden partition, per-cluster dissection, DEG, canonical-core markers,
minor-profile heatmaps, vectorised Wilcoxon. Orchestration + the unified output
tree + idempotency live in ``standissect.pipeline``.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _split_ranked_label(label: str) -> tuple[str, int | None]:
    """Parse c{parent}_{rank} while allowing underscores in parent names."""
    body = str(label)
    if body.startswith('c'):
        body = body[1:]
    parent, sep, rank = body.rpartition('_')
    if sep and rank.isdigit():
        return parent, int(rank)
    return body, None


def _canonical_group_sort_key(group: str) -> tuple[int, int | str, int, str]:
    """Sort c{parent}_rank labels with numeric parents first, then string parents."""
    parent, rank = _split_ranked_label(group)
    try:
        return (0, int(parent), -1 if rank is None else rank, str(group))
    except ValueError:
        return (1, parent, -1 if rank is None else rank, str(group))


def _composition_drift(
    obs: pd.DataFrame,
    *,
    group_col: str,
    group: str,
    reference: str,
    cat_col: str,
) -> pd.DataFrame:
    """For one categorical ``cat_col``, compare distribution in ``group`` vs ``reference``.

    Returns long DataFrame with one row per category present in either group/reference.
    log2 OR uses Haldane-Anscombe correction (+0.5). Per-category p is from a 2x2
    Fisher exact (group vs reference, this category vs others). BH FDR across categories.
    """
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests
    sj = obs.loc[obs[group_col] == group, cat_col]
    s0 = obs.loc[obs[group_col] == reference, cat_col]
    cats = pd.Index(pd.unique(pd.concat([sj, s0], ignore_index=True).dropna()))
    Nj, N0 = len(sj), len(s0)
    rows = []
    for c in cats:
        a = int((sj == c).sum())
        b = Nj - a
        cc = int((s0 == c).sum())
        d = N0 - cc
        oddsr_log2 = float(np.log2(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (cc + 0.5))))
        try:
            _, p = fisher_exact([[a, b], [cc, d]])
        except ValueError:
            p = 1.0
        rows.append({
            'category':  c,
            'n_in_j':    a,
            'frac_in_j': a / Nj if Nj else 0.0,
            'frac_in_0': cc / N0 if N0 else 0.0,
            'log2_OR':   oddsr_log2,
            'p':         p,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df['padj'] = multipletests(df['p'].fillna(1.0).values, method='fdr_bh')[1]
    else:
        df['padj'] = pd.Series(dtype=float)
    return df


def _qc_drift(
    obs: pd.DataFrame,
    *,
    group_col: str,
    group: str,
    reference: str,
    qc_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Mann-Whitney drift for each continuous QC column. BH FDR across qc_cols."""
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    mj = obs[group_col] == group
    m0 = obs[group_col] == reference
    rows = []
    for c in qc_cols:
        if c not in obs.columns:
            continue
        a = obs.loc[mj, c].dropna().to_numpy()
        b = obs.loc[m0, c].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        mean_j, mean_0 = float(a.mean()), float(b.mean())
        delta = mean_j - mean_0
        rel   = delta / (abs(mean_0) + 1e-9)
        try:
            _, p = mannwhitneyu(a, b, alternative='two-sided')
        except ValueError:
            p = 1.0
        rows.append({
            'qc_col':         c,
            'mean_j':         mean_j,
            'mean_0':         mean_0,
            'delta':          delta,
            'relative_delta': rel,
            'p':              float(p),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df['padj'] = multipletests(df['p'].fillna(1.0).values, method='fdr_bh')[1]
    else:
        df['padj'] = pd.Series(dtype=float)
    return df


def umap_leiden_partition(
    umap_xy: np.ndarray,
    *,
    target_k: int | None = None,
    resolution: float = 0.5,
    n_neighbors: int = 30,
    tol: int = 2,
    max_iter: int = 12,
    random_state: int = 0,
) -> tuple[pd.Series, dict]:
    """kNN+Leiden on UMAP-2D for all cells. Returns (labels Series, info dict).

    If ``target_k`` is given, binary-search ``resolution`` so the result has
    ``target_k ± tol`` clusters (up to ``max_iter`` iterations). Otherwise use
    ``resolution`` directly.
    """
    from sklearn.neighbors import kneighbors_graph
    import igraph as ig
    import leidenalg
    k = min(n_neighbors, max(2, len(umap_xy) - 1))
    knn = kneighbors_graph(umap_xy, n_neighbors=k, mode='connectivity', include_self=False)
    knn = knn.maximum(knn.T)
    sources, targets = knn.nonzero()
    edges = list({(int(min(s, t)), int(max(s, t))) for s, t in zip(sources, targets) if s != t})
    g = ig.Graph(n=len(umap_xy), edges=edges, directed=False)

    def _run(res):
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=res, seed=random_state,
        )
        return np.array(part.membership)

    history = []
    if target_k is None:
        labels_raw = _run(resolution)
        final_res = resolution
    else:
        lo, hi = 1e-3, 10.0
        cur_res = resolution
        labels_raw = _run(cur_res)
        n = len(np.unique(labels_raw))
        history.append((cur_res, n))
        for _ in range(max_iter):
            if abs(n - target_k) <= tol:
                break
            if n < target_k:
                lo = cur_res
                cur_res = cur_res * 1.5 if hi == 10.0 else (cur_res + hi) / 2
            else:
                hi = cur_res
                cur_res = cur_res * 0.5 if lo == 1e-3 else (cur_res + lo) / 2
            labels_raw = _run(cur_res)
            n = len(np.unique(labels_raw))
            history.append((cur_res, n))
        final_res = cur_res

    # Size-rank rename so label '0' is largest etc.
    ranked = pd.Series(labels_raw).value_counts()
    remap = {orig: rank for rank, (orig, _) in enumerate(ranked.items())}
    labels = pd.Series([remap[x] for x in labels_raw])
    info = {
        'final_resolution': float(final_res),
        'n_clusters': int(labels.nunique()),
        'history': history,
    }
    return labels, info


def wilcoxon_vs_reference(
    X,
    group_labels,
    *,
    group: str,
    reference: str,
    gene_names,
    n_genes: int = 50,
    chunk_size: int = 2000,
    apply_logfoldchanges_expm1: bool = True,
) -> pd.DataFrame:
    """Tie-corrected 2-group Mann-Whitney: ``group`` vs ``reference`` (sparse, no
    densify). ``chunk_size`` accepted for compatibility but unused. Returns the
    top ``n_genes`` by score desc: names / logfoldchanges / pvals / pvals_adj /
    scores / direction.
    """
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    group_labels = np.asarray(group_labels)
    sel = (group_labels == group) | (group_labels == reference)
    stats = _wilcoxon_sparse_stats(X[sel], group_labels[sel], [group, reference])
    z, mean_in, mean_out = stats[group]
    with np.errstate(invalid='ignore'):
        p = 2.0 * norm.sf(np.abs(z))
    p = np.where(np.isnan(z), 1.0, p)
    padj = multipletests(p, method='fdr_bh')[1]
    if apply_logfoldchanges_expm1:
        lfc = np.log2((np.expm1(mean_in) + 1e-9) / (np.expm1(mean_out) + 1e-9))
    else:
        lfc = mean_in - mean_out
    df = pd.DataFrame({
        'names':          list(gene_names),
        'logfoldchanges': lfc,
        'pvals':          p,
        'pvals_adj':      padj,
        'scores':         np.nan_to_num(z, nan=0.0),
    })
    df['direction'] = np.where(df['logfoldchanges'] > 0, 'up', 'down')
    return (df.sort_values('scores', ascending=False)
              .head(n_genes).reset_index(drop=True))


def dissect_one_cluster(
    adata,
    *,
    cluster_col: str,
    parent: str,
    umap_label_col: str,
    crosstab_row: pd.Series,
    size_rank_name: dict,
    cat_cols,
    qc_cols,
    top_n_deg: int = 50,
    deg_layer: str | None = None,
    min_subcluster_size: int = 50,
) -> dict:
    """Dissect one parent cluster on its global UMAP-Leiden fragments.

    ``crosstab_row`` is the crosstab row for ``parent`` (umap_label -> cell count).
    ``size_rank_name`` maps every umap_label present in the parent to its
    Cartesian-product name ``c{parent}_{j}`` (j=0 is the main / largest fragment).

    Each off-main fragment with >= ``min_subcluster_size`` cells is a "minor" and
    gets DEG vs the main (vectorised Mann-Whitney), composition + QC drift, and a
    compact evidence row. Final diagnosis is handled by ``standissect.diagnosis``.

    Returns a dict. The monolithic case (no minor) comes back with empty
    ``panel_rows`` / ``deg`` / ``qc_drift`` / ``composition``; ``subcluster_labels``
    is always produced so the parent can still be plotted and labelled.
    """
    parent = str(parent)
    row = crosstab_row.sort_values(ascending=False)
    reference_subcluster = f"c{parent}_0"
    main_label = next(u for u, name in size_rank_name.items()
                      if name == reference_subcluster)
    # ``row`` is the full crosstab row, so it also lists fragments with 0 cells
    # in this parent (they belong to other parents and are absent from
    # ``size_rank_name``). Guard on membership so they are never treated as
    # minors — otherwise ``--min-subcluster-size 0`` lets a 0-count foreign
    # label through and ``size_rank_name[minor]`` raises KeyError.
    minors = [u for u in row.index
              if u != main_label and u in size_rank_name
              and int(row[u]) >= min_subcluster_size]

    mask_parent = (adata.obs[cluster_col].astype(str) == parent).values
    parent_umap = adata.obs[umap_label_col].astype(str).values[mask_parent]
    subcluster_labels = pd.Series(
        [size_rank_name.get(u, f"c{parent}_?") for u in parent_umap],
        index=adata.obs_names[mask_parent], name='subcluster',
    )

    panel_rows: list = []
    deg: dict = {}
    qc_drift: dict = {}
    composition: dict = {}

    if minors:
        n_parent = int(mask_parent.sum())
        obs_parent = adata.obs.loc[mask_parent].copy()
        mm = np.where(obs_parent[umap_label_col].astype(str).values == main_label,
                      'main', obs_parent[umap_label_col].astype(str).values)
        obs_parent['__main_minor'] = pd.Categorical(mm)
        # expression matrix for DEG — the parent's cells, no per-minor copy
        if deg_layer is None:
            X_deg = adata.X[mask_parent]
        elif deg_layer == 'counts_recovered':
            import scanpy as sc
            import anndata as ad
            tmp = ad.AnnData(adata.layers['counts_recovered'][mask_parent].copy())
            sc.pp.normalize_total(tmp, target_sum=1e4)
            sc.pp.log1p(tmp)
            X_deg = tmp.X
        else:
            raise ValueError(f"unsupported deg_layer: {deg_layer!r}")
        gene_names = list(adata.var_names)

        for minor in minors:
            deg_df = wilcoxon_vs_reference(
                X_deg, mm, group=minor, reference='main',
                gene_names=gene_names, n_genes=top_n_deg,
            )
            n_sig = int(((deg_df['pvals_adj'] < 0.05) &
                         (deg_df['logfoldchanges'].abs() > 0.5)).sum())
            deg[minor] = deg_df

            top_composition = None
            for c in cat_cols:
                if c not in obs_parent.columns:
                    continue
                cdf = _composition_drift(obs_parent, group_col='__main_minor',
                                         group=minor, reference='main', cat_col=c)
                cdf.insert(0, 'parent', parent)
                cdf.insert(1, 'minor_umap', minor)
                composition[(minor, c)] = cdf
                sig = cdf[(cdf['padj'] < 0.05) & (cdf['log2_OR'] > 0)]
                if not sig.empty:
                    t = sig.sort_values('log2_OR', ascending=False).iloc[0]
                    candidate = {'cat_col': c, 'category': t['category'],
                                 'log2_OR': float(t['log2_OR']),
                                 'padj': float(t['padj'])}
                    if (top_composition is None or
                            candidate['log2_OR'] > top_composition['log2_OR']):
                        top_composition = candidate

            qdf = _qc_drift(obs_parent, group_col='__main_minor', group=minor,
                            reference='main', qc_cols=tuple(qc_cols))
            qdf.insert(0, 'parent', parent)
            qdf.insert(1, 'minor_umap', minor)
            qdf.insert(2, 'reference_subcluster', reference_subcluster)
            qc_drift[minor] = qdf
            top_qc = None
            if not qdf.empty:
                sig = qdf[qdf['padj'] < 0.05]
                if not sig.empty:
                    t = (sig.assign(absrel=sig['relative_delta'].abs())
                            .sort_values('absrel', ascending=False).iloc[0])
                    top_qc = {'qc_col': t['qc_col'], 'delta': float(t['delta']),
                              'relative_delta': float(t['relative_delta']),
                              'padj': float(t['padj'])}

            ups = deg_df[deg_df['direction'] == 'up'].sort_values(
                'scores', ascending=False)['names'].head(5).tolist()
            dns = deg_df[deg_df['direction'] == 'down'].sort_values(
                'scores')['names'].head(5).tolist()
            n_in_minor = int(row[minor])
            panel_rows.append({
                'parent_cluster':      parent,
                'subcluster':          size_rank_name[minor],
                'reference_subcluster': reference_subcluster,
                'minor_umap_label':    minor,
                'main_umap_label':     main_label,
                'n_cells':             n_in_minor,
                'frac_of_parent':      float(n_in_minor / n_parent),
                'top5_up_genes':       ','.join(ups),
                'top5_down_genes':     ','.join(dns),
                'n_sig_genes':         n_sig,
                'top_sample_enriched': (
                    f"{top_composition['cat_col']}={top_composition['category']} "
                    f"(log2OR={top_composition['log2_OR']:.2f}, "
                    f"q={top_composition['padj']:.1e})"
                    if top_composition else None),
                'top_qc_drift': (
                    f"{top_qc['qc_col']} (Δ={top_qc['delta']:+.2f}, "
                    f"q={top_qc['padj']:.1e})" if top_qc else None),
                'rule_baseline':        None,
                'likely_cause':         None,
                'cause_detail':         None,
                'diagnosis_confidence': None,
                'diagnosis_rationale':  None,
                'llm_overrode_rule':    False,
            })

    return {
        'parent':            parent,
        'main_umap_label':   main_label,
        'minors':            minors,
        'subcluster_labels': subcluster_labels,
        'panel_rows':        panel_rows,
        'deg':               deg,
        'qc_drift':          qc_drift,
        'composition':       composition,
    }


# =============================================================================
#  Canonical-core marker DEG  (one-vs-rest on c{N}_0 across all parents)
# =============================================================================

def canonical_marker_deg(
    adata,
    *,
    cluster_col: str,
    umap_label_col: str = 'umap_cluster',
    top_n_genes: int = 50,
    deg_layer: str | None = None,
    output_dir: str | None = None,
    dominant: dict | None = None,
    wilcoxon_chunk_size: int | None = None,
    wilcoxon_n_jobs: int = 1,
) -> dict:
    """For each parent cluster c_N, take its canonical core c{N}_0 (= cells whose
    UmapLeiden label is the dominant one inside c_N) and run wilcoxon one-vs-rest
    across all canonical cores.

    ``dominant`` maps parent -> dominant umap_label; if None it is derived from a
    fresh crosstab. ``output_dir``, when given, is the exact directory written to.

    Returns dict with 'deg', 'core_mask', 'sub_adata', 'fig_path'.
    """
    import scanpy as sc
    if dominant is None:
        ct = pd.crosstab(
            adata.obs[cluster_col].astype(str),
            adata.obs[umap_label_col].astype(str),
        )
        dominant = {parent: ct.loc[parent].idxmax() for parent in ct.index}

    parent_arr = adata.obs[cluster_col].astype(str).values
    umap_arr   = adata.obs[umap_label_col].astype(str).values
    core_mask  = np.array([umap_arr[i] == dominant[parent_arr[i]] for i in range(adata.n_obs)])
    print(f"canonical core cells: {core_mask.sum()} / {adata.n_obs} "
          f"({100*core_mask.sum()/adata.n_obs:.1f}%)")

    sub = adata[core_mask].copy()
    sub.obs['canonical_group'] = pd.Categorical(
        sub.obs[cluster_col].astype(str).apply(lambda p: f"c{p}_0")
    )
    if deg_layer == 'counts_recovered':
        if 'counts_recovered' not in sub.layers:
            raise KeyError("deg_layer='counts_recovered' requested but layer absent")
        sub.X = sub.layers['counts_recovered'].copy()
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)

    full = wilcoxon_one_vs_rest(
        sub.X, sub.obs['canonical_group'].astype(str).values,
        gene_names=list(sub.var_names),
        chunk_size=wilcoxon_chunk_size, n_jobs=wilcoxon_n_jobs,
        apply_logfoldchanges_expm1=True,
    )
    full['_abs_score'] = full['scores'].abs()
    full = full.sort_values(['group', '_abs_score'], ascending=[True, False])
    full['rank'] = full.groupby('group').cumcount()
    full = full[full['rank'] < top_n_genes].drop(columns=['_abs_score'])
    deg_long = full[['group', 'rank', 'gene', 'logfoldchanges',
                     'pvals', 'pvals_adj', 'scores']].reset_index(drop=True)
    groups = sorted(deg_long['group'].unique())

    fig_path = None
    if output_dir is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.cluster.hierarchy import linkage
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        deg_long.to_csv(base / 'deg_long.tsv', sep='\t', index=False)
        for g in groups:
            deg_long[deg_long['group'] == g].to_csv(base / f"markers_{g}.tsv",
                                                    sep='\t', index=False)
        top_per = (deg_long[deg_long['logfoldchanges'] > 0]
                   .sort_values(['group', 'scores'], ascending=[True, False])
                   .groupby('group').head(5))
        markers = top_per['gene'].drop_duplicates().tolist()
        groups_sorted = sorted(groups, key=_canonical_group_sort_key)
        mat = np.zeros((len(markers), len(groups_sorted)))
        for j, g in enumerate(groups_sorted):
            cells = (sub.obs['canonical_group'] == g).values
            X_block = sub[cells, markers].X
            if hasattr(X_block, 'toarray'):
                X_block = X_block.toarray()
            mat[:, j] = np.asarray(X_block).mean(axis=0)
        row_std = mat.std(axis=1, keepdims=True)
        mat_z = (mat - mat.mean(axis=1, keepdims=True)) / (row_std + 1e-9)
        df_mat = pd.DataFrame(mat_z, index=markers, columns=groups_sorted)
        nr, nc = df_mat.shape
        row_link = (linkage(df_mat.values, method='average', metric='euclidean',
                            optimal_ordering=True) if nr >= 2 else None)
        col_link = (linkage(df_mat.values.T, method='average', metric='euclidean',
                            optimal_ordering=True) if nc >= 2 else None)
        g = sns.clustermap(
            df_mat,
            cmap='RdBu_r', center=0, vmin=-2, vmax=2,
            row_linkage=row_link, col_linkage=col_link,
            row_cluster=(row_link is not None), col_cluster=(col_link is not None),
            figsize=(max(6, 0.55 * len(groups_sorted)),
                     max(8, 0.18 * len(markers))),
            xticklabels=True, yticklabels=True,
            cbar_kws={'label': 'z-score'},
            dendrogram_ratio=(0.12, 0.10),
        )
        g.ax_heatmap.set_xlabel('canonical core')
        g.ax_heatmap.set_ylabel('top markers')
        g.fig.suptitle(f'canonical-core markers — {cluster_col}', y=1.02)
        for tick in g.ax_heatmap.get_xticklabels(): tick.set_rotation(90); tick.set_fontsize(8)
        for tick in g.ax_heatmap.get_yticklabels(): tick.set_fontsize(6)
        fig_path = str(base / 'heatmap_top_markers.png')
        g.savefig(fig_path, bbox_inches='tight'); plt.close(g.fig)
        (base / 'top5_per_group.tsv').write_text(
            top_per[['group', 'rank', 'gene', 'logfoldchanges', 'pvals_adj', 'scores']]
            .to_csv(sep='\t', index=False)
        )

    return {'deg': deg_long, 'core_mask': pd.Series(core_mask, index=adata.obs_names),
            'sub_adata': sub, 'fig_path': fig_path}


# =============================================================================
#  Minor-profile heatmap (per-cluster)
# =============================================================================

def _select_gene_blocks(
    adata,
    *,
    canonical_deg_df: pd.DataFrame,
    clusters_dir,
    parent,
    minor_subcluster_names: list,
    top_n_canonical: int,
    top_n_minor: int,
) -> tuple[list, list]:
    """Return (canonical_genes, minor_genes) — two ordered lists of gene names
    present in adata.var_names. Minor block excludes anything in canonical block.
    Minor DEG is read from ``clusters_dir/c{parent}/``; minors without a DEG file
    (e.g. tiny fragments) simply contribute no minor genes.
    """
    var_set = set(adata.var_names)

    canon_order = sorted(canonical_deg_df['group'].unique(), key=_canonical_group_sort_key)
    canon = []
    for g in canon_order:
        sub_df = (canonical_deg_df[(canonical_deg_df['group'] == g) &
                                    (canonical_deg_df['logfoldchanges'] > 0)]
                  .sort_values('scores', ascending=False)
                  .head(top_n_canonical))
        for gene in sub_df['gene']:
            if gene in var_set and gene not in canon:
                canon.append(gene)

    canon_set = set(canon)
    minor = []
    base = Path(clusters_dir) / f"c{parent}"
    for sub_name in minor_subcluster_names:
        path = base / f"deg_{sub_name}.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep='\t')
        up = df[df['logfoldchanges'] > 0].sort_values('scores', ascending=False).head(top_n_minor)
        for gene in up['names'] if 'names' in up.columns else up['gene']:
            if gene in var_set and gene not in canon_set and gene not in minor:
                minor.append(gene)
    return canon, minor


def _build_expression_matrix(
    adata,
    *,
    subcluster_col: str,
    genes: list,
    columns: list,
) -> pd.DataFrame:
    """columns is a list of (name, role) tuples — role in {'core','minor'}.
    Returns DataFrame indexed by genes, columns named by the tuple's name field
    (mean expression over cells with subcluster_col == name). Empty column → NaN.
    """
    gene_idx = pd.Index(adata.var_names).get_indexer(genes)
    if (gene_idx < 0).any():
        missing = [g for g, i in zip(genes, gene_idx) if i < 0]
        raise KeyError(f"genes not in adata.var_names: {missing}")
    out = pd.DataFrame(np.full((len(genes), len(columns)), np.nan, dtype=float),
                       index=genes, columns=[c[0] for c in columns])
    sub_arr = adata.obs[subcluster_col].astype(str).values
    for col_name, _role in columns:
        mask = sub_arr == col_name
        if not mask.any():
            continue
        X_block = adata.X[mask][:, gene_idx]
        if hasattr(X_block, 'toarray'):
            X_block = X_block.toarray()
        out[col_name] = np.asarray(X_block).mean(axis=0)
    return out


def _build_qc_and_sample_matrices(
    adata,
    *,
    subcluster_col: str,
    columns: list,
    qc_cols: tuple,
    sample_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (qc_matrix, sample_matrix): qc_matrix rows = qc_cols; sample_matrix
    rows = samples, each column a fraction summing to 1. Empty if sample_col absent.
    """
    sub_arr = adata.obs[subcluster_col].astype(str).values
    col_names = [c[0] for c in columns]

    qc_rows = []
    for q in qc_cols:
        if q not in adata.obs.columns:
            continue
        row = []
        for cn in col_names:
            mask = sub_arr == cn
            row.append(float(adata.obs.loc[mask, q].mean()) if mask.any() else np.nan)
        qc_rows.append(pd.Series(row, index=col_names, name=q))
    qc_mat = pd.DataFrame(qc_rows) if qc_rows else pd.DataFrame(columns=col_names)

    if sample_col is None or sample_col not in adata.obs.columns:
        return qc_mat, pd.DataFrame(columns=col_names)
    samples = sorted(adata.obs[sample_col].astype(str).unique())
    sm = pd.DataFrame(np.zeros((len(samples), len(col_names))),
                      index=samples, columns=col_names)
    samp_arr = adata.obs[sample_col].astype(str).values
    for cn in col_names:
        mask = sub_arr == cn
        n = int(mask.sum())
        if n == 0:
            continue
        for s in samples:
            sm.loc[s, cn] = float(((samp_arr == s) & mask).sum() / n)
    return qc_mat, sm


def _cluster_columns(mat: pd.DataFrame, method: str = 'average', metric: str = 'euclidean'):
    """Return (leaf_order, linkage) for columns of mat, with optimal leaf ordering.
    NaN columns go to the end in original order, not clustered.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    arr = mat.values.T
    finite_mask = ~np.isnan(arr).any(axis=1)
    if finite_mask.sum() < 2:
        return list(range(arr.shape[0])), None
    finite_idx = np.where(finite_mask)[0]
    Z = linkage(arr[finite_idx], method=method, metric=metric, optimal_ordering=True)
    leaf = leaves_list(Z)
    order = list(finite_idx[leaf])
    order += [i for i in range(arr.shape[0]) if not finite_mask[i]]
    return order, Z


def _cluster_rows(mat: pd.DataFrame, method: str = 'average', metric: str = 'euclidean'):
    """Return (leaf_order, linkage) for rows of mat, with optimal leaf ordering."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    arr = mat.values
    finite_mask = ~np.isnan(arr).any(axis=1)
    if finite_mask.sum() < 2:
        return list(range(arr.shape[0])), None
    finite_idx = np.where(finite_mask)[0]
    Z = linkage(arr[finite_idx], method=method, metric=metric, optimal_ordering=True)
    leaf = leaves_list(Z)
    order = list(finite_idx[leaf]) + [i for i in range(arr.shape[0]) if not finite_mask[i]]
    return order, Z


def plot_minor_profile(
    adata,
    *,
    subcluster_col: str,
    canonical_deg_df: pd.DataFrame,
    clusters_dir,
    qc_cols: tuple = (),
    sample_col=None,
    top_n_canonical: int = 5,
    top_n_minor: int = 5,
    min_subcluster_size: int = 50,
    parents=None,
) -> dict:
    """Per-parent minor-profile heatmap.

    One merged gene block (canonical-core + minor-specific markers, optimal leaf
    ordering). The canonical cores ``c?_0`` and this parent's minors ``c?_i`` are
    drawn as two genuinely separate panels with a white gap; every minor fragment
    is shown regardless of size. QC + sample tracks sit below. Colorbars are on
    the left so they never overlap the gene/feature names (right side). Minor
    column names are bold.

    Reads per-minor DEG from ``clusters_dir/c{parent}/`` and writes each parent's
    heatmap to ``clusters_dir/c{parent}/minor_profile.png``.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.cluster.hierarchy import dendrogram

    if subcluster_col not in adata.obs.columns:
        raise KeyError(f"subcluster_col '{subcluster_col}' not in adata.obs")

    sub_arr = adata.obs[subcluster_col].astype(str)
    canon_groups_all = sorted(canonical_deg_df['group'].unique(),
                              key=_canonical_group_sort_key)
    canon_groups = [g for g in canon_groups_all if (sub_arr == g).any()]
    missing_canon = set(canon_groups_all) - set(canon_groups)
    if missing_canon:
        warnings.warn(
            f"canonical_deg_df has groups with no cells under subcluster_col "
            f"'{subcluster_col}': {sorted(missing_canon)}; dropping from heatmap"
        )
    sizes = sub_arr.value_counts()
    # Every parent with a c{N}_0 label; minors = ALL non-_0 fragments (any size).
    minor_by_parent: dict = {}
    for g in canon_groups:
        parent, _ = _split_ranked_label(g)
        minor_by_parent.setdefault(parent, [])
    for label in sizes.index:
        if label in canon_groups or '_' not in label:
            continue
        parent_str, rank = _split_ranked_label(label)
        if rank is None or rank == 0:
            continue
        if parent_str in minor_by_parent:
            minor_by_parent[parent_str].append(label)
    if parents is not None:
        parents = {str(p) for p in parents}
        minor_by_parent = {p: v for p, v in minor_by_parent.items() if p in parents}
    for p in minor_by_parent:
        minor_by_parent[p].sort(key=lambda x: -int(sizes[x]))

    figures: dict = {}
    out_base = Path(clusters_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    for parent, minors in minor_by_parent.items():
        pdir = out_base / f"c{parent}"
        pdir.mkdir(parents=True, exist_ok=True)
        columns = [(g, 'core') for g in canon_groups] + [(m, 'minor') for m in minors]
        canon, minor_genes = _select_gene_blocks(
            adata=adata, canonical_deg_df=canonical_deg_df,
            clusters_dir=clusters_dir, parent=parent,
            minor_subcluster_names=minors,
            top_n_canonical=top_n_canonical, top_n_minor=top_n_minor,
        )
        (pdir / 'genes_canonical.txt').write_text('\n'.join(canon))
        (pdir / 'genes_minor.txt').write_text('\n'.join(minor_genes))
        genes = canon + minor_genes
        if not genes:
            warnings.warn(f"no genes for parent c{parent}; skipping")
            continue
        mat = _build_expression_matrix(adata, subcluster_col=subcluster_col,
                                       genes=genes, columns=columns)
        mat.to_csv(pdir / 'heatmap_data.tsv', sep='\t')
        row_std = mat.std(axis=1).replace(0, np.nan)
        mat_z = mat.sub(mat.mean(axis=1), axis=0).div(row_std, axis=0).clip(-2, 2)
        mat_z_fc = mat_z.fillna(0)

        row_order, row_link = (_cluster_rows(mat_z_fc) if len(mat_z_fc) >= 2
                               else (list(range(len(mat_z_fc))), None))
        core_names = [c[0] for c in columns if c[1] == 'core']
        minor_names = [c[0] for c in columns if c[1] == 'minor']
        core_ord, core_link = (_cluster_columns(mat_z_fc[core_names])
                               if len(core_names) >= 2
                               else (list(range(len(core_names))), None))
        minor_ord, minor_link = (_cluster_columns(mat_z_fc[minor_names])
                                 if len(minor_names) >= 2
                                 else (list(range(len(minor_names))), None))
        ordered_core = [core_names[i] for i in core_ord]
        ordered_minor = [minor_names[i] for i in minor_ord]
        n_core, n_minor = len(ordered_core), len(ordered_minor)
        has_minor = n_minor > 0

        qc_mat, sm_mat = _build_qc_and_sample_matrices(
            adata, subcluster_col=subcluster_col, columns=columns,
            qc_cols=qc_cols, sample_col=sample_col,
        )
        qc_mat.to_csv(pdir / 'qc_tracks.tsv', sep='\t')
        sm_mat.to_csv(pdir / 'sample_composition.tsv', sep='\t')
        qc_z = qc_mat.sub(qc_mat.mean(axis=1), axis=0).div(
            qc_mat.std(axis=1).replace(0, np.nan), axis=0)

        gene_order = [genes[i] for i in row_order]
        heat = mat_z.loc[gene_order]
        n_gene, n_qc, n_sm = len(gene_order), len(qc_mat), len(sm_mat)

        # ---- layout: cols [colorbar, row-dendro, core block, gap, minor block]
        W_core = max(2.5, 0.34 * n_core)
        W_minor = max(0.6, 0.34 * n_minor)
        gap = 0.55
        fig_w = max(9, 0.45 + 1.0 + W_core + (gap + W_minor if has_minor else 0) + 2.4)
        fig_h = max(7, 0.17 * n_gene + 0.34 * (n_qc + n_sm) + 1.9)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=140)
        gs = GridSpec(
            4, 5, figure=fig,
            width_ratios=[0.45, 1.0, W_core,
                          gap if has_minor else 1e-3,
                          W_minor if has_minor else 1e-3],
            height_ratios=[0.8, max(2.0, 0.17 * n_gene),
                           max(0.4, 0.34 * n_qc), max(0.4, 0.34 * n_sm)],
            hspace=0.07, wspace=0.05,
        )
        cmap_main = plt.get_cmap('RdBu_r').copy(); cmap_main.set_bad('lightgray')
        cmap_qc = plt.get_cmap('coolwarm').copy(); cmap_qc.set_bad('lightgray')
        cmap_sm = plt.get_cmap('magma').copy(); cmap_sm.set_bad('lightgray')

        # column dendrograms — one over the core block, one over the minor block
        ax_dc = fig.add_subplot(gs[0, 2])
        if core_link is not None:
            dendrogram(core_link, ax=ax_dc, no_labels=True, color_threshold=0,
                       above_threshold_color='gray')
        ax_dc.set_axis_off()
        if has_minor:
            ax_dm = fig.add_subplot(gs[0, 4])
            if minor_link is not None:
                dendrogram(minor_link, ax=ax_dm, no_labels=True, color_threshold=0,
                           above_threshold_color='gray')
            ax_dm.set_axis_off()

        # gene row dendrogram
        ax_dr = fig.add_subplot(gs[1, 1])
        if row_link is not None:
            dendrogram(row_link, ax=ax_dr, orientation='left', no_labels=True,
                       color_threshold=0, above_threshold_color='gray')
        ax_dr.invert_yaxis(); ax_dr.set_axis_off()

        def _draw_block(gscol, cols, *, y_labels, bold_set):
            """Draw the gene/QC/sample imshows for one column block. Column names
            in ``bold_set`` get a bold x-tick label. Returns the three mappables."""
            ax_h = fig.add_subplot(gs[1, gscol])
            im_h = ax_h.imshow(np.ma.masked_invalid(heat[cols].values), aspect='auto',
                               cmap=cmap_main, vmin=-2, vmax=2, interpolation='nearest')
            ax_h.set_xticks([])
            ax_q = fig.add_subplot(gs[2, gscol])
            im_q = None
            if n_qc:
                im_q = ax_q.imshow(np.ma.masked_invalid(qc_z[cols].values), aspect='auto',
                                   cmap=cmap_qc, vmin=-2, vmax=2, interpolation='nearest')
            ax_q.set_xticks([])
            ax_s = fig.add_subplot(gs[3, gscol])
            im_s = None
            if n_sm:
                im_s = ax_s.imshow(np.ma.masked_invalid(sm_mat[cols].values), aspect='auto',
                                   cmap=cmap_sm, vmin=0, vmax=1, interpolation='nearest')
            for ax, nrow, labels, fs in [(ax_h, n_gene, gene_order, 6),
                                         (ax_q, n_qc, list(qc_mat.index), 7),
                                         (ax_s, n_sm, list(sm_mat.index), 7)]:
                if y_labels:
                    ax.set_yticks(range(nrow))
                    ax.set_yticklabels(labels, fontsize=fs)
                    ax.tick_params(axis='y', labelright=True, labelleft=False,
                                   right=True, left=False, pad=2)
                else:
                    ax.set_yticks([])
            ax_s.set_xticks(range(len(cols)))
            for lbl, name in zip(ax_s.set_xticklabels(cols, rotation=90, fontsize=7),
                                 cols):
                if name in bold_set:
                    lbl.set_fontweight('bold')
            return im_h, im_q, im_s

        # core block — y-labels here only when there is no minor block; the
        # parent's own home core c{parent}_0 gets a bold label too
        home_core = f"c{parent}_0"
        im_h, im_q, im_s = _draw_block(2, ordered_core,
                                       y_labels=(not has_minor), bold_set={home_core})
        if has_minor:
            # minor block is rightmost → carries the gene/feature labels; all bold
            _draw_block(4, ordered_minor, y_labels=True, bold_set=set(ordered_minor))

        # colorbars on the LEFT (col 0) — clear of the right-side labels
        fig.colorbar(im_h, cax=fig.add_subplot(gs[1, 0]),
                     label='expression z-score', ticklocation='left')
        if im_q is not None:
            fig.colorbar(im_q, cax=fig.add_subplot(gs[2, 0]),
                         label='QC z-score', ticklocation='left')
        if im_s is not None:
            fig.colorbar(im_s, cax=fig.add_subplot(gs[3, 0]),
                         label='sample frac', ticklocation='left')

        fig.suptitle(f"parent c{parent} — minor profile  "
                     f"({n_core} cores | {n_minor} minors)", fontsize=11)
        fig_path = pdir / 'minor_profile.png'
        fig.savefig(fig_path, bbox_inches='tight'); plt.close(fig)
        figures[parent] = str(fig_path)
    return {'figures': figures}


# =============================================================================
#  Sparse tie-corrected rank-once Wilcoxon kernel (presto-style)
# =============================================================================

def _wilcoxon_sparse_stats(X, group_labels, groups):
    """Tie-corrected one-vs-rest Mann-Whitney across all genes for ``groups``,
    on sparse (or dense) X WITHOUT densifying. Returns {group: (z, mean_in, mean_out)}
    of float32 (n_genes,) arrays. presto-style: rank each gene once (only its
    nonzeros), share rank sums across groups via Gᵀ@ranks sparse matmuls.

    Assumes X is non-negative (log-normalized counts) so all zeros are the
    smallest values. z is NaN where a group has <2 cells or n-2<n1.
    """
    import scipy.sparse as sp
    from scipy.stats import rankdata

    Xcsc = sp.csc_matrix(X).astype(np.float64)
    Xcsc.eliminate_zeros()
    N, n_genes = Xcsc.shape
    indptr, indices, data = Xcsc.indptr, Xcsc.indices, Xcsc.data

    # Per-gene: global ranks of the nonzeros, zero-block rank r0, tie term T.
    r0 = np.empty(n_genes, dtype=np.float64)
    Tcorr = np.empty(n_genes, dtype=np.float64)
    rnz_data = np.empty_like(data)
    for g in range(n_genes):
        s, e = indptr[g], indptr[g + 1]
        nz = e - s
        nzeros = N - nz
        r0[g] = (nzeros + 1) / 2.0
        Tg = float(nzeros) ** 3 - float(nzeros)          # zero-block tie group
        if nz:
            vals = data[s:e]
            rr = rankdata(vals, method='average')        # ranks 1..nz among nonzeros
            rnz_data[s:e] = rr + nzeros                  # global ranks
            _, counts = np.unique(vals, return_counts=True)
            tt = counts[counts > 1].astype(np.float64)
            Tg += float(np.sum(tt ** 3 - tt))
        Tcorr[g] = Tg

    Rnz = sp.csc_matrix((rnz_data, indices, indptr), shape=(N, n_genes))
    Bnz = sp.csc_matrix((np.ones_like(data), indices, indptr), shape=(N, n_genes))

    groups = list(groups)
    col_of = {k: i for i, k in enumerate(groups)}
    rows = np.arange(N)
    cols = np.fromiter((col_of[l] for l in group_labels), dtype=np.int64, count=N)
    G = sp.csr_matrix((np.ones(N), (rows, cols)), shape=(N, len(groups)))
    GT = G.T.tocsr()                                     # (K x N)
    n1 = np.asarray(G.sum(axis=0)).ravel()               # (K,)

    S_nz = np.asarray((GT @ Rnz).todense())              # (K x n_genes) rank sums of nonzeros
    C_nz = np.asarray((GT @ Bnz).todense())              # (K x n_genes) nonzero counts
    gsum = np.asarray((GT @ Xcsc).todense())             # (K x n_genes) expression sums
    total = np.asarray(Xcsc.sum(axis=0)).ravel()         # (n_genes,)

    out = {}
    denom = N * (N - 1) if N > 1 else 1
    for i, k in enumerate(groups):
        n1k = float(n1[i]); n2k = float(N - n1[i])
        if n1k < 2 or n2k < 2:
            z = np.full(n_genes, np.nan, dtype=np.float32)
        else:
            R1 = S_nz[i] + r0 * (n1k - C_nz[i])
            U1 = R1 - n1k * (n1k + 1) / 2.0
            var = (n1k * n2k / 12.0) * ((N + 1) - Tcorr / denom)
            with np.errstate(invalid='ignore', divide='ignore'):
                z = (U1 - n1k * n2k / 2.0) / np.sqrt(var)
            z = np.where(var > 0, z, np.nan).astype(np.float32)
        mean_in = (gsum[i] / n1k if n1k else np.zeros(n_genes)).astype(np.float32)
        mean_out = ((total - gsum[i]) / n2k if n2k else np.zeros(n_genes)).astype(np.float32)
        out[k] = (z, mean_in, mean_out)
    return out


def wilcoxon_one_vs_rest(
    X,
    group_labels,
    *,
    gene_names,
    n_jobs: int = 1,
    chunk_size: int | None = None,
    apply_logfoldchanges_expm1: bool = True,
) -> pd.DataFrame:
    """One-vs-rest tie-corrected Mann-Whitney U, sparse and vectorised across all
    groups (presto-style, no densify). ``n_jobs``/``chunk_size`` accepted for
    backward compatibility but unused — the kernel is single-pass over sparse X.

    Returns long: group, gene, scores (z), pvals, pvals_adj, logfoldchanges,
    mean_in, mean_out.
    """
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    group_labels = np.asarray(group_labels)
    groups = sorted(set(group_labels.tolist()))
    stats = _wilcoxon_sparse_stats(X, group_labels, groups)
    frames = []
    for k in groups:
        z, mi, mo = stats[k]
        with np.errstate(invalid='ignore'):
            p = 2.0 * norm.sf(np.abs(z))
        p = np.where(np.isnan(z), 1.0, p)
        padj = multipletests(p, method='fdr_bh')[1]
        if apply_logfoldchanges_expm1:
            lfc = np.log2((np.expm1(mi) + 1e-9) / (np.expm1(mo) + 1e-9))
        else:
            lfc = mi - mo
        frames.append(pd.DataFrame({
            'group': k, 'gene': gene_names,
            'scores': np.nan_to_num(z, nan=0.0).astype(np.float32),
            'pvals': p, 'pvals_adj': padj,
            'logfoldchanges': lfc, 'mean_in': mi, 'mean_out': mo,
        }))
    return pd.concat(frames, ignore_index=True)
