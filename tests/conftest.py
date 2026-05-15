import numpy as np
import pandas as pd
import pytest
import anndata as ad
from pathlib import Path

@pytest.fixture
def small_adata():
    rng = np.random.default_rng(0)
    n_genes = 50
    # cluster 0 has 3 subblobs (600, 300, 100); cluster 1 has 1 blob (200)
    blob_specs = [
        # (n_cells, umap_center, leiden, sample_bias, mt_bias, deg_gene_idx)
        (600, ( 0.0, 0.0), '0', None,       0.05,  None),
        (300, ( 4.0, 0.0), '0', 'sample_B', 0.05,   2),    # sample-driven + DE on gene 2
        (100, ( 0.0, 4.0), '0', None,       0.20,   7),    # mito-driven   + DE on gene 7
        (200, (10.0, 0.0), '1', None,       0.05,  None),
    ]
    obs_rows = []
    umap = []
    X_parts = []
    for size, (cx, cy), leiden, samp_bias, mt, deg_g in blob_specs:
        u = rng.normal(loc=(cx, cy), scale=0.3, size=(size, 2))
        umap.append(u)
        x = rng.poisson(lam=1.0, size=(size, n_genes)).astype(np.float32)
        if deg_g is not None:
            x[:, deg_g] += rng.poisson(lam=8.0, size=size)
        X_parts.append(x)
        if samp_bias is None:
            samples = rng.choice(['sample_A', 'sample_B', 'sample_C'], size=size, p=[0.5, 0.2, 0.3])
        else:
            samples = np.full(size, samp_bias)
        percent_mt = rng.normal(loc=mt, scale=0.01, size=size).clip(0, 1) * 100
        n_count    = x.sum(axis=1)
        n_feature  = (x > 0).sum(axis=1)
        hybrid     = rng.normal(loc=0.3, scale=0.05, size=size).clip(0, 1)
        obs_rows.append(pd.DataFrame({
            'leiden':       leiden,
            'orig.ident':   samples,
            'batch':        samples,
            'percent.mt':   percent_mt,
            'nCount_RNA':   n_count,
            'nFeature_RNA': n_feature,
            'hybrid_score': hybrid,
        }))
    X = np.vstack(X_parts)
    # log1p-normalise so deg_layer=None path uses log-norm
    lib = X.sum(axis=1, keepdims=True).clip(min=1)
    X_log = np.log1p(X / lib * 1e4).astype(np.float32)
    obs = pd.concat(obs_rows, ignore_index=True)
    obs['leiden'] = obs['leiden'].astype('category')
    var = pd.DataFrame(index=[f'g{i:02d}' for i in range(n_genes)])
    a = ad.AnnData(X=X_log, obs=obs, var=var)
    a.layers['counts_recovered'] = X
    a.obsm['X_umap'] = np.vstack(umap).astype(np.float32)
    a.obs_names = [f'cell_{i:05d}' for i in range(a.n_obs)]
    return a

@pytest.fixture
def anatomy_inputs(small_adata):
    """Adds:
      - obs['original_cluster_split']: c0_0/c0_1/c0_2/c1_0  (the Cartesian-product split label)
      - canonical_deg_df: top-3 markers per c{N}_0 (planted)
      - minor_deg_dir: a tmp directory containing deg_c0_1.tsv / deg_c0_2.tsv
    """
    import tempfile
    a = small_adata
    centers = a.obsm['X_umap']
    sub = np.where(a.obs['leiden'].astype(str) == '0',
                   np.where(centers[:,0] > 2, 'c0_1',
                            np.where(centers[:,1] > 2, 'c0_2', 'c0_0')),
                   'c1_0')
    a.obs['original_cluster_split'] = pd.Categorical(sub)
    rows = []
    for g_name, gene_idx_start in [('c0_0', 0), ('c1_0', 10)]:
        for rank in range(3):
            rows.append({'group': g_name, 'rank': rank,
                         'gene': f'g{(gene_idx_start+rank):02d}',
                         'logfoldchanges': 5.0 - rank,
                         'pvals': 0.0, 'pvals_adj': 0.0,
                         'scores': 100.0 - rank})
    canonical_deg_df = pd.DataFrame(rows)
    tmp = tempfile.mkdtemp()
    minor_deg_dir = Path(tmp) / 'leiden'
    (minor_deg_dir / 'c0').mkdir(parents=True)
    pd.DataFrame({'names': ['g02','g03'], 'logfoldchanges': [3.0, 2.0],
                  'pvals': [0,0], 'pvals_adj':[0,0], 'scores':[80,70],
                  'direction':['up','up']}).to_csv(
        minor_deg_dir / 'c0' / 'deg_c0_1.tsv', sep='\t', index=False)
    pd.DataFrame({'names': ['g07','g08'], 'logfoldchanges': [3.0, 2.0],
                  'pvals': [0,0], 'pvals_adj':[0,0], 'scores':[80,70],
                  'direction':['up','up']}).to_csv(
        minor_deg_dir / 'c0' / 'deg_c0_2.tsv', sep='\t', index=False)
    return {'adata': a, 'canonical_deg_df': canonical_deg_df,
            'clusters_dir': minor_deg_dir, 'cluster_col': 'leiden',
            'subcluster_col': 'original_cluster_split'}

@pytest.fixture
def mrvi_adata():
    """Tiny AnnData for MrVI tests: 1200 cells, 3 samples, 150 genes.
    Samples are named sample07/sample08/sample03 (real names, so add_klgrade maps them).
    Raw counts in layers['counts_recovered']; X is log1p-normalised.
    """
    rng = np.random.default_rng(0)
    n_per, n_genes = 400, 150
    samples = ['sample07', 'sample08', 'sample03']
    counts_parts, obs_rows = [], []
    for si, samp in enumerate(samples):
        # each sample gets a mild per-sample shift so MrVI has signal to learn
        lam = rng.uniform(0.5, 2.0, size=n_genes) * (1.0 + 0.3 * si)
        c = rng.poisson(lam=lam, size=(n_per, n_genes)).astype(np.float32)
        counts_parts.append(c)
        obs_rows.append(pd.DataFrame({'orig.ident': samp}, index=[f'{samp}_cell{i}' for i in range(n_per)]))
    counts = np.vstack(counts_parts)
    obs = pd.concat(obs_rows)
    lib = counts.sum(axis=1, keepdims=True).clip(min=1)
    X_log = np.log1p(counts / lib * 1e4).astype(np.float32)
    a = ad.AnnData(X=X_log, obs=obs,
                   var=pd.DataFrame(index=[f'g{i:03d}' for i in range(n_genes)]))
    a.layers['counts_recovered'] = counts
    a.obs['orig.ident'] = a.obs['orig.ident'].astype('category')
    return a
