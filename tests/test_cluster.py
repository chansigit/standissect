import numpy as np
import pandas as pd
from pathlib import Path

import scanpy as sc
from standissect.cluster import wilcoxon_vs_reference

def test_wilcoxon_vs_reference_recovers_planted_gene(small_adata):
    sub = small_adata[small_adata.obs['leiden'] == '0'].copy()
    # plant labels using the known UMAP centers
    centers = sub.obsm['X_umap']
    label = np.where(centers[:,0] > 2, 'c0_1',
            np.where(centers[:,1] > 2, 'c0_2', 'c0_0'))
    df = wilcoxon_vs_reference(sub.X, label, group='c0_1', reference='c0_0',
                               gene_names=list(sub.var_names), n_genes=10)
    assert {'names', 'logfoldchanges', 'pvals', 'pvals_adj', 'scores', 'direction'} <= set(df.columns)
    # gene g02 was planted as up-regulated in the c0_1 blob
    assert 'g02' in df.head(5)['names'].values
    assert (df['direction'].isin({'up','down'})).all()
    assert len(df) == 10

from standissect.cluster import _composition_drift

def test_composition_drift_detects_sample_bias(small_adata):
    sub = small_adata[small_adata.obs['leiden'] == '0'].copy()
    centers = sub.obsm['X_umap']
    sub.obs['__sub'] = pd.Categorical(np.where(centers[:,0] > 2, 'c0_1', 'c0_0'))
    df = _composition_drift(sub.obs, group_col='__sub', group='c0_1', reference='c0_0',
                            cat_col='orig.ident')
    assert {'category','n_in_j','frac_in_j','frac_in_0','log2_OR','p','padj'} <= set(df.columns)
    # the planted bias was 100% sample_B in c0_1
    sB = df[df['category'] == 'sample_B'].iloc[0]
    assert sB['log2_OR'] > 2
    assert sB['padj'] < 0.05

from standissect.cluster import _qc_drift

def test_qc_drift_detects_mito_bias(small_adata):
    sub = small_adata[small_adata.obs['leiden'] == '0'].copy()
    centers = sub.obsm['X_umap']
    # The (0,4) blob has elevated percent.mt; label it c0_1 here
    sub.obs['__sub'] = pd.Categorical(np.where(centers[:,1] > 2, 'c0_1', 'c0_0'))
    df = _qc_drift(sub.obs, group_col='__sub', group='c0_1', reference='c0_0',
                   qc_cols=('percent.mt','nCount_RNA','nFeature_RNA','hybrid_score'))
    assert {'qc_col','mean_j','mean_0','delta','relative_delta','p','padj'} <= set(df.columns)
    mt_row = df[df['qc_col'] == 'percent.mt'].iloc[0]
    assert mt_row['delta'] > 5      # planted +15pp
    assert mt_row['padj'] < 0.05

from standissect.cluster import _likely_cause

def test_likely_cause_precedence():
    # sample-driven wins over mito
    assert _likely_cause(top_sample={'log2_OR': 3.0, 'padj': 0.001, 'category': 'X'},
                         top_qc=None, n_sig=10) == 'sample-driven'
    # doublet
    assert _likely_cause(top_sample=None,
                         top_qc={'qc_col': 'hybrid_score', 'delta': 0.4, 'relative_delta': 0.8, 'padj': 0.01},
                         n_sig=5) == 'doublet-driven'
    # high mt
    assert _likely_cause(top_sample=None,
                         top_qc={'qc_col': 'percent.mt', 'delta': 5.0, 'relative_delta': 0.5, 'padj': 0.01},
                         n_sig=5) == 'low-quality (high mt)'
    # shallow depth
    assert _likely_cause(top_sample=None,
                         top_qc={'qc_col': 'nFeature_RNA', 'delta': -300, 'relative_delta': -0.5, 'padj': 0.01},
                         n_sig=5) == 'shallow-depth'
    # biology
    assert _likely_cause(top_sample=None, top_qc=None, n_sig=25) == 'biology-candidate'
    # unclear
    assert _likely_cause(top_sample=None, top_qc=None, n_sig=3) == 'unclear'


from standissect.cluster import _select_gene_blocks

def test_select_gene_blocks(anatomy_inputs):
    canon, minor = _select_gene_blocks(
        adata=anatomy_inputs['adata'],
        canonical_deg_df=anatomy_inputs['canonical_deg_df'],
        clusters_dir=anatomy_inputs['clusters_dir'],
        parent='0',
        minor_subcluster_names=['c0_1', 'c0_2'],
        top_n_canonical=2, top_n_minor=2,
    )
    assert canon == ['g00', 'g01', 'g10', 'g11']
    assert set(minor) == {'g02','g03','g07','g08'}
    assert canon == sorted(canon, key=lambda g: ['g00','g01','g10','g11'].index(g))

from standissect.cluster import _build_expression_matrix

def test_build_expression_matrix(anatomy_inputs):
    a = anatomy_inputs['adata']
    genes = ['g00','g02','g07','g10']
    cols  = [('c0_0','core'), ('c1_0','core'), ('c0_1','minor'), ('c0_2','minor')]
    mat = _build_expression_matrix(a, subcluster_col='original_cluster_split',
                                   genes=genes, columns=cols)
    assert mat.shape == (4, 4)
    assert list(mat.index) == genes
    assert [c[0] for c in cols] == list(mat.columns)
    assert mat.loc['g02', 'c0_1'] > mat.loc['g02', 'c0_0']
    assert mat.loc['g07', 'c0_2'] > mat.loc['g07', 'c0_0']

from standissect.cluster import _build_qc_and_sample_matrices

def test_qc_and_sample_matrices(anatomy_inputs):
    a = anatomy_inputs['adata']
    cols = [('c0_0','core'), ('c1_0','core'), ('c0_1','minor'), ('c0_2','minor')]
    qc, sm = _build_qc_and_sample_matrices(
        a, subcluster_col='original_cluster_split',
        columns=cols,
        qc_cols=('percent.mt','nCount_RNA'),
        sample_col='orig.ident',
    )
    assert qc.shape == (2, 4)
    assert list(qc.index) == ['percent.mt','nCount_RNA']
    assert qc.loc['percent.mt'].idxmax() == 'c0_2'
    assert np.allclose(sm.sum(axis=0).values, 1.0, atol=1e-6)
    assert sm.loc['sample_B', 'c0_1'] > 0.9

from standissect.cluster import _cluster_columns, _cluster_rows

def test_cluster_columns_returns_ordered_idx(anatomy_inputs):
    a = anatomy_inputs['adata']
    cols = [('c0_0','core'), ('c1_0','core'), ('c0_1','minor'), ('c0_2','minor')]
    from standissect.cluster import _build_expression_matrix, _select_gene_blocks
    canon, minor = _select_gene_blocks(
        adata=a, canonical_deg_df=anatomy_inputs['canonical_deg_df'],
        clusters_dir=anatomy_inputs['clusters_dir'], parent='0',
        minor_subcluster_names=['c0_1','c0_2'],
        top_n_canonical=2, top_n_minor=2,
    )
    mat = _build_expression_matrix(a, subcluster_col='original_cluster_split',
                                    genes=canon+minor, columns=cols)
    order, linkage = _cluster_columns(mat)
    assert len(order) == 4
    assert set(order) == {0,1,2,3}
    row_order_c, link_c = _cluster_rows(mat.loc[canon])
    assert len(row_order_c) == len(canon)

from standissect.cluster import plot_minor_anatomy

def test_plot_minor_anatomy_writes_files(anatomy_inputs):
    res = plot_minor_anatomy(
        adata=anatomy_inputs['adata'],
        subcluster_col=anatomy_inputs['subcluster_col'],
        canonical_deg_df=anatomy_inputs['canonical_deg_df'],
        clusters_dir=anatomy_inputs['clusters_dir'],
        qc_cols=('percent.mt','nCount_RNA'),
        sample_col='orig.ident',
        top_n_canonical=2, top_n_minor=2,
        min_subcluster_size=10,
    )
    assert '0' in res['figures']
    p = Path(res['figures']['0'])
    assert p.exists()
    assert p.name == 'minor_anatomy.png'
    assert (p.parent / 'heatmap_data.tsv').exists()
    assert (p.parent / 'qc_tracks.tsv').exists()
    assert (p.parent / 'sample_composition.tsv').exists()
    assert (p.parent / 'genes_canonical.txt').exists()
    assert (p.parent / 'genes_minor.txt').exists()


from standissect.cluster import wilcoxon_one_vs_rest

def test_wilcoxon_one_vs_rest_recovers_planted_genes(small_adata):
    """Vectorised one-vs-rest M-W should rank the planted up-genes top
    for the blob that received the planted bump."""
    a = small_adata
    centers = a.obsm['X_umap']
    # 4 groups: cluster1, then 3 sub-blobs of cluster0 (g02 up in (4,0), g07 up in (0,4))
    grp = np.where(a.obs['leiden'].astype(str) == '0',
                   np.where(centers[:,0] > 2, 'A',
                            np.where(centers[:,1] > 2, 'B', 'C')),
                   'D')
    df = wilcoxon_one_vs_rest(
        a.X, grp, gene_names=list(a.var_names),
        chunk_size=20, n_jobs=1,
    )
    # Top up-gene in group A (planted g02)
    top_A = df[(df['group']=='A') & (df['logfoldchanges']>0)].sort_values('scores', ascending=False)['gene'].iloc[0]
    assert top_A == 'g02'
    # Top up-gene in group B (planted g07)
    top_B = df[(df['group']=='B') & (df['logfoldchanges']>0)].sort_values('scores', ascending=False)['gene'].iloc[0]
    assert top_B == 'g07'
    # All 4 groups produced rows
    assert set(df['group'].unique()) == {'A','B','C','D'}
