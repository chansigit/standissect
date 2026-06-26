import pathlib
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_DIR))

import cluster  # noqa: E402


def _ref_z_p(x, y):
    """Reference tie-corrected two-sided MWU on group x vs group y (one gene)."""
    res = mannwhitneyu(x, y, alternative='two-sided', use_continuity=False,
                       method='asymptotic')
    # recover z from p (two-sided): |z| = isf(p/2)
    from scipy.stats import norm
    return res.statistic, res.pvalue


def test_kernel_matches_scipy_with_ties_sparse_eq_dense():
    rng = np.random.default_rng(0)
    # 60 cells x 5 genes, integer counts with many zeros (heavy ties at 0)
    dense = rng.poisson(0.4, size=(60, 5)).astype(np.float32)
    labels = np.array(['a'] * 25 + ['b'] * 35)
    out_dense = cluster._wilcoxon_sparse_stats(dense, labels, ['a', 'b'])
    out_sparse = cluster._wilcoxon_sparse_stats(sp.csr_matrix(dense), labels, ['a', 'b'])
    # sparse == dense
    for k in ('a', 'b'):
        np.testing.assert_allclose(out_dense[k][0], out_sparse[k][0], rtol=1e-4, atol=1e-4,
                                   equal_nan=True)
    # vs scipy reference (group 'a' one-vs-rest = a vs b), per gene, via p-value
    from scipy.stats import norm
    za = out_dense['a'][0]
    for g in range(5):
        x = dense[labels == 'a', g]
        y = dense[labels == 'b', g]
        _, p_ref = _ref_z_p(x, y)
        p_ours = 2.0 * norm.sf(np.abs(za[g]))
        np.testing.assert_allclose(p_ours, p_ref, rtol=1e-3, atol=1e-3)


def test_kernel_all_zero_gene_is_nonsignificant():
    dense = np.zeros((20, 3), dtype=np.float32)
    dense[:, 1] = 1.0  # constant nonzero gene also a full tie
    labels = np.array(['a'] * 10 + ['b'] * 10)
    out = cluster._wilcoxon_sparse_stats(dense, labels, ['a', 'b'])
    z = out['a'][0]
    # all-tie genes -> U at the mean -> z ~ 0 (or nan); p must be ~1
    from scipy.stats import norm
    p = 2.0 * norm.sf(np.abs(np.nan_to_num(z, nan=0.0)))
    assert np.all(p > 0.99)


def test_kernel_small_group_is_nan():
    dense = np.random.default_rng(1).poisson(0.5, size=(10, 4)).astype(np.float32)
    labels = np.array(['a'] + ['b'] * 9)   # group 'a' has 1 cell
    out = cluster._wilcoxon_sparse_stats(dense, labels, ['a', 'b'])
    assert np.all(np.isnan(out['a'][0]))


def test_one_vs_rest_columns_and_no_densify_path():
    rng = np.random.default_rng(2)
    X = sp.csr_matrix(rng.poisson(0.5, size=(40, 6)).astype(np.float32))
    labels = np.array(['g0'] * 13 + ['g1'] * 13 + ['g2'] * 14)
    df = cluster.wilcoxon_one_vs_rest(X, labels, gene_names=[f"G{i}" for i in range(6)])
    assert set(df.columns) == {'group', 'gene', 'scores', 'pvals', 'pvals_adj',
                               'logfoldchanges', 'mean_in', 'mean_out'}
    assert sorted(df['group'].unique()) == ['g0', 'g1', 'g2']
    assert len(df) == 18


def test_vs_reference_columns_and_topn():
    rng = np.random.default_rng(3)
    X = sp.csr_matrix(rng.poisson(0.5, size=(50, 8)).astype(np.float32))
    labels = np.array(['minor'] * 20 + ['core'] * 20 + ['other'] * 10)
    df = cluster.wilcoxon_vs_reference(X, labels, group='minor', reference='core',
                                       gene_names=[f"G{i}" for i in range(8)], n_genes=5)
    assert list(df.columns) == ['names', 'logfoldchanges', 'pvals', 'pvals_adj',
                                'scores', 'direction']
    assert len(df) == 5
    assert df['scores'].is_monotonic_decreasing
