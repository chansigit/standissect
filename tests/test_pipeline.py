from pathlib import Path

import pytest

from standissect.pipeline import (run_dissect_pipeline, _Layout,
                                      _skip, _normalize_force)


def test_layout_paths(tmp_path):
    lay = _Layout(str(tmp_path), 'leiden')
    assert lay.root == tmp_path / 'leiden'
    assert lay.crosstab == tmp_path / 'leiden' / 'crosstab.tsv'
    assert lay.cell_labels == tmp_path / 'leiden' / 'cell_labels.tsv'
    assert lay.cluster_dir('3') == tmp_path / 'leiden' / 'clusters' / 'c3'
    assert lay.cluster_panel('3') == tmp_path / 'leiden' / 'clusters' / 'c3' / 'panel.tsv'
    assert lay.canonical_deg == tmp_path / 'leiden' / 'canonical_markers' / 'deg_long.tsv'
    assert lay.anatomy_png('3') == tmp_path / 'leiden' / 'clusters' / 'c3' / 'minor_anatomy.png'


def test_skip_logic(tmp_path):
    p = tmp_path / 'x.tsv'
    assert not _skip(p, False)        # missing -> cannot skip
    p.write_text('hi')
    assert _skip(p, False)            # exists, not forced -> skip
    assert not _skip(p, True)         # exists but forced -> recompute


def test_normalize_force():
    assert _normalize_force(()) == set()
    assert _normalize_force('all') == {'partition', 'dissect', 'canonical', 'anatomy'}
    assert _normalize_force(True) == {'partition', 'dissect', 'canonical', 'anatomy'}
    assert _normalize_force(('dissect',)) == {'dissect'}
    assert _normalize_force('canonical') == {'canonical'}
    with pytest.raises(ValueError):
        _normalize_force(('bogus',))


def test_pipeline_end_to_end(small_adata, tmp_path):
    result = run_dissect_pipeline(
        small_adata, cluster_col='leiden', output_dir=str(tmp_path),
        cat_cols=('orig.ident', 'batch'),
        qc_cols=('percent.mt', 'nCount_RNA', 'nFeature_RNA', 'hybrid_score'),
        sample_col='orig.ident', target_k=4, min_subcluster_size=50, n_jobs=1,
    )
    root = Path(result['root'])
    assert root == tmp_path / 'leiden'
    # global files
    for name in ('crosstab.tsv', 'panel.tsv', 'cell_labels.tsv',
                 'params.json', 'global_umap_compare.png'):
        assert (root / name).exists(), name
    # canonical-marker stage
    assert (root / 'canonical_markers' / 'deg_long.tsv').exists()
    assert (root / 'canonical_markers' / 'heatmap_top_markers.png').exists()
    # every cluster gets a folder with a zoom UMAP + anatomy heatmap
    for parent in ('0', '1'):
        assert (root / 'clusters' / f'c{parent}').is_dir()
        assert (root / 'clusters' / f'c{parent}' / 'umap_subcluster.png').exists()
        assert (root / 'clusters' / f'c{parent}' / 'minor_anatomy.png').exists()
        assert (root / 'clusters' / f'c{parent}' / 'panel.tsv').exists()
    # cluster 0 has 3 planted sub-blobs -> off-main minors found
    panel = result['panel']
    assert len(panel) > 0
    assert 'parent_cluster' in panel.columns
    assert 'likely_cause' in panel.columns
    # no stale __-suffixed sibling dirs, no crosstab heatmap
    assert not (root.parent / 'leiden__canonical_markers').exists()
    assert not (root / 'crosstab_heatmap.png').exists()


def test_pipeline_idempotent(small_adata, tmp_path):
    run_dissect_pipeline(small_adata, cluster_col='leiden',
                         output_dir=str(tmp_path), cat_cols=('orig.ident', 'batch'),
                         target_k=4, n_jobs=1)
    cell_labels = tmp_path / 'leiden' / 'cell_labels.tsv'
    canon = tmp_path / 'leiden' / 'canonical_markers' / 'deg_long.tsv'
    c0_panel = tmp_path / 'leiden' / 'clusters' / 'c0' / 'panel.tsv'
    mtimes = {p: p.stat().st_mtime for p in (cell_labels, canon, c0_panel)}

    result2 = run_dissect_pipeline(small_adata, cluster_col='leiden',
                                   output_dir=str(tmp_path),
                                   cat_cols=('orig.ident', 'batch'),
                                   target_k=4, n_jobs=1)
    assert 'partition' in result2['skipped']
    assert 'canonical' in result2['skipped']
    assert 'dissect:c0' in result2['skipped']
    # cached artifacts not rewritten
    for p, mt in mtimes.items():
        assert p.stat().st_mtime == mt, f'{p} was rewritten'


def test_pipeline_force_all(small_adata, tmp_path):
    run_dissect_pipeline(small_adata, cluster_col='leiden',
                         output_dir=str(tmp_path), cat_cols=('orig.ident', 'batch'),
                         target_k=4, n_jobs=1)
    result = run_dissect_pipeline(small_adata, cluster_col='leiden',
                                  output_dir=str(tmp_path),
                                  cat_cols=('orig.ident', 'batch'),
                                  target_k=4, n_jobs=1, force='all')
    assert result['skipped'] == []


def test_pipeline_rejects_bad_umap_key(small_adata, tmp_path):
    with pytest.raises(KeyError):
        run_dissect_pipeline(small_adata, cluster_col='leiden',
                             output_dir=str(tmp_path), umap_key='X_missing',
                             n_jobs=1)


def test_pipeline_parallel(small_adata, tmp_path):
    """Stage-2 cluster parallelism (fork pool, n_jobs>1) produces the full tree."""
    result = run_dissect_pipeline(
        small_adata, cluster_col='leiden', output_dir=str(tmp_path),
        cat_cols=('orig.ident', 'batch'), target_k=4,
        min_subcluster_size=50, n_jobs=2,
    )
    root = Path(result['root'])
    for parent in ('0', '1'):
        assert (root / 'clusters' / f'c{parent}' / 'panel.tsv').exists()
        assert (root / 'clusters' / f'c{parent}' / 'umap_subcluster.png').exists()
    assert len(result['panel']) > 0
    assert result['skipped'] == []
