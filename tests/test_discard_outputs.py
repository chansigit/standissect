import pathlib
import sys

import pandas as pd

_PKG_PARENT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PKG_PARENT))

from standissect.pipeline import (_write_cell_dispositions, _Layout,  # noqa: E402
                                  _PANEL_COLS, _DIAGNOSIS_COLS,
                                  _write_proposed_cell_types, _PROPOSED_COLS,
                                  _annotate_cells, _resolve_cell_types,
                                  _write_cleaned_h5ad)


def _panel():
    return pd.DataFrame({
        'parent_cluster': ['0', '0', '1'],
        'subcluster': ['c0_1', 'c0_2', 'c1_1'],
        'n_cells': [10, 20, 5],
        'likely_cause': ['doublet-driven', 'cell-cycle', 'unclear'],
        'diagnosis_confidence': [0.9, 0.8, 0.4],
        'recommended_disposition': ['DISCARD', 'KEEP', 'UNCERTAIN'],
        'disposition_reason': ['doublets', 'cycling', 'unclear'],
        'proposed_cell_type': [None, 'cycling T', None],
    })


def test_panel_cols_include_disposition_and_proposed_columns():
    for c in ('recommended_disposition', 'disposition_baseline',
              'disposition_overridden', 'disposition_reason', 'proposed_cell_type'):
        assert c in _PANEL_COLS
        assert c in _DIAGNOSIS_COLS


def test_cell_labels_and_discard_file_with_input_row_index(tmp_path):
    lay = _Layout(tmp_path, 'leiden')
    lay.root.mkdir(parents=True)
    obs_names = ['AAA', 'BBB', 'CCC', 'DDD']
    pd.DataFrame(
        {'umap_cluster': ['a', 'a', 'b', 'b'],
         'original_cluster_split': ['c0_1', 'c0_2', 'c1_1', 'c0_1']},
        index=obs_names,
    ).to_csv(lay.cell_labels, sep='\t')

    _write_cell_dispositions(lay, _panel(), obs_names)

    labels = pd.read_csv(lay.cell_labels, sep='\t', index_col=0)
    assert list(labels['recommended_disposition']) == ['DISCARD', 'KEEP',
                                                        'UNCERTAIN', 'DISCARD']
    discard = pd.read_csv(lay.discard_cells, sep='\t')
    assert sorted(discard['barcode']) == ['AAA', 'DDD']
    assert sorted(discard['input_row_index']) == [0, 3]      # positions in obs_names
    assert list(discard.columns) == ['barcode', 'input_row_index', 'subcluster',
                                     'parent_cluster', 'likely_cause',
                                     'diagnosis_confidence', 'disposition_reason']
    assert set(discard['subcluster']) == {'c0_1'}


def test_empty_discard_writes_header_only(tmp_path):
    lay = _Layout(tmp_path, 'leiden')
    lay.root.mkdir(parents=True)
    pd.DataFrame({'umap_cluster': ['a'], 'original_cluster_split': ['c0_2']},
                 index=['AAA']).to_csv(lay.cell_labels, sep='\t')
    _write_cell_dispositions(lay, _panel(), ['AAA'])
    discard = pd.read_csv(lay.discard_cells, sep='\t')
    assert len(discard) == 0
    assert 'barcode' in discard.columns and 'input_row_index' in discard.columns


def test_proposed_cell_types_collects_minor_and_major(tmp_path):
    lay = _Layout(tmp_path, 'leiden')
    lay.root.mkdir(parents=True)
    panel = pd.DataFrame({
        'parent_cluster': ['myeloid', 'myeloid'],
        'subcluster': ['cmyeloid_1', 'cmyeloid_2'],
        'proposed_cell_type': ['pDC', None],
        'diagnosis_confidence': [0.8, 0.4],
        'diagnosis_rationale': ['pDC markers', 'n/a'],
    })
    core = pd.DataFrame({
        'parent_cluster': ['myeloid', 'tcell'],
        'core_subcluster': ['cmyeloid_0', 'ctcell_0'],
        'cell_type': ['cDC1', 'T cell'],
        'confidence': [0.9, 0.95], 'rationale': ['cDC1 markers', 'CD3'],
        'original_label': ['myeloid', 'tcell'],
        'differs_from_original': [True, False],
    })
    _write_proposed_cell_types(lay, panel, core)
    out = pd.read_csv(lay.proposed_cell_types, sep='\t')
    assert list(out.columns) == _PROPOSED_COLS
    assert set(out['level']) == {'minor', 'major'}
    assert set(out['proposed_cell_type']) == {'pDC', 'cDC1'}    # 'cycling None' & non-differing excluded


def test_proposed_cell_types_empty_header_only(tmp_path):
    lay = _Layout(tmp_path, 'leiden')
    lay.root.mkdir(parents=True)
    panel = pd.DataFrame({'parent_cluster': ['0'], 'subcluster': ['c0_1'],
                          'proposed_cell_type': [None]})
    core = pd.DataFrame({'parent_cluster': ['0'], 'core_subcluster': ['c0_0'],
                         'cell_type': ['T cell'], 'differs_from_original': [False]})
    _write_proposed_cell_types(lay, panel, core)
    out = pd.read_csv(lay.proposed_cell_types, sep='\t')
    assert len(out) == 0 and list(out.columns) == _PROPOSED_COLS


def test_proposed_major_excludes_null_cell_type(tmp_path):
    lay = _Layout(tmp_path, 'leiden')
    lay.root.mkdir(parents=True)
    panel = pd.DataFrame({'parent_cluster': ['0'], 'subcluster': ['c0_1'],
                          'proposed_cell_type': [None]})
    core = pd.DataFrame({'parent_cluster': ['m', 'n'],
                         'core_subcluster': ['cm_0', 'cn_0'],
                         'cell_type': ['cDC1', None],
                         'confidence': [0.9, 0.5], 'rationale': ['markers', 'x'],
                         'differs_from_original': [True, True]})
    _write_proposed_cell_types(lay, panel, core)
    out = pd.read_csv(lay.proposed_cell_types, sep='\t')
    assert list(out['proposed_cell_type']) == ['cDC1']


def test_apply_discard_and_celltype_obs(tmp_path):
    import anndata as ad
    import numpy as np
    obs = pd.DataFrame({'cell_ontology_class': ['granulocyte', 'granulocyte', 'B cell', 'granulocyte'],
                        'original_cluster_split': ['cgranulocyte_1', 'cgranulocyte_0',
                                                   'cB cell_0', 'cgranulocyte_2']},
                       index=['A', 'B', 'C', 'D'])
    a = ad.AnnData(X=np.zeros((4, 2), dtype='float32'), obs=obs)
    panel = pd.DataFrame({'subcluster': ['cgranulocyte_1', 'cgranulocyte_2'],
                          'recommended_disposition': ['KEEP', 'DISCARD'],
                          'proposed_cell_type': ['cycling granulocyte', None]})
    core = pd.DataFrame({'parent_cluster': ['granulocyte', 'B cell'],
                         'cell_type': ['neutrophil', None],
                         'differs_from_original': [True, False]})
    _annotate_cells(a, panel, core, 'cell_ontology_class')
    # per-cell resolution: A=minor 'cycling granulocyte'; B=major 'neutrophil';
    # C=major None->fallback original 'B cell'; D=minor None->major 'neutrophil'
    assert list(a.obs['proposed_cell_type']) == ['cycling granulocyte', 'neutrophil',
                                                 'B cell', 'neutrophil']
    out = tmp_path / 'cleaned.h5ad'
    n_disc, n_kept = _write_cleaned_h5ad(a, str(out))
    assert (n_disc, n_kept) == (1, 3)                       # D was DISCARD
    cleaned = ad.read_h5ad(out)
    assert 'D' not in list(cleaned.obs_names)
    assert 'proposed_cell_type' in cleaned.obs.columns
    assert 'recommended_disposition' in cleaned.obs.columns
