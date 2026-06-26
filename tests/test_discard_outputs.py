import pathlib
import sys

import pandas as pd

_PKG_PARENT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PKG_PARENT))

from standissect.pipeline import (_write_cell_dispositions, _Layout,  # noqa: E402
                                  _PANEL_COLS, _DIAGNOSIS_COLS)


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
