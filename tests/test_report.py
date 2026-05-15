import base64
from pathlib import Path

from standissect.report import build_report, _img, _table

# minimal valid 1x1 PNG
_PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4'
    '2mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==')


def _write_png(path):
    Path(path).write_bytes(_PNG)


def test_build_report(tmp_path):
    root = tmp_path / 'leiden_mrvi'
    (root / 'clusters' / 'c0').mkdir(parents=True)
    (root / 'clusters' / 'c1').mkdir(parents=True)
    (root / 'canonical_markers').mkdir(parents=True)
    _write_png(root / 'global_umap_compare.png')
    _write_png(root / 'canonical_markers' / 'heatmap_top_markers.png')
    (root / 'panel.tsv').write_text('parent_cluster\tsubcluster\n0\tc0_1\n')
    for c in ('0', '1'):
        _write_png(root / 'clusters' / f'c{c}' / 'minor_anatomy.png')
        _write_png(root / 'clusters' / f'c{c}' / 'umap_subcluster.png')
        (root / 'clusters' / f'c{c}' / 'panel.tsv').write_text(f'parent_cluster\n{c}\n')
    (root / 'clusters' / 'c0' / 'deg_c0_1.tsv').write_text('names\tscores\nGENE1\t5.0\n')

    out = build_report(str(root))
    assert Path(out).exists()
    html = Path(out).read_text()
    assert html.startswith('<!doctype html>')
    assert 'id="overview"' in html
    assert 'id="c0"' in html and 'id="c1"' in html
    assert 'data:image/png;base64,' in html          # images embedded inline
    assert 'deg vs main' in html.lower()             # per-minor DEG section
    # natural cluster order in the sidebar
    assert html.index('#c0') < html.index('#c1')


def test_img_missing(tmp_path):
    assert 'missing' in _img(tmp_path / 'nope.png')


def test_img_embeds(tmp_path):
    p = tmp_path / 'x.png'
    _write_png(p)
    tag = _img(p)
    assert tag.startswith('<img src="data:image/png;base64,')


def test_table_header_only(tmp_path):
    p = tmp_path / 'e.tsv'
    p.write_text('a\tb\n')           # header, no rows
    assert 'no rows' in _table(p)


def test_table_missing(tmp_path):
    assert _table(tmp_path / 'nope.tsv') == ''
