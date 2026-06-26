#!/usr/bin/env python
"""standissect.report — single-file HTML report for a standissect run.

Reads a ``run_dissect_pipeline`` output tree and writes a self-contained
``report.html`` (all images base64-embedded) with an overview section and a
per-cluster sidebar.

Usage:  python -m standissect.report <output_root>
"""
from __future__ import annotations
import base64
import html as _html
import sys
from pathlib import Path

import pandas as pd


def _img(path, *, max_width='100%'):
    """An <img> tag with the PNG base64-embedded, or a 'missing' note."""
    path = Path(path)
    if not path.exists():
        return f'<p class="missing">[missing: {path.name}]</p>'
    b64 = base64.b64encode(path.read_bytes()).decode('ascii')
    return (f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:{max_width};height:auto;border:1px solid #e3e6ee;">')


def _table(path, *, max_rows=None):
    """Render a TSV as an HTML table, or '' if absent/empty."""
    path = Path(path)
    if not path.exists():
        return ''
    try:
        df = pd.read_csv(path, sep='\t')
    except Exception:
        return ''
    if not len(df):
        return '<p class="muted">(no rows)</p>'
    if max_rows is not None and len(df) > max_rows:
        note = f'<p class="muted">showing {max_rows} of {len(df)} rows</p>'
        df = df.head(max_rows)
    else:
        note = ''
    return note + df.to_html(index=False, border=0, classes='deg',
                             float_format=lambda x: f'{x:.3g}')


def _read_tsv_safe(path):
    try:
        return pd.read_csv(path, sep='\t')
    except Exception:
        return pd.DataFrame()


def _load_core_names_map(path):
    """core_names.tsv -> {parent_cluster: cell_type} (only named clusters)."""
    df = _read_tsv_safe(path)
    out = {}
    if len(df) and 'parent_cluster' in df.columns and 'cell_type' in df.columns:
        for _, r in df.iterrows():
            ct = r.get('cell_type')
            if ct is not None and not (isinstance(ct, float) and pd.isna(ct)) and str(ct) != 'nan':
                out[str(r['parent_cluster'])] = str(ct)
    return out


def _load_narratives_map(path):
    """narratives.tsv -> {parent_cluster: narrative}."""
    df = _read_tsv_safe(path)
    out = {}
    if len(df) and 'parent_cluster' in df.columns and 'narrative' in df.columns:
        for _, r in df.iterrows():
            nv = r.get('narrative')
            if nv is not None and not (isinstance(nv, float) and pd.isna(nv)) and str(nv) != 'nan':
                out[str(r['parent_cluster'])] = str(nv)
    return out


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#222;}
#sidebar{position:fixed;top:0;left:0;width:164px;height:100%;overflow-y:auto;
  background:#1e2330;padding:14px 0;}
#sidebar a{display:block;color:#cdd3e0;text-decoration:none;padding:5px 16px;font-size:13px;}
#sidebar a:hover{background:#2d3548;color:#fff;}
#sidebar .head{color:#7f8aa3;font-size:11px;text-transform:uppercase;
  padding:10px 16px 2px;letter-spacing:.5px;}
#main{margin-left:184px;padding:24px 32px;max-width:1500px;}
h1{font-size:22px;margin:0 0 4px;}
h2{font-size:18px;border-bottom:2px solid #e3e6ee;padding-bottom:4px;margin-top:38px;}
.imgrow{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start;}
.imgrow > div{flex:1;min-width:340px;}
.cap{font-size:12px;color:#5a6473;margin:10px 0 4px;font-weight:600;}
table.deg{border-collapse:collapse;font-size:11px;margin:4px 0 10px;}
table.deg th,table.deg td{border:1px solid #dde;padding:2px 6px;text-align:right;}
table.deg th{background:#eef1f7;position:sticky;top:0;}
details{margin:5px 0;}
summary{cursor:pointer;font-size:13px;color:#2d4a73;}
.missing{color:#b00;font-size:12px;}
.muted{color:#889;font-size:11px;margin:2px 0;}
.narrative{font-size:13px;color:#333;background:#f6f8fc;border-left:3px solid #4a6da7;
  padding:8px 12px;margin:6px 0 12px;line-height:1.5;}
"""


def build_report(root, output_html=None):
    """Build a self-contained report.html for the dissect output tree at ``root``."""
    root = Path(root)
    output_html = Path(output_html) if output_html else root / 'report.html'
    clusters_dir = root / 'clusters'
    cluster_ids = sorted(
        (d.name[1:] for d in clusters_dir.glob('c*') if d.is_dir()),
        key=lambda x: int(x) if x.isdigit() else 10**9,
    )
    core_names = _load_core_names_map(root / 'core_names.tsv')
    narratives = _load_narratives_map(root / 'narratives.tsv')

    h = ['<!doctype html><html><head><meta charset="utf-8">',
         f'<title>dissect report — {root.name}</title>',
         f'<style>{_CSS}</style></head><body>']

    # --- sidebar ---
    h.append('<div id="sidebar">')
    h.append('<div class="head">overview</div>')
    h.append('<a href="#overview">Overview</a>')
    h.append('<div class="head">clusters</div>')
    for cid in cluster_ids:
        nm = core_names.get(str(cid))
        label = f'cluster {cid}' + (f' — {_html.escape(nm)}' if nm else '')
        h.append(f'<a href="#c{cid}">{label}</a>')
    h.append('</div>')

    # --- main ---
    h.append('<div id="main">')
    h.append(f'<h1>dissect cleanup-diagnosis — {root.name}</h1>')

    # overview
    h.append('<h2 id="overview">Overview</h2>')
    h.append('<div class="imgrow">')
    h.append('<div><div class="cap">global UMAP — original clusters vs UMAP-Leiden partition</div>'
             + _img(root / 'global_umap_compare.png') + '</div>')
    h.append('<div><div class="cap">canonical-core marker heatmap</div>'
             + _img(root / 'canonical_markers' / 'heatmap_top_markers.png') + '</div>')
    h.append('</div>')
    h.append('<div class="cap">minor sub-population panel — all clusters</div>')
    h.append(_table(root / 'panel.tsv'))
    core_names_html = _table(root / 'core_names.tsv')
    if core_names_html:
        h.append('<div class="cap">canonical-core cell-type names</div>')
        h.append(core_names_html)

    # per-cluster
    for cid in cluster_ids:
        cdir = clusters_dir / f'c{cid}'
        nm = core_names.get(str(cid))
        title = f'cluster {cid}' + (f' — {_html.escape(nm)}' if nm else '')
        h.append(f'<h2 id="c{cid}">{title}</h2>')
        narr = narratives.get(str(cid))
        if narr:
            h.append(f'<p class="narrative">{_html.escape(narr)}</p>')
        h.append('<div class="imgrow">')
        h.append('<div><div class="cap">minor-profile heatmap</div>'
                 + _img(cdir / 'minor_profile.png') + '</div>')
        h.append('<div><div class="cap">UMAP zoom</div>'
                 + _img(cdir / 'umap_subcluster.png') + '</div>')
        h.append('</div>')
        panel_html = _table(cdir / 'panel.tsv')
        if panel_html:
            h.append('<div class="cap">minors of this cluster</div>' + panel_html)
        markers = cdir.parent.parent / 'canonical_markers' / f'markers_c{cid}_0.tsv'
        if markers.exists():
            h.append(f'<details><summary>canonical markers — c{cid}_0</summary>'
                     + _table(markers, max_rows=50) + '</details>')
        for deg_path in sorted(cdir.glob('deg_c*.tsv')):
            name = deg_path.stem.replace('deg_', '')
            h.append(f'<details><summary>DEG vs main — {name}</summary>'
                     + _table(deg_path, max_rows=50) + '</details>')
    h.append('</div></body></html>')

    output_html.write_text('\n'.join(h), encoding='utf-8')
    return str(output_html)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m standissect.report <dissect_output_root>')
    print(f'wrote {build_report(sys.argv[1])}')
