"""standissect.annotate — LLM cell-type naming + per-cluster narrative.

Two annotation layers on top of diagnosis, mirroring its shape (evidence
dataclass -> engine over a chat client -> result dataclass):

  * core cell-type NAMING — map a canonical core's ranked markers to a cell type
    (LLM primary, local marker-overlap backup; always produces a result);
  * per-cluster NARRATIVE — one evidence-grounded paragraph (LLM only).

Stdlib + pandas only. Reuses the shared OpenAI-compatible client via
``llm_client.call_structured``; the client itself is built/owned by the caller
(see ``diagnosis.make_chat_client``) and passed in, so this module imports
neither scanpy nor diagnosis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

try:                                    # package use (standissect.annotate)
    from .llm_client import call_structured, LLMUnavailable
except ImportError:                     # standalone use (tests import top-level)
    from llm_client import call_structured, LLMUnavailable

import pandas as pd


NAMING_PROMPT_VERSION = 'standissect-naming-v1'
NARRATIVE_PROMPT_VERSION = 'standissect-narrative-v1'

# Broad, well-established lineage markers for the local naming backup. Tuned for
# synovial tissue but generally useful; override via ``naming_markers``.
DEFAULT_MARKER_SETS = {
    'Fibroblast':        ['PRG4', 'THY1', 'PDPN', 'FAP', 'COL1A1', 'COL1A2', 'DCN', 'LUM', 'PDGFRA'],
    'Macrophage/Myeloid': ['CD68', 'CD14', 'LYZ', 'AIF1', 'CD163', 'C1QA', 'C1QB', 'FCGR3A'],
    'T cell':            ['CD3D', 'CD3E', 'CD3G', 'CD2', 'TRAC', 'CD8A', 'CD4', 'IL7R'],
    'NK cell':           ['NKG7', 'GNLY', 'KLRD1', 'NCAM1', 'KLRF1'],
    'B cell':            ['CD79A', 'CD79B', 'MS4A1', 'CD19', 'BANK1'],
    'Plasma cell':       ['MZB1', 'IGHG1', 'JCHAIN', 'XBP1', 'SDC1', 'DERL3'],
    'Endothelial':       ['PECAM1', 'VWF', 'CLDN5', 'CDH5', 'EGFL7'],
    'Mural/Pericyte':    ['ACTA2', 'RGS5', 'MYH11', 'PDGFRB', 'NOTCH3'],
    'Dendritic cell':    ['FCER1A', 'CLEC10A', 'CD1C', 'LILRA4'],
    'Mast cell':         ['TPSAB1', 'TPSB2', 'CPA3', 'MS4A2'],
}


def _num(value):
    """JSON-safe float, or None for NaN/missing/non-numeric."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:                          # NaN
        return None
    return f


def _read_tsv(path):
    try:
        return pd.read_csv(path, sep='\t')
    except Exception:
        return pd.DataFrame()


def load_marker_sets(markers=None) -> dict:
    """Resolve the naming marker table.

    ``None`` -> a copy of ``DEFAULT_MARKER_SETS``; a ``dict[cell_type -> genes]``
    -> normalized copy; a path/str -> a 2-column TSV ``cell_type<TAB>gene,gene,...``
    (no header).
    """
    if markers is None:
        return {k: list(v) for k, v in DEFAULT_MARKER_SETS.items()}
    if isinstance(markers, dict):
        return {str(k): [str(g) for g in v] for k, v in markers.items()}
    out: dict = {}
    df = pd.read_csv(markers, sep='\t', header=None)
    for _, row in df.iterrows():
        cell_type = str(row.iloc[0]).strip()
        genes = [g.strip() for g in str(row.iloc[1]).split(',') if g.strip()]
        if cell_type and genes:
            out[cell_type] = genes
    return out


@dataclass
class CoreEvidence:
    """Compact, serializable evidence for one canonical core (``c{N}_0``)."""

    parent_cluster: str
    core_subcluster: str
    n_cells: int = 0
    top_markers: list[dict] = field(default_factory=list)   # {gene, logfoldchanges, scores}
    hint: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    def marker_genes(self) -> list[str]:
        return [str(m.get('gene')) for m in self.top_markers if m.get('gene')]


@dataclass
class CoreNaming:
    """Stable naming output written to core_names.tsv and naming.output.json."""

    cell_type: str | None = None
    confidence: float = 0.0
    rationale: str = ''
    markers_used: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    source: str = 'unnamed'             # 'llm' | 'local' | 'unnamed'
    model: str | None = None
    prompt_version: str = NAMING_PROMPT_VERSION
    error: str | None = None

    def __post_init__(self):
        self.confidence = float(min(1.0, max(0.0, self.confidence)))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_core_name_row(self, evidence: CoreEvidence) -> dict:
        return {
            'parent_cluster': evidence.parent_cluster,
            'core_subcluster': evidence.core_subcluster,
            'cell_type': self.cell_type,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'source': self.source,
            'model': self.model,
        }


def build_core_evidence(parent_cluster, markers_path, *, n_cells=0, hint='',
                        top_n=20) -> CoreEvidence:
    """Build core evidence from ``markers_c{N}_0.tsv`` (top up-regulated by score)."""
    top_markers: list[dict] = []
    df = _read_tsv(markers_path)
    gene_col = 'gene' if 'gene' in df.columns else ('names' if 'names' in df.columns else None)
    if len(df) and gene_col and 'scores' in df.columns:
        up = df.copy()
        if 'logfoldchanges' in up.columns:
            up = up[up['logfoldchanges'] > 0]
        up = up.sort_values('scores', ascending=False).head(top_n)
        for _, r in up.iterrows():
            top_markers.append({
                'gene': str(r[gene_col]),
                'logfoldchanges': _num(r.get('logfoldchanges')),
                'scores': _num(r.get('scores')),
            })
    return CoreEvidence(
        parent_cluster=str(parent_cluster),
        core_subcluster=f"c{parent_cluster}_0",
        n_cells=int(n_cells or 0),
        top_markers=top_markers,
        hint=str(hint or ''),
    )


class LocalNamingEngine:
    """Backup namer: overlap of the core's top markers against a marker table.

    Score = Szymkiewicz-Simpson overlap coefficient
    ``|core ∩ type| / min(|core|, |type|)``; the highest-scoring type (then
    highest raw overlap count) wins. No network, no new dependency.
    """

    source = 'local'
    model = None

    def __init__(self, markers=None, *, min_overlap=1):
        self.markers = load_marker_sets(markers)
        self.min_overlap = min_overlap

    def name(self, evidence: CoreEvidence) -> CoreNaming:
        genes = {g.upper() for g in evidence.marker_genes()}
        if not genes or not self.markers:
            return CoreNaming(cell_type=None, source='unnamed',
                              rationale='no markers available for local overlap')
        best = None                     # (coef, n, cell_type, sorted_overlap)
        for cell_type, mset in self.markers.items():
            ref = {g.upper() for g in mset}
            if not ref:
                continue
            inter = genes & ref
            n = len(inter)
            if n == 0:
                continue
            coef = n / min(len(genes), len(ref))
            cand = (coef, n, cell_type, sorted(inter))
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best is None or best[1] < self.min_overlap:
            return CoreNaming(cell_type=None, source='unnamed',
                              rationale='no marker set overlapped the core markers')
        coef, n, cell_type, inter = best
        return CoreNaming(
            cell_type=cell_type, confidence=coef,
            rationale=(f"local marker overlap: {n} of the core top markers match "
                       f"{cell_type} ({', '.join(inter)})"),
            markers_used=inter, source='local', model=None)
