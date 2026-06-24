"""Diagnosis engines for standissect minor fragments.

This module owns the interpretation layer. Upstream code computes compact
evidence packets from DEG, composition drift, and QC drift; diagnosis engines
turn those packets into a stable ``likely_cause`` plus audit fields.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re

try:                                    # package use (standissect.diagnosis)
    from .llm_client import OpenAICompatClient, call_structured, extract_json, LLMUnavailable
except ImportError:                     # standalone use (tests import top-level `diagnosis`)
    from llm_client import OpenAICompatClient, call_structured, extract_json, LLMUnavailable

import numpy as np
import pandas as pd


ALLOWED_CAUSES = (
    'sample-driven',
    'doublet-driven',
    'low-quality (high mt)',
    'shallow-depth',
    'biology-candidate',
    'unclear',
)

PROMPT_VERSION = 'standissect-diagnosis-v1'
DEFAULT_ARK_ENDPOINT = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'
DEFAULT_ARK_MODEL = 'ep-20260412124039-zjq7v'
DEFAULT_DIAGNOSIS_ROLES = {
    'source_cols': (),
    'doublet_score_col': None,
    'mitochondrial_col': None,
    'feature_count_col': None,
    'umi_count_col': None,
}


def _as_tuple(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(v for v in value if v)


def normalize_diagnosis_roles(roles=None, *, use_defaults=True) -> dict:
    """Normalize semantic column roles used by deterministic diagnosis rules."""
    merged = dict(DEFAULT_DIAGNOSIS_ROLES) if use_defaults else {}
    if roles:
        merged.update(dict(roles))
    merged['source_cols'] = _as_tuple(merged.get('source_cols'))
    for key in ('doublet_score_col', 'mitochondrial_col',
                'feature_count_col', 'umi_count_col'):
        value = merged.get(key)
        merged[key] = str(value) if value else None
    return merged


@dataclass
class MinorEvidence:
    """Compact, serializable evidence for one minor fragment."""

    parent_cluster: str
    subcluster: str
    reference_subcluster: str
    minor_umap_label: str
    main_umap_label: str
    n_cells: int
    frac_of_parent: float
    top_up_genes: list[dict] = field(default_factory=list)
    top_down_genes: list[dict] = field(default_factory=list)
    n_sig_genes: int = 0
    composition_enrichment: list[dict] = field(default_factory=list)
    qc_drift: list[dict] = field(default_factory=list)
    major_core_comparisons: list[dict] = field(default_factory=list)
    diagnosis_roles: dict = field(default_factory=dict)
    rule_baseline: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiagnosisResult:
    """Stable diagnosis output written to panel TSVs and JSON audit files."""

    likely_cause: str
    cause_detail: str | None = None
    confidence: float = 0.0
    rationale: str = ''
    evidence_used: list[str] = field(default_factory=list)
    alternative_causes: list[str] = field(default_factory=list)
    recommended_checks: list[str] = field(default_factory=list)
    rule_baseline: str | None = None
    llm_overrode_rule: bool = False
    diagnosis_source: str = 'rule'
    diagnosis_mode: str = 'rule'
    model: str | None = None
    prompt_version: str = PROMPT_VERSION
    error: str | None = None

    def __post_init__(self):
        if self.likely_cause not in ALLOWED_CAUSES:
            raise ValueError(
                f"likely_cause must be one of {ALLOWED_CAUSES}, "
                f"got {self.likely_cause!r}"
            )
        self.confidence = float(min(1.0, max(0.0, self.confidence)))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_panel_fields(self) -> dict:
        return {
            'rule_baseline': self.rule_baseline,
            'likely_cause': self.likely_cause,
            'cause_detail': self.cause_detail,
            'diagnosis_confidence': self.confidence,
            'diagnosis_rationale': self.rationale,
            'llm_overrode_rule': self.llm_overrode_rule,
        }


def _json_safe(value):
    """Convert pandas/numpy scalars and NaN values into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _clean_record(record: dict) -> dict:
    return {str(k): _json_safe(v) for k, v in record.items()}


def _records(df: pd.DataFrame | None, *, max_rows: int = 10) -> list[dict]:
    if df is None or not len(df):
        return []
    return [_clean_record(r) for r in df.head(max_rows).to_dict(orient='records')]


def _deg_records(
    deg_df: pd.DataFrame | None,
    *,
    direction: str,
    max_rows: int = 8,
) -> list[dict]:
    if deg_df is None or not len(deg_df):
        return []
    name_col = 'names' if 'names' in deg_df.columns else 'gene'
    cols = [c for c in (name_col, 'logfoldchanges', 'pvals_adj', 'scores')
            if c in deg_df.columns]
    if direction == 'up':
        sub = (deg_df[deg_df['logfoldchanges'] > 0]
               .sort_values('scores', ascending=False))
    else:
        sub = (deg_df[deg_df['logfoldchanges'] < 0]
               .sort_values('scores', ascending=True))
    out = sub[cols].head(max_rows).rename(columns={name_col: 'gene'})
    return _records(out, max_rows=max_rows)


def _composition_records(composition_frames) -> list[dict]:
    frames = []
    if isinstance(composition_frames, dict):
        iterable = composition_frames.items()
    else:
        iterable = enumerate(composition_frames or [])
    for key, df in iterable:
        if df is None or not len(df):
            continue
        tmp = df.copy()
        if 'cat_col' not in tmp.columns:
            tmp['cat_col'] = str(key)
        frames.append(tmp)
    if not frames:
        return []
    comp = pd.concat(frames, ignore_index=True)
    if 'padj' in comp.columns:
        if 'log2_OR' in comp.columns:
            sig_pos = comp[(comp['padj'] < 0.05) & (comp['log2_OR'] > 0)]
            rest = comp.drop(index=sig_pos.index)
            comp = pd.concat([
                sig_pos.sort_values('log2_OR', ascending=False),
                rest.sort_values(['padj', 'log2_OR'], ascending=[True, False]),
            ], ignore_index=True)
        else:
            comp = comp.sort_values('padj', ascending=True)
    return _records(comp, max_rows=40)


def _qc_records(qc_df: pd.DataFrame | None) -> list[dict]:
    if qc_df is None or not len(qc_df):
        return []
    tmp = qc_df.copy()
    if 'relative_delta' in tmp.columns:
        tmp['_absrel'] = tmp['relative_delta'].abs()
    else:
        tmp['_absrel'] = 0.0
    sort_cols = [c for c in ('padj', '_absrel') if c in tmp.columns]
    if sort_cols:
        tmp = tmp.sort_values(sort_cols, ascending=[True, False][:len(sort_cols)])
    tmp = tmp.drop(columns=['_absrel'], errors='ignore')
    return _records(tmp, max_rows=8)


def build_minor_evidence(
    panel_row: dict | pd.Series,
    *,
    deg_df: pd.DataFrame | None = None,
    qc_df: pd.DataFrame | None = None,
    composition_frames=None,
    major_core_comparisons=None,
    diagnosis_roles=None,
) -> MinorEvidence:
    """Build a compact evidence object from persisted per-minor artifacts."""
    if isinstance(panel_row, pd.Series):
        row = panel_row.to_dict()
    else:
        row = dict(panel_row)
    n_sig = int(row.get('n_sig_genes') or 0)
    evidence = MinorEvidence(
        parent_cluster=str(row.get('parent_cluster')),
        subcluster=str(row.get('subcluster')),
        reference_subcluster=str(
            row.get('reference_subcluster') or f"c{row.get('parent_cluster')}_0"
        ),
        minor_umap_label=str(row.get('minor_umap_label')),
        main_umap_label=str(row.get('main_umap_label')),
        n_cells=int(row.get('n_cells') or 0),
        frac_of_parent=float(row.get('frac_of_parent') or 0.0),
        top_up_genes=_deg_records(deg_df, direction='up'),
        top_down_genes=_deg_records(deg_df, direction='down'),
        n_sig_genes=n_sig,
        composition_enrichment=_composition_records(composition_frames),
        qc_drift=_qc_records(qc_df),
        major_core_comparisons=list(major_core_comparisons or []),
        diagnosis_roles=normalize_diagnosis_roles(diagnosis_roles),
    )
    evidence.rule_baseline = RuleDiagnosisEngine(
        evidence.diagnosis_roles).diagnose(evidence).likely_cause
    return evidence


class RuleDiagnosisEngine:
    """Deterministic baseline that preserves the original diagnosis rules."""

    mode = 'rule'
    model = None

    def __init__(self, diagnosis_roles=None):
        self.roles = normalize_diagnosis_roles(diagnosis_roles)

    def diagnose(self, evidence: MinorEvidence) -> DiagnosisResult:
        top_source = self._top_source_enrichment(evidence)
        cause = 'unclear'
        detail = None
        rationale = 'No strong sample, QC, or DEG signal met the rule threshold.'
        confidence = 0.35
        used = []

        if top_source and top_source.get('padj', 1) < 0.05 and top_source.get('log2_OR', 0) >= 2:
            col = top_source.get('cat_col')
            category = top_source.get('category')
            cause = 'sample-driven'
            detail = f"enriched for {col}={category}"
            rationale = (
                f"{col} category {category} is enriched "
                f"(log2OR={top_source.get('log2_OR'):.2f}, "
                f"q={top_source.get('padj'):.1e})."
            )
            confidence = 0.9
            used.append(f'{col} enrichment')

        if cause == 'unclear':
            row = self._role_qc_row(
                evidence, self.roles.get('doublet_score_col'),
                require_positive_rel=True)
            if row and row.get('relative_delta', 0) > 0.5:
                cause = 'doublet-driven'
                detail = f"{row.get('qc_col')} increased"
                confidence = 0.85
                rationale = self._qc_rationale(row)
                used.append(f"{row.get('qc_col')} drift")

        if cause == 'unclear':
            row = self._role_qc_row(
                evidence, self.roles.get('mitochondrial_col'),
                require_positive_delta=True)
            if row and row.get('delta', 0) > 2:
                cause = 'low-quality (high mt)'
                detail = f"{row.get('qc_col')} increased"
                confidence = 0.85
                rationale = self._qc_rationale(row)
                used.append(f"{row.get('qc_col')} drift")

        if cause == 'unclear':
            row = self._depth_qc_row(evidence)
            if row and row.get('relative_delta', 0) < -0.3:
                cause = 'shallow-depth'
                detail = f"{row.get('qc_col')} decreased"
                confidence = 0.85
                rationale = self._qc_rationale(row)
                used.append(f"{row.get('qc_col')} drift")

        if cause == 'unclear' and evidence.n_sig_genes >= 20:
            cause = 'biology-candidate'
            detail = 'many significant DEGs without rule-matched artifact signal'
            rationale = (
                f"{evidence.n_sig_genes} significant DEGs separate the minor "
                "from the core, without a stronger sample or QC artifact rule."
            )
            confidence = 0.65
            used.append('significant DEG count')

        return DiagnosisResult(
            likely_cause=cause,
            cause_detail=detail,
            confidence=confidence,
            rationale=rationale,
            evidence_used=used,
            rule_baseline=cause,
            llm_overrode_rule=False,
            diagnosis_source='rule',
            diagnosis_mode='rule',
        )

    def _top_source_enrichment(self, evidence: MinorEvidence) -> dict | None:
        source_cols = set(self.roles.get('source_cols') or ())
        candidates = [
            r for r in evidence.composition_enrichment
            if r.get('cat_col') in source_cols
            and (r.get('padj') is not None and r.get('padj') < 0.05)
            and (r.get('log2_OR') is not None and r.get('log2_OR') > 0)
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda r: r.get('log2_OR', 0), reverse=True)[0]

    def _role_qc_row(
        self,
        evidence: MinorEvidence,
        col: str | None,
        *,
        require_positive_delta=False,
        require_positive_rel=False,
    ) -> dict | None:
        if not col:
            return None
        rows = [
            r for r in evidence.qc_drift
            if r.get('padj') is not None and r.get('padj') < 0.05
            and r.get('qc_col') == col
        ]
        if require_positive_delta:
            rows = [r for r in rows if r.get('delta', 0) > 0]
        if require_positive_rel:
            rows = [r for r in rows if r.get('relative_delta', 0) > 0]
        if not rows:
            return None
        return sorted(rows, key=lambda r: abs(r.get('relative_delta') or 0), reverse=True)[0]

    def _depth_qc_row(self, evidence: MinorEvidence) -> dict | None:
        cols = [
            c for c in (
                self.roles.get('feature_count_col'),
                self.roles.get('umi_count_col'),
            )
            if c
        ]
        rows = [
            r for r in evidence.qc_drift
            if r.get('padj') is not None and r.get('padj') < 0.05
            and r.get('qc_col') in cols
            and r.get('relative_delta', 0) < 0
        ]
        if not rows:
            return None
        return sorted(rows, key=lambda r: r.get('relative_delta', 0))[0]

    @staticmethod
    def _qc_rationale(row: dict) -> str:
        return (
            f"{row.get('qc_col')} drift is significant "
            f"(delta={row.get('delta'):+.2f}, "
            f"relative_delta={row.get('relative_delta'):+.2f}, "
            f"q={row.get('padj'):.1e})."
        )


class CallableChatClient:
    """Wrap a user-provided callable as a small chat client."""

    def __init__(self, fn, *, model: str | None = None):
        self.fn = fn
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.fn(system_prompt, user_prompt)


class ArkChatClient(OpenAICompatClient):
    """Volcengine ARK chat client — a thin OpenAICompatClient with ARK defaults
    (temperature=0). `endpoint` may be a full /chat/completions URL."""

    def __init__(self, *, api_key, model=DEFAULT_ARK_MODEL,
                 endpoint=DEFAULT_ARK_ENDPOINT, timeout=60):
        if not api_key:
            raise ValueError("ArkChatClient requires a non-empty api_key")
        super().__init__(endpoint, api_key, model, timeout=timeout, temperature=0)

    @classmethod
    def from_env(cls, *, api_key_env="ARK_API_KEY", model=DEFAULT_ARK_MODEL,
                 endpoint=DEFAULT_ARK_ENDPOINT, timeout=60):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"environment variable {api_key_env!r} is not set")
        return cls(api_key=api_key, model=model, endpoint=endpoint, timeout=timeout)


class LLMDiagnosisEngine:
    """LLM interpretation over a compact evidence packet."""

    mode = 'llm'

    def __init__(self, client, *, mode='llm', fallback_to_rule=True,
                 diagnosis_roles=None):
        self.client = client
        self.mode = mode
        self.fallback_to_rule = fallback_to_rule
        self.diagnosis_roles = normalize_diagnosis_roles(diagnosis_roles)
        self.rule_engine = RuleDiagnosisEngine(self.diagnosis_roles)
        self.model = getattr(client, 'model', None)

    def diagnose(self, evidence: MinorEvidence) -> DiagnosisResult:
        if not evidence.diagnosis_roles:
            evidence.diagnosis_roles = self.diagnosis_roles
        baseline = self.rule_engine.diagnose(evidence)
        evidence.rule_baseline = baseline.likely_cause if self.mode == 'hybrid' else None
        system_prompt, user_prompt = build_llm_prompt(evidence, mode=self.mode)
        try:
            return call_structured(
                self.client, system_prompt, user_prompt,
                lambda data: _diagnosis_from_dict(
                    data, rule_baseline=evidence.rule_baseline,
                    mode=self.mode, model=self.model))
        except Exception as e:
            if not self.fallback_to_rule:
                raise
            fallback = baseline
            fallback.diagnosis_source = 'rule-fallback'
            fallback.diagnosis_mode = self.mode
            fallback.model = self.model
            fallback.error = str(e)
            fallback.rationale = (
                fallback.rationale + f" LLM diagnosis failed; used rule baseline. Error: {e}"
            )
            return fallback


def build_llm_prompt(evidence: MinorEvidence, *, mode='hybrid') -> tuple[str, str]:
    schema = {
        'likely_cause': list(ALLOWED_CAUSES),
        'cause_detail': 'short phrase or null',
        'confidence': 'number from 0 to 1',
        'rationale': 'one concise paragraph, using only supplied evidence',
        'evidence_used': ['short evidence labels'],
        'alternative_causes': ['plausible alternatives'],
        'recommended_checks': ['specific follow-up checks'],
        'llm_overrode_rule': 'boolean',
    }
    system = (
        "You diagnose minor fragments inside single-cell clusters. "
        "Use only the supplied statistical evidence. Do not invent measurements, "
        "cell types, markers, or experiments. Choose exactly one likely_cause "
        "from the allowed enum. Return strict JSON only."
    )
    user = json.dumps({
        'task': 'diagnose_one_minor_fragment',
        'diagnosis_mode': mode,
        'allowed_likely_cause': list(ALLOWED_CAUSES),
        'output_schema': schema,
        'evidence': evidence.to_dict(),
    }, ensure_ascii=False, indent=2)
    return system, user


def _diagnosis_from_dict(data, *, rule_baseline, mode, model) -> DiagnosisResult:
    cause = data.get('likely_cause')
    if cause not in ALLOWED_CAUSES:
        raise ValueError(f"LLM likely_cause {cause!r} is not allowed")
    return DiagnosisResult(
        likely_cause=cause,
        cause_detail=data.get('cause_detail'),
        confidence=float(data.get('confidence', 0.0)),
        rationale=str(data.get('rationale', '')),
        evidence_used=list(data.get('evidence_used') or []),
        alternative_causes=list(data.get('alternative_causes') or []),
        recommended_checks=list(data.get('recommended_checks') or []),
        rule_baseline=rule_baseline,
        llm_overrode_rule=bool(data.get(
            'llm_overrode_rule',
            rule_baseline is not None and cause != rule_baseline,
        )),
        diagnosis_source='llm',
        diagnosis_mode=mode,
        model=model,
    )


def parse_llm_result(raw, *, rule_baseline, mode, model) -> DiagnosisResult:
    """Parse raw LLM text into a DiagnosisResult, using the shared JSON
    extractor (tolerates code fences / surrounding prose)."""
    data = json.loads(extract_json(raw))
    return _diagnosis_from_dict(data, rule_baseline=rule_baseline, mode=mode, model=model)


def make_diagnosis_engine(
    *,
    mode: str = 'rule',
    llm_client=None,
    ark_api_key: str | None = None,
    ark_api_key_env: str = 'ARK_API_KEY',
    ark_model: str = DEFAULT_ARK_MODEL,
    ark_endpoint: str = DEFAULT_ARK_ENDPOINT,
    timeout: int = 60,
    fallback_to_rule: bool = True,
    diagnosis_roles=None,
):
    """Create the requested diagnosis engine.

    ``llm_client`` may be an object with ``complete(system, user)`` or a callable
    with that same signature. Without ``llm_client``, LLM modes create an Ark
    client from ``ARK_API_KEY``.
    """
    mode = str(mode)
    if mode not in {'rule', 'llm', 'hybrid'}:
        raise ValueError("diagnosis_mode must be 'rule', 'llm', or 'hybrid'")
    if mode == 'rule':
        return RuleDiagnosisEngine(diagnosis_roles)
    if llm_client is None:
        api_key = ark_api_key if ark_api_key is not None else os.environ.get(ark_api_key_env)
        if not api_key:
            raise ValueError(
                f"diagnosis_mode={mode!r} requires diagnosis_llm_client, "
                f"diagnosis_ark_api_key, or environment variable {ark_api_key_env!r}"
            )
        llm_client = ArkChatClient(
            api_key=api_key,
            model=ark_model,
            endpoint=ark_endpoint,
            timeout=timeout,
        )
    elif not hasattr(llm_client, 'complete'):
        llm_client = CallableChatClient(llm_client, model=ark_model)
    return LLMDiagnosisEngine(
        llm_client, mode=mode, fallback_to_rule=fallback_to_rule,
        diagnosis_roles=diagnosis_roles)


def safe_subcluster_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name))


def write_diagnosis_artifacts(
    cdir,
    evidence: MinorEvidence,
    result: DiagnosisResult,
):
    """Persist per-minor diagnosis input/output JSON for auditability."""
    cdir = Path(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    stem = safe_subcluster_name(evidence.subcluster)
    (cdir / f"diagnosis_{stem}.input.json").write_text(
        json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (cdir / f"diagnosis_{stem}.output.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
