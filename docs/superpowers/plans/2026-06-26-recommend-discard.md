# recommend-discard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two complementary, fully-automated per-fragment recommendations: (1) a `recommended_disposition` (DISCARD/KEEP/UNCERTAIN) surfaced in `panel.tsv`, `cell_labels.tsv`, a new cell-precise `discard_cells.tsv`, and a report section; (2) **LLM-proposed cell types** — finer/different identities for minors (diagnosis) and major cores (naming) vs the original annotation — collected in a new `proposed_cell_types.tsv` + report section.

**Architecture:** A deterministic `DISPOSITION_MAP` turns each `likely_cause` into a baseline disposition; the LLM may relax it toward KEEP (never escalate toward DISCARD); a confidence gate downgrades low-confidence DISCARDs to UNCERTAIN. Disposition + minor `proposed_cell_type` live in `diagnosis.py` (threshold held by the engine, finalized inside `diagnose()`); the major relabel flag lives in `annotate.py` naming; `pipeline.py` writes the columns/files (`discard_cells.tsv` keyed by barcode + input-adata row index; `proposed_cell_types.tsv`); `report.py` renders both sections; `cli.py` exposes the threshold.

**Tech Stack:** Python stdlib + numpy + pandas (no new deps). LLM via the existing vendored `llm_client.py` (unchanged). Tests: pytest.

## Global Constraints

- Work ONLY in the canonical project `/scratch/users/chensj16/projects/standissect` — NEVER the synovial copy.
- Branch `feat/recommend-discard` (already created off `main`). Do NOT merge without user OK. Per-task commits.
- Tests run LOCALLY on this compute node. NO `srun`/`sbatch`/Slurm.
- stdlib-only; NO new pip dependencies; do NOT modify the vendored `llm_client.py`.
- Disposition values are UPPERCASE `DISCARD`/`KEEP`/`UNCERTAIN`; `likely_cause` stays lowercase kebab-case.
- 11-cause taxonomy locked; the 5 new causes (`dissociation-effect`, `cell-cycle`, `ambient-contamination`, `sex-driven`, `interferon-response`) are **LLM-only** — `RuleDiagnosisEngine.diagnose` is NOT extended.
- **Conservative-only invariant:** LLM override + confidence gate may only move a disposition toward KEEP (`DISCARD`→`UNCERTAIN`→`KEEP`), never toward `DISCARD`.
- `proposed_cell_type` (minor) / `differs_from_original` (major) are **orthogonal to disposition** — real biology stays KEEP. Both compare against `parent_cluster` (which IS the original `cluster_col` label).
- `discard_cells.tsv` keyed by `barcode` (= `obs_name`, the stable cross-version key) + `input_row_index` (0-based position in the standissect-input adata, from `adata.obs_names`).
- Prompt-version bumps: `PROMPT_VERSION` → `standissect-diagnosis-v2`; `NAMING_PROMPT_VERSION` → `standissect-naming-v2`.
- Spec: `docs/superpowers/specs/2026-06-26-recommend-discard-design.md`.

## File Structure

- `diagnosis.py` — taxonomy, `DISPOSITION_MAP`, `derive_disposition`, `DiagnosisResult` disposition fields + `proposed_cell_type` + `finalize_disposition`, engine wiring, LLM schema/prompt. (Tasks 1, 2)
- `pipeline.py` — `_PANEL_COLS`/`_DIAGNOSIS_COLS`, `run_dissect_pipeline` threshold + `params.json`, `_Layout.{discard_cells,proposed_cell_types}`, `_write_cell_dispositions`, `_write_proposed_cell_types`. (Tasks 3, 4, 7)
- `annotate.py` — `CoreNaming.differs_from_original` + naming schema/prompt/parse + `run_naming_stage` rows + `CORE_NAME_COLS`. (Task 6)
- `report.py` — `_discards_section` + `_proposed_types_section` + sidebar anchors. (Tasks 5, 7)
- `cli.py` — `--discard-confidence-threshold`. (Task 3)
- Tests: `tests/test_disposition.py` (new), `tests/test_diagnosis_llm.py`, `tests/test_cli.py`, `tests/test_discard_outputs.py` (new), `tests/test_report.py`, `tests/test_annotate.py`.

**Test import idioms (match existing):** scanpy-free modules (`diagnosis`, `annotate`, `report`) import top-level after `sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))`. Modules pulling the package (`standissect.cli`, `standissect.pipeline`) use `parents[2]` + `from standissect.X import ...` (loads scanpy — fine on this compute node).

---

### Task 1: Disposition core + proposed_cell_type field in `diagnosis.py`

**Files:**
- Modify: `/scratch/users/chensj16/projects/standissect/diagnosis.py` (`ALLOWED_CAUSES` L29-36; `DiagnosisResult` L95-133)
- Test: `/scratch/users/chensj16/projects/standissect/tests/test_disposition.py` (create)

**Interfaces — Produces:** `ALLOWED_CAUSES` (11); `DISPOSITION_MAP`; `_DISPOSITION_RANK`; `derive_disposition(likely_cause, confidence, *, threshold, llm_disposition=None, llm_reason=None) -> (recommended, baseline, overridden, reason)`; `DiagnosisResult` gains `disposition_baseline`, `recommended_disposition`, `disposition_overridden`, `disposition_reason`, `proposed_cell_type` + `finalize_disposition(threshold, *, llm_disposition=None, llm_reason=None)`; `to_panel_fields()` emits the 4 disposition keys + `proposed_cell_type`.

- [ ] **Step 1: Write the failing test** — `tests/test_disposition.py`:

```python
import pathlib
import sys

_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_DIR))

import diagnosis  # noqa: E402
from diagnosis import (ALLOWED_CAUSES, DISPOSITION_MAP, derive_disposition,  # noqa: E402
                       DiagnosisResult)

NEW_CAUSES = {'dissociation-effect', 'cell-cycle', 'ambient-contamination',
              'sex-driven', 'interferon-response'}


def test_taxonomy_has_eleven_causes_including_new_ones():
    assert len(ALLOWED_CAUSES) == 11
    assert NEW_CAUSES.issubset(set(ALLOWED_CAUSES))


def test_disposition_map_covers_exactly_allowed_causes():
    assert set(DISPOSITION_MAP) == set(ALLOWED_CAUSES)
    assert set(DISPOSITION_MAP.values()) <= {'DISCARD', 'KEEP', 'UNCERTAIN'}


def test_baseline_mapping_per_cause():
    for c in ('doublet-driven', 'low-quality (high mt)', 'shallow-depth',
              'dissociation-effect', 'ambient-contamination'):
        assert DISPOSITION_MAP[c] == 'DISCARD'
    for c in ('cell-cycle', 'sex-driven', 'interferon-response', 'biology-candidate'):
        assert DISPOSITION_MAP[c] == 'KEEP'
    for c in ('sample-driven', 'unclear'):
        assert DISPOSITION_MAP[c] == 'UNCERTAIN'


def test_gate_downgrades_low_confidence_discard():
    assert derive_disposition('doublet-driven', 0.3, threshold=0.5)[:3] == (
        'UNCERTAIN', 'DISCARD', True)


def test_gate_keeps_high_confidence_discard():
    assert derive_disposition('doublet-driven', 0.9, threshold=0.5)[:3] == (
        'DISCARD', 'DISCARD', False)


def test_override_relax_toward_keep_is_accepted():
    assert derive_disposition('doublet-driven', 0.9, threshold=0.5,
                              llm_disposition='KEEP', llm_reason='real')[:3] == (
        'KEEP', 'DISCARD', True)


def test_override_escalate_toward_discard_is_rejected():
    final, baseline, overridden, reason = derive_disposition(
        'cell-cycle', 0.9, threshold=0.5, llm_disposition='DISCARD',
        llm_reason='looks junky')
    assert (final, baseline, overridden) == ('KEEP', 'KEEP', False)
    assert 'rejected' in reason.lower()


def test_uncertain_baseline_cannot_be_escalated_to_discard():
    assert derive_disposition('unclear', 0.9, threshold=0.5,
                              llm_disposition='DISCARD')[0] == 'UNCERTAIN'


def test_result_disposition_and_proposed_fields_round_trip():
    r = DiagnosisResult(likely_cause='cell-cycle', confidence=0.9,
                        proposed_cell_type='pDC')
    assert r.disposition_baseline == 'KEEP'
    assert r.recommended_disposition == 'KEEP'
    assert r.proposed_cell_type == 'pDC'
    pf = r.to_panel_fields()
    for k in ('recommended_disposition', 'disposition_baseline',
              'disposition_overridden', 'disposition_reason', 'proposed_cell_type'):
        assert k in pf
    assert pf['proposed_cell_type'] == 'pDC'


def test_finalize_applies_gate():
    r = DiagnosisResult(likely_cause='doublet-driven', confidence=0.2)
    r.finalize_disposition(0.5)
    assert r.recommended_disposition == 'UNCERTAIN'
    assert r.disposition_overridden is True
    assert r.proposed_cell_type is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_disposition.py -q`
Expected: FAIL — `ImportError: cannot import name 'DISPOSITION_MAP'`.

- [ ] **Step 3: Implement** — in `diagnosis.py`:

(a) Replace `ALLOWED_CAUSES` (L29-36) with the 11-value version + add the map/rank/function:

```python
ALLOWED_CAUSES = (
    'sample-driven',
    'doublet-driven',
    'low-quality (high mt)',
    'shallow-depth',
    'dissociation-effect',
    'cell-cycle',
    'ambient-contamination',
    'sex-driven',
    'interferon-response',
    'biology-candidate',
    'unclear',
)

DISPOSITION_MAP = {
    'sample-driven': 'UNCERTAIN',
    'doublet-driven': 'DISCARD',
    'low-quality (high mt)': 'DISCARD',
    'shallow-depth': 'DISCARD',
    'dissociation-effect': 'DISCARD',
    'cell-cycle': 'KEEP',
    'ambient-contamination': 'DISCARD',
    'sex-driven': 'KEEP',
    'interferon-response': 'KEEP',
    'biology-candidate': 'KEEP',
    'unclear': 'UNCERTAIN',
}

_DISPOSITION_RANK = {'DISCARD': 0, 'UNCERTAIN': 1, 'KEEP': 2}


def derive_disposition(likely_cause, confidence, *, threshold,
                       llm_disposition=None, llm_reason=None):
    """(recommended, baseline, overridden, reason). Conservative-only: an LLM
    pick is accepted only if at least as keep-leaning as the baseline; a DISCARD
    baseline below ``threshold`` confidence is downgraded to UNCERTAIN."""
    baseline = DISPOSITION_MAP[likely_cause]
    candidate = baseline
    reason = f"{likely_cause} → {baseline} (rule baseline)"
    if llm_disposition in _DISPOSITION_RANK:
        if _DISPOSITION_RANK[llm_disposition] >= _DISPOSITION_RANK[baseline]:
            candidate = llm_disposition
            reason = (str(llm_reason).strip() if llm_reason else '') or reason
        else:
            reason = (reason + f" [LLM suggested {llm_disposition}; rejected "
                      f"— cannot escalate toward DISCARD]")
    final = candidate
    if candidate == 'DISCARD' and confidence < threshold:
        final = 'UNCERTAIN'
        reason = reason + (f" (downgraded: low confidence "
                           f"{confidence:.2f} < {threshold:.2f})")
    return final, baseline, (final != baseline), reason
```

(b) `DiagnosisResult` (after `llm_overrode_rule` at L107) add five fields:

```python
    llm_overrode_rule: bool = False
    disposition_baseline: str = ''
    recommended_disposition: str = ''
    disposition_overridden: bool = False
    disposition_reason: str = ''
    proposed_cell_type: str | None = None
```

(c) Extend `__post_init__` (after the confidence clamp at L120):

```python
        self.confidence = float(min(1.0, max(0.0, self.confidence)))
        self.disposition_baseline = DISPOSITION_MAP[self.likely_cause]
        if not self.recommended_disposition:
            self.recommended_disposition = self.disposition_baseline
            if not self.disposition_reason:
                self.disposition_reason = (
                    f"{self.likely_cause} → "
                    f"{self.disposition_baseline} (rule baseline)")
```

(d) Add `finalize_disposition` (after `__post_init__`):

```python
    def finalize_disposition(self, threshold, *, llm_disposition=None,
                             llm_reason=None):
        final, baseline, overridden, reason = derive_disposition(
            self.likely_cause, self.confidence, threshold=threshold,
            llm_disposition=llm_disposition, llm_reason=llm_reason)
        self.disposition_baseline = baseline
        self.recommended_disposition = final
        self.disposition_overridden = overridden
        self.disposition_reason = reason
        return self
```

(e) Extend `to_panel_fields()` (L125-133):

```python
    def to_panel_fields(self) -> dict:
        return {
            'rule_baseline': self.rule_baseline,
            'likely_cause': self.likely_cause,
            'cause_detail': self.cause_detail,
            'diagnosis_confidence': self.confidence,
            'diagnosis_rationale': self.rationale,
            'llm_overrode_rule': self.llm_overrode_rule,
            'disposition_baseline': self.disposition_baseline,
            'recommended_disposition': self.recommended_disposition,
            'disposition_overridden': self.disposition_overridden,
            'disposition_reason': self.disposition_reason,
            'proposed_cell_type': self.proposed_cell_type,
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_disposition.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add diagnosis.py tests/test_disposition.py
git commit -m "feat(diagnosis): disposition map + conservative derive + result fields + proposed_cell_type"
```

---

### Task 2: Diagnosis engine wiring + LLM schema/prompt (disposition + proposed_cell_type)

**Files:**
- Modify: `/scratch/users/chensj16/projects/standissect/diagnosis.py` (`PROMPT_VERSION` L38; `RuleDiagnosisEngine.__init__` L272 + `.diagnose` return L337-347; `LLMDiagnosisEngine.__init__` L447-455 + `.diagnose` L457-471; `build_llm_prompt` L486-510; `_diagnosis_from_dict` L513-533; `make_diagnosis_engine` L573-609)
- Test: `/scratch/users/chensj16/projects/standissect/tests/test_diagnosis_llm.py` (extend)

**Interfaces — Consumes:** Task 1. **Produces:** `RuleDiagnosisEngine(diagnosis_roles=None, *, discard_confidence_threshold=0.5)`; `LLMDiagnosisEngine(..., discard_confidence_threshold=0.5)`; `_diagnosis_from_dict(data, *, rule_baseline, mode, model, threshold=0.5)`; `make_diagnosis_engine(..., discard_confidence_threshold=0.5)`; `CAUSE_SIGNATURES`, `DISPOSITION_POLICY`; `PROMPT_VERSION == 'standissect-diagnosis-v2'`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_diagnosis_llm.py`:

```python
def test_prompt_advertises_disposition_proposed_and_eleven_causes():
    from diagnosis import build_llm_prompt, MinorEvidence, ALLOWED_CAUSES
    ev = MinorEvidence(parent_cluster="0", subcluster="c0_1",
                       reference_subcluster="c0_0", minor_umap_label="u1",
                       main_umap_label="u0", n_cells=10, frac_of_parent=0.1)
    system, user = build_llm_prompt(ev, mode="llm")
    assert len(ALLOWED_CAUSES) == 11
    for key in ("recommended_disposition", "disposition_reason",
                "proposed_cell_type", "cause_signatures"):
        assert key in user
    assert "escalate" in (system + user).lower()


def test_rule_engine_result_carries_disposition_baseline():
    from diagnosis import RuleDiagnosisEngine, MinorEvidence
    eng = RuleDiagnosisEngine(discard_confidence_threshold=0.5)
    r = eng.diagnose(MinorEvidence(parent_cluster="0", subcluster="c0_1",
                     reference_subcluster="c0_0", minor_umap_label="u1",
                     main_umap_label="u0", n_cells=10, frac_of_parent=0.1))
    assert r.recommended_disposition == r.disposition_baseline
    assert r.disposition_overridden is False
    assert r.proposed_cell_type is None


def test_llm_disposition_parsed_and_clamped():
    payload = json.dumps({"likely_cause": "cell-cycle", "confidence": 0.9,
                          "rationale": "cycling", "recommended_disposition": "DISCARD",
                          "disposition_reason": "looks junky"})
    r = parse_llm_result(payload, rule_baseline=None, mode="llm", model=None)
    assert r.recommended_disposition == "KEEP"
    assert r.disposition_overridden is False


def test_llm_missing_disposition_falls_back_to_baseline():
    payload = json.dumps({"likely_cause": "doublet-driven", "confidence": 0.9,
                          "rationale": "doublets"})
    r = parse_llm_result(payload, rule_baseline=None, mode="llm", model=None)
    assert r.recommended_disposition == "DISCARD"
    assert r.proposed_cell_type is None


def test_llm_proposed_cell_type_parsed():
    payload = json.dumps({"likely_cause": "biology-candidate", "confidence": 0.8,
                          "rationale": "distinct program",
                          "proposed_cell_type": "pDC"})
    r = parse_llm_result(payload, rule_baseline=None, mode="llm", model=None)
    assert r.proposed_cell_type == "pDC"
    assert r.recommended_disposition == "KEEP"


def test_new_cause_dissociation_is_accepted_by_llm():
    payload = json.dumps({"likely_cause": "dissociation-effect", "confidence": 0.8,
                          "rationale": "HSP + IEG", "recommended_disposition": "DISCARD"})
    r = parse_llm_result(payload, rule_baseline=None, mode="llm", model=None)
    assert r.likely_cause == "dissociation-effect"
    assert r.recommended_disposition == "DISCARD"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_diagnosis_llm.py -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'discard_confidence_threshold'` / missing prompt keys.

- [ ] **Step 3: Implement** — in `diagnosis.py`:

(a) `PROMPT_VERSION` (L38):

```python
PROMPT_VERSION = 'standissect-diagnosis-v2'
```

(b) Module-level guidance before `build_llm_prompt`:

```python
CAUSE_SIGNATURES = {
    'sample-driven': 'fragment enriched for one sample/batch/donor (composition enrichment).',
    'doublet-driven': 'doublet score / UMI elevated; two distinct lineages co-expressed WITH high-UMI or doublet-score signal.',
    'low-quality (high mt)': 'elevated mitochondrial fraction (dying/broken cells).',
    'shallow-depth': 'low UMI/gene counts dominate the split.',
    'dissociation-effect': 'tissue-dissociation stress: immediate-early genes (AP-1: FOS/FOSB/JUN/JUNB/JUND, EGR1) and heat-shock proteins (HSPA1A/B, HSPB1, DNAJB1), SOCS3/ZFP36. Recognize species-appropriate orthologs (e.g. mouse Fos vs human FOS).',
    'cell-cycle': 'proliferation/cell-cycle genes (MKI67, TOP2A, CCNB1/2, CDK1, PCNA, CENPF, UBE2C, histones); real cells split by cycle phase.',
    'ambient-contamination': 'contaminant transcripts not native to the cluster: hemoglobin (HBA/HBB) from RBC ambient, OR a DIFFERENT compartment’s markers (epithelial EPCAM/keratins or stromal COL1A1/PDGFRB in an immune cluster) appearing diffusely WITHOUT doublet/UMI signal.',
    'sex-driven': 'sex-linked genes: XIST (female); Y genes (RPS4Y1/DDX3Y/UTY/EIF1AY; mouse Ddx3y/Uty/Eif2s3y/Kdm5d).',
    'interferon-response': 'interferon-stimulated genes: ISG15, IFIT1/2/3, MX1/2, OAS family, STAT1, IRF7, RSAD2, IFITM3.',
    'biology-candidate': 'many significant DEGs forming a coherent cell-type/state program without an artifact signature.',
    'unclear': 'no signature meets confidence.',
}

DISPOSITION_POLICY = (
    "Each likely_cause has a default recommended_disposition: "
    "doublet-driven/low-quality (high mt)/shallow-depth/dissociation-effect/"
    "ambient-contamination -> DISCARD; cell-cycle/sex-driven/interferon-response/"
    "biology-candidate -> KEEP; sample-driven/unclear -> UNCERTAIN. You MAY relax "
    "recommended_disposition toward KEEP (DISCARD->UNCERTAIN->KEEP) when evidence "
    "supports keeping the cells, and MUST give disposition_reason. You may NOT "
    "escalate toward DISCARD via recommended_disposition; to mark a cluster as "
    "junk, pick a discard-type likely_cause instead. If the fragment is a real, "
    "distinct or finer cell type than its parent (parent_cluster), set "
    "proposed_cell_type to that cell-type name; otherwise null."
)
```

(c) `build_llm_prompt` (L486-510) — add three schema keys, mention the policy, add `cause_signatures` + `disposition_policy` to the payload:

```python
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
        'recommended_disposition': ['DISCARD', 'KEEP', 'UNCERTAIN'],
        'disposition_reason': 'short phrase justifying the disposition',
        'proposed_cell_type': 'specific cell-type name if a real distinct/finer type than parent, else null',
    }
    system = (
        "You diagnose minor fragments inside single-cell clusters. "
        "Use only the supplied statistical evidence. Do not invent measurements, "
        "cell types, markers, or experiments. Choose exactly one likely_cause "
        "from the allowed enum, matching species-appropriate gene orthologs. "
        "Set recommended_disposition per disposition_policy: relax toward KEEP "
        "but NEVER escalate toward DISCARD. Return strict JSON only."
    )
    user = json.dumps({
        'task': 'diagnose_one_minor_fragment',
        'diagnosis_mode': mode,
        'allowed_likely_cause': list(ALLOWED_CAUSES),
        'cause_signatures': CAUSE_SIGNATURES,
        'disposition_policy': DISPOSITION_POLICY,
        'output_schema': schema,
        'evidence': evidence.to_dict(),
    }, ensure_ascii=False, indent=2)
    return system, user
```

(d) `_diagnosis_from_dict` (L513-533) — add `threshold=0.5`, finalize disposition, set `proposed_cell_type`:

```python
def _diagnosis_from_dict(data, *, rule_baseline, mode, model, threshold=0.5) -> DiagnosisResult:
    cause = data.get('likely_cause')
    if cause not in ALLOWED_CAUSES:
        raise ValueError(f"LLM likely_cause {cause!r} is not allowed")
    result = DiagnosisResult(
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
    result.finalize_disposition(
        threshold,
        llm_disposition=data.get('recommended_disposition'),
        llm_reason=data.get('disposition_reason'))
    pct = data.get('proposed_cell_type')
    result.proposed_cell_type = str(pct).strip() if pct and str(pct).strip() else None
    return result
```

(e) `RuleDiagnosisEngine.__init__` (L272-273):

```python
    def __init__(self, diagnosis_roles=None, *, discard_confidence_threshold=0.5):
        self.roles = normalize_diagnosis_roles(diagnosis_roles)
        self.discard_confidence_threshold = discard_confidence_threshold
```

(f) `RuleDiagnosisEngine.diagnose` (L337-347) — finalize before returning:

```python
        result = DiagnosisResult(
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
        result.finalize_disposition(self.discard_confidence_threshold)
        return result
```

(g) `LLMDiagnosisEngine.__init__` (L447-455) — threshold + pass to rule engine:

```python
    def __init__(self, client, *, mode='llm', fallback_to_rule=True,
                 diagnosis_roles=None, llm_retries=3,
                 discard_confidence_threshold=0.5):
        self.client = client
        self.mode = mode
        self.fallback_to_rule = fallback_to_rule
        self.diagnosis_roles = normalize_diagnosis_roles(diagnosis_roles)
        self.discard_confidence_threshold = discard_confidence_threshold
        self.rule_engine = RuleDiagnosisEngine(
            self.diagnosis_roles,
            discard_confidence_threshold=discard_confidence_threshold)
        self.model = getattr(client, 'model', None)
        self.llm_retries = llm_retries
```

In `.diagnose`, the inner lambda (L467-469) gains `threshold=`:

```python
                    lambda data: _diagnosis_from_dict(
                        data, rule_baseline=evidence.rule_baseline,
                        mode=self.mode, model=self.model,
                        threshold=self.discard_confidence_threshold)),
```

(h) `make_diagnosis_engine` (L573-609) — add `discard_confidence_threshold: float = 0.5` to the signature and pass it to both engines:

```python
    if mode == 'rule':
        return RuleDiagnosisEngine(
            diagnosis_roles,
            discard_confidence_threshold=discard_confidence_threshold)
    ...
    return LLMDiagnosisEngine(
        client, mode=mode, fallback_to_rule=fallback_to_rule,
        diagnosis_roles=diagnosis_roles, llm_retries=llm_retries,
        discard_confidence_threshold=discard_confidence_threshold)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_diagnosis_llm.py tests/test_disposition.py -q`
Expected: PASS (existing + 6 new + Task 1). Existing `parse_llm_result`/`call_structured` tests still pass (`threshold` defaults to 0.5).

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add diagnosis.py tests/test_diagnosis_llm.py
git commit -m "feat(diagnosis): engine threshold + LLM disposition/proposed_cell_type schema (v2)"
```

---

### Task 3: Pipeline threshold param + CLI + params.json

**Files:**
- Modify: `pipeline.py` (`run_dissect_pipeline` signature L624-668; engine build L738-749; `params.json` L953-988)
- Modify: `cli.py` (diag arg group after L92; `run_cmd` near L158)
- Test: `tests/test_cli.py` (extend)

**Interfaces — Consumes:** `make_diagnosis_engine(..., discard_confidence_threshold=...)` (Task 2). **Produces:** `run_dissect_pipeline(..., discard_confidence_threshold=0.5)`; CLI `--discard-confidence-threshold` → `args.discard_confidence_threshold`; `params.json['discard_confidence_threshold']`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_cli.py`:

```python
def test_cli_discard_threshold_default():
    a = build_parser().parse_args(["run", "x.h5ad", "--cluster-col", "leiden",
                                   "--output-dir", "o"])
    assert a.discard_confidence_threshold == 0.5


def test_cli_discard_threshold_override():
    a = build_parser().parse_args(["run", "x.h5ad", "--cluster-col", "leiden",
        "--output-dir", "o", "--discard-confidence-threshold", "0.75"])
    assert a.discard_confidence_threshold == 0.75
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_cli.py -q`
Expected: FAIL — `AttributeError: ... 'discard_confidence_threshold'`.

- [ ] **Step 3: Implement**

(a) `cli.py` — add the flag after `--ark-timeout` (L92):

```python
    diag.add_argument('--discard-confidence-threshold', type=float, default=0.5,
                      help='DISCARD calls below this diagnosis confidence are '
                           'downgraded to UNCERTAIN (kept + flagged). Default: 0.5.')
```

(b) `cli.py` — pass it in `run_cmd` (near L158):

```python
        diagnosis_timeout=args.ark_timeout,
        discard_confidence_threshold=args.discard_confidence_threshold,
        random_state=args.random_state,
```

(c) `pipeline.py` — add the param to `run_dissect_pipeline` (after `llm_retries=3,` L666):

```python
    llm_retries=3,
    discard_confidence_threshold=0.5,
    random_state=0,
```

(d) `pipeline.py` — pass to `make_diagnosis_engine` (L738-749):

```python
        diagnosis_roles=resolved_roles,
        llm_retries=llm_retries,
        discard_confidence_threshold=discard_confidence_threshold,
    )
```

(e) `pipeline.py` — record in `params.json` (near L986):

```python
        'diagnosis_timeout': diagnosis_timeout,
        'discard_confidence_threshold': discard_confidence_threshold,
        'partition_info': partition_info,
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_cli.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add pipeline.py cli.py tests/test_cli.py
git commit -m "feat(pipeline,cli): --discard-confidence-threshold + params.json"
```

---

### Task 4: panel/diagnosis columns + per-cell disposition + cell-precise discard_cells.tsv

**Files:**
- Modify: `pipeline.py` (`_PANEL_COLS` L50-55; `_DIAGNOSIS_COLS` L57-61; `_Layout` L64-95; add `_write_cell_dispositions` near L158; terminal aggregation L877)
- Test: `tests/test_discard_outputs.py` (create)

**Interfaces — Consumes:** a global `panel.tsv` carrying the disposition columns + `proposed_cell_type` (Tasks 2–3); `cell_labels.tsv` (index = barcode, column `original_cluster_split` = `c{parent}_{rank}` = `panel.subcluster`); `adata.obs_names`. **Produces:** `_Layout.discard_cells`; `_write_cell_dispositions(lay, panel, obs_names) -> None`; `_DISCARD_CELL_COLS`; the 5 new panel columns.

- [ ] **Step 1: Write the failing test** — `tests/test_discard_outputs.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_discard_outputs.py -q`
Expected: FAIL — `ImportError: cannot import name '_write_cell_dispositions'`.

- [ ] **Step 3: Implement** — in `pipeline.py`:

(a) Append 5 columns to `_PANEL_COLS` (L50-55) and `_DIAGNOSIS_COLS` (L57-61):

```python
_PANEL_COLS = ['parent_cluster', 'subcluster', 'reference_subcluster',
               'minor_umap_label', 'main_umap_label', 'n_cells', 'frac_of_parent',
               'top5_up_genes', 'top5_down_genes', 'n_sig_genes',
               'top_sample_enriched', 'top_qc_drift', 'rule_baseline',
               'likely_cause', 'cause_detail', 'diagnosis_confidence',
               'diagnosis_rationale', 'llm_overrode_rule',
               'disposition_baseline', 'recommended_disposition',
               'disposition_overridden', 'disposition_reason', 'proposed_cell_type']

_DIAGNOSIS_COLS = ['parent_cluster', 'subcluster', 'reference_subcluster',
                   'minor_umap_label', 'main_umap_label', 'n_cells', 'frac_of_parent',
                   'rule_baseline', 'likely_cause', 'cause_detail',
                   'diagnosis_confidence', 'diagnosis_rationale',
                   'llm_overrode_rule', 'disposition_baseline',
                   'recommended_disposition', 'disposition_overridden',
                   'disposition_reason', 'proposed_cell_type']
```

(b) Add `discard_cells` to `_Layout` (after `cell_labels`, L77):

```python
    @property
    def cell_labels(self):   return self.root / 'cell_labels.tsv'
    @property
    def discard_cells(self): return self.root / 'discard_cells.tsv'
```

(c) Add the helper after `_ordered_panel` (~L158):

```python
_DISCARD_CELL_COLS = ['barcode', 'input_row_index', 'subcluster', 'parent_cluster',
                      'likely_cause', 'diagnosis_confidence', 'disposition_reason']


def _write_cell_dispositions(lay, panel, obs_names):
    """Join recommended_disposition onto cell_labels.tsv (per cell), and write
    discard_cells.tsv (DISCARD cells only), keyed by barcode (obs_name) with the
    0-based row position in the standissect-input adata (from obs_names)."""
    if not lay.cell_labels.exists():
        return
    labels = pd.read_csv(lay.cell_labels, sep='\t', index_col=0)
    sub = labels['original_cluster_split'].astype(str)
    if len(panel) and 'subcluster' in panel.columns:
        p = panel.copy()
        p['subcluster'] = p['subcluster'].astype(str)
        p = p.drop_duplicates('subcluster').set_index('subcluster')
    else:
        p = pd.DataFrame()

    def col(name):
        if len(p) and name in p.columns:
            return sub.map(p[name])
        return pd.Series([None] * len(labels), index=labels.index)

    labels['recommended_disposition'] = col('recommended_disposition').fillna('')
    labels.to_csv(lay.cell_labels, sep='\t')

    pos = {str(b): i for i, b in enumerate(obs_names)}
    mask = (labels['recommended_disposition'] == 'DISCARD').values
    bc = labels.index[mask]
    discard = pd.DataFrame({
        'barcode': bc,
        'input_row_index': [pos.get(str(b)) for b in bc],
        'subcluster': sub[mask].values,
        'parent_cluster': col('parent_cluster')[mask].values,
        'likely_cause': col('likely_cause')[mask].values,
        'diagnosis_confidence': col('diagnosis_confidence')[mask].values,
        'disposition_reason': col('disposition_reason')[mask].values,
    }, columns=_DISCARD_CELL_COLS)
    discard.to_csv(lay.discard_cells, sep='\t', index=False)
```

(d) Call it in the terminal aggregation, right after `diagnosis_all` is written (after L877), passing `adata.obs_names`:

```python
    diag_cols = [c for c in _DIAGNOSIS_COLS if c in panel.columns]
    panel[diag_cols].to_csv(lay.diagnosis_all, sep='\t', index=False)
    _write_cell_dispositions(lay, panel, adata.obs_names)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_discard_outputs.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add pipeline.py tests/test_discard_outputs.py
git commit -m "feat(pipeline): disposition columns + per-cell join + cell-precise discard_cells.tsv"
```

---

### Task 5: report "Recommended discards" section

**Files:**
- Modify: `report.py` (`build_report` sidebar L125 + insert after L150; add `_discards_section`)
- Test: `tests/test_report.py` (extend)

**Interfaces — Consumes:** `<root>/panel.tsv` with `recommended_disposition`. **Produces:** `_discards_section(root) -> str`; `<h2 id="discards">` + `<a href="#discards">` in `build_report`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_report.py`:

```python
def test_report_has_discards_section(tmp_path):
    root = tmp_path / "out" / "leiden"
    (root / "clusters" / "c0").mkdir(parents=True)
    pd.DataFrame({
        "parent_cluster": ["0", "0"], "subcluster": ["c0_1", "c0_2"],
        "n_cells": [12, 7], "likely_cause": ["doublet-driven", "cell-cycle"],
        "diagnosis_confidence": [0.9, 0.8],
        "recommended_disposition": ["DISCARD", "KEEP"],
        "disposition_reason": ["doublets", "cycling"],
    }).to_csv(root / "panel.tsv", sep="\t", index=False)
    out = report.build_report(str(root))
    html = pathlib.Path(out).read_text(encoding="utf-8")
    assert 'id="discards"' in html
    assert 'href="#discards"' in html
    assert "12" in html and "c0_1" in html


def test_report_discards_section_handles_no_discards(tmp_path):
    root = tmp_path / "out" / "leiden"
    (root / "clusters" / "c0").mkdir(parents=True)
    pd.DataFrame({"parent_cluster": ["0"], "subcluster": ["c0_1"], "n_cells": [9],
                  "likely_cause": ["biology-candidate"],
                  "recommended_disposition": ["KEEP"]}
                 ).to_csv(root / "panel.tsv", sep="\t", index=False)
    out = report.build_report(str(root))
    html = pathlib.Path(out).read_text(encoding="utf-8")
    assert 'id="discards"' in html
    assert "No clusters recommended for discard" in html
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_report.py -q`
Expected: FAIL — `id="discards"` not found.

- [ ] **Step 3: Implement** — in `report.py`:

(a) Add `_discards_section` after `_table` (~L47):

```python
def _discards_section(root):
    """The 'Recommended discards' section, built from panel.tsv."""
    root = Path(root)
    panel = _read_tsv_safe(root / 'panel.tsv')
    h = ['<h2 id="discards">Recommended discards</h2>']
    if not len(panel) or 'recommended_disposition' not in panel.columns:
        h.append('<p class="muted">No disposition information available.</p>')
        return '\n'.join(h)
    disp = panel['recommended_disposition'].astype(str)
    discard = panel[disp == 'DISCARD']
    uncertain = panel[disp == 'UNCERTAIN']
    if not len(discard):
        h.append('<p class="muted">No clusters recommended for discard.</p>')
    else:
        n_cells = int(pd.to_numeric(discard.get('n_cells'),
                                    errors='coerce').fillna(0).sum())
        by_cause = (discard['likely_cause'].astype(str).value_counts().to_dict()
                    if 'likely_cause' in discard.columns else {})
        cause_str = ', '.join(f'{k}: {v}' for k, v in by_cause.items())
        h.append(f'<p><b>{len(discard)}</b> clusters recommended for discard, '
                 f'<b>{n_cells}</b> cells total. By cause — {cause_str}.</p>')
        cols = [c for c in ('subcluster', 'n_cells', 'likely_cause',
                            'diagnosis_confidence', 'disposition_reason')
                if c in discard.columns]
        h.append(discard[cols].to_html(index=False, border=0, classes='deg',
                                       float_format=lambda x: f'{x:.3g}'))
    if len(uncertain):
        cols = [c for c in ('subcluster', 'n_cells', 'likely_cause',
                            'disposition_reason') if c in uncertain.columns]
        h.append('<details><summary>flagged — UNCERTAIN (kept by default): '
                 f'{len(uncertain)} clusters</summary>'
                 + uncertain[cols].to_html(index=False, border=0, classes='deg',
                                           float_format=lambda x: f'{x:.3g}')
                 + '</details>')
    return '\n'.join(h)
```

(b) Add the sidebar anchor in `build_report` (after L125):

```python
    h.append('<a href="#overview">Overview</a>')
    h.append('<a href="#discards">Recommended discards</a>')
```

(c) Append the section after the core-names block (after L150):

```python
    core_names_html = _table(root / 'core_names.tsv')
    if core_names_html:
        h.append('<div class="cap">canonical-core cell-type names</div>')
        h.append(core_names_html)
    h.append(_discards_section(root))
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_report.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add report.py tests/test_report.py
git commit -m "feat(report): Recommended discards section + sidebar anchor"
```

---

### Task 6: Naming "differs from original" in `annotate.py`

**Files:**
- Modify: `annotate.py` (`NAMING_PROMPT_VERSION` L34; `CoreNaming` L109-138; `build_core_naming_prompt` L214-235; `_core_naming_from_dict` L238-253; `run_naming_stage` rows L485-493; `CORE_NAME_COLS` L381-382)
- Test: `tests/test_annotate.py` (extend)

**Interfaces — Consumes:** `CoreEvidence.parent_cluster` (already the original annotation). **Produces:** `CoreNaming.differs_from_original: bool`; naming schema key `differs_from_original`; `to_core_name_row` gains `original_label` + `differs_from_original`; `CORE_NAME_COLS` gains both; `NAMING_PROMPT_VERSION == 'standissect-naming-v2'`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_annotate.py`:

```python
def test_naming_prompt_advertises_differs_and_cites_parent():
    system, user = annotate.build_core_naming_prompt(_evi(["IL3RA"], parent="myeloid"))
    assert "differs_from_original" in user
    assert "parent_cluster" in (system + user)


def test_core_naming_from_dict_parses_differs_and_original_label():
    evi = _evi(["IL3RA", "CLEC4C"], parent="myeloid")
    data = {"cell_type": "pDC", "confidence": 0.9, "rationale": "pDC markers",
            "differs_from_original": True, "markers_used": ["IL3RA"]}
    naming = annotate._core_naming_from_dict(data, evi, model="m")
    assert naming.differs_from_original is True
    row = naming.to_core_name_row(evi)
    assert row["original_label"] == "myeloid"
    assert row["differs_from_original"] is True
    assert "original_label" in annotate.CORE_NAME_COLS
    assert "differs_from_original" in annotate.CORE_NAME_COLS


def test_run_naming_stage_writes_relabel_columns(tmp_path):
    from diagnosis import CallableChatClient
    clusters = tmp_path / "clusters"
    canon = tmp_path / "canonical_markers"
    (clusters / "cmyeloid").mkdir(parents=True)
    canon.mkdir(parents=True)
    pd.DataFrame({"gene": ["IL3RA", "CLEC4C"], "logfoldchanges": [2.0, 2.0],
                  "scores": [10.0, 9.0]}
                 ).to_csv(canon / "markers_cmyeloid_0.tsv", sep="\t", index=False)
    client = CallableChatClient(lambda s, u: json.dumps(
        {"cell_type": "pDC", "confidence": 0.9, "rationale": "pDC markers",
         "differs_from_original": True, "markers_used": ["IL3RA"]}), model="m")
    engine = annotate.make_naming_engine(client=client)
    annotate.run_naming_stage(clusters_dir=clusters, canonical_dir=canon,
        core_names_path=tmp_path / "core_names.tsv", parents=["myeloid"],
        engine=engine, forced=True)
    df = pd.read_csv(tmp_path / "core_names.tsv", sep="\t")
    assert df.loc[0, "cell_type"] == "pDC"
    assert str(df.loc[0, "original_label"]) == "myeloid"
    assert bool(df.loc[0, "differs_from_original"]) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_annotate.py -q`
Expected: FAIL — `differs_from_original` absent from prompt / `CoreNaming` / columns.

- [ ] **Step 3: Implement** — in `annotate.py`:

(a) `NAMING_PROMPT_VERSION` (L34):

```python
NAMING_PROMPT_VERSION = 'standissect-naming-v2'
```

(b) `CoreNaming` (L109-121) — add the field (after `error`):

```python
    error: str | None = None
    differs_from_original: bool = False
```

(c) `to_core_name_row` (L129-138) — add `original_label` + `differs_from_original`:

```python
    def to_core_name_row(self, evidence: CoreEvidence) -> dict:
        return {
            'parent_cluster': evidence.parent_cluster,
            'core_subcluster': evidence.core_subcluster,
            'cell_type': self.cell_type,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'source': self.source,
            'model': self.model,
            'original_label': evidence.parent_cluster,
            'differs_from_original': self.differs_from_original,
        }
```

(d) `build_core_naming_prompt` (L214-235) — schema key + system instruction:

```python
def build_core_naming_prompt(evidence: CoreEvidence) -> tuple[str, str]:
    schema = {
        'cell_type': 'cell type/state name, or "uncertain"',
        'confidence': 'number from 0 to 1',
        'rationale': 'one concise sentence citing supplied markers',
        'markers_used': ['subset of the supplied marker genes'],
        'alternatives': ['other plausible cell types'],
        'differs_from_original': 'true if your cell_type denotes a different '
                                 'identity than the cluster\'s existing annotation '
                                 '(evidence.parent_cluster), else false',
    }
    system = (
        "You are a single-cell biologist. Name the most likely cell type or state "
        "for a cluster from its ranked canonical marker genes, using established "
        "marker-to-cell-type knowledge. If the markers are ambiguous, return "
        '"uncertain" with low confidence. Cite only markers from the supplied '
        "list; do not introduce markers that are not listed. The cluster's "
        "existing annotation label is provided as evidence.parent_cluster; set "
        "differs_from_original=true when your cell_type denotes a semantically "
        "different identity than that label. Return strict JSON only."
    )
    user = json.dumps({
        'task': 'name_one_canonical_core',
        'tissue_hint': evidence.hint,
        'output_schema': schema,
        'evidence': evidence.to_dict(),
    }, ensure_ascii=False, indent=2)
    return system, user
```

(e) `_core_naming_from_dict` (L245-253) — parse it:

```python
    return CoreNaming(
        cell_type=cell_type,
        confidence=float(data.get('confidence', 0.0) or 0.0),
        rationale=str(data.get('rationale', '')),
        markers_used=used,
        alternatives=[str(a) for a in (data.get('alternatives') or [])],
        source='llm',
        model=model,
        differs_from_original=bool(data.get('differs_from_original', False)),
    )
```

(f) `run_naming_stage` rows (L485-493) — add the two fields (computed from `parent` + the JSON):

```python
        rows.append({
            'parent_cluster': str(parent),
            'core_subcluster': f"c{parent}_0",
            'cell_type': data.get('cell_type'),
            'confidence': data.get('confidence'),
            'rationale': data.get('rationale'),
            'source': data.get('source'),
            'model': data.get('model'),
            'original_label': str(parent),
            'differs_from_original': bool(data.get('differs_from_original', False)),
        })
```

(g) `CORE_NAME_COLS` (L381-382) — add the two columns:

```python
CORE_NAME_COLS = ['parent_cluster', 'core_subcluster', 'cell_type', 'confidence',
                  'rationale', 'source', 'model', 'original_label',
                  'differs_from_original']
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_annotate.py -q`
Expected: PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add annotate.py tests/test_annotate.py
git commit -m "feat(annotate): naming differs_from_original + original_label (naming v2)"
```

---

### Task 7: proposed_cell_types.tsv aggregation + report section

**Files:**
- Modify: `pipeline.py` (`_Layout` add `proposed_cell_types`; add `_write_proposed_cell_types` near `_write_cell_dispositions`; call after naming stage ~L917)
- Modify: `report.py` (`build_report` sidebar + insert; add `_proposed_types_section`)
- Test: `tests/test_discard_outputs.py` (extend), `tests/test_report.py` (extend)

**Interfaces — Consumes:** `panel.tsv` with `proposed_cell_type` (Tasks 2,4); `core_names.tsv` with `differs_from_original` (Task 6). **Produces:** `_Layout.proposed_cell_types`; `_write_proposed_cell_types(lay, panel, core_names_df) -> None`; `_PROPOSED_COLS`; `_proposed_types_section(root) -> str`; `<h2 id="proposed">` + anchor.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_discard_outputs.py`:

```python
from standissect.pipeline import _write_proposed_cell_types, _PROPOSED_COLS  # noqa: E402


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
```

Append to `tests/test_report.py`:

```python
def test_report_has_proposed_types_section(tmp_path):
    root = tmp_path / "out" / "leiden"
    (root / "clusters" / "c0").mkdir(parents=True)
    pd.DataFrame({"parent_cluster": ["c0_1"], "subcluster": ["c0_1"]}
                 ).to_csv(root / "clusters" / "c0" / "panel.tsv", sep="\t", index=False)
    pd.DataFrame({"level": ["minor"], "parent_cluster": ["myeloid"],
                  "subcluster": ["cmyeloid_1"], "proposed_cell_type": ["pDC"],
                  "confidence": [0.8], "rationale": ["pDC markers"]}
                 ).to_csv(root / "proposed_cell_types.tsv", sep="\t", index=False)
    out = report.build_report(str(root))
    html = pathlib.Path(out).read_text(encoding="utf-8")
    assert 'id="proposed"' in html
    assert 'href="#proposed"' in html
    assert "pDC" in html
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_discard_outputs.py tests/test_report.py -q`
Expected: FAIL — `cannot import name '_write_proposed_cell_types'` / `id="proposed"` not found.

- [ ] **Step 3: Implement**

(a) `pipeline.py` — add `proposed_cell_types` to `_Layout` (after `discard_cells`):

```python
    @property
    def proposed_cell_types(self): return self.root / 'proposed_cell_types.tsv'
```

(b) `pipeline.py` — add the helper next to `_write_cell_dispositions`:

```python
_PROPOSED_COLS = ['level', 'parent_cluster', 'subcluster', 'proposed_cell_type',
                  'confidence', 'rationale']


def _write_proposed_cell_types(lay, panel, core_names_df):
    """Collect LLM-proposed cell types: minor (panel.proposed_cell_type) +
    major (core_names.differs_from_original) into proposed_cell_types.tsv."""
    rows = []
    if len(panel) and 'proposed_cell_type' in panel.columns:
        m = panel[panel['proposed_cell_type'].notna()
                  & (panel['proposed_cell_type'].astype(str).str.strip() != '')
                  & (panel['proposed_cell_type'].astype(str).str.lower() != 'nan')]
        for _, r in m.iterrows():
            rows.append({
                'level': 'minor',
                'parent_cluster': r.get('parent_cluster'),
                'subcluster': r.get('subcluster'),
                'proposed_cell_type': r.get('proposed_cell_type'),
                'confidence': r.get('diagnosis_confidence'),
                'rationale': r.get('diagnosis_rationale'),
            })
    if len(core_names_df) and 'differs_from_original' in core_names_df.columns:
        d = core_names_df[core_names_df['differs_from_original'].apply(
            lambda v: str(v).strip().lower() in ('true', '1'))]
        for _, r in d.iterrows():
            rows.append({
                'level': 'major',
                'parent_cluster': r.get('parent_cluster'),
                'subcluster': r.get('core_subcluster'),
                'proposed_cell_type': r.get('cell_type'),
                'confidence': r.get('confidence'),
                'rationale': r.get('rationale'),
            })
    pd.DataFrame(rows, columns=_PROPOSED_COLS).to_csv(
        lay.proposed_cell_types, sep='\t', index=False)
```

(c) `pipeline.py` — call it after the naming stage, where `core_names_df` is read (after L917):

```python
    core_names_df = _read_tsv(lay.core_names)
    _write_proposed_cell_types(lay, panel, core_names_df)
```

(d) `report.py` — add `_proposed_types_section` (after `_discards_section`):

```python
def _proposed_types_section(root):
    """The 'Proposed new / re-labeled cell types' section."""
    root = Path(root)
    df = _read_tsv_safe(root / 'proposed_cell_types.tsv')
    h = ['<h2 id="proposed">Proposed new / re-labeled cell types</h2>']
    if not len(df):
        h.append('<p class="muted">No proposed new or re-labeled cell types.</p>')
        return '\n'.join(h)
    h.append(f'<p><b>{len(df)}</b> proposed.</p>')
    h.append(df.to_html(index=False, border=0, classes='deg',
                        float_format=lambda x: f'{x:.3g}'))
    return '\n'.join(h)
```

(e) `report.py` — sidebar anchor (after the `#discards` anchor added in Task 5):

```python
    h.append('<a href="#discards">Recommended discards</a>')
    h.append('<a href="#proposed">Proposed new / re-labeled cell types</a>')
```

(f) `report.py` — append the section (after the `_discards_section` append from Task 5):

```python
    h.append(_discards_section(root))
    h.append(_proposed_types_section(root))
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_discard_outputs.py tests/test_report.py -q`
Expected: PASS (Task 4/5 tests + 3 new).

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add pipeline.py report.py tests/test_discard_outputs.py tests/test_report.py
git commit -m "feat: proposed_cell_types.tsv aggregation + report section"
```

---

### Task 8: Full-suite + Marrow e2e

**Files:** No source changes (verification only). Reuses the preprocessed Marrow h5ad from the concurrency e2e.

- [ ] **Step 1: Full unit suite**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/ -q`
Expected: PASS — all prior tests (49) + the new disposition/discard/report/cli/annotate tests (~75 total). No failures.

- [ ] **Step 2: Marrow e2e (LOCAL — NO srun/sbatch)**

Locate the preprocessed Marrow file (`*_pp.h5ad` with `X_umap` + QC cols) under `/scratch/users/chensj16/standissect_test/`; if absent, re-run `preprocess_marrow.py`. With `ARK_API_KEY` exported:

```bash
cd /scratch/users/chensj16/projects/standissect
ARK_API_KEY=$ARK_API_KEY python -m standissect run <marrow_pp.h5ad> \
  --cluster-col cell_ontology_class --output-dir /scratch/users/chensj16/standissect_test/discard_out \
  --target-k 14 --mito-col pct_counts_mt --feature-count-col n_genes_by_counts \
  --umi-count-col total_counts
```

- [ ] **Step 3: Verify e2e outputs** under `<output>/cell_ontology_class/`:
- `panel.tsv` has the 5 new columns; `recommended_disposition` ∈ {DISCARD, KEEP, UNCERTAIN}.
- `cell_labels.tsv` has `recommended_disposition`.
- `discard_cells.tsv`: columns == `_DISCARD_CELL_COLS`; every row is a DISCARD cluster's cell; `input_row_index` values are valid 0-based positions; `barcode` values are real `obs_names`.
- `proposed_cell_types.tsv` exists (header at least); any minor rows have `level=='minor'`, major rows `level=='major'`.
- `params.json` has `discard_confidence_threshold`, `diagnosis_prompt_version == 'standissect-diagnosis-v2'`, and naming prompt version recorded.
- `report.html` contains `id="discards"` and `id="proposed"`.
- **Invariant spot-check:** no cell whose cluster `likely_cause` ∈ {cell-cycle, sex-driven, interferon-response, biology-candidate} appears in `discard_cells.tsv`.

- [ ] **Step 4:** Optional short e2e note to the ledger; no source change → skip commit if nothing changed.

---

## Self-Review (completed)

**1. Spec coverage:** D1 three-tier → T1,4,5. D2 11-cause → T1 (`ALLOWED_CAUSES`) + T2 (signatures). D3 conservative-only + gate → T1 (`derive_disposition`) + T2 (engines). D4 prompt → T2. **D5 proposed types** → minor (T1 field, T2 schema/parse, T4 panel col, T7 aggregation) + major (T6 naming) + T7 (`proposed_cell_types.tsv` + report). **D6 cell-precise discard** → T4 (`barcode` + `input_row_index` from `obs_names`). Outputs → T3,4,7. Report → T5,7. CLI/params → T3. e2e → T8. No gaps.

**2. Placeholder scan:** No TBD/TODO; every code/test step carries complete code. The only `<...>` are the Marrow path / API key in T8 (runtime values).

**3. Type consistency:** `derive_disposition` → `(recommended, baseline, overridden, reason)` consumed identically in `finalize_disposition`. Names identical across files: `recommended_disposition`/`disposition_baseline`/`disposition_overridden`/`disposition_reason`/`proposed_cell_type` (`DiagnosisResult`, `to_panel_fields`, `_PANEL_COLS`/`_DIAGNOSIS_COLS`, `_write_cell_dispositions`, `_write_proposed_cell_types`, `_discards_section`); `discard_confidence_threshold` (`derive_disposition` param `threshold`, engines, `make_diagnosis_engine`, `run_dissect_pipeline`, CLI); `differs_from_original`/`original_label` (`CoreNaming`, `to_core_name_row`, `_core_naming_from_dict`, `run_naming_stage`, `CORE_NAME_COLS`, `_write_proposed_cell_types`); `_Layout.discard_cells`/`_Layout.proposed_cell_types`; `_DISCARD_CELL_COLS`/`_PROPOSED_COLS` match their tests + T8 checks. `_write_cell_dispositions(lay, panel, obs_names)` called with `adata.obs_names`.
