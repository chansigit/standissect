# recommend-discard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive a per-cluster `recommended_disposition` (DISCARD / KEEP / UNCERTAIN) from the existing minor-cause diagnosis and surface explicit, fully-automated junk-discard recommendations in `panel.tsv`, `cell_labels.tsv`, a new `discard_cells.tsv`, and a report section.

**Architecture:** A deterministic `DISPOSITION_MAP` turns each `likely_cause` into a baseline disposition; an LLM may relax it toward KEEP (never escalate toward DISCARD); a confidence gate downgrades low-confidence DISCARDs to UNCERTAIN. Disposition logic lives in `diagnosis.py` (engine-held threshold, finalized inside `diagnose()`); `pipeline.py` writes the new columns/files; `report.py` renders the section; `cli.py` exposes the threshold.

**Tech Stack:** Python stdlib + numpy + pandas (no new deps). Diagnosis LLM via the existing vendored `llm_client.py` (unchanged). Tests: pytest.

## Global Constraints

- Work ONLY in the canonical project `/scratch/users/chensj16/projects/standissect` — NEVER the synovial copy.
- Branch `feat/recommend-discard` (already created off `main`). Do NOT merge without user OK. Per-task commits.
- Tests run LOCALLY on this compute node. NO `srun`/`sbatch`/Slurm.
- stdlib-only; NO new pip dependencies; do NOT modify the vendored `llm_client.py`.
- Disposition values are UPPERCASE `DISCARD` / `KEEP` / `UNCERTAIN`; `likely_cause` stays lowercase kebab-case.
- 11-cause taxonomy is locked. The 5 new causes (`dissociation-effect`, `cell-cycle`, `ambient-contamination`, `sex-driven`, `interferon-response`) are **LLM-only** — the rule cascade in `RuleDiagnosisEngine.diagnose` is NOT extended.
- **Conservative-only invariant:** automated adjustments (LLM override + confidence gate) may only move a disposition toward KEEP (`DISCARD`→`UNCERTAIN`→`KEEP`), never toward `DISCARD`. A `KEEP`/`UNCERTAIN` baseline can never be auto-escalated to `DISCARD`.
- Spec: `docs/superpowers/specs/2026-06-26-recommend-discard-design.md`.

## File Structure

- `diagnosis.py` — taxonomy, `DISPOSITION_MAP`, `derive_disposition`, `DiagnosisResult` disposition fields + `finalize_disposition`, engine wiring, LLM schema/prompt. (Tasks 1, 2)
- `pipeline.py` — `_PANEL_COLS`/`_DIAGNOSIS_COLS`, `run_dissect_pipeline` threshold param, `params.json`, `_Layout.discard_cells`, `_write_cell_dispositions`. (Tasks 3, 4)
- `report.py` — `_discards_section` + sidebar anchor. (Task 5)
- `cli.py` — `--discard-confidence-threshold`. (Task 3)
- `tests/test_disposition.py` (new), `tests/test_diagnosis_llm.py`, `tests/test_cli.py`, `tests/test_discard_outputs.py` (new), `tests/test_report.py`. (Tasks 1–5)

**Test import idioms (match existing):** scanpy-free modules import top-level after `sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))` then `import diagnosis` / `import report`. Modules pulling the package (`standissect.cli`, `standissect.pipeline`) use `parents[2]` then `from standissect.X import ...` (loads scanpy — fine on this compute node).

---

### Task 1: Disposition core in `diagnosis.py`

**Files:**
- Modify: `/scratch/users/chensj16/projects/standissect/diagnosis.py` (`ALLOWED_CAUSES` L29-36; `DiagnosisResult` L95-133)
- Test: `/scratch/users/chensj16/projects/standissect/tests/test_disposition.py` (create)

**Interfaces:**
- Produces: `ALLOWED_CAUSES` (now 11 values); `DISPOSITION_MAP: dict[str,str]`; `_DISPOSITION_RANK: dict[str,int]`; `derive_disposition(likely_cause, confidence, *, threshold, llm_disposition=None, llm_reason=None) -> (recommended:str, baseline:str, overridden:bool, reason:str)`; `DiagnosisResult` gains `disposition_baseline`, `recommended_disposition`, `disposition_overridden`, `disposition_reason` + `finalize_disposition(threshold, *, llm_disposition=None, llm_reason=None) -> DiagnosisResult`; `to_panel_fields()` emits the 4 new keys.

- [ ] **Step 1: Write the failing test** — `tests/test_disposition.py`:

```python
import pathlib
import sys

_PKG_DIR = pathlib.Path(__file__).resolve().parents[1]   # .../standissect
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
    assert DISPOSITION_MAP['doublet-driven'] == 'DISCARD'
    assert DISPOSITION_MAP['low-quality (high mt)'] == 'DISCARD'
    assert DISPOSITION_MAP['shallow-depth'] == 'DISCARD'
    assert DISPOSITION_MAP['dissociation-effect'] == 'DISCARD'
    assert DISPOSITION_MAP['ambient-contamination'] == 'DISCARD'
    assert DISPOSITION_MAP['cell-cycle'] == 'KEEP'
    assert DISPOSITION_MAP['sex-driven'] == 'KEEP'
    assert DISPOSITION_MAP['interferon-response'] == 'KEEP'
    assert DISPOSITION_MAP['biology-candidate'] == 'KEEP'
    assert DISPOSITION_MAP['sample-driven'] == 'UNCERTAIN'
    assert DISPOSITION_MAP['unclear'] == 'UNCERTAIN'


def test_gate_downgrades_low_confidence_discard():
    final, baseline, overridden, _ = derive_disposition(
        'doublet-driven', 0.3, threshold=0.5)
    assert (final, baseline, overridden) == ('UNCERTAIN', 'DISCARD', True)


def test_gate_keeps_high_confidence_discard():
    final, baseline, overridden, _ = derive_disposition(
        'doublet-driven', 0.9, threshold=0.5)
    assert (final, baseline, overridden) == ('DISCARD', 'DISCARD', False)


def test_override_relax_toward_keep_is_accepted():
    final, baseline, overridden, _ = derive_disposition(
        'doublet-driven', 0.9, threshold=0.5, llm_disposition='KEEP',
        llm_reason='clearly real cells')
    assert (final, baseline, overridden) == ('KEEP', 'DISCARD', True)


def test_override_escalate_toward_discard_is_rejected():
    # cell-cycle baseline KEEP; an LLM DISCARD must be clamped back to KEEP
    final, baseline, overridden, reason = derive_disposition(
        'cell-cycle', 0.9, threshold=0.5, llm_disposition='DISCARD',
        llm_reason='looks junky')
    assert (final, baseline, overridden) == ('KEEP', 'KEEP', False)
    assert 'rejected' in reason.lower()


def test_uncertain_baseline_cannot_be_escalated_to_discard():
    final, _, _, _ = derive_disposition(
        'unclear', 0.9, threshold=0.5, llm_disposition='DISCARD')
    assert final == 'UNCERTAIN'


def test_result_sets_disposition_fields_and_panel_fields():
    r = DiagnosisResult(likely_cause='cell-cycle', confidence=0.9)
    assert r.disposition_baseline == 'KEEP'
    assert r.recommended_disposition == 'KEEP'      # defaulted in __post_init__
    pf = r.to_panel_fields()
    for k in ('recommended_disposition', 'disposition_baseline',
              'disposition_overridden', 'disposition_reason'):
        assert k in pf


def test_finalize_applies_gate():
    r = DiagnosisResult(likely_cause='doublet-driven', confidence=0.2)
    r.finalize_disposition(0.5)
    assert r.recommended_disposition == 'UNCERTAIN'
    assert r.disposition_overridden is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_disposition.py -q`
Expected: FAIL — `ImportError: cannot import name 'DISPOSITION_MAP'` (and `derive_disposition`).

- [ ] **Step 3: Implement** — in `diagnosis.py`:

(a) Replace the `ALLOWED_CAUSES` tuple (L29-36) with the 11-value version:

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
    """Map a cause (+ optional LLM pick) to (recommended, baseline, overridden,
    reason). Conservative-only: an LLM pick is accepted only if it is at least
    as keep-leaning as the baseline; a DISCARD baseline below ``threshold``
    confidence is downgraded to UNCERTAIN. Both moves go toward KEEP only."""
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

(b) In `DiagnosisResult` (L95-133) add the four fields after `llm_overrode_rule` (all construction is keyword-based, so order is safe):

```python
    llm_overrode_rule: bool = False
    disposition_baseline: str = ''
    recommended_disposition: str = ''
    disposition_overridden: bool = False
    disposition_reason: str = ''
```

(c) Extend `__post_init__` (after the confidence clamp at L120) so a freshly-built result already carries a sensible baseline disposition:

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

(d) Add the `finalize_disposition` method (e.g. right after `__post_init__`):

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

(e) Extend `to_panel_fields()` (L125-133) — add the four keys:

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
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_disposition.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add diagnosis.py tests/test_disposition.py
git commit -m "feat(diagnosis): disposition map + conservative-only derive + result fields"
```

---

### Task 2: Engine wiring + LLM schema/prompt

**Files:**
- Modify: `/scratch/users/chensj16/projects/standissect/diagnosis.py` (`RuleDiagnosisEngine.__init__` L272 + `.diagnose` return L337-347; `LLMDiagnosisEngine.__init__` L447-455 + `.diagnose` L457-483; `build_llm_prompt` L486-510; `_diagnosis_from_dict` L513-533; `make_diagnosis_engine` L573-609; `PROMPT_VERSION` L38)
- Test: `/scratch/users/chensj16/projects/standissect/tests/test_diagnosis_llm.py` (extend)

**Interfaces:**
- Consumes: `derive_disposition`, `DiagnosisResult.finalize_disposition`, `DISPOSITION_MAP`, `ALLOWED_CAUSES` (Task 1).
- Produces: `RuleDiagnosisEngine(diagnosis_roles=None, *, discard_confidence_threshold=0.5)`; `LLMDiagnosisEngine(client, *, mode='llm', fallback_to_rule=True, diagnosis_roles=None, llm_retries=3, discard_confidence_threshold=0.5)`; `_diagnosis_from_dict(data, *, rule_baseline, mode, model, threshold=0.5)`; `make_diagnosis_engine(..., discard_confidence_threshold=0.5)`; module-level `CAUSE_SIGNATURES: dict`, `DISPOSITION_POLICY: str`; `PROMPT_VERSION == 'standissect-diagnosis-v2'`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_diagnosis_llm.py`:

```python
def test_prompt_advertises_disposition_and_eleven_causes():
    from diagnosis import build_llm_prompt, MinorEvidence, ALLOWED_CAUSES
    ev = MinorEvidence(parent_cluster="0", subcluster="c0_1",
                       reference_subcluster="c0_0", minor_umap_label="u1",
                       main_umap_label="u0", n_cells=10, frac_of_parent=0.1)
    system, user = build_llm_prompt(ev, mode="llm")
    assert len(ALLOWED_CAUSES) == 11
    assert "recommended_disposition" in user
    assert "disposition_reason" in user
    assert "cause_signatures" in user
    # conservative-only rule is communicated to the model
    assert "escalate" in (system + user).lower()


def test_rule_engine_result_carries_disposition_baseline():
    from diagnosis import RuleDiagnosisEngine, MinorEvidence
    eng = RuleDiagnosisEngine(discard_confidence_threshold=0.5)
    r = eng.diagnose(MinorEvidence(parent_cluster="0", subcluster="c0_1",
                     reference_subcluster="c0_0", minor_umap_label="u1",
                     main_umap_label="u0", n_cells=10, frac_of_parent=0.1))
    # default 'unclear' -> UNCERTAIN baseline, no override
    assert r.recommended_disposition == r.disposition_baseline
    assert r.disposition_overridden is False
    assert r.recommended_disposition in {"DISCARD", "KEEP", "UNCERTAIN"}


def test_llm_disposition_parsed_and_clamped():
    # cause cell-cycle (baseline KEEP) + llm DISCARD -> clamped to KEEP
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
    assert r.recommended_disposition == "DISCARD"   # baseline, high confidence


def test_new_cause_dissociation_is_accepted_by_llm():
    payload = json.dumps({"likely_cause": "dissociation-effect", "confidence": 0.8,
                          "rationale": "HSP + IEG", "recommended_disposition": "DISCARD",
                          "disposition_reason": "stress signature"})
    r = parse_llm_result(payload, rule_baseline=None, mode="llm", model=None)
    assert r.likely_cause == "dissociation-effect"
    assert r.recommended_disposition == "DISCARD"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_diagnosis_llm.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'discard_confidence_threshold'` / missing prompt keys.

- [ ] **Step 3: Implement** — in `diagnosis.py`:

(a) Bump `PROMPT_VERSION` (L38):

```python
PROMPT_VERSION = 'standissect-diagnosis-v2'
```

(b) Add module-level guidance near `build_llm_prompt` (before it):

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
    "recommended_disposition toward KEEP (DISCARD->UNCERTAIN->KEEP) when the "
    "evidence supports keeping the cells, and you MUST give disposition_reason. "
    "You may NOT escalate toward DISCARD via recommended_disposition; to mark a "
    "cluster as junk, pick a discard-type likely_cause instead."
)
```

(c) Extend `build_llm_prompt` (L486-510) — add the two schema keys, mention the policy in the system prompt, and add `cause_signatures` + `disposition_policy` to the user payload:

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
    }
    system = (
        "You diagnose minor fragments inside single-cell clusters. "
        "Use only the supplied statistical evidence. Do not invent measurements, "
        "cell types, markers, or experiments. Choose exactly one likely_cause "
        "from the allowed enum, matching species-appropriate gene orthologs. "
        "Then set recommended_disposition following disposition_policy: you may "
        "relax toward KEEP but must NOT escalate toward DISCARD. Return strict JSON only."
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

(d) `_diagnosis_from_dict` (L513-533) — add `threshold=0.5`, then finalize the built result with the LLM's disposition pick:

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
    return result
```

(e) `RuleDiagnosisEngine.__init__` (L272-273) — accept and store the threshold:

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

(g) `LLMDiagnosisEngine.__init__` (L447-455) — accept the threshold, store it, and pass it to the internal rule engine; then thread it into the `_diagnosis_from_dict` call in `.diagnose` (L467-469):

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

In `.diagnose`, the inner lambda becomes (note the new `threshold=`):

```python
                    lambda data: _diagnosis_from_dict(
                        data, rule_baseline=evidence.rule_baseline,
                        mode=self.mode, model=self.model,
                        threshold=self.discard_confidence_threshold)),
```

(The fallback `fallback = baseline` path already carries a finalized disposition, because `baseline` came from `self.rule_engine.diagnose(...)`.)

(h) `make_diagnosis_engine` (L573-609) — add the param and pass it to both engines:

```python
def make_diagnosis_engine(
    *,
    mode: str = 'rule',
    llm_client=None,
    ark_api_key: str | None = None,
    ark_api_key_env: str = 'ARK_API_KEY',
    ark_model: str = DEFAULT_ARK_MODEL,
    ark_endpoint: str = DEFAULT_ARK_ENDPOINT,
    timeout: int = 120,
    fallback_to_rule: bool = True,
    diagnosis_roles=None,
    llm_retries: int = 3,
    discard_confidence_threshold: float = 0.5,
):
    ...
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
Expected: PASS (existing diagnosis tests + 5 new + Task 1 tests). The existing `test_call_structured_with_callable_client_builds_result` and `parse_llm_result` tests still pass because `threshold` defaults to `0.5`.

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add diagnosis.py tests/test_diagnosis_llm.py
git commit -m "feat(diagnosis): engine threshold + LLM disposition schema/prompt (v2)"
```

---

### Task 3: Pipeline threshold param + CLI + params.json

**Files:**
- Modify: `/scratch/users/chensj16/projects/standissect/pipeline.py` (`run_dissect_pipeline` signature L624-668; engine build L738-749; `params.json` L953-988)
- Modify: `/scratch/users/chensj16/projects/standissect/cli.py` (diag arg group L91-92; `run_cmd` L122-160)
- Test: `/scratch/users/chensj16/projects/standissect/tests/test_cli.py` (extend)

**Interfaces:**
- Consumes: `make_diagnosis_engine(..., discard_confidence_threshold=...)` (Task 2).
- Produces: `run_dissect_pipeline(..., discard_confidence_threshold=0.5)`; CLI flag `--discard-confidence-threshold` → `args.discard_confidence_threshold`; `params.json['discard_confidence_threshold']`.

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
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'discard_confidence_threshold'`.

- [ ] **Step 3: Implement**

(a) `cli.py` — add the flag in the `diag` group after `--ark-timeout` (L92):

```python
    diag.add_argument('--discard-confidence-threshold', type=float, default=0.5,
                      help='DISCARD calls below this diagnosis confidence are '
                           'downgraded to UNCERTAIN (kept + flagged). Default: 0.5.')
```

(b) `cli.py` — pass it in the `run_dissect_pipeline(...)` call in `run_cmd` (add near L158, beside `diagnosis_timeout`):

```python
        diagnosis_timeout=args.ark_timeout,
        discard_confidence_threshold=args.discard_confidence_threshold,
        random_state=args.random_state,
```

(c) `pipeline.py` — add the param to `run_dissect_pipeline` (after `llm_retries=3,` at L666):

```python
    llm_retries=3,
    discard_confidence_threshold=0.5,
    random_state=0,
```

(d) `pipeline.py` — pass it to `make_diagnosis_engine` (in the call at L738-749, add a kwarg):

```python
        diagnosis_roles=resolved_roles,
        llm_retries=llm_retries,
        discard_confidence_threshold=discard_confidence_threshold,
    )
```

(e) `pipeline.py` — record it in `params.json` (add inside the dict near L986):

```python
        'diagnosis_timeout': diagnosis_timeout,
        'discard_confidence_threshold': discard_confidence_threshold,
        'partition_info': partition_info,
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_cli.py -q`
Expected: PASS (existing cli tests + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add pipeline.py cli.py tests/test_cli.py
git commit -m "feat(pipeline,cli): --discard-confidence-threshold + params.json"
```

---

### Task 4: panel/diagnosis columns + per-cell disposition + discard_cells.tsv

**Files:**
- Modify: `/scratch/users/chensj16/projects/standissect/pipeline.py` (`_PANEL_COLS` L50-55; `_DIAGNOSIS_COLS` L57-61; `_Layout` L64-95; terminal aggregation L870-880; add helper `_write_cell_dispositions`)
- Test: `/scratch/users/chensj16/projects/standissect/tests/test_discard_outputs.py` (create)

**Interfaces:**
- Consumes: a global `panel.tsv` whose rows carry `recommended_disposition` etc. (Tasks 2–3); `cell_labels.tsv` with index = barcode and columns `umap_cluster`, `original_cluster_split` (`subcluster` of the form `c{parent}_{rank}`).
- Produces: `_Layout.discard_cells` → `<root>/discard_cells.tsv`; module-level `_write_cell_dispositions(lay, panel) -> None`; `_PANEL_COLS`/`_DIAGNOSIS_COLS` each gain the 4 disposition columns.

- [ ] **Step 1: Write the failing test** — `tests/test_discard_outputs.py`:

```python
import pathlib
import sys

import pandas as pd

_PKG_PARENT = pathlib.Path(__file__).resolve().parents[2]   # .../projects
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
    })


def test_panel_cols_include_disposition_columns():
    for c in ('recommended_disposition', 'disposition_baseline',
              'disposition_overridden', 'disposition_reason'):
        assert c in _PANEL_COLS
        assert c in _DIAGNOSIS_COLS


def test_cell_labels_gets_disposition_and_discard_file(tmp_path):
    lay = _Layout(tmp_path, 'leiden')
    lay.root.mkdir(parents=True)
    pd.DataFrame(
        {'umap_cluster': ['a', 'a', 'b', 'b'],
         'original_cluster_split': ['c0_1', 'c0_2', 'c1_1', 'c0_1']},
        index=['AAA', 'BBB', 'CCC', 'DDD'],
    ).to_csv(lay.cell_labels, sep='\t')

    _write_cell_dispositions(lay, _panel())

    labels = pd.read_csv(lay.cell_labels, sep='\t', index_col=0)
    assert list(labels['recommended_disposition']) == ['DISCARD', 'KEEP',
                                                        'UNCERTAIN', 'DISCARD']
    discard = pd.read_csv(lay.discard_cells, sep='\t')
    assert sorted(discard['barcode']) == ['AAA', 'DDD']        # only DISCARD cells
    assert set(discard.columns) == {'barcode', 'subcluster', 'parent_cluster',
                                    'likely_cause', 'diagnosis_confidence',
                                    'disposition_reason'}
    assert set(discard['subcluster']) == {'c0_1'}


def test_empty_discard_writes_header_only(tmp_path):
    lay = _Layout(tmp_path, 'leiden')
    lay.root.mkdir(parents=True)
    pd.DataFrame({'umap_cluster': ['a'], 'original_cluster_split': ['c0_2']},
                 index=['AAA']).to_csv(lay.cell_labels, sep='\t')
    _write_cell_dispositions(lay, _panel())          # c0_2 -> KEEP
    discard = pd.read_csv(lay.discard_cells, sep='\t')
    assert len(discard) == 0
    assert 'barcode' in discard.columns
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_discard_outputs.py -q`
Expected: FAIL — `ImportError: cannot import name '_write_cell_dispositions'`.

- [ ] **Step 3: Implement** — in `pipeline.py`:

(a) Append the 4 columns to `_PANEL_COLS` (L50-55) and `_DIAGNOSIS_COLS` (L57-61):

```python
_PANEL_COLS = ['parent_cluster', 'subcluster', 'reference_subcluster',
               'minor_umap_label', 'main_umap_label', 'n_cells', 'frac_of_parent',
               'top5_up_genes', 'top5_down_genes', 'n_sig_genes',
               'top_sample_enriched', 'top_qc_drift', 'rule_baseline',
               'likely_cause', 'cause_detail', 'diagnosis_confidence',
               'diagnosis_rationale', 'llm_overrode_rule',
               'disposition_baseline', 'recommended_disposition',
               'disposition_overridden', 'disposition_reason']

_DIAGNOSIS_COLS = ['parent_cluster', 'subcluster', 'reference_subcluster',
                   'minor_umap_label', 'main_umap_label', 'n_cells', 'frac_of_parent',
                   'rule_baseline', 'likely_cause', 'cause_detail',
                   'diagnosis_confidence', 'diagnosis_rationale',
                   'llm_overrode_rule', 'disposition_baseline',
                   'recommended_disposition', 'disposition_overridden',
                   'disposition_reason']
```

(b) Add a `discard_cells` property to `_Layout` (after `cell_labels`, L77):

```python
    @property
    def cell_labels(self):   return self.root / 'cell_labels.tsv'
    @property
    def discard_cells(self): return self.root / 'discard_cells.tsv'
```

(c) Add the helper near the other module-level pipeline helpers (e.g. just below `_ordered_panel`, ~L158):

```python
_DISCARD_CELL_COLS = ['barcode', 'subcluster', 'parent_cluster', 'likely_cause',
                      'diagnosis_confidence', 'disposition_reason']


def _write_cell_dispositions(lay, panel):
    """Join recommended_disposition onto cell_labels.tsv (per cell) and write
    discard_cells.tsv (only cells whose cluster is recommended DISCARD)."""
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

    mask = (labels['recommended_disposition'] == 'DISCARD').values
    discard = pd.DataFrame({
        'barcode': labels.index[mask],
        'subcluster': sub[mask].values,
        'parent_cluster': col('parent_cluster')[mask].values,
        'likely_cause': col('likely_cause')[mask].values,
        'diagnosis_confidence': col('diagnosis_confidence')[mask].values,
        'disposition_reason': col('disposition_reason')[mask].values,
    }, columns=_DISCARD_CELL_COLS)
    discard.to_csv(lay.discard_cells, sep='\t', index=False)
```

(d) Call it in the terminal aggregation, right after `diagnosis_all` is written (after L877):

```python
    diag_cols = [c for c in _DIAGNOSIS_COLS if c in panel.columns]
    panel[diag_cols].to_csv(lay.diagnosis_all, sep='\t', index=False)
    _write_cell_dispositions(lay, panel)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/test_discard_outputs.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add pipeline.py tests/test_discard_outputs.py
git commit -m "feat(pipeline): disposition panel columns + per-cell join + discard_cells.tsv"
```

---

### Task 5: report "Recommended discards" section

**Files:**
- Modify: `/scratch/users/chensj16/projects/standissect/report.py` (`build_report` sidebar L125 + overview insert after L150; add `_discards_section`)
- Test: `/scratch/users/chensj16/projects/standissect/tests/test_report.py` (extend)

**Interfaces:**
- Consumes: `<root>/panel.tsv` with a `recommended_disposition` column.
- Produces: `_discards_section(root) -> str`; a `<h2 id="discards">` section + a `<a href="#discards">` sidebar anchor in `build_report`.

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
    assert "1" in html and "12" in html          # 1 cluster, 12 cells
    assert "c0_1" in html                          # the DISCARD cluster row


def test_report_discards_section_handles_no_discards(tmp_path):
    root = tmp_path / "out" / "leiden"
    (root / "clusters" / "c0").mkdir(parents=True)
    pd.DataFrame({"parent_cluster": ["0"], "subcluster": ["c0_1"],
                  "n_cells": [9], "likely_cause": ["biology-candidate"],
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

(a) Add the section builder (e.g. after `_table`, ~L47):

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

(b) Add the sidebar anchor in `build_report` (after L125 `h.append('<a href="#overview">Overview</a>')`):

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
Expected: PASS (existing report tests + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /scratch/users/chensj16/projects/standissect
git add report.py tests/test_report.py
git commit -m "feat(report): Recommended discards section + sidebar anchor"
```

---

### Task 6: Full-suite + Marrow e2e

**Files:**
- No source changes (verification only). Reuses the preprocessed Marrow h5ad from the concurrency feature's e2e.

**Interfaces:**
- Consumes: everything from Tasks 1–5.

- [ ] **Step 1: Run the full unit suite**

Run: `cd /scratch/users/chensj16/projects/standissect && python -m pytest tests/ -q`
Expected: PASS — all prior tests (49) plus the new disposition/discard/report/cli tests, ~60+ total. No failures.

- [ ] **Step 2: Marrow e2e (LOCAL — NO srun/sbatch)**

Use the preprocessed Marrow file from the concurrency e2e (`/scratch/users/chensj16/standissect_test/...` — locate the `*_pp.h5ad` with `X_umap` + QC cols; if absent, re-run `preprocess_marrow.py`). With `ARK_API_KEY` exported, run a small dissect into a fresh output dir, e.g.:

```bash
cd /scratch/users/chensj16/projects/standissect
ARK_API_KEY=$ARK_API_KEY python -m standissect run <marrow_pp.h5ad> \
  --cluster-col cell_ontology_class --output-dir /scratch/users/chensj16/standissect_test/discard_out \
  --target-k 14 --mito-col pct_counts_mt --feature-count-col n_genes_by_counts \
  --umi-count-col total_counts
```

- [ ] **Step 3: Verify e2e outputs**

Confirm under `<output>/cell_ontology_class/`:
- `panel.tsv` contains the 4 disposition columns; `recommended_disposition` ∈ {DISCARD, KEEP, UNCERTAIN}.
- `cell_labels.tsv` has a `recommended_disposition` column.
- `discard_cells.tsv` exists; every row's cluster has `recommended_disposition == DISCARD`; columns match `_DISCARD_CELL_COLS`.
- `params.json` has `discard_confidence_threshold` and `diagnosis_prompt_version == 'standissect-diagnosis-v2'`.
- `report.html` contains `id="discards"`.
- **Invariant spot-check:** no cell whose cluster `likely_cause` ∈ {cell-cycle, sex-driven, interferon-response, biology-candidate} appears in `discard_cells.tsv`.

- [ ] **Step 4: Commit a short e2e note** (optional, to the ledger or a NOTES file — no source change). If nothing to commit, skip.

---

## Self-Review (completed)

**1. Spec coverage:** D1 three-tier → Tasks 1,4,5. D2 11-cause taxonomy → Task 1 (`ALLOWED_CAUSES`) + Task 2 (prompt signatures). D3 hybrid + conservative-only + gate → Task 1 (`derive_disposition`) + Task 2 (engine/LLM wiring). D4 function-based cross-species prompt → Task 2 (`CAUSE_SIGNATURES`, `DISPOSITION_POLICY`). Outputs (panel/cell_labels/discard_cells) → Tasks 3,4. Report → Task 5. CLI + params → Task 3. `DISPOSITION_MAP` coverage test → Task 1. e2e → Task 6. No gaps.

**2. Placeholder scan:** No TBD/TODO; every code/test step carries complete code. The only `<...>` are the Marrow file path / API key in Task 6 (runtime values, not code).

**3. Type consistency:** `derive_disposition` returns `(recommended, baseline, overridden, reason)` — consumed identically in `finalize_disposition` (Task 1) and not re-shaped elsewhere. `recommended_disposition`/`disposition_baseline`/`disposition_overridden`/`disposition_reason` names are identical across `DiagnosisResult`, `to_panel_fields`, `_PANEL_COLS`/`_DIAGNOSIS_COLS`, `_write_cell_dispositions`, and `_discards_section`. `discard_confidence_threshold` is spelled identically across `derive_disposition`(`threshold`)/engines/`make_diagnosis_engine`/`run_dissect_pipeline`/CLI. `_Layout.discard_cells` and `_DISCARD_CELL_COLS` match the Task 4 test and Task 6 checks.
