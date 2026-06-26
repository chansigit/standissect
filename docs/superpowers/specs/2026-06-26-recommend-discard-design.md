# recommend-discard — Design Spec

**Date:** 2026-06-26
**Status:** Approved (brainstorm complete; ready for implementation plan)
**Branch:** `feat/recommend-discard` (off `main`; do NOT merge without user OK)

## Goal

Make the pipeline produce explicit, **fully automated** "throw out the junk"
recommendations. For every minor fragment we already diagnose a `likely_cause`;
this feature derives a per-cluster `recommended_disposition` ∈
{`DISCARD`, `KEEP`, `UNCERTAIN`} from that diagnosis and surfaces it in the
existing TSV outputs, a new `discard_cells.tsv`, and a report section.

**No human-in-the-loop.** The pipeline never pauses for review. The output is a
*recommendation* a human consumes afterward (e.g. to subset the AnnData); the
tool always commits to a disposition on its own.

## Background (current state, verified)

- `diagnosis.py` produces one `DiagnosisResult` per minor fragment.
  `likely_cause` is validated against `ALLOWED_CAUSES` (currently **6** values:
  `sample-driven`, `doublet-driven`, `low-quality (high mt)`, `shallow-depth`,
  `biology-candidate`, `unclear`) in `DiagnosisResult.__post_init__`.
- Two engines: `RuleDiagnosisEngine` (deterministic cascade) and
  `LLMDiagnosisEngine`. Default `diagnosis_mode` is `llm`; the rule baseline is
  always computed and carried as `rule_baseline`, with `llm_overrode_rule`
  recording whether the LLM changed the cause.
- `DiagnosisResult` already has a `confidence: float` in `[0,1]`
  (`to_panel_fields()` renames it `diagnosis_confidence`). Rule path assigns
  discrete confidences (0.35 unclear, 0.65 biology-candidate, 0.85
  doublet/mt/shallow, 0.90 sample-driven); LLM path takes the model's number.
- `MinorEvidence` carries `top_up_genes` / `top_down_genes` (gene-level DEG),
  composition enrichment, and qc_drift — enough for the LLM to recognize
  expression signatures.
- Outputs flow through `_Layout` (pipeline.py): root `panel.tsv`,
  `diagnosis_all.tsv`, `cell_labels.tsv`, plus per-cluster `panel.tsv`;
  `report.html` is built separately by `build_report` (report.py), invoked from
  the CLI post-run.
- There is **no** existing disposition / discard concept anywhere — greenfield.
- Downstream `annotate.py` reads only a fixed 4 columns from `panel.tsv`, so
  adding columns is safe.

## Design decisions (locked during brainstorm)

### D1. Three-tier disposition, no human gate

`recommended_disposition` ∈ {`DISCARD`, `KEEP`, `UNCERTAIN`} (uppercase, for
visual prominence in tables; distinct from the lowercase `likely_cause` enum).

- `DISCARD` — confidently junk. These cells (and only these) populate
  `discard_cells.tsv`.
- `KEEP` — real biology / clean; retained.
- `UNCERTAIN` — **kept + flagged.** The tool's default is to *retain* these
  cells (they are NOT in `discard_cells.tsv`); the label only flags them as
  borderline for the human reading the report. This is NOT a "human must
  review" task — the pipeline never blocks. (We deliberately do **not** use the
  word "REVIEW", which implies a human step.)

### D2. Taxonomy expansion: 6 → 11 causes

Five new causes are added to `ALLOWED_CAUSES`. **All five are LLM-only** — they
are detected by the LLM from the cluster's DEG, not by the rule cascade.
Rationale for LLM detection: the discriminating signatures are gene-name based
and **species-specific** (mouse `Fos` vs human `FOS`, `Hba-a1` vs `HBA1`,
`Mki67` vs `MKI67`, …); hardcoding gene tables would require per-species
maintenance and would miss aliases/orthologs, whereas the LLM recognizes
species-appropriate orthologs natively. The rule path is **unchanged** and
still emits only the original 6 causes; since the default mode is `llm`, the new
causes are available by default.

| # | likely_cause | disposition (baseline) | detection | signature / notes |
|---|---|---|---|---|
| 1 | `sample-driven` | `UNCERTAIN` | rule + llm | batch/sample enrichment |
| 2 | `doublet-driven` | `DISCARD` | rule + llm | physical doublets; UMI/score signal. **Foreign-compartment markers WITH doublet/UMI signal are judged here.** |
| 3 | `low-quality (high mt)` | `DISCARD` | rule + llm | high mito % |
| 4 | `shallow-depth` | `DISCARD` | rule + llm | depth-dominated (user chose DISCARD; the confidence gate is its safety net in llm mode) |
| 5 | `dissociation-effect` | `DISCARD` | **llm** | immediate-early / stress genes: AP-1 (FOS/FOSB/JUN/JUNB/JUND), EGR1, heat-shock proteins (HSPA1A/B, HSPB1, DNAJB1), SOCS3, ZFP36, IER2 |
| 6 | `cell-cycle` | `KEEP` | **llm** | proliferation/cell-cycle genes: MKI67, TOP2A, CCNB1/2, CDK1, PCNA, CENPF, UBE2C, BIRC5, histones. Real cells, spurious split → keep |
| 7 | `ambient-contamination` | `DISCARD` | **llm** | contaminant transcripts not native to the cluster: hemoglobin (HBA/HBB) for RBC ambient; OR a *different* cellular compartment's markers in this cluster (e.g. epithelial EPCAM/keratins or stromal COL1A1/PDGFRB in an immune cluster) that are diffuse and **without** doublet/UMI inflation |
| 8 | `sex-driven` | `KEEP` | **llm** | sex-linked genes: XIST; Y genes (RPS4Y1/DDX3Y/UTY/EIF1AY; mouse Ddx3y/Uty/Eif2s3y/Kdm5d). Real biology → keep |
| 9 | `interferon-response` | `KEEP` | **llm** | type I/II ISGs: ISG15, IFIT1/2/3, MX1/2, OAS family, STAT1, IRF7, RSAD2, IFITM3. Real biology → keep |
| 10 | `biology-candidate` | `KEEP` | rule + llm | real biology |
| 11 | `unclear` | `UNCERTAIN` | rule + llm | nothing met threshold |

Disposition tally: DISCARD = {doublet-driven, low-quality (high mt),
shallow-depth, dissociation-effect, ambient-contamination}; KEEP = {cell-cycle,
sex-driven, interferon-response, biology-candidate}; UNCERTAIN = {sample-driven,
unclear}.

Final cause strings (kebab-case, mirroring existing style):
`dissociation-effect`, `cell-cycle`, `ambient-contamination`, `sex-driven`,
`interferon-response`.

### D3. Hybrid disposition with a conservative-only invariant

The disposition is computed deterministically from the **final** `likely_cause`,
with the LLM permitted to override **only toward more conservative** (more
keep-leaning) outcomes.

Define conservativeness rank: `DISCARD` = 0 < `UNCERTAIN` = 1 < `KEEP` = 2.

Computation order, per fragment:

1. `disposition_baseline = DISPOSITION_MAP[likely_cause]`.
2. **Override (conservative-only):** if the LLM supplied a disposition
   `llm_disp`, accept it iff `rank(llm_disp) >= rank(baseline)` (i.e. it relaxes
   toward KEEP, or is equal); otherwise ignore it and keep `baseline`. In rule
   mode there is no `llm_disp`, so the baseline stands.
   `candidate = llm_disp if accepted else baseline`.
3. **Confidence gate:** if `candidate == DISCARD` and
   `confidence < threshold` (default `0.5`, CLI-tunable) → `final = UNCERTAIN`;
   else `final = candidate`. (The gate, too, only moves toward KEEP.)
4. `recommended_disposition = final`;
   `disposition_overridden = (final != disposition_baseline)`;
   `disposition_reason` carries the explanation (LLM rationale if it relaxed the
   call; the rule template `"<cause> → <baseline> (rule baseline)"` otherwise;
   with `" (downgraded: low confidence)"` appended when the gate fired).

**Invariant (the guarantee for "must keep"):** automated adjustments (LLM
override + confidence gate) can only move a disposition toward KEEP, never
toward DISCARD. Consequently a `KEEP`- or `UNCERTAIN`-baseline cause can **never**
be auto-escalated to `DISCARD`; the four real-biology causes (cell-cycle,
sex-driven, interferon-response, biology-candidate) are never auto-discarded.
The *only* path to `DISCARD` is a high-confidence `DISCARD`-cause baseline. To
mark a cluster as junk, the LLM must instead pick a `DISCARD` **cause**
(cause-level override, already tracked by `llm_overrode_rule`), not raise the
disposition.

### D4. LLM prompt: function-based, cross-species signatures

`build_llm_prompt` is extended so the schema advertises the 11-value enum
(auto-propagates from `ALLOWED_CAUSES`) plus two new fields,
`recommended_disposition` (enum DISCARD/KEEP/UNCERTAIN) and
`disposition_reason` (string). The prompt:

- describes each cause's signature by **function/family with example genes**,
  explicitly instructing the model to recognize species-appropriate orthologs
  (no fixed gene list);
- states the cause→disposition default map and the **conservative-only override
  rule** ("you may relax a disposition toward KEEP when evidence supports it;
  to mark a cluster as junk, choose a discard-type *cause* — do not raise the
  disposition");
- for `ambient-contamination` vs `doublet-driven`: foreign-compartment markers
  **with** doublet/UMI signal → `doublet-driven`; diffuse contaminant
  transcripts **without** doublet signal → `ambient-contamination`.

`PROMPT_VERSION` is bumped to `standissect-diagnosis-v2`.

## Component / file-level design

### diagnosis.py

- `ALLOWED_CAUSES`: append the 5 new strings.
- `DISPOSITION_MAP: dict[str, str]`: module-level, the 11-entry table above.
  `_DISPOSITION_RANK = {'DISCARD': 0, 'UNCERTAIN': 1, 'KEEP': 2}`.
- `derive_disposition(likely_cause, confidence, *, threshold, llm_disposition=None, llm_reason=None) -> (recommended, baseline, overridden, reason)`:
  pure function implementing D3 steps 1–4. Unit-tested in isolation.
- `DiagnosisResult`: add fields `disposition_baseline: str = ''`,
  `recommended_disposition: str = ''`, `disposition_overridden: bool = False`,
  `disposition_reason: str = ''`. `__post_init__` sets `disposition_baseline =
  DISPOSITION_MAP[self.likely_cause]` (after the cause validation). A
  `finalize_disposition(threshold, llm_disposition=None, llm_reason=None)`
  method calls `derive_disposition` and fills the remaining three fields. (The
  threshold is not known at construction, so finalize is a separate call.)
- `to_panel_fields()`: add the four new keys
  (`recommended_disposition`, `disposition_baseline`, `disposition_overridden`,
  `disposition_reason`).
- `RuleDiagnosisEngine`: unchanged cause cascade; after building the result,
  call `result.finalize_disposition(threshold)` (no llm disposition). Threshold
  comes from a new constructor arg.
- `LLMDiagnosisEngine` / `build_llm_prompt` / `_diagnosis_from_dict`: parse
  `recommended_disposition` + `disposition_reason` from the model JSON
  (tolerate missing/invalid → treat as no override), then call
  `finalize_disposition(threshold, llm_disposition=..., llm_reason=...)`. The
  conservative-only clamp lives in `derive_disposition`, so a model that returns
  a DISCARD escalation is silently clamped to the baseline.
- `make_diagnosis_engine(...)`: new `discard_confidence_threshold=0.5` param,
  threaded to both engines.

### pipeline.py

- `_PANEL_COLS` / `_DIAGNOSIS_COLS`: append the four disposition columns
  (after the existing diagnosis columns) so they flow into `panel.tsv`,
  per-cluster `panel.tsv`, and `diagnosis_all.tsv`. `_ordered_panel` keeps order.
- `_apply_diagnosis_to_cluster_panel`: accept/forward
  `discard_confidence_threshold` to the engine; `to_panel_fields()` already
  carries the disposition, so the existing `row_dict.update(...)` suffices.
- `cell_labels.tsv`: add one column `recommended_disposition`, joined per cell
  via `cell_labels.original_cluster_split == panel.subcluster`. Cells whose
  subcluster has no panel row (if any) get empty string.
- `discard_cells.tsv` (NEW, root level; add to `_Layout`): one row per cell
  whose cluster's `recommended_disposition == 'DISCARD'`. Columns: `barcode`,
  `subcluster`, `parent_cluster`, `likely_cause`, `diagnosis_confidence`,
  `disposition_reason`. Written from the joined cell→panel table at the terminal
  aggregation block. If no DISCARD cells exist, write a header-only file.
- `params.json`: record `discard_confidence_threshold`.
- `run_dissect_pipeline(...)`: new `discard_confidence_threshold=0.5` param,
  threaded to diagnosis.

### report.py

- `build_report`: add a `<h2 id="discards">Recommended discards</h2>` section
  plus a sidebar `<a href="#discards">` anchor, placed right after the overview
  panel table. Content:
  - one-line summary: `N clusters recommended for discard, M cells total`, with
    a per-cause count breakdown;
  - a **DISCARD cluster table** (one row per DISCARD cluster): `subcluster`,
    `n_cells`, `likely_cause`, `diagnosis_confidence`, `disposition_reason`;
  - a smaller **flagged (UNCERTAIN)** list, for awareness (kept by default).
  - If there are no DISCARD clusters, the section says so explicitly.
- Note: because the disposition columns now live in `panel.tsv`, the existing
  overview and per-cluster panel tables (rendered by `_table`) will also show
  them. This is intended (disposition visible in context).

### cli.py

- Add `--discard-confidence-threshold` (float, default `0.5`), passed through
  `run_cmd` to `run_dissect_pipeline`. Help text: "DISCARD calls below this
  diagnosis confidence are downgraded to UNCERTAIN (kept + flagged)."

## Data flow

diagnose (rule baseline always; LLM in llm mode) → `finalize_disposition`
(baseline → conservative-only override → confidence gate) → `to_panel_fields`
→ per-cluster `panel.tsv` → root `panel.tsv` / `diagnosis_all.tsv` → join onto
`cell_labels.tsv` (per cell) → filter DISCARD → `discard_cells.tsv` → report
"Recommended discards" section.

## Error handling / edge cases

- LLM omits `recommended_disposition` or returns a value outside the enum →
  treated as "no override" → baseline stands (then gate). Logged in reason.
- LLM tries to escalate (e.g. baseline KEEP, llm DISCARD) → clamped to baseline
  by the conservative-only rule; reason notes the rejected escalation.
- `rule` mode → no override; disposition = baseline (then gate). The 5 new
  causes never appear (rule cascade unchanged).
- LLM diagnosis failure → existing `rule-fallback` path; the fallback result
  still gets `finalize_disposition(threshold)` so it carries a disposition.
- No DISCARD cells → `discard_cells.tsv` is header-only; report section states
  "no clusters recommended for discard".
- A subcluster present in `cell_labels` but absent from `panel` → empty
  disposition for those cells (defensive; should not happen).

## Testing

- `derive_disposition`: each of the 11 causes → expected baseline; conservative
  override accepted (DISCARD baseline + llm KEEP → KEEP) and rejected (KEEP
  baseline + llm DISCARD → KEEP; UNCERTAIN + llm DISCARD → UNCERTAIN);
  confidence gate (DISCARD + low conf → UNCERTAIN; DISCARD + high conf →
  DISCARD; KEEP/UNCERTAIN never gated).
- `DISPOSITION_MAP` has exactly one entry per `ALLOWED_CAUSES` value (no
  missing, no extra) — guards taxonomy/map drift.
- `DiagnosisResult`: `disposition_baseline` set in `__post_init__`;
  `to_panel_fields` emits the 4 keys; the 5 new causes validate.
- Rule engine: result carries `recommended_disposition == baseline`,
  `disposition_overridden == False`.
- LLM parse: `_diagnosis_from_dict` reads disposition fields; missing/invalid →
  baseline; escalation clamped.
- `pipeline`: `panel.tsv` has the 4 columns in order; `cell_labels.tsv` has
  `recommended_disposition` correctly joined; `discard_cells.tsv` contains
  exactly the DISCARD cells with correct columns; empty-DISCARD → header-only.
- `report`: "Recommended discards" section + sidebar anchor render; summary
  counts correct; no-DISCARD message path.
- `cli`: `--discard-confidence-threshold` parsed and recorded in `params.json`.
- e2e: Marrow run (local, on the compute node — NO srun/sbatch) produces
  `discard_cells.tsv` + the report section; serial vs concurrent unaffected.

## Out of scope (YAGNI)

- Per-cause CLI overrides of `DISPOSITION_MAP` (map is fixed in code).
- Rule-path detection of the 5 new causes (LLM-only by design).
- Actually subsetting/writing a cleaned `.h5ad` — this feature only *recommends*
  (emits `discard_cells.tsv`); the user acts on it. Auto-removal could be a
  future opt-in flag.

## Open items for spec review

- `panel.tsv` gains **4** disposition columns (full parity with the existing
  `rule_baseline` / `llm_overrode_rule` audit columns). If a leaner panel is
  preferred, `disposition_baseline` + `disposition_overridden` could move to the
  per-minor JSON / `diagnosis_all.tsv` only — flag at review.
