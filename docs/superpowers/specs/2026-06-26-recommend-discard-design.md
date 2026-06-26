# recommend-discard — Design Spec

**Date:** 2026-06-26
**Status:** Approved (brainstorm complete; ready for implementation plan)
**Branch:** `feat/recommend-discard` (off `main`; do NOT merge without user OK)

## Goal

Make the pipeline produce explicit, **fully automated** recommendations about
what to do with every minor fragment, on two complementary axes:

1. **Discard junk.** Derive a per-cluster `recommended_disposition` ∈
   {`DISCARD`, `KEEP`, `UNCERTAIN`} from the existing minor-cause diagnosis and
   surface it in the TSV outputs + a new `discard_cells.tsv` + a report section.
   The discard recommendation is **precise to the cell** (`barcode` +
   input-adata row index), not just a cluster name.
2. **Preserve novel biology.** Capture **LLM-proposed cell types** — a finer or
   different identity the LLM proposes for a minor fragment (e.g. pDC inside a
   myeloid parent) or for a major core (the core looks mislabeled) versus the
   original annotation — so newly recognized cell types are recorded rather
   than lost.

**No human-in-the-loop.** The pipeline never pauses for review. Outputs are
*recommendations* a human consumes afterward; the tool always commits to a
disposition / proposal on its own.

## Background (current state, verified)

- `diagnosis.py` produces one `DiagnosisResult` per minor fragment.
  `likely_cause` is validated against `ALLOWED_CAUSES` (currently **6**) in
  `DiagnosisResult.__post_init__`; `to_panel_fields()` feeds the panel TSV.
  `DiagnosisResult` already has `confidence: float` in `[0,1]`.
- Two diagnosis engines: `RuleDiagnosisEngine` (deterministic cascade) and
  `LLMDiagnosisEngine`. Default `diagnosis_mode` is `llm`; the rule baseline is
  always computed (`rule_baseline`), with `llm_overrode_rule` recording cause
  changes. `MinorEvidence` carries `top_up_genes`/`top_down_genes` and
  `parent_cluster` — enough for the LLM to recognize signatures and to know the
  fragment's parent.
- `annotate.py` names canonical cores: `LLMNamingEngine`/`LocalNamingEngine`
  produce a `CoreNaming` per core; `run_naming_stage` writes `core_names.tsv`
  (`CORE_NAME_COLS = parent_cluster, core_subcluster, cell_type, confidence,
  rationale, source, model`). `CoreEvidence` already carries `parent_cluster`.
- **`parent_cluster` IS the original annotation:** it equals the value of the
  user's `cluster_col`. So "differs from the original label" is a comparison
  against `parent_cluster` for both minors and majors — no extra lookup needed.
- `run_dissect_pipeline(adata, ...)` holds the input `adata` in memory; output
  paths funnel through `_Layout`. `cell_labels.tsv` is written per cell with
  index = `adata.obs_names` (barcodes) and column `original_cluster_split` (the
  subcluster, `c{parent}_{rank}`, equal to `panel.subcluster`). `report.html`
  is built separately by `build_report` (report.py).
- There is **no** existing disposition / discard / proposed-type concept
  (greenfield). Downstream `annotate.py` reads only a fixed 4 columns from
  `panel.tsv`, so adding columns is safe.

## Design decisions (locked during brainstorm)

### D1. Three-tier disposition, no human gate

`recommended_disposition` ∈ {`DISCARD`, `KEEP`, `UNCERTAIN`} (uppercase; the
`likely_cause` enum stays lowercase). `DISCARD` cells (and only those) populate
`discard_cells.tsv`. `KEEP` = retained. `UNCERTAIN` = **kept + flagged** (the
tool retains them; the label only flags them as borderline — never a human
task, never a pipeline pause). We avoid the word "REVIEW".

### D2. Taxonomy expansion: 6 → 11 causes

Five new causes are added to `ALLOWED_CAUSES`. **All five are LLM-only** —
detected by the LLM from the cluster's DEG, not by the rule cascade (the
discriminating signatures are species-specific gene names; the LLM recognizes
orthologs natively; hardcoding gene tables would need per-species maintenance).
The rule cascade in `RuleDiagnosisEngine.diagnose` is NOT extended; since the
default mode is `llm`, the new causes are available by default.

| # | likely_cause | disposition (baseline) | detection | signature / notes |
|---|---|---|---|---|
| 1 | `sample-driven` | `UNCERTAIN` | rule + llm | batch/sample enrichment |
| 2 | `doublet-driven` | `DISCARD` | rule + llm | physical doublets; UMI/score signal. Foreign-compartment markers WITH doublet/UMI signal are judged here. |
| 3 | `low-quality (high mt)` | `DISCARD` | rule + llm | high mito % |
| 4 | `shallow-depth` | `DISCARD` | rule + llm | depth-dominated (user chose DISCARD; the confidence gate is its safety net in llm mode) |
| 5 | `dissociation-effect` | `DISCARD` | **llm** | immediate-early / stress genes: AP-1 (FOS/FOSB/JUN/JUNB/JUND), EGR1, heat-shock proteins (HSPA1A/B, HSPB1, DNAJB1), SOCS3, ZFP36, IER2 |
| 6 | `cell-cycle` | `KEEP` | **llm** | proliferation/cell-cycle genes: MKI67, TOP2A, CCNB1/2, CDK1, PCNA, CENPF, UBE2C, BIRC5, histones. Real cells, spurious split → keep |
| 7 | `ambient-contamination` | `DISCARD` | **llm** | contaminant transcripts not native to the cluster: hemoglobin (HBA/HBB) for RBC ambient; OR a *different* cellular compartment's markers in this cluster (e.g. epithelial EPCAM/keratins or stromal COL1A1/PDGFRB in an immune cluster) that are diffuse and **without** doublet/UMI inflation |
| 8 | `sex-driven` | `KEEP` | **llm** | sex-linked genes: XIST; Y genes (RPS4Y1/DDX3Y/UTY/EIF1AY; mouse Ddx3y/Uty/Eif2s3y/Kdm5d). Real biology → keep |
| 9 | `interferon-response` | `KEEP` | **llm** | type I/II ISGs: ISG15, IFIT1/2/3, MX1/2, OAS family, STAT1, IRF7, RSAD2, IFITM3. Real biology → keep |
| 10 | `biology-candidate` | `KEEP` | rule + llm | real biology |
| 11 | `unclear` | `UNCERTAIN` | rule + llm | nothing met threshold |

Final cause strings: `dissociation-effect`, `cell-cycle`,
`ambient-contamination`, `sex-driven`, `interferon-response`.

### D3. Hybrid disposition with a conservative-only invariant

Disposition is computed deterministically from the **final** `likely_cause`,
with the LLM permitted to override **only toward more conservative** (more
keep-leaning) outcomes. Rank: `DISCARD` = 0 < `UNCERTAIN` = 1 < `KEEP` = 2.

Per fragment:
1. `disposition_baseline = DISPOSITION_MAP[likely_cause]`.
2. **Override (conservative-only):** an LLM disposition `llm_disp` is accepted
   iff `rank(llm_disp) >= rank(baseline)`; otherwise ignored (baseline stands).
   Rule mode has no `llm_disp`.
3. **Confidence gate:** if the candidate is `DISCARD` and
   `confidence < threshold` (default `0.5`, CLI-tunable) → `UNCERTAIN`.
4. `recommended_disposition` = result; `disposition_overridden = (final !=
   baseline)`; `disposition_reason` explains (LLM rationale when relaxed; rule
   template otherwise; `" (downgraded: low confidence)"` appended when gated).

**Invariant:** automated adjustments (LLM override + gate) can only move toward
KEEP, never toward DISCARD. A `KEEP`/`UNCERTAIN`-baseline cause can never be
auto-escalated to `DISCARD`; the only path to `DISCARD` is a high-confidence
`DISCARD`-cause baseline. To mark a cluster as junk the LLM must pick a
`DISCARD` **cause** (cause-level override, tracked by `llm_overrode_rule`).

### D4. LLM diagnosis prompt: function-based, cross-species signatures

`build_llm_prompt` advertises the 11-value enum (auto-propagated) plus
`recommended_disposition` (enum) and `disposition_reason`. The prompt describes
each cause's signature by **function/family with example genes** and instructs
the model to recognize species-appropriate orthologs (no fixed gene list); it
states the cause→disposition map and the conservative-only override rule.
`PROMPT_VERSION` → `standissect-diagnosis-v2`.

### D5. Preserve LLM-proposed cell types (minor + major)

A second, orthogonal axis: capture identities the LLM proposes that differ from
the original annotation (`parent_cluster`). Disposition is unaffected (these are
real biology → KEEP).

- **Minor (diagnosis):** add `proposed_cell_type` (string or null) to the
  diagnosis LLM schema + `DiagnosisResult`. The LLM fills it when a minor is a
  **real, distinct/finer cell type than its parent** (e.g. `pDC` inside a
  myeloid parent); null for pure states (cell-cycle/sex/interferon) and
  artifacts. Flows into `panel.tsv`.
- **Major (naming):** add `differs_from_original` (bool) to the core-naming LLM
  schema + `CoreNaming`. The naming evidence already carries `parent_cluster`
  (the original annotation); the prompt instructs the model to set it true when
  its `cell_type` denotes a **semantically different identity** than
  `parent_cluster` (the LLM judges synonymy, e.g. "T cell" vs "T cells").
  `core_names.tsv` gains `original_label` (= `parent_cluster`) and
  `differs_from_original`. `NAMING_PROMPT_VERSION` → `standissect-naming-v2`.
- **Aggregation:** a new `proposed_cell_types.tsv` (root) collects both — minor
  rows (every panel row with a non-null `proposed_cell_type`) and major rows
  (every `core_names` row with `differs_from_original == True`). Columns:
  `level` (`minor`/`major`), `parent_cluster` (the original label),
  `subcluster`, `proposed_cell_type`, `confidence`, `rationale`. A report
  subsection lists them.
- **Caveat:** the major `differs_from_original` signal is only meaningful when
  `cluster_col` holds cell-type names. When it is a numeric cluster id, any
  cell-type name "differs"; the row still records `original_label` so the user
  can judge. Documented, not gated.

### D6. Discard recommendation precise to the cell

`discard_cells.tsv` is keyed by **`barcode`** (the `obs_name` of the
standissect-input adata, which is the user's QC-filtered object) — a stable
identifier that survives any later filtering/reordering (match by ID, never by
position). It also carries **`input_row_index`**: the 0-based integer position
of that cell in the standissect-input adata, taken directly from
`adata.obs_names` (the in-memory input object). Columns: `barcode`,
`input_row_index`, `subcluster`, `parent_cluster`, `likely_cause`,
`diagnosis_confidence`, `disposition_reason`. Documented contract: `barcode` is
the cross-version key; `input_row_index` indexes the standissect-input
(post-QC-filter) object only.

## Component / file-level design

### diagnosis.py

- `ALLOWED_CAUSES`: append the 5 new strings.
- `DISPOSITION_MAP: dict[str,str]` (11 entries) + `_DISPOSITION_RANK`.
- `derive_disposition(likely_cause, confidence, *, threshold, llm_disposition=None, llm_reason=None) -> (recommended, baseline, overridden, reason)` — pure, conservative-only + gate.
- `DiagnosisResult`: add `disposition_baseline`, `recommended_disposition`,
  `disposition_overridden`, `disposition_reason`, **`proposed_cell_type`**.
  `__post_init__` sets `disposition_baseline = DISPOSITION_MAP[likely_cause]`
  and defaults `recommended_disposition` to it; `finalize_disposition(threshold,
  *, llm_disposition=None, llm_reason=None)` applies override + gate.
  `to_panel_fields()` emits the 4 disposition keys + `proposed_cell_type`.
- LLM path: `build_llm_prompt` schema gains `recommended_disposition`,
  `disposition_reason`, **`proposed_cell_type`** + `CAUSE_SIGNATURES` /
  `DISPOSITION_POLICY` guidance; `_diagnosis_from_dict(..., threshold=0.5)`
  parses them, finalizes disposition, and sets `proposed_cell_type`.
- Engines: `RuleDiagnosisEngine` / `LLMDiagnosisEngine` /
  `make_diagnosis_engine` gain `discard_confidence_threshold=0.5`; engines call
  `finalize_disposition` before returning. (Rule path leaves
  `proposed_cell_type=None`.)

### annotate.py

- `CoreNaming`: add `differs_from_original: bool = False`.
- `build_core_naming_prompt`: schema gains `differs_from_original`; system
  prompt instructs comparison against `evidence.parent_cluster` (already in the
  evidence payload). `_core_naming_from_dict` parses it.
- `to_core_name_row(evidence)`: add `original_label = evidence.parent_cluster`
  and `differs_from_original`.
- `run_naming_stage`: include `original_label` (= the parent) and
  `differs_from_original` in the rows; `CORE_NAME_COLS` gains the two columns.
- `NAMING_PROMPT_VERSION` → `standissect-naming-v2`.
- Local backup (`LocalNamingEngine`): leaves `differs_from_original=False`
  (no original-label judgment without the LLM).

### pipeline.py

- `_PANEL_COLS` / `_DIAGNOSIS_COLS`: append the 4 disposition columns +
  `proposed_cell_type`.
- `_Layout`: add `discard_cells` and `proposed_cell_types` properties.
- `_write_cell_dispositions(lay, panel, obs_names)`: join
  `recommended_disposition` onto `cell_labels.tsv` (per cell); write
  `discard_cells.tsv` for DISCARD cells with `barcode` + `input_row_index`
  (position from `obs_names`). Called after `diagnosis_all` is written
  (pipeline.py ~L877), passing `adata.obs_names`.
- `_write_proposed_cell_types(lay, panel, core_names_df)`: assemble
  `proposed_cell_types.tsv` from minor (`panel.proposed_cell_type`) + major
  (`core_names.differs_from_original`). Called after the naming stage
  (pipeline.py ~L917, where `core_names_df` is read).
- `run_dissect_pipeline(...)`: new `discard_confidence_threshold=0.5`, passed to
  `make_diagnosis_engine`; recorded in `params.json`.

### report.py

- `_discards_section(root)`: `<h2 id="discards">Recommended discards</h2>` —
  summary (N clusters, M cells, by-cause), a DISCARD-cluster table, and a
  collapsed UNCERTAIN (flagged-kept) list; explicit "none" message when empty.
- `_proposed_types_section(root)`: `<h2 id="proposed">Proposed new / re-labeled
  cell types</h2>` — a table from `proposed_cell_types.tsv` (level / parent /
  subcluster / proposed_cell_type / confidence / rationale); "none" when empty.
- `build_report`: append both sections after the overview; add matching sidebar
  anchors (`#discards`, `#proposed`).

### cli.py

- `--discard-confidence-threshold` (float, default `0.5`), passed through
  `run_cmd`. Help: "DISCARD calls below this diagnosis confidence are downgraded
  to UNCERTAIN (kept + flagged)."

## Data flow

diagnose (rule baseline always; LLM in llm mode) → `finalize_disposition`
(baseline → conservative-only override → gate) + `proposed_cell_type` →
`to_panel_fields` → per-cluster `panel.tsv` → root `panel.tsv` /
`diagnosis_all.tsv` → (a) join onto `cell_labels.tsv`, filter DISCARD →
`discard_cells.tsv` (barcode + input_row_index); naming → `core_names.tsv`
(+ original_label / differs_from_original) → (b) minor + major →
`proposed_cell_types.tsv`. Report renders both new sections.

## Error handling / edge cases

- LLM omits/invalid `recommended_disposition` → no override → baseline (+ gate).
- LLM escalation attempt → clamped to baseline (conservative-only).
- LLM omits `proposed_cell_type` → null (no proposal); rule mode → always null.
- LLM omits `differs_from_original` → False (no relabel proposed); local/unnamed
  cores → False.
- `rule` mode → disposition = baseline (+ gate); the 5 new causes never appear.
- LLM diagnosis failure → `rule-fallback`; the fallback result is finalized so
  it still carries a disposition (and `proposed_cell_type=None`).
- No DISCARD cells → `discard_cells.tsv` header-only; report says "none".
- No proposed types → `proposed_cell_types.tsv` header-only; report says "none".
- A subcluster present in `cell_labels` but absent from `panel` → empty
  disposition (defensive).

## Testing

- `derive_disposition`: each of 11 causes → baseline; conservative override
  accept/reject; gate (DISCARD+low→UNCERTAIN; high→DISCARD; KEEP/UNCERTAIN never
  gated). `DISPOSITION_MAP` keys == `ALLOWED_CAUSES`.
- `DiagnosisResult`: disposition fields + `proposed_cell_type` in
  `to_panel_fields`; 5 new causes validate; `proposed_cell_type` round-trips.
- LLM parse: disposition parsed/clamped/missing→baseline; `proposed_cell_type`
  parsed (and null when absent); prompt schema advertises the new keys + 11
  causes + the conservative-only rule.
- Naming: `build_core_naming_prompt` schema has `differs_from_original` + cites
  `parent_cluster`; `_core_naming_from_dict` parses it; `run_naming_stage` rows
  / `CORE_NAME_COLS` include `original_label` + `differs_from_original`.
- `pipeline`: `panel.tsv` column order incl. the 5 new columns;
  `_write_cell_dispositions` → `cell_labels` gains `recommended_disposition`,
  `discard_cells.tsv` has only DISCARD cells with `barcode` + `input_row_index`
  (matching `obs_names` positions); empty-DISCARD header-only;
  `_write_proposed_cell_types` → minor + major rows correct; empty header-only.
- `report`: both sections + anchors render; counts correct; no-data messages.
- `cli`: `--discard-confidence-threshold` default/override.
- e2e: Marrow run (local, NO srun/sbatch) produces `discard_cells.tsv` (with
  `input_row_index`), `proposed_cell_types.tsv`, both report sections,
  `params.json['discard_confidence_threshold']`, prompt versions v2.
  Invariant spot-check: no KEEP-cause cluster in `discard_cells.tsv`.

## Out of scope (YAGNI)

- Per-cause CLI overrides of `DISPOSITION_MAP`.
- Rule-path detection of the 5 new causes (LLM-only by design).
- Auto-writing a cleaned `.h5ad` — this feature only *recommends*
  (`discard_cells.tsv`); the user acts on it.
- Emitting a pre-QC-filter (original raw) index — out of standissect's
  knowledge; `barcode` is the bridge. (Could be a future opt-in
  `--original-index-col`.)

## Panel-column decision

`panel.tsv` / `diagnosis_all.tsv` gain the 4 disposition columns
(`disposition_baseline`, `recommended_disposition`, `disposition_overridden`,
`disposition_reason`) — full parity with the existing `rule_baseline` /
`llm_overrode_rule` audit columns — plus `proposed_cell_type`. Approved at spec
review.
