# Core cell-type naming + per-cluster narrative — design spec

**Date:** 2026-06-25
**Scope:** Add two LLM annotation features on top of standissect's existing
per-minor diagnosis — (1) canonical-core **cell-type naming**, (2) per-cluster
**narrative** — and flip the default `diagnosis_mode` to `llm`. New module
`annotate.py`, reusing the existing ARK chat client. The May snapshot at
`synovial.prep-231031/standissect` is **not** touched (separate, stale copy).

## 1. Goal & background

`diagnosis.py` already turns a compact per-minor evidence packet into a
`likely_cause` via `RuleDiagnosisEngine` / `LLMDiagnosisEngine` over an ARK
`OpenAICompatClient`, writing results into `panel.tsv` and
`clusters/cN/diagnosis_<sub>.{input,output}.json`, idempotent on
`prompt_version`+`mode`+`model` (`pipeline.py:_diagnosis_current`). It does **not**:

- **name** the canonical core `c{N}_0` — markers exist in
  `canonical_markers/markers_c{N}_0.tsv`, but no cell type is assigned;
- produce a per-cluster **narrative**.

This spec adds both, mirroring the diagnosis pattern (evidence packet → engine
over a chat client → result dataclass → pipeline stage → report), and makes LLM
diagnosis the default.

## 2. Design decisions (user-approved)

1. **`diagnosis_mode` default `rule` → `llm`** in both `run_dissect_pipeline`
   and `cli.py`. ARK is the only LLM backend (`ArkChatClient`, `DEFAULT_ARK_*`).
   **Graceful construction-time degrade:** if no client can be built (missing
   `ARK_API_KEY` and no injected client), the run **falls back to `rule`** with a
   warning instead of raising — partition/dissect/canonical/profile and rule
   diagnosis still complete. (`fallback_to_rule` already covers per-call
   failures; this adds the construction-time guard so the default flip can't hard
   -crash a keyless run.)
2. **New module `annotate.py`** — keeps `diagnosis.py` focused; reuses the shared
   chat client.
3. **Naming uses a knowledge-allowed prompt** — the deliberate opposite of the
   diagnosis system prompt's "Use only the supplied statistical evidence. Do not
   invent ... cell types, markers ..." (`diagnosis.py:490`). Naming **must** map
   markers→cell type from cell-biology knowledge; guarded by a required
   `confidence`, an allowed `uncertain`, and a rule to cite only markers from the
   supplied list (no invented markers). **Narrative stays evidence-grounded** (no
   new cell types or causes beyond what is supplied) — same stance as diagnosis.
4. **Naming = LLM primary, local backup; narrative = LLM-only** (per user: "LLM
   为主, 本地为 backup"). `LLMNamingEngine` calls ARK first; on `LLMUnavailable`
   it falls back to a `LocalNamingEngine` (overlap of the core's top markers
   against a marker table) — mirroring `LLMDiagnosisEngine(fallback_to_rule=...)`.
   **Naming therefore always runs**: LLM when a client exists, else local; both
   fail → `unnamed`. **Narrative has no local backup** (prose synthesis is
   meaningless without the model) — it runs *only* when a client exists, and is
   skipped otherwise / on failure.
5. **Marker table** for the local backup is a `naming_markers` param (a
   `dict[cell_type -> [genes]]` or a TSV path). Default = a small bundled
   `DEFAULT_MARKER_SETS` of broad synovial-relevant lineages, so the backup works
   zero-config and is overridable with a tissue-tuned table. No new ARK
   parameters; the diagnosis client is reused.

## 3. `annotate.py`

**Shared client construction.** Extract
`make_chat_client(*, mode, llm_client, ark_api_key, ark_api_key_env, ark_model,
ark_endpoint, timeout) -> client | None` (currently inline in
`make_diagnosis_engine`). Returns `None` for `rule` / no key. `make_diagnosis_engine`
is refactored to call it; the pipeline calls it **once** and shares the client
across diagnosis + the two new engines. Construction failure (no key) returns
`None` (caller degrades), not an exception.

### 3.1 Core naming
- `CoreEvidence` (dataclass): `parent_cluster`, `core_subcluster` (`c{N}_0`),
  `n_cells`, `top_markers` (list of {gene, logfoldchanges, scores} from
  `markers_c{N}_0.tsv`, top ~20 by score), `hint`.
- `CoreNaming` (dataclass): `cell_type` (str | None), `confidence` (0..1),
  `rationale`, `markers_used` (subset of supplied), `alternatives` (list),
  `source` (`'llm'` | `'local'` | `'unnamed'`), `model`, `prompt_version`
  (`standissect-naming-v1`), `error`.
- `LocalNamingEngine(markers)` — the **backup**. Scores each cell type by overlap
  of the core's top-up markers against the type's marker set (Szymkiewicz–Simpson
  `overlap_coef`, or a hypergeometric p over the gene universe; plain
  pandas/python, optionally scanpy's `sc.tl.marker_gene_overlap`). Returns the top
  type above a small threshold (`source='local'`, `confidence` = the score), else
  `cell_type=None, source='unnamed'`. No network, no new dependency.
- `LLMNamingEngine(client, *, local=None, hint='', fallback_to_local=True)` — the
  **primary**. `build_core_naming_prompt(evidence)` →
  `call_structured(client, system, user, _core_naming_from_dict)` (`source='llm'`).
  On `LLMUnavailable` → `local.name(evidence)` when `local` and
  `fallback_to_local`, else `CoreNaming(cell_type=None, source='unnamed',
  error=...)`. Never raises. Hallucination guard: `markers_used` intersected with
  the supplied list.
- `make_naming_engine(*, client, markers, hint='', fallback_to_local=True)` →
  `LLMNamingEngine(client, local=LocalNamingEngine(markers), ...)` when a client
  exists, else `LocalNamingEngine(markers)` alone.

### 3.2 Narrative
- `ClusterNarrativeEvidence` (dataclass): `parent_cluster`, `cell_type` (the core
  name), `minors` (list of {subcluster, likely_cause, cause_detail,
  diagnosis_rationale} read from the cluster panel).
- `ClusterNarrative` (dataclass): `narrative` (str), `source`
  (`'llm'` | `'skipped'`), `model`, `prompt_version`
  (`standissect-narrative-v1`), `error`.
- `NarrativeEngine(client)`: evidence-grounded prompt → `narrative`. On failure →
  `ClusterNarrative(narrative='', source='skipped', error=str(e))`. Never raises.

### 3.3 Prompts
- **Naming system:** "You are a single-cell biologist. Name the most likely cell
  type/state for a cluster from its ranked canonical marker genes, using
  established marker→cell-type knowledge. If the markers are ambiguous, return
  `uncertain` with low confidence. Cite only markers from the supplied list; do
  not introduce markers not listed. Return strict JSON." Schema: `{cell_type,
  confidence, rationale, markers_used:[subset], alternatives:[...]}`.
- **Narrative system:** "Summarize this cluster for a report using only the
  supplied facts — its core cell-type identity and each minor fragment's
  diagnosis. One concise paragraph, plain prose. Do not introduce new cell types
  or causes beyond those supplied. Return strict JSON `{narrative}`."

## 4. Pipeline integration

- `_STAGES = ('partition','dissect','diagnosis','canonical','naming','narrative','profile')`
  — naming + narrative inserted after `canonical`; `profile` stays last.
- Build the client once:
  `chat_client = make_chat_client(mode=diagnosis_mode, llm_client=diagnosis_llm_client, ...)`;
  on failure → warn, `chat_client=None`, effective diagnosis mode `rule`.
- **diagnosis stage** uses `make_diagnosis_engine(mode, llm_client=chat_client, ...)`
  — behavior unchanged.
- **naming stage** (always runs): `naming_engine = make_naming_engine(
  client=chat_client, markers=naming_markers, hint=annotation_hint)`; for each
  core `c{N}_0` (one per cluster) read `canonical_markers/markers_c{N}_0.tsv` →
  `naming_engine.name(...)` → write `clusters/cN/naming.output.json`; reassemble
  global `core_names.tsv` (parent_cluster, core_subcluster, cell_type, confidence,
  rationale, source, model). Uses LLM when `chat_client` exists, else the local
  backup.
- **narrative stage** (only if `chat_client`): for each cluster, from
  `core_names.tsv` + that cluster's `panel.tsv` minor diagnoses →
  `NarrativeEngine` → write `clusters/cN/narrative.output.json`; reassemble
  global `narratives.tsv` (parent_cluster, cell_type, narrative).
- **Idempotency:** mirror `_diagnosis_current` — `_naming_current` /
  `_narrative_current` check `prompt_version`+`model` in the saved
  `.output.json`. Add `naming` / `narrative` to `_STAGES`, `_normalize_force`,
  and the cli `--force` choices. Forcing `canonical` cascades into
  `naming`→`narrative`; forcing `naming` cascades into `narrative`; forcing
  `diagnosis` cascades into `narrative` (minor causes feed the narrative).
- `_Layout`: add `core_names` (`core_names.tsv`), `narratives`
  (`narratives.tsv`), `naming_output(parent)`, `narrative_output(parent)`.
- `params.json`: add `naming_prompt_version`, `narrative_prompt_version`,
  annotation model, `annotation_hint`.
- Return dict gains `'core_names'`, `'narratives'`.

## 5. `cli.py`

- `--diagnosis-mode` default `rule` → `llm`.
- `--force` choices add `naming`, `narrative`.
- New `--annotation-hint` (optional free text, e.g. `"synovial tissue, OA vs RA"`)
  → `run_dissect_pipeline(annotation_hint=...)` → naming/narrative prompts.
- New `--naming-markers PATH` (optional TSV, `cell_type<TAB>gene,gene,...`) →
  `run_dissect_pipeline(naming_markers=...)`; defaults to the bundled
  `DEFAULT_MARKER_SETS`.

## 6. `report.py`

- Per-cluster header: `cluster N — {cell_type}` (from `core_names.tsv`) and the
  narrative paragraph (from `narratives.tsv`), above the existing image row.
- Overview: a small `core_names.tsv` table.
- Tolerant of absent files (reuse the `_table` / `_img` missing-file handling) —
  a `rule`-mode run renders exactly as today.

## 7. Testing

- `tests/test_annotate.py` (mirror `tests/test_diagnosis_llm.py`):
  `CallableChatClient` stub returning canned JSON → `CoreNaming` /
  `ClusterNarrative` parse; fenced JSON tolerated; empty/garbled →
  `unnamed` / `skipped` (never raises); a `markers_used` entry not in the supplied
  list is dropped.
- Pipeline test (stub client): `naming` + `narrative` stages produce
  `core_names.tsv` + `narratives.tsv` and the report shows the name; a second run
  reports them in `skipped` (idempotent). `diagnosis_mode='rule'` (or no key) →
  naming/narrative skipped, run still completes (degrade path).
- Run via the venv pytest with `PYTHONPATH` set (canonical has no
  `pyproject.toml`; the package is imported by path).

## 8. Out of scope (YAGNI)

- Local naming is a coarse marker-overlap **backup**, not a full annotator (no
  SingleR / CellTypist / ScType integration); narrative has no local backup.
- No new dependencies — stdlib `OpenAICompatClient` (ARK) + plain pandas overlap;
  no pydantic (consistent with standissect's stdlib-only stance).
- `synovial.prep-231031/standissect` is untouched.
- No Batch API, streaming, retries.

## 9. Risks / notes

- **Default flip needs `ARK_API_KEY`** for the LLM path; without it the run does
  not crash — diagnosis degrades to `rule`, narrative is skipped, and **naming
  still emits local marker-overlap names**. The degrade path is explicitly tested.
- Changing `naming_markers` (or the bundled default) does **not** auto-invalidate
  cached `naming.output.json` — pass `--force naming` (consistent with the
  project's file-existence idempotency model).
- **canonical `main` has uncommitted WIP** (`README.md`/`__init__.py`/`cluster.py`
  /`pipeline.py` modified; `cli.py`/`__main__.py`/`METHODS.md` untracked).
  Implement on a **new branch**; stage only this feature's files; do **not** sweep
  the pre-existing edits. No git commit performed for this spec.
- `llm_client.py` is the vendored shared module — **not modified** here, so the
  byte-identical-with-stanmetacols invariant is unaffected.
