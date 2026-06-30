# Interactive minor-subcluster review server — design

**Date:** 2026-06-29
**Status:** approved
**Topic:** `standissect serve` — an interactive web UI to review per-cluster minor
subclusters and record human keep/discard decisions, with an interactive UMAP
for minor-vs-major comparison and free lasso selection of cells.

## Motivation

`standissect` already produces a static, self-contained `report.html`
(`report.py`). It is read-only: you can look at each cluster's minors, their
DEG/QC/composition, and the LLM diagnosis, but you cannot record your own
verdict. Reviewing a run means scrolling the report and keeping decisions in
your head or a side file.

This feature adds a small local web server that projects the same run-output
tree into an **interactive** page where the user can, per cluster:

1. inspect each minor subcluster (heatmap, UMAP, DEG, QC drift, LLM diagnosis);
2. click a minor to **compare it against the major (core `_0`) cluster**;
3. record a human **KEEP / DISCARD / UNCERTAIN** verdict (+ free-text note) per
   minor, persisted to a decisions file;
4. on an **interactive UMAP**, free-lasso/box-select cells to (a) inspect the
   selection's stats, (b) export its barcodes, and (c) record a hand-picked
   manual keep/discard set.

The server **only reads** the run tree (TSV / PNG / JSON / `cell_coords`) and
**only writes** decision/selection files. It never opens the source `.h5ad` —
so it is safe to run on a Sherlock login node and expose via an SSH tunnel or
ngrok. Coordinates for the interactive UMAP are produced by a separate,
explicitly-invoked exporter (run in `sh_dev`/a job), never by the server.

## Non-goals (YAGNI)

- **No on-the-fly DEG / re-clustering / re-embedding.** Those need the
  expression matrix and are heavy compute; out of scope for this server.
- **No applying discards / writing a cleaned `.h5ad`.** Decisions-file-only by
  design. The files are *shaped* so a future `apply-discard` could consume
  them, but that wiring is not built here.
- **No multi-user auth / locking.** Single-user tool behind a private tunnel.
  (We document ngrok basic-auth as the recommended protection.)
- **No database.** Flat TSV files in the run dir, consistent with the rest of
  the package.

## Architecture

Three new flat modules plus a static-asset directory and two new CLI
subcommands. Nothing in the existing pipeline changes; `report.py` helpers are
reused where convenient.

```
standissect/
  review_store.py        # ReviewStore + ManualStore: own the decision/manual files
  export_coords.py       # export_cell_coords(): h5ad -> cell_coords.tsv.gz (backed read)
  webreview.py           # build_app(root,...) -> FastAPI ; serve(root, host, port)
  webreview_static/
    index.html           # single-page dashboard (no build step)
    app.js               # vanilla JS: fetch APIs, render dashboard + Plotly UMAP
    style.css
    # Plotly (WebGL scatter + lasso/box select) loaded from CDN at runtime, not vendored
  cli.py                 # + `serve` and `export-coords` subcommands
```

### Unit boundaries

- **`ReviewStore`** — owns `human_review.tsv`. Loads existing rows on init
  (resume), exposes `get_all()`, `set(subcluster, parent_cluster,
  llm_disposition, human_disposition, note)`, `progress()`. Writes through
  atomically on every mutation. No web/anndata knowledge. Independently
  testable.
- **`ManualStore`** — owns `manual_cells.tsv` and `selections/`. Exposes
  `add_manual(label, barcodes, disposition)` and `write_selection(label,
  barcodes)`. Atomic writes. No web knowledge.
- **`export_cell_coords`** — pure data prep: read `obsm[umap_key]` + barcodes +
  any present QC columns from an `.h5ad` in **backed mode**, write
  `cell_coords.tsv.gz`. No web knowledge; importing it does not import FastAPI.
- **FastAPI app** — thin HTTP layer. Reads the run tree (reusing `report.py`'s
  `_read_tsv_safe` and the `core_names`/`narratives` loaders), serves static
  files, and delegates all writes to `ReviewStore` / `ManualStore`. Holds the
  joined cell table in memory (lazy, first `/api/cells`) for index→barcode
  resolution.
- **SPA** — presentation only; talks to the app over JSON. The
  dashboard-with-inline-controls layout plus a Plotly UMAP panel.

The server is read-only against the run tree and write-only against the
decision/selection files. It never imports `anndata` and never reads the
`.h5ad`.

## Data files

All paths are relative to the run output root (the `serve` argument).

### Read (existing, produced by the pipeline)

- `core_names.tsv` → `{parent_cluster: cell_type}`
- `narratives.tsv` → `{parent_cluster: narrative}`
- `panel.tsv` (root and `clusters/c{cid}/panel.tsv`) — the diagnosed minors and
  all their fields (`subcluster, parent_cluster, n_cells, frac_of_parent,
  top5_up_genes, top5_down_genes, likely_cause, diagnosis_confidence,
  diagnosis_rationale, recommended_disposition, proposed_cell_type,
  disposition_reason, …`).
- `clusters/c{cid}/subcluster_labels.tsv` — every subcluster of the parent and
  its per-cell membership (used to list **all** subclusters incl. below-threshold
  ones and the core, with sizes).
- `clusters/c{cid}/minor_profile.png`, `umap_subcluster.png` — static images.
- `clusters/c{cid}/deg_c{cid}_{k}.tsv`, `qc_drift_c{cid}_{k}.tsv` — per-minor
  DEG and QC-drift tables (the "minor vs major" evidence).
- `cell_labels.tsv` — per-cell `umap_cluster` (= subcluster, e.g. `c14_2`),
  `recommended_disposition`, `proposed_cell_type`; index = barcode.

### Read (new, produced by `export-coords`)

- `cell_coords.tsv.gz` — `barcode, umap_x, umap_y[, <qc cols…>]`, one row per
  cell. Gzipped TSV (no new dependency; ~3–5 MB for ~80k cells). Optional: if
  absent, the server runs with the interactive UMAP disabled.

### Write (new, produced by the server)

- `human_review.tsv` — one row per reviewed minor:
  `subcluster · parent_cluster · llm_disposition · human_disposition · note ·
  reviewer · updated_at`. `human_disposition ∈ {KEEP, DISCARD, UNCERTAIN}` (a
  cleared decision removes the row). Columns chosen so a future `apply-discard`
  could read `human_disposition` directly.
- `manual_cells.tsv` — hand-picked cells:
  `barcode · disposition · label · reviewer · updated_at`.
- `selections/selection_<label>.tsv` — exported barcode lists (one `barcode`
  column; `<label>` slugified).

All writes are atomic: write to a temp file in the same directory, then
`os.replace`. A dropped tunnel/ngrok session never corrupts a file, and
restarting `serve` resumes from the on-disk state.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the SPA (`index.html`) |
| GET | `/static/*` | static assets (`app.js`, `style.css`); Plotly loads from CDN |
| GET | `/api/run` | run summary: clusters (id, core name, #minors, #decided), totals, `has_coords` |
| GET | `/api/cluster/{cid}` | one cluster: core name, narrative, all subclusters (minors reviewable; core + below-threshold read-only), each merged with any existing human decision; image + table descriptors |
| GET | `/api/image/{cid}/{name}` | stream a PNG (`minor_profile` / `umap_subcluster`) via `FileResponse` |
| GET | `/api/table/{cid}/{name}` | a DEG/QC TSV as JSON (lazy, on expand) |
| GET | `/api/cells` | interactive-UMAP data (only if `has_coords`) |
| POST | `/api/decision` | record one minor's verdict |
| POST | `/api/selection/export` | write a barcode list |
| POST | `/api/selection/manual` | append a manual keep/discard set |

### `/api/cells` payload

Column-arrays, compact, HTTP-gzipped. **No barcodes** are sent to the browser;
cells are in a stable server-side order and selections refer to cells by their
integer index in that order.

```json
{
  "n": 81597,
  "x": [...float...],
  "y": [...float...],
  "parent_cluster": [...int code...],
  "parent_categories": ["0","3","6", ...],
  "subcluster": [...int code...],
  "subcluster_categories": ["c0_0","c0_1", ...],
  "disposition": [...int code...],
  "disposition_categories": ["", "KEEP", "DISCARD", "UNCERTAIN"],
  "qc": { "doublet_score": [...], "pct_counts_mt": [...] }   // whatever was exported
}
```

Built by inner-joining `cell_coords.tsv.gz` (coords/QC) with `cell_labels.tsv`
(subcluster via `umap_cluster`, disposition) on barcode. The server keeps the
joined `barcode` array in memory so selection POSTs (which send `indices`) can
be resolved to barcodes server-side.

### POST bodies

- `/api/decision` — `{subcluster, disposition, note}`. Validates
  `disposition ∈ {"KEEP","DISCARD","UNCERTAIN",""}` (`""` clears) and that
  `subcluster` is a known reviewable minor. Returns updated `{progress}`. 400 on
  bad input.
- `/api/selection/export` — `{label, indices:[int,...]}`. Resolves indices →
  barcodes, writes `selections/selection_<label>.tsv`. Returns `{path, n}`.
- `/api/selection/manual` — `{label, indices:[int,...], disposition}`. Resolves,
  appends to `manual_cells.tsv`. Returns `{n, total}`.

## UI / interaction

Dashboard-with-inline-controls (the approved mockup):

- **Left sidebar** — cluster list (id + core name), each with a decided/total
  badge; a global progress bar (`▓ decided/total`); a top-level toggle between
  **Dashboard** and **Interactive UMAP**.
- **Cluster view** — core name + narrative; the `minor_profile` and
  `umap_subcluster` PNGs; then a table of subclusters:
  - **Reviewable minors** (in `panel.tsv`): a row with size, frac, top up/down
    genes, `likely_cause`, confidence, the LLM `recommended_disposition`, and
    inline **[KEEP] [DISCARD] [UNCERTAIN]** buttons + a note field. An **"adopt
    LLM"** shortcut sets the human verdict to the LLM's recommendation. The
    LLM rationale and the DEG/QC tables expand on demand.
  - **Core (`_0`) and below-threshold fragments**: shown greyed/read-only with
    their size and a "not diagnosed (below min_subcluster_size)" note, so the
    full composition is visible but you cannot "decide" what the pipeline can't
    act on.
  - Clicking a minor row, when coords are present, jumps to the **Interactive
    UMAP** focused on that parent cluster with the minor highlighted vs the
    core (the "compare minor vs major" view) and the DEG/QC shown beside it.
- **Interactive UMAP view** — a Plotly `scattergl` plot of all cells.
  - *Global mode*: colored by parent cluster; hover shows subcluster.
  - *Cluster-focus mode*: zoomed to one parent, the chosen minor highlighted
    against the core.
  - **Lasso / box select** (Plotly built-in) opens a selection panel:
    - **Inspect** — live count, cluster/subcluster breakdown, disposition mix,
      and QC distributions, computed browser-side from the served arrays.
    - **Export barcodes** — name + save (`/api/selection/export`).
    - **Manual set** — name + KEEP/DISCARD, record (`/api/selection/manual`).

Decisions autosave on every click. Decided rows show a check + the saved
verdict; reloading the page (or restarting the server) restores all decisions
from `human_review.tsv`.

## Error handling

- Missing run files degrade gracefully, mirroring `report.py`'s `[missing]`
  behavior: a cluster without `panel.tsv` shows no minors (not a 500); a missing
  PNG → the `/api/image` route returns 404 and the UI shows a placeholder; a
  missing table → empty JSON.
- `cell_coords.tsv.gz` absent → `/api/run` reports `has_coords:false`, the
  UMAP toggle is hidden, `/api/cells` returns 404. The dashboard still works.
- Bad POST (unknown subcluster, invalid disposition, out-of-range index) → 400
  with a message; the SPA shows a toast and keeps the user's unsaved note.
- `fastapi`/`uvicorn` not installed → `serve` exits with a clear
  `pip install fastapi uvicorn` hint (run in `sh_dev`, not the login node).
  Importing `standissect` or running the pipeline never requires these.

## CLI

```
standissect serve <output_root>
    [--host 127.0.0.1] [--port 8050]
    [--decisions-file human_review.tsv] [--reviewer $USER]

standissect export-coords <h5ad> --output-dir <output_root>
    [--umap-key X_umap]
    [--doublet-score-col ...] [--mito-col ...]
    [--feature-count-col ...] [--umi-count-col ...]
    [--extra-qc-col ... (repeatable)]
```

`serve` prints the local URL on startup. `export-coords` reads the h5ad in
backed mode, extracts the embedding + present QC columns, and writes
`cell_coords.tsv.gz` into the run dir.

## Testing

`tests/test_webreview.py`, using FastAPI's `TestClient` (no network); the whole
module is guarded by `pytest.importorskip("fastapi")` so the suite still passes
where the optional deps are absent.

- **`ReviewStore`**: set/get round-trip; resume from an existing file; clearing
  a decision; atomic-write leaves no temp files; `progress()` math.
- **`ManualStore`**: `add_manual` append + `write_selection` round-trip; label
  slugification.
- **Synthetic run tree** (built in `tmp_path`): root + `clusters/c0` with
  `panel.tsv`, `subcluster_labels.tsv`, a 1×1 PNG, a `deg`/`qc` TSV,
  `core_names.tsv`, `narratives.tsv`, `cell_labels.tsv`, `cell_coords.tsv.gz`.
  - `GET /api/run` → expected clusters, counts, `has_coords:true`.
  - `GET /api/cluster/c0` → minors marked reviewable, core/below-threshold
    read-only, existing decision merged.
  - `GET /api/cells` → array lengths consistent; categories present.
  - `GET /api/image` / `/api/table` → 200 for present, 404 for missing.
  - `POST /api/decision` → persisted to file; invalid disposition → 400;
    unknown subcluster → 400.
  - `POST /api/selection/export` / `manual` → file written; indices resolved to
    the right barcodes; out-of-range index → 400.
  - `has_coords:false` path: omit `cell_coords.tsv.gz`, assert UMAP disabled and
    `/api/cells` → 404.

## Implementation milestones

- **M1 — dashboard + minor-vs-major review.** `review_store.py`, the FastAPI
  app's read endpoints + `/api/decision`, the SPA dashboard, the `serve` CLI,
  tests for the store and dashboard endpoints. Usable with only existing run
  output (no coords needed). Ships standalone value.
- **M2 — interactive UMAP + lasso.** `export_coords.py` + `export-coords` CLI,
  `/api/cells` and the selection endpoints, `ManualStore`, the Plotly UMAP +
  selection panel in the SPA, Plotly loaded from CDN, the coords/selection
  tests.
```
