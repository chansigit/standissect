"use strict";
// standissect review — dashboard with inline interactive UMAPs + lasso review.

const STATE = {run: null, cid: null, cells: null, selIndices: [], heat: null,
  _heatHL: null, degA: null, degB: null, degArm: null, feature: null, degMode: false};
const PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
  "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1f77b4", "#d62728",
  "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];

// ---------------------------------------------------------------- tiny helpers
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const k in (attrs || {})) {
    if (k === "class") e.className = attrs[k];
    else if (k === "html") e.innerHTML = attrs[k];
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), attrs[k]);
    else if (attrs[k] === true) e.setAttribute(k, "");
    else if (attrs[k] !== false && attrs[k] != null) e.setAttribute(k, attrs[k]);
  }
  for (const c of kids) if (c != null) e.append(c.nodeType ? c : String(c));
  return e;
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = String(r.status);
    try { msg = (await r.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
}
let _toastT;
function toast(msg, isErr) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.className = "show" + (isErr ? " err" : "");
  clearTimeout(_toastT);
  _toastT = setTimeout(() => { t.className = ""; }, 2800);
}
function causeClass(c) {
  c = (c || "").toLowerCase();
  if (c.includes("biology")) return "biology";
  if (c.includes("doublet")) return "doublet";
  if (c.includes("shallow")) return "shallow";
  if (c.includes("sample") || c.includes("batch") || c.includes("donor")) return "sample";
  return "";
}

// ---------------------------------------------------------------- init
async function init() {
  STATE.run = await api("/api/run");
  document.getElementById("runroot").textContent = STATE.run.root;
  renderSidebar();
  setProgress();
  if (STATE.run.clusters.length) loadCluster(STATE.run.clusters[0].cid);
}

function setProgress() {
  const t = STATE.run.totals;
  document.getElementById("progtext").textContent = `${t.decided} / ${t.minors} decided`;
  document.getElementById("progfill").style.width =
    (t.minors ? (100 * t.decided / t.minors) : 0) + "%";
}

function renderSidebar() {
  const sb = document.getElementById("sidebar");
  sb.innerHTML = "";
  sb.append(el("div", {class: "side-title"}, "Clusters"));
  for (const c of STATE.run.clusters) {
    const done = c.n_minors > 0 && c.n_decided === c.n_minors;
    sb.append(el("a", {
      class: "side-item" + (done ? " done" : "") + (c.cid === STATE.cid ? " active" : ""),
      onclick: () => loadCluster(c.cid)},
      el("span", {class: "dot"}),
      el("span", {class: "nm"}, `c${c.cid}${c.core_name ? " · " + c.core_name : ""}`),
      el("span", {class: "badge"}, `${c.n_decided}/${c.n_minors}`)));
  }
}

// ---------------------------------------------------------------- cluster panel
async function loadCluster(cid) {
  STATE.cid = cid;
  renderSidebar();
  const box = document.getElementById("clusterbox");
  box.innerHTML = "<p class='muted'>loading…</p>";
  let d;
  try { d = await api(`/api/cluster/${cid}`); }
  catch (e) { box.innerHTML = `<p class='muted'>error: ${e.message}</p>`; return; }
  if (STATE.cid === cid) renderCluster(d);
}

function renderCluster(d) {
  const box = document.getElementById("clusterbox");
  box.innerHTML = "";

  const head = el("div", {class: "panel-head"},
    el("div", {class: "crumb"}, `cluster ${d.cid}`),
    el("h2", {}, d.core_name || `cluster ${d.cid}`));
  if (d.narrative) head.append(el("p", {class: "narrative"}, d.narrative));
  box.append(head);

  // interactive minor-profile heatmap (left) + interactive UMAP (right)
  const viz = el("div", {class: "viz"});
  viz.append(el("figure", {},
    el("figcaption", {}, "Minor-profile heatmap"),
    el("div", {id: "heat", class: "heat loading"}, "loading heatmap…")));
  if (STATE.run.has_coords) {
    const hlSel = el("select", {id: "hlSel",
      onchange: () => drawUmap(d.cid, hlSel.value)});
    const featBox = STATE.run.deg_enabled
      ? el("input", {id: "featbox", class: "featbox", placeholder: "colour by gene / obs col…",
          title: "type a gene name or an .obs column, then Enter",
          onkeydown: e => { if (e.key === "Enter") showFeature(e.target.value); }})
      : null;
    viz.append(el("figure", {},
      el("figcaption", {},
        el("span", {id: "umapcap"}, `UMAP — cluster ${d.cid}`),
        el("span", {class: "hl-sel"}, "view ", hlSel, featBox,
          STATE.run.deg_enabled
            ? el("button", {class: "deg-open", onclick: openDeg,
                title: "differential expression between two lassoed groups (all-clusters view)"}, "Lasso DEG")
            : null)),
      el("div", {class: "hint umaphint"},
        "left-drag = pan · hold Shift + drag = lasso · click = select group · scroll = zoom · double-click = reset"),
      el("div", {id: "umap", class: "plot loading"}, "loading cells…"),
      el("div", {id: "umaplegend", class: "umap-legend", hidden: true})));
    box.append(viz);
    drawHeatmap(d.cid);                 // after viz is in the DOM (getElementById works)
    drawClusterUmaps(d.cid);
  } else {
    if (d.images.umap_subcluster) {
      const img = el("img", {src: `/api/image/${d.cid}/umap_subcluster`, loading: "lazy"});
      img.addEventListener("error", () => { img.closest("figure").style.display = "none"; });
      viz.append(el("figure", {}, el("figcaption", {}, "UMAP zoom"), img));
    }
    box.append(viz);
    drawHeatmap(d.cid);                 // after viz is in the DOM (getElementById works)
  }

  // minors
  box.append(el("div", {class: "section-title"}, "Minor subclusters",
    el("span", {class: "count"}, String(d.minors.length))));
  if (!d.minors.length)
    box.append(el("p", {class: "muted"}, "No diagnosed minor subclusters in this cluster."));
  for (const m of d.minors) box.append(renderMinor(m, d.cid));

  // read-only fragments
  if (d.others.length) {
    box.append(el("div", {class: "section-title"}, "Core + below-threshold fragments",
      el("span", {class: "count"}, "read-only")));
    const wrap = el("div", {class: "others"});
    for (const o of d.others)
      wrap.append(el("span", {class: "chip" + (o.kind === "core" ? " core" : "")},
        `${o.subcluster} · ${o.n_cells}${o.kind === "core" ? " · core" : " · <min size"}`));
    box.append(wrap);
  }
}

function renderMinor(m, cid) {
  const card = el("div", {class: "minor"});
  const applyState = () => {
    card.classList.remove("v-keep", "v-discard", "v-uncertain");
    if (m.human_disposition) card.classList.add("v-" + m.human_disposition.toLowerCase());
  };

  const conf = m.diagnosis_confidence != null ? ` · conf ${m.diagnosis_confidence}` : "";
  const frac = m.frac_of_parent != null ? ` · ${(m.frac_of_parent * 100).toFixed(1)}%` : "";

  const verdict = el("div", {class: "verdict"});
  const mk = (label, val, cls) => el("button", {
    class: cls + (m.human_disposition === val ? " on" : ""),
    onclick: () => decide(val)}, label);
  const bK = mk("Keep", "KEEP", "keep"), bD = mk("Discard", "DISCARD", "discard"),
        bU = mk("Uncertain", "UNCERTAIN", "uncertain");
  verdict.append(bK, bD, bU);

  const note = el("input", {class: "note", placeholder: "note…", value: m.note || ""});
  note.addEventListener("blur", () => { if (m.human_disposition) decide(m.human_disposition, true); });

  async function decide(val, noteOnly) {
    const newVal = (!noteOnly && m.human_disposition === val) ? "" : val;
    try {
      const r = await api("/api/decision", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({subcluster: m.subcluster, disposition: newVal, note: note.value})});
      m.human_disposition = newVal; m.note = note.value;
      for (const [b, v] of [[bK, "KEEP"], [bD, "DISCARD"], [bU, "UNCERTAIN"]])
        b.classList.toggle("on", newVal === v);
      applyState();
      STATE.run.totals = r.progress;
      const cl = STATE.run.clusters.find(c => c.cid === cid);
      if (cl) cl.n_decided = document.querySelectorAll("#clusterbox .verdict button.on").length;
      renderSidebar(); setProgress();
      if (!noteOnly) toast(newVal ? `${m.subcluster} → ${newVal}` : `${m.subcluster} cleared`);
    } catch (e) { toast("save failed: " + e.message, true); }
  }

  const top = el("div", {class: "minor-top"},
    el("span", {class: "sc"}, m.subcluster),
    m.likely_cause ? el("span", {class: "cause " + causeClass(m.likely_cause)}, m.likely_cause) : null,
    el("span", {class: "meta"}, `${m.n_cells != null ? m.n_cells + " cells" : ""}${frac}${conf}`),
    el("span", {class: "grow"}),
    el("span", {class: "llm", html: `LLM&nbsp;→&nbsp;<b>${m.recommended_disposition || "—"}</b>`}),
    verdict);

  const actions = el("div", {class: "minor-actions"});
  if (m.recommended_disposition)
    actions.append(el("a", {class: "link", onclick: () => decide(m.recommended_disposition)}, "adopt LLM"));
  if (STATE.run.has_coords)
    actions.append(el("a", {class: "link", onclick: () => showOnUmap(cid, m.subcluster)}, "show on UMAP"));
  const detail = el("div", {class: "detail"});
  const expander = el("a", {class: "link", onclick: () => {
    detail.classList.toggle("open");
    if (detail.classList.contains("open") && !detail.dataset.loaded) fillDetail();
  }}, "details");
  actions.append(expander, note);

  async function fillDetail() {
    detail.dataset.loaded = "1";
    if (m.diagnosis_rationale) detail.append(el("div", {class: "rationale"}, m.diagnosis_rationale));
    if (m.proposed_cell_type)
      detail.append(el("div", {class: "proposed"}, el("b", {}, "proposed: "), m.proposed_cell_type));
    for (const [tbl, cap] of [[m.deg_table, "DEG vs main"], [m.qc_table, "QC drift"]]) {
      if (!tbl) continue;
      const holder = el("div", {});
      detail.append(el("a", {class: "link", onclick: async () => {
        if (holder.dataset.open) { holder.innerHTML = ""; holder.dataset.open = ""; return; }
        holder.dataset.open = "1";
        try { holder.append(renderTable(await api(`/api/table/${cid}/${tbl}`))); }
        catch (e) { holder.append(el("p", {class: "muted"}, "table error")); }
      }}, `▸ ${cap}`), holder);
    }
  }

  card.append(top, actions, detail);
  applyState();
  return card;
}

function renderTable(t) {
  const tab = el("table", {class: "tbl"});
  tab.append(el("tr", {}, ...t.columns.map(c => el("th", {}, c))));
  for (const row of t.rows)
    tab.append(el("tr", {}, ...t.columns.map(c => {
      let v = row[c];
      if (typeof v === "number")
        v = (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2)
          : (Number.isInteger(v) ? v : v.toFixed(3));
      return el("td", {}, v == null ? "" : v);
    })));
  return el("div", {class: "tbl-wrap"}, tab);
}

// ---------------------------------------------------------------- UMAPs
async function ensureCells() {
  if (STATE.cells) return true;
  try { STATE.cells = await api("/api/cells"); return true; }
  catch (e) { return false; }
}

async function drawClusterUmaps(cid) {
  if (!(await ensureCells())) {
    const g = document.getElementById("umap");
    if (g) { g.classList.remove("loading"); g.textContent = "";
             g.append(el("p", {class: "muted"}, "cells unavailable")); }
    return;
  }
  if (STATE.cid !== cid) return;                 // user switched away while loading
  const g = document.getElementById("umap");
  if (g) { g.classList.remove("loading"); g.textContent = ""; }
  fillSelect(cid, "");
  drawUmap(cid, "");
}

function minorsOf(cid) {
  if (!cid || !STATE.cells) return [];
  const pre = `c${cid}_`;
  return STATE.cells.subcluster_categories
    .filter(s => s.startsWith(pre) && !s.endsWith("_0")).sort();
}

// the dropdown doubles as the view switch: all clusters (global) | all minors | a minor
function fillSelect(cid, sel) {
  const ms = document.getElementById("hlSel");
  if (!ms) return;
  ms.innerHTML = "";
  ms.append(el("option", {value: "__all__"}, "all clusters (global)"));
  ms.append(el("option", {value: ""}, `cluster ${cid} · all minors`));
  for (const s of minorsOf(cid)) ms.append(el("option", {value: s}, `cluster ${cid} · ${s}`));
  ms.value = sel != null ? sel : "";
}

function showOnUmap(cid, minor) {
  const ms = document.getElementById("hlSel");
  if (ms) ms.value = minor;
  drawUmap(cid, minor);
  const lo = document.getElementById("umap");
  if (lo) lo.scrollIntoView({behavior: "smooth", block: "center"});
}

// equal x/y span centred on the kept points → with scaleanchor below, a true 1:1 UMAP
function _squareRange(C, keep) {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < C.n; i++) {
    if (keep && !keep(i)) continue;
    const x = C.x[i], y = C.y[i];
    if (x == null || y == null) continue;
    if (x < xmin) xmin = x; if (x > xmax) xmax = x;
    if (y < ymin) ymin = y; if (y > ymax) ymax = y;
  }
  if (!(xmin <= xmax)) return [undefined, undefined];
  const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
  const half = (Math.max(xmax - xmin, ymax - ymin) / 2) * 1.06 || 1;
  return [[cx - half, cx + half], [cy - half, cy + half]];
}

// Two traces so only the focus is selectable and it draws on top:
//   trace 0 = background (not selectable; selection styling neutralised)
//   trace 1 = focus cells of interest (selectable, on top)
// customdata on each point is the global cell index.
function _react2(bg, fg, xr, yr, gene) {
  const mkTrace = (d, neutral) => {
    const marker = gene
      ? {size: d.s, color: d.c, colorscale: "Viridis", cmin: 0, cmax: gene.cmax,
         showscale: !neutral, line: {width: 0},
         colorbar: {len: 0.92, thickness: 8, x: 1.005, xanchor: "left", outlinewidth: 0,
                    tickfont: {size: 8}, title: {text: gene.name, side: "right", font: {size: 9}}}}
      : {size: d.s, color: d.c, opacity: 1, line: {width: 0}};
    return {
      type: "scattergl", mode: "markers", x: d.x, y: d.y, text: d.t, customdata: d.cd,
      hovertemplate: gene ? "%{text}<br>%{marker.color:.2f}<extra></extra>"
                          : "%{text}<extra></extra>",
      marker,
      selected: {marker: {opacity: 1}},
      unselected: {marker: {opacity: neutral ? 1 : 0.15}},
    };
  };
  const layout = {
    margin: {l: 30, r: gene ? 48 : 8, t: 6, b: 26}, dragmode: "pan", hovermode: "closest",
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: {range: xr, zeroline: false, showticklabels: false, constrain: "domain"},
    yaxis: {range: yr, zeroline: false, showticklabels: false,
            scaleanchor: "x", scaleratio: 1, constrain: "domain"},
    showlegend: false, font: {family: "ui-sans-serif, system-ui, sans-serif", size: 11},
  };
  const gd = document.getElementById("umap");
  if (!gd) return;
  Plotly.react(gd, [mkTrace(bg, true), mkTrace(fg, false)], layout, {
    responsive: true, displaylogo: false, scrollZoom: true, doubleClick: "reset",
    modeBarButtonsToRemove: ["autoScale2d"],
  });
  gd.removeAllListeners && gd.removeAllListeners("plotly_selected");
  gd.on("plotly_selected", onSelected);
  gd.on("plotly_deselect", closeSel);
  gd.removeAllListeners && gd.removeAllListeners("plotly_click");
  gd.on("plotly_click", onPointClick);
  _activeGd = gd;
}

// left-drag = pan (default); hold Shift to momentarily switch to lasso, release
// to return to pan. Plotly's drag only honours the LEFT button and dragmode
// changes are async, so a right-button gesture can't lasso — Shift flips the
// mode on keydown (well before the mouse moves), which is reliable.
let _activeGd = null;
window.addEventListener("keydown", e => {
  if (e.key === "Shift" && !e.repeat && _activeGd && _activeGd._fullLayout &&
      _activeGd._fullLayout.dragmode !== "lasso")
    Plotly.relayout(_activeGd, "dragmode", "lasso");
});
window.addEventListener("keyup", e => {
  if (e.key === "Shift" && _activeGd && _activeGd._fullLayout &&
      _activeGd._fullLayout.dragmode === "lasso")
    Plotly.relayout(_activeGd, "dragmode", "pan");
});

// hl: "__all__" = all clusters (coloured by cluster) | "" = this cluster, all minors
//     | "cX_k" = highlight that minor. Every mode zooms to the current cluster and
//     draws it on top; only the focus (the cluster, or the chosen minor) is selectable.
function drawUmap(cid, hl) {
  const C = STATE.cells; if (!C) return;
  const subCats = C.subcluster_categories, N = C.n;
  const parPref = `c${cid}_`, corePref = `c${cid}_0`;
  const global = (hl === "__all__");
  const minor = (!global && hl) ? hl : null;
  const inCluster = i => subCats[C.subcluster[i]].startsWith(parPref);
  const isFocus = minor ? (i => subCats[C.subcluster[i]] === minor) : inCluster;
  const feat = STATE.feature;                      // feature colouring when set
  const cont = feat && feat.kind === "cont" ? feat : null;   // gene / numeric obs
  const cat = feat && feat.kind === "cat" ? feat : null;     // categorical obs
  const degMode = STATE.degMode;                   // show the two DEG groups (A/B)
  const aSet = degMode && STATE.degA ? new Set(STATE.degA) : null;
  const bSet = degMode && STATE.degB ? new Set(STATE.degB) : null;

  function colorOf(i) {
    const s = subCats[C.subcluster[i]];
    if (global) return PALETTE[C.parent_cluster[i] % PALETTE.length];
    if (!s.startsWith(parPref)) return "#d7dbe3";
    if (minor) return s === minor ? "#dc2626" : (s === corePref ? "#2563eb" : "#a8b6e0");
    return s === corePref ? "#2563eb" : "#dc2626";
  }
  const bg = {x: [], y: [], c: [], s: [], cd: [], t: []};
  const fg = {x: [], y: [], c: [], s: [], cd: [], t: []};
  for (let i = 0; i < N; i++) {
    const inA = aSet && aSet.has(i), inB = bSet && bSet.has(i);
    const foc = (degMode && (inA || inB)) || isFocus(i);   // DEG groups draw on top
    const d = foc ? fg : bg;
    let color, size;
    if (cont) {
      color = cont.vals[i] == null ? 0 : cont.vals[i];
      size = foc ? (minor ? 9 : 6) : (inCluster(i) ? 6 : 3);
    } else if (cat) {
      color = cat.codes[i] == null ? "#d7dbe3" : PALETTE[cat.codes[i] % PALETTE.length];
      size = foc ? (minor ? 9 : 6) : (inCluster(i) ? 6 : 3);
    } else if (degMode) {
      color = inA ? "#111827" : inB ? "#e11d48"            // A = ink, B = crimson
        : PALETTE[C.parent_cluster[i] % PALETTE.length];   // clusters stay visible
      size = (inA || inB) ? 8 : 3;
    } else {
      color = colorOf(i);
      size = foc ? (minor ? 9 : 6) : (inCluster(i) ? 6 : 3);
    }
    d.x.push(C.x[i]); d.y.push(C.y[i]); d.c.push(color); d.s.push(size);
    d.cd.push(i); d.t.push(subCats[C.subcluster[i]]);
  }
  STATE._umapGlobal = global;                     // global view = lasso selects ALL clusters
  // global fits the whole UMAP (so you can lasso across clusters); cluster/minor
  // views zoom to the cluster of interest.
  const [xr, yr] = _squareRange(C, global ? null : inCluster);
  _react2(bg, fg, xr, yr, cont ? {name: cont.name, cmax: cont.vmax} : null);
  highlightHeatCols(heatFocusCols(cid, hl));      // sync the heatmap column highlight
  renderUmapLegend(cat);
  const cap = document.getElementById("umapcap");
  if (cap) {
    cap.textContent = "";
    if (feat)
      cap.append(`UMAP · ${feat.name} `,
        el("a", {class: "gene-clear", onclick: clearFeature, title: "back to cluster colours"}, "✕"));
    else
      cap.append(global
        ? `UMAP — cluster ${cid} in context (all clusters)`
        : `UMAP — cluster ${cid}` + (minor ? ` · ${minor}` : " · all minors"));
  }
}

// discrete legend for a categorical feature (swatch + label); cleared otherwise.
function renderUmapLegend(cat) {
  const lg = document.getElementById("umaplegend");
  if (!lg) return;
  lg.innerHTML = "";
  if (!cat) { lg.hidden = true; return; }
  lg.hidden = false;
  cat.categories.forEach((name, i) => {
    lg.append(el("span", {class: "lg-item"},
      el("span", {class: "lg-sw", style: `background:${PALETTE[i % PALETTE.length]}`}),
      String(name)));
  });
}

// ---------------------------------------------------------------- heatmap
function currentHl() {
  const s = document.getElementById("hlSel");
  return s ? s.value : "";
}

// which heatmap columns correspond to the current UMAP focus:
//   a specific minor "cX_k" → just that column; otherwise (all minors / global)
//   → this cluster's own columns (its home core + all its minors).
function heatFocusCols(cid, hl) {
  const H = STATE.heat;
  if (!H || !H.cols) return [];
  if (hl && hl !== "__all__" && hl !== "") return [hl];
  return [H.home_core, ...(H.minor_cols || [])].filter(c => H.cols.includes(c));
}

async function drawHeatmap(cid) {
  let g = document.getElementById("heat");
  if (!g) return;
  let H = null;
  try { H = await api(`/api/heatmap/${cid}`); } catch (e) { H = null; }
  if (STATE.cid !== cid) return;                  // switched away while loading
  g = document.getElementById("heat"); if (!g) return;
  g.classList.remove("loading"); g.textContent = "";
  if (!H) {                                       // fall back to the static PNG
    STATE.heat = null;
    const img = el("img", {src: `/api/image/${cid}/minor_profile`, loading: "lazy"});
    img.addEventListener("error", () => { const f = img.closest("figure"); if (f) f.style.display = "none"; });
    g.append(img);
    return;
  }
  STATE.heat = H; STATE._heatHL = null;
  renderHeatPlot(g, H);
}

// three stacked heatmaps (gene / QC / sample) sharing the subcluster x-axis,
// reproducing minor_profile.png's order + matplotlib colormaps, but live.
function renderHeatPlot(g, H) {
  const cs = H.colorscales, R = H.ranges, cols = H.cols;
  const present = [
    {k: "gene",   y: H.genes,       z: H.gene_z, scale: cs.gene,   zr: R.gene,   w: 0.17 * H.genes.length,       title: "expr z"},
    {k: "qc",     y: H.qc_rows,     z: H.qc_z,   scale: cs.qc,     zr: R.qc,     w: 0.42 * (H.qc_rows.length),   title: "QC z"},
    {k: "sample", y: H.sample_rows, z: H.sample, scale: cs.sample, zr: R.sample, w: 0.30 * (H.sample_rows.length), title: "samp"},
  ].filter(b => b.y && b.y.length && b.z && b.z.length);

  const gap = 0.02, totW = present.reduce((s, b) => s + b.w, 0) || 1;
  const avail = 1 - gap * (present.length - 1);
  let top = 1;
  for (const b of present) { const h = avail * b.w / totW; b.dom = [Math.max(0, top - h), top]; top = b.dom[0] - gap; }
  const lastAx = present.length === 1 ? "y" : "y" + present.length;
  const totalRows = present.reduce((s, b) => s + b.y.length, 0);

  // colour THIS cluster's own column labels: major (home core) red, minors blue;
  // other clusters' cores stay muted grey.
  const tcolor = c => c === H.home_core ? "#dc2626"
                    : (H.minor_cols.indexOf(c) >= 0 ? "#2563eb" : "#94a3b8");
  const tbold = c => (c === H.home_core || H.minor_cols.indexOf(c) >= 0) ? ";font-weight:700" : "";
  const ticktext = cols.map(c => `<span style="color:${tcolor(c)}${tbold(c)}">${c}</span>`);

  const data = [], layout = {
    margin: {l: 78, r: 60, t: 6, b: 92},
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#f1f2f5", showlegend: false,
    font: {family: "ui-sans-serif, system-ui, sans-serif", size: 10},
    height: Math.min(1000, Math.max(420, 8 * totalRows + 110)),
    xaxis: {anchor: lastAx, domain: [0, 1], side: "bottom", tickangle: -90,
            tickmode: "array", tickvals: cols.map((_, i) => i), ticktext,
            tickfont: {size: 8}, ticks: "", showgrid: false, automargin: true,
            range: [-0.5, cols.length - 0.5]},
  };
  present.forEach((b, i) => {
    const ax = i === 0 ? "y" : "y" + (i + 1), axn = i === 0 ? "yaxis" : "yaxis" + (i + 1);
    data.push({
      type: "heatmap", x: cols, y: b.y, z: b.z, xaxis: "x", yaxis: ax,
      colorscale: b.scale, zmin: b.zr[0], zmax: b.zr[1], hoverongaps: false,
      xgap: 0, ygap: 0, zsmooth: false,
      colorbar: {len: b.dom[1] - b.dom[0], y: (b.dom[0] + b.dom[1]) / 2, yanchor: "middle",
                 x: 1.004, xanchor: "left", thickness: 8, outlinewidth: 0,
                 tickfont: {size: 7}, title: {text: b.title, side: "right", font: {size: 8}}},
      hovertemplate: "%{x} · %{y}<br>%{z:.2f}<extra></extra>",
    });
    layout[axn] = {domain: b.dom, anchor: "x", autorange: "reversed",
                   tickfont: {size: b.k === "gene" ? 6 : 8}, ticks: "",
                   showgrid: false, zeroline: false, automargin: true};
  });
  Plotly.react(g, data, layout, {responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"]});
  highlightHeatCols(STATE._heatHL || heatFocusCols(H.cid, currentHl()));
}

// the heatmap shapes for a given glow level (0..1): a white core|minor
// separator plus an amber outline box per highlighted column.
function _heatShapes(H, want, glow) {
  const shapes = [];
  if (H.n_core > 0 && H.n_minor > 0)
    shapes.push({type: "line", xref: "x", yref: "paper",
                 x0: H.n_core - 0.5, x1: H.n_core - 0.5, y0: 0, y1: 1,
                 line: {color: "#ffffff", width: 3}});
  H.cols.forEach((c, i) => {
    if (want.has(c))
      shapes.push({type: "rect", xref: "x", yref: "paper",
                   x0: i - 0.5, x1: i + 0.5, y0: 0, y1: 1,
                   line: {color: `rgba(217,119,6,${(0.45 + 0.55 * glow).toFixed(3)})`,
                          width: 0.8 + 0.8 * glow},
                   fillcolor: `rgba(245,158,11,${(0.02 + 0.15 * glow).toFixed(3)})`});
  });
  return shapes;
}

let _heatPulse = null;
function stopHeatPulse() { if (_heatPulse) { clearInterval(_heatPulse); _heatPulse = null; } }

// outline the given subcluster columns with a pulsing ("闪光") amber box across
// all three blocks. No-op on the PNG fallback.
function highlightHeatCols(names) {
  const g = document.getElementById("heat"), H = STATE.heat;
  if (!g || !g.data || !H || !H.cols) { stopHeatPulse(); return; }
  STATE._heatHL = names;
  stopHeatPulse();
  const want = new Set(names || []);
  if (!want.size) { Plotly.relayout(g, {shapes: _heatShapes(H, want, 0)}); return; }
  let phase = 0;
  const timer = setInterval(() => {
    const gg = document.getElementById("heat");
    if (timer !== _heatPulse || !gg || !gg.data || STATE.heat !== H) { clearInterval(timer); return; }
    phase += 0.5;
    Plotly.relayout(gg, {shapes: _heatShapes(H, want, 0.5 + 0.5 * Math.sin(phase))});
  }, 130);
  _heatPulse = timer;
}

// ---------------------------------------------------------------- selection
function onSelected(ev) {
  if (!ev || !ev.points || !ev.points.length) { if (!STATE.degArm) closeSel(); return; }
  if (STATE.degArm) {
    // DEG capture: a SEPARATE flow from keep/discard — take every lassoed cell
    // (any cluster, no focus restriction) into the armed group.
    const idx = ev.points.map(p => p.customdata);
    if (idx.length < 2) { toast("lasso at least 2 cells", true); return; }
    const w = STATE.degArm;
    if (w === "a") STATE.degA = idx; else STATE.degB = idx;
    STATE.degArm = null;
    _syncDegbar();
    if (STATE.cid != null) drawUmap(STATE.cid, "__all__");   // colour A/B; dragmode -> pan
    const h = document.getElementById("degHint");
    if (h) h.textContent = (STATE.degA && STATE.degB)
      ? "both groups set — click Compute." : `group ${w.toUpperCase()} set · arm the other.`;
    toast(`group ${w.toUpperCase()} = ${idx.length} cells`);
    return;
  }
  // keep/discard: only the focus (curveNumber 1) is selectable, as before.
  const pts = ev.points.filter(p => p.curveNumber === 1);
  if (!pts.length) { closeSel(); return; }
  STATE.selIndices = pts.map(p => p.customdata);
  showSelStats();
}

// click a focus point → select the WHOLE focus group (the cluster/minor of
// interest), producing the same selection payload + panel as a lasso. Only the
// focus trace (curveNumber 1) is selectable; clicking a background cell is ignored.
function onPointClick(ev) {
  if (STATE.degArm) return;               // armed for a DEG lasso — ignore clicks
  if (!ev || !ev.points) return;
  const pt = ev.points.find(p => p.curveNumber === 1);
  if (!pt || !pt.data || !pt.data.customdata || !pt.data.customdata.length) return;
  const cd = pt.data.customdata;
  STATE.selIndices = cd.slice();                   // every focus cell (global indices)
  const gd = document.getElementById("umap");
  if (gd) Plotly.restyle(gd, {selectedpoints: [[...cd.keys()]]}, [1]);  // mark focus selected
  showSelStats();
  const C = STATE.cells;                            // light up the clicked subcluster's column
  if (C && pt.customdata != null) {
    const sub = C.subcluster_categories[C.subcluster[pt.customdata]];
    if (sub) highlightHeatCols([sub]);
  }
}

function showSelStats() {
  const C = STATE.cells, idx = STATE.selIndices;
  document.getElementById("selcount").textContent = `${idx.length} cells`;
  const bySub = {}, byDisp = {};
  for (const i of idx) {
    const s = C.subcluster_categories[C.subcluster[i]];
    const d = C.disposition_categories[C.disposition[i]] || "(none)";
    bySub[s] = (bySub[s] || 0) + 1;
    byDisp[d || "(none)"] = (byDisp[d || "(none)"] || 0) + 1;
  }
  const box = document.getElementById("selstats");
  box.innerHTML = "";
  box.append(el("div", {class: "k"}, "Top subclusters"));
  const t1 = el("table", {});
  for (const [s, n] of Object.entries(bySub).sort((a, b) => b[1] - a[1]).slice(0, 10))
    t1.append(el("tr", {}, el("td", {}, s), el("td", {}, n)));
  box.append(t1);
  box.append(el("div", {class: "k"}, "LLM disposition"));
  const t2 = el("table", {});
  for (const [d, n] of Object.entries(byDisp).sort((a, b) => b[1] - a[1]))
    t2.append(el("tr", {}, el("td", {}, d), el("td", {}, n)));
  box.append(t2);
  for (const key in (C.qc || {})) {
    const vals = idx.map(i => C.qc[key][i]).filter(v => v != null).sort((a, b) => a - b);
    if (!vals.length) continue;
    const med = vals[Math.floor(vals.length / 2)];
    box.append(el("div", {class: "k"},
      `${key}: ${vals[0].toFixed(2)} / ${med.toFixed(2)} / ${vals[vals.length - 1].toFixed(2)} (min·med·max)`));
  }
  document.getElementById("selpanel").hidden = false;
}

function closeSel() {
  document.getElementById("selpanel").hidden = true;
  STATE.selIndices = [];
}

async function exportSel() {
  if (!STATE.selIndices.length) return;
  const label = prompt("Name this selection (→ selections/selection_<name>.tsv):", "selection");
  if (label == null) return;
  try {
    const r = await api("/api/selection/export", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({label, indices: STATE.selIndices})});
    toast(`exported ${r.n} barcodes → ${r.path}`);
  } catch (e) { toast("export failed: " + e.message, true); }
}

async function manualSel(disp) {
  if (!STATE.selIndices.length) return;
  const label = prompt(`Name this manual ${disp} set:`, "manual");
  if (label == null) return;
  try {
    const r = await api("/api/selection/manual", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({label, indices: STATE.selIndices, disposition: disp})});
    toast(`recorded ${r.n} cells as ${disp} (total ${r.total})`);
  } catch (e) { toast("manual save failed: " + e.message, true); }
}

// ---------------------------------------------------------------- DEG (A vs B)
// A dedicated flow, fully separate from the keep/discard selection: open the DEG
// panel from the UMAP's "DEG" button, "Lasso A"/"Lasso B" arm the next lasso to
// fill that group (any cells, across clusters — no focus restriction), then
// Compute runs Mann-Whitney server-side.
function openDeg() {
  if (!(STATE.run && STATE.run.deg_enabled)) return;
  const p = document.getElementById("degpanel");
  p.hidden = false;
  if (!p.dataset.placed) {                          // first open: place + make draggable
    p.dataset.placed = "1";
    p.style.bottom = "auto";
    p.style.left = "20px";
    p.style.height = Math.min(window.innerHeight * 0.7, 520) + "px";   // explicit -> resizable
    p.style.top = Math.max(64, window.innerHeight - p.offsetHeight - 24) + "px";
    _makeDegDraggable(p);
  }
  STATE.degMode = true;                             // DEG works only in all-clusters view
  const ms = document.getElementById("hlSel");
  if (ms) { ms.value = "__all__"; ms.disabled = true; }
  if (STATE.cid != null) drawUmap(STATE.cid, "__all__");
  _syncDegbar();
}

// drag the header to move the panel (native CSS resize:both handles the size).
function _makeDegDraggable(p) {
  const head = p.querySelector(".sel-head");
  if (!head) return;
  head.style.cursor = "move";
  head.addEventListener("mousedown", e => {
    if (e.target.closest(".sel-x")) return;         // not when hitting the close ×
    e.preventDefault();
    const r = p.getBoundingClientRect();
    const ox = e.clientX - r.left, oy = e.clientY - r.top;
    const move = ev => {
      p.style.left = Math.max(0, Math.min(window.innerWidth - 60, ev.clientX - ox)) + "px";
      p.style.top = Math.max(0, Math.min(window.innerHeight - 30, ev.clientY - oy)) + "px";
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  });
}

function closeDeg() {
  document.getElementById("degpanel").hidden = true;
  STATE.degArm = null;
  STATE.degMode = false;
  const ms = document.getElementById("hlSel");
  if (ms) ms.disabled = false;
  _setArmedUI();
  if (STATE.cid != null) drawUmap(STATE.cid, currentHl());   // back to normal colouring
}

function armDeg(which) {
  STATE.degArm = which;
  if (which === "a") STATE.degA = null; else STATE.degB = null;   // re-picking this group
  _syncDegbar();
  if (STATE.cid != null) drawUmap(STATE.cid, "__all__");          // redraw (drops old colour)
  const gd = document.getElementById("umap");
  if (gd && gd.data) Plotly.relayout(gd, "dragmode", "lasso");    // auto lasso — just drag
  const h = document.getElementById("degHint");
  if (h) h.textContent = `drag on the UMAP to lasso group ${which.toUpperCase()} (no Shift needed).`;
}

function _setArmedUI() {
  for (const w of ["a", "b"]) {
    const btn = document.getElementById("degArm" + w.toUpperCase());
    if (btn) btn.classList.toggle("armed", STATE.degArm === w);
  }
}

function _syncDegbar() {
  const ca = document.getElementById("degAchip"), cb = document.getElementById("degBchip");
  if (ca) ca.textContent = STATE.degA ? `A · ${STATE.degA.length}` : "A —";
  if (cb) cb.textContent = STATE.degB ? `B · ${STATE.degB.length}` : "B —";
  const run = document.getElementById("degRun");
  if (run) run.disabled = !(STATE.degA && STATE.degB);
  _setArmedUI();
}

function clearDeg() {
  STATE.degA = null; STATE.degB = null; STATE.degArm = null;
  const r = document.getElementById("degResult"); if (r) r.innerHTML = "";
  const h = document.getElementById("degHint");
  if (h) h.textContent = "Lasso A, then Lasso B, then Compute.";
  _syncDegbar();
  if (STATE.cid != null) drawUmap(STATE.cid, "__all__");   // drop the A/B colouring
}

async function computeDeg() {
  if (!STATE.degA || !STATE.degB) return;
  const res = document.getElementById("degResult");
  res.innerHTML = "<p class='muted'>computing DEG"
    + "<span class='dot-anim'><span>.</span><span>.</span><span>.</span></span></p>";
  try {
    const d = await api("/api/deg", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({a: STATE.degA, b: STATE.degB, top_n: 25})});
    renderDeg(d);
  } catch (e) { res.innerHTML = `<p class='muted'>DEG error: ${e.message}</p>`; }
}

function _fmtP(p) {
  if (p == null) return "";
  return p < 1e-3 ? p.toExponential(1) : p.toFixed(3);
}

function renderDeg(d) {
  const res = document.getElementById("degResult");
  res.innerHTML = "";
  const note = [`A=${d.n_a}`, `B=${d.n_b}`,
    d.dropped_overlap ? `overlap −${d.dropped_overlap}` : null,
    d.dropped_unknown ? `unknown −${d.dropped_unknown}` : null,
    `layer ${d.layer}`].filter(Boolean).join(" · ");
  res.append(el("div", {class: "deg-note"}, note));
  const mk = (title, rows, cls) => {
    const t = el("table", {class: "tbl deg-tbl"});
    t.append(el("tr", {}, el("th", {}, title), el("th", {}, "log2FC"), el("th", {}, "padj")));
    for (const r of rows)
      t.append(el("tr", {},
        el("td", {class: "g", title: "colour the UMAP by this gene's expression",
          onclick: () => showFeature(r.gene)}, r.gene),
        el("td", {}, r.log2fc == null ? "" : r.log2fc.toFixed(2)),
        el("td", {}, _fmtP(r.padj))));
    return el("div", {class: "deg-col " + cls}, t);
  };
  res.append(el("div", {class: "deg-cols"},
    mk("↑ in A", d.up_in_a, "a"), mk("↑ in B", d.up_in_b, "b")));
}

// colour the UMAP by a gene OR an .obs column (numeric -> continuous Viridis,
// categorical -> discrete palette + legend). Used by the feature box and by
// clicking a gene in a DEG result.
async function showFeature(name) {
  name = (name || "").trim();
  if (!name) { clearFeature(); return; }
  try {
    const d = await api(`/api/feature/${encodeURIComponent(name)}`);
    STATE.feature = d.kind === "cont"
      ? {name: d.name, kind: "cont", vals: d.values, vmax: d.vmax || 1}
      : {name: d.name, kind: "cat", codes: d.codes, categories: d.categories};
    if (STATE.cid != null) drawUmap(STATE.cid, currentHl());
    toast(`UMAP coloured by ${d.name}` + (d.kind === "cat" ? ` (${d.categories.length} groups)` : ""));
  } catch (e) { toast("feature failed: " + e.message, true); }
}

function clearFeature() {
  const box = document.getElementById("featbox");
  if (box) box.value = "";
  if (!STATE.feature) return;
  STATE.feature = null;
  if (STATE.cid != null) drawUmap(STATE.cid, currentHl());
}

window.addEventListener("DOMContentLoaded", init);
