"use strict";
// standissect review — interactive dashboard + UMAP lasso review.

const STATE = {run: null, cid: null, cells: null, view: "dashboard",
               selIndices: []};
const PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
  "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1f77b4", "#d62728",
  "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];

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
    let msg = r.status + "";
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
  _toastT = setTimeout(() => { t.className = ""; }, 2600);
}

// ----------------------------------------------------------------- init
async function init() {
  STATE.run = await api("/api/run");
  document.getElementById("runroot").textContent = STATE.run.root;
  if (STATE.run.has_coords) document.getElementById("umapBtn").hidden = false;
  renderSidebar();
  setProgress();
  document.querySelectorAll("#viewtoggle button").forEach(b =>
    b.addEventListener("click", () => showView(b.dataset.view)));
  if (STATE.run.clusters.length) loadCluster(STATE.run.clusters[0].cid);
}

function setProgress() {
  const t = STATE.run.totals;
  document.getElementById("progress").textContent =
    `decided ${t.decided}/${t.minors}`;
}

function renderSidebar() {
  const sb = document.getElementById("sidebar");
  sb.innerHTML = "";
  for (const c of STATE.run.clusters) {
    const done = c.n_minors > 0 && c.n_decided === c.n_minors;
    const a = el("a", {class: (done ? "done " : "") + (c.cid === STATE.cid ? "active" : ""),
                       onclick: () => loadCluster(c.cid)},
      el("span", {}, `c${c.cid}${c.core_name ? " · " + c.core_name : ""}`),
      el("span", {class: "badge"}, `${c.n_decided}/${c.n_minors}`));
    a.dataset.cid = c.cid;
    sb.append(a);
  }
}

// ----------------------------------------------------------------- dashboard
async function loadCluster(cid) {
  STATE.cid = cid;
  renderSidebar();
  showView("dashboard");
  const main = document.getElementById("dashboard");
  main.innerHTML = "<p class='muted'>loading…</p>";
  let d;
  try { d = await api(`/api/cluster/${cid}`); }
  catch (e) { main.innerHTML = `<p class='muted'>error: ${e.message}</p>`; return; }
  renderCluster(d);
}

function renderCluster(d) {
  const main = document.getElementById("dashboard");
  main.innerHTML = "";
  main.append(el("h2", {}, `cluster ${d.cid}${d.core_name ? " — " + d.core_name : ""}`));
  if (d.narrative) main.append(el("p", {class: "narrative"}, d.narrative));

  const imgrow = el("div", {class: "imgrow"});
  for (const [key, cap] of [["minor_profile", "minor-profile heatmap"],
                            ["umap_subcluster", "UMAP zoom"]]) {
    if (d.images[key]) {
      const img = el("img", {src: `/api/image/${d.cid}/${key}`, loading: "lazy"});
      img.addEventListener("error", () => { img.style.display = "none"; });
      imgrow.append(el("figure", {}, el("figcaption", {}, cap), img));
    }
  }
  if (imgrow.children.length) main.append(imgrow);

  if (!d.minors.length) main.append(el("p", {class: "muted"}, "no diagnosed minor subclusters."));
  for (const m of d.minors) main.append(renderMinor(m, d.cid));

  if (d.others.length) {
    const box = el("div", {class: "othersbox"},
      el("div", {class: "muted"}, "core + below-threshold fragments (not individually reviewable):"));
    const wrap = el("div", {class: "others"});
    for (const o of d.others)
      wrap.append(el("span", {class: "ro " + (o.kind === "core" ? "core" : "")},
        `${o.subcluster} · ${o.n_cells} cells${o.kind === "core" ? " · core" : " · below min size"}`));
    box.append(wrap);
    main.append(box);
  }
}

function renderMinor(m, cid) {
  const card = el("div", {class: "minor"});
  const setDim = () => card.classList.toggle("dimmed",
    m.human_disposition === "DISCARD");

  const conf = m.diagnosis_confidence != null ? ` · conf ${m.diagnosis_confidence}` : "";
  const frac = m.frac_of_parent != null ? ` · ${(m.frac_of_parent * 100).toFixed(1)}%` : "";
  const llm = el("span", {class: "llm", html:
    `LLM: <b>${m.recommended_disposition || "—"}</b>${m.likely_cause ? " (" + m.likely_cause + ")" : ""}`});

  const btns = el("div", {class: "btns"});
  const mk = (label, val, cls) => {
    const b = el("button", {class: cls + (m.human_disposition === val ? " on" : ""),
      onclick: () => decide(val)}, label);
    return b;
  };
  const bK = mk("KEEP", "KEEP", "keep"), bD = mk("DISCARD", "DISCARD", "discard"),
        bU = mk("UNCERTAIN", "UNCERTAIN", "uncertain");
  btns.append(bK, bD, bU);

  const noteInput = el("input", {class: "note", placeholder: "note…", value: m.note || ""});
  noteInput.addEventListener("blur", () => {
    if (m.human_disposition) decide(m.human_disposition, true);
  });

  async function decide(val, noteOnly) {
    const newVal = (!noteOnly && m.human_disposition === val) ? "" : val; // toggle off
    try {
      const r = await api("/api/decision", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({subcluster: m.subcluster, disposition: newVal,
                              note: noteInput.value})});
      m.human_disposition = newVal;
      m.note = noteInput.value;
      for (const [b, v] of [[bK, "KEEP"], [bD, "DISCARD"], [bU, "UNCERTAIN"]])
        b.classList.toggle("on", newVal === v);
      setDim();
      STATE.run.totals = r.progress;
      const cl = STATE.run.clusters.find(c => c.cid === cid);
      if (cl) { // recount decided for this cluster from the DOM
        cl.n_decided = document.querySelectorAll("#dashboard .minor .btns button.on").length;
      }
      renderSidebar(); setProgress();
      if (!noteOnly) toast(newVal ? `${m.subcluster} → ${newVal}` : `${m.subcluster} cleared`);
    } catch (e) { toast("save failed: " + e.message, true); }
  }

  const adopt = m.recommended_disposition
    ? el("span", {class: "adopt", onclick: () => decide(m.recommended_disposition)}, "adopt LLM")
    : null;

  const row1 = el("div", {class: "row1"},
    el("span", {class: "sc"}, m.subcluster),
    el("span", {class: "meta"}, `${m.n_cells != null ? m.n_cells + " cells" : ""}${frac}${conf}`),
    m.likely_cause ? el("span", {class: "cause"}, m.likely_cause) : null,
    llm, adopt,
    el("span", {class: "spacerflex"}),
    btns);
  card.append(row1, noteInput);

  // expandable detail (rationale + DEG/QC + compare-on-umap)
  const detail = el("div", {class: "detail"});
  const expander = el("span", {class: "expand", onclick: () => {
    detail.classList.toggle("open");
    if (detail.classList.contains("open") && !detail.dataset.loaded) fillDetail();
  }}, "▸ details (rationale · DEG · QC)");
  card.append(expander, detail);

  async function fillDetail() {
    detail.dataset.loaded = "1";
    if (m.diagnosis_rationale)
      detail.append(el("div", {class: "rationale"}, m.diagnosis_rationale));
    if (m.proposed_cell_type)
      detail.append(el("div", {}, el("span", {class: "k muted"}, "proposed: "), m.proposed_cell_type));
    if (STATE.run.has_coords)
      detail.append(el("span", {class: "adopt", onclick: () =>
        focusOnUmap(cid, m.subcluster)}, "compare vs core on UMAP ▶"));
    for (const [tbl, cap] of [[m.deg_table, "DEG vs main"], [m.qc_table, "QC drift"]]) {
      if (!tbl) continue;
      const holder = el("div", {});
      detail.append(el("div", {class: "expand", onclick: async () => {
        if (holder.dataset.loaded) { holder.innerHTML = ""; holder.dataset.loaded = ""; return; }
        holder.dataset.loaded = "1";
        try {
          const t = await api(`/api/table/${cid}/${tbl}`);
          holder.append(renderTable(t));
        } catch (e) { holder.append(el("p", {class: "muted"}, "table error")); }
      }}, `▸ ${cap}`), holder);
    }
  }
  setDim();
  return card;
}

function renderTable(t) {
  const tab = el("table", {class: "tbl"});
  tab.append(el("tr", {}, ...t.columns.map(c => el("th", {}, c))));
  for (const row of t.rows)
    tab.append(el("tr", {}, ...t.columns.map(c => {
      let v = row[c];
      if (typeof v === "number") v = Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2)
        : (Number.isInteger(v) ? v : v.toFixed(3));
      return el("td", {}, v == null ? "" : v);
    })));
  return tab;
}

// ----------------------------------------------------------------- view switch
function showView(view) {
  STATE.view = view;
  document.querySelectorAll("#viewtoggle button").forEach(b =>
    b.classList.toggle("active", b.dataset.view === view));
  document.getElementById("dashboard").hidden = view !== "dashboard";
  document.getElementById("umapview").hidden = view !== "umap";
  if (view === "umap") ensureUmap();
}

// ----------------------------------------------------------------- umap
async function ensureUmap() {
  if (!STATE.run.has_coords) return;
  if (!STATE.cells) {
    document.getElementById("umap").innerHTML = "<p class='muted'>loading cells…</p>";
    try { STATE.cells = await api("/api/cells"); }
    catch (e) { document.getElementById("umap").innerHTML =
      `<p class='muted'>cells error: ${e.message}</p>`; return; }
    buildFocusSelectors();
  }
  drawUmap();
}

function buildFocusSelectors() {
  const fs = document.getElementById("focusSelect");
  fs.innerHTML = "";
  fs.append(el("option", {value: ""}, "— global —"));
  for (const c of STATE.run.clusters)
    fs.append(el("option", {value: c.cid}, `c${c.cid}${c.core_name ? " · " + c.core_name : ""}`));
  fs.addEventListener("change", () => { buildMinorSelect(fs.value); drawUmap(); });
  document.getElementById("minorSelect").addEventListener("change", drawUmap);
  buildMinorSelect("");
}

function minorsOf(cid) {
  if (!cid || !STATE.cells) return [];
  const pre = `c${cid}_`;
  return STATE.cells.subcluster_categories
    .filter(s => s.startsWith(pre) && !s.endsWith("_0")).sort();
}

function buildMinorSelect(cid) {
  const ms = document.getElementById("minorSelect");
  ms.innerHTML = "";
  ms.append(el("option", {value: ""}, "— all —"));
  for (const s of minorsOf(cid)) ms.append(el("option", {value: s}, s));
}

function focusOnUmap(cid, minor) {
  showView("umap");
  const setSel = () => {
    document.getElementById("focusSelect").value = cid;
    buildMinorSelect(cid);
    document.getElementById("minorSelect").value = minor || "";
    drawUmap();
  };
  if (STATE.cells) setSel(); else ensureUmap().then(setSel);
}

function drawUmap() {
  const C = STATE.cells; if (!C) return;
  const focusCid = document.getElementById("focusSelect").value;
  const focusMinor = document.getElementById("minorSelect").value;
  const subCats = C.subcluster_categories, parCats = C.parent_categories;
  const colors = new Array(C.n);
  let xr = null, yr = null;
  if (!focusCid) {
    for (let i = 0; i < C.n; i++) colors[i] = PALETTE[C.parent_cluster[i] % PALETTE.length];
  } else {
    const corePref = `c${focusCid}_0`, parPref = `c${focusCid}_`;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (let i = 0; i < C.n; i++) {
      const s = subCats[C.subcluster[i]];
      if (focusMinor && s === focusMinor) colors[i] = "#d62728";
      else if (s === corePref) colors[i] = "#1f77b4";
      else if (s.startsWith(parPref)) colors[i] = focusMinor ? "#9bb3d4" : "#d62728";
      else { colors[i] = "#e2e2e2"; continue; }
      if (C.x[i] < xmin) xmin = C.x[i]; if (C.x[i] > xmax) xmax = C.x[i];
      if (C.y[i] < ymin) ymin = C.y[i]; if (C.y[i] > ymax) ymax = C.y[i];
    }
    if (xmin < xmax) {
      const px = (xmax - xmin) * 0.08, py = (ymax - ymin) * 0.08;
      xr = [xmin - px, xmax + px]; yr = [ymin - py, ymax + py];
    }
  }
  const trace = {
    type: "scattergl", mode: "markers", x: C.x, y: C.y,
    text: C.subcluster.map(c => subCats[c]),
    hovertemplate: "%{text}<extra></extra>",
    marker: {size: focusCid ? 4 : 3, color: colors, opacity: 0.8},
  };
  const layout = {
    margin: {l: 30, r: 10, t: 10, b: 30}, dragmode: "lasso", hovermode: "closest",
    xaxis: {title: "UMAP1", range: xr, zeroline: false},
    yaxis: {title: "UMAP2", range: yr, zeroline: false},
    showlegend: false,
  };
  const gd = document.getElementById("umap");
  Plotly.react(gd, [trace], layout,
    {responsive: true, displaylogo: false, modeBarButtonsToRemove: ["autoScale2d"]});
  gd.removeAllListeners && gd.removeAllListeners("plotly_selected");
  gd.on("plotly_selected", onSelected);
  gd.on("plotly_deselect", closeSel);
}

function onSelected(ev) {
  if (!ev || !ev.points || !ev.points.length) { closeSel(); return; }
  STATE.selIndices = ev.points.map(p => p.pointNumber);
  showSelStats();
}

function showSelStats() {
  const C = STATE.cells, idx = STATE.selIndices;
  document.getElementById("selcount").textContent = `${idx.length} cells`;
  const bySub = {}, byDisp = {};
  for (const i of idx) {
    const s = C.subcluster_categories[C.subcluster[i]];
    const d = C.disposition_categories[C.disposition[i]] || "(none)";
    bySub[s] = (bySub[s] || 0) + 1;
    byDisp[d] = (byDisp[d] || 0) + 1;
  }
  const box = document.getElementById("selstats");
  box.innerHTML = "";
  const subSorted = Object.entries(bySub).sort((a, b) => b[1] - a[1]).slice(0, 12);
  box.append(el("div", {class: "k"}, "by subcluster:"));
  const t1 = el("table", {});
  for (const [s, n] of subSorted) t1.append(el("tr", {}, el("td", {}, s), el("td", {}, n)));
  box.append(t1);
  box.append(el("div", {class: "k"}, "by LLM disposition:"));
  const t2 = el("table", {});
  for (const [d, n] of Object.entries(byDisp).sort((a, b) => b[1] - a[1]))
    t2.append(el("tr", {}, el("td", {}, d), el("td", {}, n)));
  box.append(t2);
  for (const key in (C.qc || {})) {
    const vals = idx.map(i => C.qc[key][i]).filter(v => v != null).sort((a, b) => a - b);
    if (!vals.length) continue;
    const med = vals[Math.floor(vals.length / 2)];
    box.append(el("div", {class: "k"},
      `${key}: min ${vals[0].toFixed(3)} · med ${med.toFixed(3)} · max ${vals[vals.length - 1].toFixed(3)}`));
  }
  document.getElementById("selpanel").hidden = false;
}

function closeSel() {
  document.getElementById("selpanel").hidden = true;
  STATE.selIndices = [];
}

async function exportSel() {
  if (!STATE.selIndices.length) return;
  const label = prompt("Name this selection (file: selections/selection_<name>.tsv):", "selection");
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

window.addEventListener("DOMContentLoaded", init);
