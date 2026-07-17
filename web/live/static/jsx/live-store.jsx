/* eslint-disable */
// CiteClaw live-data layer.
// Replaces the static demo globals with a tiny external store fed by the
// backend (REST + WebSocket). Components read slices via `useLive(key)`.
// The design's components are untouched except for swapping their data
// source to this store; all layout/style stays as authored.

function _emptyMetrics() {
  return {
    accepted: 0, rejected: 0, rejectionReasons: [],
    llmTokensIn: 0, llmTokensOut: 0, llmReasoningTokens: 0,
    llmCalls: 0, llmCacheHits: 0, cost: 0, costBySource: [],
    s2Requests: 0, s2CacheHits: 0, s2CacheHitPct: 0, elapsedSec: 0,
  };
}

const LIVE = (function () {
  const listeners = new Set();
  let state = {
    // build mode
    seeds: (window.SEED_PAPERS || []).slice(),   // seed panel: demo template until searched
    searchQuery: "",                              // last seed-search query → run topic
    // run mode
    accepted: [],
    progress: { steps: [], done: 0, total: 0, current: null, overallPct: 0 },
    network: { nodes: [], edges: [], version: 0 },
    metrics: _emptyMetrics(),
    running: false,
    runStatus: "idle",       // idle | starting | running | done | error | stopped
    runId: null,
    error: null,
    logs: [],
    lastAddedId: null,
    settings: { model: "gemini-3.1-flash-lite", effort: "minimal", maxPapers: 200, keys: {}, models: [], loaded: false },
    // explore mode — "live" mirrors the current session; a run path swaps in
    // an on-disk collection loaded through /api/explore/run
    explore: { source: "live", papers: [], edges: [], runs: [], runsLoaded: false,
               runPath: null, meta: null, loading: false, error: null, version: 0,
               // author co-authorship view of the loaded run (fetched lazily)
               collab: { forPath: null, papers: [], edges: [], loading: false, error: null } },
  };
  return {
    getState: () => state,
    get: (k) => state[k],
    set(patch) { state = Object.assign({}, state, patch); listeners.forEach(l => l()); },
    subscribe(l) { listeners.add(l); return () => listeners.delete(l); },
    emptyMetrics: _emptyMetrics,
  };
})();

// Slice-level subscription. getSnapshot returns the *same* reference when a
// slice is unchanged, so unrelated updates don't re-render a component.
function useLive(key) {
  return React.useSyncExternalStore(
    React.useCallback((cb) => LIVE.subscribe(cb), []),
    () => LIVE.get(key)
  );
}

// ---- formatting helpers used across the run-mode panels ----
function fmtK(n) {
  n = Number(n) || 0;
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "K";
  return String(Math.round(n));
}
function fmtNum(n) { return (Number(n) || 0).toLocaleString(); }
function fmtDur(sec) {
  sec = Math.max(0, Math.round(Number(sec) || 0));
  const m = Math.floor(sec / 60), s = sec % 60;
  return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
}

// ---- backend calls ----
async function _api(path, opts) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch (_) {}
  if (!res.ok) throw new Error((body && (body.detail || body.message)) || res.statusText);
  return body;
}

async function refreshSettings() {
  try {
    const s = await _api("/api/settings");
    let models = [];
    try { models = await _api("/api/models"); } catch (_) {}
    LIVE.set({ settings: Object.assign({}, LIVE.get("settings"),
      { model: s.model, effort: s.reasoning_effort, maxPapers: s.max_papers, keys: s.keys, models, loaded: true }) });
  } catch (_) { /* backend not up yet */ }
}

async function saveSettings(patch) {
  const s = await _api("/api/settings", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch),
  });
  LIVE.set({ settings: Object.assign({}, LIVE.get("settings"),
    { model: s.model, effort: s.reasoning_effort, maxPapers: s.max_papers, keys: s.keys }) });
  return s;
}

async function searchSeeds(q, filters) {
  const f = filters || {};
  const params = new URLSearchParams({ q: q || "", limit: "25" });
  if (f.year && f.year !== "Any") params.set("year", f.year);
  if (f.minCites) params.set("min_cites", String(f.minCites));
  return _api("/api/seeds/search?" + params.toString());
}

// Lazy abstract fallback (OpenAlex) for a seed paper whose S2 abstract is empty.
async function fetchSeedAbstract(paper) {
  return _api("/api/seeds/abstract", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_id: paper.id, externalIds: paper.externalIds || {},
      title: paper.title, year: paper.year,
    }),
  });
}

function _pushLog(tag, msg) {
  const now = new Date();
  const t = now.toTimeString().slice(0, 8);
  LIVE.set({ logs: [{ t, tag, msg }].concat(LIVE.get("logs")).slice(0, 40) });
}

let _netThrottle = 0;
function _handleEvent(m) {
  if (m.type === "hello") {
    LIVE.set({ progress: m.progress });
  } else if (m.type === "progress") {
    LIVE.set({ progress: m.progress });
    if (m.progress.current) _pushLog("STEP", m.progress.current);
  } else if (m.type === "paper_added") {
    const acc = [m.paper].concat(LIVE.get("accepted")).slice(0, 1500);
    window.ACCEPTED_PAPERS = acc;
    LIVE.set({ accepted: acc, lastAddedId: m.paper.id });
  } else if (m.type === "metrics") {
    LIVE.set({ metrics: m.metrics });
  } else if (m.type === "graph") {
    window.NETWORK = { nodes: m.graph.nodes, edges: m.graph.edges };
    window.ACCEPTED_PAPERS = LIVE.get("accepted");
    const now = Date.now();
    if (now - _netThrottle > 2500) {   // throttle expensive canvas remounts
      _netThrottle = now;
      const cur = LIVE.get("network");
      LIVE.set({ network: { nodes: m.graph.nodes, edges: m.graph.edges, version: (cur.version || 0) + 1 } });
    }
  } else if (m.type === "error") {
    LIVE.set({ running: false, runStatus: "error", error: m.message });
    _pushLog("ERR", m.message);
  } else if (m.type === "done") {
    const cur = LIVE.get("network");
    window.NETWORK = { nodes: cur.nodes, edges: cur.edges };
    LIVE.set({ running: false, runStatus: m.status,
      network: { nodes: cur.nodes, edges: cur.edges, version: (cur.version || 0) + 1 } });
    _pushLog("DONE", "run " + m.status);
  }
}

function _openStream(runId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(proto + "://" + location.host + "/api/run/" + runId + "/stream");
  ws.onmessage = (ev) => { try { _handleEvent(JSON.parse(ev.data)); } catch (_) {} };
  ws.onerror = () => {};
  ws.onclose = () => {};
  return ws;
}

async function startRun(pipeline, seeds) {
  const st = LIVE.get("settings");
  const starred = (seeds || []).filter(s => s.starred);
  const body = {
    pipeline: pipeline,
    seeds: starred.map(s => ({ paper_id: s.id, title: s.title })),
    model: st.model, reasoning_effort: st.effort,
    limits: { max_papers: st.maxPapers || 200 },
    topic: (LIVE.get("searchQuery") || "").trim(),
  };
  const netVer = (LIVE.get("network").version || 0) + 1;
  LIVE.set({
    running: true, runStatus: "starting", error: null, accepted: [], lastAddedId: null, logs: [],
    progress: { steps: [], done: 0, total: 0, current: null, overallPct: 0 },
    network: { nodes: [], edges: [], version: netVer },
    metrics: LIVE.emptyMetrics(),
  });
  window.NETWORK = { nodes: [], edges: [] };
  window.ACCEPTED_PAPERS = [];
  let data;
  try {
    data = await _api("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
  } catch (e) {
    LIVE.set({ running: false, runStatus: "error", error: e.message });
    _pushLog("ERR", e.message);
    return { ok: false, error: e.message };
  }
  LIVE.set({
    runId: data.run_id, runStatus: "running",
    progress: { steps: data.steps, done: 0, total: data.steps.length, current: null, overallPct: 0 },
  });
  _openStream(data.run_id);
  return { ok: true };
}

async function stopRun() {
  const id = LIVE.get("runId");
  if (id) { try { await _api("/api/run/" + id + "/stop", { method: "POST" }); } catch (_) {} }
  LIVE.set({ running: false });
}

// ---- exploration mode ----
function _patchExplore(patch) {
  LIVE.set({ explore: Object.assign({}, LIVE.get("explore"), patch) });
}

async function refreshExploreRuns() {
  try {
    const runs = await _api("/api/explore/runs");
    _patchExplore({ runs, runsLoaded: true });
    return runs;
  } catch (_) {
    _patchExplore({ runs: [], runsLoaded: true });
    return [];
  }
}

const _EMPTY_COLLAB = { forPath: null, papers: [], edges: [], loading: false, error: null };

async function loadExploreRun(path) {
  _patchExplore({ loading: true, error: null });
  try {
    const d = await _api("/api/explore/run?path=" + encodeURIComponent(path));
    const ex = LIVE.get("explore");
    _patchExplore({
      source: "run", runPath: path, meta: d.meta || null,
      papers: d.papers || [], edges: d.edges || [],
      loading: false, version: (ex.version || 0) + 1,
      collab: { ..._EMPTY_COLLAB },
    });
    return { ok: true };
  } catch (e) {
    _patchExplore({ loading: false, error: e.message });
    return { ok: false, error: e.message };
  }
}

// Lazy-fetch the author co-authorship network for the loaded run (Finalize's
// collaboration_network.graphml, or derived from the collection JSON).
async function loadExploreCollab(path) {
  const cur = LIVE.get("explore").collab || _EMPTY_COLLAB;
  if (cur.forPath === path && !cur.error && (cur.papers.length || cur.loading)) return;
  _patchExplore({ collab: { forPath: path, papers: [], edges: [], loading: true, error: null } });
  try {
    const d = await _api("/api/explore/collab?path=" + encodeURIComponent(path));
    _patchExplore({ collab: { forPath: path, papers: d.papers || [], edges: d.edges || [],
                              loading: false, error: null } });
  } catch (e) {
    _patchExplore({ collab: { forPath: path, papers: [], edges: [],
                              loading: false, error: e.message } });
  }
}

function exploreUseLiveSession() {
  const ex = LIVE.get("explore");
  _patchExplore({ source: "live", runPath: null, meta: null, error: null,
                  version: (ex.version || 0) + 1 });
}

// Derive explore-shaped {papers, edges} from the live session's stores.
// Network nodes carry seed flags + cites for papers the accepted stream may
// not have (top-cited cap ordering); join the two by paper id.
function exploreFromLive() {
  const accepted = LIVE.get("accepted") || [];
  const net = LIVE.get("network") || { nodes: [], edges: [] };
  const byId = {};
  for (const p of accepted) byId[p.id] = p;
  const papers = [];
  const have = new Set();
  for (const n of net.nodes || []) {
    const p = byId[n.paperId];
    papers.push({
      id: n.paperId,
      title: (p && p.title) || n.paperId,
      authors: (p && p.authors) || "",
      year: (p && p.year) || n.year || 0,
      venue: (p && p.venue) || "",
      cites: (p && p.cites) != null ? p.cites : (n.cites || 0),
      seed: !!n.seed,
      depth: p ? p.depth : 0,
      source: (p && p.source) || (n.seed ? "seed" : ""),
      score: (p && p.score) || 0,
      abstract: "",
      addedAt: p ? p.addedAt : 0,
    });
    have.add(n.paperId);
  }
  for (const p of accepted) {
    if (have.has(p.id)) continue;
    papers.push({ ...p, cites: p.cites || 0, seed: p.source === "seed", abstract: "" });
  }
  const edges = [];
  for (const e of net.edges || []) {
    const a = net.nodes[e.a], b = net.nodes[e.b];
    if (a && b) edges.push({ source: a.paperId, target: b.paperId });
  }
  return { papers, edges };
}

Object.assign(window, {
  LIVE, useLive, fmtK, fmtNum, fmtDur,
  refreshSettings, saveSettings, searchSeeds, fetchSeedAbstract, startRun, stopRun,
  refreshExploreRuns, loadExploreRun, loadExploreCollab, exploreUseLiveSession,
  exploreFromLive,
});
