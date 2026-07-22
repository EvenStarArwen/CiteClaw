/* eslint-disable */
// Shared citation-graph surface — used by BOTH the Run and Explore pages.
//
// Engine: the design's module_a_network_f5 reference stack (graphology graph
// + ForceAtlas2 layout + sigma WebGL renderer). Two execution modes of the
// SAME algorithm: a one-shot synchronous pre-warm gives a fresh dataset
// coarse structure before its first paint, then the web-worker runs a
// bounded fast-settle window (double step size) and drops to the configured
// gentle dynamics — full speed even in a background tab, UI never blocked.
//
// Look: the Run network's — camera-synced dot grid, ramp-coloured fills,
// cream node strokes, plum seed dot with soft halo, plum selection ring.
// Labels are opt-in (hidden by default); hover always shows the tooltip.
//
// Data shape (papers double as author rows when kind="author"):
//   papers: [{id,title,authors,year,venue,cites,seed,depth,source,addedAt,
//             hIndex?,nPapers?}]
//   edges:  [{source,target,weight?}]  (id pairs)
//
// Filters here are Gephi-style: filtered nodes/edges are REMOVED from the
// graphology graph (not reducer-hidden), so ForceAtlas2 re-flows the layout
// live; removed nodes keep a position cache so un-filtering restores them
// where they were.

function cgCssVar(name, fallback) {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  } catch (_) { return fallback; }
}

function cgColorRgb(color, fallback) {
  const s = String(color || "").trim();
  const h = s.replace("#", "");
  if (/^[0-9a-fA-F]{6}$/.test(h)) {
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  const m = s.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const parts = m[1].split(",").map(x => parseFloat(x));
    if (parts.length >= 3) return [parts[0], parts[1], parts[2]];
  }
  return fallback;
}

// The run canvas dimmed via globalAlpha; sigma has no per-item alpha, so we
// pre-blend toward the paper background (theme-aware) instead.
function cgFade(color, alpha, bgRgb) {
  const c = cgColorRgb(color, [107, 118, 129]);
  const out = c.map((v, i) => Math.round(v * alpha + bgRgb[i] * (1 - alpha)));
  return "#" + out.map(v => Math.max(0, Math.min(255, v)).toString(16).padStart(2, "0")).join("");
}

function cgMulberry32(seed) {
  return function () {
    let t = (seed += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------- palettes
// The default "Ember" ramp is the design's --cc-year-p-* tokens (theme-aware
// via CSS). The alternatives are hand-tuned low-saturation ramps that sit on
// the cream/charcoal backgrounds: dim → bright on dark, pale → deep on light
// (old → recent in both cases). 8 stops each, like the token ramp.
const CG_PALETTES = {
  ember: null,  // read from CSS tokens
  moss: {
    light: ["#b7bfa4", "#a4b18f", "#91a27b", "#7e9268", "#6b8156", "#587045", "#465e36", "#354c28"],
    dark:  ["#4b5442", "#57624c", "#647157", "#728062", "#80906e", "#90a07b", "#a0b089", "#b1c098"],
  },
  slate: {
    light: ["#b3bcc6", "#9fabb9", "#8b99ab", "#77889d", "#64778e", "#51657e", "#3f546d", "#2e435b"],
    dark:  ["#49525e", "#545f6e", "#606d7e", "#6d7b8e", "#7b8a9e", "#8a99ae", "#9aa9be", "#abb9cd"],
  },
  dusk: {
    light: ["#c2b3bd", "#b29fad", "#a28b9d", "#92788d", "#82657d", "#71536c", "#60425a", "#4e3248"],
    dark:  ["#564b56", "#635665", "#706274", "#7e6e83", "#8c7b92", "#9b89a1", "#aa97b0", "#b9a6bf"],
  },
  ash: {
    light: ["#bdbab1", "#aca99f", "#9b988e", "#8a877e", "#79766e", "#68655e", "#57554f", "#464440"],
    dark:  ["#565550", "#62615b", "#6f6e67", "#7c7b73", "#8a887f", "#98968c", "#a6a499", "#b5b2a7"],
  },
};
const CG_PALETTE_NAMES = [
  ["ember", "Ember (default)"], ["moss", "Moss"], ["slate", "Slate"],
  ["dusk", "Dusk"], ["ash", "Ash"],
];

function cgTheme() {
  const t = document.documentElement.getAttribute("data-theme");
  return t === "dark" ? "dark" : "light";
}

function cgPaletteStops(name, theme) {
  const p = CG_PALETTES[name];
  if (p) return p[theme === "dark" ? "dark" : "light"];
  const stops = [];
  for (let y = 2018; y <= 2025; y++) stops.push(cgCssVar(`--cc-year-p-${y}`, "#8a8a8a"));
  return stops;
}

function cgReadPalette(paletteName) {
  const bg = cgCssVar("--cc-bg", "#f5f3ee");
  const bgRgb = cgColorRgb(bg, [245, 243, 238]);
  return {
    bg, bgRgb,
    stops: cgPaletteStops(paletteName || "ember", cgTheme()),
    seed: cgCssVar("--cc-net-seed", "#3e3548"),
    seedHalo: cgCssVar("--cc-net-seed-halo", "rgba(62, 53, 72, 0.10)"),
    seedStroke: cgCssVar("--cc-net-seed-stroke", "#faf8f2"),
    nodeStroke: cgCssVar("--cc-net-node-stroke", "rgba(250, 248, 242, 0.7)"),
    selStrong: cgCssVar("--cc-net-sel-strong", "#3e3548"),
    selSoft: cgCssVar("--cc-net-sel-soft", "rgba(62, 53, 72, 0.55)"),
    gridDot: cgCssVar("--cc-grid-dot-soft", "rgba(68, 60, 40, 0.09)"),
    ink1: cgCssVar("--cc-ink-1", "#1e2735"),
    ink2: cgCssVar("--cc-ink-2", "#55606d"),
    inkFaint: cgCssVar("--cc-ink-faint", "#94a3b8"),
    // the demo's ink for node outlines + edges — theme-aware (dark on light,
    // light on dark) via --cc-ink-1; a --cc-net-ink token overrides if defined.
    netInk: cgCssVar("--cc-net-ink", cgCssVar("--cc-ink-1", "#26231f")),
  };
}

// The design ships 8 ramp stops. Data can span any era (a Gephi import may
// reach back decades), so the stops are mapped linearly over the dataset's
// actual range; the default domain reproduces the fixed 2018→2025 behaviour.
function cgYearDomain(papers) {
  let min = Infinity, max = -Infinity;
  for (const p of papers || []) {
    const y = Number(p.year) || 0;
    if (y > 1800) { if (y < min) min = y; if (y > max) max = y; }
  }
  if (!isFinite(min)) return { min: 2018, max: 2025 };
  if (min >= 2018 && max <= 2025) return { min: 2018, max: 2025 };
  return { min, max: Math.max(max, min + 1) };
}

// Author view: colour by the year the author first appeared in the
// collection (author_graph.py's `year_entered`, or derived). Data without
// years (older exports) falls back to h-index, then papers-in-collection.
function cgAuthorDomain(papers) {
  let yMin = Infinity, yMax = -Infinity, hMax = 0, pMax = 0;
  for (const p of papers || []) {
    const y = Number(p.year) || 0;
    if (y > 1800) { if (y < yMin) yMin = y; if (y > yMax) yMax = y; }
    if ((p.hIndex || 0) > hMax) hMax = p.hIndex || 0;
    if ((p.nPapers || 0) > pMax) pMax = p.nPapers || 0;
  }
  if (isFinite(yMin)) {
    return { min: yMin, max: Math.max(yMax, yMin + 1), field: "year",
             caption: `first paper ${yMin} → ${yMax}` };
  }
  if (hMax > 0) return { min: 0, max: hMax, field: "hIndex", caption: `h-index 0 → ${hMax}` };
  return { min: 0, max: Math.max(1, pMax), field: "nPapers", caption: `papers 0 → ${pMax}` };
}

function cgColorValue(p, kind, domain) {
  if (kind === "author" && domain.field !== "year") return Number(p[domain.field]) || 0;
  const y = Number(p.year) || 0;
  return y > 1800 ? y : null;
}

function cgRampColor(pal, domain, v) {
  if (v == null) return pal.stops[3] || "#8a8a8a";
  const t = Math.max(0, Math.min(1, (v - domain.min) / Math.max(1e-9, domain.max - domain.min)));
  return pal.stops[Math.round(t * 7)] || "#8a8a8a";
}

// ------------------------------------------------------------------ sizing
// Legacy curve = the Run canvas's radii (seeds fixed, satellites log-cites).
function cgLegacyRadius(p) {
  if (p.seed) return 8;
  const c = Math.max(0, Number(p.cites) || 0);
  return 3.5 + Math.min(4.5, Math.log10(1 + c) * 1.2);
}

// Gephi-style ranking: the user sets the EXACT min/max radius (so a 1 → 100
// range is a genuine 100× difference) and picks a transformation applied to
// the normalized value — the analogue of Gephi's spline presets. Concave
// curves (√, ∛, log) lift the long tail's small values; convex ones (², ³)
// reserve size for the giants. Min = max gives uniform nodes.
const CG_CURVES = {
  linear: t => t,
  sqrt: t => Math.sqrt(t),
  cbrt: t => Math.cbrt(t),
  log: t => Math.log10(1 + 99 * t) / 2,   // two decades of dynamic range
  pow2: t => t * t,
  pow3: t => t * t * t,
};
const CG_CURVE_NAMES = [
  ["linear", "Linear"], ["sqrt", "Square root"], ["cbrt", "Cube root"],
  ["log", "Log"], ["pow2", "Squared"], ["pow3", "Cubed"],
];

// Node size value: citations for papers, in-collection citations for authors
// (the collab payload's `cites` is intra_network_citation).
function cgParamRadius(p, kind, vis, vmax) {
  const v = Math.max(0, Number(p.cites) || 0);
  const lo = Math.max(0.5, Number(vis.minSize) || 0.5);
  const hi = Math.max(lo, Number(vis.maxSize) || lo);
  const t = vmax > 0 ? Math.max(0, Math.min(1, v / vmax)) : 0;
  const f = CG_CURVES[vis.sizeCurve] || CG_CURVES.sqrt;
  let r = lo + (hi - lo) * f(t);
  if (p.seed) r = Math.max(r, lo + 0.45 * (hi - lo));  // seeds never vanish
  return r;
}

function cgRadius(p, kind, vis, vmax, legacy) {
  return legacy ? cgLegacyRadius(p) : cgParamRadius(p, kind, vis, vmax);
}
// Border budget added around the fill (ring space + halo/stroke), so the
// visible fill keeps the run-canvas sizes.
function cgNodeSize(p, kind, vis, vmax, legacy) {
  return cgRadius(p, kind, vis, vmax, legacy) + (p.seed ? 5 : 2.5);
}

// Edge width ∝ edge weight over the VISIBLE edges' weight range, with its
// own min/max + transformation (mirrors the node ranking). Graphs without
// meaningful weights (all equal) render every edge at the minimum width.
function cgEdgeWidth(w, wrange, vis) {
  const lo = Math.max(0.1, Number(vis.edgeMin) || 0.1);
  const hi = Math.max(lo, Number(vis.edgeMax) || lo);
  if (!wrange || !(wrange.hi > wrange.lo)) return lo;
  const v = w == null ? 1 : Number(w) || 0;
  const t = Math.max(0, Math.min(1, (v - wrange.lo) / (wrange.hi - wrange.lo)));
  const f = CG_CURVES[vis.edgeCurve] || CG_CURVES.linear;
  return lo + (hi - lo) * f(t);
}

function cgTrunc(s, n) { s = String(s || ""); return s.length > n ? s.slice(0, n - 2) + "…" : s; }
function cgEaseOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

const CG_TRANSPARENT = "#00000000";

// FA2 defaults — Gephi's citation-network recipe rather than the f5 demo's
// small-graph aesthetics. edgeWeightInfluence defaults to 0 (pure topology):
// similarity-style GEXF weights are often degenerate (CitNet: 60% exactly 0)
// and turn the layout into a core-knot + repulsion shell.
const CG_FA2_DEFAULTS = {
  linLogMode: true,
  adjustSizes: false,
  outboundAttractionDistribution: false,
  strongGravityMode: false,
  barnesHutOptimize: true,
  scalingRatio: 10,
  gravity: 1,
  slowDown: 5,
  edgeWeightInfluence: 0,
};
const CG_VIS_DEFAULTS = {
  minSize: 3, maxSize: 20, sizeCurve: "sqrt",
  edgeMin: 0.25, edgeMax: 1.1, edgeCurve: "linear",
  palette: "ember",
};
const CG_GF_DEFAULTS = { minDegree: 0, minEdgeW: 0, largestOnly: false };

// Appearance + force-layout settings persist across BOTH the Run and Explore
// pages. Each page mounts its own CiteGraph, so without a shared store the
// graph-settings popover reset to defaults every time you switched pages.
// One module-level object is the single source of truth; localStorage makes
// the choice survive a reload too. Structural filters (gf: min degree / edge
// weight / largest component) are deliberately NOT shared — they are tuned to
// one dataset's ranges and reset on every dataset switch.
const CG_PREFS_KEY = "citeclaw.graphPrefs.v1";
function cgLoadPrefs() {
  const out = { fa2: { ...CG_FA2_DEFAULTS }, vis: { ...CG_VIS_DEFAULTS }, labels: null };
  try {
    const raw = JSON.parse(localStorage.getItem(CG_PREFS_KEY) || "{}");
    if (raw && typeof raw === "object") {
      if (raw.fa2 && typeof raw.fa2 === "object") Object.assign(out.fa2, raw.fa2);
      if (raw.vis && typeof raw.vis === "object") Object.assign(out.vis, raw.vis);
      if (typeof raw.labels === "boolean") out.labels = raw.labels;
    }
  } catch (_) {}
  return out;
}
const CG_PREFS = cgLoadPrefs();
function cgSavePrefs() {
  try { localStorage.setItem(CG_PREFS_KEY, JSON.stringify(CG_PREFS)); } catch (_) {}
}

const CG_PREWARM_ITERS = 300; // one-shot sync FA2 before first paint (~0.3s)
const CG_SETTLE_MS = 6500;    // fast-stepping worker window for a fresh dataset
const CG_EDIT_MS = 1800;      // fast re-flow after filter edits
const CG_POLISH_MS = 2500;    // gentle worker run after the fast phase

// ------------------------------------------------- active-subgraph pipeline
// Gephi-style filter chain, applied in order:
//   1. drop nodes hidden upstream (facet/keyword filters, subtree lens)
//   2. drop edges below the weight threshold
//   3. drop nodes below the degree threshold (degree counted on 2's result)
//   4. optionally keep only the largest connected component
function cgComputeActive(papers, edges, hiddenIds, gf) {
  let nodes = hiddenIds && hiddenIds.size
    ? papers.filter(p => !hiddenIds.has(p.id))
    : papers.slice();
  let nodeSet = new Set(nodes.map(p => p.id));

  const minW = Number(gf.minEdgeW) || 0;
  const pairSeen = new Set();
  let act = [];
  for (const e of edges) {
    if (!e || e.source === e.target) continue;
    if (!nodeSet.has(e.source) || !nodeSet.has(e.target)) continue;
    if ((e.weight == null ? 1 : e.weight) < minW) continue;
    const key = e.source < e.target ? e.source + "|" + e.target
                                    : e.target + "|" + e.source;
    if (pairSeen.has(key)) continue;
    pairSeen.add(key);
    act.push(e);
  }

  if ((Number(gf.minDegree) || 0) > 0) {
    const deg = {};
    for (const e of act) {
      deg[e.source] = (deg[e.source] || 0) + 1;
      deg[e.target] = (deg[e.target] || 0) + 1;
    }
    nodes = nodes.filter(p => (deg[p.id] || 0) >= gf.minDegree);
    nodeSet = new Set(nodes.map(p => p.id));
    act = act.filter(e => nodeSet.has(e.source) && nodeSet.has(e.target));
  }

  if (gf.largestOnly && nodes.length) {
    const comp = cgComponents(nodes, act);
    let best = 0, bestSize = 0;
    for (const [cid, size] of comp.sizes) {
      if (size > bestSize) { best = cid; bestSize = size; }
    }
    nodes = nodes.filter(p => comp.of[p.id] === best);
    nodeSet = new Set(nodes.map(p => p.id));
    act = act.filter(e => nodeSet.has(e.source) && nodeSet.has(e.target));
  }

  return { nodes, nodeSet, edges: act };
}

// Union-find over the id lists; returns {of: id->compId, sizes: Map}.
function cgComponents(nodes, edges) {
  const idx = {}, parent = [], rank = [];
  nodes.forEach((p, i) => { idx[p.id] = i; parent.push(i); rank.push(0); });
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const union = (a, b) => {
    a = find(a); b = find(b);
    if (a === b) return;
    if (rank[a] < rank[b]) [a, b] = [b, a];
    parent[b] = a;
    if (rank[a] === rank[b]) rank[a]++;
  };
  for (const e of edges) union(idx[e.source], idx[e.target]);
  const of = {}, sizes = new Map();
  for (const p of nodes) {
    const r = find(idx[p.id]);
    of[p.id] = r;
    sizes.set(r, (sizes.get(r) || 0) + 1);
  }
  return { of, sizes };
}

// Whole-network statistics for the stats card. Diameter is exact all-sources
// BFS up to 400 nodes; above that a pseudo-diameter (alternating
// farthest-node sweeps — exact on trees, tight on real citation graphs) keeps
// it O(edges). The exact version at 1500 nodes was the multi-second UI stall
// whenever a filter changed.
function cgComputeStats(active) {
  const n = active.nodes.length, m = active.edges.length;
  if (!n) return null;
  const comp = cgComponents(active.nodes, active.edges);
  let largest = 0, bestComp = -1;
  for (const [cid, size] of comp.sizes) {
    if (size > largest) { largest = size; bestComp = cid; }
  }

  let diameter = null;
  if (n >= 2 && largest >= 2) {
    const idx = {}, ids = [];
    active.nodes.forEach((p, i) => { idx[p.id] = i; ids.push(p.id); });
    const adj = Array.from({ length: n }, () => []);
    for (const e of active.edges) {
      adj[idx[e.source]].push(idx[e.target]);
      adj[idx[e.target]].push(idx[e.source]);
    }
    const inBest = (i) => comp.of[ids[i]] === bestComp;
    const dist = new Int32Array(n);
    const queue = new Int32Array(n);
    const bfs = (s) => {  // -> [farthest node, its distance]
      dist.fill(-1); dist[s] = 0;
      let head = 0, tail = 0, far = s, fd = 0;
      queue[tail++] = s;
      while (head < tail) {
        const u = queue[head++];
        for (const v of adj[u]) {
          if (dist[v] === -1) {
            dist[v] = dist[u] + 1;
            if (dist[v] > fd) { fd = dist[v]; far = v; }
            queue[tail++] = v;
          }
        }
      }
      return [far, fd];
    };
    if (n <= 400) {
      diameter = 0;
      for (let s = 0; s < n; s++) {
        if (!inBest(s)) continue;
        const [, fd] = bfs(s);
        if (fd > diameter) diameter = fd;
      }
    } else {
      let s = 0;
      while (s < n && !inBest(s)) s++;
      diameter = 0;
      for (let sweep = 0; sweep < 4 && s < n; sweep++) {
        const [far, fd] = bfs(s);
        if (fd <= diameter) break;
        diameter = fd;
        s = far;
      }
    }
  }
  return {
    nodes: n,
    edges: m,
    avgDeg: n ? (2 * m) / n : 0,
    density: n > 1 ? (2 * m) / (n * (n - 1)) : 0,
    components: comp.sizes.size,
    largest,
    diameter,
  };
}

function CiteGraph({ papers, edges, dataKey, selectedId, onSelect, onHover,
                     theme, kind = "paper", labels = false, hiddenIds = null,
                     sizing, tools = {}, legendSeed = "auto", sizeHint = false,
                     emptyHint, onStats, onGraphHidden, topLeft, children }) {
  const wrapRef = React.useRef(null);
  const boxRef = React.useRef(null);
  const gridRef = React.useRef(null);
  const tipRef = React.useRef(null);
  const st = React.useRef({
    sigma: null, graph: null, layout: null, rand: cgMulberry32(7),
    pal: null, cdomain: { min: 2018, max: 2025 }, adj: {}, adjPairs: {},
    selected: null, hovered: null, anchor: null, connected: new Set(),
    top3: new Set(), labels: false, paperById: {}, posCache: {},
    activeData: { nodes: [], nodeSet: new Set(), edges: [] },
    kind: "paper", legacy: false, vmax: 0, wrange: { lo: 0, hi: 0 }, origHover: null,
    sizeK: 1, origZoomFn: null,
    dataKey: null, stopTimer: null, animRAF: 0, replayTimer: null,
    syncTimer: null,
    dragging: null, dragHome: null, dragLast: null, dragVel: { x: 0, y: 0 },
    _dragMoved: false, bounceRAF: 0, _bounceId: null, layoutWasRunning: false,
    fa2: { ...CG_FA2_DEFAULTS }, vis: { ...CG_VIS_DEFAULTS }, gf: { ...CG_GF_DEFAULTS },
    onSelect: null, onHover: null, onStats: null,
  }).current;
  st.onSelect = onSelect || null;
  st.onHover = onHover || null;
  st.onStats = onStats || null;
  st.kind = kind;
  st.legacy = sizing === "legacy";

  const [libs, setLibs] = React.useState(() =>
    window.GraphLibs ? (window.GraphLibs.error ? "error" : "ready") : "loading");
  const [zoomPct, setZoomPct] = React.useState(100);
  const [layoutOn, setLayoutOn] = React.useState(false);
  const [replaying, setReplaying] = React.useState(false);
  const [optsOpen, setOptsOpen] = React.useState(false);
  // fa2 / vis / showLabels seed from the shared cross-page prefs (see CG_PREFS);
  // gf stays per-dataset.
  const [fa2, setFa2] = React.useState(() => ({ ...CG_PREFS.fa2 }));
  const [vis, setVis] = React.useState(() => ({ ...CG_PREFS.vis }));
  const [gf, setGf] = React.useState({ ...CG_GF_DEFAULTS });
  const [showLabels, setShowLabels] = React.useState(
    () => (typeof CG_PREFS.labels === "boolean" ? CG_PREFS.labels : !!labels));
  const [stats, setStats] = React.useState(null);
  const [palTick, setPalTick] = React.useState(0);
  st.fa2 = fa2; st.vis = vis; st.gf = gf; st.labels = showLabels;

  // Filter edits hit the simulation only after a short pause — dragging the
  // edge-weight slider fires dozens of onChange ticks, and rebuilding the
  // active subgraph + stats + layout per tick froze the UI for seconds.
  const [gfApplied, setGfApplied] = React.useState(() => ({ ...CG_GF_DEFAULTS }));
  React.useEffect(() => {
    const t = setTimeout(() => setGfApplied(gf), 220);
    return () => clearTimeout(t);
  }, [gf]);
  // Structural filters are dataset-scoped (a min-edge-weight tuned to one
  // graph's weight range is meaningless on another). Reset them the moment
  // the dataset switches — during render, so the rebuild effect can never
  // see the previous dataset's thresholds through the debounce window.
  const [gfForKey, setGfForKey] = React.useState(dataKey);
  if (gfForKey !== dataKey) {
    setGfForKey(dataKey);
    setGf({ ...CG_GF_DEFAULTS });
    setGfApplied({ ...CG_GF_DEFAULTS });
  }

  React.useEffect(() => {
    if (libs !== "loading") return;
    const onReady = () => setLibs(window.GraphLibs && window.GraphLibs.error ? "error" : "ready");
    window.addEventListener("graphlibs", onReady);
    if (window.GraphLibs) onReady();
    return () => window.removeEventListener("graphlibs", onReady);
  }, [libs]);

  // ---- derived data ------------------------------------------------------
  const active = React.useMemo(
    () => cgComputeActive(papers, edges, hiddenIds, gfApplied),
    [papers, edges, hiddenIds, gfApplied]);
  st.activeData = active;

  const cdomain = React.useMemo(
    () => (kind === "author" ? cgAuthorDomain(papers) : cgYearDomain(papers)),
    [papers, kind]);

  // Size-normalization ceiling over the VISIBLE nodes, not the full dataset:
  // capping citations at 1000 while a 224k-cite giant exists off-screen must
  // not squash every visible node onto the size floor — sizes rank within
  // the filtered view, like Gephi's ranking on a filtered workspace.
  const vmax = React.useMemo(() => {
    let v = 0;
    for (const p of active.nodes) {
      const x = Number(p.cites) || 0;
      if (x > v) v = x;
    }
    return v;
  }, [active, kind]);
  st.vmax = vmax;
  React.useEffect(() => {
    st.vmax = vmax;
    if (!st.legacy) resizeNodes();  // renormalize what's already on canvas
  }, [vmax]);  // eslint-disable-line

  // Weight range over the VISIBLE edges (same renormalize-on-filter rule as
  // node sizes) — feeds the width mapping in the edge reducer.
  const wrange = React.useMemo(() => {
    let lo = Infinity, hi = 0;
    for (const e of active.edges) {
      const w = e.weight == null ? 1 : Number(e.weight) || 0;
      if (w < lo) lo = w;
      if (w > hi) hi = w;
    }
    return { lo: lo === Infinity ? 0 : lo, hi };
  }, [active]);
  st.wrange = wrange;
  React.useEffect(() => {
    st.wrange = wrange;
    if (st.sigma) st.sigma.refresh();
  }, [wrange]);  // eslint-disable-line

  const ranges = React.useMemo(() => {
    let maxW = 0, minW = Infinity, deg = {}, maxDeg = 0;
    for (const e of edges) {
      if (!e || e.source === e.target) continue;
      const w = e.weight == null ? 1 : e.weight;
      if (w > maxW) maxW = w;
      if (w < minW) minW = w;
      deg[e.source] = (deg[e.source] || 0) + 1;
      deg[e.target] = (deg[e.target] || 0) + 1;
    }
    for (const id in deg) if (deg[id] > maxDeg) maxDeg = deg[id];
    return { maxW, weighted: maxW > (minW === Infinity ? 0 : minW),
             maxDeg: Math.min(50, maxDeg) };
  }, [edges]);

  // ids removed by the graph-side filters (degree/weight/component) beyond
  // the upstream facet filters — reported so the paper list can agree.
  React.useEffect(() => {
    if (!onGraphHidden) return;
    const removed = new Set();
    for (const p of papers) {
      if (hiddenIds && hiddenIds.has(p.id)) continue;
      if (!active.nodeSet.has(p.id)) removed.add(p.id);
    }
    onGraphHidden(removed);
  }, [active]);  // eslint-disable-line

  // stats card (debounced — diameter is the pricey part)
  React.useEffect(() => {
    if (!tools.stats) return;
    const t = setTimeout(() => setStats(cgComputeStats(active)), 220);
    return () => clearTimeout(t);
  }, [active, tools.stats]);

  const emitStats = () => {
    if (st.onStats && st.graph) {
      st.onStats({ nodes: st.graph.order, edges: st.graph.size });
    }
  };

  // ---- node / edge insertion (f5 spawn-near-neighbour + grow animation) --
  const nodeAttrs = (p, animate) => {
    const size = cgNodeSize(p, st.kind, st.vis, st.vmax, st.legacy) * (st.sizeK || 1);
    const cv = cgColorValue(p, st.kind, st.cdomain);
    return {
      label: cgTrunc(p.title, 48),
      size: animate ? 0.001 : size,
      _targetSize: size,
      _animSize: animate ? 0.001 : size,
      _animStart: performance.now(),
      color: p.seed ? st.pal.seed : cgRampColor(st.pal, st.cdomain, cv),
      _seed: !!p.seed,
      _cval: cv,
      haloColor: p.seed ? st.pal.seedHalo : CG_TRANSPARENT,
      strokeColor: p.seed ? st.pal.seedStroke : st.pal.netInk,
      haloSize: p.seed ? 0.22 : 0.0,
      // demo's node "line style": an ink ring whose width grows with the node
      strokeSize: p.seed ? 0.09 : 0.14,
    };
  };

  const addPaperNode = (p, animate, pos) => {
    const g = st.graph;
    if (!g || g.hasNode(p.id)) return;
    let x, y;
    const present = pos ? [] : (st.adj[p.id] || []).filter(o => g.hasNode(o));
    if (pos) {
      x = pos.x; y = pos.y;
    } else if (present.length) {
      const nb = present[Math.floor(st.rand() * present.length)];
      x = g.getNodeAttribute(nb, "x") + (st.rand() - 0.5) * 10;
      y = g.getNodeAttribute(nb, "y") + (st.rand() - 0.5) * 10;
    } else if (g.order > 0) {
      const nodes = g.nodes();
      const nb = nodes[Math.floor(st.rand() * nodes.length)];
      x = g.getNodeAttribute(nb, "x") + (st.rand() - 0.5) * 14;
      y = g.getNodeAttribute(nb, "y") + (st.rand() - 0.5) * 14;
    } else {
      x = (st.rand() - 0.5) * 20;
      y = (st.rand() - 0.5) * 20;
    }
    g.addNode(p.id, { x, y, ...nodeAttrs(p, animate) });
  };

  const addEdgesFor = (id) => {
    const g = st.graph;
    for (const [a, b, w] of (st.adjPairs[id] || [])) {
      if (a !== b && g.hasNode(a) && g.hasNode(b) && !g.hasEdge(a, b)) {
        // `weight` feeds FA2's edgeWeightInfluence (0-weight links don't pull)
        g.addEdge(a, b, { size: 0.8 * (st.sizeK || 1), weight: w });
      }
    }
  };

  const rebuildAdjacency = (edgeArr) => {
    st.adj = {}; st.adjPairs = {};
    for (const e of edgeArr) {
      if (!e || e.source === e.target) continue;
      const w = e.weight == null ? 1 : e.weight;
      (st.adj[e.source] ||= []).push(e.target);
      (st.adj[e.target] ||= []).push(e.source);
      (st.adjPairs[e.source] ||= []).push([e.source, e.target, w]);
      (st.adjPairs[e.target] ||= []).push([e.source, e.target, w]);
    }
  };

  const refreshTop3 = (ps) => {
    st.top3 = new Set([...ps].sort((a, b) => (b.cites || 0) - (a.cites || 0))
      .slice(0, 3).map(p => p.id));
  };

  // ---- layout: worker (live streaming) + synchronous bursts (instant) ----
  const heatLayout = (ms) => {
    if (!st.layout) return;
    try { st.layout.start(); } catch (_) {}
    setLayoutOn(true);
    clearTimeout(st.stopTimer);
    // dense graphs need FA2 time roughly ∝ edges, not just nodes
    const win = ms != null ? ms : Math.min(45000,
      6000 + (st.graph ? st.graph.order * 12 + st.graph.size * 5 : 0));
    st.stopTimer = setTimeout(() => {
      if (st.layout) { try { st.layout.stop(); } catch (_) {} }
      // overlap-prevention invariant: whenever the layout comes to rest,
      // residual collisions are swept away
      if (st.fa2.adjustSizes) separationSweep();
      setLayoutOn(false);
    }, win);
  };

  // One-shot synchronous FA2 head start (same algorithm + settings): a few
  // hundred iterations before the next paint, so a fresh dataset appears
  // with coarse structure instead of the random init. Cheap (~0.3s for 640
  // nodes) and skipped silently if the sync entry failed to load.
  // graphology-FA2's Barnes-Hut branch computes repulsion on the quad-tree
  // WITHOUT the size-aware anti-collision term — with BH on, adjustSizes is
  // silently ignored (measured: ~10k overlapping pairs that never resolve).
  // Exact O(n²) repulsion honours it and is cheap at this scale (~2ms/iter
  // at 640 nodes), so overlap-prevention forces BH off.
  const layoutSettings = (settings) =>
    settings.adjustSizes ? { ...settings, barnesHutOptimize: false } : { ...settings };

  const prewarmLayout = () => {
    const sync = window.GraphLibs && window.GraphLibs.fa2Sync;
    if (!st.graph || !sync || st.graph.order < 20) return;
    try {
      sync.assign(st.graph, {
        iterations: CG_PREWARM_ITERS,
        settings: layoutSettings({ ...st.fa2, slowDown: Math.max(2, st.fa2.slowDown / 2) }),
      });
    } catch (_) {}
  };

  // Fast settle: the SAME worker, run for a bounded window at half slowDown
  // (double step size — many gentle iterations' worth per second), then
  // swapped back to the configured dynamics for a short polish. Runs in the
  // web worker, so it converges at full speed even in a background tab and
  // never blocks the UI.
  const makeLayout = (settings) => {
    if (st.layout) { try { st.layout.kill(); } catch (_) {} st.layout = null; }
    if (!st.graph || !window.GraphLibs || !window.GraphLibs.FA2Layout) return;
    st.layout = new window.GraphLibs.FA2Layout(st.graph, { settings: layoutSettings(settings) });
  };
  const fastSettle = (ms) => {
    if (!st.graph) return;
    clearTimeout(st.stopTimer);
    makeLayout({ ...st.fa2, slowDown: Math.max(2, st.fa2.slowDown / 2) });
    if (!st.layout) return;
    try { st.layout.start(); } catch (_) {}
    setLayoutOn(true);
    st.stopTimer = setTimeout(() => {
      makeLayout(st.fa2);
      heatLayout(CG_POLISH_MS);
    }, ms);
  };

  // Final deterministic separation (Gephi's Noverlap idea): FA2's
  // anti-collision converges the bulk, but mega-hubs pinned by hundreds of
  // edges keep a small overlap tail — push residual overlapping pairs apart
  // along their axis until none remain. ~100ms at 640 nodes.
  const separationSweep = () => {
    const g = st.graph;
    if (!g || g.order < 2) return;
    const ids = g.nodes();
    const at = ids.map(id => {
      const a = g.getNodeAttributes(id);
      return { x: a.x, y: a.y, size: a.size };
    });
    for (let pass = 0; pass < 50; pass++) {
      let moved = 0;
      for (let i = 0; i < at.length; i++) {
        for (let j = i + 1; j < at.length; j++) {
          let dx = at[j].x - at[i].x, dy = at[j].y - at[i].y;
          let d = Math.hypot(dx, dy);
          const need = at[i].size + at[j].size + 0.5;
          if (d >= need) continue;
          if (d < 1e-6) { dx = 1; dy = 0; d = 1; }
          const push = (need - d) / 2, ux = dx / d, uy = dy / d;
          at[i].x -= ux * push; at[i].y -= uy * push;
          at[j].x += ux * push; at[j].y += uy * push;
          moved++;
        }
      }
      if (!moved) break;
    }
    ids.forEach((id, i) => g.mergeNodeAttributes(id, { x: at[i].x, y: at[i].y }));
    if (st.sigma) st.sigma.refresh();
    drawGrid();
  };

  const toggleLayout = () => {
    if (!st.layout) return;
    if (st.layout.isRunning()) {
      clearTimeout(st.stopTimer);
      st.layout.stop();
      if (st.fa2.adjustSizes) separationSweep();
      setLayoutOn(false);
    } else heatLayout();
  };

  const rebuildLayout = (settings, ms) => {
    if (!st.graph || !window.GraphLibs || !window.GraphLibs.FA2Layout) return;
    st.fa2 = { ...settings };
    fastSettle(ms != null ? ms : CG_EDIT_MS);
  };

  // Gephi-true geometry for "Prevent node overlap": FA2's adjustSizes keeps
  // the discs separated in LAYOUT coordinates, but sigma normally renders
  // node sizes in a screen-pixel reference decoupled from the graph→viewport
  // scale — compressing the layout to fit the pane shrinks the gaps but not
  // the dots, so separated discs still LOOK overlapped. While the toggle is
  // on, sizes join the position coordinate system (like Gephi's canvas) and
  // are scaled by the layout's units-per-pixel (sizeK) so they LOOK the same
  // as before — FA2 then separates exactly the discs you see: zero overlap
  // in the simulation is zero overlap on screen.
  const computeSizeK = () => {
    if (!st.graph || !st.graph.order || !wrapRef.current) return 1;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    st.graph.forEachNode((id, a) => {
      if (a.x < minX) minX = a.x;
      if (a.x > maxX) maxX = a.x;
      if (a.y < minY) minY = a.y;
      if (a.y > maxY) maxY = a.y;
    });
    const vw = Math.max(200, wrapRef.current.clientWidth - 80);
    const vh = Math.max(200, wrapRef.current.clientHeight - 80);
    return Math.max(1, Math.max((maxX - minX) / vw, (maxY - minY) / vh));
  };
  const applySizeReference = (on) => {
    if (!st.sigma) return;
    st.sizeK = on ? computeSizeK() : 1;
    resizeNodes();
    if (st.graph) {
      st.graph.forEachEdge((eid) => st.graph.setEdgeAttribute(eid, "size", 0.8 * st.sizeK));
    }
    try {
      st.sigma.setSetting("itemSizesReference", on ? "positions" : "screen");
      st.sigma.setSetting("zoomToSizeRatioFunction",
        on ? ((r) => r) : (st.origZoomFn || ((r) => Math.sqrt(r))));
      st.sigma.refresh();
    } catch (_) {}
  };

  const stopReplay = () => {
    clearInterval(st.replayTimer);
    st.replayTimer = null;
    setReplaying(false);
  };

  const teardown = () => {
    stopReplay();
    clearTimeout(st.stopTimer);
    clearTimeout(st.syncTimer);
    clearTimeout(st.fa2Timer);
    cancelAnimationFrame(st.animRAF);
    cancelAnimationFrame(st.bounceRAF);
    st.dragging = null; st._bounceId = null; st.bounceRAF = 0;
    if (st.layout) { try { st.layout.kill(); } catch (_) {} st.layout = null; }
    if (st.sigma) { try { st.sigma.kill(); } catch (_) {} st.sigma = null; }
    st.graph = null;
    st.posCache = {};
  };

  // ---- camera-synced dot grid (the run canvas backdrop) -----------------
  const drawGrid = () => {
    const canvas = gridRef.current, wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const cssW = wrap.clientWidth, cssH = wrap.clientHeight;
    if (!cssW || !cssH) return;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
      canvas.width = cssW * dpr; canvas.height = cssH * dpr;
      canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = st.pal ? st.pal.gridDot : "rgba(68,60,40,0.09)";
    const step = 16;
    let ox = 0, oy = 0;
    if (st.sigma) {
      try {
        const o = st.sigma.graphToViewport({ x: 0, y: 0 });
        ox = ((o.x % step) + step) % step;
        oy = ((o.y % step) + step) % step;
      } catch (_) {}
    }
    for (let gx = ox; gx < cssW; gx += step) {
      for (let gy = oy; gy < cssH; gy += step) {
        ctx.fillRect(gx, gy, 1, 1);
      }
    }
  };

  // ---- node drag + bounce (the demo's grab-and-spring-home) --------------
  const CG_BOUNCE = 0.5;
  const beginNodeDrag = (id) => {
    const g = st.graph;
    if (!g || !g.hasNode(id)) return;
    cancelAnimationFrame(st.bounceRAF); st.bounceRAF = 0; st._bounceId = null;
    st.dragging = id;
    st.dragHome = { x: g.getNodeAttribute(id, "x"), y: g.getNodeAttribute(id, "y") };
    st.dragLast = { ...st.dragHome };
    st.dragVel = { x: 0, y: 0 };
    st._dragMoved = false;
    st.layoutWasRunning = false;
    if (boxRef.current) boxRef.current.style.cursor = "grabbing";
    hideTip();
    if (st.onSelect) st.onSelect(id);   // grab reveals the neighbourhood, like the demo
  };
  const moveNodeDrag = (e) => {
    const g = st.graph;
    if (st.dragging === null || !g || !st.sigma || !g.hasNode(st.dragging)) return false;
    if (!st._dragMoved) {
      // pause the live layout only once the drag really starts, so a plain
      // click never disturbs the running simulation
      st._dragMoved = true;
      st.layoutWasRunning = !!(st.layout && st.layout.isRunning());
      if (st.layoutWasRunning) {
        clearTimeout(st.stopTimer);
        try { st.layout.stop(); } catch (_) {}
        setLayoutOn(false);
      }
    }
    const pos = st.sigma.viewportToGraph({ x: e.x, y: e.y });
    st.dragVel = { x: pos.x - st.dragLast.x, y: pos.y - st.dragLast.y };
    st.dragLast = { x: pos.x, y: pos.y };
    g.setNodeAttribute(st.dragging, "x", pos.x);
    g.setNodeAttribute(st.dragging, "y", pos.y);
    st.sigma.refresh();
    return true;
  };
  const endNodeDrag = () => {
    if (st.dragging === null) return;
    const id = st.dragging;
    st.dragging = null;
    if (boxRef.current) boxRef.current.style.cursor = "";
    if (!st._dragMoved) return;   // a click, not a drag — nothing to spring back
    const g = st.graph;
    if (!g || !g.hasNode(id) || !st.dragHome) return;
    st._bounceId = id;
    const home = st.dragHome;
    const damp = 0.92 - CG_BOUNCE * 0.30;   // bouncier release ⇒ more overshoot
    const stiff = 0.16;
    const step = () => {
      st.bounceRAF = 0;
      if (st._bounceId !== id || !st.graph || !st.graph.hasNode(id)) return;
      let x = st.graph.getNodeAttribute(id, "x");
      let y = st.graph.getNodeAttribute(id, "y");
      const dx = home.x - x, dy = home.y - y;
      if (Math.abs(st.dragVel.x) > 0.03 || Math.abs(st.dragVel.y) > 0.03
          || Math.abs(dx) > 0.05 || Math.abs(dy) > 0.05) {
        st.dragVel.x = (st.dragVel.x + dx * stiff) * damp;
        st.dragVel.y = (st.dragVel.y + dy * stiff) * damp;
        x += st.dragVel.x; y += st.dragVel.y;
        st.graph.setNodeAttribute(id, "x", x);
        st.graph.setNodeAttribute(id, "y", y);
        if (st.sigma) st.sigma.refresh();
        st.bounceRAF = requestAnimationFrame(step);
      } else {
        st.graph.setNodeAttribute(id, "x", home.x);
        st.graph.setNodeAttribute(id, "y", home.y);
        if (st.sigma) st.sigma.refresh();
        st._bounceId = null;
        if (st.layoutWasRunning) heatLayout();   // hand the node back to the live layout
      }
    };
    st.bounceRAF = requestAnimationFrame(step);
  };

  // ---- build / rebuild on data-source change ----------------------------
  React.useEffect(() => {
    if (libs !== "ready" || !boxRef.current) return;
    teardown();
    st.dataKey = dataKey;
    if (!papers.length) {
      drawGrid();
      if (st.onStats) st.onStats({ nodes: 0, edges: 0 });
      return;
    }
    const { Graph, Sigma, FA2Layout, createNodeBorderProgram } = window.GraphLibs;
    st.rand = cgMulberry32(7);
    st.pal = cgReadPalette(st.vis.palette);
    st.cdomain = cdomain;
    rebuildAdjacency(st.activeData.edges);
    refreshTop3(st.activeData.nodes);

    const g = new Graph({ multi: false, type: "undirected" });
    st.graph = g;
    // Gephi-style init: scatter over a wide disc and let FA2 contract into
    // structure. Seeding everything near the origin (the f5 spawn rule, kept
    // for incremental/live additions) jams dense graphs into a uniform ball.
    const R = 12 * Math.sqrt(Math.max(1, st.activeData.nodes.length));
    for (const p of st.activeData.nodes) {
      const a = st.rand() * Math.PI * 2;
      const r = R * Math.sqrt(st.rand());
      addPaperNode(p, false, { x: Math.cos(a) * r, y: Math.sin(a) * r });
    }
    for (const p of st.activeData.nodes) addEdgesFor(p.id);
    if (dataKey !== "live") prewarmLayout();  // coarse structure before paint

    // Node program: ring (selection) + halo (seeds) + stroke + fill.
    // If @sigma/node-border failed to load, plain circles still work.
    const settings = {
      allowInvalidContainer: true,
      zIndex: true,
      labelFont: "Inter, -apple-system, sans-serif",
      labelSize: 11,
      labelWeight: "400",
      labelRenderedSizeThreshold: st.labels ? 7 : 10000,
      labelColor: { attribute: "labelColor", color: st.pal.ink1 },
      defaultEdgeColor: st.pal.inkFaint,
      minCameraRatio: 0.05,
      maxCameraRatio: 8,
      nodeReducer: (id, data) => {
        const size = data._animSize ?? data.size;
        const base = { ...data, size, labelColor: st.pal.ink1,
                       ringColor: CG_TRANSPARENT, ringSize: 0.12,
                       forceLabel: st.labels && st.top3.has(id) };
        if (id === st.selected) {
          return { ...base, zIndex: 3, forceLabel: st.labels,
                   ringColor: st.pal.selStrong };
        }
        if (id === st.hovered) {
          return { ...base, zIndex: 3, ringColor: st.pal.selSoft };
        }
        if (st.anchor) {
          if (st.connected.has(id)) return { ...base, zIndex: 2 };
          // dim the out-of-neighbourhood nodes (demo-style) against the paper base
          return { ...base,
                   color: cgFade(data.color, 0.15, st.pal.bgRgb),
                   haloColor: CG_TRANSPARENT,
                   strokeColor: cgFade(data.strokeColor, 0.15, st.pal.bgRgb),
                   labelColor: cgFade(st.pal.ink1, 0.2, st.pal.bgRgb),
                   forceLabel: false };
        }
        return base;
      },
      edgeReducer: (edge, data) => {
        const src = st.graph.source(edge), tgt = st.graph.target(edge);
        const k = st.sizeK || 1;
        // width stays weight-mapped (our project); only the colour follows the
        // demo's ink. Selection lights the whole neighbourhood sub-graph.
        const bw = (st.legacy ? 0.3 : cgEdgeWidth(data.weight, st.wrange, st.vis)) * k;
        if (st.anchor) {
          const inHood = (id) => id === st.anchor || st.connected.has(id);
          if (inHood(src) && inHood(tgt)) {
            return { ...data, color: cgFade(st.pal.selStrong, 0.72, st.pal.bgRgb),
                     size: Math.max(bw * 1.6, bw + 0.8 * k), zIndex: 1 };
          }
          return { ...data, color: cgFade(st.pal.netInk, 0.06, st.pal.bgRgb), size: bw };
        }
        return { ...data, color: cgFade(st.pal.netInk, 0.22, st.pal.bgRgb), size: bw };
      },
    };
    if (createNodeBorderProgram) {
      try {
        settings.defaultNodeType = "bordered";
        settings.nodeProgramClasses = {
          bordered: createNodeBorderProgram({
            borders: [
              { size: { attribute: "ringSize", defaultValue: 0.12 },
                color: { attribute: "ringColor" } },
              { size: { attribute: "haloSize", defaultValue: 0.0 },
                color: { attribute: "haloColor" } },
              { size: { attribute: "strokeSize", defaultValue: 0.12 },
                color: { attribute: "strokeColor" } },
              { size: { fill: true }, color: { attribute: "color" } },
            ],
          }),
        };
      } catch (_) {
        delete settings.defaultNodeType;
        delete settings.nodeProgramClasses;
      }
    }

    const sigma = new Sigma(g, boxRef.current, settings);
    st.sigma = sigma;
    // The .net-tip tooltip is the hover surface; sigma's own white
    // hover-label box only makes sense when labels are on.
    st.origHover = sigma.getSetting("defaultDrawNodeHover");
    if (!st.labels) sigma.setSetting("defaultDrawNodeHover", () => {});
    st.origZoomFn = sigma.getSetting("zoomToSizeRatioFunction");
    if (st.fa2.adjustSizes) applySizeReference(true);

    sigma.on("clickNode", ({ node }) => st.onSelect && st.onSelect(node));
    sigma.on("clickStage", () => { if (st.selected) st.onSelect && st.onSelect(null); });
    sigma.on("enterNode", ({ node, event }) => {
      st.hovered = node;
      st.anchor = st.selected || st.hovered;
      recomputeConnected();
      if (st.onHover) st.onHover(node);
      showTip(node, event);
      sigma.refresh();
    });
    sigma.on("leaveNode", () => {
      st.hovered = null;
      st.anchor = st.selected;
      recomputeConnected();
      if (st.onHover) st.onHover(null);
      hideTip();
      sigma.refresh();
    });
    sigma.on("downNode", ({ node }) => beginNodeDrag(node));
    sigma.getMouseCaptor().on("mousemovebody", (e) => {
      if (st.dragging !== null) {
        if (moveNodeDrag(e)) {
          e.preventSigmaDefault();
          if (e.original) { e.original.preventDefault(); e.original.stopPropagation(); }
        }
        return;
      }
      moveTip(e);
    });
    sigma.getMouseCaptor().on("mouseup", () => endNodeDrag());
    const cam = sigma.getCamera();
    cam.on("updated", () => { setZoomPct(Math.round(100 / cam.ratio)); drawGrid(); });
    setZoomPct(Math.round(100 / cam.ratio));

    if (dataKey === "live") {
      makeLayout(st.fa2);
      heatLayout();                  // streaming source: keep it warm
    } else {
      fastSettle(CG_SETTLE_MS);      // static dataset: settle now, then polish
    }
    emitStats();
    drawGrid();

    // f5 grow-animation ticker (also keeps the grid glued under FA2 motion)
    const tick = () => {
      const now = performance.now();
      let dirty = false;
      if (st.graph) {
        st.graph.forEachNode((id, attrs) => {
          if (attrs._targetSize !== attrs._animSize) {
            const t = Math.min(1, (now - attrs._animStart) / 400);
            const v = Math.max(0.001, cgEaseOutCubic(t) * attrs._targetSize);
            st.graph.setNodeAttribute(id, "_animSize", t >= 1 ? attrs._targetSize : v);
            st.graph.setNodeAttribute(id, "size", t >= 1 ? attrs._targetSize : v);
            dirty = true;
          }
        });
      }
      if (dirty && st.sigma) st.sigma.refresh();
      if (dirty || (st.layout && st.layout.isRunning())) drawGrid();
      st.animRAF = requestAnimationFrame(tick);
    };
    st.animRAF = requestAnimationFrame(tick);

    return teardown;
  }, [libs, dataKey, papers.length === 0]);  // eslint-disable-line

  // ---- keep the graph in sync with data + filters (removal-based) --------
  // One debounced diff covers live streaming (new papers appear with the
  // grow animation) AND filter edits (nodes/edges leave the simulation and
  // FA2 re-flows). Removed nodes park their positions in posCache so
  // loosening a filter puts them back where they were.
  const syncGraph = () => {
    const g = st.graph;
    if (!g) return;
    const act = st.activeData;

    if (st.cdomain.min !== cdomain.min || st.cdomain.max !== cdomain.max
        || st.cdomain.field !== cdomain.field) {
      st.cdomain = cdomain;
      recolorNodes();
    }
    rebuildAdjacency(act.edges);

    let changed = false;
    const toDrop = [];
    g.forEachNode((id) => { if (!act.nodeSet.has(id)) toDrop.push(id); });
    for (const id of toDrop) {
      st.posCache[id] = { x: g.getNodeAttribute(id, "x"), y: g.getNodeAttribute(id, "y") };
      g.dropNode(id);
      changed = true;
    }
    let freshNodes = 0;
    for (const p of act.nodes) {
      if (!g.hasNode(p.id)) {
        addPaperNode(p, true, st.posCache[p.id] || null);
        freshNodes++;
        changed = true;
      }
    }
    const want = new Set();
    for (const e of act.edges) {
      want.add(e.source < e.target ? e.source + "|" + e.target
                                   : e.target + "|" + e.source);
    }
    const dropEdges = [];
    g.forEachEdge((eid, attrs, s, t) => {
      if (!want.has(s < t ? s + "|" + t : t + "|" + s)) dropEdges.push(eid);
    });
    for (const eid of dropEdges) { g.dropEdge(eid); changed = true; }
    for (const e of act.edges) {
      if (g.hasNode(e.source) && g.hasNode(e.target) && !g.hasEdge(e.source, e.target)) {
        g.addEdge(e.source, e.target,
          { size: 0.8 * (st.sizeK || 1), weight: e.weight == null ? 1 : e.weight });
        changed = true;
      }
    }

    if (changed) {
      refreshTop3(act.nodes);
      if (st.dataKey === "live" && freshNodes) heatLayout();
      else fastSettle(CG_EDIT_MS);
      emitStats();
      if (st.sigma) st.sigma.refresh();
    }
  };

  React.useEffect(() => {
    if (!st.sigma || st.dataKey !== dataKey || st.replayTimer) return;
    clearTimeout(st.syncTimer);
    st.syncTimer = setTimeout(syncGraph, 140);
    return () => clearTimeout(st.syncTimer);
  }, [active, cdomain]);  // eslint-disable-line

  // ---- selection --------------------------------------------------------
  const recomputeConnected = () => {
    st.connected = new Set();
    if (st.anchor && st.graph && st.graph.hasNode(st.anchor)) {
      for (const nb of st.graph.neighbors(st.anchor)) st.connected.add(nb);
    }
  };
  React.useEffect(() => {
    st.selected = selectedId && st.graph && st.graph.hasNode(selectedId) ? selectedId : null;
    st.anchor = st.selected || st.hovered;
    recomputeConnected();
    if (st.sigma) st.sigma.refresh();
  }, [selectedId, papers]);

  // ---- palette / theme --------------------------------------------------
  const recolorNodes = () => {
    if (!st.graph) return;
    st.graph.forEachNode((nid) => {
      const p = st.paperById[nid];
      if (!p) return;
      const cv = cgColorValue(p, st.kind, st.cdomain);
      st.graph.mergeNodeAttributes(nid, {
        _cval: cv,
        color: p.seed ? st.pal.seed : cgRampColor(st.pal, st.cdomain, cv),
        haloColor: p.seed ? st.pal.seedHalo : CG_TRANSPARENT,
        strokeColor: p.seed ? st.pal.seedStroke : st.pal.netInk,
      });
    });
  };
  const applyPalette = () => {
    if (!st.sigma) return;
    st.pal = cgReadPalette(st.vis.palette);
    recolorNodes();
    st.sigma.setSetting("labelColor", { attribute: "labelColor", color: st.pal.ink1 });
    st.sigma.setSetting("defaultEdgeColor", st.pal.inkFaint);
    st.sigma.refresh();
    drawGrid();
    setPalTick(t => t + 1);  // legend re-reads CSS tokens post-flip
  };

  // Theme flip: deferred one frame — this child effect fires BEFORE app.jsx's
  // effect that stamps data-theme on <html>, so an immediate read would get
  // the OLD theme's tokens.
  React.useEffect(() => {
    if (!st.sigma) return;
    const id = requestAnimationFrame(applyPalette);
    return () => cancelAnimationFrame(id);
  }, [theme]);  // eslint-disable-line

  // ---- appearance edits (panel) -----------------------------------------
  const resizeNodes = () => {
    if (!st.graph) return;
    st.graph.forEachNode((nid) => {
      const p = st.paperById[nid];
      if (!p) return;
      const size = cgNodeSize(p, st.kind, st.vis, st.vmax, st.legacy) * (st.sizeK || 1);
      st.graph.mergeNodeAttributes(nid, { size, _targetSize: size, _animSize: size });
    });
    if (st.sigma) st.sigma.refresh();
  };

  const applyFa2 = (patch) => {
    const next = { ...st.fa2, ...patch };
    setFa2(next);
    st.fa2 = next;
    CG_PREFS.fa2 = next; cgSavePrefs();  // share force-layout choice across pages
    if (patch.adjustSizes !== undefined) applySizeReference(patch.adjustSizes);
    // separating a settled dense core needs a real settle window; the sweep
    // at layout-stop then removes any residual collisions
    const ms = patch.adjustSizes ? CG_SETTLE_MS : CG_EDIT_MS;
    clearTimeout(st.fa2Timer);
    st.fa2Timer = setTimeout(() => rebuildLayout(st.fa2, ms), 180);
  };
  const applyVis = (patch) => {
    const next = { ...st.vis, ...patch };
    setVis(next);
    st.vis = next;
    CG_PREFS.vis = next; cgSavePrefs();  // share appearance choice across pages
    if (patch.palette != null) applyPalette();
    if (patch.minSize != null || patch.maxSize != null || patch.sizeCurve != null) resizeNodes();
    if (patch.edgeMin != null || patch.edgeMax != null || patch.edgeCurve != null) {
      if (st.sigma) st.sigma.refresh();  // widths live in the edge reducer
    }
  };
  const applyGf = (patch) => setGf(g0 => ({ ...g0, ...patch }));
  const toggleLabels = (on) => {
    setShowLabels(on);
    st.labels = on;
    CG_PREFS.labels = on; cgSavePrefs();  // share label toggle across pages
    if (st.sigma) {
      st.sigma.setSetting("labelRenderedSizeThreshold", on ? 7 : 10000);
      try {
        st.sigma.setSetting("defaultDrawNodeHover", on ? st.origHover : () => {});
      } catch (_) {}
      st.sigma.refresh();
    }
  };
  const resetAll = () => {
    applyFa2({ ...CG_FA2_DEFAULTS });
    applyVis({ ...CG_VIS_DEFAULTS });
    applyGf({ ...CG_GF_DEFAULTS });
  };

  // ---- tooltip (run skin) -----------------------------------------------
  const paperById = React.useMemo(() => {
    const m = {}; for (const p of papers) m[p.id] = p; return m;
  }, [papers]);
  st.paperById = paperById;

  const showTip = (id, event) => {
    const p = st.paperById[id];
    const tip = tipRef.current;
    if (!p || !tip) return;
    tip.innerHTML = "<b></b><span class=\"net-tip-meta\"></span>";
    tip.firstChild.textContent = p.title;
    tip.lastChild.textContent = (st.kind === "author"
      ? [p.venue, p.hIndex != null ? `h-index ${p.hIndex}` : null,
         `${p.nPapers || 0} papers`,
         (p.cites || 0).toLocaleString() + " cites here",
         p.year ? `since ${p.year}` : null]
      : [p.authors, p.year, p.venue, (p.cites || 0).toLocaleString() + " cites"]
    ).filter(Boolean).join(" · ");
    tip.style.display = "block";
    moveTip(event);
  };
  const hideTip = () => { if (tipRef.current) tipRef.current.style.display = "none"; };
  const moveTip = (event) => {
    const tip = tipRef.current, wrap = wrapRef.current;
    if (!tip || !wrap || tip.style.display !== "block") return;
    const rect = wrap.getBoundingClientRect();
    const ev = (event && event.original) || event || {};
    const cx = (ev.clientX ?? ev.x ?? 0) - rect.left;
    const cy = (ev.clientY ?? ev.y ?? 0) - rect.top;
    tip.style.left = Math.max(4, Math.min(cx + 14, rect.width - 270)) + "px";
    tip.style.top = (cy + 14) + "px";
  };

  // ---- toolbar ----------------------------------------------------------
  const zoomBy = (f) => {
    if (!st.sigma) return;
    const cam = st.sigma.getCamera();
    cam.animate({ ratio: Math.max(0.05, Math.min(8, cam.ratio / f)) }, { duration: 180 });
  };
  const fitView = () => {
    if (!st.sigma) return;
    st.sigma.getCamera().animate({ x: 0.5, y: 0.5, ratio: 1, angle: 0 }, { duration: 220 });
  };
  // Manual 30°-per-click rotation of the whole field (no auto-spin) — rotates
  // the camera, so node positions / drag / hit-testing all stay correct.
  const rotate30 = () => {
    if (!st.sigma) return;
    const cam = st.sigma.getCamera();
    cam.animate({ angle: cam.angle + Math.PI / 6 }, { duration: 240 });
  };

  // f5's "Simulate live growth" over the active collection: seeds first, then
  // acceptance order (depth -> stream order -> citations), batched to finish
  // in roughly 13s regardless of size.
  const replayGrowth = () => {
    if (!st.sigma || !st.activeData.nodes.length) return;
    if (st.replayTimer) { stopReplay(); return; }
    if (st.onSelect) st.onSelect(null);
    const order = [...st.activeData.nodes].sort((a, b) =>
      (b.seed === true) - (a.seed === true)
      || (a.depth || 0) - (b.depth || 0)
      || (a.addedAt || 0) - (b.addedAt || 0)
      || (b.cites || 0) - (a.cites || 0));
    makeLayout(st.fa2);  // replay animates under the configured dynamics
    st.graph.clear();
    hideTip();
    emitStats();
    setReplaying(true);
    heatLayout();
    const batch = Math.max(1, Math.ceil(order.length / 110));
    let i = 0;
    st.replayTimer = setInterval(() => {
      if (i >= order.length) {
        stopReplay();
        refreshTop3(st.activeData.nodes);
        heatLayout();
        return;
      }
      for (let k = 0; k < batch && i < order.length; k++, i++) {
        addPaperNode(order[i], true);
        addEdgesFor(order[i].id);
      }
      heatLayout();  // keep FA2 hot for the whole replay, not just the first 6s
      emitStats();
    }, 120);
  };

  const empty = libs !== "ready" || !papers.length;
  const filteredOut = !empty && papers.length > 0 && !active.nodes.length;

  // Sigma only emits leaveNode inside the canvas — clear hover state when the
  // pointer leaves the pane entirely (else the tooltip + ring stick around).
  const onPaneLeave = () => {
    if (!st.sigma) return;
    if (st.hovered) {
      st.hovered = null;
      st.anchor = st.selected;
      recomputeConnected();
      if (st.onHover) st.onHover(null);
      st.sigma.refresh();
    }
    hideTip();
  };

  // ---- legend content ---------------------------------------------------
  const legendStops = cgPaletteStops(vis.palette, theme === "dark" ? "dark" : "light");
  const showSeedLegend = legendSeed === "always"
    || (kind === "paper" && papers.some(p => p.seed));
  const domainCaption = kind === "author"
    ? cdomain.caption
    : `${cdomain.min} → ${cdomain.max}`;

  const gfCount = (gf.minDegree > 0 ? 1 : 0) + (gf.minEdgeW > 0 ? 1 : 0) + (gf.largestOnly ? 1 : 0);

  return (
    <div className="network" ref={wrapRef} onMouseLeave={onPaneLeave}>
      <canvas ref={gridRef} className="cg-grid" />
      <div ref={boxRef} className="cg-sigma" />
      <div ref={tipRef} className="net-tip" />

      {empty && (
        <div className="cg-empty">
          {libs === "loading" && <span>Loading graph engine…</span>}
          {libs === "error" && (
            <span>
              The graph engine could not be loaded (CDN unreachable). This
              view needs internet access — check the connection and reload.
            </span>
          )}
          {libs === "ready" && !papers.length && <span>{emptyHint || "No papers yet."}</span>}
        </div>
      )}
      {filteredOut && (
        <div className="cg-empty">
          <span>Every node is filtered out — loosen the filters.</span>
        </div>
      )}

      {libs === "ready" && (
        <div className="cg-tl" data-paltick={palTick}>
          <div className="net-legend">
            {showSeedLegend && (
              <span className="net-legend-item">
                <span className="net-dot seed" />
                <span>Seed</span>
              </span>
            )}
            <span className="net-legend-item">
              <span className="net-ramp">
                {legendStops.map((c, i) => (
                  <span key={i + theme + vis.palette} className="net-ramp-step" style={{ background: c }} />
                ))}
              </span>
              <span>{domainCaption}</span>
            </span>
            {sizeHint && (
              <span className="net-legend-item">
                <span className="net-hint-txt">
                  size ∝ {kind === "author" ? "citations in collection" : "citations"}
                </span>
              </span>
            )}
            <span className="net-legend-item">
              <span className="net-hint-txt">drag · scroll · click</span>
            </span>
          </div>

          {tools.stats && stats && (
            <div className="cg-stats">
              <div className="cg-stats-row"><span>nodes</span><b>{stats.nodes.toLocaleString()}</b></div>
              <div className="cg-stats-row"><span>edges</span><b>{stats.edges.toLocaleString()}</b></div>
              <div className="cg-stats-row"><span>avg degree</span><b>{stats.avgDeg.toFixed(2)}</b></div>
              <div className="cg-stats-row"><span>density</span><b>{stats.density < 0.001 && stats.density > 0 ? stats.density.toExponential(1) : stats.density.toFixed(3)}</b></div>
              <div className="cg-stats-row"><span>components</span><b>{stats.components.toLocaleString()}</b></div>
              <div className="cg-stats-row">
                <span>diameter</span>
                <b>{stats.diameter == null ? "—" : stats.diameter}</b>
              </div>
            </div>
          )}

          {topLeft}
        </div>
      )}

      {!empty && (
        <div className="net-toolbar">
          <button className="btn-icon btn" onClick={() => zoomBy(1 / 1.25)} title="Zoom out"><Icon name="minus" size={11} /></button>
          <span className="pipe-zoom">{zoomPct}%</span>
          <button className="btn-icon btn" onClick={() => zoomBy(1.25)} title="Zoom in"><Icon name="plus" size={11} /></button>
          <button className="btn-icon btn" onClick={fitView} title="Fit to view"><Icon name="maximize-2" size={11} /></button>
          <button className="btn-icon btn" onClick={rotate30} title="Rotate 30°"><Icon name="rotate-cw" size={11} /></button>
          {(tools.layout || tools.replay || tools.layoutOptions || tools.labels) && <span className="cg-toolbar-sep" />}
          {tools.layout && (
            <button className={"btn-icon btn" + (layoutOn ? " is-on" : "")} onClick={toggleLayout}
                    title={layoutOn ? "Pause force layout" : "Resume force layout"}>
              <Icon name={layoutOn ? "pause" : "play"} size={11} />
            </button>
          )}
          {tools.labels && (
            <button className={"btn-icon btn" + (showLabels ? " is-on" : "")}
                    onClick={() => toggleLabels(!showLabels)}
                    title={showLabels ? "Hide labels" : "Show labels"}>
              <Icon name="type" size={11} />
            </button>
          )}
          {tools.layoutOptions && (
            <button className={"btn-icon btn" + (optsOpen ? " is-on" : "")}
                    onClick={() => setOptsOpen(v => !v)} title="Graph settings">
              <Icon name="sliders-horizontal" size={11} />
            </button>
          )}
          {tools.replay && (
            <button className={"btn-icon btn" + (replaying ? " is-on" : "")} onClick={replayGrowth}
                    title={replaying ? "Stop growth replay" : "Replay collection growth"}>
              <Icon name="history" size={11} />
            </button>
          )}
        </div>
      )}

      {optsOpen && !empty && (
        <div className="cg-pop">
          <div className="cg-pop-title">Force layout</div>
          <label className="cg-pop-row">
            <span className="cg-pop-k">Scaling</span>
            <input type="range" min="1" max="40" step="1" value={fa2.scalingRatio}
                   onChange={e => applyFa2({ scalingRatio: +e.target.value })} />
            <span className="cg-pop-v">{fa2.scalingRatio}</span>
          </label>
          <label className="cg-pop-row">
            <span className="cg-pop-k">Gravity</span>
            <input type="range" min="0.1" max="5" step="0.1" value={fa2.gravity}
                   onChange={e => applyFa2({ gravity: +e.target.value })} />
            <span className="cg-pop-v">{fa2.gravity.toFixed(1)}</span>
          </label>
          <label className="cg-pop-row cg-pop-check">
            <input type="checkbox" checked={fa2.strongGravityMode}
                   onChange={e => applyFa2({ strongGravityMode: e.target.checked })} />
            <span>Strong gravity</span>
          </label>
          <label className="cg-pop-row cg-pop-check">
            <input type="checkbox" checked={fa2.outboundAttractionDistribution}
                   onChange={e => applyFa2({ outboundAttractionDistribution: e.target.checked })} />
            <span>Dissuade hubs</span>
          </label>
          <label className="cg-pop-row cg-pop-check">
            <input type="checkbox" checked={fa2.linLogMode}
                   onChange={e => applyFa2({ linLogMode: e.target.checked })} />
            <span>Tight clusters (LinLog)</span>
          </label>
          <label className="cg-pop-row cg-pop-check">
            <input type="checkbox" checked={fa2.adjustSizes}
                   onChange={e => applyFa2({ adjustSizes: e.target.checked })} />
            <span>Prevent node overlap</span>
          </label>
          <label className="cg-pop-row cg-pop-check">
            <input type="checkbox" checked={fa2.edgeWeightInfluence > 0}
                   onChange={e => applyFa2({ edgeWeightInfluence: e.target.checked ? 1 : 0 })} />
            <span>Use edge weights</span>
          </label>

          <div className="cg-pop-title cg-pop-sec">Appearance</div>
          <label className="cg-pop-row"
                 title="Exact min / max node radius, like Gephi's ranking (1 → 100 is a real 100× difference). Set min = max for uniform nodes.">
            <span className="cg-pop-k">Node size</span>
            <span className="cg-pop-pair">
              <input type="number" className="cg-pop-num" min="0.5" max="200" step="1"
                     value={vis.minSize}
                     onChange={e => applyVis({ minSize: e.target.value === "" ? "" : +e.target.value })} />
              <span className="cg-pop-dash">–</span>
              <input type="number" className="cg-pop-num" min="0.5" max="200" step="1"
                     value={vis.maxSize}
                     onChange={e => applyVis({ maxSize: e.target.value === "" ? "" : +e.target.value })} />
            </span>
          </label>
          <label className="cg-pop-row"
                 title={"How " + (kind === "author" ? "in-collection citations" : "citations") + " map onto the size range — Gephi's spline presets. √ / ∛ / Log lift the long tail's small values; Squared / Cubed reserve size for the giants."}>
            <span className="cg-pop-k">Size scale</span>
            <select className="cg-pop-select" value={vis.sizeCurve}
                    onChange={e => applyVis({ sizeCurve: e.target.value })}>
              {CG_CURVE_NAMES.map(([id, name]) => (
                <option key={id} value={id}>{name}</option>
              ))}
            </select>
            <span className="cg-pop-v" />
          </label>
          <label className="cg-pop-row"
                 title={ranges.weighted
                   ? "Edge width is proportional to edge weight, mapped into this min–max range."
                   : "This graph has no varying edge weights — every edge renders at the minimum width."}>
            <span className="cg-pop-k">Edge width</span>
            <span className="cg-pop-pair">
              <input type="number" className="cg-pop-num" min="0.1" max="20" step="0.1"
                     value={vis.edgeMin}
                     onChange={e => applyVis({ edgeMin: e.target.value === "" ? "" : +e.target.value })} />
              <span className="cg-pop-dash">–</span>
              <input type="number" className="cg-pop-num" min="0.1" max="20" step="0.1"
                     value={vis.edgeMax}
                     onChange={e => applyVis({ edgeMax: e.target.value === "" ? "" : +e.target.value })} />
            </span>
          </label>
          {ranges.weighted && (
            <label className="cg-pop-row"
                   title="Transformation from edge weight to width — same presets as the node sizes.">
              <span className="cg-pop-k">Width scale</span>
              <select className="cg-pop-select" value={vis.edgeCurve}
                      onChange={e => applyVis({ edgeCurve: e.target.value })}>
                {CG_CURVE_NAMES.map(([id, name]) => (
                  <option key={id} value={id}>{name}</option>
                ))}
              </select>
              <span className="cg-pop-v" />
            </label>
          )}
          <label className="cg-pop-row">
            <span className="cg-pop-k">Palette</span>
            <select className="cg-pop-select" value={vis.palette}
                    onChange={e => applyVis({ palette: e.target.value })}>
              {CG_PALETTE_NAMES.map(([id, name]) => (
                <option key={id} value={id}>{name}</option>
              ))}
            </select>
            <span className="cg-pop-v" />
          </label>
          <label className="cg-pop-row cg-pop-check">
            <input type="checkbox" checked={showLabels}
                   onChange={e => toggleLabels(e.target.checked)} />
            <span>Show labels</span>
          </label>

          <div className="cg-pop-title cg-pop-sec">
            Graph filters{gfCount ? ` · ${gfCount}` : ""}
          </div>
          <label className="cg-pop-row">
            <span className="cg-pop-k">Min degree</span>
            <input type="range" min="0" max={Math.max(1, ranges.maxDeg)} step="1"
                   value={gf.minDegree}
                   onChange={e => applyGf({ minDegree: +e.target.value })} />
            <span className="cg-pop-v">{gf.minDegree || "off"}</span>
          </label>
          {ranges.weighted && (
            <label className="cg-pop-row" title="Edges below the threshold are removed from the simulation — the layout re-flows">
              <span className="cg-pop-k">Min weight</span>
              <input type="range" min="0" max={ranges.maxW} step={ranges.maxW / 100}
                     value={gf.minEdgeW}
                     onChange={e => applyGf({ minEdgeW: +e.target.value })} />
              <span className="cg-pop-v">{gf.minEdgeW ? gf.minEdgeW.toFixed(2) : "off"}</span>
            </label>
          )}
          <label className="cg-pop-row cg-pop-check">
            <input type="checkbox" checked={gf.largestOnly}
                   onChange={e => applyGf({ largestOnly: e.target.checked })} />
            <span>Largest component only</span>
          </label>

          <button className="btn btn-ghost cg-pop-reset" onClick={resetAll}>
            <Icon name="rotate-ccw" size={11} /> Defaults
          </button>
        </div>
      )}

      {children}
    </div>
  );
}

Object.assign(window, { CiteGraph, cgYearDomain, cgAuthorDomain });
