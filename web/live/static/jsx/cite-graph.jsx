/* eslint-disable */
// Shared citation-graph surface — used by BOTH the Run and Explore pages.
//
// Engine: the design's module_a_network_f5 reference stack (graphology graph
// + ForceAtlas2 web-worker layout + sigma WebGL renderer) with its algorithms:
// spawn-near-neighbour node insertion, 400ms easeOutCubic grow animation,
// continuous FA2 with auto-settle, fade-toward-paper de-emphasis.
//
// Look: the Run network's — camera-synced dot grid, year-ramp fills
// (--cc-year-p-*), cream node strokes, plum seed dot with soft halo,
// plum selection ring, run-style radii. Labels are opt-in (Explore only);
// hovering always shows title · authors · year · venue · cites.
//
// Wrappers own their legends/counters (children overlays) and data shape:
//   papers: [{id,title,authors,year,venue,cites,seed,depth,source,addedAt}]
//   edges:  [{source,target}]  (paper-id pairs)

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

function cgReadPalette() {
  const bg = cgCssVar("--cc-bg", "#f5f3ee");
  const bgRgb = cgColorRgb(bg, [245, 243, 238]);
  const years = {};
  for (let y = 2018; y <= 2025; y++) years[y] = cgCssVar(`--cc-year-p-${y}`, "#8a8a8a");
  return {
    bg, bgRgb, years,
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
  };
}

function cgYearColor(pal, y) {
  const yy = Math.max(2018, Math.min(2025, Number(y) || 2021));
  return pal.years[yy] || "#8a8a8a";
}

// Run-mode radii (snapshots._radius): seeds fixed, satellites log-cites.
function cgRadius(p) {
  if (p.seed) return 8;
  const c = Math.max(0, Number(p.cites) || 0);
  return 3.5 + Math.min(4.5, Math.log10(1 + c) * 1.2);
}
// Border budget added around the fill (ring space + halo/stroke), so the
// visible fill keeps the run-canvas sizes.
function cgNodeSize(p) { return cgRadius(p) + (p.seed ? 5 : 2.5); }

function cgTrunc(s, n) { s = String(s || ""); return s.length > n ? s.slice(0, n - 2) + "…" : s; }
function cgEaseOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

const CG_TRANSPARENT = "#00000000";

// f5 FA2 settings — the shared default; Explore can override via the
// layout-options popover.
const CG_FA2_DEFAULTS = {
  linLogMode: true,
  adjustSizes: true,
  barnesHutOptimize: true,
  scalingRatio: 10,
  gravity: 1,
  slowDown: 10,
  edgeWeightInfluence: 1,
};

function CiteGraph({ papers, edges, dataKey, selectedId, onSelect, onHover,
                     theme, labels = false, hiddenIds = null,
                     tools = {}, emptyHint, onStats, children }) {
  const wrapRef = React.useRef(null);
  const boxRef = React.useRef(null);
  const gridRef = React.useRef(null);
  const tipRef = React.useRef(null);
  const st = React.useRef({
    sigma: null, graph: null, layout: null, rand: cgMulberry32(7),
    pal: null, adj: {}, adjPairs: {}, edgeList: [],
    selected: null, hovered: null, active: null, connected: new Set(),
    hidden: null, top3: new Set(), labels: false, paperById: {},
    dataKey: null, stopTimer: null, animRAF: 0, replayTimer: null,
    fa2: { ...CG_FA2_DEFAULTS },
    onSelect: null, onHover: null, onStats: null,
  }).current;
  st.onSelect = onSelect || null;
  st.onHover = onHover || null;
  st.onStats = onStats || null;
  st.labels = !!labels;

  const [libs, setLibs] = React.useState(() =>
    window.GraphLibs ? (window.GraphLibs.error ? "error" : "ready") : "loading");
  const [zoomPct, setZoomPct] = React.useState(100);
  const [layoutOn, setLayoutOn] = React.useState(false);
  const [replaying, setReplaying] = React.useState(false);
  const [optsOpen, setOptsOpen] = React.useState(false);
  const [fa2, setFa2] = React.useState({ ...CG_FA2_DEFAULTS });

  React.useEffect(() => {
    if (libs !== "loading") return;
    const onReady = () => setLibs(window.GraphLibs && window.GraphLibs.error ? "error" : "ready");
    window.addEventListener("graphlibs", onReady);
    if (window.GraphLibs) onReady();
    return () => window.removeEventListener("graphlibs", onReady);
  }, [libs]);

  const emitStats = () => {
    if (st.onStats && st.graph) {
      st.onStats({ nodes: st.graph.order, edges: st.graph.size });
    }
  };

  // ---- node / edge insertion (f5 spawn-near-neighbour + grow animation) --
  const nodeAttrs = (p, animate) => {
    const size = cgNodeSize(p);
    return {
      label: cgTrunc(p.title, 48),
      size: animate ? 0.001 : size,
      _targetSize: size,
      _animSize: animate ? 0.001 : size,
      _animStart: performance.now(),
      color: p.seed ? st.pal.seed : cgYearColor(st.pal, p.year),
      _seed: !!p.seed,
      _year: p.year,
      haloColor: p.seed ? st.pal.seedHalo : CG_TRANSPARENT,
      strokeColor: p.seed ? st.pal.seedStroke : st.pal.nodeStroke,
      haloSize: p.seed ? 0.22 : 0.0,
      strokeSize: p.seed ? 0.09 : 0.12,
    };
  };

  const addPaperNode = (p, animate) => {
    const g = st.graph;
    if (!g || g.hasNode(p.id)) return;
    let x, y;
    const present = (st.adj[p.id] || []).filter(o => g.hasNode(o));
    if (present.length) {
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
    for (const [a, b] of (st.adjPairs[id] || [])) {
      if (a !== b && g.hasNode(a) && g.hasNode(b) && !g.hasEdge(a, b)) {
        g.addEdge(a, b, { size: 0.8 });
      }
    }
  };

  const rebuildAdjacency = (edgeArr) => {
    st.adj = {}; st.adjPairs = {}; st.edgeList = edgeArr;
    for (const e of edgeArr) {
      if (!e || e.source === e.target) continue;
      (st.adj[e.source] ||= []).push(e.target);
      (st.adj[e.target] ||= []).push(e.source);
      (st.adjPairs[e.source] ||= []).push([e.source, e.target]);
      (st.adjPairs[e.target] ||= []).push([e.source, e.target]);
    }
  };

  const refreshTop3 = (ps) => {
    st.top3 = new Set([...ps].sort((a, b) => (b.cites || 0) - (a.cites || 0))
      .slice(0, 3).map(p => p.id));
  };

  // Layout runs hot after (re)builds, then settles; the toolbar resumes it.
  const heatLayout = () => {
    if (!st.layout) return;
    try { st.layout.start(); } catch (_) {}
    setLayoutOn(true);
    clearTimeout(st.stopTimer);
    const ms = Math.min(20000, 6000 + (st.graph ? st.graph.order * 12 : 0));
    st.stopTimer = setTimeout(() => {
      if (st.layout) { try { st.layout.stop(); } catch (_) {} }
      setLayoutOn(false);
    }, ms);
  };
  const toggleLayout = () => {
    if (!st.layout) return;
    if (st.layout.isRunning()) {
      clearTimeout(st.stopTimer);
      st.layout.stop();
      setLayoutOn(false);
    } else heatLayout();
  };

  const rebuildLayout = (settings) => {
    if (!st.graph || !window.GraphLibs || !window.GraphLibs.FA2Layout) return;
    st.fa2 = { ...st.fa2, ...settings };
    if (st.layout) { try { st.layout.kill(); } catch (_) {} }
    st.layout = new window.GraphLibs.FA2Layout(st.graph, { settings: { ...st.fa2 } });
    heatLayout();
  };

  const stopReplay = () => {
    clearInterval(st.replayTimer);
    st.replayTimer = null;
    setReplaying(false);
  };

  const teardown = () => {
    stopReplay();
    clearTimeout(st.stopTimer);
    cancelAnimationFrame(st.animRAF);
    if (st.layout) { try { st.layout.kill(); } catch (_) {} st.layout = null; }
    if (st.sigma) { try { st.sigma.kill(); } catch (_) {} st.sigma = null; }
    st.graph = null;
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
    st.pal = cgReadPalette();
    rebuildAdjacency(edges);
    refreshTop3(papers);

    const g = new Graph({ multi: false, type: "undirected" });
    st.graph = g;
    for (const p of papers) addPaperNode(p, false);
    for (const p of papers) addEdgesFor(p.id);

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
        if (st.hidden && st.hidden.has(id)) return { ...data, hidden: true };
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
        if (st.active) {
          if (st.connected.has(id)) return { ...base, zIndex: 2 };
          // run-canvas dim: alpha 0.22 against the paper base
          return { ...base,
                   color: cgFade(data.color, 0.22, st.pal.bgRgb),
                   haloColor: CG_TRANSPARENT,
                   strokeColor: cgFade(data.strokeColor, 0.22, st.pal.bgRgb),
                   labelColor: cgFade(st.pal.ink1, 0.25, st.pal.bgRgb),
                   forceLabel: false };
        }
        return base;
      },
      edgeReducer: (edge, data) => {
        const src = st.graph.source(edge), tgt = st.graph.target(edge);
        if (st.hidden && (st.hidden.has(src) || st.hidden.has(tgt))) {
          return { ...data, hidden: true };
        }
        const yc = cgYearColor(st.pal, st.graph.getNodeAttribute(tgt, "_year"));
        const touches = st.active && (src === st.active || tgt === st.active);
        if (touches) return { ...data, color: cgFade(yc, 0.85, st.pal.bgRgb), size: 1.6, zIndex: 1 };
        return { ...data, color: cgFade(yc, 0.32, st.pal.bgRgb), size: 0.8 };
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

    sigma.on("clickNode", ({ node }) => st.onSelect && st.onSelect(node));
    sigma.on("clickStage", () => { if (st.selected) st.onSelect && st.onSelect(null); });
    sigma.on("enterNode", ({ node, event }) => {
      st.hovered = node;
      st.active = st.selected || st.hovered;
      recomputeConnected();
      if (st.onHover) st.onHover(node);
      showTip(node, event);
      sigma.refresh();
    });
    sigma.on("leaveNode", () => {
      st.hovered = null;
      st.active = st.selected;
      recomputeConnected();
      if (st.onHover) st.onHover(null);
      hideTip();
      sigma.refresh();
    });
    sigma.getMouseCaptor().on("mousemovebody", (e) => moveTip(e));
    const cam = sigma.getCamera();
    cam.on("updated", () => { setZoomPct(Math.round(100 / cam.ratio)); drawGrid(); });
    setZoomPct(Math.round(100 / cam.ratio));

    st.layout = new FA2Layout(g, { settings: { ...st.fa2 } });
    heatLayout();
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

  // ---- incremental growth (live run streams into the same dataKey) ------
  React.useEffect(() => {
    if (!st.sigma || st.dataKey !== dataKey || st.replayTimer) return;
    rebuildAdjacency(edges);
    const fresh = papers.filter(p => !st.graph.hasNode(p.id));
    for (const p of fresh) addPaperNode(p, true);
    let newEdges = false;
    for (const e of st.edgeList) {
      if (e.source !== e.target && st.graph.hasNode(e.source) && st.graph.hasNode(e.target)
          && !st.graph.hasEdge(e.source, e.target)) {
        st.graph.addEdge(e.source, e.target, { size: 0.8 });
        newEdges = true;
      }
    }
    if (fresh.length || newEdges) {
      refreshTop3(papers);
      heatLayout();
      emitStats();
    }
  }, [papers, edges]);  // eslint-disable-line

  // ---- selection / hidden set -> reducer inputs -------------------------
  const recomputeConnected = () => {
    st.connected = new Set();
    if (st.active && st.graph && st.graph.hasNode(st.active)) {
      for (const nb of st.graph.neighbors(st.active)) st.connected.add(nb);
    }
  };
  React.useEffect(() => {
    st.selected = selectedId && st.graph && st.graph.hasNode(selectedId) ? selectedId : null;
    st.active = st.selected || st.hovered;
    recomputeConnected();
    if (st.sigma) st.sigma.refresh();
  }, [selectedId, papers]);
  React.useEffect(() => {
    st.hidden = hiddenIds && hiddenIds.size ? hiddenIds : null;
    if (st.sigma) st.sigma.refresh();
  }, [hiddenIds]);

  // ---- theme flip -> re-read tokens, recolour in place ------------------
  React.useEffect(() => {
    if (!st.sigma) return;
    st.pal = cgReadPalette();
    st.graph.forEachNode((id) => {
      const p = st.paperById[id];
      if (!p) return;
      st.graph.mergeNodeAttributes(id, {
        color: p.seed ? st.pal.seed : cgYearColor(st.pal, p.year),
        haloColor: p.seed ? st.pal.seedHalo : CG_TRANSPARENT,
        strokeColor: p.seed ? st.pal.seedStroke : st.pal.nodeStroke,
      });
    });
    st.sigma.setSetting("labelColor", { attribute: "labelColor", color: st.pal.ink1 });
    st.sigma.refresh();
    drawGrid();
  }, [theme]);  // eslint-disable-line

  // ---- tooltip (run skin: title + authors · year · venue · cites) -------
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
    tip.lastChild.textContent = [p.authors, p.year, p.venue,
      (p.cites || 0).toLocaleString() + " cites"].filter(Boolean).join(" · ");
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

  // f5's "Simulate live growth" over the real collection: seeds first, then
  // acceptance order (depth -> stream order -> citations), batched to finish
  // in roughly 13s regardless of size.
  const replayGrowth = () => {
    if (!st.sigma || !papers.length) return;
    if (st.replayTimer) { stopReplay(); return; }
    if (st.onSelect) st.onSelect(null);
    const order = [...papers].sort((a, b) =>
      (b.seed === true) - (a.seed === true)
      || (a.depth || 0) - (b.depth || 0)
      || (a.addedAt || 0) - (b.addedAt || 0)
      || (b.cites || 0) - (a.cites || 0));
    if (st.layout) { try { st.layout.stop(); } catch (_) {} }
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
        refreshTop3(papers);
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

  const applyFa2 = (patch) => {
    const next = { ...fa2, ...patch };
    setFa2(next);
    rebuildLayout(next);
  };

  const empty = libs !== "ready" || !papers.length;

  // Sigma only emits leaveNode inside the canvas — clear hover state when the
  // pointer leaves the pane entirely (else the tooltip + ring stick around).
  const onPaneLeave = () => {
    if (!st.sigma) return;
    if (st.hovered) {
      st.hovered = null;
      st.active = st.selected;
      recomputeConnected();
      if (st.onHover) st.onHover(null);
      st.sigma.refresh();
    }
    hideTip();
  };

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

      {!empty && (
        <div className="net-toolbar">
          <button className="btn-icon btn" onClick={() => zoomBy(1 / 1.25)} title="Zoom out"><Icon name="minus" size={11} /></button>
          <span className="pipe-zoom">{zoomPct}%</span>
          <button className="btn-icon btn" onClick={() => zoomBy(1.25)} title="Zoom in"><Icon name="plus" size={11} /></button>
          <button className="btn-icon btn" onClick={fitView} title="Fit to view"><Icon name="maximize-2" size={11} /></button>
          {(tools.layout || tools.replay || tools.layoutOptions) && <span className="cg-toolbar-sep" />}
          {tools.layout && (
            <button className={"btn-icon btn" + (layoutOn ? " is-on" : "")} onClick={toggleLayout}
                    title={layoutOn ? "Pause force layout" : "Resume force layout"}>
              <Icon name={layoutOn ? "pause" : "play"} size={11} />
            </button>
          )}
          {tools.layoutOptions && (
            <button className={"btn-icon btn" + (optsOpen ? " is-on" : "")}
                    onClick={() => setOptsOpen(v => !v)} title="Layout options">
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
            <span className="cg-pop-k">Spacing</span>
            <input type="range" min="2" max="30" step="1" value={fa2.scalingRatio}
                   onChange={e => applyFa2({ scalingRatio: +e.target.value })} />
            <span className="cg-pop-v">{fa2.scalingRatio}</span>
          </label>
          <label className="cg-pop-row">
            <span className="cg-pop-k">Gravity</span>
            <input type="range" min="0.2" max="5" step="0.2" value={fa2.gravity}
                   onChange={e => applyFa2({ gravity: +e.target.value })} />
            <span className="cg-pop-v">{fa2.gravity.toFixed(1)}</span>
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
          <button className="btn btn-ghost cg-pop-reset"
                  onClick={() => applyFa2({ ...CG_FA2_DEFAULTS })}>
            <Icon name="rotate-ccw" size={11} /> Defaults
          </button>
        </div>
      )}

      {children}
    </div>
  );
}

Object.assign(window, { CiteGraph });
