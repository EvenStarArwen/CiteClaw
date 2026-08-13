// network-viz.js — citation-network visualizer for KnowledgeLab (Runs page).
// ForceAtlas2 (Jacomy et al. 2014) with size-aware overlap prevention
// (off / partial / full), destructive filters (min degree, min edge weight,
// k-core, largest component), year→teal node ramp, weight→ink edge ramp,
// ranked labels, rubber-band node drag, live node arrival. No dependencies.

const XF = {
  linear: t => t,
  sqrt: t => Math.sqrt(t),
  log: t => Math.log(1 + 19 * t) / Math.log(20),
  pow2: t => t * t
};
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const cubicOut = t => 1 - Math.pow(1 - t, 3);
const backOut = t => { const c = 1.70158, u = t - 1; return 1 + (c + 1) * u * u * u + c * u * u; };

const MIX = (c, p, base) => 'color-mix(in oklab, var(--v-' + c + ') ' + p + '%, var(--' + base + '))';
export const PALETTES = {
  'Accent ink': ['color-mix(in oklab, var(--accent) 15%, var(--canvas))', 'color-mix(in oklab, var(--accent) 44%, var(--canvas))', 'color-mix(in oklab, var(--accent) 32%, var(--fg2))', 'var(--fg)'],
  'Ochre → slate': [MIX('gold', 30, 'canvas'), MIX('gold', 88, 'canvas'), MIX('blue', 88, 'canvas'), MIX('indigo', 82, 'fg')],
  'Slate ink': [MIX('blue', 22, 'canvas'), MIX('blue', 70, 'canvas'), 'var(--v-indigo)', MIX('indigo', 45, 'fg')],
  'Teal depth': [MIX('teal', 20, 'canvas'), MIX('teal', 62, 'canvas'), 'var(--v-teal)', MIX('teal', 38, 'fg')],
  'Moss': [MIX('green', 22, 'canvas'), MIX('green', 68, 'canvas'), 'var(--v-green)', MIX('green', 36, 'fg')],
  'Purple dusk': [MIX('purple', 20, 'canvas'), MIX('purple', 64, 'canvas'), 'var(--v-purple)', MIX('indigo', 60, 'fg')],
  'Rose clay': [MIX('rose', 22, 'canvas'), MIX('rose', 68, 'canvas'), 'var(--v-rose)', MIX('rose', 40, 'fg')],
  'Gold ember': [MIX('gold', 24, 'canvas'), MIX('gold', 72, 'canvas'), 'var(--v-gold)', MIX('gold', 38, 'fg')],
  'Neutral ink': ['color-mix(in oklab, var(--fg) 14%, var(--canvas))', 'color-mix(in oklab, var(--fg) 40%, var(--canvas))', 'color-mix(in oklab, var(--fg) 70%, var(--canvas))', 'var(--fg)'],
  'Teal → rose': [MIX('teal', 26, 'canvas'), 'var(--v-teal)', 'var(--v-purple)', MIX('rose', 72, 'fg')],
  'Blue dawn': [MIX('blue', 18, 'canvas'), MIX('blue', 52, 'canvas'), 'var(--v-blue-dot)', 'var(--v-blue)']
};

export class CitationNetwork {
  static PALETTE_DEFAULT = PALETTES['Neutral ink'];
  constructor(o) {
    this.canvas = o.canvas;
    this.host = o.host;
    this.onSelect = o.onSelect || (() => {});
    this.onHover = o.onHover || (() => {});
    this.onHoverMove = o.onHoverMove || (() => {});
    this.anchors = o.anchors || CitationNetwork.PALETTE_DEFAULT;
    this.onStats = o.onStats || (() => {});
    this.fmt = o.fmt || (n => String(n));
    this.nodeAlpha = o.nodeAlpha == null ? .8 : o.nodeAlpha;
    this.opts = { zoom: o.zoom !== false, pan: o.pan !== false };
    this.params = { scaling: 3, gravity: 1, dissuade: false, linlog: false, useWeights: true, overlap: 'partial' };
    this.appearance = { metric: 'cites', sizeScale: 'sqrt', sizeMin: 5, sizeMax: 22, edgeScale: 'linear', stroke: 1.25, edgeMin: 0.6, edgeMax: 2.2 };
    this.filters = { minDeg: 0, minW: 0, kcore: 0, largest: false };
    this.labelScale = 1;
    this.nodes = []; this.edges = []; this.byId = new Map(); this.fullAdj = new Map();
    this.shown = []; this.shownE = []; this.adj = new Map(); this.onSet = new Set();
    this.drawOrder = []; this.leaving = []; this.pulses = [];
    this.sel = null; this.nbrs = new Set(); this.hover = null; this.spot = null; this.spotSet = null;
    this.cam = { x: 0, y: 0, k: 1 };
    this.speed = 1; this.calm = 0; this.hot = true; this.dirty = true;
    this.userCam = false; this.didFit = false; this.visible = true;
    this.w = 0; this.h = 0; this._lw = {};
    this.ctx = this.canvas.getContext('2d');
    this._probe = document.createElement('span');
    this._probe.style.cssText = 'position:absolute;left:-9999px;top:0;visibility:hidden;pointer-events:none;transition:none !important;';
    this.host.appendChild(this._probe);
    this._cc = document.createElement('canvas');
    this._cc.width = 1; this._cc.height = 1;
    this._cctx = this._cc.getContext('2d', { willReadFrequently: true });
    this._down = this._down.bind(this); this._move = this._move.bind(this); this._up = this._up.bind(this);
    this._cancel = this._cancel.bind(this);
    this._wheel = this._wheel.bind(this); this._dbl = this._dbl.bind(this); this._loop = this._loop.bind(this);
    this._ptrs = new Map(); this._pinch = null; this._tap = null;
    // touch: the canvas owns its gestures — without this iPadOS reads a drag as page scroll and cancels it
    this.canvas.style.touchAction = 'none';
    this.canvas.style.webkitUserSelect = 'none'; this.canvas.style.userSelect = 'none';
    this.canvas.style.webkitTouchCallout = 'none';
    this.canvas.addEventListener('pointerdown', this._down);
    this.canvas.addEventListener('pointermove', this._move);
    window.addEventListener('pointerup', this._up);
    window.addEventListener('pointercancel', this._cancel);
    this.canvas.addEventListener('wheel', this._wheel, { passive: false });
    this.canvas.addEventListener('dblclick', this._dbl);
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(this.host);
    this.resize();
    this._raf = requestAnimationFrame(this._loop);
  }

  destroy() {
    cancelAnimationFrame(this._raf);
    this._ro.disconnect();
    this.canvas.removeEventListener('pointerdown', this._down);
    this.canvas.removeEventListener('pointermove', this._move);
    window.removeEventListener('pointerup', this._up);
    window.removeEventListener('pointercancel', this._cancel);
    this.canvas.removeEventListener('wheel', this._wheel);
    this.canvas.removeEventListener('dblclick', this._dbl);
    if (this._probe.parentNode) this._probe.parentNode.removeChild(this._probe);
  }

  // ------------------------------------------------------------- data
  setData(d) {
    this.ghost = null;
    this.nodes = d.nodes.map(n => Object.assign({
      vx: 0, vy: 0, fx: 0, fy: 0, pfx: 0, pfy: 0,
      r: 6, deg: 0, pr: 0, enter: 1, present: false, drag: false, ret: null
    }, n));
    this.byId = new Map(this.nodes.map(n => [n.id, n]));
    const ws = d.edges.map(e => e.w);
    const wmin = Math.min.apply(null, ws), wmax = Math.max.apply(null, ws);
    this.edges = d.edges.map(e => ({ s: e.s, t: e.t, w: e.w, tn: wmax > wmin ? (e.w - wmin) / (wmax - wmin) : .5 }));
    this.fullAdj = new Map();
    this.edges.forEach(e => {
      if (!this.fullAdj.has(e.s)) this.fullAdj.set(e.s, []);
      if (!this.fullAdj.has(e.t)) this.fullAdj.set(e.t, []);
      this.fullAdj.get(e.s).push(e.t);
      this.fullAdj.get(e.t).push(e.s);
    });
    let y0 = Infinity, y1 = -Infinity;
    this.nodes.forEach(n => { y0 = Math.min(y0, n.year); y1 = Math.max(y1, n.year); });
    this._yd = [y0, y1];
    const pc = d.presentCount == null ? this.nodes.length : d.presentCount;
    this.nodes.forEach((n, i) => { n.present = i < pc; });
    this.refreshTheme();
    this.rebuild();
  }

  yearDomain() { return this._yd || [0, 0]; }

  stream(count, opts) {
    opts = opts || {};
    const rest = this.nodes.filter(n => !n.present);
    const batch = rest.slice(0, count === Infinity ? rest.length : count);
    if (!batch.length) return 0;
    batch.forEach(n => this._land(n, opts));
    this.rebuild();
    if (!this.userCam && this.w && !opts.instant) {
      const out = batch.some(n => {
        const x = this.sX(n.x), y = this.sY(n.y);
        return x < 30 || x > this.w - 30 || y < 30 || y > this.h - 30;
      });
      if (out) this.fit(true);
    }
    return batch.length;
  }

  _land(n, opts) {
    n.present = true;
    const nb = (this.fullAdj.get(n.id) || []).map(id => this.byId.get(id)).filter(x => x.present && x !== n);
    if (nb.length) {
      let ax = 0, ay = 0;
      nb.forEach(b => { ax += b.x; ay += b.y; });
      n.x = ax / nb.length + (Math.random() - .5) * 42;
      n.y = ay / nb.length + (Math.random() - .5) * 42;
    }
    n.enter = 0;
    if (!opts.instant) this.pulses.push({ n: n, t0: performance.now() });
  }

  // land ONE not-yet-streamed node right now — picking an accepted paper must
  // always respond on the canvas, even while the run is still streaming nodes in
  present(id) {
    const n = this.byId.get(id);
    if (!n || n.present) return;
    this._land(n, {});
    this.rebuild();
  }

  // -------------------------------------------------- filters → shown graph
  rebuild() {
    const F = this.filters;
    let keep = new Set(this.nodes.filter(n => n.present).map(n => n.id));
    if (this._local) keep = new Set([...keep].filter(i => this._local.has(i)));
    let es = this.edges.filter(e => keep.has(e.s) && keep.has(e.t) && e.tn >= F.minW - 1e-9);
    const degOf = () => {
      const d = new Map();
      keep.forEach(i => d.set(i, 0));
      es.forEach(e => { d.set(e.s, d.get(e.s) + 1); d.set(e.t, d.get(e.t) + 1); });
      return d;
    };
    const cutE = () => { es = es.filter(e => keep.has(e.s) && keep.has(e.t)); };
    if (F.minDeg > 0) { const d = degOf(); keep = new Set([...keep].filter(i => d.get(i) >= F.minDeg)); cutE(); }
    if (F.kcore > 1) {
      for (;;) {
        const d = degOf();
        const nk = [...keep].filter(i => d.get(i) >= F.kcore);
        if (nk.length === keep.size) break;
        keep = new Set(nk); cutE();
        if (!keep.size) break;
      }
    }
    if (F.largest && keep.size) {
      const adj = new Map();
      keep.forEach(i => adj.set(i, []));
      es.forEach(e => { adj.get(e.s).push(e.t); adj.get(e.t).push(e.s); });
      const seen = new Set();
      let best = [];
      keep.forEach(i => {
        if (seen.has(i)) return;
        const q = [i], comp = [];
        seen.add(i);
        while (q.length) {
          const x = q.pop();
          comp.push(x);
          adj.get(x).forEach(y => { if (!seen.has(y)) { seen.add(y); q.push(y); } });
        }
        if (comp.length > best.length) best = comp;
      });
      keep = new Set(best); cutE();
    }
    const prev = this.onSet;
    this.nodes.forEach(n => {
      const on = keep.has(n.id);
      if (on && !prev.has(n.id) && n.enter >= 1) n.enter = 0.35;
      if (!on && prev.has(n.id)) this.leaving.push({ x: n.x, y: n.y, r: n.r, fillS: n.fillS, t0: performance.now() });
    });
    this.onSet = keep;
    this.shown = this.nodes.filter(n => keep.has(n.id));
    this.shownE = es;
    this.adj = new Map();
    this.shown.forEach(n => this.adj.set(n.id, []));
    es.forEach(e => { this.adj.get(e.s).push(e); this.adj.get(e.t).push(e); });
    const dm = new Map();
    this.shown.forEach(n => dm.set(n.id, 0));
    es.forEach(e => { dm.set(e.s, dm.get(e.s) + 1); dm.set(e.t, dm.get(e.t) + 1); });
    this.shown.forEach(n => { n.deg = dm.get(n.id); });
    if (this.sel != null && !keep.has(this.sel)) { this.sel = null; this.nbrs = new Set(); this.onSelect(null); }
    else if (this.sel != null) this.nbrs = this.computeNbrs(this.sel);
    if (this.hover != null && !keep.has(this.hover)) { this.hover = null; this.onHover(null); }
    if (this.spot != null && !keep.has(this.spot)) this.spot = null;
    if (this.spotSet) { this.spotSet = new Set([...this.spotSet].filter(id => keep.has(id))); if (!this.spotSet.size) this.spotSet = null; }
    this.pagerank();
    this._btwOk = false;
    this.computeStyles();
    this.stats();
    if (!this.didFit) this.fit(false);
    this.wake();
  }

  pagerank() {
    const n = this.shown.length;
    if (!n) return;
    const idx = new Map(this.shown.map((x, i) => [x.id, i]));
    const wdeg = new Float64Array(n);
    const L = this.shownE.map(e => {
      const a = idx.get(e.s), b = idx.get(e.t);
      wdeg[a] += e.w; wdeg[b] += e.w;
      return [a, b, e.w];
    });
    let pr = new Float64Array(n).fill(1 / n);
    const d = 0.85;
    for (let it = 0; it < 40; it++) {
      const nx = new Float64Array(n).fill((1 - d) / n);
      for (let k = 0; k < L.length; k++) {
        const a = L[k][0], b = L[k][1], w = L[k][2];
        if (wdeg[a]) nx[b] += d * pr[a] * (w / wdeg[a]);
        if (wdeg[b]) nx[a] += d * pr[b] * (w / wdeg[b]);
      }
      pr = nx;
    }
    this.shown.forEach((x, i) => { x.pr = pr[i]; });
  }

  // Brandes betweenness centrality (unweighted, undirected), computed lazily
  betweenness() {
    const n = this.shown.length;
    if (!n) return;
    const idx = new Map(this.shown.map((x, i) => [x.id, i]));
    const adj = Array.from({ length: n }, () => []);
    this.shownE.forEach(e => { const a = idx.get(e.s), b = idx.get(e.t); adj[a].push(b); adj[b].push(a); });
    const bc = new Float64Array(n);
    const dist = new Int32Array(n), sigma = new Float64Array(n), delta = new Float64Array(n), Q = new Int32Array(n);
    const P = Array.from({ length: n }, () => []);
    for (let s = 0; s < n; s++) {
      dist.fill(-1); sigma.fill(0); delta.fill(0);
      for (let i = 0; i < n; i++) P[i].length = 0;
      dist[s] = 0; sigma[s] = 1;
      let qh = 0, qt = 0; Q[qt++] = s;
      const order = [];
      while (qh < qt) {
        const v = Q[qh++]; order.push(v);
        const nb = adj[v];
        for (let k = 0; k < nb.length; k++) {
          const w = nb[k];
          if (dist[w] < 0) { dist[w] = dist[v] + 1; Q[qt++] = w; }
          if (dist[w] === dist[v] + 1) { sigma[w] += sigma[v]; P[w].push(v); }
        }
      }
      for (let i = order.length - 1; i >= 0; i--) {
        const w = order[i], pw = P[w];
        for (let k = 0; k < pw.length; k++) { const v = pw[k]; delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w]); }
        if (w !== s) bc[w] += delta[w];
      }
    }
    this.shown.forEach((x, i) => { x.btw = bc[i] / 2; });
    this._btwOk = true;
  }

  computeStyles() {
    const A = this.appearance, xf = XF[A.sizeScale] || XF.sqrt;
    if (A.metric === 'betweenness' && !this._btwOk) this.betweenness();
    const val = n => A.metric === 'degree' ? n.deg : A.metric === 'pagerank' ? n.pr : A.metric === 'betweenness' ? (n.btw || 0) : n.cites;
    let mn = Infinity, mx = -Infinity;
    this.shown.forEach(n => { const v = val(n); if (v < mn) mn = v; if (v > mx) mx = v; });
    const lo = Math.min(A.sizeMin, A.sizeMax), hi = Math.max(A.sizeMin, A.sizeMax);
    this.shown.forEach(n => {
      const t = mx > mn ? xf((val(n) - mn) / (mx - mn)) : .5;
      n.rT = lo + t * (hi - lo);
      if (n.r == null || n.enter >= 1) n.r = n.rT;
    });
    const order = this.shown.slice().sort((a, b) => b.rT - a.rT || b.cites - a.cites);
    order.forEach((n, i) => { n.lrank = i; });
    this.drawOrder = order;
    this.restyleEdges();
    this.dirty = true;
  }

  stats() {
    const F = this.filters;
    const act = (F.minDeg > 0 ? 1 : 0) + (F.minW > 0 ? 1 : 0) + (F.kcore > 1 ? 1 : 0) + (F.largest ? 1 : 0);
    let tn = 0, te = 0;
    if (this._local) {
      tn = this._local.size;
      this.edges.forEach(e => { if (this._local.has(e.s) && this._local.has(e.t) && this.byId.get(e.s).present && this.byId.get(e.t).present) te++; });
    } else {
      this.nodes.forEach(n => { if (n.present) tn++; });
      this.edges.forEach(e => { if (this.byId.get(e.s).present && this.byId.get(e.t).present) te++; });
    }
    this.onStats({ sn: this.shown.length, se: this.shownE.length, tn: tn, te: te, activeFilters: act, local: !!this._local });
  }

  // ------------------------------------------------------------- theme
  refreshTheme() {
    const res = c => {
      this._probe.style.color = c;
      const v = getComputedStyle(this._probe).color;
      const x = this._cctx;
      x.fillStyle = '#000';
      x.fillStyle = v;
      x.clearRect(0, 0, 1, 1);
      x.fillRect(0, 0, 1, 1);
      const d = x.getImageData(0, 0, 1, 1).data;
      return [d[0], d[1], d[2]];
    };
    const T = this.T = {};
    T.canvas = res('var(--canvas)');
    T.fg = res('var(--fg)');
    T.muted2 = res('var(--muted2)');
    T.accent = res('var(--accent)');
    T.success = res('var(--success)') || T.accent;
    // year ramp anchors: old → recent, all token-derived (theme/scheme aware)
    const anchors = this.anchors;
    T.ramp = [];
    const seg = anchors.length - 1;
    for (let i = 0; i <= 8; i++) {
      const t = i / 8 * seg, s = Math.min(seg - 1, Math.floor(t)), f = t - s;
      T.ramp.push(res('color-mix(in oklab, ' + anchors[s + 1] + ' ' + (f * 100).toFixed(1) + '%, ' + anchors[s] + ')'));
    }
    T.edgeLo = res('color-mix(in oklab, var(--muted2) 60%, var(--canvas))');
    T.edgeHi = res('color-mix(in oklab, var(--fg) 90%, var(--canvas))');
    const yd = this._yd || [0, 1], y0 = yd[0], y1 = yd[1];
    // categorical override: one CSS colour per node id (community colouring).
    // Resolutions are memoised per colour string — 500 nodes over ~12 colours.
    const G = this.groupColors, gcache = {};
    const gres = s => (gcache[s] || (gcache[s] = res(s)));
    this.nodes.forEach(n => {
      const gc = G ? G[n.id] : null;
      const t = y1 > y0 ? (n.year - y0) / (y1 - y0) : .5;
      const c = gc ? gres(gc) : this.rampAt(t);
      n.fillS = 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',';
      const s = [Math.round(lerp(c[0], T.fg[0], .3)), Math.round(lerp(c[1], T.fg[1], .3)), Math.round(lerp(c[2], T.fg[2], .3))];
      n.strokeS = 'rgba(' + s[0] + ',' + s[1] + ',' + s[2] + ',';
    });
    this.restyleEdges();
    this.dirty = true;
  }

  rampAt(t) {
    const R = this.T.ramp, x = clamp(t, 0, 1) * (R.length - 1), i = Math.floor(x), f = x - i;
    const a = R[i], b = R[Math.min(R.length - 1, i + 1)];
    return [Math.round(lerp(a[0], b[0], f)), Math.round(lerp(a[1], b[1], f)), Math.round(lerp(a[2], b[2], f))];
  }

  rampCSSAt(t) { return this.T ? 'rgb(' + this.rampAt(t).join(',') + ')' : ''; }

  // -------------------------------------------------- neighborhood mode
  isLocal() { return !!this._local; }

  enterLocal(id) {
    if (id == null || !this.onSet.has(id)) return false;
    const ids = new Set([id]);
    (this.adj.get(id) || []).forEach(e => { ids.add(e.s); ids.add(e.t); });
    return this.enterLocalSet(ids, id);
  }

  // same subgraph mode from an arbitrary node set (a community's members)
  enterLocalSet(ids, selId) {
    if (!ids || !ids.size) return false;
    const keep = new Set(Array.from(ids).filter(i => { const n = this.byId.get(i); return n && n.present; }));
    if (!keep.size) return false;
    const id = selId != null && keep.has(selId) ? selId : null;
    if (this._local) this.exitLocal({ silent: true });
    this._mainSnap = {
      pos: new Map(this.nodes.map(n => [n.id, [n.x, n.y]])),
      cam: { x: this.cam.x, y: this.cam.y, k: this.cam.k },
      filters: Object.assign({}, this.filters),
      params: Object.assign({}, this.params),
      appearance: Object.assign({}, this.appearance),
      userCam: this.userCam, sel: id
    };
    this._local = keep;
    this.rebuild();
    // gravity pulls toward the world origin, so a subgraph sitting far from it
    // would migrate as a whole and slide out of view — recenter it first
    let cx = 0, cy = 0;
    this.shown.forEach(n => { cx += n.x; cy += n.y; });
    cx /= (this.shown.length || 1); cy /= (this.shown.length || 1);
    this.shown.forEach(n => { n.x -= cx; n.y -= cy; n.vx = 0; n.vy = 0; n.pfx = 0; n.pfy = 0; n.ret = null; });
    this.cam.x = 0; this.cam.y = 0;
    this.select(id);
    this.userCam = false;
    this._localFit = performance.now() + 3000;
    setTimeout(() => { if (this._local) this.fit(true); }, 60);
    return true;
  }

  exitLocal(opts) {
    opts = opts || {};
    const S = this._mainSnap;
    if (!S) return;
    this._local = null;
    this._localFit = 0;
    this._mainSnap = null;
    this.filters = S.filters;
    this.params = S.params;
    // a subgraph's style is a scratch copy — restore the main appearance wholesale
    if (S.appearance) { this.appearance = S.appearance; this.computeStyles(); }
    this.nodes.forEach(n => {
      const p = S.pos.get(n.id);
      if (p) { n.x = p[0]; n.y = p[1]; n.vx = 0; n.vy = 0; n.ret = null; }
    });
    this.rebuild();
    // restored positions ARE the converged main layout — freeze, don't re-run
    this.hot = false; this.calm = 99; this.speed = 1;
    if (!opts.silent) {
      this.select(S.sel);
      this.userCam = false;
      this.camTween(S.cam, 520);
      this.userCam = S.userCam;
    }
    this.dirty = true;
  }

  legendCSS() {
    return this.T ? 'linear-gradient(90deg,' + this.T.ramp.map(c => 'rgb(' + c.join(',') + ')').join(',') + ')' : '';
  }

  restyleEdges() {
    if (!this.T) return;
    const xf = XF[this.appearance.edgeScale] || XF.linear;
    const lo = this.T.edgeLo, hi = this.T.edgeHi;
    const A = this.appearance;
    const w0 = Math.min(A.edgeMin, A.edgeMax), w1 = Math.max(A.edgeMin, A.edgeMax);
    this.edges.forEach(e => {
      const t = xf(e.tn);
      const c = [Math.round(lerp(lo[0], hi[0], t)), Math.round(lerp(lo[1], hi[1], t)), Math.round(lerp(lo[2], hi[2], t))];
      e.colS = 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',';
      e.a = .26 + .5 * t;
      e.wpx = w0 + (w1 - w0) * t;
    });
    this.dirty = true;
  }

  // ------------------------------------------------------------- setters
  setParams(p) { Object.assign(this.params, p); this.wake(); }
  setAppearance(p) { Object.assign(this.appearance, p); this.computeStyles(); this.wake(); }

  zoomBy(f) {
    const k1 = clamp(this.cam.k * f, .18, 6);
    if (k1 === this.cam.k) return;
    this.cam.k = k1;
    this.userCam = true; this._camT = null; this.dirty = true;
  }
  setFilters(p) { Object.assign(this.filters, p); this.rebuild(); }
  setLabelScale(v) { this.labelScale = v; this.dirty = true; }
  setPalette(anchors) { this.anchors = anchors; this.refreshTheme(); }
  // categorical node colouring (null → back to the year ramp)
  setGroupColors(arr) { this.groupColors = arr && arr.length ? arr : null; this.refreshTheme(); }
  // emphasise one group: members keep full ink, everything else recedes
  setGroupEmph(ids) {
    const s = ids && ids.size ? ids : null;
    if (this._gemph !== s) { this._gemph = s; this.dirty = true; }
  }
  // camera onto a set of node ids (a community's members)
  focusSet(ids, animate) {
    if (!this.w || !ids || !ids.size) return;
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9, n0 = 0;
    this.shown.forEach(n => {
      if (!ids.has(n.id)) return;
      n0++;
      if (n.x - n.r < x0) x0 = n.x - n.r;
      if (n.x + n.r > x1) x1 = n.x + n.r;
      if (n.y - n.r < y0) y0 = n.y - n.r;
      if (n.y + n.r > y1) y1 = n.y + n.r;
    });
    if (!n0) return;
    const pad = 76;
    const k = clamp(Math.min((this.w - pad * 2) / Math.max(120, x1 - x0), (this.h - pad * 2) / Math.max(120, y1 - y0)), .18, 1.9);
    const to = { x: (x0 + x1) / 2, y: (y0 + y1) / 2, k: k };
    this.userCam = false;
    if (animate === false) { this.cam = to; this._camT = null; this.dirty = true; return; }
    this.camTween(to, 560);
  }
  setVisible(v) { this.visible = v; if (v) { this.resize(); this.wake(); } }

  // locate-only highlight (paper-card hover / a card left selected): accent-rings
  // ONE node, no neighbour or edge treatment
  spotlight(id) { if (this.spot !== id) { this.spot = id; this.dirty = true; } }
  // trace ONE link out of the selection: the selected node stays selected, but only
  // this counterpart node and the edge between them keep the accent treatment —
  // every other neighbour recedes with the rest of the graph. (Citing-passage hover.)
  setTrace(id) {
    const v = id == null ? null : id;
    if (this.traceId === v) return;
    this.traceId = v;
    this._trcSet = v == null ? null : new Set([v]);
    this.dirty = true;
  }
  // group mark (a group card left active after backing out of its drill): accent-rings
  // EVERY member node — rings only, no neighbour/edge/dim treatment
  spotlightSet(ids) { this.spotSet = ids && ids.length ? new Set(ids) : null; this.dirty = true; }
  // suggestion preview (Runs · Refine · Add papers · Suggested): a candidate paper
  // that is NOT in the corpus, shown as a dashed ghost node anchored to the accepted
  // papers it cites. g = { links: [nodeIds], label } or null. Success ink (the
  // vocabulary for additions), dashed = not in the corpus yet; accent stays
  // selection-only. Cleared by setData.
  setGhost(g) { this.ghost = g && g.links && g.links.length ? g : null; this.dirty = true; }

  camGet() { return { x: this.cam.x, y: this.cam.y, k: this.cam.k }; }
  camGoTo(c, dur) { if (!c) return; this._camT = null; this.userCam = false; this.camTween({ x: c.x, y: c.y, k: c.k }, dur == null ? 520 : dur); }

  select(id, opts) {
    opts = opts || {};
    if (id !== this.sel) this.setTrace(null);
    if (id != null && !this.onSet.has(id)) {
      this.present(id); // mid-stream pick: land the node instead of silently ignoring it
      if (!this.onSet.has(id)) id = null; // still hidden → a canvas filter excludes it
    }
    this.sel = id;
    this.nbrs = id != null ? this.computeNbrs(id) : new Set();
    this.dirty = true;
    if (opts.focus && id != null) {
      const n = this.byId.get(id);
      const sx = (n.x - this.cam.x) * this.cam.k + this.w / 2;
      const sy = (n.y - this.cam.y) * this.cam.k + this.h / 2;
      const m = 70;
      if (sx < m || sx > this.w - m || sy < m || sy > this.h - m || this.cam.k < .5)
        this.camTween({ x: n.x, y: n.y, k: Math.max(this.cam.k, .85) }, 480);
    }
  }

  computeNbrs(id) {
    const s = new Set();
    (this.adj.get(id) || []).forEach(e => s.add(e.s === id ? e.t : e.s));
    return s;
  }

  // ------------------------------------------------------------- camera
  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.w = w; this.h = h;
    // effective on-screen scale: the DC preview scales the page with a CSS
    // transform, so backing-store px must cover dpr × that scale or the
    // canvas (esp. labels) renders soft on hidpi monitors
    const rect = this.host.getBoundingClientRect();
    const cssScale = rect.width ? rect.width / w : 1;
    this._scale = cssScale;
    const d = this.dpr = clamp((window.devicePixelRatio || 1) * cssScale, 1, 3);
    const pw = Math.round(w * d), ph = Math.round(h * d);
    if (this.canvas.width !== pw || this.canvas.height !== ph) { this.canvas.width = pw; this.canvas.height = ph; }
    this.dirty = true;
    if (!this.didFit && this.shown.length) this.fit(false);
  }

  fit(animate) {
    if (!this.w || !this.shown.length) return;
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    this.shown.forEach(n => {
      if (n.x - n.r < x0) x0 = n.x - n.r;
      if (n.x + n.r > x1) x1 = n.x + n.r;
      if (n.y - n.r < y0) y0 = n.y - n.r;
      if (n.y + n.r > y1) y1 = n.y + n.r;
    });
    const pad = 48;
    const k = clamp(Math.min((this.w - pad * 2) / Math.max(80, x1 - x0), (this.h - pad * 2) / Math.max(80, y1 - y0)), .18, 2.4);
    const to = { x: (x0 + x1) / 2, y: (y0 + y1) / 2, k: k };
    this.didFit = true;
    if (!animate) { this.cam = to; this._camT = null; this.dirty = true; return; }
    this.camTween(to, 520);
  }

  camTween(to, dur) { this._camT = { from: { x: this.cam.x, y: this.cam.y, k: this.cam.k }, to: to, t0: performance.now(), dur: dur }; }

  wake() { this.hot = true; this.calm = 0; this.speed = Math.max(this.speed, 1); this.dirty = true; }

  // ------------------------------------------------------------- physics
  step() {
    const P = this.params, ns = this.shown, m = ns.length;
    if (m < 2) { this.calm = 99; return; }
    const kr = P.scaling, kg = P.gravity * 0.85;
    const full = P.overlap === 'full', part = P.overlap === 'partial';
    const ovF = full ? 1 : .62, ovPad = full ? 1.5 : 0;
    for (let i = 0; i < m; i++) { const n = ns[i]; n.fx = 0; n.fy = 0; }
    // repulsion (degree-weighted; size-aware when overlap prevention is on)
    for (let i = 0; i < m; i++) {
      const a = ns[i], ma = a.deg + 1;
      for (let j = i + 1; j < m; j++) {
        const b = ns[j], mb = b.deg + 1;
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < .01) { dx = (Math.random() - .5) * .2; dy = (Math.random() - .5) * .2; d2 = dx * dx + dy * dy; }
        const d = Math.sqrt(d2);
        let f;
        if (full || part) {
          const bd = d - (a.r + b.r) * ovF - ovPad;
          f = bd > 0 ? kr * ma * mb / (bd * d) : kr * 80 * ma * mb / d;
        } else {
          f = kr * ma * mb / d2;
        }
        a.fx += dx * f; a.fy += dy * f;
        b.fx -= dx * f; b.fy -= dy * f;
      }
    }
    // attraction along edges
    const es = this.shownE;
    for (let k = 0; k < es.length; k++) {
      const e = es[k], a = this.byId.get(e.s), b = this.byId.get(e.t);
      const dx = a.x - b.x, dy = a.y - b.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1e-4;
      if ((full || part) && d < (a.r + b.r) * ovF + ovPad) continue;
      let f = P.useWeights ? e.w : 1;
      if (P.linlog) f *= Math.log(1 + d) / d;
      if (P.dissuade) f /= (a.deg + 1);
      a.fx -= dx * f; a.fy -= dy * f;
      b.fx += dx * f; b.fy += dy * f;
    }
    // gravity
    for (let i = 0; i < m; i++) {
      const n = ns[i];
      const d = Math.sqrt(n.x * n.x + n.y * n.y) || 1e-4;
      const f = kg * (n.deg + 1) / d;
      n.fx -= n.x * f; n.fy -= n.y * f;
    }
    // adaptive speed (FA2 swinging / traction)
    let sw = 0, tr = 0;
    for (let i = 0; i < m; i++) {
      const n = ns[i], mass = n.deg + 1;
      n._sw = Math.sqrt((n.fx - n.pfx) * (n.fx - n.pfx) + (n.fy - n.pfy) * (n.fy - n.pfy));
      sw += mass * n._sw;
      tr += mass * Math.sqrt((n.fx + n.pfx) * (n.fx + n.pfx) + (n.fy + n.pfy) * (n.fy + n.pfy)) / 2;
    }
    const jt = clamp(.05 * Math.sqrt(m), .1, 1.1);
    const target = jt * jt * tr / Math.max(sw, 1e-6);
    this.speed = Math.min(target, this.speed * 1.5) || 1;
    let disp = 0;
    for (let i = 0; i < m; i++) {
      const n = ns[i];
      n.pfx = n.fx; n.pfy = n.fy;
      if (n.drag || n.ret) continue;
      let factor = .1 * this.speed / (1 + this.speed * Math.sqrt(n._sw));
      const fl = Math.sqrt(n.fx * n.fx + n.fy * n.fy);
      const maxD = 12;
      if (fl * factor > maxD) factor = maxD / fl;
      n.x += n.fx * factor;
      n.y += n.fy * factor;
      disp += Math.min(maxD, fl * factor);
    }
    disp /= m;
    if (disp < .015 * (kr + 2)) this.calm++; else this.calm = 0;
    if (this.calm > 50) this.hot = false;
  }

  collide(iter) {
    const P = this.params;
    if (P.overlap === 'off') return;
    const full = P.overlap === 'full';
    const fac = full ? 1 : .62, str = full ? .85 : .3, pad = full ? 1.2 : 0;
    const ns = this.shown, m = ns.length;
    for (let it = 0; it < iter; it++) {
      for (let i = 0; i < m; i++) {
        const a = ns[i];
        for (let j = i + 1; j < m; j++) {
          const b = ns[j];
          let dx = b.x - a.x, dy = b.y - a.y;
          let d = Math.sqrt(dx * dx + dy * dy);
          const min = (a.r + b.r) * fac + pad;
          if (d >= min) continue;
          if (d < .01) { dx = Math.random() - .5; dy = Math.random() - .5; d = Math.sqrt(dx * dx + dy * dy); }
          const push = (min - d) / d * str;
          const wa = b.r * b.r / (a.r * a.r + b.r * b.r), wb = 1 - wa;
          if (!a.drag && !a.ret) { a.x -= dx * push * wa; a.y -= dy * push * wa; }
          if (!b.drag && !b.ret) { b.x += dx * push * wb; b.y += dy * push * wb; }
        }
      }
    }
  }

  // ------------------------------------------------------------- loop
  _loop(ts) {
    this._raf = requestAnimationFrame(this._loop);
    if (!this.visible || !this.w) return;
    if (!this._scaleT || ts - this._scaleT > 800) {
      this._scaleT = ts;
      const r = this.host.getBoundingClientRect();
      const s = r.width ? r.width / this.host.clientWidth : 1;
      if (Math.abs(s - (this._scale || 1)) > .04) this.resize();
    }
    if (this.hot) {
      const t0 = performance.now();
      let it = 0;
      do { this.step(); it++; } while (this.hot && it < 3 && performance.now() - t0 < 3.5);
      this.collide(1);
      this.dirty = true;
      // keep a settling neighborhood framed until it calms (or the user pans)
      if (this._local && this._localFit && !this.userCam) {
        if (ts > this._localFit) this._localFit = 0;
        else if (!this._localFitT || ts - this._localFitT > 260) { this._localFitT = ts; this.fit(true); }
      }
    } else if (this._local && this._localFit) {
      this._localFit = 0;
      if (!this.userCam) this.fit(true);
    }
    // rubber-band returns
    let anyRet = false;
    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i];
      if (!n.ret) continue;
      anyRet = true;
      const ax = (n.ret.x - n.x) * .03 - n.vx * .12;
      const ay = (n.ret.y - n.y) * .03 - n.vy * .12;
      n.vx += ax; n.vy += ay;
      n.x += n.vx; n.y += n.vy;
      if (Math.abs(n.ret.x - n.x) < .4 && Math.abs(n.ret.y - n.y) < .4 && Math.abs(n.vx) + Math.abs(n.vy) < .12) {
        n.x = n.ret.x; n.y = n.ret.y; n.ret = null; n.vx = 0; n.vy = 0;
      }
      this.dirty = true;
    }
    if (anyRet) this.wake();
    // enter animations
    for (let i = 0; i < this.shown.length; i++) {
      const n = this.shown[i];
      if (n.enter < 1) { n.enter = Math.min(1, n.enter + .05); n.r = n.rT; this.dirty = true; }
      else if (n.r !== n.rT) { n.r += (n.rT - n.r) * .2; if (Math.abs(n.r - n.rT) < .05) n.r = n.rT; this.dirty = true; }
      if (n.enter < 1 && this.calm > 90) this.dirty = true;
    }
    // camera tween
    if (this._camT) {
      const c = this._camT, t = Math.min(1, (ts - c.t0) / c.dur), e = cubicOut(t);
      this.cam.x = lerp(c.from.x, c.to.x, e);
      this.cam.y = lerp(c.from.y, c.to.y, e);
      this.cam.k = lerp(c.from.k, c.to.k, e);
      if (t >= 1) this._camT = null;
      this.dirty = true;
    }
    if (this.pulses.length || this.leaving.length) this.dirty = true;
    if (this.dirty) { this.draw(); this.dirty = false; }
  }

  // ------------------------------------------------------------- render
  sX(x) { return (x - this.cam.x) * this.cam.k + this.w / 2; }
  sY(y) { return (y - this.cam.y) * this.cam.k + this.h / 2; }

  draw() {
    const ctx = this.ctx, k = this.cam.k, now = performance.now();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    const selOn = this.sel != null;
    const selN = selOn ? this.byId.get(this.sel) : null;
    const trc = selOn && this.traceId != null && this.traceId !== this.sel && this.onSet.has(this.traceId) ? this.traceId : null;
    const NB = trc != null ? this._trcSet : this.nbrs;
    // edges (incident ones are deferred to an overlay pass ON TOP of nodes)
    for (let i = 0; i < this.shownE.length; i++) {
      const e = this.shownE[i];
      const a = this.byId.get(e.s), b = this.byId.get(e.t);
      const inc = selOn && (e.s === this.sel || e.t === this.sel);
      if (inc && (trc == null || e.s === trc || e.t === trc)) continue;
      const between = selOn && NB.has(e.s) && NB.has(e.t);
      let alpha = e.a, col = e.colS, wpx = e.wpx;
      if (selOn) alpha *= between ? .5 : .07;
      const ea = Math.min(a.enter, b.enter);
      if (ea < 1) alpha *= Math.max(0, ea * 1.6 - .2);
      if (alpha <= .01) continue;
      ctx.strokeStyle = col + alpha + ')';
      ctx.lineWidth = Math.max(.35, wpx); // screen-fixed: stays hairline-thin when zoomed in
      ctx.beginPath();
      ctx.moveTo(this.sX(a.x), this.sY(a.y));
      ctx.lineTo(this.sX(b.x), this.sY(b.y));
      ctx.stroke();
    }
    // leaving ghosts
    for (let i = this.leaving.length - 1; i >= 0; i--) {
      const g = this.leaving[i], t = (now - g.t0) / 240;
      if (t >= 1) { this.leaving.splice(i, 1); continue; }
      ctx.fillStyle = g.fillS + (.6 * (1 - t)) + ')';
      ctx.beginPath();
      ctx.arc(this.sX(g.x), this.sY(g.y), Math.max(.1, g.r * (1 - .35 * t) * k), 0, 7);
      ctx.fill();
    }
    // nodes: dimmed rest → neighbors → selected, so related never hides behind noise
    const strokeW = this.appearance.stroke;
    const acc = 'rgba(' + this.T.accent.join(',') + ',';
    const cvs = 'rgba(' + this.T.canvas.join(',') + ',';
    const ink = 'rgba(' + this.T.fg.join(',') + ',';
    const hovId = this.hover;
    const hovN = hovId != null ? this.byId.get(hovId) : null;
    const spotN = this.spot != null && this.spot !== this.sel ? this.byId.get(this.spot) : null;
    let order = this.drawOrder;
    if (selOn) {
      const rest = [], nb = [];
      for (let i = 0; i < order.length; i++) {
        const n = order[i];
        if (n.id === this.sel) continue;
        (NB.has(n.id) ? nb : rest).push(n);
      }
      order = rest.concat(nb, selN ? [selN] : []);
    }
    if (spotN) order = order.filter(n => n !== spotN).concat([spotN]);
    if (hovN && hovN !== selN) order = order.filter(n => n !== hovN).concat([hovN]); // hovered node paints last
    for (let i = 0; i < order.length; i++) {
      const n = order[i];
      const isSel = n.id === this.sel;
      const isNbr = selOn && NB.has(n.id);
      const gdim = this._gemph && !this._gemph.has(n.id);
      const dim = (selOn && !isSel && !isNbr) || gdim;
      const isHov = n === hovN;
      const isSpot = n === spotN;
      const isMark = !isSpot && !isSel && this.spotSet != null && this.spotSet.has(n.id);
      const er = n.enter < 1 ? backOut(n.enter) : 1;
      const sr = Math.max(.2, n.r * er * k) * (isHov ? 1.22 : isSpot ? 1.14 : 1);
      const ea = n.enter < 1 ? Math.min(1, n.enter * 1.8) : 1;
      let fa = this.nodeAlpha * ea, sa = .95 * ea;
      if (dim && !isHov && !isSpot) { const f = gdim && !selOn ? [.2, .14] : [.1, .07]; fa *= f[0]; sa *= f[1]; }
      if (isNbr) fa = Math.max(.92, this.nodeAlpha) * ea;
      if (isHov || isSpot || isMark) { fa = Math.max(.96, this.nodeAlpha) * ea; sa = .95 * ea; }
      const x = this.sX(n.x), y = this.sY(n.y);
      if (x < -sr - 60 || x > this.w + sr + 60 || y < -sr - 60 || y > this.h + sr + 60) continue;
      if (isSel || isNbr || isHov || isSpot) { // canvas casing lifts related nodes off the dim cluster
        ctx.beginPath();
        ctx.arc(x, y, sr + ((isHov || isSpot) && !isSel ? 3.4 : 2.2), 0, 7);
        ctx.fillStyle = cvs + '.92)';
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(x, y, sr, 0, 7);
      ctx.fillStyle = isSel ? acc + (.92 * ea) + ')' : n.fillS + fa + ')';
      ctx.fill();
      if (isSpot && !isSel) { // located node: accent ring + halo, nothing else touched
        ctx.lineWidth = 2.2;
        ctx.strokeStyle = acc + '.92)';
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, sr + 7, 0, 7);
        ctx.lineWidth = 5.5;
        ctx.strokeStyle = acc + '.18)';
        ctx.stroke();
      } else if (isMark) { // group-mark member: plain accent ring (no halo — many nodes carry it)
        ctx.lineWidth = 2;
        ctx.strokeStyle = acc + (.9 * ea) + ')';
        ctx.stroke();
      } else if (isHov && !isSel) { // hover = neutral ink ring + soft halo (accent stays selection-only)
        ctx.lineWidth = 2.2;
        ctx.strokeStyle = ink + '.92)';
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, sr + 7, 0, 7);
        ctx.lineWidth = 5.5;
        ctx.strokeStyle = ink + '.16)';
        ctx.stroke();
      } else if (isNbr) {
        ctx.lineWidth = 2;
        ctx.strokeStyle = acc + '.85)';
        ctx.stroke();
      } else if (strokeW > 0 && !isSel) {
        ctx.lineWidth = strokeW * Math.min(k, 1.6);
        ctx.strokeStyle = n.strokeS + sa + ')';
        ctx.stroke();
      }
      if (isSel) {
        ctx.beginPath();
        ctx.arc(x, y, sr + 3.4, 0, 7);
        ctx.lineWidth = 2.4;
        ctx.strokeStyle = acc + '.95)';
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, sr + 7, 0, 7);
        ctx.lineWidth = 6;
        ctx.strokeStyle = acc + '.16)';
        ctx.stroke();
      }
    }
    // incident edges ON TOP: casing + accent line, so links stay traceable in clusters
    if (selOn && selN) {
      const inc = (this.adj.get(this.sel) || []).filter(e => trc == null || e.s === trc || e.t === trc);
      const sx = this.sX(selN.x), sy = this.sY(selN.y);
      for (let pass = 0; pass < 2; pass++) {
        ctx.strokeStyle = pass === 0 ? cvs + '.75)' : acc + '.95)';
        for (let i = 0; i < inc.length; i++) {
          const e = inc[i], o = this.byId.get(e.s === this.sel ? e.t : e.s);
          const w = Math.max(trc == null ? 1.1 : 2, e.wpx * (trc == null ? .85 : 1.6));
          ctx.lineWidth = pass === 0 ? w + 1.8 : w;
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(this.sX(o.x), this.sY(o.y));
          ctx.stroke();
        }
      }
    }
    if (hovN && !selOn) { // hovered node's links, cased so they read in dense clusters
      const inc = this.adj.get(hovId) || [];
      const hx = this.sX(hovN.x), hy = this.sY(hovN.y);
      for (let pass = 0; pass < 2; pass++) {
        ctx.strokeStyle = pass === 0 ? cvs + '.7)' : ink + '.42)';
        for (let i = 0; i < inc.length; i++) {
          const e = inc[i], o = this.byId.get(e.s === hovId ? e.t : e.s);
          if (!o || !this.onSet.has(o.id)) continue;
          const w = Math.max(.9, e.wpx * .8);
          ctx.lineWidth = pass === 0 ? w + 1.7 : w;
          ctx.beginPath();
          ctx.moveTo(hx, hy);
          ctx.lineTo(this.sX(o.x), this.sY(o.y));
          ctx.stroke();
        }
      }
    }
    // arrival pulses
    for (let i = this.pulses.length - 1; i >= 0; i--) {
      const p = this.pulses[i], t = (now - p.t0) / 950;
      if (t >= 1 || !this.onSet.has(p.n.id)) { this.pulses.splice(i, 1); continue; }
      ctx.beginPath();
      ctx.arc(this.sX(p.n.x), this.sY(p.n.y), (p.n.r + 3 + t * 30) * k, 0, 7);
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = 'rgba(' + (this.T.success || this.T.accent).join(',') + ',' + (.55 * (1 - t)) + ')'; // newly landed = success green (accepted), accent stays selection-only
      ctx.stroke();
    }
    this.drawLabels(ctx, k, selOn);
    // suggestion-preview ghost: dashed success node + dashed links to the accepted
    // papers it cites, drawn on top of everything (it is the current interaction)
    if (this.ghost && this.T) {
      const G = this.ghost, suc = 'rgba(' + (this.T.success || this.T.accent).join(',') + ',';
      let gx0 = 0, gy0 = 0, gc = 0;
      const tg = [];
      for (let i = 0; i < G.links.length; i++) {
        const n = this.byId.get(G.links[i]);
        if (!n || !n.present || !this.onSet.has(n.id)) continue;
        tg.push(n); gx0 += n.x; gy0 += n.y; gc++;
      }
      if (gc) {
        const gx = this.sX(gx0 / gc), gy = this.sY(gy0 / gc), gr = 9;
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = suc + '.38)';
        ctx.lineWidth = 1;
        for (let i = 0; i < tg.length; i++) {
          ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(this.sX(tg[i].x), this.sY(tg[i].y)); ctx.stroke();
        }
        ctx.beginPath(); ctx.arc(gx, gy, gr, 0, 7);
        ctx.fillStyle = 'rgba(' + this.T.canvas.join(',') + ',.92)'; ctx.fill();
        ctx.fillStyle = suc + '.14)'; ctx.fill();
        ctx.lineWidth = 1.5; ctx.strokeStyle = suc + '.92)'; ctx.stroke();
        ctx.setLineDash([]);
        ctx.lineWidth = 1.7; ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(gx - 3.4, gy); ctx.lineTo(gx + 3.4, gy);
        ctx.moveTo(gx, gy - 3.4); ctx.lineTo(gx, gy + 3.4);
        ctx.stroke();
        if (G.label) {
          ctx.font = '600 11px "Hanken Grotesk",sans-serif';
          ctx.textAlign = 'center'; ctx.textBaseline = 'top';
          ctx.lineWidth = 4; ctx.lineJoin = 'round';
          ctx.strokeStyle = 'rgba(' + this.T.canvas.join(',') + ',.85)';
          ctx.strokeText(G.label, gx, gy + gr + 4);
          ctx.fillStyle = suc + '.98)';
          ctx.fillText(G.label, gx, gy + gr + 4);
        }
        ctx.restore();
      }
    }
    if (this.hover != null) {
      const hn = this.byId.get(this.hover);
      if (hn) this.onHoverMove({ x: this.sX(hn.x), y: this.sY(hn.y), r: Math.max(5, hn.r * k) });
    }
  }

  drawLabels(ctx, k, selOn) {
    const halo = 'rgba(' + this.T.canvas.join(',') + ',.82)';
    const ink = 'rgba(' + this.T.fg.join(',') + ',.92)';
    const inkDim = 'rgba(' + this.T.fg.join(',') + ',.5)';
    const cands = [];
    for (let i = 0; i < this.drawOrder.length; i++) {
      const n = this.drawOrder[i];
      const NBL = this.traceId != null && this._trcSet ? this._trcSet : this.nbrs;
      const related = n.id === this.sel || NBL.has(n.id) || n.id === this.hover || n.id === this.spot;
      if (selOn && !related) continue;
      const fw = clamp(n.r * .58, 6.5, 26) * this.labelScale;
      const fs = fw * k;
      const sr = n.r * k;
      if (fs < 6.75 && !related) continue;
      if (!related && n.lrank >= 50 && sr < 12.5) continue;
      if (fs < 5.5) continue;
      cands.push({ n: n, fs: Math.min(fs, 30) * (n.id === this.hover || n.id === this.spot ? 1.22 : 1), pri: related ? -1 : n.lrank });
    }
    cands.sort((a, b) => a.pri - b.pri);
    const rects = [];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    const clash = r => {
      for (let j = 0; j < rects.length; j++) {
        const o = rects[j];
        if (r.x0 < o.x1 && r.x1 > o.x0 && r.y0 < o.y1 && r.y1 > o.y0) return true;
      }
      return false;
    };
    for (let i = 0; i < cands.length && i < 90; i++) {
      const c = cands[i], n = c.n;
      const rel = c.pri < 0;
      const font = (rel ? '600 ' : '500 ') + c.fs.toFixed(1) + 'px "Hanken Grotesk", sans-serif';
      const key = n.id + '|' + c.fs.toFixed(1) + (rel ? 'b' : '');
      let w = this._lw[key];
      if (w == null) { ctx.font = font; w = ctx.measureText(n.label).width; this._lw[key] = w; }
      const cx = this.sX(n.x), cy = this.sY(n.y), rr = n.r * k;
      // related labels get fallback placements so neighbour names don't pile up
      const spots = rel
        ? [[0, -(rr + 5)], [0, rr + 5 + c.fs], [rr + 7 + w / 2, c.fs * .34], [-(rr + 7 + w / 2), c.fs * .34], [0, -(rr + 7 + c.fs * 1.5)]]
        : [[0, -(rr + 4)]];
      let x = cx, y = cy - rr - 4, r = null;
      for (let s = 0; s < spots.length; s++) {
        const tx = cx + spots[s][0], ty = cy + spots[s][1];
        const cand = { x0: tx - w / 2 - 2, x1: tx + w / 2 + 2, y0: ty - c.fs - 2, y1: ty + 2 };
        if (!clash(cand) || s === spots.length - 1) { x = tx; y = ty; r = cand; break; }
      }
      if (!r) continue;
      if (!rel && clash(r)) continue;
      rects.push(r);
      ctx.font = font;
      ctx.lineWidth = Math.max(rel ? 3.4 : 2.5, c.fs / (rel ? 3 : 4));
      ctx.strokeStyle = halo;
      ctx.strokeText(n.label, x, y);
      ctx.fillStyle = rel ? ink : (selOn ? inkDim : ink);
      ctx.fillText(n.label, x, y);
    }
    if (Object.keys(this._lw).length > 4000) this._lw = {};
  }

  // ------------------------------------------------------------- pointer
  _hit(mx, my) {
    // small nodes drawn last are on top → test ascending size
    for (let i = this.drawOrder.length - 1; i >= 0; i--) {
      const n = this.drawOrder[i];
      const dx = mx - this.sX(n.x), dy = my - this.sY(n.y);
      const r = Math.max(5, n.r * this.cam.k) + 3;
      if (dx * dx + dy * dy <= r * r) return n;
    }
    return null;
  }

  _pos(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  _down(e) {
    if (e.button !== 0) return;
    this._ptrs.set(e.pointerId, this._pos(e));
    this.canvas.setPointerCapture && this.canvas.setPointerCapture(e.pointerId);
    if (this._ptrs.size >= 2) { this._pinchStart(); return; }
    const p = this._pos(e), n = this._hit(p[0], p[1]);
    if (n) {
      this._dragN = { n: n, x0: n.x, y0: n.y, px: p[0], py: p[1], moved: false };
    } else {
      this._pan = { cx: this.cam.x, cy: this.cam.y, px: p[0], py: p[1], moved: false };
    }
  }

  _move(e) {
    const p = this._pos(e);
    if (this._ptrs.has(e.pointerId)) this._ptrs.set(e.pointerId, p);
    if (this._pinch) { this._pinchMove(); return; }
    if (this._dragN) {
      const d = this._dragN;
      if (!d.moved && Math.abs(p[0] - d.px) + Math.abs(p[1] - d.py) > 4) {
        d.moved = true;
        d.n.drag = true; d.n.ret = null; d.n.vx = 0; d.n.vy = 0;
        if (this.hover != null) { this.hover = null; this.onHover(null); }
        this.canvas.style.cursor = 'grabbing';
      }
      if (d.moved) {
        d.n.x = (p[0] - this.w / 2) / this.cam.k + this.cam.x;
        d.n.y = (p[1] - this.h / 2) / this.cam.k + this.cam.y;
        this.wake();
      }
      return;
    }
    if (this._pan) {
      if (!this.opts.pan) return;
      const pn = this._pan;
      if (!pn.moved && Math.abs(p[0] - pn.px) + Math.abs(p[1] - pn.py) > 3) pn.moved = true;
      if (pn.moved) {
        this.cam.x = pn.cx - (p[0] - pn.px) / this.cam.k;
        this.cam.y = pn.cy - (p[1] - pn.py) / this.cam.k;
        if (this.hover != null) { this.hover = null; this.onHover(null); }
        this.userCam = true; this._camT = null; this.dirty = true;
      }
      return;
    }
    const n = this._hit(p[0], p[1]);
    const id = n ? n.id : null;
    if (id !== this.hover) { this.hover = id; this.dirty = true; this.onHover(id); }
    this.canvas.style.cursor = n ? 'pointer' : (this.opts.pan ? 'grab' : 'default');
  }

  _up(e) {
    this._ptrs.delete(e.pointerId);
    if (this._pinch) { if (this._ptrs.size < 2) this._endPinch(); return; }
    if (this._dragN) {
      const d = this._dragN;
      this._dragN = null;
      if (d.moved) {
        d.n.drag = false;
        d.n.ret = { x: d.x0, y: d.y0 };
        d.n.vx = 0; d.n.vy = 0;
        this.canvas.style.cursor = 'pointer';
        this.wake();
      } else if (this._tapDbl(e)) {
        if (this.opts.zoom) this.fit(true);
      } else {
        const next = this.sel === d.n.id ? null : d.n.id;
        this.select(next);
        this.onSelect(next);
      }
      return;
    }
    if (this._pan) {
      const moved = this._pan.moved;
      this._pan = null;
      if (moved) return;
      if (this._tapDbl(e)) { if (this.opts.zoom) this.fit(true); return; }
      // empty-canvas click clears everything — a located/pinned node or a group mark counts too
      if (this.sel != null || this.spot != null || this.spotSet != null) { this.select(null); this.spot = null; this.spotSet = null; this.dirty = true; this.onSelect(null); }
    }
  }

  // a system gesture (or a lost pointer) must tear the drag down, not leave a ghost
  _cancel(e) {
    if (e && e.pointerId != null) this._ptrs.delete(e.pointerId); else this._ptrs.clear();
    if (this._pinch && this._ptrs.size < 2) this._endPinch();
    if (this._ptrs.size) return;
    this._abortDrag();
  }

  _abortDrag() {
    if (this._dragN) {
      const d = this._dragN; this._dragN = null;
      if (d.moved) { d.n.drag = false; d.n.ret = { x: d.x0, y: d.y0 }; d.n.vx = 0; d.n.vy = 0; this.wake(); }
      this.canvas.style.cursor = this.opts.pan ? 'grab' : 'default';
    }
    this._pan = null; this._tap = null;
  }

  // double-tap = fit (dblclick is unreliable on iPad); mouse keeps its native dblclick path
  _tapDbl(e) {
    if (e.pointerType === 'mouse') return false;
    const p = this._pos(e), t = e.timeStamp || Date.now(), l = this._tap;
    this._tap = { t: t, x: p[0], y: p[1] };
    if (l && t - l.t < 300 && Math.abs(p[0] - l.x) < 24 && Math.abs(p[1] - l.y) < 24) { this._tap = null; return true; }
    return false;
  }

  _gest() {
    const a = Array.from(this._ptrs.values()).slice(0, 2);
    return { d: Math.max(1, Math.hypot(a[1][0] - a[0][0], a[1][1] - a[0][1])), m: [(a[0][0] + a[1][0]) / 2, (a[0][1] + a[1][1]) / 2] };
  }

  _pinchStart() {
    this._abortDrag();
    if (this.hover != null) { this.hover = null; this.onHover(null); this.dirty = true; }
    this._pinch = this._gest();
  }

  _pinchMove() {
    if (this._ptrs.size < 2) return;
    const g = this._gest(), st = this._pinch;
    if (this.opts.zoom) {
      const k0 = this.cam.k, k1 = clamp(k0 * (g.d / st.d), .18, 6);
      const wx = (st.m[0] - this.w / 2) / k0 + this.cam.x;
      const wy = (st.m[1] - this.h / 2) / k0 + this.cam.y;
      this.cam.k = k1;
      this.cam.x = wx - (st.m[0] - this.w / 2) / k1;
      this.cam.y = wy - (st.m[1] - this.h / 2) / k1;
    }
    if (this.opts.pan) {
      this.cam.x -= (g.m[0] - st.m[0]) / this.cam.k;
      this.cam.y -= (g.m[1] - st.m[1]) / this.cam.k;
    }
    st.d = g.d; st.m = g.m;
    this.userCam = true; this._camT = null; this.dirty = true;
  }

  // lifting to one finger continues as a pan, re-seeded from the surviving pointer
  _endPinch() {
    this._pinch = null; this._tap = null;
    const a = Array.from(this._ptrs.values())[0];
    this._pan = a ? { cx: this.cam.x, cy: this.cam.y, px: a[0], py: a[1], moved: true } : null;
  }

  _wheel(e) {
    if (!this.opts.zoom) return;
    e.preventDefault();
    const p = this._pos(e);
    const k0 = this.cam.k;
    const k1 = clamp(k0 * Math.exp(-e.deltaY * .00125), .18, 6);
    if (k1 === k0) return;
    const wx = (p[0] - this.w / 2) / k0 + this.cam.x;
    const wy = (p[1] - this.h / 2) / k0 + this.cam.y;
    this.cam.k = k1;
    this.cam.x = wx - (p[0] - this.w / 2) / k1;
    this.cam.y = wy - (p[1] - this.h / 2) / k1;
    this.userCam = true; this._camT = null; this.dirty = true;
  }

  _dbl() { if (this.opts.zoom) this.fit(true); }
}
