// topic-viz.js — topic-map scatter for KnowledgeLab (Explore page).
// Fixed 2D embedding (one point per paper), topic-colored with a recycling
// 12-color muted palette (noise = light grey), screen-fixed point sizes,
// zoomable camera (wheel / buttons / dblclick-fit / marquee region zoom),
// centroid topic labels with greedy de-collision that reveal on zoom,
// hover + selection states, topic hover/focus emphasis. No dependencies.

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const cubicOut = t => 1 - Math.pow(1 - t, 3);

// Recycled categorical set: the muted --v-* identifier band + darker ink
// mixes of the same hues. Topic id t -> TOPIC_COLORS[t % 12]; -1 -> NOISE.
export const TOPIC_COLORS = [
  'var(--v-teal-dot)',
  'var(--v-gold-dot)',
  'var(--v-purple-dot)',
  'var(--v-rose-dot)',
  'var(--v-blue-dot)',
  'var(--v-green-dot)',
  'var(--v-indigo)',
  'color-mix(in oklab, var(--v-teal) 55%, var(--fg))',
  'color-mix(in oklab, var(--v-gold) 58%, var(--fg))',
  'color-mix(in oklab, var(--v-rose) 58%, var(--fg))',
  'color-mix(in oklab, var(--v-blue) 55%, var(--fg))',
  'color-mix(in oklab, var(--v-green) 55%, var(--fg))'
];
export const NOISE_COLOR = 'color-mix(in oklab, var(--muted2) 46%, var(--canvas))';
export const topicColor = t => t < 0 ? NOISE_COLOR : TOPIC_COLORS[t % TOPIC_COLORS.length];

export class TopicMap {
  constructor(o) {
    this.canvas = o.canvas;
    this.host = o.host;
    this.onSelect = o.onSelect || (() => {});
    this.onHover = o.onHover || (() => {});
    this.onHoverMove = o.onHoverMove || (() => {});
    this.onMarqueeEnd = o.onMarqueeEnd || (() => {});
    this.pts = []; this.topics = new Map(); this.labelOrder = [];
    this.sel = null; this.hover = null; this._spot = null; this._spotSet = null;
    this._hovT = null; this._focT = null;
    this.labelScale = 1;
    // style, driven by the Topic map's Style panel (same grammar as the network's)
    this.appearance = { sizeBy: 'cites', sizeScale: 'log', sizeMin: 3, sizeMax: 13, alpha: .84, stroke: 1, noise: true };
    this.marqueeMode = false;
    this.cam = { x: 0, y: 0, k: 1 };
    this._k0 = 1;
    this.visible = true; this.dirty = true; this.didFit = false;
    this.w = 0; this.h = 0;
    this.ctx = this.canvas.getContext('2d');
    this._probe = document.createElement('span');
    this._probe.style.cssText = 'position:absolute;left:-9999px;top:0;visibility:hidden;pointer-events:none;transition:none !important;';
    this.host.appendChild(this._probe);
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
  // papers: [{x, y, tp}] (index = id) · topics: [{id, name, n}]
  setData(d) {
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    d.papers.forEach(p => {
      if (p.x < x0) x0 = p.x; if (p.x > x1) x1 = p.x;
      if (p.y < y0) y0 = p.y; if (p.y > y1) y1 = p.y;
    });
    const s = 640 / Math.max(1e-6, Math.max(x1 - x0, y1 - y0));
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    this.pts = d.papers.map((p, i) => ({
      id: i, t: p.tp,
      w: Math.max(0, +(p.ci != null ? p.ci : (p.cites || 0))),
      x: (p.x - cx) * s,
      y: (cy - p.y) * s   // flip: data y up, screen y down
    }));
    // log-scaled size domain — the same visual grammar as the citation network:
    // area carries citations, so the map reads as a field of weighted papers
    // rather than uniform polka dots
    let wmax = 0;
    this.pts.forEach(p => { if (p.w > wmax) wmax = p.w; });
    this._wmax = wmax || 1;
    this._wl = Math.log1p(wmax) || 1;
    this.topics = new Map();
    d.topics.forEach(t => this.topics.set(t.id, { id: t.id, name: t.name, n: t.n }));
    // centroids (median resists stragglers)
    const mem = new Map();
    this.pts.forEach(p => { if (!mem.has(p.t)) mem.set(p.t, []); mem.get(p.t).push(p); });
    this._mem = mem;
    mem.forEach((arr, t) => {
      const meta = this.topics.get(t); if (!meta) return;
      const xs = arr.map(p => p.x).sort((a, b) => a - b);
      const ys = arr.map(p => p.y).sort((a, b) => a - b);
      meta.cx = xs[Math.floor(xs.length / 2)];
      meta.cy = ys[Math.floor(ys.length / 2)];
      meta.ly = ys[Math.min(ys.length - 1, Math.floor(ys.length * 0.94))];
    });
    this.labelOrder = [...this.topics.values()].filter(t => t.id >= 0).sort((a, b) => b.n - a.n);
    this._reindex();
    this.refreshTheme();
    this.didFit = false;
    this.fit(false);
    this.dirty = true;
  }

  // ------------------------------------------------------------- theme
  refreshTheme() {
    const res = c => {
      this._probe.style.color = c;
      return getComputedStyle(this._probe).color;
    };
    const cvs = res('var(--canvas)');
    const lm = /rgb\((\d+),\s*(\d+),\s*(\d+)/.exec(cvs || '');
    const lum = lm ? (.2126 * lm[1] + .7152 * lm[2] + .0722 * lm[3]) / 255 : 1;
    const anchor = lum < .5 ? 'var(--canvas)' : 'var(--fg)';
    this.C = {
      topics: TOPIC_COLORS.map(res),
      noise: res(NOISE_COLOR),
      canvas: res('var(--canvas)'),
      accent: res('var(--accent)'),
      strokes: TOPIC_COLORS.map(c => res('color-mix(in oklab, ' + c + ' 55%, ' + anchor + ')')),
      noiseStroke: res('color-mix(in oklab, ' + NOISE_COLOR + ' 60%, ' + anchor + ')'),
      label: res('color-mix(in oklab, var(--fg) 80%, var(--canvas))'),
      labelDim: res('color-mix(in oklab, var(--fg) 34%, var(--canvas))'),
      ink: res('color-mix(in oklab, var(--fg) 90%, var(--canvas))')
    };
    this.dirty = true;
  }
  colorOf(t) { return t < 0 ? this.C.noise : this.C.topics[t % this.C.topics.length]; }
  strokeOf(t) { return t < 0 ? this.C.noiseStroke : this.C.strokes[t % this.C.strokes.length]; }

  // ------------------------------------------------------------- camera
  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.w = w; this.h = h;
    const rect = this.host.getBoundingClientRect();
    const cssScale = rect.width ? rect.width / w : 1;
    const d = this.dpr = clamp((window.devicePixelRatio || 1) * cssScale, 1, 3);
    const pw = Math.round(w * d), ph = Math.round(h * d);
    if (this.canvas.width !== pw || this.canvas.height !== ph) { this.canvas.width = pw; this.canvas.height = ph; }
    this.dirty = true;
    if (!this.didFit && this.pts.length) this.fit(false);
  }

  _bbox(pts) {
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    pts.forEach(p => {
      if (p.x < x0) x0 = p.x; if (p.x > x1) x1 = p.x;
      if (p.y < y0) y0 = p.y; if (p.y > y1) y1 = p.y;
    });
    return { x0, y0, x1, y1 };
  }

  fit(animate, bb) {
    if (!this.w || !this.pts.length) return;
    const b = bb || this._bbox(this.pts);
    const pad = 52;
    const k = clamp(Math.min((this.w - pad * 2) / Math.max(40, b.x1 - b.x0), (this.h - pad * 2) / Math.max(40, b.y1 - b.y0)), .01, 40);
    if (!bb) this._k0 = k;
    const to = { x: (b.x0 + b.x1) / 2, y: (b.y0 + b.y1) / 2, k: k };
    this.didFit = true;
    if (!animate) { this.cam = to; this._camT = null; this.dirty = true; return; }
    this.camTween(to, 520);
  }

  fitTopic(id, animate) {
    const arr = this._mem && this._mem.get(id);
    if (!arr || !arr.length) return;
    const b = this._bbox(arr);
    const g = 26; b.x0 -= g; b.x1 += g; b.y0 -= g; b.y1 += g;
    // never zoom past 5.5x the overview scale for a tight cluster
    const pad = 52;
    let k = Math.min((this.w - pad * 2) / Math.max(40, b.x1 - b.x0), (this.h - pad * 2) / Math.max(40, b.y1 - b.y0));
    k = clamp(k, this._k0 * .6, this._k0 * 5.5);
    const to = { x: (b.x0 + b.x1) / 2, y: (b.y0 + b.y1) / 2, k: k };
    animate === false ? (this.cam = to, this.dirty = true) : this.camTween(to, 560);
  }

  camTween(to, dur) {
    this._camT = { from: { x: this.cam.x, y: this.cam.y, k: this.cam.k }, to: to, t0: performance.now(), dur: dur };
    this.dirty = true;
  }

  zoomBy(f) {
    const k1 = clamp(this.cam.k * f, this._k0 * .5, this._k0 * 30);
    if (k1 === this.cam.k) return;
    this.camTween({ x: this.cam.x, y: this.cam.y, k: k1 }, 260);
  }

  _sx(p) { return (p.x - this.cam.x) * this.cam.k + this.w / 2; }
  _sy(p) { return (p.y - this.cam.y) * this.cam.k + this.h / 2; }
  _wx(sx) { return (sx - this.w / 2) / this.cam.k + this.cam.x; }
  _wy(sy) { return (sy - this.h / 2) / this.cam.k + this.cam.y; }

  // ------------------------------------------------------------- interaction
  _pos(e) {
    const r = this.canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * (this.w / r.width), y: (e.clientY - r.top) * (this.h / r.height) };
  }

  _down(e) {
    if (e.button !== 0) return;
    const p = this._pos(e);
    this._ptrs.set(e.pointerId, p);
    this.canvas.setPointerCapture && this.canvas.setPointerCapture(e.pointerId);
    if (this._ptrs.size >= 2) { this._pinchStart(); return; }
    this._pd = { x: p.x, y: p.y, cam: { x: this.cam.x, y: this.cam.y }, moved: false };
    if (this.marqueeMode || e.shiftKey) { this._mq = { x0: p.x, y0: p.y, x1: p.x, y1: p.y }; }
  }

  _move(e) {
    const p = this._pos(e);
    if (this._ptrs.has(e.pointerId)) this._ptrs.set(e.pointerId, p);
    if (this._pinch) { this._pinchMove(); return; }
    if (this._pd) {
      const dx = p.x - this._pd.x, dy = p.y - this._pd.y;
      if (Math.abs(dx) > 5 || Math.abs(dy) > 5) this._pd.moved = true;
      if (this._mq) { this._mq.x1 = p.x; this._mq.y1 = p.y; this.dirty = true; }
      else if (this._pd.moved) {
        this._camT = null;
        this.cam.x = this._pd.cam.x - dx / this.cam.k;
        this.cam.y = this._pd.cam.y - dy / this.cam.k;
        this.dirty = true;
      }
      this._cursor();
      return;
    }
    const id = this._hit(p);
    if (id !== this.hover) {
      this.hover = id;
      this.dirty = true;
      this.onHover(id);
    }
    const hp = this._ptOf(id);
    if (hp) this.onHoverMove({ x: this._sx(hp), y: this._sy(hp), r: this._r(hp) });
    this._cursor();
  }

  _up(e) {
    this._ptrs.delete(e.pointerId);
    if (this._pinch) { if (this._ptrs.size < 2) this._endPinch(); return; }
    if (!this._pd) return;
    const pd = this._pd, mq = this._mq;
    this._pd = null; this._mq = null;
    if (mq) {
      const x0 = Math.min(mq.x0, mq.x1), x1 = Math.max(mq.x0, mq.x1);
      const y0 = Math.min(mq.y0, mq.y1), y1 = Math.max(mq.y0, mq.y1);
      if (x1 - x0 > 14 && y1 - y0 > 14) {
        const b = { x0: this._wx(x0), x1: this._wx(x1), y0: this._wy(y0), y1: this._wy(y1) };
        const pad = 52;
        let k = Math.min((this.w - pad * 2) / Math.max(8, b.x1 - b.x0), (this.h - pad * 2) / Math.max(8, b.y1 - b.y0));
        k = clamp(k, this._k0 * .5, this._k0 * 30);
        this.camTween({ x: (b.x0 + b.x1) / 2, y: (b.y0 + b.y1) / 2, k: k }, 480);
      }
      if (this.marqueeMode) { this.marqueeMode = false; this.onMarqueeEnd(); }
      this.dirty = true;
      this._cursor();
      return;
    }
    if (!pd.moved) {
      if (this._tapDbl(e)) { this.fit(true); this._cursor(); return; }
      const id = this._hit({ x: pd.x, y: pd.y });
      if (id == null && this.sel == null && (this._spot != null || this._spotSet != null)) { this._spot = null; this._spotSet = null; this.dirty = true; this.onSelect(null); }
      else this.select(id, { fire: true });
    }
    this._cursor();
  }

  // a system gesture (or a lost pointer) must tear the drag down, not leave a ghost
  _cancel(e) {
    if (e && e.pointerId != null) this._ptrs.delete(e.pointerId); else this._ptrs.clear();
    if (this._pinch && this._ptrs.size < 2) this._endPinch();
    if (this._ptrs.size) return;
    this._pd = null; this._mq = null; this._tap = null;
    if (this.marqueeMode) { this.marqueeMode = false; this.onMarqueeEnd(); }
    this.dirty = true; this._cursor();
  }

  // double-tap = fit (dblclick is unreliable on iPad); mouse keeps its native dblclick path
  _tapDbl(e) {
    if (e.pointerType === 'mouse') return false;
    const p = this._pos(e), t = e.timeStamp || Date.now(), l = this._tap;
    this._tap = { t: t, x: p.x, y: p.y };
    if (l && t - l.t < 300 && Math.abs(p.x - l.x) < 24 && Math.abs(p.y - l.y) < 24) { this._tap = null; return true; }
    return false;
  }

  _gest() {
    const a = Array.from(this._ptrs.values()).slice(0, 2);
    return { d: Math.max(1, Math.hypot(a[1].x - a[0].x, a[1].y - a[0].y)), m: { x: (a[0].x + a[1].x) / 2, y: (a[0].y + a[1].y) / 2 } };
  }

  _pinchStart() {
    this._pd = null; this._mq = null;
    if (this.hover != null) { this.hover = null; this.onHover(null); this.dirty = true; }
    this._pinch = this._gest();
    this._cursor();
  }

  _pinchMove() {
    if (this._ptrs.size < 2) return;
    const g = this._gest(), st = this._pinch;
    const k1 = clamp(this.cam.k * (g.d / st.d), this._k0 * .5, this._k0 * 30);
    const wx = this._wx(st.m.x), wy = this._wy(st.m.y);
    this._camT = null;
    this.cam.k = k1;
    this.cam.x = wx - (st.m.x - this.w / 2) / k1;
    this.cam.y = wy - (st.m.y - this.h / 2) / k1;
    this.cam.x -= (g.m.x - st.m.x) / this.cam.k;
    this.cam.y -= (g.m.y - st.m.y) / this.cam.k;
    st.d = g.d; st.m = g.m;
    this.dirty = true;
  }

  // lifting to one finger continues as a pan, re-seeded from the surviving pointer
  _endPinch() {
    this._pinch = null; this._tap = null;
    const a = Array.from(this._ptrs.values())[0];
    this._pd = a ? { x: a.x, y: a.y, cam: { x: this.cam.x, y: this.cam.y }, moved: true } : null;
    this._cursor();
  }

  _wheel(e) {
    e.preventDefault();
    const p = this._pos(e);
    const f = Math.exp(-e.deltaY * .0016);
    const k1 = clamp(this.cam.k * f, this._k0 * .5, this._k0 * 30);
    if (k1 === this.cam.k) return;
    const wx = this._wx(p.x), wy = this._wy(p.y);
    this._camT = null;
    this.cam.k = k1;
    this.cam.x = wx - (p.x - this.w / 2) / k1;
    this.cam.y = wy - (p.y - this.h / 2) / k1;
    this.dirty = true;
  }

  _dbl() { this.fit(true); }

  _cursor() {
    this.canvas.style.cursor = (this.marqueeMode || this._mq) ? 'crosshair' :
      this._pd && this._pd.moved ? 'grabbing' :
      this.hover != null ? 'pointer' : 'grab';
  }

  // point lookup is BY ID (the paper's corpus index), never by array position
  _reindex() { this._byId = new Map(); for (const p of this.pts) this._byId.set(p.id, p); }
  _ptOf(id) { return id == null || !this._byId ? null : (this._byId.get(id) || null); }

  _hit(p) {
    let best = null, bd = 1e9;
    for (let i = 0; i < this.pts.length; i++) {
      const pt = this.pts[i];
      if (pt.t < 0 && !this.appearance.noise) continue;
      const dx = this._sx(pt) - p.x, dy = this._sy(pt) - p.y;
      const d = dx * dx + dy * dy;
      const rr = this._r(pt) + 3.5;
      if (d < rr * rr && d < bd) { bd = d; best = pt.id; }
    }
    return best;
  }

  // ------------------------------------------------------------- state
  select(id, opts) {
    opts = opts || {};
    if (id === this.sel && !opts.focus) { if (opts.fire) return; }
    const changed = id !== this.sel;
    this.sel = id;
    if (id != null) this._focT = null; // a picked paper supersedes topic focus
    this.dirty = true;
    if (opts.fire && changed) this.onSelect(id);
    if (opts.focus && id != null) {
      const pt = this._ptOf(id);
      if (!pt) return;
      const sx = this._sx(pt), sy = this._sy(pt);
      const mx = this.w * .16, my = this.h * .16;
      if (sx < mx || sx > this.w - mx || sy < my || sy > this.h - my)
        this.camTween({ x: pt.x, y: pt.y, k: this.cam.k }, 460);
    }
  }

  camGet() { return { x: this.cam.x, y: this.cam.y, k: this.cam.k }; }
  camGoTo(c, dur) { if (!c) return; this.camTween({ x: c.x, y: c.y, k: c.k }, dur == null ? 480 : dur); }

  hoverTopic(t) { if (this._hovT !== t) { this._hovT = t; this.dirty = true; } }

  spotlight(id) { if (this._spot !== id) { this._spot = id; this.dirty = true; } }

  // group mark (a topic card left active after backing out of its drill): accent-rings
  // every member point — rings only, no fade treatment
  spotlightSet(ids) { this._spotSet = ids && ids.length ? new Set(ids) : null; this.dirty = true; }

  focusTopic(t, opts) {
    opts = opts || {};
    this._focT = t;
    const sp = this._ptOf(this.sel);
    if (t != null && sp && sp.t !== t) this.sel = null;
    this.dirty = true;
    if (t != null && opts.fit !== false) this.fitTopic(t, true);
    if (t == null && opts.fit !== false) this.fit(true);
  }
  focusedTopic() { return this._focT; }

  setMarquee(v) { this.marqueeMode = !!v; this._cursor(); }
  setLabelScale(v) { this.labelScale = v; this.dirty = true; }
  setAppearance(o) { Object.assign(this.appearance, o || {}); this.dirty = true; }
  setVisible(v) { this.visible = v; if (v) { this.resize(); this.dirty = true; } }

  // entry animation: points pop in staggered by topic size rank
  enter() {
    const rank = new Map();
    this.labelOrder.forEach((t, i) => rank.set(t.id, i));
    const now = performance.now();
    const b = {};
    this.pts.forEach(p => {
      const r = p.t < 0 ? this.labelOrder.length + 4 : (rank.get(p.t) || 0);
      b[p.id] = now + 120 + r * 16 + Math.random() * 200;
    });
    this._birth = b;
    this._enterEnd = now + 120 + (this.labelOrder.length + 6) * 16 + 200 + 340;
    this.dirty = true;
  }

  // ------------------------------------------------------------- loop
  _loop(now) {
    this._raf = requestAnimationFrame(this._loop);
    if (!this.visible) return;
    if (this._camT) {
      const c = this._camT, t = clamp((now - c.t0) / c.dur, 0, 1), e = cubicOut(t);
      this.cam = { x: lerp(c.from.x, c.to.x, e), y: lerp(c.from.y, c.to.y, e), k: lerp(c.from.k, c.to.k, e) };
      if (t >= 1) this._camT = null;
      this.dirty = true;
    }
    if (this._enterEnd && now < this._enterEnd) this.dirty = true;
    else if (this._enterEnd) { this._enterEnd = 0; this._birth = null; this.dirty = true; }
    if (this.dirty) { this.dirty = false; this.draw(now); }
  }

  _norm(pt) {
    const A = this.appearance;
    if (A.sizeBy === 'uniform') return .42;
    const w = pt.w || 0;
    if (A.sizeScale === 'log') return this._wl ? Math.log1p(w) / this._wl : 0;
    const u = this._wmax ? w / this._wmax : 0;
    if (A.sizeScale === 'sqrt') return Math.sqrt(u);
    if (A.sizeScale === 'pow2') return u * u;
    return u;
  }
  _r(pt) {
    const A = this.appearance;
    const z = clamp(Math.sqrt(Math.max(.2, this.cam.k / this._k0)), .8, 1.55);
    let r = (A.sizeMin + this._norm(pt) * Math.max(0, A.sizeMax - A.sizeMin)) * z;
    if (pt.t < 0) r *= .62;
    if (pt.id === this.sel) r += 2.2;
    else if (pt.id === this._spot) r += 2;
    else if (pt.id === this.hover) r += 1.8;
    return clamp(r, 1.5, 44);
  }

  draw(now) {
    const ctx = this.ctx, d = this.dpr;
    ctx.setTransform(d, 0, 0, d, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    if (!this.pts.length) return;
    const selPt = this._ptOf(this.sel);
    const spotPt = this._ptOf(this._spot);
    const E = this._hovT != null ? this._hovT : (this._focT != null ? this._focT : (selPt ? selPt.t : null));
    const birth = this._birth;
    const drawPt = pt => {
      if (pt === selPt || pt === spotPt) return; // drawn last
      const A0 = this.appearance;
      if (pt.t < 0 && !A0.noise) return;
      let a = pt.t < 0 ? A0.alpha * .66 : A0.alpha;
      if (E != null) a = pt.t === E ? Math.min(1, a + .06) : a * .17;
      let r = this._r(pt);
      if (birth) {
        const bt = birth[pt.id];
        const t = clamp((now - bt) / 300, 0, 1);
        if (t <= 0) return;
        const e = cubicOut(t);
        a *= e; r *= .3 + .7 * e;
      }
      const x = this._sx(pt), y = this._sy(pt);
      if (x < -12 || x > this.w + 12 || y < -12 || y > this.h + 12) return;
      ctx.globalAlpha = a;
      ctx.fillStyle = this.colorOf(pt.t);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 6.2832);
      ctx.fill();
      if (A0.stroke > 0) {
        ctx.globalAlpha = a * .85;
        ctx.lineWidth = Math.max(.5, Math.min(1.6, r * .12)) * A0.stroke;
        ctx.strokeStyle = this.strokeOf(pt.t);
        ctx.stroke();
      }
      if (pt.id === this.hover) {
        ctx.globalAlpha = Math.min(1, a + .2);
        ctx.lineWidth = 1.6;
        ctx.strokeStyle = this.C.canvas;
        ctx.beginPath();
        ctx.arc(x, y, r + 1.3, 0, 6.2832);
        ctx.stroke();
      }
    };
    // noise underneath, then topics; emphasized topic on top
    for (const pt of this.pts) if (pt.t < 0 && pt.t !== E) drawPt(pt);
    for (const pt of this.pts) if (pt.t >= 0 && pt.t !== E) drawPt(pt);
    if (E != null) for (const pt of this.pts) if (pt.t === E) drawPt(pt);
    // group-mark members — plain accent ring on each (no halo; many points carry it)
    if (this._spotSet) {
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1.8; ctx.strokeStyle = this.C.accent;
      for (const pt of this.pts) {
        if (!this._spotSet.has(pt.id) || pt === selPt || pt === spotPt) continue;
        const x = this._sx(pt), y = this._sy(pt), r = this._r(pt);
        ctx.beginPath(); ctx.arc(x, y, r + 2.2, 0, 6.2832); ctx.stroke();
      }
    }
    // spotlighted point (row hover in the topic drill) — accent ring, on top
    if (spotPt && spotPt !== selPt) { // located point: accent ring, nothing else touched
      const pt = spotPt;
      const x = this._sx(pt), y = this._sy(pt), r = this._r(pt);
      ctx.globalAlpha = 1;
      ctx.fillStyle = this.colorOf(pt.t);
      ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
      if (this.appearance.stroke > 0) { ctx.lineWidth = Math.max(.7, Math.min(1.6, r * .13)) * this.appearance.stroke; ctx.strokeStyle = this.strokeOf(pt.t); ctx.stroke(); }
      ctx.lineWidth = 2; ctx.strokeStyle = this.C.accent;
      ctx.beginPath(); ctx.arc(x, y, r + 2.6, 0, 6.2832); ctx.stroke();
    }
    // selected point: halo + casing + accent ring
    if (selPt) {
      const pt = selPt;
      const x = this._sx(pt), y = this._sy(pt), r = this._r(pt);
      ctx.globalAlpha = .16;
      ctx.fillStyle = this.C.accent;
      ctx.beginPath(); ctx.arc(x, y, r + 8, 0, 6.2832); ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = this.colorOf(pt.t);
      ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
      ctx.lineWidth = 2.4; ctx.strokeStyle = this.C.canvas;
      ctx.beginPath(); ctx.arc(x, y, r + 1.1, 0, 6.2832); ctx.stroke();
      ctx.lineWidth = 2; ctx.strokeStyle = this.C.accent;
      ctx.beginPath(); ctx.arc(x, y, r + 2.8, 0, 6.2832); ctx.stroke();
    }
    this._labels(ctx, E, now);
    // marquee rectangle
    if (this._mq) {
      const m = this._mq;
      const x = Math.min(m.x0, m.x1), y = Math.min(m.y0, m.y1);
      const w = Math.abs(m.x1 - m.x0), h = Math.abs(m.y1 - m.y0);
      ctx.globalAlpha = .08;
      ctx.fillStyle = this.C.accent;
      ctx.fillRect(x, y, w, h);
      ctx.globalAlpha = 1;
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = this.C.accent;
      ctx.strokeRect(x + .5, y + .5, w, h);
      ctx.setLineDash([]);
    }
    ctx.globalAlpha = 1;
  }

  _labels(ctx, E, now) {
    if (this._birth && now < (this._enterEnd || 0) - 260) return;
    const z = this.cam.k / this._k0;
    const nShow = Math.max(10, Math.round(12 * Math.pow(Math.max(z, .6), 1.7)));
    const placed = [];
    const tryPlace = (t, forced) => {
      const x = this._sx({ x: t.cx, y: t.cy }), y = this._sy({ x: t.cx, y: t.ly != null ? t.ly : t.cy }) + 15;
      if (x < -80 || x > this.w + 80 || y < -20 || y > this.h + 20) return;
      const fs = clamp((9.5 + Math.sqrt(t.n) * .5) * this.labelScale, 9, 16);
      ctx.font = '600 ' + fs.toFixed(1) + 'px "Hanken Grotesk", sans-serif';
      const txt = t.name.toUpperCase();
      const tw = ctx.measureText(txt).width;
      let cy = y;
      const box = { x0: x - tw / 2 - 4, x1: x + tw / 2 + 4, y0: cy - fs / 2 - 3, y1: cy + fs / 2 + 3 };
      if (!forced) {
        for (const b of placed) {
          if (box.x0 < b.x1 && box.x1 > b.x0 && box.y0 < b.y1 && box.y1 > b.y0) return;
        }
      }
      placed.push(box);
      const dim = E != null && t.id !== E;
      ctx.globalAlpha = 1;
      ctx.lineWidth = 5;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = this.C.canvas;
      ctx.strokeText(txt, x - tw / 2, cy + fs * .35);
      ctx.fillStyle = dim ? this.C.labelDim : this.C.label;
      ctx.fillText(txt, x - tw / 2, cy + fs * .35);
    };
    try { ctx.letterSpacing = '0.5px'; } catch (e) { /* older engines */ }
    if (E != null && E >= 0) { const t = this.topics.get(E); if (t) tryPlace(t, true); }
    let n = 0;
    for (const t of this.labelOrder) {
      if (n >= nShow) break;
      if (E != null && t.id === E) { n++; continue; }
      tryPlace(t, false);
      n++;
    }
    try { ctx.letterSpacing = '0px'; } catch (e) { }
  }
}
