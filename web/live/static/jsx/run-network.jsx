/* eslint-disable */
// Section C (Run mode) — Network visualization.
// Reproducible seed-based layout, 3 seed nodes + ~34 satellites,
// edges colored by target-paper year via the year ramp.

// Canvas renderer + light force-directed relaxation (ForceAtlas2-style:
// repulsion + attraction + gravity). clickNode → onSelectPaper + detail panel;
// wheel → zoom; drag → pan; hover → tooltip.
function RunNetwork({ selectedPaperId, onSelectPaper, hoverPaperId, onHoverPaper, theme }) {
  const { nodes: rawNodes, edges } = window.NETWORK;
  const canvasRef = React.useRef(null);
  const wrapRef = React.useRef(null);
  const tipRef = React.useRef(null);
  const stateRef = React.useRef(null);
  const [, forceRender] = React.useReducer(x => x + 1, 0);

  // One-time: seed simulation state.
  // World is NOT bounded — we run a real force-directed layout in world units
  // (positions in arbitrary scale), then fit-to-viewport with the camera.
  // Initial positions: de-center the spawn around origin to avoid biased grid.
  if (!stateRef.current) {
    const rng = (() => { let s = 29; return () => { s = (s*9301+49297)%233280; return s/233280; }; })();
    const pos = rawNodes.map((n) => {
      // Start in a tiny jittered disc near origin; the layout will explode
      // outward into its natural clusters.
      const r = 20 + rng() * 30;
      const a = rng() * Math.PI * 2;
      return { x: Math.cos(a) * r, y: Math.sin(a) * r, vx: 0, vy: 0 };
    });
    stateRef.current = {
      pos,
      camera: { fitScale: 1, userZoom: 1, cx: 0, cy: 0 },  // fitScale is world→px; userZoom is user pinch
      hoverIdx: -1,
      dragging: null,
      settled: 0,
      fitted: false,
    };
  }
  const st = stateRef.current;

  const cssVar = (name, fallback) => {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fallback;
    } catch (_) { return fallback; }
  };
  const yearColor = (y) => {
    const yy = Math.max(2018, Math.min(2025, y));
    return cssVar(`--cc-year-p-${yy}`, "#8a8a8a");
  };

  // --- Simulation step (ForceAtlas2-style, unbounded world) ----------------
  React.useEffect(() => {
    let raf;
    let frame = 0;
    const step = () => {
      const p = st.pos;
      // Cooling schedule: start hot (big moves), settle to crawl
      const t = Math.min(1, frame / 320);
      const speed = 1 - t * 0.85;        // 1.0 → 0.15
      // ForceAtlas2-ish forces in world units
      const kr = 420;                     // repulsion strength (Coulomb)
      const ka = 0.012;                   // linear attraction
      const kg = 0.08;                    // gravity toward origin (weak, not boxy)
      const damp = 0.78;

      // Repulsion — O(N²) ok for ~60 nodes. Soft-core to avoid blowups.
      for (let i = 0; i < p.length; i++) {
        const ni = rawNodes[i];
        for (let j = i + 1; j < p.length; j++) {
          const nj = rawNodes[j];
          let dx = p[i].x - p[j].x;
          let dy = p[i].y - p[j].y;
          let d2 = dx*dx + dy*dy + 0.1;
          let d = Math.sqrt(d2);
          // Scale repulsion by (1 + degree) proxy — seeds push harder
          const w = (1 + (ni.seed ? 6 : 0)) * (1 + (nj.seed ? 6 : 0));
          const f = kr * w / d2;
          const rx = (dx / d) * f, ry = (dy / d) * f;
          p[i].vx += rx * speed; p[i].vy += ry * speed;
          p[j].vx -= rx * speed; p[j].vy -= ry * speed;
        }
      }
      // Edge attraction — linear (FA2 LinLog-ish). Edge length target ~ 80 units.
      for (const e of edges) {
        const a = p[e.a], b = p[e.b];
        const dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.sqrt(dx*dx + dy*dy) + 0.01;
        // Attraction proportional to distance (longer edges pull harder)
        const f = ka * d;
        a.vx += (dx / d) * f * speed;
        a.vy += (dy / d) * f * speed;
        b.vx -= (dx / d) * f * speed;
        b.vy -= (dy / d) * f * speed;
      }
      // Gravity toward origin — very soft, just to prevent drift to infinity
      for (let i = 0; i < p.length; i++) {
        const dx = -p[i].x, dy = -p[i].y;
        const d = Math.sqrt(dx*dx + dy*dy) + 0.01;
        p[i].vx += (dx / d) * kg * speed;
        p[i].vy += (dy / d) * kg * speed;
        p[i].vx *= damp; p[i].vy *= damp;
        p[i].x += p[i].vx;
        p[i].y += p[i].vy;
        // NO hard bounds. The world can be as wide as the layout needs.
      }

      // Fit-to-viewport after an initial settle, then again periodically
      // while the graph is still expanding.
      if (frame === 60 || frame === 140 || frame === 260) fitCameraToGraph(0.9);

      frame++;
      st.settled = frame;
      draw();
      if (frame < 360) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []); // eslint-disable-line

  // Compute bbox of positions and set camera zoom/center so everything fits
  // inside `margin` fraction of the viewport.
  const fitCameraToGraph = React.useCallback((margin = 0.88) => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const cssW = wrap.clientWidth, cssH = wrap.clientHeight;
    if (!cssW || !cssH) return;
    const p = st.pos;
    let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
    for (let i = 0; i < p.length; i++) {
      const r = (rawNodes[i].r || 4) + 6;
      if (p[i].x - r < minX) minX = p[i].x - r;
      if (p[i].y - r < minY) minY = p[i].y - r;
      if (p[i].x + r > maxX) maxX = p[i].x + r;
      if (p[i].y + r > maxY) maxY = p[i].y + r;
    }
    const worldW = Math.max(1, maxX - minX);
    const worldH = Math.max(1, maxY - minY);
    const zx = (cssW * margin) / worldW;
    const zy = (cssH * margin) / worldH;
    st.camera.fitScale = Math.min(zx, zy);
    st.camera.cx = (minX + maxX) / 2;
    st.camera.cy = (minY + maxY) / 2;
    st.fitted = true;
  }, []);

  // --- Canvas draw ------------------------------------------------------
  const draw = React.useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const wrap = wrapRef.current;
    const cssW = wrap.clientWidth, cssH = wrap.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
      canvas.width = cssW * dpr; canvas.height = cssH * dpr;
      canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const { fitScale, userZoom, cx, cy } = st.camera;
    // world→screen: fitScale pins the graph to viewport, userZoom is user pinch
    const scale = fitScale * userZoom;
    const nodeScale = userZoom;  // nodes grow slightly as user zooms in, not with fit
    const toScreen = (wx, wy) => ({
      x: (wx - cx) * scale + cssW / 2,
      y: (wy - cy) * scale + cssH / 2,
    });

    // subtle dot grid
    ctx.fillStyle = cssVar('--cc-grid-dot-soft', 'rgba(68, 60, 40, 0.09)');
    const gridStep = 16;
    for (let gx = (cssW / 2 - cx * scale) % gridStep; gx < cssW; gx += gridStep) {
      for (let gy = (cssH / 2 - cy * scale) % gridStep; gy < cssH; gy += gridStep) {
        ctx.fillRect(gx, gy, 1, 1);
      }
    }

    // edges
    const selectedIdx = selectedPaperId ? rawNodes.findIndex(n => n.paperId === selectedPaperId) : -1;
    const hoverIdx = st.hoverIdx;
    const connected = new Set();
    if (selectedIdx >= 0 || hoverIdx >= 0) {
      const k = selectedIdx >= 0 ? selectedIdx : hoverIdx;
      for (const e of edges) {
        if (e.a === k) connected.add(e.b);
        if (e.b === k) connected.add(e.a);
      }
    }
    for (const e of edges) {
      const pa = st.pos[e.a], pb = st.pos[e.b];
      const sa = toScreen(pa.x, pa.y), sb = toScreen(pb.x, pb.y);
      const highlight = (e.a === selectedIdx || e.b === selectedIdx || e.a === hoverIdx || e.b === hoverIdx);
      ctx.strokeStyle = yearColor(rawNodes[e.b].year);
      ctx.globalAlpha = highlight ? 0.85 : 0.32;
      ctx.lineWidth = highlight ? 1.6 : 0.7;
      ctx.beginPath();
      ctx.moveTo(sa.x, sa.y); ctx.lineTo(sb.x, sb.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // nodes
    for (let i = 0; i < rawNodes.length; i++) {
      const n = rawNodes[i], p = st.pos[i];
      const s = toScreen(p.x, p.y);
      const isSelected = i === selectedIdx;
      const isHover = i === hoverIdx;
      const dim = (selectedIdx >= 0 || hoverIdx >= 0) && !isSelected && !isHover && !connected.has(i);

      ctx.globalAlpha = dim ? 0.22 : 1;

      if (n.seed) {
        // Seed: larger, filled ink-dark dot with a thin halo
        const rr = (n.r + 1.5) * nodeScale;
        ctx.beginPath();
        ctx.arc(s.x, s.y, rr + 3, 0, Math.PI * 2);
        ctx.fillStyle = cssVar('--cc-net-seed-halo', 'rgba(62, 53, 72, 0.10)');
        ctx.fill();
        ctx.beginPath();
        ctx.arc(s.x, s.y, rr, 0, Math.PI * 2);
        ctx.fillStyle = cssVar('--cc-net-seed', '#3e3548');
        ctx.fill();
        ctx.strokeStyle = cssVar('--cc-net-seed-stroke', '#faf8f2');
        ctx.lineWidth = 1.2;
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(s.x, s.y, n.r * nodeScale, 0, Math.PI * 2);
        ctx.fillStyle = yearColor(n.year);
        ctx.fill();
        ctx.strokeStyle = cssVar('--cc-net-node-stroke', 'rgba(250, 248, 242, 0.7)');
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
      if (isSelected || isHover) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, (n.r + 3) * nodeScale, 0, Math.PI * 2);
        ctx.strokeStyle = isSelected
          ? cssVar('--cc-net-sel-strong', '#3e3548')
          : cssVar('--cc-net-sel-soft', 'rgba(62, 53, 72, 0.55)');
        ctx.lineWidth = isSelected ? 1.8 : 1.2;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }, [selectedPaperId]);

  // Redraw whenever selection changes
  React.useEffect(() => { draw(); }, [draw, selectedPaperId]);

  // Redraw when theme flips — CSS vars change but positions remain
  React.useEffect(() => {
    // Give the style pass a frame to commit the new custom properties
    const id = requestAnimationFrame(() => draw());
    return () => cancelAnimationFrame(id);
  }, [theme, draw]);

  // --- Pointer interactions --------------------------------------------
  const pickNode = (ev) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    const { fitScale, userZoom, cx, cy } = st.camera;
    const scale = fitScale * userZoom;
    const nodeScale = userZoom;
    let best = -1, bestD = Infinity;
    for (let i = 0; i < rawNodes.length; i++) {
      const p = st.pos[i];
      const sx = (p.x - cx) * scale + rect.width / 2;
      const sy = (p.y - cy) * scale + rect.height / 2;
      const dx = mx - sx, dy = my - sy;
      const r = (rawNodes[i].r + 3) * nodeScale;
      const d2 = dx * dx + dy * dy;
      if (d2 < r * r && d2 < bestD) { best = i; bestD = d2; }
    }
    return { idx: best, mx, my };
  };

  const onMouseMove = (ev) => {
    if (st.dragging && st.dragging.type === "pan") {
      const scale = st.camera.fitScale * st.camera.userZoom;
      const dx = (ev.clientX - st.dragging.startX) / scale;
      const dy = (ev.clientY - st.dragging.startY) / scale;
      st.camera.cx = st.dragging.startCam.cx - dx;
      st.camera.cy = st.dragging.startCam.cy - dy;
      draw();
      return;
    }
    const { idx, mx, my } = pickNode(ev);
    if (idx !== st.hoverIdx) {
      st.hoverIdx = idx;
      onHoverPaper && onHoverPaper(idx >= 0 ? rawNodes[idx].paperId : null);
      draw();
    }
    const tip = tipRef.current;
    if (tip) {
      if (idx >= 0) {
        const paper = (window.ACCEPTED_PAPERS || []).find(p => p.id === rawNodes[idx].paperId);
        tip.style.display = "block";
        tip.style.left = (mx + 14) + "px";
        tip.style.top = (my + 14) + "px";
        tip.innerHTML = paper
          ? `<b>${paper.title}</b><br/><span class="net-tip-meta">${paper.authors} · ${paper.year} · ${paper.venue} · ${paper.cites.toLocaleString()} cites</span>`
          : `<b>${rawNodes[idx].id}</b>`;
      } else {
        tip.style.display = "none";
      }
    }
    wrapRef.current.style.cursor = idx >= 0 ? "pointer" : (st.dragging ? "grabbing" : "grab");
  };

  const onMouseDown = (ev) => {
    const { idx } = pickNode(ev);
    if (idx >= 0) {
      onSelectPaper && onSelectPaper(rawNodes[idx].paperId);
      return;
    }
    st.dragging = {
      type: "pan",
      startX: ev.clientX, startY: ev.clientY,
      startCam: { ...st.camera },
    };
    wrapRef.current.style.cursor = "grabbing";
  };
  const onMouseUp = () => { st.dragging = null; };
  const onMouseLeave = () => {
    st.dragging = null; st.hoverIdx = -1;
    if (tipRef.current) tipRef.current.style.display = "none";
    onHoverPaper && onHoverPaper(null);
    draw();
  };

  const onWheel = (ev) => {
    ev.preventDefault();
    const f = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
    const next = Math.max(0.4, Math.min(5, st.camera.userZoom * f));
    st.camera.userZoom = next;
    forceRender();
    draw();
  };

  React.useEffect(() => {
    const w = wrapRef.current;
    if (!w) return;
    const handler = (e) => onWheel(e);
    w.addEventListener("wheel", handler, { passive: false });
    return () => w.removeEventListener("wheel", handler);
  }, []); // eslint-disable-line

  React.useEffect(() => {
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

  // Zoom controls
  const zoomBy = (f) => {
    st.camera.userZoom = Math.max(0.4, Math.min(5, st.camera.userZoom * f));
    forceRender();
    draw();
  };
  const resetView = () => {
    st.camera.userZoom = 1;
    fitCameraToGraph(0.9);
    forceRender();
    draw();
  };

  return (
    <section className="pane pane-top">
      <div className="network" ref={wrapRef}
        onMouseMove={onMouseMove}
        onMouseDown={onMouseDown}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
      >
        <canvas ref={canvasRef} className="network-canvas" />
        <div ref={tipRef} className="net-tip" />

        <div className="net-toolbar">
          <button className="btn-icon btn" onClick={() => zoomBy(1 / 1.25)}><Icon name="minus" size={11}/></button>
          <span className="pipe-zoom">{Math.round(st.camera.userZoom * 100)}%</span>
          <button className="btn-icon btn" onClick={() => zoomBy(1.25)}><Icon name="plus" size={11}/></button>
          <button className="btn-icon btn" onClick={resetView} title="Reset view"><Icon name="maximize-2" size={11}/></button>
        </div>

        <div className="net-legend">
          <span className="net-legend-item">
            <span className="net-dot seed" />
            <span>Seed</span>
          </span>
          <span className="net-legend-item">
            <span className="net-ramp">
              {[2018,2019,2020,2021,2022,2023,2024,2025].map(y => (
                <span key={y} className="net-ramp-step" style={{ background: yearColor(y) }} />
              ))}
            </span>
            <span>2018 → 2025</span>
          </span>
          <span className="net-legend-item">
            <span className="net-hint-txt">drag · scroll · click</span>
          </span>
        </div>

        <div className="net-counter">
          <span>
            <span className="net-counter-num">{rawNodes.length}</span>
            <span className="net-counter-lbl">nodes</span>
          </span>
          <span className="net-counter-sep">·</span>
          <span>
            <span className="net-counter-num">{edges.length}</span>
            <span className="net-counter-lbl">edges</span>
          </span>
          <span className="net-counter-sep">·</span>
          <span>
            <span className="net-counter-num">+42</span>
            <span className="net-counter-lbl">/ min</span>
          </span>
        </div>
      </div>
    </section>
  );
}

Object.assign(window, { RunNetwork });
