/* eslint-disable */
// Section C (Build mode) — Pipeline canvas.
// Renders a horizontal chain of blocks with connecting edges, 4 visual styles
// (selectable via Tweaks): card / chip / ghost / tag.

function PipelineBlock({ node, selected, onClick, style, index }) {
  const iconFor = {
    seed: "flag", fwd: "arrow-right", bwd: "arrow-left",
    rerank: "sliders-horizontal", rsc: "filter", sink: "inbox",
  };

  // Count leaves in the screener tree (nodes whose kind is not a composite)
  const COMPOSITE = new Set(["Sequential", "Parallel", "Any", "Not", "Route"]);
  const countLeaves = (t) => {
    if (!t) return 0;
    if (!COMPOSITE.has(t.kind)) return 1;
    const kids = t.children || (t.layer ? [t.layer] : []);
    return kids.reduce((s, c) => s + countLeaves(c), 0);
  };
  const filterCount = countLeaves(node.screener);

  const common = {
    className: "pblock style-" + style + (selected ? " is-selected" : ""),
    onClick,
  };

  const kindLabel = {
    seed: "Source", fwd: "Expander", bwd: "Expander",
    rerank: "Reranker", rsc: "Reranker", sink: "Sink",
  };
  const stepNum = String(index + 1).padStart(2, "0");
  const iconName = iconFor[node.kind] || "square";

  // ─── Variant A: catalog — matches the right-panel block list aesthetic ──
  if (style === "catalog") {
    return (
      <div {...common} data-kind={node.kind}>
        <span className="pb-cat-icon">
          <Icon name={iconName} size={13} />
        </span>
        <div className="pb-cat-body">
          <div className="pb-cat-name">{node.name}</div>
          <div className="pb-cat-hint">
            <span>{node.localId}</span>
            {filterCount > 0 && (
              <>
                <span className="pb-cat-sep">·</span>
                <span>{filterCount} filter{filterCount > 1 ? "s" : ""}</span>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ─── Variant B: bookmark — tall index-slip with step number at top ──────
  if (style === "bookmark") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-bm-top">
          <span className="pb-bm-num">{stepNum}</span>
          <span className="pb-bm-kind">{kindLabel[node.kind]}</span>
        </div>
        <div className="pb-bm-name">{node.name}</div>
        <div className="pb-bm-foot">
          <Icon name={iconName} size={11} />
          <span>{node.localId}</span>
          {filterCount > 0 && <span className="pb-bm-badge">{filterCount}f</span>}
        </div>
      </div>
    );
  }

  // ─── Variant C: rail — inline station with a thin left kind-rule ────────
  if (style === "rail") {
    return (
      <div {...common} data-kind={node.kind}>
        <span className="pb-rail-rule" aria-hidden="true" />
        <span className="pb-rail-icon">
          <Icon name={iconName} size={12} />
        </span>
        <div className="pb-rail-body">
          <div className="pb-rail-row">
            <span className="pb-rail-num">{stepNum}</span>
            <span className="pb-rail-name">{node.name}</span>
          </div>
          <div className="pb-rail-meta">
            <span>{kindLabel[node.kind]}</span>
            <span className="pb-rail-sep">·</span>
            <span>{node.localId}</span>
            {filterCount > 0 && (
              <>
                <span className="pb-rail-sep">·</span>
                <span>{filterCount} filter{filterCount > 1 ? "s" : ""}</span>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ─── Variant D: specimen — scientific specimen card ─────────────────────
  if (style === "specimen") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-sp-caption">
          <span>{kindLabel[node.kind]}</span>
          <span className="pb-sp-num">№ {stepNum}</span>
        </div>
        <div className="pb-sp-name">{node.name}</div>
        <div className="pb-sp-divider" />
        <div className="pb-sp-meta">
          <span className="pb-sp-id">{node.localId}</span>
          {filterCount > 0 && (
            <span className="pb-sp-pill">{filterCount}&nbsp;filter{filterCount > 1 ? "s" : ""}</span>
          )}
        </div>
      </div>
    );
  }

  // ─── Variant E: monogram — dominant step number, thin name line ─────────
  if (style === "monogram") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-mo-num">{stepNum}</div>
        <div className="pb-mo-body">
          <div className="pb-mo-kind">
            <Icon name={iconName} size={10} />
            <span>{kindLabel[node.kind]}</span>
          </div>
          <div className="pb-mo-name">{node.name}</div>
          <div className="pb-mo-meta">
            <span>{node.localId}</span>
            {filterCount > 0 && (
              <>
                <span className="pb-mo-sep">·</span>
                <span>{filterCount}&nbsp;filter{filterCount > 1 ? "s" : ""}</span>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ─── Variant F: ticket — perforated receipt slip ─────────────────────────
  if (style === "ticket") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-tk-head">
          <span>{kindLabel[node.kind]}</span>
          <span className="pb-tk-num">№ {stepNum}</span>
        </div>
        <div className="pb-tk-body">
          <div className="pb-tk-name">{node.name}</div>
        </div>
        <div className="pb-tk-foot">
          <span>{node.localId}</span>
          {filterCount > 0 && <span>{filterCount}f</span>}
        </div>
      </div>
    );
  }

  // ─── Variant G: badge — circular icon badge + inline name ────────────────
  if (style === "badge") {
    return (
      <div {...common} data-kind={node.kind}>
        <span className="pb-bg-icon">
          <Icon name={iconName} size={15} />
          <span className="pb-bg-num">{stepNum}</span>
        </span>
        <div className="pb-bg-body">
          <div className="pb-bg-name">{node.name}</div>
          <div className="pb-bg-meta">
            <span>{kindLabel[node.kind]}</span>
            <span className="pb-bg-sep">·</span>
            <span>{node.localId}</span>
          </div>
        </div>
      </div>
    );
  }

  // ─── Variant H: stamp — notary-style double-bordered stamp ───────────────
  if (style === "stamp") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-st-kind">{kindLabel[node.kind]}</div>
        <div className="pb-st-name">{node.name}</div>
        <div className="pb-st-foot">
          <span>{node.localId}</span>
          <span className="pb-st-num">№ {stepNum}</span>
        </div>
      </div>
    );
  }

  // ─── Variant I: ledger — horizontal ruled row with columns ───────────────
  if (style === "ledger") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-lg-num">{stepNum}</div>
        <div className="pb-lg-body">
          <div className="pb-lg-kind">{kindLabel[node.kind]} · {node.localId}</div>
          <div className="pb-lg-name">{node.name}</div>
        </div>
        <div className="pb-lg-right">{filterCount}f</div>
      </div>
    );
  }

  // ─── Variant J: ribbon — dark kind-ribbon on top ─────────────────────────
  if (style === "ribbon") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-rb-ribbon">
          <span>{kindLabel[node.kind]}</span>
          <span className="pb-rb-num">{stepNum}</span>
        </div>
        <div className="pb-rb-body">
          <div className="pb-rb-name">{node.name}</div>
          <div className="pb-rb-meta">
            <span>{node.localId}</span>
            {filterCount > 0 && (
              <>
                <span className="pb-rb-sep">·</span>
                <span>{filterCount}&nbsp;filter{filterCount > 1 ? "s" : ""}</span>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ─── Variant K: minimal — borderless, underline-on-select ────────────────
  if (style === "minimal") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-mn-head">
          <span className="pb-mn-num">{stepNum}</span>
          <span className="pb-mn-kind">{kindLabel[node.kind]}</span>
        </div>
        <div className="pb-mn-name">{node.name}</div>
        <div className="pb-mn-meta">
          {node.localId}
          {filterCount > 0 && ` · ${filterCount} filter${filterCount > 1 ? "s" : ""}`}
        </div>
      </div>
    );
  }

  // ─── Variant L: blueprint — technical drawing, corner crosshairs ─────────
  if (style === "blueprint") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-bp-top">
          <span>{kindLabel[node.kind]}</span>
          <span className="pb-bp-num">[{stepNum}]</span>
        </div>
        <div className="pb-bp-name">{node.name}</div>
        <div className="pb-bp-foot">
          <span>id={node.localId}</span>
          {filterCount > 0 && <span>filt={filterCount}</span>}
        </div>
      </div>
    );
  }

  // ─── Variant M: numbered — oversize step number dominates ────────────────
  if (style === "numbered") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-nb-num">{stepNum}</div>
        <div className="pb-nb-body">
          <div className="pb-nb-kind">{kindLabel[node.kind]} · {node.localId}</div>
          <div className="pb-nb-name">{node.name}</div>
        </div>
      </div>
    );
  }

  // ─── Variant N: chip — ultra-compact pill ────────────────────────────────
  if (style === "chip") {
    return (
      <div {...common} data-kind={node.kind}>
        <span className="pb-ch-num">{stepNum}</span>
        <span className="pb-ch-icon"><Icon name={iconName} size={12} /></span>
        <span className="pb-ch-name">{node.name}</span>
        {filterCount > 0 && <span className="pb-ch-count">{filterCount}f</span>}
      </div>
    );
  }

  // ─── Variant V2a: card-v2 — clean card, no divider, horizontal header ───
  if (style === "card-v2") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-cv-head">
          <span className="pb-cv-num">{stepNum}</span>
          <span className="pb-cv-kind">{kindLabel[node.kind]}</span>
          {filterCount > 0 && <span className="pb-cv-filt">{filterCount}f</span>}
        </div>
        <div className="pb-cv-name">{node.name}</div>
        <div className="pb-cv-id">{node.localId}</div>
      </div>
    );
  }

  // ─── Variant V2b: chip-v2 — single-row chip, step number left ──────────
  if (style === "chip-v2") {
    return (
      <div {...common} data-kind={node.kind}>
        <span className="pb-cv2-num">{stepNum}</span>
        <span className="pb-cv2-sep" aria-hidden="true" />
        <span className="pb-cv2-name">{node.name}</span>
        <span className="pb-cv2-meta">
          {kindLabel[node.kind]}
          {filterCount > 0 && <span className="pb-cv2-dot">·</span>}
          {filterCount > 0 && <span>{filterCount}f</span>}
        </span>
      </div>
    );
  }

  // ─── Variant V2c: label-v2 — typographic, no card, underline selected ──
  if (style === "label-v2") {
    return (
      <div {...common} data-kind={node.kind}>
        <div className="pb-lb-kind">
          <span className="pb-lb-num">{stepNum}</span>
          {kindLabel[node.kind]}
        </div>
        <div className="pb-lb-name">{node.name}</div>
        <div className="pb-lb-meta">
          {node.localId}
          {filterCount > 0 && ` · ${filterCount} filter${filterCount > 1 ? "s" : ""}`}
        </div>
      </div>
    );
  }

  // Fallback: catalog
  return (
    <div {...common} data-kind={node.kind}>
      <span className="pb-cat-icon">
        <Icon name={iconName} size={13} />
      </span>
      <div className="pb-cat-body">
        <div className="pb-cat-name">{node.name}</div>
        <div className="pb-cat-hint">{node.localId}</div>
      </div>
    </div>
  );
}

// Vertical connector between steps (down-arrow).
function VEdge() {
  return (
    <div className="pipe-vedge" aria-hidden="true">
      <svg viewBox="0 0 14 34" preserveAspectRatio="none">
        <line x1="7" y1="0" x2="7" y2="26" stroke="var(--cc-ink-2)" strokeWidth="1.25" strokeLinecap="round" />
        <path d="M 3 26 L 7 32 L 11 26 Z" fill="var(--cc-ink-2)" />
      </svg>
    </div>
  );
}

// ── Pipeline data model + helpers (shared with app.jsx / build-step-config) ──
// A pipeline is a SEQUENCE (array) of "elements". An element is either a step
// node ({id, kind, name, localId, config, screener}) or a parallel node
// ({id, kind:"parallel", branches:[Seq, Seq]}). Each branch is its own sub-
// sequence (max 2 per parallel); the branches run concurrently and merge
// (union) into whatever element follows the parallel in the parent sequence.
// Recursive: a branch Seq may itself contain parallels — bounded so no more
// than 3 blocks ever sit at the same horizontal level.
const MAX_BRANCHES = 2;   // a parallel may fan out to at most 2 branches
const MAX_COLS = 3;       // at most 3 blocks at the same horizontal level
let __sid = 100;
const STEP_META = {
  seed:   { name: "Seed set",          prefix: "SED" },
  fwd:    { name: "Forward screener",  prefix: "FWD" },
  bwd:    { name: "Backward screener", prefix: "BWD" },
  rerank: { name: "Diversified rerank", prefix: "RRK" },
  rsc:    { name: "Rescreener",        prefix: "RSC" },
};
// Kinds offered by the "add" pickers (seed is always first; sink is auto).
const ADDABLE_STEPS = [
  { kind: "fwd",    label: "Forward screener",  desc: "follow citations",  icon: "arrow-right" },
  { kind: "bwd",    label: "Backward screener", desc: "walk references",   icon: "arrow-left" },
  { kind: "rerank", label: "Diversified rerank", desc: "MMR top-K",        icon: "sliders-horizontal" },
  { kind: "rsc",    label: "Rescreener",        desc: "LLM re-filter",     icon: "filter" },
];

function newStep(kind) {
  const m = STEP_META[kind] || { name: kind, prefix: "STP" };
  const n = ++__sid;
  const localId = m.prefix + "-" + String(n).padStart(2, "0");
  const node = { id: "n" + n, kind, name: m.name, localId, config: {}, screener: null };
  if (kind === "fwd")         node.config = { maxCitations: 100 };
  else if (kind === "bwd")    node.config = {};
  else if (kind === "rerank") node.config = { metric: "citation", targetN: 100, diversity: "walktrap" };
  else if (kind === "seed")   node.config = { query: "", years: "2019-2025", maxSeeds: 42 };
  return node;
}
function newParallel(branches) { return { id: "par" + (++__sid), kind: "parallel", branches }; }

// Find a step node by id anywhere in the sequence tree (recurses into branches).
function findStep(seq, id) {
  for (const el of seq) {
    if (el.id === id) return el;
    if (el.kind === "parallel")
      for (const b of (el.branches || [])) { const r = findStep(b, id); if (r) return r; }
  }
  return null;
}
// Replace the element with `id` via fn(), anywhere in the tree.
function mapStep(seq, id, fn) {
  return seq.map(el => {
    if (el.id === id) return fn(el);
    if (el.kind === "parallel")
      return { ...el, branches: (el.branches || []).map(b => mapStep(b, id, fn)) };
    return el;
  });
}
// Remove the element with `id`; empties collapse (drop empty branch, and a
// parallel left with a single branch is inlined back into the parent sequence).
function removeStep(seq, id) {
  const out = [];
  for (const el of seq) {
    if (el.id === id) continue;
    if (el.kind === "parallel") {
      const branches = (el.branches || []).map(b => removeStep(b, id)).filter(b => b.length > 0);
      if (branches.length === 0) continue;                 // whole parallel gone
      if (branches.length === 1) { out.push(...branches[0]); continue; }  // collapse to serial
      out.push({ ...el, branches });
      continue;
    }
    out.push(el);
  }
  return out;
}
// Insert `el` immediately after the element with `afterId`, wherever it lives.
// This is the single primitive behind "add step" (append to a sequence),
// "add parallel" (append a parallel after the last step) and "merge" (append a
// step after a parallel) — each anchors on the id of the element it sits under.
function insertAfter(seq, afterId, el) {
  const out = [];
  for (const row of seq) {
    let r = row;
    if (row.kind === "parallel" && row.id !== afterId)
      r = { ...row, branches: (row.branches || []).map(b => insertAfter(b, afterId, el)) };
    out.push(r);
    if (row.id === afterId) out.push(el);
  }
  return out;
}
// Add a new branch (a one-element sub-sequence) to the parallel `parId`.
function addParallelBranch(seq, parId, el) {
  return seq.map(row => {
    if (row.kind === "parallel") {
      if (row.id === parId) return { ...row, branches: [...(row.branches || []), [el]] };
      return { ...row, branches: (row.branches || []).map(b => addParallelBranch(b, parId, el)) };
    }
    return row;
  });
}

// ── Column budget (keep at most MAX_COLS blocks at one horizontal level) ──
function seqCols(seq) { let m = 1; for (const el of seq) m = Math.max(m, elCols(el)); return m; }
function elCols(el) {
  if (el.kind !== "parallel") return 1;
  return (el.branches || []).reduce((s, b) => s + seqCols(b), 0);
}
// Would the pipeline still fit if we added a 1-col branch to parallel `parId`?
function _seqColsExtraBranch(seq, parId) { let m = 1; for (const el of seq) m = Math.max(m, _elColsExtraBranch(el, parId)); return m; }
function _elColsExtraBranch(el, parId) {
  if (el.kind !== "parallel") return 1;
  let sum = 0;
  for (const b of (el.branches || [])) sum += _seqColsExtraBranch(b, parId);
  if (el.id === parId) sum += 1;
  return sum;
}
function canAddBranch(pipeline, parNode) {
  if (!parNode || parNode.kind !== "parallel") return false;
  if ((parNode.branches || []).length >= MAX_BRANCHES) return false;
  return _seqColsExtraBranch(pipeline, parNode.id) <= MAX_COLS;
}
// Would the pipeline still fit if a 2-col parallel were appended after `anchorId`?
function _seqColsExtraParallel(seq, anchorId) {
  let m = 1;
  for (const el of seq) {
    m = Math.max(m, _elColsExtraParallel(el, anchorId));
    if (el.id === anchorId) m = Math.max(m, 2);
  }
  return m;
}
function _elColsExtraParallel(el, anchorId) {
  if (el.kind !== "parallel") return 1;
  let sum = 0;
  for (const b of (el.branches || [])) sum += _seqColsExtraParallel(b, anchorId);
  return sum;
}
function canAddParallel(pipeline, anchorId) {
  return _seqColsExtraParallel(pipeline, anchorId) <= MAX_COLS;
}

// Total number of step nodes (pre-order), for the header / block count.
function countSteps(seq) {
  let n = 0;
  for (const el of seq) {
    if (el.kind === "parallel") (el.branches || []).forEach(b => { n += countSteps(b); });
    else n += 1;
  }
  return n;
}
// Map every step id → a pre-order display index (for the "№ NN" caption).
function stepIndexMap(seq) {
  const map = {};
  let k = 0;
  const walk = (s) => { for (const el of s) { if (el.kind === "parallel") (el.branches || []).forEach(walk); else map[el.id] = k++; } };
  walk(seq);
  return map;
}

// Geometry compare — skip re-render when measured fork/join geometry is stable.
function _sameGeom(a, b) {
  if (!a || !b) return false;
  if (Math.abs(a.w - b.w) > 0.5) return false;
  if (a.centers.length !== b.centers.length) return false;
  for (let i = 0; i < a.centers.length; i++) if (Math.abs(a.centers[i] - b.centers[i]) > 0.5) return false;
  return true;
}

// Popover that lists the addable step kinds, anchored to its trigger button.
function StepPicker({ onPick, onClose, anchorRef }) {
  const [pos, setPos] = React.useState(null);
  React.useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  React.useLayoutEffect(() => {
    const compute = () => {
      const el = anchorRef?.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const popW = 240, popH = 230, margin = 8;
      let top = r.bottom + 4;
      if (top + popH > window.innerHeight - margin) top = Math.max(margin, r.top - 4 - popH);
      let left = r.left;
      if (left + popW > window.innerWidth - margin) left = Math.max(margin, window.innerWidth - popW - margin);
      setPos({ top, left });
    };
    compute();
    window.addEventListener("scroll", compute, true);
    window.addEventListener("resize", compute);
    return () => { window.removeEventListener("scroll", compute, true); window.removeEventListener("resize", compute); };
  }, [anchorRef]);
  return (
    <>
      <div className="ft-pop-scrim" onClick={onClose} />
      <div className="ft-pop ft-pop-fixed" style={pos ? { top: pos.top, left: pos.left, width: 240 } : { visibility: "hidden" }}>
        <div className="ft-pop-group">
          <div className="ft-pop-label">Add a step</div>
          <div className="ft-pop-list">
            {ADDABLE_STEPS.map(s => (
              <button key={s.kind} className="ft-pop-item" onClick={() => { onPick(s.kind); onClose(); }}>
                <span className="ft-pop-name"><Icon name={s.icon} size={11} /> {s.label}</span>
                <span className="ft-pop-desc">{s.desc}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

// A button that opens the StepPicker and calls onPick(kind).
function StepAddButton({ onPick, label, className, icon }) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);
  return (
    <>
      <button ref={ref} className={"pipe-add " + (className || "")} onClick={() => setOpen(true)}>
        <Icon name={icon || "plus"} size={12} /> {label}
      </button>
      {open && <StepPicker anchorRef={ref} onPick={onPick} onClose={() => setOpen(false)} />}
    </>
  );
}

// ── Fork / join connectors (measured, so they stay symmetric + centered for
//    any branch widths, matching the classic split→merge diamond) ──
const FORK_H = 34;
function ForkSvg({ geom }) {
  if (!geom) return null;
  const { w, centers } = geom;
  const apexX = w / 2, busY = FORK_H * 0.55, tip = FORK_H, base = FORK_H - 5;
  const minX = Math.min(apexX, ...centers), maxX = Math.max(apexX, ...centers);
  const S = "var(--cc-ink-2)";
  return (
    <svg width={w} height={FORK_H} viewBox={`0 0 ${w} ${FORK_H}`} style={{ display: "block", overflow: "visible" }}>
      <line x1={apexX} y1={0} x2={apexX} y2={busY} stroke={S} strokeWidth={1.25} strokeLinecap="round" />
      {centers.length > 1 && <line x1={minX} y1={busY} x2={maxX} y2={busY} stroke={S} strokeWidth={1.25} strokeLinecap="round" />}
      {centers.map((cx, i) => (
        <g key={i}>
          <line x1={cx} y1={busY} x2={cx} y2={base} stroke={S} strokeWidth={1.25} strokeLinecap="round" />
          <path d={`M ${cx - 4} ${base} L ${cx} ${tip} L ${cx + 4} ${base} Z`} fill={S} />
        </g>
      ))}
    </svg>
  );
}
function JoinSvg({ geom }) {
  if (!geom) return null;
  const { w, centers } = geom;
  const apexX = w / 2, busY = FORK_H * 0.45, tip = FORK_H, base = FORK_H - 5;
  const minX = Math.min(apexX, ...centers), maxX = Math.max(apexX, ...centers);
  const S = "var(--cc-ink-2)";
  return (
    <svg width={w} height={FORK_H} viewBox={`0 0 ${w} ${FORK_H}`} style={{ display: "block", overflow: "visible" }}>
      {centers.map((cx, i) => (
        <line key={i} x1={cx} y1={0} x2={cx} y2={busY} stroke={S} strokeWidth={1.25} strokeLinecap="round" />
      ))}
      {centers.length > 1 && <line x1={minX} y1={busY} x2={maxX} y2={busY} stroke={S} strokeWidth={1.25} strokeLinecap="round" />}
      <line x1={apexX} y1={busY} x2={apexX} y2={base} stroke={S} strokeWidth={1.25} strokeLinecap="round" />
      <path d={`M ${apexX - 4} ${base} L ${apexX} ${tip} L ${apexX + 4} ${base} Z`} fill={S} />
    </svg>
  );
}

// A parallel element: fork band → dashed branch group → join band. Each branch
// is its own recursive SeqView (so branches extend + nest independently). The
// "+ branch" affordance floats to the right (doesn't shift the centered axis).
function ParallelBlock({ node, ctx, depth }) {
  const unitRef = React.useRef(null);
  const branchRefs = React.useRef([]);
  const [geom, setGeom] = React.useState(null);
  const nB = (node.branches || []).length;

  const measure = React.useCallback(() => {
    const unit = unitRef.current;
    if (!unit) return;
    const ur = unit.getBoundingClientRect();
    const centers = [];
    for (let i = 0; i < nB; i++) {
      const el = branchRefs.current[i];
      if (el) { const r = el.getBoundingClientRect(); centers.push(r.left - ur.left + r.width / 2); }
      else centers.push(ur.width / 2);
    }
    const next = { w: ur.width, centers };
    setGeom(prev => (_sameGeom(prev, next) ? prev : next));
  }, [nB]);

  React.useLayoutEffect(() => { measure(); });
  React.useEffect(() => {
    const unit = unitRef.current;
    if (!unit || !window.ResizeObserver) return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(unit);
    return () => ro.disconnect();
  }, [measure]);

  const showBranchAdd = canAddBranch(ctx.pipeline, node);
  return (
    <div className="pipe-parallel-unit" ref={unitRef}>
      <div className="pipe-fork" style={{ height: FORK_H }}><ForkSvg geom={geom} /></div>
      {/* A hidden ghost of equal width on the left balances the "+ branch" on the
          right, so the box stays centred on the axis — in flow, never overlapping. */}
      <div className="pipe-parallel-row">
        {showBranchAdd && (
          <div className="pipe-branch-add pipe-branch-add-ghost" aria-hidden="true">
            <span className="pipe-add pipe-add-branch"><Icon name="git-branch" size={12} /> branch</span>
          </div>
        )}
        <div className="pipe-parallel-box">
          <span className="pipe-parallel-tag">parallel</span>
          <div className="pipe-parallel-branches">
            {(node.branches || []).map((b, i) => (
              <div className="pipe-branch" key={i} ref={el => { branchRefs.current[i] = el; }}>
                <SeqView seq={b} ctx={ctx} depth={depth + 1} />
              </div>
            ))}
          </div>
        </div>
        {showBranchAdd && (
          <div className="pipe-branch-add">
            <StepAddButton label="branch" className="pipe-add-branch" icon="git-branch"
              onPick={(k) => ctx.addBranch(node.id, k)} />
          </div>
        )}
      </div>
      <div className="pipe-join" style={{ height: FORK_H }}><JoinSvg geom={geom} /></div>
    </div>
  );
}

// The bottom-of-sequence add controls. After a step: "add step" (serial) +
// optional "parallel" (fan out). After a parallel: "merge" (converge branches
// into a new trunk step).
function SeqAdder({ last, ctx }) {
  if (!last) return null;
  if (last.kind === "parallel") {
    return (
      <div className="pipe-adder">
        <StepAddButton label="merge branches" className="pipe-add-merge" icon="git-merge"
          onPick={(k) => ctx.appendAfter(last.id, k)} />
      </div>
    );
  }
  const showParallel = last.kind !== "seed" && canAddParallel(ctx.pipeline, last.id);
  return (
    <div className="pipe-adder">
      <VEdge />
      <div className="pipe-adder-row">
        {/* ghost of equal width balances "parallel" so "add step" stays centred */}
        {showParallel && (
          <span className="pipe-add pipe-add-parallel pipe-adder-ghost" aria-hidden="true">
            <Icon name="git-branch" size={12} /> parallel
          </span>
        )}
        <StepAddButton label="add step" className="pipe-add-serial" onPick={(k) => ctx.appendAfter(last.id, k)} />
        {showParallel && (
          <StepAddButton label="parallel" className="pipe-add-parallel" icon="git-branch"
            onPick={(k) => ctx.appendParallel(last.id, k)} />
        )}
      </div>
    </div>
  );
}

// A sequence: elements stacked vertically + a trailing adder. Simple down-edges
// join consecutive steps; parallels carry their own fork/join bands.
function SeqView({ seq, ctx, depth }) {
  const last = seq[seq.length - 1];
  return (
    <div className="pipe-seq">
      {seq.map((el, i) => {
        const prev = seq[i - 1];
        const needEdge = i > 0 && prev.kind !== "parallel" && el.kind !== "parallel";
        return (
          <React.Fragment key={el.id}>
            {needEdge && <VEdge />}
            {el.kind === "parallel" ? (
              <ParallelBlock node={el} ctx={ctx} depth={depth} />
            ) : (
              <div className="pipe-step-row">
                <PipelineBlock
                  node={el} index={ctx.indexOf[el.id] || 0}
                  selected={ctx.selectedId === el.id}
                  onClick={() => ctx.onSelect(el.id)}
                  style={ctx.style}
                />
              </div>
            )}
          </React.Fragment>
        );
      })}
      <SeqAdder last={last} ctx={ctx} />
    </div>
  );
}

function BuildPipeline({ pipeline, selectedId, setSelectedId, blockStyle,
                         onAppendAfter, onAppendParallel, onAddBranch }) {
  const ctx = {
    pipeline,
    selectedId,
    onSelect: setSelectedId,
    style: blockStyle,
    indexOf: stepIndexMap(pipeline),
    appendAfter: onAppendAfter,
    appendParallel: onAppendParallel,
    addBranch: onAddBranch,
  };
  const n = countSteps(pipeline);
  return (
    <section className="pane pane-top">
      <div className="pipe pipe-vert">
        <div className="pipe-header pipe-header-vert">
          <span className="pipe-vtitle">Pipeline</span>
          <span className="pipe-vhint">{n} step{n === 1 ? "" : "s"} · click a step to configure</span>
        </div>
        <div className="pipe-canvas">
          <div className="pipe-inner pipe-inner-vert">
            <div className="pipe-col">
              <SeqView seq={pipeline} ctx={ctx} depth={0} />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

Object.assign(window, {
  BuildPipeline, PipelineBlock,
  newStep, newParallel, findStep, mapStep, removeStep,
  insertAfter, addParallelBranch, countSteps,
});
