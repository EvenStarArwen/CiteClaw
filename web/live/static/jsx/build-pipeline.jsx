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
// A pipeline is a serial list of "rows". A row is either a regular step node
// ({id, kind, name, localId, config, screener}) or a parallel node
// ({id, kind:"parallel", branches:[node,...]}) whose branches run concurrently.
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
  if (kind === "fwd")         node.config = { depth: 2, maxChildren: 200 };
  else if (kind === "bwd")    node.config = { depth: 2, maxChildren: 100 };
  else if (kind === "rerank") node.config = { lambda: 0.4, targetN: 500 };
  else if (kind === "seed")   node.config = { query: "", years: "2019-2025", maxSeeds: 42 };
  return node;
}
function newParallel(branches) { return { id: "par" + (++__sid), kind: "parallel", branches }; }

function findStep(pipeline, id) {
  for (const row of pipeline) {
    if (row.id === id) return row;
    if (row.kind === "parallel")
      for (const b of (row.branches || [])) if (b.id === id) return b;
  }
  return null;
}
function mapStep(pipeline, id, fn) {
  return pipeline.map(row => {
    if (row.id === id) return fn(row);
    if (row.kind === "parallel")
      return { ...row, branches: (row.branches || []).map(b => b.id === id ? fn(b) : b) };
    return row;
  });
}
function removeStep(pipeline, id) {
  const out = [];
  for (const row of pipeline) {
    if (row.id === id) continue;
    if (row.kind === "parallel") {
      const branches = (row.branches || []).filter(b => b.id !== id);
      if (branches.length === 0) continue;              // drop empty parallel
      if (branches.length === 1) { out.push(branches[0]); continue; }  // collapse to serial
      out.push({ ...row, branches });
      continue;
    }
    out.push(row);
  }
  return out;
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
function StepAddButton({ onPick, label, className }) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);
  return (
    <>
      <button ref={ref} className={"pipe-add " + (className || "")} onClick={() => setOpen(true)}>
        <Icon name="plus" size={12} /> {label}
      </button>
      {open && <StepPicker anchorRef={ref} onPick={onPick} onClose={() => setOpen(false)} />}
    </>
  );
}

function ParallelRow({ row, selectedId, onSelect, style, onAddBranch }) {
  return (
    <div className="pipe-parallel">
      <span className="pipe-parallel-tag">parallel</span>
      <div className="pipe-parallel-branches">
        {(row.branches || []).map(b => (
          <div key={b.id} className="pipe-branch">
            <PipelineBlock
              node={b} index={0}
              selected={selectedId === b.id}
              onClick={() => onSelect(b.id)}
              style={style}
            />
          </div>
        ))}
        <div className="pipe-branch pipe-branch-add">
          <StepAddButton label="branch" className="pipe-add-branch" onPick={(k) => onAddBranch(row.id, k)} />
        </div>
      </div>
    </div>
  );
}

function BuildPipeline({ pipeline, selectedId, setSelectedId, blockStyle,
                         onAddSerial, onWrapParallel, onAddBranch }) {
  const last = pipeline[pipeline.length - 1];
  const canWrap = last && last.kind !== "parallel" && last.kind !== "seed";
  return (
    <section className="pane pane-top">
      <div className="pipe pipe-vert">
        <div className="pipe-header pipe-header-vert">
          <span className="pipe-vtitle">Pipeline</span>
          <span className="pipe-vhint">{pipeline.length} step{pipeline.length === 1 ? "" : "s"} · click a step to configure</span>
        </div>
        <div className="pipe-canvas">
          <div className="pipe-inner pipe-inner-vert">
            <div className="pipe-col">
              {pipeline.map((row, i) => (
                <React.Fragment key={row.id}>
                  {i > 0 && <VEdge />}
                  {row.kind === "parallel" ? (
                    <ParallelRow
                      row={row} selectedId={selectedId} onSelect={setSelectedId}
                      style={blockStyle} onAddBranch={onAddBranch}
                    />
                  ) : (
                    <div className="pipe-node-line">
                      <PipelineBlock
                        node={row} index={i}
                        selected={selectedId === row.id}
                        onClick={() => setSelectedId(row.id)}
                        style={blockStyle}
                      />
                      {i === pipeline.length - 1 && canWrap && (
                        <StepAddButton
                          label="parallel" className="pipe-add-side"
                          onPick={(k) => onWrapParallel(k)}
                        />
                      )}
                    </div>
                  )}
                </React.Fragment>
              ))}
              <VEdge />
              <StepAddButton label="Add step" className="pipe-add-serial" onPick={onAddSerial} />
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
});
