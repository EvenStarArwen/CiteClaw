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

function PipelineEdge() {
  return (
    <div className="pipe-edge">
      <svg viewBox="0 0 40 14" preserveAspectRatio="none">
        <line x1="0" y1="7" x2="32" y2="7"
              stroke="var(--cc-ink-2)" strokeWidth="1.25"
              strokeLinecap="round"/>
        <path d="M 32 3 L 38 7 L 32 11 Z" fill="var(--cc-ink-2)" />
      </svg>
    </div>
  );
}

const PIPELINE_STYLES = [
  { value: "specimen", label: "Specimen" },
  { value: "catalog",  label: "Catalog"  },
  { value: "bookmark", label: "Bookmark" },
  { value: "rail",     label: "Rail"     },
  { value: "monogram", label: "Monogram" },
  { value: "ticket",   label: "Ticket"   },
  { value: "badge",    label: "Badge"    },
  { value: "stamp",    label: "Stamp"    },
  { value: "ledger",   label: "Ledger"   },
  { value: "ribbon",   label: "Ribbon"   },
  { value: "minimal",  label: "Minimal"  },
  { value: "blueprint",label: "Blueprint"},
  { value: "numbered", label: "Numbered" },
  { value: "chip",     label: "Chip"     },
];

function BuildPipeline({ pipeline, selectedId, setSelectedId, blockStyle, setBlockStyle }) {
  return (
    <section className="pane pane-top">
      <div className="pipe">
        <div className="pipe-header">
          <div className="pipe-toolbar">
            <button className="btn-icon btn"><Icon name="minus" size={11}/></button>
            <span className="pipe-zoom">100%</span>
            <button className="btn-icon btn"><Icon name="plus" size={11}/></button>
            <button className="btn-icon btn" title="Fit"><Icon name="maximize-2" size={11}/></button>
          </div>
        </div>
        <div className="pipe-canvas">
          <div className="pipe-inner">
          <div className="pipe-row">
            {pipeline.map((n, i) => (
              <React.Fragment key={n.id}>
                {i > 0 && <PipelineEdge />}
                <PipelineBlock
                  node={n}
                  index={i}
                  selected={selectedId === n.id}
                  onClick={() => setSelectedId(n.id)}
                  style={blockStyle}
                />
              </React.Fragment>
            ))}
          </div>
          </div>
        </div>
      </div>
    </section>
  );
}

Object.assign(window, { BuildPipeline, PipelineBlock });
