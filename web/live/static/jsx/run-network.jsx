/* eslint-disable */
// Section C (Run mode) — live network visualization.
// Same engine as the Exploration page (shared CiteGraph: graphology +
// ForceAtlas2 worker + sigma — the design's f5 reference stack), kept to the
// run look: dot grid, year-ramp nodes, plum seed halo, no on-graph labels.
// New papers stream in incrementally with the grow animation while the FA2
// worker keeps untangling off the main thread — no full re-layout per
// snapshot. Hover shows title · authors · year · venue · cites.

function RunNetwork({ selectedPaperId, onSelectPaper, onHoverPaper, theme }) {
  const network = useLive("network");
  const accepted = useLive("accepted");
  const running = useLive("running");
  const [counts, setCounts] = React.useState({ nodes: 0, edges: 0 });

  const cssVar = (name, fallback) => {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fallback;
    } catch (_) { return fallback; }
  };
  const yearColor = (y) => cssVar(`--cc-year-p-${y}`, "#8a8a8a");

  // Normalize the live stores into the shared graph shape (id-based edges).
  const data = React.useMemo(() => exploreFromLive(),
    [network.version, accepted.length]);  // eslint-disable-line

  // Real acceptance rate over the last minute (was a hardcoded demo number).
  const perMin = React.useMemo(() => {
    const cut = Date.now() - 60000;
    return accepted.filter(p => (p.addedAt || 0) >= cut).length;
  }, [accepted, network.version]);  // eslint-disable-line

  return (
    <section className="pane pane-top">
      <CiteGraph
        papers={data.papers}
        edges={data.edges}
        dataKey="live"
        selectedId={selectedPaperId}
        onSelect={onSelectPaper}
        onHover={onHoverPaper}
        theme={theme}
        labels={false}
        emptyHint={running
          ? "Waiting for the first accepted papers…"
          : "The citation graph grows here during a run."}
        onStats={setCounts}
      >
        <div className="net-legend">
          <span className="net-legend-item">
            <span className="net-dot seed" />
            <span>Seed</span>
          </span>
          <span className="net-legend-item">
            <span className="net-ramp">
              {[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025].map(y => (
                <span key={y + theme} className="net-ramp-step" style={{ background: yearColor(y) }} />
              ))}
            </span>
            <span>{cgYearDomain(data.papers).min} → {cgYearDomain(data.papers).max}</span>
          </span>
          <span className="net-legend-item">
            <span className="net-hint-txt">drag · scroll · click</span>
          </span>
        </div>

        <div className="net-counter">
          <span>
            <span className="net-counter-num">{counts.nodes}</span>
            <span className="net-counter-lbl">nodes</span>
          </span>
          <span className="net-counter-sep">·</span>
          <span>
            <span className="net-counter-num">{counts.edges}</span>
            <span className="net-counter-lbl">edges</span>
          </span>
          <span className="net-counter-sep">·</span>
          <span>
            <span className="net-counter-num">+{perMin}</span>
            <span className="net-counter-lbl">/ min</span>
          </span>
        </div>
      </CiteGraph>
    </section>
  );
}

Object.assign(window, { RunNetwork });
