/* eslint-disable */
// Section C (Run mode) — live network visualization.
// Same engine as the Exploration page (shared CiteGraph: graphology +
// ForceAtlas2 worker + sigma — the design's f5 reference stack) with the same
// parametric sizing, so the label toggle + graph-settings popover (node size /
// curve, edge width, palette, filters) work here exactly like on Explore.
// New papers stream in incrementally with the grow animation while the FA2
// worker keeps untangling off the main thread.
// Hover shows title · authors · year · venue · cites.

function RunNetwork({ selectedPaperId, onSelectPaper, onHoverPaper, theme }) {
  const network = useLive("network");
  const accepted = useLive("accepted");
  const running = useLive("running");
  const [counts, setCounts] = React.useState({ nodes: 0, edges: 0 });

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
        tools={{ labels: true, layoutOptions: true }}
        legendSeed="always"
        emptyHint={running
          ? "Waiting for the first accepted papers…"
          : "The citation graph grows here during a run."}
        onStats={setCounts}
      >
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
