/* eslint-disable */
// Section C (Explore mode) — full-page citation-network exploration.
// A thin wrapper over the shared CiteGraph (see cite-graph.jsx): same engine
// and look as the Run network, plus the richer tools — labels, growth
// replay, force-layout options, facet filters (from the Papers panel) and
// the 2-hop "Explore subtree" lens.

function ExploreNetwork({ papers, edges, dataKey, selectedId, onSelect,
                          subtreeId, onClearSubtree, filterHiddenIds, theme }) {
  const [counts, setCounts] = React.useState({ nodes: 0, edges: 0 });

  const cssVar = (name, fallback) => {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fallback;
    } catch (_) { return fallback; }
  };
  const yearColor = (y) => cssVar(`--cc-year-p-${y}`, "#8a8a8a");

  // 2-hop neighbourhood of the subtree anchor, from the edge list.
  const subtreeSet = React.useMemo(() => {
    if (!subtreeId) return null;
    const adj = {};
    for (const e of edges) {
      if (!e || e.source === e.target) continue;
      (adj[e.source] ||= []).push(e.target);
      (adj[e.target] ||= []).push(e.source);
    }
    const seen = new Set([subtreeId]);
    let frontier = [subtreeId];
    for (let hop = 0; hop < 2; hop++) {
      const next = [];
      for (const id of frontier) {
        for (const nb of (adj[id] || [])) {
          if (!seen.has(nb)) { seen.add(nb); next.push(nb); }
        }
      }
      frontier = next;
    }
    return seen;
  }, [subtreeId, edges]);

  // hidden = failing the facet filters ∪ outside the subtree lens
  const hiddenIds = React.useMemo(() => {
    if (!subtreeSet && (!filterHiddenIds || !filterHiddenIds.size)) return null;
    const hidden = new Set(filterHiddenIds || []);
    if (subtreeSet) {
      for (const p of papers) if (!subtreeSet.has(p.id)) hidden.add(p.id);
    }
    return hidden;
  }, [papers, subtreeSet, filterHiddenIds]);

  const visible = hiddenIds ? papers.filter(p => !hiddenIds.has(p.id)).length : papers.length;
  const visibleLinks = React.useMemo(() => {
    if (!hiddenIds) return null;  // fall back to the live graph count
    let n = 0;
    for (const e of edges) if (!hiddenIds.has(e.source) && !hiddenIds.has(e.target)) n++;
    return n;
  }, [edges, hiddenIds]);
  const subtreePaper = subtreeId ? papers.find(p => p.id === subtreeId) : null;

  return (
    <section className="pane">
      <CiteGraph
        papers={papers}
        edges={edges}
        dataKey={dataKey}
        selectedId={selectedId}
        onSelect={onSelect}
        theme={theme}
        labels={true}
        hiddenIds={hiddenIds}
        tools={{ layout: true, layoutOptions: true, replay: true }}
        emptyHint="Nothing to explore yet. Run a pipeline, or pick a finished run in the Papers panel."
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
            <span>2018 → 2025</span>
          </span>
          <span className="net-legend-item">
            <span className="net-hint-txt">size ∝ citations</span>
          </span>
          <span className="net-legend-item">
            <span className="net-hint-txt">drag · scroll · click</span>
          </span>
        </div>

        {subtreePaper && (
          <div className="cg-chip">
            <Icon name="git-branch" size={11} />
            <span className="cg-chip-txt">
              Subtree · {subtreePaper.title.length > 34
                ? subtreePaper.title.slice(0, 32) + "…" : subtreePaper.title}
              {subtreeSet ? ` · ${subtreeSet.size} papers` : ""}
            </span>
            <button className="ph-btn" onClick={onClearSubtree} title="Show full graph">
              <Icon name="x" size={11} />
            </button>
          </div>
        )}

        <div className="net-counter">
          <span>
            <span className="net-counter-num">{Math.min(visible, counts.nodes).toLocaleString()}</span>
            <span className="net-counter-lbl">papers</span>
          </span>
          <span className="net-counter-sep">·</span>
          <span>
            <span className="net-counter-num">{(visibleLinks ?? counts.edges).toLocaleString()}</span>
            <span className="net-counter-lbl">links</span>
          </span>
          {visible !== papers.length && (
            <>
              <span className="net-counter-sep">·</span>
              <span>
                <span className="net-counter-lbl" style={{ marginLeft: 0 }}>
                  of {papers.length.toLocaleString()}
                </span>
              </span>
            </>
          )}
        </div>
      </CiteGraph>
    </section>
  );
}

Object.assign(window, { ExploreNetwork });
