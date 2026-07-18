/* eslint-disable */
// Section C (Explore mode) — full-page network exploration.
// A wrapper over the shared CiteGraph (see cite-graph.jsx): same engine and
// look as the Run network, plus the richer tools — the graph-settings panel
// (layout / appearance / Gephi-style structural filters), stats card, label
// toggle, growth replay, facet filters (from the Papers panel), the 2-hop
// "Explore subtree" lens, and the citation ↔ collaboration network switch.

function ExploreNetwork({ papers, edges, dataKey, kind, selectedId, onSelect,
                          subtreeId, onClearSubtree, filterHiddenIds,
                          onGraphHidden, netMode, onSwitchNet, citeEnabled = true,
                          collabEnabled, collabHint, emptyHint, theme }) {
  const [counts, setCounts] = React.useState({ nodes: 0, edges: 0 });

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
  // (CiteGraph REMOVES these from the simulation, so the layout re-flows)
  const hiddenIds = React.useMemo(() => {
    if (!subtreeSet && (!filterHiddenIds || !filterHiddenIds.size)) return null;
    const hidden = new Set(filterHiddenIds || []);
    if (subtreeSet) {
      for (const p of papers) if (!subtreeSet.has(p.id)) hidden.add(p.id);
    }
    return hidden;
  }, [papers, subtreeSet, filterHiddenIds]);

  const subtreePaper = subtreeId ? papers.find(p => p.id === subtreeId) : null;

  return (
    <section className="pane">
      <CiteGraph
        papers={papers}
        edges={edges}
        dataKey={dataKey}
        kind={kind}
        selectedId={selectedId}
        onSelect={onSelect}
        theme={theme}
        labels={false}
        hiddenIds={hiddenIds}
        onGraphHidden={onGraphHidden}
        tools={{ layout: true, layoutOptions: true, replay: true, labels: true, stats: true }}
        sizeHint={true}
        emptyHint={emptyHint || "Nothing to explore yet. Run a pipeline, or pick a finished run in the Papers panel."}
        onStats={setCounts}
        topLeft={subtreePaper && (
          <div className="cg-chip">
            <Icon name="git-branch" size={11} />
            <span className="cg-chip-txt">
              {kind === "author" ? "Ego network" : "Subtree"} · {subtreePaper.title.length > 34
                ? subtreePaper.title.slice(0, 32) + "…" : subtreePaper.title}
              {subtreeSet ? ` · ${subtreeSet.size}` : ""}
            </span>
            <button className="ph-btn" onClick={onClearSubtree} title="Show full graph">
              <Icon name="x" size={11} />
            </button>
          </div>
        )}
      >
        <div className="net-counter">
          <span>
            <span className="net-counter-num">{counts.nodes.toLocaleString()}</span>
            <span className="net-counter-lbl">{kind === "author" ? "authors" : "papers"}</span>
          </span>
          <span className="net-counter-sep">·</span>
          <span>
            <span className="net-counter-num">{counts.edges.toLocaleString()}</span>
            <span className="net-counter-lbl">links</span>
          </span>
          {counts.nodes !== papers.length && papers.length > 0 && (
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

        <div className="cg-seg" title={collabEnabled ? "" : (collabHint || "")}>
          <button
            className={"cg-seg-btn" + (netMode === "cite" ? " is-on" : "")}
            disabled={!citeEnabled}
            onClick={() => citeEnabled && onSwitchNet("cite")}
            title={citeEnabled ? "" : (collabHint || "")}
          >
            <Icon name="git-fork" size={11} /> Citation
          </button>
          <button
            className={"cg-seg-btn" + (netMode === "collab" ? " is-on" : "")}
            disabled={!collabEnabled}
            onClick={() => collabEnabled && onSwitchNet("collab")}
            title={collabEnabled ? "Author co-authorship network" : (collabHint || "No author data for this source")}
          >
            <Icon name="users" size={11} /> Authors
          </button>
        </div>
      </CiteGraph>
    </section>
  );
}

Object.assign(window, { ExploreNetwork });
