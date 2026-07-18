/* eslint-disable */
// Section E (Build mode, right sidebar) — Selected step's configuration.
// Replaces the old Building-Blocks catalog + the lower BuildConfig panel.
//   · Step parameters (depth / max children / λ …) at the top.
//   · The step's FILTER PIPELINE (tree) below, for screener steps.
//   · Clicking a leaf filter opens its full config in a detail view (the same
//     "slide into detail" pattern used for paper abstracts), with a Back link.
// Reuses the tree helpers + FilterTree / FilterParams / BlockParams defined in
// build-config.jsx (all files are concatenated into one script).

function BuildStepConfig({ node, pipeline, onPatchConfig, onUpdateScreener, onRemove, onDuplicate, onPatchStep }) {
  const [selFilterId, setSelFilterId] = React.useState(null);

  React.useEffect(() => { setSelFilterId(null); }, [node ? node.id : null]);

  // Linked-copy state: a synced duplicate shows (and runs with) its ORIGIN's
  // parameters + filters and is locked here; unsyncing snapshots them into
  // this step, which then configures independently.
  const origin = node && node.syncOf ? findStep(pipeline || [], node.syncOf) : null;
  const isSynced = !!(node && node.syncOf && node.synced !== false && origin);
  const view = isSynced ? resolveStepView(node, pipeline) : node;
  const deps = node ? syncDependents(pipeline || [], node.id) : [];
  const unsync = () => {
    const src = resolveStepView(node, pipeline);
    onPatchStep({
      synced: false,
      config: JSON.parse(JSON.stringify(src.config || {})),
      screener: src.screener ? cloneFilterNode(src.screener) : null,
    });
  };
  const resync = () => onPatchStep({ synced: true });

  const selNode = view && view.screener ? findNode(view.screener, selFilterId) : null;
  // Leaves open the filter detail; so does Route (its conditions editor).
  const selFilter = selNode && (!COMPOSITE_KINDS.includes(selNode.kind) || selNode.kind === "Route")
    ? selNode : null;

  React.useEffect(() => {
    if (!selFilter) return;
    const onKey = (e) => { if (e.key === "Escape") setSelFilterId(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selFilter]);

  if (!node) {
    return (
      <aside className="panel panel-right">
        <div className="ph"><span className="ph-title">Configure</span></div>
        <div className="cfg-right-empty">
          <Icon name="mouse-pointer-click" size={18} />
          <span>Select a pipeline step to configure it.</span>
        </div>
      </aside>
    );
  }

  // Seed set → the accepted-papers "cart" (its own panel), not a param form.
  if (node.kind === "seed") return <SeedSetConfig />;

  const isScreener = node.kind === "fwd" || node.kind === "bwd" || node.kind === "rsc"
    || node.kind === "search";

  // --- screener tree ops (mirror the old BuildConfig) ---
  const setScreener = (t) => onUpdateScreener(t);
  const addRoot = (child) => { setScreener(child); setSelFilterId(child.id); };
  const addChild = (parentId, child) => {
    setScreener(mapTree(node.screener, parentId, (p) =>
      p.kind === "Not" ? { ...p, layer: child } : { ...p, children: [...(p.children || []), child] }));
    setSelFilterId(child.id);
  };
  const removeFilter = (id) => {
    if (node.screener && node.screener.id === id) { setScreener(null); setSelFilterId(null); return; }
    setScreener(removeFromTree(node.screener, id));
    if (selFilterId === id) setSelFilterId(null);
  };
  const patchFilter = (id, params) => setScreener(mapTree(node.screener, id, (n) => ({ ...n, params })));
  const patchFilterNode = (id, patch) => setScreener(mapTree(node.screener, id, (n) => ({ ...n, ...patch })));
  const moveFilter = (id, dir) => setScreener(moveInTree(node.screener, id, dir));
  // Drop a picked filter into a Route branch slot (a route row's pass_to,
  // or "__else" for the else branch), then open it for configuration.
  const addToRoute = (routeNodeId, slotId, child) => {
    setScreener(mapTree(node.screener, routeNodeId, (n) => slotId === "__else"
      ? { ...n, else: child }
      : { ...n, routes: (n.routes || []).map(r => (r.id === slotId ? { ...r, pass_to: child } : r)) }));
    setSelFilterId(child.id);
  };

  // --- FILTER DETAIL (full-panel config for one leaf filter) ---
  // Synced copies never drill in: their filters belong to the origin.
  if (selFilter && !isSynced) {
    return (
      <aside className="panel panel-right">
        <div className="ph">
          <span className="ph-title">Filter</span>
          <span className="ph-count">{node.localId}</span>
        </div>
        <div className="seed-detail">
          <button className="seed-detail-back" onClick={() => setSelFilterId(null)} title="Back (Esc)">
            <Icon name="arrow-left" size={13} /> Back to filters
          </button>
          <div className="seed-detail-body">
            <div className="cfg-detail-kind">
              {selFilter.kind === "Route" ? "Route" : selFilter.kind.replace("Filter", "") + " filter"}
            </div>
            <div className="cfg-detail-summary">{filterSummary(selFilter)}</div>
            <div className="config-inspect-grid cfg-detail-grid">
              <FilterParams node={selFilter}
                onPatch={(params) => patchFilter(selFilter.id, params)}
                onPatchNode={(patch) => patchFilterNode(selFilter.id, patch)} />
            </div>
          </div>
          <div className="seed-detail-foot">
            <button className="btn btn-ghost" style={{ flex: "1 1 auto", justifyContent: "center" }}
              onClick={() => removeFilter(selFilter.id)}>
              <Icon name="trash-2" size={12} /> Remove filter
            </button>
          </div>
        </div>
      </aside>
    );
  }

  // --- STEP CONFIG (params + filter tree) ---
  return (
    <aside className="panel panel-right">
      <div className="ph">
        <span className="ph-title">Configure step</span>
        <span className="ph-count">{node.localId}</span>
      </div>
      <div className="pb-scroll cfg-right-body">
        <div className="cfg-step-head">
          <div className="cfg-step-headmain">
            <div className="cfg-step-title">{node.name}</div>
            <div className="cfg-step-sub">{node.kind} · {node.localId}</div>
          </div>
          {node.kind !== "seed" && (
            <>
              <button className="ph-btn cfg-step-dup" onClick={onDuplicate}
                title="Duplicate this step — an identical copy (filters included) is inserted right after it">
                <Icon name="copy" size={14} />
              </button>
              <button className="ph-btn cfg-step-remove" onClick={onRemove} title="Remove step">
                <Icon name="trash-2" size={14} />
              </button>
            </>
          )}
        </div>

        {isSynced && (
          <div className="cfg-note cfg-sync-note">
            <Icon name="link-2" size={13} />
            <span>
              <strong>Synced to {origin.name} · {origin.localId}.</strong> This
              linked copy always runs with {origin.localId}'s parameters and
              filters, so it's locked here — edit {origin.localId} to change
              every linked copy at once, or unsync to configure this one on
              its own.
            </span>
            <button className="cfg-sync-btn" onClick={unsync}
              title="Make this copy independent — it keeps a snapshot of the current configuration and becomes editable">
              Unsync
            </button>
          </div>
        )}
        {!isSynced && node.syncOf && origin && (
          <div className="cfg-note cfg-sync-note is-off">
            <Icon name="unlink" size={13} />
            <span>Independent copy of {origin.localId} — configured on its own.</span>
            <button className="cfg-sync-btn" onClick={resync}
              title={"Sync back to " + origin.localId + " — this copy's own configuration is discarded and it mirrors " + origin.localId + " again"}>
              Re-sync
            </button>
          </div>
        )}
        {deps.length > 0 && (
          <div className="cfg-note cfg-sync-note">
            <Icon name="link-2" size={13} />
            <span>
              <strong>{deps.map(d => d.localId).join(", ")}</strong>
              {deps.length === 1 ? " is a linked copy" : " are linked copies"} of
              this step — edits here apply there too.
            </span>
          </div>
        )}

        <div className={isSynced ? "cfg-locked" : undefined}>
          <div className="cfg-section">
            <div className="cfg-section-head">Step parameters</div>
            <div className="config-inspect-grid cfg-step-params">
              <BlockParams node={view} onPatchConfig={onPatchConfig} />
            </div>
          </div>

          {isScreener && (
            <div className="cfg-section">
              <div className="cfg-section-head">
                Filter pipeline
                <span className="cfg-section-hint">
                  {isSynced ? "locked · synced to " + origin.localId : "click a filter to configure"}
                </span>
              </div>
              <div className="cfg-tree-wrap">
                <FilterTree
                  screener={view.screener}
                  selectedId={selFilterId}
                  onSelect={setSelFilterId}
                  onAddRoot={addRoot}
                  onAddChild={addChild}
                  onRemove={removeFilter}
                  onMove={moveFilter}
                  onAddToRoute={addToRoute}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}

Object.assign(window, { BuildStepConfig });
