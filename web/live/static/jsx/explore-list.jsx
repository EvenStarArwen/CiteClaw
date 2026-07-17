/* eslint-disable */
// Section B (Explore mode) — the paper list, data-source picker and facet
// filters. Mirrors the f5 reference's sidebar (sortable rows driving the
// graph selection) plus the richer tools: a source select swapping between
// the live session and finished runs on disk, and a Filters bar (year
// window / min citations / seeds only — same UI as the seed search) that
// hides papers in both this list and the graph.

const XP_LIST_CAP = 400;
const XP_EMPTY_FILTERS = { yearMin: "", yearMax: "", minCites: "", seedsOnly: false };

// Shared with app.jsx (graph hiddenIds) so list + graph always agree.
function xpFilterPredicate(f) {
  const yMin = Number(f.yearMin) || 0;
  const yMax = Number(f.yearMax) || 0;
  const cMin = Number(f.minCites) || 0;
  return (p) => {
    if (f.seedsOnly && !p.seed) return false;
    if (yMin && (p.year || 0) < yMin) return false;
    if (yMax && (p.year || 0) > yMax) return false;
    if (cMin && (p.cites || 0) < cMin) return false;
    return true;
  };
}
function xpFilterCount(f) {
  return (f.yearMin ? 1 : 0) + (f.yearMax ? 1 : 0) + (f.minCites ? 1 : 0) + (f.seedsOnly ? 1 : 0);
}

function ExploreList({ papers, selectedId, onSelect, sort, setSort,
                       filters, setFilters, explore, onPickSource }) {
  const scrollRef = React.useRef(null);
  const [showFilters, setShowFilters] = React.useState(false);
  const nFilters = xpFilterCount(filters);

  const visible = React.useMemo(
    () => papers.filter(xpFilterPredicate(filters)), [papers, filters]);

  const sorted = React.useMemo(() => {
    const copy = [...visible];
    if (sort === "citations") copy.sort((a, b) => (b.cites || 0) - (a.cites || 0));
    else if (sort === "year") copy.sort((a, b) => (b.year || 0) - (a.year || 0));
    else if (sort === "title") copy.sort((a, b) => String(a.title).localeCompare(String(b.title)));
    return copy.slice(0, XP_LIST_CAP);
  }, [visible, sort]);

  React.useEffect(() => {
    if (!selectedId || !scrollRef.current) return;
    const el = scrollRef.current.querySelector(`[data-paper-id="${CSS.escape(selectedId)}"]`);
    if (!el) return;
    const c = scrollRef.current;
    const eTop = el.offsetTop - c.offsetTop;
    const eBot = eTop + el.offsetHeight;
    if (eTop < c.scrollTop || eBot > c.scrollTop + c.clientHeight) {
      c.scrollTo({ top: eTop - 60, behavior: "smooth" });
    }
  }, [selectedId]);

  const srcValue = explore.source === "run" ? explore.runPath : "live";
  const liveCount = (LIVE.get("accepted") || []).length;
  const patchF = (patch) => setFilters({ ...filters, ...patch });

  return (
    <aside className="panel panel-left">
      <div className="ph">
        <span className="ph-title">Papers</span>
        <span className="ph-count">
          <span style={{ color: "var(--cc-ink-1)", fontWeight: 600 }}>{visible.length.toLocaleString()}</span>
          <span>{visible.length !== papers.length ? ` of ${papers.length.toLocaleString()}` : " in graph"}</span>
        </span>
      </div>

      <div className="xp-controls">
        <select
          className="accepted-sort"
          value={srcValue}
          onChange={e => onPickSource(e.target.value)}
          title="Data source"
        >
          <option value="live">Current session · {liveCount.toLocaleString()} papers</option>
          {explore.runs.map(r => (
            <option key={r.path} value={r.path}>
              {r.label} · {r.papers.toLocaleString()} papers · {r.modified}
            </option>
          ))}
        </select>
        <button className="ph-btn" onClick={() => refreshExploreRuns()} title="Rescan runs on disk">
          <Icon name="refresh-cw" size={12} />
        </button>
      </div>
      <div className="xp-controls">
        <select className="accepted-sort" value={sort} onChange={e => setSort(e.target.value)}>
          <option value="citations">Sort by: Citations</option>
          <option value="year">Sort by: Year</option>
          <option value="title">Sort by: Title</option>
        </select>
      </div>

      <div className="seed-filterbar">
        <button className={"seed-filter-toggle" + (showFilters ? " is-open" : "")}
                onClick={() => setShowFilters(v => !v)}>
          <Icon name="sliders-horizontal" size={12} /> Filters{nFilters ? ` · ${nFilters}` : ""}
          <Icon name={showFilters ? "chevron-up" : "chevron-down"} size={11} />
        </button>
        {nFilters > 0 && (
          <button className="ph-btn" onClick={() => setFilters({ ...XP_EMPTY_FILTERS })}
                  title="Clear filters">
            <Icon name="x" size={11} />
          </button>
        )}
      </div>
      {showFilters && (
        <div className="seed-filters">
          <label className="seed-filter-row">
            <span className="seed-filter-k">Year ≥</span>
            <input type="number" placeholder="any" value={filters.yearMin}
                   onChange={e => patchF({ yearMin: e.target.value })} />
          </label>
          <label className="seed-filter-row">
            <span className="seed-filter-k">Year ≤</span>
            <input type="number" placeholder="any" value={filters.yearMax}
                   onChange={e => patchF({ yearMax: e.target.value })} />
          </label>
          <label className="seed-filter-row">
            <span className="seed-filter-k">Min citations</span>
            <input type="number" min="0" step="10" placeholder="0" value={filters.minCites}
                   onChange={e => patchF({ minCites: e.target.value })} />
          </label>
          <label className="seed-filter-row">
            <span className="seed-filter-k">Seeds only</span>
            <input type="checkbox" checked={filters.seedsOnly}
                   onChange={e => patchF({ seedsOnly: e.target.checked })} />
          </label>
        </div>
      )}

      <div className="xp-scroll" ref={scrollRef}>
        {explore.loading && <div className="xp-note">Loading run…</div>}
        {explore.error && <div className="xp-note xp-note-err">{explore.error}</div>}
        {!explore.loading && !visible.length && !explore.error && (
          <div className="xp-note">
            {papers.length
              ? "No papers match the filters."
              : explore.source === "live"
                ? "No papers in this session yet. Run a pipeline, or pick a finished run above."
                : "This run has no accepted papers."}
          </div>
        )}
        {sorted.map(p => (
          <div
            key={p.id}
            data-paper-id={p.id}
            className={"xp-item" + (selectedId === p.id ? " is-selected" : "")}
            onClick={() => onSelect(p.id === selectedId ? null : p.id)}
          >
            <div className="xp-item-title">
              {p.seed && <span className="xp-seed-dot" title="Seed paper" />}
              {p.title}
            </div>
            <div className="xp-item-meta">
              <span className="xp-item-sub">
                {[p.authors, p.year || null].filter(Boolean).join(" · ")}
              </span>
              <span className="xp-item-cites">{fmtK(p.cites || 0)}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="acc-foot">
        <span className="acc-foot-stream">
          {explore.source === "run"
            ? <>run · <span className="cc-mono-xs">{explore.runPath}</span></>
            : "live session"}
        </span>
        <span>{Math.min(XP_LIST_CAP, visible.length)} of {visible.length.toLocaleString()}</span>
      </div>
    </aside>
  );
}

Object.assign(window, { ExploreList, xpFilterPredicate, XP_EMPTY_FILTERS });
