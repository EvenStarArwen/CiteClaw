/* eslint-disable */
// Section B (Build mode) — Seed search + the accepted-seeds "cart".
//
// LEFT sidebar (BuildSeeds): a search box + a collapsible Filters bar
//   (year window, min citations) → candidate results. Starring a paper
//   ACCEPTS it as a seed, moving it out of this list into the seed-set block.
// RIGHT sidebar (SeedSetConfig, shown when the Seed set block is selected):
//   the accepted papers as cards; removing one returns it to the search list.
// Both share SeedAbstractDetail — click a card to read the full abstract.

const YEAR_PRESETS = ["Any", "2015-2025", "2019-2025", "2022-2025"];

// Fetch an OpenAlex abstract fallback into the store for a paper with none.
function _loadSeedAbstract(p) {
  return fetchSeedAbstract(p).then(res => {
    LIVE.set({ seeds: LIVE.get("seeds").map(s => s.id === p.id
      ? { ...s, abstract: (res && res.abstract) || "", _absTried: true, _absSource: res && res.source }
      : s) });
  }).catch(() => {});
}

function _s2Url(p) {
  const ext = p.externalIds || {};
  if (ext.DOI) return "https://doi.org/" + ext.DOI;
  if (ext.ArXiv) return "https://arxiv.org/abs/" + ext.ArXiv;
  return "https://www.semanticscholar.org/paper/" + p.id;
}

// Accept / remove a paper (shared by both panels; mutates the store).
function acceptSeed(id) { LIVE.set({ seeds: LIVE.get("seeds").map(p => p.id === id ? { ...p, starred: true } : p) }); }
function removeSeed(id) { LIVE.set({ seeds: LIVE.get("seeds").map(p => p.id === id ? { ...p, starred: false } : p) }); }

// Full-panel abstract view (shared by the search list + the seed cart).
function SeedAbstractDetail({ paper, onBack, footer }) {
  const [absLoading, setAbsLoading] = React.useState(false);
  React.useEffect(() => {
    if (!paper || paper.abstract || paper._absTried) return;
    let cancelled = false;
    setAbsLoading(true);
    _loadSeedAbstract(paper).finally(() => { if (!cancelled) setAbsLoading(false); });
    return () => { cancelled = true; };
  }, [paper ? paper.id : null]);
  React.useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onBack(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onBack]);
  if (!paper) return null;
  return (
    <div className="seed-detail">
      <button className="seed-detail-back" onClick={onBack} title="Back (Esc)">
        <Icon name="arrow-left" size={13} /> Back
      </button>
      <div className="seed-detail-body">
        <div className="acc-detail-title">{paper.title}</div>
        <div className="acc-detail-meta">{paper.authors || "Unknown authors"}</div>
        <div className="acc-detail-grid">
          <div><div className="acc-detail-k">Year</div><div className="acc-detail-v">{paper.year || "—"}</div></div>
          <div><div className="acc-detail-k">Citations</div><div className="acc-detail-v">{(paper.cites || 0).toLocaleString()}</div></div>
          <div style={{ gridColumn: "1 / 3" }}>
            <div className="acc-detail-k">Venue</div>
            <div className="acc-detail-v" style={{ whiteSpace: "normal" }}>{paper.venue || "—"}</div>
          </div>
        </div>
        <div>
          <div className="seed-abstract-k">
            Abstract
            {paper.abstract && paper._absSource && <span className="seed-abstract-src"> · via {paper._absSource}</span>}
          </div>
          <div className={"seed-abstract" + (paper.abstract ? "" : " is-empty")}>
            {paper.abstract
              ? paper.abstract
              : absLoading
                ? "Looking for an abstract…"
                : "No abstract available from Semantic Scholar or OpenAlex for this paper."}
          </div>
        </div>
      </div>
      <div className="seed-detail-foot">{footer}</div>
    </div>
  );
}

// One paper row. mode "candidate" → star to accept; mode "accepted" → × to remove.
function SeedCard({ paper, onOpen, onAction, mode }) {
  const accepted = mode === "accepted";
  return (
    <div className="seed-card" onClick={() => onOpen(paper.id)} title="Click to read the abstract">
      <div className="seed-card-head">
        <button
          className={accepted ? "seed-remove-btn" : "seed-star-btn"}
          onClick={e => { e.stopPropagation(); onAction(paper.id); }}
          title={accepted ? "Remove from seeds" : "Star as seed"}
          aria-label={accepted ? "Remove from seeds" : "Star as seed"}
        >
          <span className={accepted ? "seed-remove" : "seed-star"}>
            <Icon name={accepted ? "x" : "star"} size={13} />
          </span>
        </button>
        <span className="seed-title">{paper.title}</span>
      </div>
      <div className="seed-meta">
        <span>{paper.authors}</span>
        <span className="seed-meta-sep">·</span>
        <span>{paper.year}</span>
        <span className="seed-meta-sep">·</span>
        <span>{paper.venue}</span>
        <span className="seed-meta-sep">·</span>
        <span className="seed-cites">{(paper.cites || 0).toLocaleString()} cites</span>
      </div>
    </div>
  );
}

function BuildSeeds({ onSelectSeed }) {
  const seeds = useLive("seeds");
  const [query, setQuery] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState(null);
  const [detailId, setDetailId] = React.useState(null);
  const [showFilters, setShowFilters] = React.useState(false);
  const [year, setYear] = React.useState("Any");
  const [minCites, setMinCites] = React.useState(0);
  // Pagination over S2's relevance search: total matches, the next page's
  // offset (null = exhausted), and how many results this query has fetched.
  const [page, setPage] = React.useState({ total: 0, next: null, fetched: 0 });

  const candidates = seeds.filter(p => !p.starred);
  const acceptedCount = seeds.length - candidates.length;
  const detailPaper = detailId ? seeds.find(p => p.id === detailId) : null;

  const runSearch = async (q, f) => {
    q = (q || "").trim();
    if (!q) return;
    setBusy(true); setErr(null);
    try {
      const res = await searchSeeds(q, f || { year, minCites });
      const results = res.items || [];
      const starredMap = {};
      seeds.forEach(p => { if (p.starred) starredMap[p.id] = p; });
      const merged = results.map(r => ({ ...r, starred: !!starredMap[r.id] }));
      const seen = new Set(results.map(r => r.id));
      Object.values(starredMap).forEach(p => { if (!seen.has(p.id)) merged.push(p); });
      LIVE.set({ seeds: merged, searchQuery: q });
      setPage({ total: res.total || 0, next: res.next, fetched: results.length });
    } catch (e) {
      setErr(e.message || "search failed");
    }
    setBusy(false);
  };

  const loadMore = async () => {
    if (busy || page.next == null || !query.trim()) return;
    setBusy(true); setErr(null);
    try {
      const res = await searchSeeds(query, { year, minCites }, page.next);
      const items = res.items || [];
      const cur = LIVE.get("seeds");
      const have = new Set(cur.map(p => p.id));
      const fresh = items.filter(r => !have.has(r.id)).map(r => ({ ...r, starred: false }));
      LIVE.set({ seeds: cur.concat(fresh) });
      setPage(pg => ({ total: res.total || pg.total, next: res.next, fetched: pg.fetched + items.length }));
    } catch (e) {
      setErr(e.message || "search failed");
    }
    setBusy(false);
  };

  // Change a filter → re-run the search immediately (only if there's a query).
  const applyFilter = (patch) => {
    if (patch.year !== undefined) setYear(patch.year);
    if (patch.minCites !== undefined) setMinCites(patch.minCites);
    const next = { year, minCites, ...patch };
    if (query.trim()) runSearch(query, next);
  };

  if (detailPaper) {
    return (
      <aside className="panel panel-left">
        <div className="ph"><span className="ph-title">Seeds</span><span className="ph-count">abstract</span></div>
        <SeedAbstractDetail
          paper={detailPaper}
          onBack={() => setDetailId(null)}
          footer={<>
            <button className="btn btn-primary" onClick={() => { acceptSeed(detailPaper.id); setDetailId(null); }}>
              <Icon name="star" size={13} /> Star as seed
            </button>
            <a className="btn btn-ghost" href={_s2Url(detailPaper)} target="_blank" rel="noreferrer" title="Open source">
              <Icon name="external-link" size={12} />
            </a>
          </>}
        />
      </aside>
    );
  }

  return (
    <aside className="panel panel-left">
      <div className="ph">
        <span className="ph-title">Seeds</span>
        <span className="ph-count" title={page.total > page.fetched
          ? `Semantic Scholar matched ${page.total.toLocaleString()} papers — ${page.fetched} loaded so far`
          : undefined}>
          {busy && !candidates.length ? "…"
            : page.total > page.fetched
              ? `${candidates.length} of ${page.total.toLocaleString()}`
              : candidates.length + " results"}
        </span>
      </div>

      <div className="searchbox">
        <Icon name="search" size={12} />
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") runSearch(query); }}
          placeholder="Search Semantic Scholar — press Enter"
        />
        {query && (
          <button className="ph-btn" onClick={() => setQuery("")} title="Clear"><Icon name="x" size={11} /></button>
        )}
      </div>

      <div className="seed-filterbar">
        <button className={"seed-filter-toggle" + (showFilters ? " is-open" : "")} onClick={() => setShowFilters(v => !v)}>
          <Icon name="sliders-horizontal" size={12} /> Filters
          <Icon name={showFilters ? "chevron-up" : "chevron-down"} size={11} />
        </button>
        <button className="seed-accepted-badge" onClick={onSelectSeed} title="Show the seed set (right panel)">
          <span className="seeds-counter-n">{acceptedCount}</span>
          <Icon name="star" size={11} style={{ color: "var(--cc-warning)" }} /> accepted
        </button>
      </div>

      {showFilters && (
        <div className="seed-filters">
          <label className="seed-filter-row">
            <span className="seed-filter-k">Year</span>
            <select value={year} onChange={e => applyFilter({ year: e.target.value })}>
              {YEAR_PRESETS.map(y => <option key={y} value={y}>{y === "Any" ? "Any year" : y}</option>)}
            </select>
          </label>
          <label className="seed-filter-row">
            <span className="seed-filter-k">Min citations</span>
            <input
              type="number" min="0" step="10" value={minCites}
              onChange={e => setMinCites(Math.max(0, +e.target.value || 0))}
              onBlur={e => applyFilter({ minCites: Math.max(0, +e.target.value || 0) })}
              onKeyDown={e => { if (e.key === "Enter") applyFilter({ minCites: Math.max(0, +e.target.value || 0) }); }}
            />
          </label>
        </div>
      )}

      <div className="pb-scroll">
        {err && <div className="seeds-counter" style={{ color: "var(--cc-danger)" }}>{err}</div>}
        {candidates.length === 0 && !busy && (
          <div className="seeds-empty">
            {query.trim()
              ? "No results. Try a different query or loosen the filters."
              : "Search Semantic Scholar to find seed papers."}
          </div>
        )}
        <div className="seeds-list">
          {candidates.map(p => (
            <SeedCard key={p.id} paper={p} mode="candidate" onOpen={setDetailId} onAction={acceptSeed} />
          ))}
        </div>
        {page.next != null && candidates.length > 0 && (
          <button className="seeds-more" onClick={loadMore} disabled={busy}>
            {busy ? "Loading…" : <><Icon name="chevron-down" size={12} /> Load {Math.min(100, page.total - page.fetched)} more
              <span className="seeds-more-total"> · {page.total.toLocaleString()} matched</span></>}
          </button>
        )}
        {page.next == null && page.fetched >= 900 && page.total > page.fetched && (
          <div className="seeds-cap-note">
            Semantic Scholar serves at most the first 1,000 matches per query —
            refine the query or tighten the filters to reach the rest.
          </div>
        )}
      </div>
    </aside>
  );
}

// Right sidebar when the Seed set block is selected: the accepted papers as
// cards. Removing one returns it to the search list on the left.
function SeedSetConfig() {
  const seeds = useLive("seeds");
  const [detailId, setDetailId] = React.useState(null);
  const accepted = seeds.filter(p => p.starred);
  const detailPaper = detailId ? seeds.find(p => p.id === detailId) : null;

  if (detailPaper) {
    return (
      <aside className="panel panel-right">
        <div className="ph"><span className="ph-title">Seed paper</span><span className="ph-count">SED-01</span></div>
        <SeedAbstractDetail
          paper={detailPaper}
          onBack={() => setDetailId(null)}
          footer={<>
            <button className="btn btn-ghost" style={{ flex: "1 1 auto", justifyContent: "center" }}
              onClick={() => { removeSeed(detailPaper.id); setDetailId(null); }}>
              <Icon name="x" size={13} /> Remove from seeds
            </button>
            <a className="btn btn-ghost" href={_s2Url(detailPaper)} target="_blank" rel="noreferrer" title="Open source">
              <Icon name="external-link" size={12} />
            </a>
          </>}
        />
      </aside>
    );
  }

  return (
    <aside className="panel panel-right">
      <div className="ph">
        <span className="ph-title">Seed set</span>
        <span className="ph-count">{accepted.length} paper{accepted.length === 1 ? "" : "s"}</span>
      </div>
      {accepted.length === 0 ? (
        <div className="cfg-right-empty">
          <Icon name="star" size={18} />
          <span>No seed papers yet. Search on the left and star papers (☆) to add them here.</span>
        </div>
      ) : (
        <div className="pb-scroll">
          <div className="seeds-list seed-cart">
            {accepted.map(p => (
              <SeedCard key={p.id} paper={p} mode="accepted" onOpen={setDetailId} onAction={removeSeed} />
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}

Object.assign(window, { BuildSeeds, SeedSetConfig });
