/* eslint-disable */
// Section B (Build mode) — Seed search + the accepted-seeds "cart".
//
// LEFT sidebar (BuildSeeds): a search box + a collapsible Filters bar
//   (year window, min citations) → candidate results. Starring a paper
//   ACCEPTS it as a seed, moving it out of this list into the seed-set block.
// RIGHT sidebar (SeedSetConfig, shown when the Seed set block is selected):
//   the accepted papers as cards; removing one returns it to the search list.
// Both share SeedAbstractDetail — click a card to read the full abstract.

// ---- local filter helpers ---------------------------------------------
// The loaded search results ARE the database: every filter below runs
// client-side over them — no extra Semantic Scholar calls.

// Parse a boolean keyword expression — same syntax as the pipeline's
// keyword filters: AND OR NOT ( ) "quoted phrases" (& | ! also accepted)
// and trailing-* prefix wildcards. Returns an AST or null when invalid/empty.
function _kwParse(expr) {
  const toks = [];
  const re = /"([^"]*)"(\*?)|(and|or|not)(?![\w-])|([A-Za-z0-9_À-￿][A-Za-z0-9_À-￿-]*\*?)|([&|!()])|(\s+)|(.)/gi;
  let m;
  while ((m = re.exec(expr)) !== null) {
    if (m[6] != null) continue;                           // whitespace
    if (m[7] != null) return null;                        // stray char (e.g. a lone *)
    if (m[3] != null) {                                   // AND / OR / NOT word -> operator
      const g = m[3].toLowerCase();
      toks.push({ t: g === "and" ? "&" : g === "or" ? "|" : "!" });
      continue;
    }
    if (m[5] != null) { toks.push({ t: m[5] }); continue; }   // glyph operator
    const raw = m[1] != null ? (m[1] + (m[2] || "")) : m[4];   // quoted (+*) or bare word
    const v = (raw || "").trim().toLowerCase();
    if (!v) return null;                                  // empty quotes
    toks.push({ t: "term", v });
  }
  if (!toks.length) return null;
  let i = 0;
  const peek = () => toks[i];
  const parseAtom = () => {
    const tk = peek();
    if (!tk) return null;
    if (tk.t === "term") { i++; return tk; }
    if (tk.t === "(") {
      i++;
      const n = parseOr();
      if (!n || !peek() || peek().t !== ")") return null;
      i++;
      return n;
    }
    return null;
  };
  const parseNot = () => {
    if (peek() && peek().t === "!") { i++; const c = parseNot(); return c ? { t: "not", c } : null; }
    return parseAtom();
  };
  const parseAnd = () => {
    let n = parseNot();
    while (n && peek() && peek().t === "&") { i++; const r = parseNot(); n = r ? { t: "and", l: n, r } : null; }
    return n;
  };
  const parseOr = () => {
    let n = parseAnd();
    while (n && peek() && peek().t === "|") { i++; const r = parseAnd(); n = r ? { t: "or", l: n, r } : null; }
    return n;
  };
  const ast = parseOr();
  return ast && i === toks.length ? ast : null;
}
function _kwEval(n, text) {
  switch (n.t) {
    case "term": {
      const v = n.v;
      if (v.length > 1 && v.endsWith("*")) {
        const stem = v.slice(0, -1).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        return new RegExp("\\b" + stem, "i").test(text);
      }
      return text.includes(v);
    }
    case "and":  return _kwEval(n.l, text) && _kwEval(n.r, text);
    case "or":   return _kwEval(n.l, text) || _kwEval(n.r, text);
    case "not":  return !_kwEval(n.c, text);
    default:     return true;
  }
}

// Field text → matcher fn. Valid expressions get the boolean treatment;
// anything else (plain prose like "Nature Communications") falls back to a
// case-insensitive substring — so both styles Just Work, no error states.
function _kwMatcher(raw) {
  const s = (raw || "").trim();
  if (!s) return null;
  const ast = _kwParse(s);
  if (ast) return (text) => _kwEval(ast, text);
  const plain = s.toLowerCase();
  return (text) => text.includes(plain);
}

// Dual-handle range slider (two overlaid native ranges; thumbs-only hit
// areas). `log` compresses heavy-tailed domains (citation counts) so the
// low end keeps resolution.
function DualRange({ min, max, lo, hi, onChange, log, disabled }) {
  const N = 400;
  const flat = !(max > min);
  const toVal = (pos) => {
    if (flat) return min;
    const t = pos / N;
    if (log) {
      const a = Math.log(min + 1), b = Math.log(max + 1);
      return Math.max(min, Math.min(max, Math.round(Math.exp(a + t * (b - a)) - 1)));
    }
    return Math.round(min + t * (max - min));
  };
  const toPos = (val) => {
    if (flat) return 0;
    const v = Math.max(min, Math.min(max, val));
    if (log) {
      const a = Math.log(min + 1), b = Math.log(max + 1);
      return Math.round(N * (Math.log(v + 1) - a) / (b - a));
    }
    return Math.round(N * (v - min) / (max - min));
  };
  const pLo = toPos(lo), pHi = toPos(hi);
  return (
    <div className={"dr" + (disabled || flat ? " is-off" : "")}>
      <div className="dr-track" />
      <div className="dr-fill" style={{ left: (100 * pLo / N) + "%", width: (100 * Math.max(0, pHi - pLo) / N) + "%" }} />
      <input type="range" min={0} max={N} value={pLo} disabled={disabled || flat}
        style={{ zIndex: pLo > N / 2 ? 5 : 3 }}
        onChange={e => onChange([Math.min(toVal(+e.target.value), hi), hi])}
        aria-label="lower bound" />
      <input type="range" min={0} max={N} value={pHi} disabled={disabled || flat}
        style={{ zIndex: 4 }}
        onChange={e => onChange([lo, Math.max(toVal(+e.target.value), lo)])}
        aria-label="upper bound" />
    </div>
  );
}

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
    <div className="seed-card pcard" onClick={() => onOpen(paper.id)} title="Click to read the abstract">
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
        <span className="seed-title pcard-title">{paper.title}</span>
      </div>
      <div className="seed-meta pcard-meta pcard-venue">{paper.venue || "—"}</div>
      <div className="seed-meta pcard-meta">
        <span>{paper.authors}</span>
        <span className="seed-meta-sep">·</span>
        <span>{paper.year}</span>
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
  // Local filters over the LOADED results (no re-search): year + citation
  // ranges (null = follow the data's min/max), venue substring, and a
  // boolean keyword expression over title+abstract.
  const [fYear, setFYear] = React.useState(null);     // [lo, hi] | null
  const [fCites, setFCites] = React.useState(null);   // [lo, hi] | null
  const [fVenue, setFVenue] = React.useState("");
  const [fTitle, setFTitle] = React.useState("");
  const [fAbs, setFAbs] = React.useState("");
  // Pagination over S2's relevance search: total matches, the next page's
  // offset (null = exhausted), and how many results this query has fetched.
  const [page, setPage] = React.useState({ total: 0, next: null, fetched: 0 });

  const candidates = seeds.filter(p => !p.starred);
  const acceptedCount = seeds.length - candidates.length;
  const detailPaper = detailId ? seeds.find(p => p.id === detailId) : null;

  // Slider bounds track whatever is currently loaded.
  const bounds = React.useMemo(() => {
    let yMin = Infinity, yMax = -Infinity, cMax = 0;
    for (const p of candidates) {
      if (p.year) { yMin = Math.min(yMin, p.year); yMax = Math.max(yMax, p.year); }
      cMax = Math.max(cMax, p.cites || 0);
    }
    if (!isFinite(yMin)) { yMin = 2000; yMax = new Date().getFullYear(); }
    return { yMin, yMax, cMax };
  }, [seeds]);
  const yr = fYear || [bounds.yMin, bounds.yMax];
  const ct = fCites || [0, bounds.cMax];
  const mVenue = React.useMemo(() => _kwMatcher(fVenue), [fVenue]);
  const mTitle = React.useMemo(() => _kwMatcher(fTitle), [fTitle]);
  const mAbs = React.useMemo(() => _kwMatcher(fAbs), [fAbs]);
  const filtersActive = !!(fYear || fCites || mVenue || mTitle || mAbs);

  const shown = React.useMemo(() => {
    if (!filtersActive) return candidates;
    return candidates.filter(p => {
      if (fYear && ((p.year || 0) < fYear[0] || (p.year || 0) > fYear[1])) return false;
      if (fCites) { const c = p.cites || 0; if (c < fCites[0] || c > fCites[1]) return false; }
      if (mVenue && !mVenue((p.venue || "").toLowerCase())) return false;
      if (mTitle && !mTitle((p.title || "").toLowerCase())) return false;
      if (mAbs && !mAbs((p.abstract || "").toLowerCase())) return false;
      return true;
    });
  }, [candidates, fYear, fCites, mVenue, mTitle, mAbs, filtersActive]);

  const clearFilters = () => { setFYear(null); setFCites(null); setFVenue(""); setFTitle(""); setFAbs(""); };

  const runSearch = async (q) => {
    q = (q || "").trim();
    if (!q) return;
    setBusy(true); setErr(null);
    try {
      const res = await searchSeeds(q);
      const results = res.items || [];
      const starredMap = {};
      seeds.forEach(p => { if (p.starred) starredMap[p.id] = p; });
      const merged = results.map(r => ({ ...r, starred: !!starredMap[r.id] }));
      const seen = new Set(results.map(r => r.id));
      Object.values(starredMap).forEach(p => { if (!seen.has(p.id)) merged.push(p); });
      LIVE.set({ seeds: merged, searchQuery: q });
      setPage({ total: res.total || 0, next: res.next, fetched: results.length });
      // fresh corpus → range filters snap back to the new data's extent
      setFYear(null); setFCites(null);
    } catch (e) {
      setErr(e.message || "search failed");
    }
    setBusy(false);
  };

  const loadMore = async () => {
    if (busy || page.next == null || !query.trim()) return;
    setBusy(true); setErr(null);
    try {
      const res = await searchSeeds(query, null, page.next);
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
            : filtersActive ? `${shown.length} of ${candidates.length} loaded`
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
          <div className="seed-filter-note">
            Filters apply to the {candidates.length.toLocaleString()} loaded result{candidates.length === 1 ? "" : "s"} — no new search.
          </div>
          <div className="seed-filter-row seed-filter-row-slider">
            <span className="seed-filter-k">Year</span>
            <DualRange min={bounds.yMin} max={bounds.yMax} lo={yr[0]} hi={yr[1]}
              disabled={!candidates.length}
              onChange={([a, b]) => setFYear(a <= bounds.yMin && b >= bounds.yMax ? null : [a, b])} />
            <span className="seed-filter-v">{yr[0]} – {yr[1]}</span>
          </div>
          <div className="seed-filter-row seed-filter-row-slider">
            <span className="seed-filter-k">Citations</span>
            <DualRange min={0} max={bounds.cMax} lo={ct[0]} hi={ct[1]} log
              disabled={!candidates.length}
              onChange={([a, b]) => setFCites(a <= 0 && b >= bounds.cMax ? null : [a, b])} />
            <span className="seed-filter-v">{fmtK(ct[0])} – {fmtK(ct[1])}</span>
          </div>
          <label className="seed-filter-field">
            <span className="seed-filter-k">Venue</span>
            <input className="seed-filter-input" value={fVenue}
              placeholder={'"Nature" | "Science" | ICML'}
              onChange={e => setFVenue(e.target.value)} />
          </label>
          <label className="seed-filter-field">
            <span className="seed-filter-k">Title</span>
            <textarea className="cfg-expr-ta seed-filter-kw" rows={1} value={fTitle}
              ref={autoGrowTextarea} onInput={e => autoGrowTextarea(e.target)}
              placeholder={'"bayesian optimization" AND NOT survey'}
              onChange={e => setFTitle(e.target.value)} />
          </label>
          <label className="seed-filter-field">
            <span className="seed-filter-k">Abstract</span>
            <textarea className="cfg-expr-ta seed-filter-kw" rows={1} value={fAbs}
              ref={autoGrowTextarea} onInput={e => autoGrowTextarea(e.target)}
              placeholder={'benchmark OR "ablation study"'}
              onChange={e => setFAbs(e.target.value)} />
          </label>
          <div className="seed-filter-hint">
            Plain text or an expression — AND · OR · NOT · ( ) · "quoted phrase" · term*.
          </div>
          <div className="seed-filter-foot">
            <span>{filtersActive ? `${shown.length.toLocaleString()} of ${candidates.length.toLocaleString()} match` : "no filters active"}</span>
            {filtersActive && (
              <button className="seed-filter-clear" onClick={clearFilters}>
                <Icon name="x" size={10} /> clear
              </button>
            )}
          </div>
        </div>
      )}

      <div className="pb-scroll">
        {err && <div className="seeds-counter" style={{ color: "var(--cc-danger)" }}>{err}</div>}
        {candidates.length === 0 && !busy && (
          <div className="seeds-empty">
            {query.trim()
              ? "No results. Try a different query."
              : "Search Semantic Scholar to find seed papers."}
          </div>
        )}
        {candidates.length > 0 && shown.length === 0 && (
          <div className="seeds-empty">
            No loaded paper matches the local filters — loosen them or load more results.
          </div>
        )}
        <div className="seeds-list">
          {shown.map(p => (
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
