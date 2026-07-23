/* eslint-disable */
// Section B (Explore mode) — the paper/author list, data-source picker and
// facet filters. Mirrors the f5 reference's sidebar (sortable rows driving
// the graph selection) plus the richer tools: a source select swapping
// between the live session and finished runs on disk, and a Filters bar
// (year window / min citations / seeds only / keyword formula — same UI as
// the seed search). Filters REMOVE nodes from the graph simulation, so the
// force layout re-flows as you filter.

const XP_LIST_CAP = 400;
const XP_EMPTY_FILTERS = { yearMin: "", yearMax: "", minCites: "", maxCites: "", seedsOnly: false, kw: "" };

// Keyword formula, same DSL as the search pipeline's keyword filters:
// bare words / "quoted phrases" combined with AND OR NOT and parentheses
// (& | ! also accepted), e.g. (transformer OR attention) AND NOT survey.
// A trailing * is a prefix wildcard. Case-insensitive. Anything that
// doesn't parse falls back to matching the raw text as one plain phrase.
function xpCompileQuery(q) {
  const src = String(q || "").trim();
  if (!src) return null;
  const toks = src.match(/\(|\)|&|\||!|"[^"]*"|[^\s()&|!]+/g) || [];
  let i = 0;
  const isOr = (x) => x === "|" || (typeof x === "string" && x.toLowerCase() === "or");
  const isAnd = (x) => x === "&" || (typeof x === "string" && x.toLowerCase() === "and");
  const isNot = (x) => x === "!" || (typeof x === "string" && x.toLowerCase() === "not");
  const parseOr = () => {
    let l = parseAnd();
    while (isOr(toks[i])) { i++; const a = l, b = parseAnd(); l = t => a(t) || b(t); }
    return l;
  };
  const parseAnd = () => {
    let l = parseNot();
    while (isAnd(toks[i])) { i++; const a = l, b = parseNot(); l = t => a(t) && b(t); }
    return l;
  };
  const parseNot = () => {
    if (isNot(toks[i])) { i++; const inner = parseNot(); return t => !inner(t); }
    return parseAtom();
  };
  const parseAtom = () => {
    const tok = toks[i];
    if (tok === "(") {
      i++;
      const e = parseOr();
      if (toks[i] !== ")") throw 0;
      i++;
      return e;
    }
    if (tok == null || tok === ")" || isAnd(tok) || isOr(tok)) throw 0;
    i++;
    const raw = (tok[0] === '"' ? tok.slice(1, -1) : tok).toLowerCase();
    if (raw.length > 1 && raw.endsWith("*")) {
      const stem = raw.slice(0, -1).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const rx = new RegExp("\\b" + stem, "i");
      return t => rx.test(t);
    }
    return t => t.includes(raw);
  };
  try {
    const fn = parseOr();
    if (i !== toks.length) throw 0;
    return fn;
  } catch (_) {
    const needle = src.toLowerCase();
    return t => t.includes(needle);
  }
}

// Shared with app.jsx (graph hiddenIds) so list + graph always agree.
// kind="author": year/seeds don't apply, "min citations" reads as min papers,
// and the keyword formula searches name + affiliation.
function xpFilterPredicate(f, kind) {
  const yMin = Number(f.yearMin) || 0;
  const yMax = Number(f.yearMax) || 0;
  const cMin = Number(f.minCites) || 0;
  const cMax = Number(f.maxCites) || 0;
  const kw = xpCompileQuery(f.kw);
  const author = kind === "author";
  return (p) => {
    if (!author) {
      if (f.seedsOnly && !p.seed) return false;
      if (yMin && (p.year || 0) < yMin) return false;
      if (yMax && (p.year || 0) > yMax) return false;
      if (cMin && (p.cites || 0) < cMin) return false;
      if (cMax && (p.cites || 0) > cMax) return false;
    } else {
      if (cMin && (p.nPapers || 0) < cMin) return false;
      if (cMax && (p.nPapers || 0) > cMax) return false;
    }
    if (kw) {
      const hay = author
        ? `${p.title || ""} ${p.venue || ""}`.toLowerCase()
        : `${p.title || ""} ${p.abstract || ""}`.toLowerCase();
      if (!kw(hay)) return false;
    }
    return true;
  };
}
function xpFilterCount(f, kind) {
  // kw is its own search box now (see the find-box), so it no longer counts
  // toward the Filters badge.
  const base = (f.minCites ? 1 : 0) + (f.maxCites ? 1 : 0);
  if (kind === "author") return base;
  return base + (f.yearMin ? 1 : 0) + (f.yearMax ? 1 : 0) + (f.seedsOnly ? 1 : 0);
}

function ExploreList({ papers, kind, selectedId, onSelect, sort, setSort,
                       filters, setFilters, explore, onPickSource, onPickFile,
                       graphHidden }) {
  const scrollRef = React.useRef(null);
  const fileRef = React.useRef(null);
  const [showFilters, setShowFilters] = React.useState(false);
  const author = kind === "author";
  const nFilters = xpFilterCount(filters, kind);

  // Facet filters remove rows; graph-side structural filters (min degree /
  // edge weight / largest component) do NOT — those papers stay listed but
  // are dimmed, since they exist in the collection yet not in the network
  // view. Removing them entirely would make the collection look smaller than
  // it is; keeping them undimmed would invite hunting for missing nodes.
  const visible = React.useMemo(() => {
    const pred = xpFilterPredicate(filters, kind);
    return papers.filter(pred);
  }, [papers, filters, kind]);
  const offSet = graphHidden && graphHidden.size ? graphHidden : null;
  const nOff = React.useMemo(() => {
    if (!offSet) return 0;
    let k = 0;
    for (const p of visible) if (offSet.has(p.id)) k++;
    return k;
  }, [visible, offSet]);
  const inGraph = visible.length - nOff;

  const sorted = React.useMemo(() => {
    const copy = [...visible];
    if (sort === "citations") copy.sort((a, b) => (b.cites || 0) - (a.cites || 0));
    else if (sort === "year") copy.sort((a, b) => (b.year || 0) - (a.year || 0));
    else if (sort === "papers") copy.sort((a, b) => (b.nPapers || 0) - (a.nPapers || 0));
    else if (sort === "hindex") copy.sort((a, b) => (b.hIndex || 0) - (a.hIndex || 0));
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

  const srcValue = explore.source === "run" ? explore.runPath
    : explore.source === "upload" ? "upload" : "live";
  const liveCount = (LIVE.get("accepted") || []).length;
  const patchF = (patch) => setFilters({ ...filters, ...patch });

  return (
    <aside className="panel panel-left">
      <div className="ph">
        <span className="ph-title">{author ? "Authors" : "Papers"}</span>
        <span className="ph-count">
          <span style={{ color: "var(--cc-ink-1)", fontWeight: 600 }}>{inGraph.toLocaleString()}</span>
          <span>{visible.length !== papers.length ? ` of ${papers.length.toLocaleString()}` : " in graph"}</span>
          {nOff > 0 && (
            <span title="In the collection, but removed from the network view by the graph filters (min degree / edge weight / largest component)">
              {" "}· {nOff.toLocaleString()} off
            </span>
          )}
        </span>
      </div>

      <div className="list-find">
        <Icon name="search" size={13} className="list-find-ic" />
        <input type="text" className="list-find-input" spellCheck="false"
          placeholder={author ? "Find an author…" : "Find a paper…"}
          title={author
            ? "Search author name or affiliation. Supports AND OR NOT and ( )."
            : "Search title + abstract. Supports AND OR NOT ( ) and \"quoted phrases\" — e.g. (graph OR network) AND NOT survey"}
          value={filters.kw} onChange={e => patchF({ kw: e.target.value })} />
        {filters.kw && (
          <button className="list-find-x" onClick={() => patchF({ kw: "" })}
            aria-label="Clear search" title="Clear search">
            <Icon name="x" size={12} />
          </button>
        )}
      </div>

      <div className="xp-controls">
        <select
          className="accepted-sort"
          value={srcValue}
          onChange={e => onPickSource(e.target.value)}
          title="Data source"
        >
          <option value="live">Current session · {liveCount.toLocaleString()} papers</option>
          {explore.upload && (
            <option value="upload">
              File · {explore.upload.name} · {(explore.upload.papers || []).length.toLocaleString()} nodes
            </option>
          )}
          {explore.runs.map(r => (
            <option key={r.path} value={r.path}>
              {r.label} · {r.papers.toLocaleString()} papers · {r.modified}
            </option>
          ))}
        </select>
        <button className="ph-btn" onClick={() => refreshExploreRuns()} title="Rescan runs on disk">
          <Icon name="refresh-cw" size={12} />
        </button>
        <button className="ph-btn" onClick={() => fileRef.current && fileRef.current.click()}
                title="Open a local graph file (.graphml / .gexf) — a CiteClaw citation or collaboration network">
          <Icon name="upload" size={12} />
        </button>
        <input ref={fileRef} type="file" accept=".graphml,.gexf,.xml"
               style={{ display: "none" }}
               onChange={e => {
                 const f = e.target.files && e.target.files[0];
                 e.target.value = "";  // same file re-picked should re-fire
                 if (f && onPickFile) onPickFile(f);
               }} />
      </div>
      <div className="xp-controls">
        <select className="accepted-sort" value={sort} onChange={e => setSort(e.target.value)}>
          {author ? (
            <>
              <option value="papers">Sort by: Papers</option>
              <option value="hindex">Sort by: h-index</option>
              <option value="citations">Sort by: Citations</option>
              <option value="title">Sort by: Name</option>
            </>
          ) : (
            <>
              <option value="citations">Sort by: Citations</option>
              <option value="year">Sort by: Year</option>
              <option value="title">Sort by: Title</option>
            </>
          )}
        </select>
      </div>

      <div className="seed-filterbar">
        <button className={"seed-filter-toggle" + (showFilters ? " is-open" : "")}
                onClick={() => setShowFilters(v => !v)}>
          <Icon name="sliders-horizontal" size={12} /> Filters{nFilters ? ` · ${nFilters}` : ""}
          <Icon name={showFilters ? "chevron-up" : "chevron-down"} size={11} />
        </button>
        {nFilters > 0 && (
          <button className="ph-btn" onClick={() => setFilters({ ...XP_EMPTY_FILTERS, kw: filters.kw })}
                  title="Clear filters">
            <Icon name="x" size={11} />
          </button>
        )}
      </div>
      {showFilters && (
        <div className="seed-filters">
          {!author && (
            <>
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
            </>
          )}
          <label className="seed-filter-row">
            <span className="seed-filter-k">{author ? "Min papers" : "Min citations"}</span>
            <input type="number" min="0" step={author ? 1 : 10} placeholder="0" value={filters.minCites}
                   onChange={e => patchF({ minCites: e.target.value })} />
          </label>
          <label className="seed-filter-row">
            <span className="seed-filter-k">{author ? "Max papers" : "Max citations"}</span>
            <input type="number" min="0" step={author ? 1 : 10} placeholder="any" value={filters.maxCites}
                   onChange={e => patchF({ maxCites: e.target.value })} />
          </label>
          {!author && (
            <label className="seed-filter-row">
              <span className="seed-filter-k">Seeds only</span>
              <input type="checkbox" checked={filters.seedsOnly}
                     onChange={e => patchF({ seedsOnly: e.target.checked })} />
            </label>
          )}
        </div>
      )}

      <div className="xp-scroll" ref={scrollRef}>
        {explore.loading && <div className="xp-note">Loading run…</div>}
        {explore.error && <div className="xp-note xp-note-err">{explore.error}</div>}
        {!explore.loading && !visible.length && !explore.error && (
          <div className="xp-note">
            {papers.length
              ? `No ${author ? "authors" : "papers"} match the filters.`
              : explore.source === "live"
                ? "No papers in this session yet. Run a pipeline, or pick a finished run above."
                : author
                  ? "No author data for this run."
                  : "This run has no accepted papers."}
          </div>
        )}
        {sorted.map(p => {
          const off = offSet ? offSet.has(p.id) : false;
          return (
            <PaperCard
              key={p.id}
              paperId={p.id}
              title={p.title}
              titlePrefix={p.seed ? <span className="pcard-seed-dot" title="Seed paper" /> : null}
              venue={p.venue}
              rows={[
                <span key="a">
                  {off && <span className="pcard-offnet-ic"><Icon name="unlink" size={10} /></span>}
                  {author ? (p.affiliation || "—") : (p.authors || "—")}
                </span>,
                author
                  ? ([p.hIndex != null ? `h ${p.hIndex}` : null, `${(p.nPapers || 0).toLocaleString()} papers`].filter(Boolean).join(" · ") || "—")
                  : ([p.year || null, `${fmtK(p.cites || 0)} cites`].filter(Boolean).join(" · ") || "—"),
              ]}
              selected={selectedId === p.id}
              offnet={off}
              onClick={() => onSelect(p.id === selectedId ? null : p.id)}
              tooltip={off
                ? "In the collection, but not in the network view — removed by the graph filters (min degree / edge weight / largest component)"
                : undefined}
            />
          );
        })}
      </div>

      <div className="acc-foot">
        <span className="acc-foot-stream">
          {explore.source === "run"
            ? <>run · <span className="cc-mono-xs">{explore.runPath}</span></>
            : explore.source === "upload"
              ? <>file · <span className="cc-mono-xs">{explore.upload ? explore.upload.name : ""}</span></>
              : "live session"}
        </span>
        <span>{Math.min(XP_LIST_CAP, visible.length)} of {visible.length.toLocaleString()}</span>
      </div>
    </aside>
  );
}

Object.assign(window, { ExploreList, xpFilterPredicate, xpCompileQuery, XP_EMPTY_FILTERS });
