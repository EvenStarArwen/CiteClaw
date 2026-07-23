/* eslint-disable */
// Section E (Run mode) — Accepted / Rejected papers sidebar.
// Two tabs over one paginated list:
//   • Accepted — the live stream (client-side paginated; newest flashes in).
//   • Rejected — server-side paginated (/api/run/:id/rejected), each row
//     carrying the human reason it was filtered out. Server-side so a run
//     that rejects tens of thousands stays snappy: the browser only ever
//     holds the page in view, and the detail buffer lives on the server.
// Tunable page size (10 / 25 / 50) applies to both tabs.

const ACC_SORTS = [
  ["recent", "Recently accepted"], ["score", "Highest score"],
  ["year", "Newest year"], ["cites", "Most cited"],
];
const REJ_SORTS = [
  ["recent", "Recently rejected"], ["category", "By filter / reason"],
  ["year", "Newest year"], ["cites", "Most cited"],
];
const PAGE_SIZES = [10, 25, 50];

function RunAccepted({ selectedPaperId, onSelectPaper, detailOpen, onCloseDetail }) {
  const all = useLive("accepted");
  const lastAddedId = useLive("lastAddedId");
  const running = useLive("running");
  const runId = useLive("runId");
  const metrics = useLive("metrics");

  const [tab, setTab] = React.useState("accepted");
  const [sort, setSort] = React.useState("recent");
  const [pageSize, setPageSize] = React.useState(25);
  const [page, setPage] = React.useState(0);
  const [freshIds, setFreshIds] = React.useState(() => new Set());
  const [rej, setRej] = React.useState({ items: [], total: 0, capped: false, loading: false, error: null });
  // Object of the currently-selected rejected paper, cached at click time so
  // its detail card survives pagination (the row may scroll off rej.items).
  const [pickedRej, setPickedRej] = React.useState(null);
  // One find-box over the visible list (both tabs). "Is my paper here?" —
  // list-only, never touches the graph. Accepted filters client-side; rejected
  // is server-paginated, so it searches the whole buffer via a ?q= param.
  const [query, setQuery] = React.useState("");
  const [debouncedQ, setDebouncedQ] = React.useState("");
  React.useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(query.trim()), 250);
    return () => clearTimeout(t);
  }, [query]);
  React.useEffect(() => { setPage(0); }, [debouncedQ]);

  const selected = selectedPaperId;
  const setSelected = (id) => onSelectPaper && onSelectPaper(id);

  // Reset paging whenever the view definition changes.
  const switchTab = (t) => { if (t === tab) return; setTab(t); setSort("recent"); setPage(0); };
  const changeSort = (s) => { setSort(s); setPage(0); };
  const changeSize = (n) => { setPageSize(n); setPage(0); };

  // flash the most-recently accepted paper as it streams in
  React.useEffect(() => {
    if (!lastAddedId) return;
    setFreshIds(prev => new Set(prev).add(lastAddedId));
    const t = setTimeout(() => {
      setFreshIds(prev => { const n = new Set(prev); n.delete(lastAddedId); return n; });
    }, 1800);
    return () => clearTimeout(t);
  }, [lastAddedId]);

  // Accepted — client-side sort over the live store.
  const sortedAccepted = React.useMemo(() => {
    const copy = [...all];
    if (sort === "recent") copy.sort((a, b) => b.addedAt - a.addedAt);
    else if (sort === "score") copy.sort((a, b) => b.score - a.score);
    else if (sort === "year") copy.sort((a, b) => b.year - a.year);
    else if (sort === "cites") copy.sort((a, b) => b.cites - a.cites);
    return copy;
  }, [sort, all]);

  // Accepted find — over title + authors, client-side (the full list is in
  // memory). Rejected search happens server-side (see the fetch below).
  const filteredAccepted = React.useMemo(() => {
    const q = debouncedQ.toLowerCase();
    if (!q) return sortedAccepted;
    return sortedAccepted.filter(p =>
      (`${p.title || ""} ${p.authors || ""}`).toLowerCase().includes(q));
  }, [sortedAccepted, debouncedQ]);

  // Rejected — server-side page, re-fetched on navigation and polled while
  // the run is live (each poll re-reads LIVE.running so it stops on its own).
  React.useEffect(() => {
    if (tab !== "rejected" || !runId) return;
    let alive = true, timer = null;
    const load = async (showLoading) => {
      if (showLoading) setRej(r => ({ ...r, loading: true }));
      try {
        const d = await fetchRejected(runId, { offset: page * pageSize, limit: pageSize, sort, q: debouncedQ });
        if (!alive) return;
        setRej({ items: d.items || [], total: d.total || 0, capped: !!d.capped, loading: false, error: null });
      } catch (e) {
        if (!alive) return;
        setRej(r => ({ ...r, loading: false, error: (e && e.message) || "failed to load" }));
      }
      if (alive && LIVE.get("running")) timer = setTimeout(() => load(false), 2500);
    };
    load(true);
    return () => { alive = false; if (timer) clearTimeout(timer); };
  }, [tab, runId, page, pageSize, sort, running, debouncedQ]);

  const total = tab === "accepted" ? filteredAccepted.length : rej.total;
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const curPage = Math.min(page, pageCount - 1);
  const items = tab === "accepted"
    ? filteredAccepted.slice(curPage * pageSize, curPage * pageSize + pageSize)
    : rej.items;

  const scrollRef = React.useRef(null);
  React.useEffect(() => {
    if (!selectedPaperId) return;
    const el = scrollRef.current?.querySelector(`[data-paper-id="${selectedPaperId}"]`);
    if (!el) return;
    const c = scrollRef.current;
    const eTop = el.offsetTop - c.offsetTop;
    const eBot = eTop + el.offsetHeight;
    if (eTop < c.scrollTop || eBot > c.scrollTop + c.clientHeight) {
      c.scrollTo({ top: eTop - 40, behavior: "smooth" });
    }
  }, [selectedPaperId, tab, curPage]);

  const detailPaper = detailOpen && selectedPaperId
    ? (all.find(p => p.id === selectedPaperId)
       || (pickedRej && pickedRej.id === selectedPaperId ? pickedRej : null)
       || rej.items.find(p => p.id === selectedPaperId)
       || null)
    : null;
  const isRej = !!(detailPaper && detailPaper.category != null);

  const sortOpts = tab === "accepted" ? ACC_SORTS : REJ_SORTS;
  const rangeLo = total === 0 ? 0 : curPage * pageSize + 1;
  const rangeHi = Math.min(total, curPage * pageSize + pageSize);
  const rejBadge = metrics ? (metrics.rejected || 0) : 0;

  const pick = (p, rejected) => { setSelected(p.id); setPickedRej(rejected ? p : null); };

  return (
    <aside className="panel panel-right">
      <div className="ph">
        <span className="ph-title">{tab === "accepted" ? "Accepted" : "Rejected"}</span>
        <span className="ph-count">
          <span style={{ color: "var(--cc-ink-1)", fontWeight: 600 }}>{total.toLocaleString()}</span>
          <span>{debouncedQ ? " matching"
            : tab === "rejected" && rej.capped ? " (first 20k)" : " total"}</span>
        </span>
      </div>

      <div className="acc-tabs" role="tablist">
        <button role="tab" aria-selected={tab === "accepted"}
          className={"acc-tab" + (tab === "accepted" ? " is-active" : "")}
          onClick={() => switchTab("accepted")}>
          Accepted <span className="acc-tab-n">{all.length.toLocaleString()}</span>
        </button>
        <button role="tab" aria-selected={tab === "rejected"}
          className={"acc-tab" + (tab === "rejected" ? " is-active" : "")}
          onClick={() => switchTab("rejected")}>
          Rejected <span className="acc-tab-n">{rejBadge.toLocaleString()}</span>
        </button>
      </div>

      <div className="list-find">
        <Icon name="search" size={13} className="list-find-ic" />
        <input type="text" className="list-find-input" spellCheck="false"
          placeholder={tab === "accepted" ? "Find an accepted paper…" : "Find a rejected paper…"}
          title="Search this list by title or author — does not change the graph"
          value={query} onChange={e => setQuery(e.target.value)} />
        {query && (
          <button className="list-find-x" onClick={() => setQuery("")}
            aria-label="Clear search" title="Clear search">
            <Icon name="x" size={12} />
          </button>
        )}
      </div>

      <div className="accepted-controls">
        <select className="accepted-sort" value={sort} onChange={e => changeSort(e.target.value)}>
          {sortOpts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        <select className="accepted-pagesize" value={pageSize}
          onChange={e => changeSize(Number(e.target.value))} title="Papers per page">
          {PAGE_SIZES.map(n => <option key={n} value={n}>{n} / page</option>)}
        </select>
      </div>

      <div className="accepted-scroll" ref={scrollRef}>
        {tab === "accepted" && all.length === 0 && (
          <div className="seeds-counter" style={{ color: "var(--cc-ink-3)" }}>
            No papers yet. Accepted papers stream in here as the run progresses.
          </div>
        )}
        {tab === "accepted" && all.length > 0 && total === 0 && debouncedQ && (
          <div className="seeds-counter" style={{ color: "var(--cc-ink-3)" }}>
            No accepted paper matches “{debouncedQ}”.
          </div>
        )}
        {tab === "rejected" && rej.total === 0 && !rej.loading && !rej.error && (
          <div className="seeds-counter" style={{ color: "var(--cc-ink-3)" }}>
            {debouncedQ
              ? `No rejected paper matches “${debouncedQ}”.`
              : "No rejected papers yet. Papers filtered out by your screeners appear here with the reason."}
          </div>
        )}
        {tab === "rejected" && rej.error && (
          <div className="seeds-counter" style={{ color: "var(--cc-danger)" }}>
            Couldn’t load rejected papers: {rej.error}
          </div>
        )}

        {tab === "accepted" && items.map(p => (
          <div
            key={p.id}
            data-paper-id={p.id}
            className={
              "acc-item pcard" +
              (selected === p.id ? " is-selected" : "") +
              (freshIds.has(p.id) ? " is-fresh" : "")
            }
            onClick={() => pick(p, false)}
          >
            <div className="acc-title pcard-title">{p.title}</div>
            <span className="acc-depth">d{p.depth}</span>
            <div className="acc-meta pcard-meta pcard-venue" style={{ gridColumn: "1 / 3" }}>{p.venue || "—"}</div>
            <div className="acc-meta pcard-meta" style={{ gridColumn: "1 / 3" }}>
              <span>{p.authors}</span>
              <span className="acc-meta-sep">·</span>
              <span>{p.year}</span>
              <span className="acc-meta-sep">·</span>
              <span>{(p.cites || 0).toLocaleString()} cites</span>
            </div>
          </div>
        ))}

        {tab === "rejected" && items.map(p => (
          <div
            key={p.id}
            data-paper-id={p.id}
            className={"acc-item is-reject pcard" + (selected === p.id ? " is-selected" : "")}
            onClick={() => pick(p, true)}
          >
            <span className="acc-reject-mark" title="Rejected"><Icon name="x" size={12} /></span>
            <div className="acc-title pcard-title">{p.title}</div>
            <span className="acc-cat" title={p.category}>{fmtReason(p.category)}</span>
            <div className="acc-meta pcard-meta pcard-venue" style={{ gridColumn: "2 / 4" }}>{p.venue || "—"}</div>
            <div className="acc-meta pcard-meta" style={{ gridColumn: "2 / 4" }}>
              <span>{p.authors || "—"}</span>
              <span className="acc-meta-sep">·</span>
              <span>{p.year || "—"}</span>
              <span className="acc-meta-sep">·</span>
              <span>{(p.cites || 0).toLocaleString()} cites</span>
            </div>
            {p.reason && (
              <div className="acc-reason" style={{ gridColumn: "2 / 4" }} title={p.reason}>{p.reason}</div>
            )}
          </div>
        ))}
      </div>

      <div className="acc-foot">
        <span className="acc-foot-stream">
          <span className={"sb-dot " + (running ? "sb-dot-run" : "sb-dot-idle")} />
          {running ? "Streaming" : "Complete"}
        </span>
        <div className="acc-pager">
          <button className="acc-page-btn" disabled={curPage <= 0}
            onClick={() => setPage(p => Math.max(0, p - 1))} title="Previous page" aria-label="Previous page">
            <Icon name="chevron-left" size={14} />
          </button>
          <span className="acc-page-info">
            {total === 0 ? "0" : `${rangeLo.toLocaleString()}–${rangeHi.toLocaleString()}`} of {total.toLocaleString()}
          </span>
          <button className="acc-page-btn" disabled={curPage >= pageCount - 1}
            onClick={() => setPage(p => Math.min(pageCount - 1, p + 1))} title="Next page" aria-label="Next page">
            <Icon name="chevron-right" size={14} />
          </button>
        </div>
      </div>

      {detailPaper && (
        <div className="acc-detail">
          <div className="acc-detail-head">
            <span className={"acc-detail-tag" + (isRej ? " is-reject" : "")}>{isRej ? "Rejected" : "Selected"}</span>
            <button className="ph-btn" onClick={onCloseDetail} title="Close">
              <Icon name="x" size={14} />
            </button>
          </div>
          <div className="acc-detail-title">{detailPaper.title}</div>
          <div className="acc-detail-meta">
            {(detailPaper.authors || "—")} · {detailPaper.year || "—"} · {detailPaper.venue || "—"}
          </div>

          {isRej && (
            <div className="acc-detail-reject">
              <div className="acc-detail-k">Rejected by</div>
              <div className="acc-detail-catrow">
                <span className="acc-cat" title={detailPaper.category}>{fmtReason(detailPaper.category)}</span>
              </div>
              {detailPaper.reason && <div className="acc-detail-reason">{detailPaper.reason}</div>}
            </div>
          )}

          <div className="acc-detail-grid">
            {!isRej && (
              <div>
                <div className="acc-detail-k">Score</div>
                <div className="acc-detail-v">{(detailPaper.score || 0).toFixed(3)}</div>
              </div>
            )}
            <div>
              <div className="acc-detail-k">Cites</div>
              <div className="acc-detail-v">{(detailPaper.cites || 0).toLocaleString()}</div>
            </div>
            <div>
              <div className="acc-detail-k">Depth</div>
              <div className="acc-detail-v">d{detailPaper.depth}</div>
            </div>
            <div>
              <div className="acc-detail-k">Source</div>
              <div className="acc-detail-v">{detailPaper.source || "—"}</div>
            </div>
          </div>
          <div className="acc-detail-trail">
            <div className="acc-detail-k">Paper ID</div>
            <div className="acc-detail-v cc-mono-xs" style={{ wordBreak: "break-all" }}>{detailPaper.id}</div>
          </div>
          <div className="acc-detail-actions">
            <a className="btn btn-ghost" href={"https://www.semanticscholar.org/paper/" + detailPaper.id}
               target="_blank" rel="noreferrer">
              <Icon name="external-link" size={12} /> Open in Semantic Scholar
            </a>
          </div>
        </div>
      )}
    </aside>
  );
}

Object.assign(window, { RunAccepted });
