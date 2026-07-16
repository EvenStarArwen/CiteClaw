/* eslint-disable */
// Section E (Run mode) — Accepted papers (live stream).
// Sources from the live store; newest first; freshly-arrived paper flashes.

function RunAccepted({ selectedPaperId, onSelectPaper, detailOpen, onCloseDetail }) {
  const all = useLive("accepted");
  const lastAddedId = useLive("lastAddedId");
  const running = useLive("running");
  const [sort, setSort] = React.useState("recent");
  const [freshIds, setFreshIds] = React.useState(() => new Set());

  const selected = selectedPaperId ?? (all[0] && all[0].id);
  const setSelected = (id) => onSelectPaper && onSelectPaper(id);

  // flash the most-recently accepted paper as it streams in
  React.useEffect(() => {
    if (!lastAddedId) return;
    setFreshIds(prev => new Set(prev).add(lastAddedId));
    const t = setTimeout(() => {
      setFreshIds(prev => { const n = new Set(prev); n.delete(lastAddedId); return n; });
    }, 1800);
    return () => clearTimeout(t);
  }, [lastAddedId]);

  const sorted = React.useMemo(() => {
    const copy = [...all];
    if (sort === "recent") copy.sort((a, b) => b.addedAt - a.addedAt);
    else if (sort === "score") copy.sort((a, b) => b.score - a.score);
    else if (sort === "year") copy.sort((a, b) => b.year - a.year);
    else if (sort === "cites") copy.sort((a, b) => b.cites - a.cites);
    return copy.slice(0, 50);
  }, [sort, all]);

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
  }, [selectedPaperId]);

  const detailPaper = detailOpen && selectedPaperId
    ? all.find(p => p.id === selectedPaperId)
    : null;

  return (
    <aside className="panel panel-right">
      <div className="ph">
        <span className="ph-title">Accepted</span>
        <span className="ph-count">
          <span style={{ color: "var(--cc-ink-1)", fontWeight: 600 }}>{all.length.toLocaleString()}</span>
          <span> total · showing {Math.min(50, all.length)}</span>
        </span>
      </div>

      <div className="accepted-controls">
        <select className="accepted-sort" value={sort} onChange={e => setSort(e.target.value)}>
          <option value="recent">Recently accepted</option>
          <option value="score">Highest score</option>
          <option value="year">Newest year</option>
          <option value="cites">Most cited</option>
        </select>
        <button className="ph-btn" title="Export"><Icon name="download" size={13} /></button>
        <button className="ph-btn" title="Filter"><Icon name="filter" size={13} /></button>
      </div>

      <div className="accepted-scroll" ref={scrollRef}>
        {all.length === 0 && (
          <div className="seeds-counter" style={{ color: "var(--cc-ink-3)" }}>
            No papers yet. Accepted papers stream in here as the run progresses.
          </div>
        )}
        {sorted.map(p => (
          <div
            key={p.id}
            data-paper-id={p.id}
            className={
              "acc-item" +
              (selected === p.id ? " is-selected" : "") +
              (freshIds.has(p.id) ? " is-fresh" : "")
            }
            onClick={() => setSelected(p.id)}
          >
            <span className="acc-score">{(p.score || 0).toFixed(2)}</span>
            <div className="acc-title">{p.title}</div>
            <span className="acc-depth">d{p.depth}</span>
            <div className="acc-meta" style={{ gridColumn: "2 / 4" }}>
              <span>{p.authors}</span>
              <span className="acc-meta-sep">·</span>
              <span>{p.year}</span>
              <span className="acc-meta-sep">·</span>
              <span>{p.venue}</span>
              <span className="acc-meta-sep">·</span>
              <span>{(p.cites || 0).toLocaleString()} cites</span>
            </div>
          </div>
        ))}
      </div>

      <div className="acc-foot">
        <span className="acc-foot-stream">
          <span className={"sb-dot " + (running ? "sb-dot-run" : "sb-dot-idle")} />
          {running ? "Streaming" : "Complete"}
        </span>
        <span>{Math.min(50, all.length)} of {all.length.toLocaleString()}</span>
      </div>

      {detailPaper && (
        <div className="acc-detail">
          <div className="acc-detail-head">
            <span className="acc-detail-tag">Selected</span>
            <button className="ph-btn" onClick={onCloseDetail} title="Close">
              <Icon name="x" size={14} />
            </button>
          </div>
          <div className="acc-detail-title">{detailPaper.title}</div>
          <div className="acc-detail-meta">
            {detailPaper.authors} · {detailPaper.year} · {detailPaper.venue}
          </div>
          <div className="acc-detail-grid">
            <div>
              <div className="acc-detail-k">Score</div>
              <div className="acc-detail-v">{(detailPaper.score || 0).toFixed(3)}</div>
            </div>
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
