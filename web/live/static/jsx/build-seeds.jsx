/* eslint-disable */
// Section B (Build mode) — Seeds search.
// Live: searches Semantic Scholar (press Enter), stars feed the seed block.
// Starred papers stick across searches so they aren't lost when results change.
//
// Interactions:
//   · click the STAR on a card  -> accept/unaccept it as a seed
//   · click the card body       -> open the full abstract (fills the panel)

function BuildSeeds() {
  const seeds = useLive("seeds");
  const [query, setQuery] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState(null);
  const [detailId, setDetailId] = React.useState(null);

  const selected = seeds.filter(p => p.starred).length;
  const detailPaper = detailId ? seeds.find(p => p.id === detailId) : null;

  const runSearch = async (q) => {
    q = (q || "").trim();
    if (!q) return;
    setBusy(true); setErr(null);
    try {
      const results = await searchSeeds(q);
      const starredMap = {};
      seeds.forEach(p => { if (p.starred) starredMap[p.id] = p; });
      const merged = results.map(r => ({ ...r, starred: !!starredMap[r.id] }));
      const seen = new Set(results.map(r => r.id));
      Object.values(starredMap).forEach(p => { if (!seen.has(p.id)) merged.push(p); });
      LIVE.set({ seeds: merged });
    } catch (e) {
      setErr(e.message || "search failed");
    }
    setBusy(false);
  };

  const toggle = (id) => LIVE.set({
    seeds: seeds.map(p => p.id === id ? { ...p, starred: !p.starred } : p),
  });

  // Esc closes the detail view (matches the Settings modal behaviour).
  React.useEffect(() => {
    if (!detailPaper) return;
    const onKey = (e) => { if (e.key === "Escape") setDetailId(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [detailPaper]);

  const s2Url = (p) => {
    const ext = p.externalIds || {};
    if (ext.DOI) return "https://doi.org/" + ext.DOI;
    if (ext.ArXiv) return "https://arxiv.org/abs/" + ext.ArXiv;
    return "https://www.semanticscholar.org/paper/" + p.id;
  };

  return (
    <aside className="panel panel-left">
      <div className="ph">
        <span className="ph-title">Seeds</span>
        <span className="ph-count">
          {detailPaper ? "abstract" : (busy ? "…" : seeds.length + " results")}
        </span>
      </div>

      {detailPaper ? (
        <div className="seed-detail">
          <button className="seed-detail-back" onClick={() => setDetailId(null)} title="Back (Esc)">
            <Icon name="arrow-left" size={13} /> Back to results
          </button>
          <div className="seed-detail-body">
            <div className="acc-detail-title">{detailPaper.title}</div>
            <div className="acc-detail-meta">{detailPaper.authors || "Unknown authors"}</div>
            <div className="acc-detail-grid">
              <div>
                <div className="acc-detail-k">Year</div>
                <div className="acc-detail-v">{detailPaper.year || "—"}</div>
              </div>
              <div>
                <div className="acc-detail-k">Citations</div>
                <div className="acc-detail-v">{(detailPaper.cites || 0).toLocaleString()}</div>
              </div>
              <div style={{ gridColumn: "1 / 3" }}>
                <div className="acc-detail-k">Venue</div>
                <div className="acc-detail-v" style={{ whiteSpace: "normal" }}>{detailPaper.venue || "—"}</div>
              </div>
            </div>
            <div>
              <div className="seed-abstract-k">Abstract</div>
              <div className={"seed-abstract" + (detailPaper.abstract ? "" : " is-empty")}>
                {detailPaper.abstract || "No abstract available from Semantic Scholar for this paper."}
              </div>
            </div>
          </div>
          <div className="seed-detail-foot">
            <button
              className={"btn " + (detailPaper.starred ? "btn-primary" : "btn-ghost")}
              onClick={() => toggle(detailPaper.id)}
            >
              <Icon name="star" size={13} />
              {detailPaper.starred ? "Seed ✓ — starred" : "Star as seed"}
            </button>
            <a className="btn btn-ghost" href={s2Url(detailPaper)} target="_blank" rel="noreferrer" title="Open source">
              <Icon name="external-link" size={12} />
            </a>
          </div>
        </div>
      ) : (
        <>
          <div className="searchbox">
            <Icon name="search" size={12} />
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter") runSearch(query); }}
              placeholder="Search Semantic Scholar — press Enter"
            />
            {query && (
              <button className="ph-btn" onClick={() => setQuery("")} title="Clear">
                <Icon name="x" size={11} />
              </button>
            )}
          </div>

          <div className="pb-scroll">
            <div className="seeds-counter">
              <span>
                <span className="seeds-counter-n">{selected}</span> starred · feeds seed block
              </span>
              <Icon name="star" size={12} style={{ color: "var(--cc-warning)" }} />
            </div>
            {err && (
              <div className="seeds-counter" style={{ color: "var(--cc-danger)" }}>{err}</div>
            )}
            <div className="seeds-list">
              {seeds.map(p => (
                <div
                  key={p.id}
                  className={"seed-card" + (p.starred ? " is-selected" : "")}
                  onClick={() => setDetailId(p.id)}
                  title="Click to read the abstract"
                >
                  <div className="seed-card-head">
                    <button
                      className="seed-star-btn"
                      onClick={e => { e.stopPropagation(); toggle(p.id); }}
                      title={p.starred ? "Unstar (remove from seeds)" : "Star as seed"}
                      aria-label={p.starred ? "Unstar" : "Star as seed"}
                      aria-pressed={!!p.starred}
                    >
                      <span className="seed-star"><Icon name="star" size={13} /></span>
                    </button>
                    <span className="seed-title">{p.title}</span>
                  </div>
                  <div className="seed-meta">
                    <span>{p.authors}</span>
                    <span className="seed-meta-sep">·</span>
                    <span>{p.year}</span>
                    <span className="seed-meta-sep">·</span>
                    <span>{p.venue}</span>
                    <span className="seed-meta-sep">·</span>
                    <span className="seed-cites">{(p.cites || 0).toLocaleString()} cites</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </aside>
  );
}

Object.assign(window, { BuildSeeds });
