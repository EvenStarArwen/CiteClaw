/* eslint-disable */
// Section B (Build mode) — Seeds search.
// Live: searches Semantic Scholar (press Enter), stars feed the seed block.
// Starred papers stick across searches so they aren't lost when results change.

function BuildSeeds() {
  const seeds = useLive("seeds");
  const [query, setQuery] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState(null);

  const selected = seeds.filter(p => p.starred).length;

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

  return (
    <aside className="panel panel-left">
      <div className="ph">
        <span className="ph-title">Seeds</span>
        <span className="ph-count">{busy ? "…" : seeds.length + " results"}</span>
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
              onClick={() => toggle(p.id)}
            >
              <div className="seed-card-head">
                <span className="seed-star">
                  <Icon name="star" size={13} />
                </span>
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
    </aside>
  );
}

Object.assign(window, { BuildSeeds });
