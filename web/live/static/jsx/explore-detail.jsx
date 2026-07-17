/* eslint-disable */
// Section E (Explore mode) — selected-paper detail.
// The f5 reference's slide-in drawer, recast as the app's fixed right panel:
// title / authors / meta grid / abstract (with the OpenAlex lazy fallback the
// seed panels use) / provenance chips / actions, all on existing tokens.

const _xpAbsCache = {};  // paper id -> {abstract, source} (module-lifetime)

function ExploreDetail({ paper, onExploreSubtree, subtreeActive }) {
  const [abs, setAbs] = React.useState(null);
  const [absLoading, setAbsLoading] = React.useState(false);

  React.useEffect(() => {
    setAbs(null);
    if (!paper || paper.abstract) return;
    const cached = _xpAbsCache[paper.id];
    if (cached) { setAbs(cached); return; }
    let cancelled = false;
    setAbsLoading(true);
    fetchSeedAbstract(paper)
      .then(res => {
        const entry = { abstract: (res && res.abstract) || "", source: res && res.source };
        _xpAbsCache[paper.id] = entry;
        if (!cancelled) setAbs(entry);
      })
      .catch(() => { if (!cancelled) setAbs({ abstract: "", source: null }); })
      .finally(() => { if (!cancelled) setAbsLoading(false); });
    return () => { cancelled = true; };
  }, [paper ? paper.id : null]);

  if (!paper) {
    return (
      <aside className="panel panel-right">
        <div className="ph">
          <span className="ph-title">Details</span>
        </div>
        <div className="xp-detail-empty">No paper selected. Click a node or a list row.</div>
      </aside>
    );
  }

  const abstract = paper.abstract || (abs && abs.abstract) || "";
  const absSource = paper.abstract ? null : (abs && abs.source);

  return (
    <aside className="panel panel-right">
      <div className="ph">
        <span className="ph-title">Details</span>
        <span className="ph-actions">
          {paper.seed && <span className="xp-tag xp-tag-seed">Seed</span>}
          {paper.source && !paper.seed && <span className="xp-tag">{paper.source}</span>}
          <span className="xp-tag">d{paper.depth || 0}</span>
        </span>
      </div>

      <div className="xp-detail-scroll">
        <div className="acc-detail-title" style={{ fontSize: 15 }}>{paper.title}</div>
        <div className="acc-detail-meta">{paper.authors || "Unknown authors"}</div>

        <div className="acc-detail-grid">
          <div>
            <div className="acc-detail-k">Year</div>
            <div className="acc-detail-v">{paper.year || "—"}</div>
          </div>
          <div>
            <div className="acc-detail-k">Citations</div>
            <div className="acc-detail-v">{(paper.cites || 0).toLocaleString()}</div>
          </div>
          <div style={{ gridColumn: "1 / 3" }}>
            <div className="acc-detail-k">Venue</div>
            <div className="acc-detail-v" style={{ whiteSpace: "normal" }}>{paper.venue || "—"}</div>
          </div>
        </div>

        <div>
          <div className="seed-abstract-k">
            Abstract
            {abstract && absSource && <span className="seed-abstract-src"> · via {absSource}</span>}
          </div>
          <div className={"seed-abstract" + (abstract ? "" : " is-empty")}>
            {abstract
              ? abstract
              : absLoading
                ? "Looking for an abstract…"
                : "No abstract available from Semantic Scholar or OpenAlex for this paper."}
          </div>
        </div>

        <div className="acc-detail-trail">
          <div className="acc-detail-k">Paper ID</div>
          <div className="acc-detail-v cc-mono-xs" style={{ wordBreak: "break-all" }}>{paper.id}</div>
        </div>

        <div className="acc-detail-actions">
          <a className="btn btn-ghost" href={"https://www.semanticscholar.org/paper/" + paper.id}
             target="_blank" rel="noreferrer">
            <Icon name="external-link" size={12} /> Open in Semantic Scholar
          </a>
          <button className={"btn btn-ghost" + (subtreeActive ? " is-on" : "")}
                  onClick={onExploreSubtree}>
            <Icon name="git-branch" size={12} /> {subtreeActive ? "Show full graph" : "Explore subtree"}
          </button>
        </div>
      </div>
    </aside>
  );
}

Object.assign(window, { ExploreDetail });
