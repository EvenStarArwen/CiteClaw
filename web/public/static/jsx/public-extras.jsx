/* eslint-disable */
// Public-deployment extras, appended after app.jsx by the public server's
// index assembly. Declares PublicTopbarExtra — the top bar renders it in
// its actions row (next to Reset / Run) via a typeof-guarded slot, so the
// shared component needs no fork and the local build is untouched:
//   * "Results" button + dropdown — the session's runs with per-artifact
//     download links + one-zip bundle
//   * scale-to-zero notice — while a run is active, warn that the server
//     sleeps shortly after the tab closes (the /me poll doubles as the
//     keep-alive while the tab stays open)

function PublicTopbarExtra() {
  const [open, setOpen] = React.useState(false);
  const [runs, setRuns] = React.useState([]);
  const [me, setMe] = React.useState(null);
  const [toastRun, setToastRun] = React.useState(null);
  const toastSeen = React.useRef(new Set());

  const loadRuns = React.useCallback(() => {
    fetch("/api/session/runs").then(r => r.ok ? r.json() : [])
      .then(setRuns).catch(() => {});
  }, []);

  React.useEffect(() => {
    const tick = () => fetch("/api/auth/me").then(r => r.ok ? r.json() : null)
      .then(d => { if (d && d.authed) setMe(d); }).catch(() => {});
    tick();
    const t = setInterval(tick, 15000);
    return () => clearInterval(t);
  }, []);

  // know about downloadable runs before the popover ever opens — the
  // Results button stays greyed until there is actually something to get
  React.useEffect(() => { loadRuns(); }, [loadRuns, me && me.active_run]);  // eslint-disable-line

  React.useEffect(() => {
    if (!open) return;
    loadRuns();
    const t = setInterval(loadRuns, 10000);
    return () => clearInterval(t);
  }, [open, loadRuns]);

  // one-time "keep this tab open" toast per active run
  React.useEffect(() => {
    const rid = me && me.active_run;
    if (rid && !toastSeen.current.has(rid)) {
      toastSeen.current.add(rid);
      setToastRun(rid);
      const t = setTimeout(() => setToastRun(null), 14000);
      return () => clearTimeout(t);
    }
  }, [me && me.active_run]);  // eslint-disable-line

  const running = !!(me && me.active_run);
  const hasResults = runs.some(r => r.artifacts && Object.values(r.artifacts).some(Boolean));
  const enabled = hasResults || running;
  const btnTitle = enabled
    ? "Download your run results"
    : "No results yet — run a pipeline first. Downloads appear here as soon as a run finalizes.";
  const label = (r) => {
    const bits = [];
    if (r.papers != null) bits.push(`${r.papers} papers`);
    bits.push(r.modified);
    return bits.join(" · ");
  };
  const arts = (r) => {
    const a = r.artifacts || {};
    const items = [
      ["collection", "JSON"], ["bib", "BibTeX"],
      ["citation", "Graph"], ["collab", "Authors"],
    ].filter(([k]) => a[k]);
    return (
      <span className="pubx-links">
        {items.map(([k, name]) => (
          <a key={k} href={`/api/download/${r.run_id}/${k}`} download>{name}</a>
        ))}
        {items.length > 0 && <a href={`/api/download/${r.run_id}/zip`} download>ZIP</a>}
      </span>
    );
  };

  return (
    <div className="pubx-tb">
      <button className={"btn btn-ghost" + (open ? " is-on" : "") + (enabled ? "" : " pubx-off")}
              onClick={() => { if (enabled) setOpen(v => !v); }}
              aria-disabled={!enabled} title={btnTitle}>
        <Icon name="download" size={13} /> Results
        {running && <span className="pubx-dot" />}
      </button>
      {open && (
        <div className="pubx-pop">
          <div className="pubx-pop-head">
            <span>Your runs</span>
            <button className="pubx-x" onClick={() => setOpen(false)} aria-label="Close">
              <Icon name="x" size={12} />
            </button>
          </div>
          {me && (
            <div className="pubx-quota">
              {me.runs_today} / {me.runs_per_day} runs today · ≤{me.max_papers_ceiling} papers per run
            </div>
          )}
          <div className="pubx-list">
            {runs.length === 0 && (
              <div className="pubx-empty">No runs yet — results appear here as
                soon as a run finalizes, ready to download.</div>
            )}
            {runs.map(r => (
              <div key={r.run_id} className="pubx-row">
                <div className="pubx-row-top">
                  <span className="pubx-rid">{r.run_id.slice(0, 6)}</span>
                  <span className={"pubx-st pubx-st-" + r.status}>{r.status}</span>
                  <span className="pubx-meta">{label(r)}</span>
                </div>
                {arts(r)}
              </div>
            ))}
          </div>
        </div>
      )}
      {toastRun && (
        <div className="pubx-toast">
          <Icon name="alert-circle" size={12} />
          <span>
            Run in progress — <b>keep this tab open</b>. This server goes to
            sleep about a minute after the last visitor leaves, which would
            end the run (everything found so far is finalized + downloadable).
          </span>
        </div>
      )}
    </div>
  );
}

(function mountPublicExtras() {
  if (!window.__PUBLIC__) return;
  const style = document.createElement("style");
  style.textContent = `
    .pubx-tb { position: relative; display: flex; align-items: center;
               font-family: Inter, system-ui, sans-serif; }
    .pubx-off { opacity: 0.45; cursor: not-allowed; }
    .pubx-dot { width: 7px; height: 7px; border-radius: 50%; background: #2e7d43;
                margin-left: 5px; animation: pubx-pulse 1.6s ease-in-out infinite; }
    @keyframes pubx-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
    .pubx-pop { position: absolute; top: calc(100% + 10px); right: 0; z-index: 900;
                width: 330px; max-height: 56vh; overflow-y: auto; background: var(--cc-bg-0, #fff);
                border: 1px solid var(--cc-line-1, #d6d3cb); border-radius: 12px;
                box-shadow: 0 8px 28px rgba(20,18,12,0.13); padding: 12px 13px 10px; }
    .pubx-pop-head { display: flex; align-items: center; justify-content: space-between;
                     font: 600 12px Inter, sans-serif; letter-spacing: 0.02em;
                     color: var(--cc-ink-1, #1a1915); margin-bottom: 4px; }
    .pubx-x { border: none; background: none; cursor: pointer; color: inherit;
              padding: 3px; border-radius: 6px; }
    .pubx-x:hover { background: #eceae5; }
    .pubx-quota { font-size: 10.5px; color: var(--cc-ink-3, #8b877d); margin-bottom: 8px; }
    .pubx-empty { font-size: 12px; line-height: 1.5; color: var(--cc-ink-3, #8b877d);
                  padding: 8px 0 6px; }
    .pubx-row { padding: 8px 0 7px; border-top: 1px solid var(--cc-line-2, #eceae4); }
    .pubx-row-top { display: flex; align-items: baseline; gap: 7px; margin-bottom: 4px; }
    .pubx-rid { font: 600 11px "IBM Plex Mono", monospace; color: var(--cc-ink-1, #1a1915); }
    .pubx-st { font: 500 10px Inter, sans-serif; padding: 1px 6px; border-radius: 999px;
               background: #eceae5; color: #57544c; }
    .pubx-st-running, .pubx-st-starting { background: #e3efe6; color: #2e7d43; }
    .pubx-st-error { background: #f6e3df; color: #a03424; }
    .pubx-meta { font-size: 10.5px; color: var(--cc-ink-3, #8b877d); }
    .pubx-links { display: flex; flex-wrap: wrap; gap: 5px 10px; }
    .pubx-links a { font: 500 11px Inter, sans-serif; color: #3652a3; text-decoration: none; }
    .pubx-links a:hover { text-decoration: underline; }
    .pubx-toast { position: fixed; top: 50px; right: 14px; z-index: 899;
                  display: flex; gap: 8px; align-items: flex-start; width: 330px;
                  background: #fff8ec; border: 1px solid #e8d9b8; border-radius: 10px;
                  padding: 10px 12px; font-size: 11.5px; line-height: 1.5; color: #6b5722;
                  box-shadow: 0 4px 16px rgba(20,18,12,0.10); }
    .pubx-toast svg { flex: none; margin-top: 2px; }
  `;
  document.head.appendChild(style);
})();
