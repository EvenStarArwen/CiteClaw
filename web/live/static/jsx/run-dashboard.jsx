/* eslint-disable */
// Section D (Run mode) — Dashboard (live).
// Metric strip (always visible) + a tab-switched body: Overview (top
// rejections + cost), Rejections (every bucket, human-readable), Cost
// (source + token detail), Logs (full-width live log).

// "llm__anon.layer0.default.layer2" → "LLM · else · filter 3" etc. Raw
// bucket ids stay visible via the row tooltip.
function fmtReason(raw) {
  const KNOWN = {
    citation: "Citation (β)", year_filter: "Year range", similarity: "Similarity",
    abstract_keyword: "Abstract keywords", title_keyword: "Title keywords",
    venue_keyword: "Venue keywords", human_in_the_loop: "Human review",
  };
  if (KNOWN[raw]) return KNOWN[raw];
  let s = raw;
  let prefix = "";
  if (s.startsWith("llm_")) { prefix = "LLM"; s = s.slice(4); }
  const parts = s.split(".").filter(p => p && p !== "_anon" && !/^layer0$/.test(p));
  const bits = parts.map(p => {
    const mCase = p.match(/^case(\d+)$/);
    if (mCase) return "branch " + (Number(mCase[1]) + 1);
    if (p === "default") return "else";
    const mLayer = p.match(/^layer(\d+)$/);
    if (mLayer) return "filter " + (Number(mLayer[1]) + 1);
    return p.replace(/_/g, " ");
  });
  const label = [prefix, ...bits].filter(Boolean).join(" · ");
  return label || raw;
}

function BarList({ title, total, rows, fill, fmtVal }) {
  const max = rows.length ? Math.max(...rows.map(r => r.v), 1e-9) : 1;
  return (
    <div className="bars-col">
      <div className="bars-title">
        <span>{title}</span>
        <span className="bars-total">{total}</span>
      </div>
      <div className="bars-list">
        {rows.length === 0 && <div className="bar-row"><span className="bar-label">nothing yet</span></div>}
        {rows.map(r => (
          <div key={r.key} className="bar-row" title={r.tip || r.label}>
            <div className="bar-head">
              <span className={"bar-label" + (r.mono ? " mono" : "")}>{r.label}</span>
              <span className="bar-val">{fmtVal(r.v)}</span>
            </div>
            <div className="bar-track">
              <div className={"bar-fill " + fill} style={{ width: (r.v / max) * 100 + "%" }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function RunDashboard() {
  const [tab, setTab] = React.useState("overview");
  const metrics = useLive("metrics");
  const progress = useLive("progress");
  const logs = useLive("logs");
  const [openLogs, setOpenLogs] = React.useState(() => new Set());

  const rejections = metrics.rejectionReasons || [];
  const costs = metrics.costBySource || [];
  const rejTotal = metrics.rejected || 0;
  const costTotal = metrics.cost || 0;
  const tokTotal = (metrics.llmTokensIn || 0) + (metrics.llmTokensOut || 0);

  const rejRows = (list) => list.map(r => ({
    key: r.reason, label: fmtReason(r.reason), tip: r.reason, v: r.count,
  }));
  const costRows = costs.map(c => ({ key: c.source, label: c.source, v: c.cost }));

  const tagClass = (tag) =>
    tag === "ERR" ? "prog-log-tag--err"
    : tag === "WARN" || tag === "RETRY" ? "prog-log-tag--warn"
    : tag === "DONE" ? "prog-log-tag--ok"
    : tag === "S2" ? "prog-log-tag--s2"
    : tag === "LLM" ? "prog-log-tag--llm"
    : tag === "PHASE" ? "prog-log-tag--phase"
    : "prog-log-tag--info";
  const toggleLog = (id) => setOpenLogs(cur => {
    const next = new Set(cur);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  return (
    <section className="dashboard">
      <div className="dash-head">
        <span className="dash-head-title">Dashboard · {progress.current || "run"}</span>
        <div className="dash-tabs">
          {[["overview", "Overview"], ["rejects", "Rejections"], ["cost", "Cost"], ["logs", "Logs"]].map(([k, lbl]) => (
            <button key={k} className={"dash-tab" + (tab === k ? " on" : "")}
              onClick={() => setTab(k)}>{lbl}</button>
          ))}
        </div>
      </div>

      <div className="dash-metrics">
        <div className="metric">
          <span className="metric-label">Accepted</span>
          <span className="metric-value">{(metrics.accepted || 0).toLocaleString()}</span>
          <span className="metric-sub">papers</span>
        </div>
        <div className="metric">
          <span className="metric-label">Rejected</span>
          <span className="metric-value">{rejTotal.toLocaleString()}</span>
          <span className="metric-sub">{rejections.length} reasons</span>
        </div>
        <div className="metric">
          <span className="metric-label">LLM tokens</span>
          <span className="metric-value">{fmtK(tokTotal)}</span>
          <span className="metric-sub">{fmtK(metrics.llmTokensIn)} in · {fmtK(metrics.llmTokensOut)} out</span>
        </div>
        <div className="metric">
          <span className="metric-label">LLM calls</span>
          <span className="metric-value">{(metrics.llmCalls || 0).toLocaleString()}</span>
          <span className="metric-sub">{(metrics.llmCacheHits || 0).toLocaleString()} cache hits</span>
        </div>
        <div className="metric">
          <span className="metric-label">Cost</span>
          <span className="metric-value">${costTotal.toFixed(2)}</span>
          <span className="metric-sub">this run</span>
        </div>
        <div className="metric">
          <span className="metric-label">S2 cache</span>
          <span className="metric-value">{metrics.s2CacheHitPct || 0}%</span>
          <span className="metric-sub">{(metrics.s2CacheHits || 0).toLocaleString()} / {((metrics.s2Requests || 0) + (metrics.s2CacheHits || 0)).toLocaleString()} reqs</span>
        </div>
      </div>

      {tab === "overview" && (
        <div className="dash-bars">
          <BarList title="Top rejection reasons" total={rejTotal.toLocaleString() + " total"}
            rows={rejRows(rejections.slice(0, 5))} fill="reject" fmtVal={v => v.toLocaleString()} />
          <BarList title="Cost by source" total={"$" + costTotal.toFixed(2) + " total"}
            rows={costRows} fill="cost" fmtVal={v => "$" + v.toFixed(2)} />
        </div>
      )}

      {tab === "rejects" && (
        <div className="dash-bars dash-bars-single">
          <BarList title="Every rejection bucket — which filter rejected, and where in the cascade"
            total={rejTotal.toLocaleString() + " total"}
            rows={rejRows(rejections)} fill="reject"
            fmtVal={v => v.toLocaleString() + (rejTotal ? " · " + Math.round(100 * v / rejTotal) + "%" : "")} />
        </div>
      )}

      {tab === "cost" && (
        <div className="dash-bars">
          <BarList title="Cost by source" total={"$" + costTotal.toFixed(2) + " total"}
            rows={costRows} fill="cost" fmtVal={v => "$" + v.toFixed(4)} />
          <div className="bars-col">
            <div className="bars-title"><span>LLM &amp; API detail</span></div>
            <div className="dash-kv">
              <div><span>Input tokens</span><b>{(metrics.llmTokensIn || 0).toLocaleString()}</b></div>
              <div><span>Output tokens</span><b>{(metrics.llmTokensOut || 0).toLocaleString()}</b></div>
              <div><span>Reasoning tokens</span><b>{(metrics.llmReasoningTokens || 0).toLocaleString()}</b></div>
              <div><span>LLM calls</span><b>{(metrics.llmCalls || 0).toLocaleString()}</b></div>
              <div><span>LLM cache hits</span><b>{(metrics.llmCacheHits || 0).toLocaleString()}</b></div>
              <div><span>S2 API requests</span><b>{(metrics.s2Requests || 0).toLocaleString()}</b></div>
              <div><span>S2 cache hits</span><b>{(metrics.s2CacheHits || 0).toLocaleString()}</b></div>
              <div><span>Elapsed</span><b>{fmtDur(metrics.elapsedSec || 0)}</b></div>
            </div>
          </div>
        </div>
      )}

      {tab === "logs" && (
        <div className="dash-logs">
          {logs.length === 0 && <div className="prog-log-row"><span>no activity yet</span></div>}
          {logs.map(l => {
            const open = openLogs.has(l.id);
            return (
              <div key={l.id} className={"dash-log-row" + (open ? " open" : "")}
                onClick={() => toggleLog(l.id)}
                title={open ? "Click to collapse" : "Click to expand"}>
                <span className="prog-log-t">{l.t}</span>
                <span className={"prog-log-tag " + tagClass(l.tag)}>{l.tag}</span>
                <span className={"dash-log-msg" + (open ? " open" : "")}>{l.msg}</span>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

Object.assign(window, { RunDashboard });
