/* eslint-disable */
// Section D (Run mode) — Dashboard (live).
// Metric strip (6 metrics) + rejection & cost-by-source bar-lists, all from
// the backend metric snapshots.

function RunDashboard() {
  const [tab, setTab] = React.useState("overview");
  const metrics = useLive("metrics");
  const progress = useLive("progress");

  const rejections = metrics.rejectionReasons || [];
  const costs = metrics.costBySource || [];
  const rejTotal = metrics.rejected || 0;
  const costTotal = metrics.cost || 0;
  const rejMax = rejections.length ? Math.max(...rejections.map(r => r.count)) : 1;
  const costMax = costs.length ? Math.max(...costs.map(c => c.cost), 0.0001) : 1;
  const tokTotal = (metrics.llmTokensIn || 0) + (metrics.llmTokensOut || 0);

  return (
    <section className="dashboard">
      <div className="dash-head">
        <span className="dash-head-title">Dashboard · {progress.current || "run"}</span>
        <div className="dash-tabs">
          <button className={"dash-tab" + (tab === "overview" ? " on" : "")}
            onClick={() => setTab("overview")}>Overview</button>
          <button className={"dash-tab" + (tab === "rejects" ? " on" : "")}
            onClick={() => setTab("rejects")}>Rejections</button>
          <button className={"dash-tab" + (tab === "cost" ? " on" : "")}
            onClick={() => setTab("cost")}>Cost</button>
          <button className={"dash-tab" + (tab === "logs" ? " on" : "")}
            onClick={() => setTab("logs")}>Logs</button>
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

      <div className="dash-bars">
        <div className="bars-col">
          <div className="bars-title">
            <span>Top rejection reasons</span>
            <span className="bars-total">{rejTotal.toLocaleString()} total</span>
          </div>
          <div className="bars-list">
            {rejections.length === 0 && <div className="bar-row"><span className="bar-label">no rejections yet</span></div>}
            {rejections.map(r => (
              <div key={r.reason} className="bar-row">
                <div className="bar-head">
                  <span className="bar-label mono">{r.reason}</span>
                  <span className="bar-val">{r.count.toLocaleString()}</span>
                </div>
                <div className="bar-track">
                  <div className="bar-fill reject" style={{ width: (r.count / rejMax) * 100 + "%" }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bars-col">
          <div className="bars-title">
            <span>Cost by source</span>
            <span className="bars-total">${costTotal.toFixed(2)} total</span>
          </div>
          <div className="bars-list">
            {costs.length === 0 && <div className="bar-row"><span className="bar-label">no cost yet</span></div>}
            {costs.map(c => (
              <div key={c.source} className="bar-row">
                <div className="bar-head">
                  <span className="bar-label">{c.source}</span>
                  <span className="bar-val">${c.cost.toFixed(2)}</span>
                </div>
                <div className="bar-track">
                  <div className="bar-fill cost" style={{ width: (c.cost / costMax) * 100 + "%" }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

Object.assign(window, { RunDashboard });
