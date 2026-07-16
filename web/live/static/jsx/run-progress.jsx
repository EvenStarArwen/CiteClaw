/* eslint-disable */
// Section B (Run mode) — Pipeline progress (live).
// Steps + overall summary + live log come from the backend event stream.

function RunProgress() {
  const progress = useLive("progress");
  const logs = useLive("logs");
  const steps = progress.steps || [];

  const tagClass = (tag) => tag === "ERR" ? "prog-log-tag--err"
    : tag === "DONE" ? "prog-log-tag--ok" : "prog-log-tag--info";

  return (
    <aside className="panel panel-left">
      <div className="ph">
        <span className="ph-title">Pipeline progress</span>
        <span className="ph-count">{progress.done || 0} / {progress.total || 0}</span>
      </div>

      <div className="pb-scroll">
        <div className="progress-list">
          {steps.length === 0 && (
            <div className="prog-step is-idle">
              <span className="prog-badge">–</span>
              <div className="prog-body">
                <span className="prog-name">Waiting to start</span>
                <span className="prog-meta">press Run pipeline</span>
              </div>
            </div>
          )}
          {steps.map((s, i) => (
            <div key={s.localId || i} className={"prog-step is-" + s.status}>
              <span className="prog-badge">
                {s.status === "done" ? "✓" : i + 1}
              </span>
              <div className="prog-body">
                <span className="prog-name">{s.name}</span>
                <span className="prog-meta">{s.localId} · {s.sub}</span>
                {s.status === "active" && (
                  <div className="prog-bar-track">
                    <div className="prog-bar-fill" style={{ width: (s.pct || 50) + "%" }} />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="prog-summary">
          <div className="prog-summary-row">
            <span className="prog-summary-lbl">Overall</span>
            <span className="prog-summary-val">{progress.overallPct || 0}%</span>
          </div>
          <div className="prog-summary-row">
            <span className="prog-summary-lbl">Current</span>
            <span className="prog-summary-val">{progress.current || "—"}</span>
          </div>
          <div className="prog-summary-row">
            <span className="prog-summary-lbl">Steps</span>
            <span className="prog-summary-val">{progress.done || 0} / {progress.total || 0}</span>
          </div>
        </div>

        <div className="prog-log">
          <div className="prog-log-head">
            <span>Live log</span>
            <span className="prog-log-dot" aria-hidden="true" />
          </div>
          <div className="prog-log-body">
            {logs.length === 0 && (
              <div className="prog-log-row"><span>no activity yet</span></div>
            )}
            {logs.map((l, i) => (
              <div key={i} className="prog-log-row">
                <span className="prog-log-t">{l.t}</span>
                <span className={"prog-log-tag " + tagClass(l.tag)}>{l.tag}</span>
                <span>{l.msg}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
}

Object.assign(window, { RunProgress });
