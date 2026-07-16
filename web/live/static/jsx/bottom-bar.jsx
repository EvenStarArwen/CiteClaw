/* eslint-disable */
// Section F — Xcode-style status bar.

function BottomBar({ mode, running, pipelineLen, edgesLen, accepted, elapsed, progressPct,
                    metrics, current, done, total, seedsSelected }) {
  metrics = metrics || {};
  return (
    <footer className="statusbar">
      <span className="sb-item">
        <span className={"sb-dot " + (running ? "sb-dot-run" : "sb-dot-idle")} />
        <span className="sb-key">{running ? "Running" : "Idle"}</span>
      </span>
      <span className="sb-sep">·</span>
      {mode === "build" ? (
        <>
          <span className="sb-item">
            <Icon name="git-branch" size={11} style={{ color: "var(--cc-ink-3)" }} />
            <span className="sb-key">Session</span>
            <span className="sb-val">demo · v4</span>
          </span>
          <span className="sb-sep">·</span>
          <span className="sb-item">
            <span className="sb-key">Pipeline</span>
            <span className="sb-val">{pipelineLen} blocks</span>
            <span className="sb-key">·</span>
            <span className="sb-val">{edgesLen} edges</span>
          </span>
          <span className="sb-sep">·</span>
          <span className="sb-item">
            <span className="sb-key">Seeds</span>
            <span className="sb-val">{seedsSelected != null ? seedsSelected : 0} selected</span>
          </span>
        </>
      ) : (
        <>
          <span className="sb-item">
            <span className="sb-key">Step</span>
            <span className="sb-val">{(done || 0)} / {(total || 0)} · {current || "—"}</span>
          </span>
          <span className="sb-sep">·</span>
          <span className="sb-item">
            <span className="sb-key">Accepted</span>
            <span className="sb-val">{accepted.toLocaleString()}</span>
          </span>
          <span className="sb-sep">·</span>
          <span className="sb-item">
            <span className="sb-key">Elapsed</span>
            <span className="sb-val">{elapsed}</span>
          </span>
          <span className="sb-sep">·</span>
          <div className="sb-progress">
            <div className="sb-progress-fill" style={{ width: progressPct + "%" }} />
          </div>
          <span className="sb-val">{progressPct}%</span>
        </>
      )}
      <span className="sb-spacer" />
      <span className="sb-item">
        <Icon name="database" size={11} style={{ color: "var(--cc-ink-3)" }} />
        <span className="sb-key">S2 cache</span>
        <span className="sb-val">{metrics.s2CacheHitPct || 0}%</span>
      </span>
      <span className="sb-sep">·</span>
      <span className="sb-item">
        <Icon name="dollar-sign" size={11} style={{ color: "var(--cc-ink-3)" }} />
        <span className="sb-val">${(metrics.cost || 0).toFixed(2)}</span>
      </span>
      <span className="sb-sep">·</span>
      <span className="sb-item">
        <span className={"sb-dot " + (running ? "sb-dot-run" : "sb-dot-idle")} />
        <span className="sb-key">{running ? "running" : "connected"}</span>
      </span>
    </footer>
  );
}

Object.assign(window, { BottomBar });
