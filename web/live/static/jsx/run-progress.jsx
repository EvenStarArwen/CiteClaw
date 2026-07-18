/* eslint-disable */
// Section B (Run mode) — Pipeline progress (live).
// Steps + CURRENT ACTIVITY (the CLI's double progress bar: outer bar over
// source papers, inner bar per phase, S2 retry banner, liveness heartbeat)
// + overall summary + live log, all from the backend event stream.

// One activity bar: label · counts · determinate fill (or an indeterminate
// shimmer when the total is unknown, e.g. paginated fetches).
function ActivityBar({ bar, inner }) {
  if (!bar) return null;
  const hasTotal = bar.total != null && bar.total > 0;
  const pct = hasTotal ? Math.min(100, (100 * (bar.done || 0)) / bar.total) : 0;
  return (
    <div className={"act-bar" + (inner ? " act-bar-inner" : "")}>
      <div className="act-bar-row">
        <span className="act-bar-desc">{bar.desc}</span>
        <span className="act-bar-n">
          {hasTotal
            ? `${(bar.done || 0).toLocaleString()} / ${bar.total.toLocaleString()}`
            : (bar.done || 0) > 0 ? (bar.done).toLocaleString() : "…"}
        </span>
      </div>
      <div className="prog-bar-track">
        {hasTotal
          ? <div className="prog-bar-fill" style={{ width: pct + "%" }} />
          : <div className="prog-bar-fill act-bar-indet" />}
      </div>
    </div>
  );
}

// Liveness heartbeat: seconds since the last backend event, colour-coded so
// "is it dead?" has an answer at a glance.
function Heartbeat({ lastEventAt, nowMs }) {
  const ago = Math.max(0, Math.round((nowMs - (lastEventAt || nowMs)) / 1000));
  const state = ago < 10 ? "ok" : ago < 60 ? "slow" : "stale";
  const label = ago < 10 ? `live · ${ago}s`
    : ago < 60 ? `quiet · ${ago}s since last event`
    : `no events for ${ago}s — long API call, backoff, or stall`;
  return (
    <span className={"act-heart is-" + state} title="Time since the last event from the pipeline — green: streaming; amber: mid API call or backoff; red: check the server terminal">
      <span className="act-heart-dot" /> {label}
    </span>
  );
}

function RunProgress() {
  const progress = useLive("progress");
  const logs = useLive("logs");
  const activity = useLive("activity");
  const running = useLive("running");
  const lastEventAt = useLive("lastEventAt");
  const nowMs = useLive("nowMs");
  const steps = progress.steps || [];

  const tagClass = (tag) =>
    tag === "ERR" ? "prog-log-tag--err"
    : tag === "WARN" || tag === "RETRY" ? "prog-log-tag--warn"
    : tag === "DONE" ? "prog-log-tag--ok"
    : tag === "S2" ? "prog-log-tag--s2"
    : tag === "LLM" ? "prog-log-tag--llm"
    : tag === "PHASE" ? "prog-log-tag--phase"
    : "prog-log-tag--info";

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
                    <div className="prog-bar-fill act-bar-indet" />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {running && (
          <div className="prog-activity">
            <div className="prog-activity-head">
              <span>Current activity</span>
              <Heartbeat lastEventAt={lastEventAt} nowMs={nowMs} />
            </div>
            {activity && activity.retry && (
              <div className="act-retry" title="The Semantic Scholar client is retrying with backoff — the run is alive, just waiting out a rate limit or a flaky response">
                <Icon name="refresh-cw" size={11} /> {activity.retry}
              </div>
            )}
            <ActivityBar bar={activity && activity.outer} />
            <ActivityBar bar={activity && activity.inner} inner />
            {!(activity && (activity.outer || activity.inner)) && (
              <div className="act-idle">
                <div className="prog-bar-track"><div className="prog-bar-fill act-bar-indet" /></div>
                <span>working — waiting for the step to report…</span>
              </div>
            )}
            {activity && activity.seen > 0 && (
              <div className="act-seen">{activity.seen.toLocaleString()} candidates seen this step</div>
            )}
          </div>
        )}

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
            <span className="prog-log-n">{logs.length ? logs.length + (logs.length >= 200 ? "+" : "") : ""}</span>
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
                <span className="prog-log-msg" title={l.msg}>{l.msg}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
}

Object.assign(window, { RunProgress });
