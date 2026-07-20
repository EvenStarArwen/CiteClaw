/* eslint-disable */
// Section B (Run mode) — Pipeline progress (live).
// The step list navigates: clicking a step OPENS ITS OWN PAGE (the same
// slide-in pattern as the paper-abstract view) showing the step's internal
// ROADMAP — its stages as a visual sub-pipeline with the live position
// highlighted and the progress bars embedded. The sidebar root keeps the
// step list + Current activity + summary + live log (click a line for the
// full message; only the last 100 are kept).

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

// The within-step live view (bars + retry + counter) — Current-activity card
// and the step-detail page both use it.
function ActivityDetail({ activity }) {
  return (
    <>
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
    </>
  );
}

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

// Which roadmap stage is live? Match the inner phase description against the
// stage keys; "__screen" wins when nothing else matches (screening phases
// carry the individual filter names).
function _liveStageKey(stages, activity) {
  const desc = activity && activity.inner && activity.inner.desc;
  if (!desc) return null;
  for (const st of stages) {
    if (st.key !== "__screen" && desc.startsWith(st.key)) return st.key;
  }
  return stages.some(st => st.key === "__screen") ? "__screen" : null;
}

// One roadmap node. `laneBar` is the node's own outer bar (source papers
// k/n for a Parallel branch sub-step) — kept visible after the lane
// finishes, so every branch card carries its progress; `liveBar` is the
// inner phase bar and only shows while the node is the live one.
function RoadNode({ index, label, hint, filters, state, liveBar, laneBar }) {
  return (
    <div className={"road-node is-" + state}>
      <span className="road-dot">
        {state === "done" ? "✓" : state === "active" ? "" : index}
      </span>
      <div className="road-body">
        <span className="road-label">{label}</span>
        {hint ? <span className="road-hint">{hint}</span> : null}
        {filters && filters.length > 0 && (
          <div className="road-filters">
            {filters.map((f, i) => <div key={i} className="road-filter">{f}</div>)}
          </div>
        )}
        {laneBar && <div className="road-live"><ActivityBar bar={laneBar} /></div>}
        {state === "active" && liveBar && <div className="road-live"><ActivityBar bar={liveBar} inner={!!laneBar} /></div>}
        {state === "active" && !liveBar && !laneBar && (
          <div className="road-live"><div className="prog-bar-track"><div className="prog-bar-fill act-bar-indet" /></div></div>
        )}
      </div>
    </div>
  );
}

// Full-page detail for one pipeline step: header, live bars, and the ROADMAP.
function StepDetailPage({ s, index, activity, running, lastEventAt, nowMs, onBack }) {
  React.useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onBack(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onBack]);

  const road = s.road || { stages: [], loop: false, blurb: "" };
  const isActive = s.status === "active";
  const liveKey = isActive ? _liveStageKey(road.stages || [], activity) : null;
  const stages = road.stages || [];
  const liveIdx = liveKey ? stages.findIndex(st => st.key === liveKey) : -1;
  const lane = isActive && activity ? activity.lane : null;
  const lanes = (isActive && activity && activity.lanes) || {};

  const stageState = (st, i) => {
    if (s.status === "done") return "done";
    if (s.status === "skipped" || s.status === "idle") return "pending";
    if (s.status === "error") return i < liveIdx ? "done" : i === liveIdx ? "error" : "pending";
    if (!isActive) return "pending";
    if (liveIdx < 0) return "pending";
    if (i === liveIdx) return "active";
    // looping steps revisit stages per source paper — only the current one
    // is meaningful; linear steps genuinely progress top to bottom.
    if (road.loop) return "pending";
    return i < liveIdx ? "done" : "pending";
  };

  const statusChip = { idle: "pending", active: "running", done: "done", skipped: "skipped", error: "failed" };

  return (
    <aside className="panel panel-left">
      <div className="ph">
        <span className="ph-title">Step detail</span>
        <span className="ph-count">{s.localId}</span>
      </div>
      <div className="seed-detail">
        <button className="seed-detail-back" onClick={onBack} title="Back (Esc)">
          <Icon name="arrow-left" size={13} /> Back to progress
        </button>
        <div className="seed-detail-body">
          <div className="sd-step-head">
            <span className={"prog-badge sd-badge is-" + s.status}>
              {s.status === "done" ? "✓" : s.status === "skipped" ? "→" : s.status === "error" ? "!" : index + 1}
            </span>
            <div className="sd-step-title">
              <span className="sd-step-name">{s.name}</span>
              <span className={"sd-step-chip is-" + s.status}>{statusChip[s.status] || s.status}</span>
            </div>
          </div>
          <div className="sd-step-sub">{s.localId} · {s.sub}</div>
          {road.blurb ? <div className="sd-step-blurb">{road.blurb}</div> : null}

          {isActive && (
            <div className="prog-activity sd-activity">
              <div className="prog-activity-head">
                <span>Live</span>
                <Heartbeat lastEventAt={lastEventAt} nowMs={nowMs} />
              </div>
              <ActivityBar bar={activity && activity.outer} />
              {activity && activity.retry && (
                <div className="act-retry"><Icon name="refresh-cw" size={11} /> {activity.retry}</div>
              )}
              {activity && activity.seen > 0 && (
                <div className="act-seen">{activity.seen.toLocaleString()} candidates seen</div>
              )}
            </div>
          )}

          {road.branches && road.branches.length > 0 ? (
            <div className="road">
              <div className="road-title">Branches</div>
              {road.branches.map((b, i) => (
                <div key={i} className="road-lane">
                  <div className="road-lane-head">Branch {i + 1}</div>
                  {b.map((node, j) => {
                    const li = lanes[node.key];
                    const nodeActive = !!((li && li.state === "run") || (lane && lane === node.key));
                    const nodeDone = s.status === "done" || !!(li && li.state === "done");
                    return (
                      <RoadNode key={j} index={j + 1} label={node.label} hint={node.hint}
                        state={nodeActive ? "active" : nodeDone ? "done" : "pending"}
                        laneBar={li ? li.outer : null}
                        liveBar={nodeActive && activity ? activity.inner : null} />
                    );
                  })}
                </div>
              ))}
            </div>
          ) : stages.length > 0 ? (
            <div className="road">
              <div className="road-title">
                Stages{road.loop ? <span className="road-loop-chip">↻ per source paper</span> : null}
              </div>
              {stages.map((st, i) => (
                <RoadNode key={st.key} index={i + 1} label={st.label} hint={st.hint}
                  filters={st.filters} state={stageState(st, i)}
                  liveBar={i === liveIdx && activity ? activity.inner : null} />
              ))}
            </div>
          ) : (
            <div className="sd-step-blurb">No internal stages for this step.</div>
          )}
        </div>
      </div>
    </aside>
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
  const [detailStep, setDetailStep] = React.useState(null);   // localId | null
  const [openLogs, setOpenLogs] = React.useState(() => new Set());

  const toggleLog = (id) => setOpenLogs(cur => {
    const next = new Set(cur);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  const tagClass = (tag) =>
    tag === "ERR" ? "prog-log-tag--err"
    : tag === "WARN" || tag === "RETRY" ? "prog-log-tag--warn"
    : tag === "DONE" ? "prog-log-tag--ok"
    : tag === "S2" ? "prog-log-tag--s2"
    : tag === "LLM" ? "prog-log-tag--llm"
    : tag === "PHASE" ? "prog-log-tag--phase"
    : "prog-log-tag--info";

  const detail = detailStep ? steps.find(x => x.localId === detailStep) : null;
  if (detail) {
    return <StepDetailPage s={detail} index={steps.indexOf(detail)}
      activity={activity} running={running}
      lastEventAt={lastEventAt} nowMs={nowMs}
      onBack={() => setDetailStep(null)} />;
  }

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
            <div key={s.localId || i}
              className={"prog-step is-" + s.status + " is-navigable"}
              onClick={() => setDetailStep(s.localId)}
              title="Open this step's roadmap">
              <span className="prog-badge">
                {s.status === "done" ? "✓" : s.status === "skipped" ? "→" : s.status === "error" ? "!" : i + 1}
              </span>
              <div className="prog-body">
                <span className="prog-name">{s.name}
                  <Icon name="chevron-right" size={10} className="prog-chev" />
                </span>
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
            <ActivityDetail activity={activity} />
          </div>
        )}

        <div className="prog-log">
          <div className="prog-log-head">
            <span>Live log</span>
            <span className="prog-log-n">
              {logs.length ? (logs.length >= 100 ? "last 100" : logs.length) : ""}
            </span>
            <span className="prog-log-dot" aria-hidden="true" />
          </div>
          <div className="prog-log-body">
            {logs.length === 0 && (
              <div className="prog-log-row"><span>no activity yet</span></div>
            )}
            {logs.map((l) => {
              const open = openLogs.has(l.id);
              return open ? (
                <div key={l.id} className="prog-log-open" onClick={() => toggleLog(l.id)}
                  title="Click to collapse">
                  <div className="prog-log-open-head">
                    <span className="prog-log-t">{l.t}</span>
                    <span className={"prog-log-tag " + tagClass(l.tag)}>{l.tag}</span>
                  </div>
                  <div className="prog-log-open-msg">{l.msg}</div>
                </div>
              ) : (
                <div key={l.id} className="prog-log-row is-clickable" onClick={() => toggleLog(l.id)}
                  title="Click to read the full message">
                  <span className="prog-log-t">{l.t}</span>
                  <span className={"prog-log-tag " + tagClass(l.tag)}>{l.tag}</span>
                  <span className="prog-log-msg">{l.msg}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </aside>
  );
}

Object.assign(window, { RunProgress });
