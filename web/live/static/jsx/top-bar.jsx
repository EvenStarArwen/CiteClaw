/* eslint-disable */
// Section A — top bar.
// Claude-ish chrome: brand, mode segmented toggle (Build/Run),
// lightweight meta counters, primary action on the right.

function TopBar({ mode, setMode, running, onRun, onPause, onReset, onOpenSettings,
                 theme, onToggleTheme, blocks, accepted, elapsed, runStatus, explorePapers }) {
  return (
    <header className="topbar">
      <div className="tb-brand">
        <svg width="20" height="20" viewBox="0 0 48 46" fill="none" aria-hidden="true">
          <path fill="#863bff" d="M25.946 44.938c-.664.845-2.021.375-2.021-.698V33.937a2.26 2.26 0 0 0-2.262-2.262H10.287c-.92 0-1.456-1.04-.92-1.788l7.48-10.471c1.07-1.497 0-3.578-1.842-3.578H1.237c-.92 0-1.456-1.04-.92-1.788L10.013.474c.214-.297.556-.474.92-.474h28.894c.92 0 1.456 1.04.92 1.788l-7.48 10.471c-1.07 1.498 0 3.579 1.842 3.579h11.377c.943 0 1.473 1.088.89 1.83L25.947 44.94z"/>
        </svg>
        <span className="tb-brand-name">CiteClaw</span>
        <span className="tb-brand-sep">/</span>
        <span className="tb-brand-run">{runStatus && runStatus !== "idle" ? runStatus : "workspace"}</span>
      </div>

      <div className="mode-toggle" role="tablist">
        <button
          className={"mode-btn" + (mode === "build" ? " on" : "")}
          onClick={() => setMode("build")}
        >Build</button>
        <button
          className={"mode-btn" + (mode === "run" ? " on" : "")}
          onClick={() => setMode("run")}
        >Run</button>
        <button
          className={"mode-btn" + (mode === "explore" ? " on" : "")}
          onClick={() => setMode("explore")}
        >Explore</button>
      </div>

      <div className="tb-spacer" />

      {mode === "build" ? (
        <div className="tb-meta">
          <span className="tb-meta-num">{blocks != null ? blocks : 0}</span>
          <span className="tb-meta-lbl">blocks</span>
        </div>
      ) : mode === "explore" ? (
        <div className="tb-meta">
          <span className="tb-meta-num">{(explorePapers || 0).toLocaleString()}</span>
          <span className="tb-meta-lbl">papers</span>
        </div>
      ) : (
        <div className="tb-meta">
          <span className="tb-meta-num">{(accepted || 0).toLocaleString()}</span>
          <span className="tb-meta-lbl">accepted</span>
          <span style={{ color: "var(--cc-ink-faint)", opacity: 0.5, margin: "0 2px" }}>·</span>
          <span className="tb-meta-num">{elapsed || "00:00"}</span>
          <span className="tb-meta-lbl">elapsed</span>
        </div>
      )}

      <div className="tb-actions">
        <button
          className="btn btn-ghost btn-icon"
          onClick={onOpenSettings}
          title="API keys & model"
          aria-label="Settings"
        >
          <Icon name="settings" size={13} />
        </button>
        <button
          className="btn btn-ghost btn-icon"
          onClick={onToggleTheme}
          title={theme === "dark" ? "Switch to light" : "Switch to dark"}
          aria-label="Toggle theme"
        >
          <Icon name={theme === "dark" ? "sun" : "moon"} size={13} />
        </button>
        <button className="btn btn-ghost" onClick={onReset} title="Reset session">
          <Icon name="rotate-ccw" size={13} /> Reset
        </button>
        {running ? (
          <button className="btn" onClick={onPause}>
            <Icon name="pause" size={13} /> Pause
          </button>
        ) : (
          <button className="btn btn-primary" onClick={onRun}>
            <Icon name="play" size={13} /> Run pipeline
          </button>
        )}
      </div>
    </header>
  );
}

Object.assign(window, { TopBar });
