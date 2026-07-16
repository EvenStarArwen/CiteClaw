/* eslint-disable */
// CiteClaw WebUI — root app.
// Layout per sketch:
//   A = TopBar                      (top)
//   B = Build: BuildSeeds | Run: RunProgress   (left sidebar)
//   C = Build: BuildPipeline | Run: RunNetwork (center top)
//   D = Build: BuildConfig   | Run: RunDashboard (center bottom)
//   E = Build: BuildBlocks   | Run: RunAccepted  (right sidebar)
//   F = BottomBar                   (status bar)

function App() {
  const { useState, useEffect } = React;
  // Tweaks host protocol + defaults
  const [tweaks, setTweaks] = useState(() => ({ ...(window.__TWEAKS || {}) }));
  const [tweaksVisible, setTweaksVisible] = useState(false);

  useEffect(() => {
    const onMsg = (e) => {
      const d = e.data || {};
      if (d.type === "__activate_edit_mode") setTweaksVisible(true);
      else if (d.type === "__deactivate_edit_mode") setTweaksVisible(false);
    };
    window.addEventListener("message", onMsg);
    window.parent.postMessage({ type: "__edit_mode_available" }, "*");
    return () => window.removeEventListener("message", onMsg);
  }, []);

  const setTweak = (key, value) => {
    setTweaks(prev => {
      const next = { ...prev, [key]: value };
      window.parent.postMessage({ type: "__edit_mode_set_keys", edits: { [key]: value } }, "*");
      return next;
    });
  };

  // Mode (build / run) — Tweaks may override
  const mode = tweaks.mode || "build";
  const setMode = (m) => setTweak("mode", m);
  const blockStyle = tweaks.blockStyle || "specimen";
  const setBlockStyle = (v) => setTweak("blockStyle", v);
  const showBottomBar = tweaks.showBottomBar !== false;
  const theme = tweaks.theme === "dark" ? "dark" : "light";
  const palette = tweaks.palette || "citrus";
  const toggleTheme = () => setTweak("theme", theme === "dark" ? "light" : "dark");

  // Mono-specific component variants (only applied when palette === "mono")
  const monoRadius  = tweaks.monoRadius  || "soft";
  const monoPrimary = tweaks.monoPrimary || "black";
  const monoKind    = tweaks.monoKind    || "pill-black";
  const monoLeaf    = tweaks.monoLeaf    || "card";
  const monoSeed    = tweaks.monoSeed    || "card";
  const monoSeedFill = tweaks.monoSeedFill || "orange-bar";
  const monoCompLine = tweaks.monoCompLine || "gray";
  const monoRunPill  = tweaks.monoRunPill  || "black";
  const monoTrail    = tweaks.monoTrail    || "black";
  const monoAcc      = tweaks.monoAcc      || "black";
  const monoRunDot   = tweaks.monoRunDot   || "green";
  const monoCanvas   = tweaks.monoCanvas   || "paper";  // paper | snow | fog
  const monoAccent   = tweaks.monoAccent   || "ink";    // ink | indigo | cobalt | crimson

  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute("data-theme", theme);
    root.setAttribute("data-palette", palette);
    if (palette === "mono") {
      root.setAttribute("data-mono-radius",  monoRadius);
      root.setAttribute("data-mono-primary", monoPrimary);
      root.setAttribute("data-mono-kind",    monoKind);
      root.setAttribute("data-mono-leaf",    monoLeaf);
      root.setAttribute("data-mono-seed",    monoSeed);
      root.setAttribute("data-mono-seedfill", monoSeedFill);
      root.setAttribute("data-mono-compline", monoCompLine);
      root.setAttribute("data-mono-runpill",  monoRunPill);
      root.setAttribute("data-mono-trail",    monoTrail);
      root.setAttribute("data-mono-acc",      monoAcc);
      root.setAttribute("data-mono-rundot",   monoRunDot);
      root.setAttribute("data-mono-canvas",   monoCanvas);
      root.setAttribute("data-mono-accent",   monoAccent);
    } else {
      ["radius","primary","kind","leaf","seed","seedfill","compline","runpill","trail","acc","rundot","canvas","accent"]
        .forEach(k => root.removeAttribute("data-mono-" + k));
    }
  }, [theme, palette, monoRadius, monoPrimary, monoKind, monoLeaf, monoSeed, monoSeedFill, monoCompLine, monoRunPill, monoTrail, monoAcc, monoRunDot, monoCanvas, monoAccent]);

  // Run state — driven by the live backend store (see live-store.jsx)
  const running = useLive("running");
  const liveProgress = useLive("progress");
  const liveMetrics = useLive("metrics");
  const liveNetwork = useLive("network");
  const liveSeeds = useLive("seeds");
  const runStatus = useLive("runStatus");
  const progressPct = liveProgress.overallPct || 0;
  const seedsSelected = liveSeeds.filter(s => s.starred).length;
  const [settingsOpen, setSettingsOpen] = useState(false);
  useEffect(() => { refreshSettings(); }, []);

  // Pipeline state (Build mode)
  const [pipeline, setPipeline] = useState(window.INITIAL_PIPELINE);
  const [selectedId, setSelectedId] = useState("n3");  // Backward selected by default

  // Run-mode paper selection (shared by network + accepted list)
  const [runSelectedPaperId, setRunSelectedPaperId] = useState(null);
  const [runHoverPaperId, setRunHoverPaperId] = useState(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const handleSelectRunPaper = (id) => {
    setRunSelectedPaperId(id);
    setDetailOpen(!!id);
  };
  const selectedNode = findStep(pipeline, selectedId);

  const patchSelectedConfig = (patch) => {
    setPipeline(ps => mapStep(ps, selectedId, n => ({ ...n, config: { ...n.config, ...patch } })));
  };
  const updateScreener = (newScreener) => {
    setPipeline(ps => mapStep(ps, selectedId, n => ({ ...n, screener: newScreener })));
  };

  // Pipeline construction (serial extend / parallel branch / remove).
  const addSerial = (kind) => {
    const step = newStep(kind);
    setPipeline(ps => [...ps, step]);
    setSelectedId(step.id);
  };
  const wrapParallel = (kind) => {
    setPipeline(ps => {
      if (ps.length === 0) return ps;
      const last = ps[ps.length - 1];
      const fresh = newStep(kind);
      if (last.kind === "parallel") {
        return [...ps.slice(0, -1), { ...last, branches: [...last.branches, fresh] }];
      }
      if (last.kind === "seed") return [...ps, fresh];  // can't parallel the seed
      setSelectedId(fresh.id);
      return [...ps.slice(0, -1), newParallel([last, fresh])];
    });
  };
  const addBranch = (rowId, kind) => {
    const fresh = newStep(kind);
    setPipeline(ps => mapStep(ps, rowId, row =>
      row.kind === "parallel" ? { ...row, branches: [...row.branches, fresh] } : row));
    setSelectedId(fresh.id);
  };
  const removeSelected = () => {
    setPipeline(ps => removeStep(ps, selectedId));
    setSelectedId(null);
  };

  // Top-bar actions — launch / stop a real run through the backend
  const handleRun = () => { setMode("run"); startRun(pipeline, LIVE.getState().seeds); };
  const handlePause = () => { stopRun(); };
  const handleReset = () => { stopRun(); };

  useEffect(() => {
    if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.75 } });
  });

  return (
    <div
      className={"app" + (showBottomBar ? "" : " no-foot")}
      style={{
        "--cc-col-left": (tweaks.colLeft ?? 248) + "px",
        "--cc-col-right": (tweaks.colRight ?? 312) + "px",
      }}
    >
      <TopBar
        mode={mode}
        setMode={setMode}
        running={running}
        onRun={handleRun}
        onPause={handlePause}
        onReset={handleReset}
        onOpenSettings={() => setSettingsOpen(true)}
        theme={theme}
        onToggleTheme={toggleTheme}
        blocks={pipeline.length}
        accepted={liveMetrics.accepted}
        elapsed={fmtDur(liveMetrics.elapsedSec)}
        runStatus={runStatus}
      />

      <ColSplitter
        side="left"
        value={tweaks.colLeft ?? 248}
        onChange={(v) => setTweak("colLeft", v)}
        min={220}
        max={460}
      />
      <ColSplitter
        side="right"
        value={tweaks.colRight ?? 312}
        onChange={(v) => setTweak("colRight", v)}
        min={260}
        max={560}
      />

      {mode === "build" ? (
        <>
          <BuildSeeds />
          <div className="main main-solo">
            <BuildPipeline
              pipeline={pipeline}
              selectedId={selectedId}
              setSelectedId={setSelectedId}
              blockStyle={blockStyle}
              onAddSerial={addSerial}
              onWrapParallel={wrapParallel}
              onAddBranch={addBranch}
            />
          </div>
          <BuildStepConfig
            node={selectedNode}
            onPatchConfig={patchSelectedConfig}
            onUpdateScreener={updateScreener}
            onRemove={removeSelected}
          />
        </>
      ) : (
        <>
          <RunProgress />
          <div className="main" style={{ gridTemplateRows: `${(tweaks.runSplit ?? 0.62) * 100}% 6px 1fr` }}>
            <RunNetwork
              key={liveNetwork.version}
              selectedPaperId={runSelectedPaperId}
              onSelectPaper={handleSelectRunPaper}
              hoverPaperId={runHoverPaperId}
              onHoverPaper={setRunHoverPaperId}
              theme={theme}
            />
            <PaneSplitter
              value={tweaks.runSplit ?? 0.62}
              onChange={(v) => setTweak("runSplit", v)}
            />
            <RunDashboard />
          </div>
          <RunAccepted
            selectedPaperId={runSelectedPaperId}
            onSelectPaper={handleSelectRunPaper}
            detailOpen={detailOpen}
            onCloseDetail={() => setDetailOpen(false)}
          />
        </>
      )}

      {showBottomBar && (
        <BottomBar
          mode={mode}
          running={running}
          pipelineLen={pipeline.length}
          edgesLen={pipeline.length - 1}
          accepted={liveMetrics.accepted}
          elapsed={fmtDur(liveMetrics.elapsedSec)}
          progressPct={Math.round(progressPct)}
          metrics={liveMetrics}
          current={liveProgress.current}
          done={liveProgress.done}
          total={liveProgress.total}
          seedsSelected={seedsSelected}
        />
      )}

      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />

      <TweaksPanel
        visible={tweaksVisible}
        tweaks={tweaks}
        setTweak={setTweak}
      />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
