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
  const [runError, setRunError] = useState(null);
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

  // ---- Explore mode (selection, sort, facet filters, subtree lens) ----
  const explore = useLive("explore");
  const liveAccepted = useLive("accepted");
  const [exploreSort, setExploreSort] = useState("citations");
  const [exploreFilters, setExploreFilters] = useState(() => ({ ...XP_EMPTY_FILTERS }));
  const [exploreSelectedId, setExploreSelectedId] = useState(null);
  const [exploreSubtreeId, setExploreSubtreeId] = useState(null);

  // Citation (default) ↔ collaboration network switch. The author view is
  // loaded lazily per run (Finalize graphml or derived from the collection
  // JSON); the live session and GEXF drop-ins have no author lists. An
  // uploaded file fixes its own kind (a collaboration_network.graphml IS the
  // author view; a citation file has no author lists).
  const [exploreNet, setExploreNet] = useState("cite");
  const exploreCollab = explore.collab || { papers: [], edges: [] };
  const exploreUpload = explore.source === "upload" ? (explore.upload || null) : null;
  const uploadIsAuthor = !!(exploreUpload && exploreUpload.kind === "author");
  const collabEnabled = explore.source === "run";
  const exploreKind = exploreUpload
    ? (uploadIsAuthor ? "author" : "paper")
    : (exploreNet === "collab" ? "author" : "paper");
  useEffect(() => {
    if (exploreNet === "collab" && explore.source === "run" && explore.runPath) {
      loadExploreCollab(explore.runPath);
    }
    if (exploreNet === "collab" && explore.source !== "run") setExploreNet("cite");
  }, [exploreNet, explore.source, explore.runPath]);

  const exploreData = React.useMemo(() => {
    if (exploreUpload) {
      return { papers: exploreUpload.papers || [], edges: exploreUpload.edges || [] };
    }
    if (exploreNet === "collab") {
      return { papers: exploreCollab.papers || [], edges: exploreCollab.edges || [] };
    }
    return explore.source === "run"
      ? { papers: explore.papers, edges: explore.edges }
      : exploreFromLive();
  }, [exploreNet, exploreCollab, exploreUpload, explore.source, explore.version,
      liveNetwork.version, liveAccepted.length]);
  const exploreDataKey = exploreUpload
    ? "upload:" + exploreUpload.token + ":" + exploreUpload.name
    : (explore.source === "run" ? "run:" + explore.runPath : "live")
      + (exploreNet === "collab" ? ":collab" : "");
  // Nodes removed by the graph-side filters (degree / edge weight / largest
  // component) inside CiteGraph — mirrored into the list so both agree.
  const [exploreGraphHidden, setExploreGraphHidden] = useState(null);
  // Facet edits reach the simulation debounced — typing "250" into Min
  // citations re-laid-out the graph on every keystroke; the list stays live.
  const [exploreFiltersApplied, setExploreFiltersApplied] = useState(exploreFilters);
  useEffect(() => {
    const t = setTimeout(() => setExploreFiltersApplied(exploreFilters), 250);
    return () => clearTimeout(t);
  }, [exploreFilters]);
  const exploreFilterHidden = React.useMemo(() => {
    const pred = xpFilterPredicate(exploreFiltersApplied, exploreKind);
    const hidden = new Set();
    for (const p of exploreData.papers) if (!pred(p)) hidden.add(p.id);
    return hidden;
  }, [exploreData, exploreFiltersApplied, exploreKind]);
  const exploreSelectedPaper = exploreSelectedId
    ? exploreData.papers.find(p => p.id === exploreSelectedId) || null
    : null;

  // First entry into Explore: scan disk runs; if the live session is empty,
  // auto-load the freshest finished run so the page never opens blank.
  useEffect(() => {
    if (mode !== "explore") return;
    (async () => {
      const st = LIVE.get("explore");
      const runs = st.runsLoaded ? st.runs : await refreshExploreRuns();
      const ex = LIVE.get("explore");
      const liveEmpty = !(LIVE.get("network").nodes || []).length
        && !(LIVE.get("accepted") || []).length;
      if (ex.source === "live" && liveEmpty && runs.length) loadExploreRun(runs[0].path);
    })();
  }, [mode]);

  // Data source / network mode changed → stale selection/lens/filters must
  // not survive (author facets differ from paper facets).
  useEffect(() => {
    setExploreSelectedId(null);
    setExploreSubtreeId(null);
    setExploreGraphHidden(null);
  }, [exploreDataKey]);
  useEffect(() => {
    setExploreFilters({ ...XP_EMPTY_FILTERS });
    setExploreSort(exploreKind === "author" ? "papers" : "citations");
  }, [exploreKind]);

  const handleExplorePickSource = (v) => {
    setExploreNet("cite");
    if (v === "live") exploreUseLiveSession();
    else if (v === "upload") exploreUseUpload();
    else loadExploreRun(v);
  };
  const handleExploreFile = (file) => {
    setExploreNet("cite");
    loadExploreUpload(file);
  };
  const handleExploreSubtree = () => {
    if (!exploreSelectedPaper) return;
    setExploreSubtreeId(cur => cur === exploreSelectedPaper.id ? null : exploreSelectedPaper.id);
  };

  const patchSelectedConfig = (patch) => {
    setPipeline(ps => mapStep(ps, selectedId, n => ({ ...n, config: { ...n.config, ...patch } })));
  };
  const updateScreener = (newScreener) => {
    setPipeline(ps => mapStep(ps, selectedId, n => ({ ...n, screener: newScreener })));
  };

  // Pipeline construction — a single insertion primitive (insertAfter) drives
  // serial extend + parallel fan-out; addParallelBranch adds a branch (max 2);
  // removeStep prunes and collapses empties.
  const appendAfter = (anchorId, kind) => {
    const step = newStep(kind);
    setPipeline(ps => insertAfter(ps, anchorId, step));
    setSelectedId(step.id);
  };
  const appendParallel = (anchorId, kind) => {
    const step = newStep(kind);
    setPipeline(ps => insertAfter(ps, anchorId, newParallel([[step]])));
    setSelectedId(step.id);
  };
  const addBranch = (parId, kind) => {
    const step = newStep(kind);
    setPipeline(ps => addParallelBranch(ps, parId, step));
    setSelectedId(step.id);
  };
  const removeSelected = () => {
    setPipeline(ps => removeStep(ps, selectedId));
    setSelectedId(null);
  };

  // Top-bar actions — launch / stop a real run through the backend.
  // Validate first (client-side) so a missing key / no seeds pops a dialog
  // instead of silently switching to an empty Run view or wiping prior results.
  const handleRun = async () => {
    const st = LIVE.get("settings");
    const starred = (LIVE.get("seeds") || []).filter(s => s.starred);
    if (!starred.length) {
      setRunError("No seed papers selected. Star at least one paper (☆ → ★) in the Seeds panel before running.");
      return;
    }
    if (st.loaded) {
      const keys = st.keys || {};
      // Key requirements come from EVERY model the run will touch: the
      // Settings default plus each LLM filter's own override.
      const used = [{ model: st.model || "", where: "the default screening model (Settings)" }];
      forEachLlmFilter(pipeline, (f) => {
        const m = f.params && f.params.model;
        if (m) used.push({ model: m, where: "an LLM filter's model override" });
      });
      for (const u of used) {
        const m = u.model.toLowerCase();
        if (m.startsWith("gemini") && !keys.gemini_api_key) {
          setRunError(`No Gemini API key set — needed by ${u.where}. Open Settings (the gear, top-right) and add it before running.`);
          return;
        }
        if ((m.startsWith("gpt") || m.startsWith("o")) && !keys.openai_api_key) {
          setRunError(`No OpenAI API key set — needed by ${u.where}. Open Settings (the gear, top-right) and add it before running.`);
          return;
        }
      }
    }
    const res = await startRun(pipeline, LIVE.getState().seeds);
    if (res && res.ok) setMode("run");
    else setRunError((res && res.error) || "Could not start the run. Check your settings and try again.");
  };
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
        blocks={countSteps(pipeline)}
        accepted={liveMetrics.accepted}
        elapsed={fmtDur(liveMetrics.elapsedSec)}
        runStatus={runStatus}
        explorePapers={exploreData.papers.length}
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
          <BuildSeeds onSelectSeed={() => { const s = pipeline.find(el => el.kind === "seed"); if (s) setSelectedId(s.id); }} />
          <div className="main main-solo">
            <BuildPipeline
              pipeline={pipeline}
              selectedId={selectedId}
              setSelectedId={setSelectedId}
              blockStyle={blockStyle}
              onAppendAfter={appendAfter}
              onAppendParallel={appendParallel}
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
      ) : mode === "run" ? (
        <>
          <RunProgress />
          <div className="main" style={{ gridTemplateRows: `${(tweaks.runSplit ?? 0.62) * 100}% 6px 1fr` }}>
            <RunNetwork
              selectedPaperId={runSelectedPaperId}
              onSelectPaper={handleSelectRunPaper}
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
      ) : (
        <>
          <ExploreList
            papers={exploreData.papers}
            kind={exploreKind}
            selectedId={exploreSelectedId}
            onSelect={setExploreSelectedId}
            sort={exploreSort}
            setSort={setExploreSort}
            filters={exploreFilters}
            setFilters={setExploreFilters}
            explore={!exploreUpload && exploreNet === "collab"
              ? { ...explore, loading: exploreCollab.loading, error: exploreCollab.error }
              : explore}
            onPickSource={handleExplorePickSource}
            onPickFile={handleExploreFile}
            graphHidden={exploreGraphHidden}
          />
          <div className="main main-solo">
            <ExploreNetwork
              papers={exploreData.papers}
              edges={exploreData.edges}
              dataKey={exploreDataKey}
              kind={exploreKind}
              selectedId={exploreSelectedId}
              onSelect={setExploreSelectedId}
              subtreeId={exploreSubtreeId}
              onClearSubtree={() => setExploreSubtreeId(null)}
              filterHiddenIds={exploreFilterHidden}
              onGraphHidden={setExploreGraphHidden}
              netMode={exploreUpload ? (uploadIsAuthor ? "collab" : "cite") : exploreNet}
              onSwitchNet={setExploreNet}
              citeEnabled={!uploadIsAuthor}
              collabEnabled={collabEnabled}
              collabHint={exploreUpload
                ? (uploadIsAuthor
                    ? "This uploaded file IS a collaboration network"
                    : "Uploaded citation files carry no author lists")
                : explore.source === "live"
                  ? "Author view needs a finished run — pick one in the Papers panel"
                  : ""}
              emptyHint={!exploreUpload && exploreNet === "collab"
                ? (exploreCollab.loading
                    ? "Loading the collaboration network…"
                    : (exploreCollab.error || "No author data for this run."))
                : null}
              theme={theme}
            />
          </div>
          <ExploreDetail
            paper={exploreSelectedPaper}
            kind={exploreKind}
            onExploreSubtree={handleExploreSubtree}
            subtreeActive={!!exploreSelectedPaper && exploreSubtreeId === exploreSelectedPaper.id}
          />
        </>
      )}

      {showBottomBar && (
        <BottomBar
          mode={mode}
          running={running}
          pipelineLen={countSteps(pipeline)}
          edgesLen={Math.max(0, countSteps(pipeline) - 1)}
          accepted={liveMetrics.accepted}
          elapsed={fmtDur(liveMetrics.elapsedSec)}
          progressPct={Math.round(progressPct)}
          metrics={liveMetrics}
          current={liveProgress.current}
          done={liveProgress.done}
          total={liveProgress.total}
          seedsSelected={seedsSelected}
          explorePapers={exploreData.papers.length}
          exploreSource={explore.source === "run"
            ? explore.runPath
            : exploreUpload ? exploreUpload.name : "live session"}
        />
      )}

      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} pipeline={pipeline} />

      <RunErrorDialog
        error={runError}
        onClose={() => setRunError(null)}
        onOpenSettings={() => { setRunError(null); setSettingsOpen(true); }}
      />

      <TweaksPanel
        visible={tweaksVisible}
        tweaks={tweaks}
        setTweak={setTweak}
      />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
