/**
 * BuildScreen — the Build screen (the demo component named "Paper Card").
 *
 * Composition only. Everything visible here is transplanted:
 *   markup    -> TopBar / BuildSidebar / BuildPipelineCanvas / BuildConfigPanel
 *                (mechanical JSX conversions of the demo template)
 *   CSS       -> ../../styles/demo-screens.css (all seven screens' verbatim
 *                helmet <style> blocks, in the demo's own head order — see the
 *                comment at the top of that file, this is load-bearing)
 *   behaviour -> ./build-logic.js (verbatim `class Component extends DCLogic`)
 *
 * The demo's template is NOT what the demo renders. On mount its own logic
 * rewrites large parts of the DOM: every `.pr` row is repainted through
 * paper-row.js (`prPaint`), the static 7-step example inside `.pl-flow` is
 * thrown away and rebuilt by `plInit`/`plRender`, the config panel's parameter
 * controls come from `plRenderParams`, and the sidebar grows to 91 rows via
 * `expandMock`. Reproducing that by hand would be a re-implementation, so the
 * demo's own class is transplanted instead and driven from the effect below.
 *
 * Data: the Build screen's fixtures live INSIDE the transplanted logic
 * (`mockPapers`, `plInit`'s pipeline, `plParamDefs`, `dlLatestRun`) and inside
 * the markup (the 14 seed `.pr` rows, the project list, the 5 `.cf-filter`
 * rows). None of it is in src/design-data, so there is nothing for the
 * DataSource to serve yet — H-BLD-01 in web/wiring/hardening-todo.md lists
 * exactly what has to move behind `DataSource` when Build is wired to CiteClaw.
 */

import { useEffect, useRef } from 'react';

// Hover paint for the demo's `style-hover` directive (compiled away by
// support.js in the demo; a real stylesheet here). Shared across screens, so it
// is imported by every screen that renders demo markup rather than by main.tsx.
import '../../styles/demo-style-hover.css';
// NOT just this screen's stylesheet. The demo renders Build with all seven
// screens' <style> blocks in one head, and later ones override Build's own —
// read the header of demo-screens.css before changing this import.
import '../../styles/demo-screens.css';

import { TopBar } from '../../components/shared/TopBar';
import { BuildSidebar } from './BuildSidebar';
import { BuildPipelineCanvas } from './BuildPipelineCanvas';
import { BuildConfigPanel } from './BuildConfigPanel';
import { BuildLogic } from './build-logic.js';

/**
 * Exactly the props the iPad Demo hands this screen.
 *
 * cardEmphasis / logoStyle / layout / pageState / pipelineStyle /
 * overscrollBounce come from the shell template ("KnowledgeLab iPad
 * Demo.dc.html" line 80 together with its renderVals, where `scheme` is pinned
 * to 'Warm paper' and `bounce` to false); cardLayout is the screen's own
 * data-props default. LOCKED — these are the defaults the product owner signed
 * off on. Changing any of them changes the picture.
 */
const BUILD_PROPS = Object.freeze({
  cardLayout: 'Inline index',
  pipelineStyle: 'Flow chart (6d)',
  colorScheme: 'Warm paper',
  logoStyle: 'Terracotta tile',
  layout: 'List',
  pageState: 'Has results',
  cardEmphasis: 'Muted',
  overscrollBounce: false,
});

export default function BuildScreen() {
  const rootRef = useRef<HTMLDivElement>(null);
  const { cardLayout, pipelineStyle } = BUILD_PROPS;

  useEffect(() => {
    const root = rootRef.current as (HTMLDivElement & { __klBuildLogic?: unknown }) | null;
    if (!root) return;
    // Guard, not cleanup. The transplanted componentDidMount installs document
    // listeners and rewrites the DOM, and the demo gave it no teardown. React
    // 18 StrictMode runs effects twice in dev against the SAME nodes, so a
    // second pass would double-wire. Writing the componentWillUnmount the demo
    // never had would be authoring behaviour, which this pass may not do.
    if (root.__klBuildLogic) return;
    const logic = new BuildLogic({ current: root }, BUILD_PROPS);
    root.__klBuildLogic = logic;
    logic.componentDidMount();

    // paper-row.js / import-resolver.js are loaded AFTER mount on purpose.
    //
    // The demo pulls them in with <script> tags that its helmet's __resources
    // shim appends to the head, so they land after the screen has mounted, and
    // the transplanted logic is built around exactly that: setupSidebar bails
    // out when window.KLPaperRow is missing and re-runs on `kl-paper-row-ready`.
    // The ordering is visible. seedRender clones the config panel's `.cs-row`
    // seed rows out of the sidebar, and in the demo it does so BEFORE prPaint
    // has repainted those rows through paper-row.js — so the seed rows carry the
    // template's own markup. Import these two eagerly and the clones come out
    // repainted through prCardInner instead, which is a different picture in
    // .cfg-seed-list. Measured against the demo's settled DOM; do not "tidy"
    // these into static imports.
    void import('../../design-data/paper-row.js');
    void import('../../design-data/import-resolver.js');
  }, []);

  return (
    // The screen's host inside the demo shell: `.pg` is the shell's page slot
    // (styled in src/styles/ipad-demo-shell.css), and the inner box is the
    // dc-import element the shell mounts "Paper Card" through, which carries
    // style="width:100%; height:100%".
    <div className="pg" data-pg="Build" data-screen-label="Build" data-on="1">
      <div style={{ width: '100%', height: '100%' }}>
        <div
          className="pc-root"
          ref={rootRef}
          data-cardlayout={cardLayout}
          data-theme="light"
          data-scheme="Recessed canvas"
          data-color="Warm paper"
          data-srcpage="set"
          data-pstyle={pipelineStyle}
          style={{ width: "100%", height: "100%", background: "var(--bg)", fontFamily: "'Hanken Grotesk',-apple-system,sans-serif", color: "var(--fg)", display: "flex", flexDirection: "column", overflow: "hidden", border: "1px solid var(--border)" }}
        >
          {"\n\n  "}
          <TopBar />
          {"\n\n  "}
          <div
            style={{ flex: "none", height: "var(--bn-inset, 0px)" }}
          />
          {"\n\n  "}
          <main
            style={{ position: "relative", flex: "1", minHeight: "0", display: "flex", gap: "var(--gut, 14px)", padding: "8px var(--gut, 14px) var(--gut, 14px)" }}
          >
            {"\n    "}
            <BuildSidebar />
            {"\n\n    "}
            <BuildPipelineCanvas />
            {"\n\n    "}
            <BuildConfigPanel />
            {"\n\n  "}
            {/* The template has a trailing "\n" text node before </div>; the
                dc-runtime compiler drops the last whitespace-only child, so the
                demo's .pc-root ends "…</main></div>". Matched here. */}
          </main>
        </div>
      </div>
    </div>
  );
}
