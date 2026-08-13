# iPad Pro 12.9" demo — audit & action plan

Audited: 2026-08-11. Scope: `KnowledgeLab App Fluid.dc.html` composition (Login, Home, Build/Paper Card, Runs, Explore, Settings, System Banners) + `network-viz.js` / `topic-viz.js` + packaging for Safari on iPadOS.
Read CLAUDE.md first. Edit pages in their own files, then **re-bundle** (see § Demo artifacts). Target device: iPad Pro 12.9 — 1366 × 1024 pt landscape, 1024 × 1366 portrait.

Legend: [DONE] fixed this session · [P0] demo-blocking · [P1] felt immediately in anyone's hands · [P2] polish.

## Decisions locked (2026-08-11, user)

- **Landscape only** — Explore's narrow tier stays deferred; demo entry gets a rotate-device notice at narrow widths.
- **Hosted URL** launch via Netlify (runbook below); `deploy/` folder is prepped (`index.html` = the bundle + `apple-touch-icon.png`).
- **Audience is hands-on**; Guided Access is the guard — no extra in-app lockdown beyond the P1 touch work.
- **Hidden demo-state switcher: build it.** Trigger: 5 quick taps on the logo. Scope: page-state flips for Build / Runs / Explore + a "Reset demo" action ONLY (no connection/key-probe controls — too risky in audience hands).
- **Fix scope: everything P0–P2.** Demo script pages: Home, Build, Runs, Explore, Settings (Login/banners lower priority).
- **Auto-sync rule:** any session that edits a shipped page/module MUST regenerate the bundle AND refresh `deploy/index.html` in the same turn (also recorded in CLAUDE.md).

## The two headline questions

**"Wrap as a file, open in Safari, Add to Home Screen, full-screen" — true?** Half. Wrapping into ONE self-contained offline HTML file: true, and done (`KnowledgeLab iPad Demo.html`). But iPadOS Safari cannot open local files (no file:// browsing; Files app opens HTML in a preview sheet with a Done bar, and there is no Add to Home Screen from there). Add to Home Screen only works from a real URL open in Safari. Served over HTTP(S), the entry file's Apple meta tags (added) make the home-screen icon launch truly chrome-less full-screen. See § Launch paths.

**"Fully supports touch?"** No — solid foundation (pointer events, tap alternatives for every drag-and-drop, scrim-closable overlays), but real gaps: no pinch-zoom on any canvas, canvas/slider drags fight page scrolling (missing `touch-action`), hover-only reveals invisible on touch, input-focus auto-zoom (fixed in demo entry), portrait Explore loses the Agent panel. Everything is itemized below with fixes.

## What already works on touch (verified in code)

- All activation is `click`-based → taps work everywhere; panels scroll natively with `overscroll-behavior` containment already set (`.sb-list`, `.th-scroll`, `.rv-scroll`, …).
- Both canvas engines use **Pointer Events** (`pointerdown/move/up` + `setPointerCapture`) → single-finger node-drag / pan / tap-select work once `touch-action` is set (P0-1). Zoom fallbacks exist: −/+/⛶ header buttons.
- Config-popover sliders (`.nv-sl`, `.tm-sl`) already carry `touch-action:none` ✓.
- Every HTML5 drag-and-drop has a tap alternative: import dropzones have Browse + "use a sample"; agent-panel attach has ⊕ menu + `data-psel` multi-select + selection bar; pipeline presets menu for reorder-averse users.
- Popovers/modals close on scrim/outside click (document `click` listeners fire for taps on iPadOS); Esc is never the only close path.
- `localStorage` access is try/catch-guarded (`kl-filter-groups`).
- Pages self-measure with ResizeObserver element queries (`data-vp`) → real device widths engage tiers correctly, no host needed.
- Fluid shell uses `100dvh` (correct for Safari's dynamic chrome) and `min-width:1024px` (= iPad portrait width exactly).

## Launch paths (pick one; A recommended)

**A. Hosted URL → Add to Home Screen (true full-screen).** Put `KnowledgeLab iPad Demo.html` on any static HTTPS host. Safari on iPad → open URL → Share → Add to Home Screen. The icon launches with zero browser chrome (status bar only): the entry already carries `apple-mobile-web-app-capable`, `apple-mobile-web-app-status-bar-style: black`, `apple-mobile-web-app-title`, `viewport-fit=cover`. Each launch re-reads the URL (updates = redeploy, no re-install). Needs network at launch (file is self-contained, so nothing else is fetched).

**B. Mac on the same Wi-Fi.** In the project folder: `python3 -m http.server 8080` → on iPad open `http://<mac-name>.local:8080/KnowledgeLab%20iPad%20Demo.html` → Add to Home Screen. Same full-screen result; Mac must be reachable at every launch.

**C. Fully offline, zero infrastructure — DOES NOT WORK for this app.** Verified on device: AirDrop / Files opens HTML in iPadOS Quick Look, which **disables JavaScript** — only the K splash (the bundle's no-JS fallback) renders. There is no way to open a local HTML file in real Safari on iPadOS. Use path A or B; a third-party kiosk-browser app that opens local files is the only true-offline option.

Also worth doing: **[P1-A] in-app "Enter full screen"** row in the account menu (`document.documentElement.webkitRequestFullscreen?.() || requestFullscreen()`, only shown when `matchMedia('(hover:none)')` or not `navigator.standalone`) — chrome-less fullscreen from plain Safari without any home-screen install. Covers paths B/C.

Demo-day resilience: Settings → Display → Auto-Lock: Never; for hands-on audiences enable **Guided Access** (blocks edge-swipes/app-switch). External keyboard optional; all flows must pass touch-only.

## Deploy runbook (Netlify, ~2 minutes)

1. One-time: create a free Netlify account (a claimed site keeps a stable URL across updates).
2. Drag the **`deploy/` folder** onto https://app.netlify.com/drop → you get `https://<name>.netlify.app`.
3. iPad Safari → open that URL → Share → **Add to Home Screen** → launch from the icon: chrome-less full screen (the K tile is the icon).
4. Updating: after any re-bundle, refresh `deploy/index.html` (the auto-sync rule does this) and drag `deploy/` onto the site's Deploys page again — same URL, icon keeps working.

Alternative hosts work identically (GitHub Pages, S3, Vercel) — the only requirements are HTTPS and `apple-touch-icon.png` sitting next to `index.html`.

## Demo artifacts (created this session — [DONE])

- **`KnowledgeLab iPad Demo.dc.html`** — demo entry. It is the Fluid shell plus: `viewport` tweak defaults to **Fill the window** (100% × 100dvh); hardened viewport meta (`maximum-scale=1, user-scalable=no` — kills iOS input-focus auto-zoom, which would otherwise fire on every 12.5–13px input in the app); Apple standalone metas + `<title>`; `__bundler_thumbnail` splash; base touch CSS (`-webkit-tap-highlight-color:transparent`, `overscroll-behavior:none`, `touch-action:manipulation` on buttons/tabs); the **ext-resource-dependency manifest** (9 entries) for every module the pages load via dynamic `import()`.
- **`KnowledgeLab iPad Demo.html`** — the compiled 6.9 MB self-contained offline bundle. Never edit it; regenerate: `super_inline_html({input: "KnowledgeLab iPad Demo.dc.html", output: "KnowledgeLab iPad Demo.html"})` after ANY page edit, then copy it over `deploy/index.html`. **This is the auto-sync rule: same turn as the page edit, every time.**
- **Bundle-resolution rules (found the hard way; both regressions reached a user build):** the bundler only sees the ENTRY file; everything the pages reference gets into the single file via these three mechanisms, and NOTHING else works:
  1. **Dynamic module imports** (Explore/Runs/Login, [DONE]): literal-first with bundle fallback — `import('./x.js').catch(function(){return import((window.__resources||{})['id'])})` + a matching `ext-resource-dependency` meta in the demo entry. Never `import(resources || literal)` in one call — that broke LIVE resolution.
  2. **Local helmet scripts** (paper-row.js, import-resolver.js, run-mock.js, [DONE]): support.js clones helmet `<script src>` tags verbatim — a relative src 404s at file:// and on a host. Pages now mount them through the shared inline shim (`s.src = __resources['paperrow'] || './paper-row.js'`; identical shim text in every page, so helmet dedup mounts it once) + metas in the demo entry. Any NEW local helmet script must join that shim + manifest.
  3. **Helmet stylesheet links** (Google Fonts, KaTeX css): auto-handled — the bundler crawls link hrefs across the dc-import chain and support.js inlines them from `__resources[href]`. No action needed.
  - **CDN helmet scripts** (marked / KaTeX / turndown / smooth-scrollbar js) stay network-loaded — the runtime has no bundle path for them; fine because the demo launch path is hosted/online. A truly offline demo would need them shimmed like (2).

Bundle verification status: the user's Mac file:// test (2026-08-11) caught what my project-served sandbox structurally cannot — relative URLs resolve against the project here, so missing-from-bundle assets are masked. That round: paper-row.js / import-resolver.js were absent → Runs/Explore lists empty, Build sidebar wiring threw at the first row and died (search/filter/sort/Import all dead). Fixed via mechanism (2) above; manifest ids verified in the compiled file. Expected residue: one masked detail-free `Script error.` (toast-guarded, no symptom), and each literal import 404s once before falling back — harmless. **Re-run the P0-4 click-through after every bundling-mechanism change.**

## P0 — demo-blocking

1. **`touch-action:none` on all four interaction canvases** — Runs citation network, Explore citation network, Explore topic map, Login background. Without it, iPadOS treats a canvas drag as page scroll and fires `pointercancel`: node-drag/pan dies mid-gesture and the page rubber-bands. Set at canvas creation (inline style in `network-viz.js` / `topic-viz.js` constructors: `this.canvas.style.touchAction = 'none'` — one line covers every mount) + `-webkit-user-select:none` on the canvas hosts. Acceptance: dragging on canvas never scrolls the page; drags never abort.
2. **Pinch-zoom + two-finger pan in both viz engines** (`network-viz.js`, `topic-viz.js`). Recipe: track active pointers in a Map keyed by `pointerId` in `_down/_move/_up/_cancel`; when 2 pointers are live, suspend `_dragN`/`_pan` and drive zoom from inter-pointer distance about the midpoint (reuse `_wheel`'s world-anchor math), pan from midpoint delta; on returning to 1 pointer, re-seed. Add **double-tap = fit** detection in `_up` (two non-moved taps < 300 ms, < 24 px apart → `fit(true)`) — `dblclick` is unreliable on iPad. Keep `wheel` for desktop. ~60 lines per file. Acceptance: pinch zooms about the fingers on iPad; single-finger drag/tap-select unchanged on desktop.
3. **`pointercancel` = abort** for every window/document-level drag: both viz engines, `.dr-h` range handles (Build/Runs/Explore), pipeline reorder (`wireCfDrag`, `plDragWire` in Paper Card), Runs `rscDragStart`, Explore/Runs `.nv-sl` sliders. Anywhere `pointerup` tears down, `pointercancel` must too, else a system gesture mid-drag leaves a stuck ghost row / zombie listener.
4. **Click through all five pages + Settings in the BUNDLE** on Mac Safari (closest engine to iPadOS), then on the iPad. My sandbox's capture bridge times out on this composition (three concurrent force sims saturate it — see P1-9), so Runs/Explore/Login inside the bundle are code-verified but not screen-verified.
5. **Portrait: DECIDED landscape-only.** Landscape 1366 → compact tier (≤1440): every panel present — works today. Portrait 1024 → narrow tier, where **Explore's Agent panel hides** (narrow tier deferred). Task: add a rotate-device notice to the **demo entry only** (overlay when root width < ~1160, serif invite style, "Rotate to landscape"), so a hands-on visitor who rotates the iPad isn't confused. Explore's real narrow tier stays a separate, non-demo work item.

## P1 — felt in anyone's hands

6. **Filter range sliders `.dr-h`** (year/citations, in Build sidebar + Runs/Explore filter panels): 16 px handles, no `touch-action`. Add `touch-action:none` on the `.dr` track container; extend each handle's hit area to ≥ 32 px (transparent `::after` or padding-box) keeping the 16 px visual. Handles already use pointer events + window-level move ✓.
7. **Hover-reveals invisible on touch** — add `@media (hover:none)` always-visible rules (reduced opacity is fine) per page: `.rsc-drag` reorder grip (Runs re-screen; `opacity:0` until `.rsc-row:hover`), row ⋯ menus (Runs run rows; Home project/collection rows), paper-row star (JS-driven opacity — gate with `matchMedia('(hover:none)')` at paint: always visible), `.cc-open` arrow in In-this-corpus, abstract chevron trigger, `Changes · N` style hover-only affordances. Rule of thumb: **on touch, anything actionable is visible at rest.**
8. **Sticky hover paint.** Dozens of `mouseenter/mouseleave` inline-style pairs (rows, buttons, cards) — on iPad the first tap applies hover paint and it sticks until tapping elsewhere. Cheap mitigation: in each page's wiring helpers, skip attaching the mouseenter/leave cosmetic paints when `matchMedia('(hover:none)')` (behavior/click paths untouched; `:active` still gives touch feedback). Do the shared helpers first: paper-row hover paint (`row.__paint`), ghost-button pairs. Acceptable to defer stragglers — it's cosmetic, not blocking.
9. **Pause sims on hidden pages (battery/heat/perf).** All five pages stay mounted; Login bg + Runs network + Explore network/topic map each run their own rAF loop — measured cost: this composition saturates a mid-power WebKit iframe. `CitationNetwork` already has `setVisible()` (loop-gates + resumes); wire it: shells' `show()` already flips `.pg[data-on]` — dispatch a document event (`kl-page-shown {name}`) there; each page listens and calls `nv.setVisible(on)` / topic-map equivalent (add `setVisible` to `TopicMap` if missing — same 3 lines as network-viz). Login should stop its sim after `kl-login`. Acceptance: only the visible page's canvas loop runs.
10. **Keyboard avoidance.** Focusing the Explore composer / search fields raises the on-screen keyboard (landscape ≈ half the screen). Panels scroll internally so it's likely acceptable, but test; if the composer is obscured, listen to `visualViewport` resize and adjust the panel's `scrollTop` (project rule: never `scrollIntoView`). Demo tip: a paired hardware keyboard sidesteps this entirely.
11. **Text selection & callouts.** Long-press on rows/chrome pops iOS selection/callout UI mid-demo. Add `-webkit-user-select:none; -webkit-touch-callout:none` to list rows, top bar, panel headers, canvas hosts — but NOT to abstracts, In-this-corpus passages, or the review draft (reading surfaces stay selectable). Note: paper rows are `draggable` — long-press correctly starts HTML5 drag (works on iPadOS, keep it; it's how rows reach the agent panel by touch).
12. **[P1-A] "Enter full screen" account-menu row** — see § Launch paths.

## P2 — polish / after the first on-device pass

13. **Hit-target pass (44 pt rule).** Known undersized: canvas header ghost icons (A−/A+/⛶ ~26 px), paper-row star (3 px padding), pager chevrons, seg controls (~26 px tall), `.lg-eye` (26 px), `.lg-sw` switch (30×17), popover ⋯ triggers, `.nv-num` steppers. Bump with `@media (pointer:coarse)` padding or transparent hit-extension — visuals unchanged on desktop.
14. ~~Identify the anonymous `Script error.`~~ **RESOLVED / CONTAINED**: the original one was the masked twin of an unhandled module rejection from the first import-rewrite attempt — fixed by the literal-first pattern (live files log clean). A residual masked, detail-free `Script error.` still fires intermittently **in the bundle only** (opaque-origin artifact of the bundler's script inlining; even a first-in-line capture hook gets no message/file/stack; the app renders and works regardless). Contained: the demo entry hides the bundler's red toast for exactly that empty signature (`bundle-toast-guard` in the head script) — errors WITH details still toast, and the `[trace]` / `[trace-rej]` hooks print full details to the console for on-device debugging. If the bundler runtime ever changes, re-check the guard's text match.
15. **Home-screen icon — [DONE].** `apple-touch-icon.png` (180×180 terracotta K tile) generated; `<link rel="apple-touch-icon">` added to the demo entry; both live in `deploy/`. Caveat: if the bundler inlines the link href to a data: URI, iOS silently falls back to a page snapshot — after first hosted deploy, check the icon on the home screen; if it's a snapshot, strip the link from the bundle and rely on the sibling file convention (`/apple-touch-icon.png` is auto-probed by iOS at the site root).
16. **Hidden demo-state switcher — DECIDED: build it.** Demo entry only. Trigger: 5 taps on the logo within ~2.5 s (count `pointerdown`s on the top-bar logo tile; works mouse + touch). Panel: small ink card (klModal styling family) with the shell's existing prop plumbing — Build state (Before first run / Has results), Runs state (Before first run / Idle with history / Running), Explore state (Before first run / Has results), plus **Reset demo** (restore entry defaults + `show('Build')`). Nothing else — no connection/key-probe controls (locked decision). Implementation note: the shell already passes these as `renderVals` fallbacks; hold overrides in shell state so the switcher works without the Tweaks host.
17. **Tooltip audit**: delegated `mouseover` tooltips (Paper Card meta tooltip, topic-drill histogram) appear on tap and clear on next tap — fine; just confirm no information lives ONLY in a tooltip (the ⓘ "How this was computed" is a click popover ✓).

## On-device test checklist (run in landscape AND portrait, touch-only)

- Global: tab nav Build/Runs/Explore; account menu → Settings open/close on scrim; Log out exit animation → Login; theme toggle syncs across pages; no rubber-band anywhere; no tap-flash; page never zooms (type in every input!); no horizontal creep.
- Login: type in both fields (no auto-zoom), eye toggle, sign-in busy→success→Home; background net: drag a node, pinch, double-tap fit.
- Home: composer grows, example chips, wizard ①②③ (preset chips, star toggles, import via Browse, boundary segs), Create → Build fresh; project row ⋯ menu (after P1-7), collection drill + back; search filters both panels.
- Build: search, star a result, Import tab (Browse + sample path), filter panel range sliders (year + citations — both handles), sort menu, abstract expand, pipeline: tap step, drag-reorder step, presets menu, config panel focus-view editors, Run pipeline button states.
- Runs: run switcher popover, version chain popover (View/Restore → global modal), step drill, Progress/Refine seg, Re-screen shell scope seg + criteria editors, Grow presets, Merge picker, Add papers (search + import), left list: change markers toggle, row ⋯ Reject/Rescue, bulk Edit verdicts; canvas: pinch/pan/node drag/double-tap fit/−/+/⛶, Layout/Style/Filters popovers + their sliders; monitor rows expand.
- Explore: Papers/Groups seg, row tap → abstract drill + canvas select, In-this-corpus passage tap (trace), facet rail (horizontal scroll), Summarize in chat, topic search + card tap → drill (histogram tooltip, papers list) → paper above drill → Back; canvas view switcher; both canvases gestures; drag a row and a topic card onto the agent panel (long-press drag), ⊕ menu attach, multi-select bar, `/` skills menu (type / on the on-screen keyboard), send "literature review" → task card → artifact opens; artifact: Contents popover, View/Source edit, cite chips, close ✕.
- Bundle-specific: launch from home-screen icon → no Safari chrome; kill + relaunch (state resets — expected); airplane mode relaunch (path A needs network at launch; path C works).

## iPadOS facts behind the fixes

- Files/AirDrop open HTML in a preview sheet; Safari cannot browse file:// — Add to Home Screen requires a served URL; `apple-mobile-web-app-capable` (or a manifest with `display: standalone`) makes the web clip launch chrome-less.
- Inputs with font-size < 16 px auto-zoom the page on focus; `maximum-scale=1` in the viewport meta suppresses it (Safari still permits manual pinch for accessibility — fine).
- Without `touch-action`, a touch drag is a scroll: the element gets `pointercancel` and the drag dies. This is the single most common "works with mouse, broken on iPad" cause.
- Synthetic hover: first tap = `mouseenter` (+ hover styles), second tap = `click`; hover sticks until a tap elsewhere. Real hover exists on iPad with trackpad or Apple Pencil — hover polish still pays off for Magic Keyboard users.
- HTML5 drag & drop works on iPadOS via long-press-then-drag, including file drops from Split View Files. The Fullscreen API works in iPadOS Safari (`webkitRequestFullscreen` fallback).
- `dblclick` is flaky on iPad even with double-tap-zoom disabled — detect double-taps from pointer timestamps.

## Task board

- [x] Demo entry + Apple metas + focus-zoom guard + thumbnail (`KnowledgeLab iPad Demo.dc.html`)
- [x] Offline bundle builds & boots (`KnowledgeLab iPad Demo.html`); dynamic-import manifest pattern in Explore/Runs/Login
- [x] P0-1 `touch-action:none` + user-select on 4 canvases (network-viz.js, topic-viz.js constructors)
- [x] P0-2 pinch-zoom / two-finger pan / double-tap-fit (network-viz.js, topic-viz.js)
- [x] P0-3 `pointercancel` teardown on all pointer drags (viz ×2, `.dr-h` ×3 pages, Paper Card ×2 drags, Runs rsc, `.nv-sl` ×2)
- [ ] P0-4 click-through of the bundle on Mac Safari, then iPad — **the one open item: needs a human on the device**
- [x] P0-5 rotate-to-landscape notice in demo entry (DECIDED: landscape only; Explore narrow tier stays deferred)
- [x] P1-6 `.dr-h` touch-action + ≥32px hit areas
- [x] P1-7 `(hover:none)` always-visible reveals (`.rsc-drag`, row ⋯, star, `.cc-open`, …)
- [x] P1-8 skip cosmetic mouseenter paints on touch
- [x] P1-9 `setVisible` wiring on page switch + TopicMap.setVisible + Login sim stop
- [x] P1-10 keyboard avoidance (visualViewport → the focused field's own panel; still worth an on-device check)
- [x] P1-11 user-select/touch-callout on chrome (not reading surfaces)
- [x] P1-A fullscreen row in account menu
- [x] P2-13 pointer:coarse hit-target bump list
- [x] P2-14 anonymous Script error — root-caused (import rewrite) and fixed; trace hooks left in demo entry
- [x] P2-16 hidden demo-state switcher (DECIDED build: 5-tap logo → Build/Runs/Explore states + Reset demo, nothing else)
- [x] P2-15 apple-touch-icon.png + entry link + `deploy/` folder (index.html + icon) + Netlify runbook
- [x] P2-17 tooltip-only-info audit — nothing lives only in a tooltip: the topic-histogram tip restates the bar plus the drill's paper list, the paper-meta tip restates the row, and ⓘ "How this was computed" is a click popover. The hover guard (P1-8) suppresses those two `mouseenter` tips on touch by design.

## Where the fixes live (2026-08-11 implementation pass)

- **Engines** (`network-viz.js`, `topic-viz.js`): constructors set `touch-action:none` + `user-select`/`touch-callout` on the canvas; a `_ptrs` Map drives `_pinchStart/_pinchMove/_endPinch` (zoom about the finger midpoint, pan from the midpoint delta, one-finger pan re-seeded on lift); `_tapDbl` gives double-tap-fit for non-mouse pointers only (desktop keeps native `dblclick`); `_cancel` on window `pointercancel` aborts node-drag / pan / marquee.
- **Per-page touch CSS** — one identical block appended to each page's helmet `<style>` (Paper Card, Runs, Explore, Home, Login, Settings): `.dr` touch-action + `.dr-h::after` hit extension; `user-select`/`callout` off on chrome and list rows (reading surfaces untouched); `@media (hover:none)` reveals every hover-only affordance at rest (`.cf-tools`, `.rsc-drag`, `.rsc-tools`, `.rr-mn` — shifted left on the viewed run so it never covers `.rr-chk` — `.rr-go`, `.pr-act`, `.hm-dots`, `.cc-open`, `.th-hist-del`, `.pr-star` with a `[data-saved]` exception); `@media (pointer:coarse)` bumps hit targets (canvas toolbar 36px, seg/tab padding, ⋯ triggers and pager buttons 34px, `.nv-num` padding).
- **Shared touch helper** — one identical inline helmet `<script>` per page, guarded by `window.__klTouchAid`: (1) capture-phase `mouseenter`/`mouseleave` suppression when the last pointer was touch, so a tap never paints or sticks hover while trackpad/Pencil hover keeps working; (2) `visualViewport` keyboard avoidance that scrolls the focused field's own panel (never the page, never `scrollIntoView`).
- **Page-visibility** — all three shells dispatch `kl-page-shown {name}` from `show()`; Runs/Explore/Login listen and call `pgVis(on)`, and `_pgOff` also gates their internal `setVisible` calls, so only the visible page's rAF loop runs.
- **`klFsRow(wrap, close)`** in Paper Card / Runs / Explore / Home clones the Settings row into an "Enter full screen" row, mounted only when `matchMedia('(hover:none)')` and not already `navigator.standalone`.
- **Demo entry only**: the `.rot` rotate invite (root width < 1160, ResizeObserver-driven) and the `.dsw` state switcher (5 `pointerdown`s on `.tb-mark` within 2.5 s → Build/Runs/Explore state segs + Reset demo; overrides live in `this._ov` and feed `renderVals`, so it works without the Tweaks host).

## Touch-feel pass (2026-08-11 PM, after first on-device test)

User-reported fixes, all in the page/engine files (shared desktop + iPad):
- **Pipeline canvas drag** (`plDragWire`): `.pb-card,.v6d-node{touch-action:none}` so the carry-chip ghost survives on touch (scroll no longer fires `pointercancel` mid-drag); touch lift threshold 13px (finger wobble ≠ drag), ghost rides 64px above the finger, drop-line radius 160px on touch + glow.
- **First-tap select on pipeline cards**: finger wobble past the browser click slop swallowed the click (two-tap feel) — a touch pointerup with no drag now selects directly; `@media (hover:none)` neutralizes the sticky `.pb-card:hover` paint.
- **Filter-pipeline reorder** (`wireCfDrag`): rewritten iOS-style — row rides the pointer 1:1 (mouse from 3px, touch after a 240ms hold so the panel still scrolls), neighbours slide aside via transforms, single FLIP commit on release, spring-back under the crossing point, `pointercancel` springs back.
- **Runs swipe verdicts** (`rpSwipe`): Outlook-style — accepted rows swipe LEFT → Reject, rejected rows swipe RIGHT → Rescue; 1:1 follow over a tinted action layer, icon pop + haptic past threshold, slide-out + collapse commit via `rpVerdict`, spring-back otherwise; `touch-action:pan-y` keeps vertical scroll native; works with mouse too.
- **Build star** = press-scale + accent bounce + the row visibly slides/collapses into the seed set (was an instant hide). **Build sort** now staggers rows like Runs/Explore. **Presets** apply with a fade/stagger transition (`plPresetFx`).
- **Dark mode**: rotate notice is theme-aware (demo entry `.rot` via `body:has(.pc-root[data-theme=dark])`); Explore composer focus ring uses `color-mix` off `--fg` (was near-white `--fg2` border in dark).
- **Explore header overflow @1366**: `.gl-info` ⓘ hidden at compact/narrow tiers (full desktop keeps it).
- ~~Open: user report "Build projects button light in dark mode"~~ **RESOLVED (3rd pass, reproduced on Mac too)**: the light pill was the UA default `ButtonFace` (#efefef), not the tap glow. The proj-menu `close()` cleared the inline background (`=''`) that the open handler had set behind React's back — deleting the template's `transparent` with it (React never re-writes a style key it thinks is unchanged), so the raw `<button>` UA fill showed. Fix: `close()` restores `'transparent'` explicitly; same closer patched in Paper Card / Runs / Explore. Rule: a handler that sets `el.style.x` outside React must restore the template's literal value, never `''`, on a `<button>`.

## Consistency & feedback pass (2026-08-11 evening, 2nd on-device report)

- a. Panel headers unified (see design-system.md § Panel header anatomy): one container recipe on Build (both sidebars), Runs papers panel, Explore Papers AND Groups (Groups search moved into the header container - it no longer sits below the divider).
- b. Runs right header: date hides at compact/narrow (.rn-stat-date), stat gets min-width:0; "Completed" never abbreviates.
- c. Header pills act: version chip -> rnGotoVersions (Progress pane + scroll + flash); In-Explore chip -> clicks the Explore tab. Both get role=button + hover.
- d. Pressed tint on all tappable rows/cards (:active + color-mix, :not(:has(button:active))); iOS :active enabled by a touchstart listener in paper-row.js.
- e. Open-state rule [aria-expanded="true"] -> fg 10% mix in all pages; Presets button now toggles aria-expanded; sticky touch hover neutralized for th-histbtn/xp-vbtn/gl-info.
- f1. Explore glCanvasReset() on tab switch; rpDrillBack re-applies group focus (tcFocus(id, noSave)) instead of leaving paper neighbourhood lit - no stacked canvas states.
- f2. Refine launchers mutually exclusive: opening any launcher closes an active Add-papers search or verdict selection.
- f3. Up/down arrows hidden on touch: .cf-tool[data-act=up/down] (Build), .rsc-mv (Runs re-screen).

## Teammate-review pass (2026-08-11, Mingyu's comments)

- Type audit: floors set (10.5px caps-labels only, 11.5px paper-row meta/venue, 13px body) - 49 sub-floor sizes swept across pages; spec in design-system.md § Type minimums.
- cfg-add-menu flash: popIn() animates transform and was knocking out the translateX(-50%) centering for 180ms - menus now center with margin-left; rule recorded in design-system.md.
- Citations unified: author-surname + year sans pills everywhere (.rv-cite/.cc-ref shared style); no accent [n] numbering, no serif/mono chip text; artifact + thread + corpus-says all use rvCiteLbl().
- Project switch: 420ms fade-through on main (all 3 pages), same family as shell page transitions.
- Explore left panel named "Corpus" (title row + count), Papers|Groups is now a full-width Segmented seg under it - no more naked seg butting the panel corner.

## Review pass 3 (2026-08-11, A-F)

A copy-link feedback (abCopyFlash, 3 pages) · B Explore Download menu wired (corpus-scoped) · C left-side drop slots on canvas drag · D import triage order + collapsible sections (resolver ORDER central; Build sidebar collapse) · E reuse menu excludes copies (step.reused) · F "N filters" text replaces capped pips + add-menu dims duplicate-function filters with explain-on-tap. Specs in design-system.md § Feedback & structure rules.

## Review pass 4 (2026-08-11): drill choreography + drag slots

- Runs step drill (PRISMA flow) content now composes in: hero stagger -> per-stage rise -> square sweep (rdEnterFx; spec: design-system.md § Drill-in choreography). Other drills already carry the slide motion.
- Build canvas drag: trunk slots keep neighbour fallbacks so dropping between two parallel groups / above a group works even when pulling the drag out dissolves its old group (plSlots prevId/trunkTail + plMoveFn fallbacks).

After every page edit: re-run the bundle, reload it once, check console, and re-test the touched flow on the iPad. Keep `KnowledgeLab App.dc.html` untouched by demo-only changes — demo hardening lives in the demo entry; touch fixes that improve the real product (P0-1..3, P1) belong in the page/engine files themselves.
