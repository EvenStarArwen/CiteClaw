# demo-architecture.md — how the KnowledgeLab iPad Demo actually works

> 中文摘要：本文档拆解 `KnowledgeLab iPad Demo.html`（视觉真相源）的运行机制。**关键结论：该 6.5 MB 单文件 bundle 里嵌的每一个页面/模块，与 `ui_design/` 目录下的同名 sibling 文件逐字节相同**（已验证，见 §1.3），因此后续所有 agent 可以直接读 `Paper Card.dc.html` / `Runs.dc.html` 等源文件并引用行号，不必解包 bundle。第二个关键结论：`import-resolver.js` **不是**构建工具/模块解析器，而是「文献批量导入」的业务 mock（§4）。
>
> Generated 2026-08-13 by the dissection pass. Read-only source:
> `/private/tmp/claude-501/-Users-arwen-Downloads/1080a093-8b62-4b36-a8d0-32beaa09e80a/scratchpad/ui_design/`
> (referred to below as `$UI`). Nothing in that directory was modified.

---

## 1. The artifact chain

### 1.1 Three layers

| Layer | File | Role |
|---|---|---|
| Runtime | `$UI/support.js` (1911 lines) | The **dc-runtime**: a ~70 KB IIFE that turns `<x-dc>` HTML templates + a `class Component extends DCLogic` script into React components. Generated from `dc-runtime/src/*.ts`; header says *do not edit*. |
| Demo entry | `$UI/KnowledgeLab iPad Demo.dc.html` (302 lines) | The shell. Composes the 7 screens by reference, owns page routing, theme sync, the rotate notice, and the hidden demo-state switcher. |
| Compiled bundle | `$UI/KnowledgeLab iPad Demo.html` (389 lines, 6.5 MB) | **The source of visual truth.** Self-contained offline build produced by `super_inline_html` from the entry. Never edited by hand. |

### 1.2 Anatomy of the compiled bundle

The bundle is not minified markup — it is a **loader + a resource archive**:

| Bundle lines | Content |
|---|---|
| 1–29 | `<head>`: charset, `<title>KnowledgeLab</title>`, splash CSS (`#__bundler_thumbnail`, `#__bundler_loading`), `<noscript>` fallback |
| 22–28 | The K-tile splash SVG (`#__bundler_thumbnail`) — this is what iPadOS Quick Look renders when JS is disabled |
| 30–374 | The unpacker `<script>`: base64→bytes→gunzip→`Blob`, builds `window.__resources` (id → blob URL) and `window.__resourceBlobs` (url → Blob), installs an error sink, then `replaceWith`s the real document |
| **375** | One 6.7 MB JSON line: `{ "<uuid>": { mime, compressed, data(base64 gzip) }, … }` — every page, every JS module, every stylesheet |
| **379** | `<script type="__bundler/ext_resources">` — the manifest `[{id, uuid}, …]` mapping logical ids to blobs |
| 383 | `<script type="__bundler/page_order">` = `[]` (unused) |
| **387** | `<script type="__bundler/template">` — the entry HTML as a JSON string, with every `src`/`href` rewritten to the blob uuid |

Resolution model: `support.js` reads `window.__resources[url]` first (`ensureFetched`, `cdnScriptFor`, the helmet `<link>` inliner), and `window.__resourceBlobs` via `bundledBlob(url)`; only if there is no entry does it `fetch()` the network/relative URL. That is why the bundle works offline for everything except the CDN `<script>`s (marked/KaTeX/turndown/smooth-scrollbar), which stay network-loaded.

### 1.3 VERIFIED: bundle == sibling files, byte for byte

Every embedded resource was decompressed and compared with its `$UI` sibling. **All 21 pairs are byte-identical** (sha1 prefix match on identical byte lengths):

```
paper-row.js 15582  graph-data.js 13721  import-resolver.js 16056  community-data.js 5454
topic-viz.js 26406  review-draft.js 9532 network-viz.js 53529     stream-text.js 2699
run-mock.js 48940   topic-desc.js 1786   citation-context.js 64987 topic-data.js 651453
support.js 69150
Login.dc.html 42923      Home.dc.html 107598     Paper Card.dc.html 593353
Runs.dc.html 575993      Explore.dc.html 480062  Settings.dc.html 49925
System Banners.dc.html 23630
```

Consequences for every downstream agent:

1. **Read the sibling `.dc.html` files, cite their line numbers.** They *are* what the demo renders. No need to unpack the bundle.
2. Any parity mismatch is therefore a *rewrite* defect, never a "the bundle is stale" defect.
3. The one place the bundle differs from `KnowledgeLab iPad Demo.dc.html` is mechanical: `<link rel=apple-touch-icon href>` and `<script src=./support.js>` become uuid filenames, and the 15 `<meta name="ext-resource-dependency">` tags are consumed and stripped.
4. Bundle-only quirk to keep out of the rewrite: the `bundle-toast-guard` MutationObserver in the entry head (entry lines 8–11) that hides a detail-free `[bundle] Script error.` toast. It is an artifact of the bundler's script inlining, not app behaviour.
5. Minor bundler wart: `Login.dc.html` appears **twice** in the manifest (as `./Login.dc.html` and as `Login.dc.html`, two different uuids, identical bytes). Harmless; do not reproduce.

---

## 2. The dc-runtime component model (`support.js`)

### 2.1 What a "DC" is

A Design Component file is an ordinary HTML document containing exactly two things the runtime cares about:

```html
<x-dc>
  <helmet> …head material… </helmet>
  …the template…
</x-dc>
<script type="text/x-dc" data-dc-script data-props="{…JSON…}">
class Component extends DCLogic { … }
</script>
```

* `parseDcDocument` / `parseDcText` (support.js 24–74) pull out `{ template, js, props, preview }`.
* `parseDataProps` (56–74) parses the `data-props` attribute: a map `propName → { editor, default, options, tsType, section, label }`. Keys starting with `$` are stripped; `$preview` is kept separately.
* `boot()` (150–200) runs on the entry document: it registers the root DC under the name derived from the filename, replaces `<x-dc>` with `<div id="dc-root">`, and React-renders `StandaloneRoot`, whose props are `{ …propsMeta defaults, …propOverrides }`.

### 2.2 Template compilation

`compileTemplate` (467–482) → `encodeCase` (372–387) → `walk*` builders. Notable rules:

* `<helmet>` is rewritten to `<sc-helmet>` and compiled by the **helmet manager** (1420–1494): its children are hoisted into `document.head`, deduped by a key (`SCRIPT|src-or-text`, `LINK|href`, `META|…`). **`<style>` and other elements are keyed by `componentName|index` and live-patched** — this is why every page ships its own giant `<style>` block and they coexist.
  * Helmet `<link rel=stylesheet>` is inlined from `window.__resources[href]` when bundled (1452–1473) — that is mechanism (3) in the audit.
  * Helmet `<script src>` is **cloned verbatim** (1437–1446). A relative `src` would 404 in the bundle, hence the shared `__resources` shim each page uses (see §3.3).
* Control-flow tags: `<sc-if value="{{ expr }}">`, `<sc-for list="{{ … }}">`, `<dc-import name="X">`, `<x-import from="./m.js" component="Y">` (walk dispatch at 555–559).
* `{{ … }}` interpolation is **not** JavaScript. `resolve()` (205–236) is a hand-rolled mini-expression evaluator: literals, `!`, `===/!==/==/!=`, dotted/bracket paths. No function calls, no arithmetic. Anything more complex has to live in `renderVals()`.
* `style="…"` strings become React style objects (`cssToObj`, 391–400). `style-hover="…"`, `style-before="…"` etc. become generated pseudo-classes injected into a runtime stylesheet with `!important` (`createPseudoSheet`, 1567–1588). **This is why so much CSS is inline in these files** and why a rewrite must reproduce hover paint either as real CSS classes or as the same generated rules.
* `class` → `className`, `for` → `htmlFor`, `on*` → React handler names via `EVENT_MAP` (317–359).

### 2.3 `DCLogic`

`StreamableLogic` (support.js 817–841), exported to page scripts as **both** `DCLogic` and `StreamableLogic`:

```js
class StreamableLogic {
  props; state = {}; __host;
  setState(update, cb)   // proxies to the React wrapper
  forceUpdate()
  componentDidMount() {} componentDidUpdate(prev) {} componentWillUnmount() {}
  renderVals() { return {} }   // the flat object the template renders against
}
```

`evalDcLogic` (842–851) `new Function("DCLogic","StreamableLogic","React", src + ';return Component')` — so a page's script is *evaluated*, has no module scope, and can only reach globals (`window.KLPaperRow`, `window.KLRunMock`, `window.KLImport`, …) or dynamic `import()`.

The React wrapper is `StreamableComponent` (891–1108): an error boundary per DC (`getDerivedStateFromError`), a registry subscription so hot-swapping the logic class re-renders, and a `__failedLogic/__failedVer` memo so a throwing constructor is not retried every parent render.

**Practical consequence:** every page's entire behaviour is imperative DOM manipulation inside `componentDidMount`, keyed off `rootRef.current.querySelector(...)`. The template is rendered once by React; after that the pages hand-edit the DOM. React "does not know" about most of the runtime state. (The audit's *"a handler that sets `el.style.x` outside React must restore the template's literal value, never `''`, on a `<button>`"* rule comes from exactly this collision.)

### 2.4 `dc-import` — how a screen name becomes a file

`walkComponent` (661–689) turns `<dc-import name="Paper Card" page-state="{{ buildState }}">` into `h(getDC("Paper Card"), props)`. Attribute names are kebab→camel for component props (`page-state` → `pageState`); `hint-size` is consumed as a placeholder size hint.

`ensureFetched(name)` (1642–1685) resolves the name to a **sibling URL**:

```js
const url = "." + "/" + encodeURIComponent(name) + ".dc.html";   // "./Paper%20Card.dc.html"
const target = window.__resources?.[url] ?? url;                  // bundle-aware
```

so `name="Paper Card"` ⇒ `./Paper%20Card.dc.html`. This is exactly why the bundle manifest contains `./Paper%20Card.dc.html` and `./System%20Banners.dc.html` with percent-encoded spaces.

### 2.5 `x-import` — real ES modules

`walkXImport` (690–770) + `createExternalModules` (1171–…). `from="./x.js"` is fetched (bundle blob first), evaluated with `new Function("React","module","exports","require", code)`, and the named export is used as a React component. JSX/TSX goes through a CDN Babel. **The KnowledgeLab pages do not use `x-import` for their own logic** — they use plain dynamic `import()` inside `componentDidMount`, with the documented fallback shape:

```js
import('./network-viz.js').catch(function(){ return import((window.__resources||{})['netviz']) })
```

Literal first, bundle id second. The audit records that the reversed form (`import(resources || literal)`) broke live resolution and shipped twice.

---

## 3. Which sibling files are loaded, and what each provides

### 3.1 Static data + logic modules (all pure `window.*` globals or ES modules)

| File | Loaded by | Mechanism | Provides |
|---|---|---|---|
| `support.js` | demo entry `<script src>` | head script | the dc-runtime (§2) |
| `paper-row.js` | Build, Runs, Explore, Home | **helmet `<script src>` via `__resources` shim** | `window.KLPaperRow` — the ONE paper-card builder (`prCardInner`, `prPdf`, `PR_ICONS`); fires `kl-paper-row-ready`; also installs the iOS `touchstart` listener that enables `:active` |
| `import-resolver.js` | Build, Runs, Home | helmet shim | `window.KLImport` — bulk-import **mock** (see §4) |
| `run-mock.js` | Runs | helmet shim | `window.KLRunMock`; fires `kl-run-mock-ready` |
| `network-viz.js` | Runs, Explore, Login | dynamic `import()` id `netviz` | `CitationNetwork` canvas engine (pointer events, pinch, `setVisible`) |
| `topic-viz.js` | Explore | `import()` id `topicviz` | `TopicMap` engine |
| `topic-data.js` (636 KB) | Explore | `import()` id `topicdata` | the 500-paper MOEA/D corpus, 67 topics + 86 noise |
| `topic-desc.js` | Explore | `import()` id `topicdesc` | topic descriptions |
| `graph-data.js` | Runs, Login | `import()` id `graphdata` | citation graph for the Runs canvas / Login background |
| `community-data.js` | Explore | `import()` id `commdata` | Leiden communities (`C##`) |
| `citation-context.js` (63 KB) | Explore | `import()` id `citctx` | "In this corpus" citing passages — **for exactly one paper** (Boiko et al.); every other paper renders the empty state |
| `stream-text.js` | Explore | `import()` id `streamtext` | `streamHtml(host, html, {cps, caret, onDone, onTick})` — rAF word-by-word reveal, default 220 chars/s |
| `review-draft.js` | Explore | `import()` id `reviewdraft` | the literature-review draft markdown |

All 15 ids are declared as `<meta name="ext-resource-dependency" content="…" data-resource-id="…">` in the demo entry (lines 20–35). **Any new module must be added there or it silently drops out of the bundle.**

### 3.2 CDN scripts that are NOT bundled

`marked`, `KaTeX` js, `turndown`, `smooth-scrollbar` load from the network at runtime. KaTeX **css** and the four Google-Fonts stylesheets *are* bundled (manifest ids `fontsA`…`fontsD`, plus the KaTeX css url). A rewrite that must work offline has to vendor the four JS libs.

### 3.3 The shared resource shim (duplicated per page)

Each page's helmet carries one identical inline `<script>` that mounts the local helmet scripts through `__resources`:

```
Build   / Explore / Home : 230 normalized chars, sha 17f30278  (identical)
Runs                     : 258 normalized chars, sha 6d4fc3d5  (adds run-mock.js)
```

Because the helmet manager dedupes by script text, identical shims mount once.

### 3.4 The shared touch helper

One inline `<script>` guarded by `window.__klTouchAid`, **byte-identical (1323 chars, sha ad73234e) in Build, Runs, Explore, Home, Login, Settings** (absent from System Banners). It does two things: (1) capture-phase suppression of `mouseenter`/`mouseleave` when the last pointer was touch, (2) `visualViewport`-driven keyboard avoidance that scrolls the focused field's own panel (never the page, never `scrollIntoView`).

---

## 4. SETTLED: what `import-resolver.js` actually is

**It is a product-domain mock, not build tooling.** Despite the name, it has nothing to do with module resolution, the design tool, or the bundler.

Evidence (file header + full read, 174 lines):

```js
// import-resolver.js — KnowledgeLab's shared bulk-import mock (see design-system.md § Import papers).
// One flow, many parsers: extract references → resolve on Semantic Scholar → match-review → add.
window.KLImport = { parse, sample, groups, rowHtml, groupHtml, candPopHtml, fileRowHtml, extOf }
```

"Resolver" here means **resolving imported bibliographic references against Semantic Scholar**. It is consumed by three UI surfaces so they cannot diverge:

* Home wizard step ② (Search | Import seg)
* Runs → Refine → Add papers / Add seed papers
* **Build → seed sidebar → Import tab** (see build-page-spec.md §4)

What it exposes:

* `parse(files)` → `{ files:[{name, ext, n, err?}], entries:[…] }`. Deterministic: a seeded LCG keyed on `hash(filename + ':' + index)`. Per-extension entry counts `EXT_N = { bib:[24,12], ris:[16,10], csv:[10,6], txt:[5,4], zip:[5,4], pdf:[1,0] }`; unknown extensions produce `err: 'Unsupported format: use .bib, .ris, .csv, a DOI list, or PDFs'`.
* **`refs.bib` is a hard-coded fixture**: exactly 34 entries, with indices 3/14/25 = `none`, 5/22 = `multi`, 9 = `dupe`. `sample()` = `parse(['refs.bib'])`. This is the "Or try a sample refs.bib" path and the only fully deterministic import case.
* Entry states: `ok` (matched, checkbox), `multi` (needs a decision, "N matches" pill), `none` (couldn't match, error line, **never added**), `dupe` (already in corpus, rendered at 55 % opacity).
* `groups(entries, dupeLabel)` imposes the **central triage order**: `Needs a decision → Couldn't match → Matched → Already in the corpus` (`ORDER`, lines 89–94). Empty groups are dropped.
* `rowHtml/groupHtml/candPopHtml/fileRowHtml` — the shared markup builders. `groupHtml(..., {flat:true})` drops the card frame for hosts that already draw one (Home's wizard box).

Backend-contract note: none of this is a backend contract. The real wiring must supply its own parse/resolve service; only the **triage order, the four states, and the "unmatched entries are reported, never added" rule** are product decisions to preserve.

---

## 5. Screens and routing

### 5.1 Composition (demo entry lines 72–137)

All seven screens are mounted **simultaneously** inside one relatively-positioned root (`width: {{vpW}}; height: {{vpH}}; min-width:1024px`):

| Order | Element | `data-pg` | z | Notes |
|---|---|---|---|---|
| 1 | `<dc-import name="Login">` | `Login` | — | `network-colors="Ink & white"` (fixed) |
| 2 | `<dc-import name="Home">` | `Home` | — | homeState / libraryLayout / projectRows / collectionStyle |
| 3 | `<dc-import name="Paper Card">` | `Build` | — | `layout="List"` **pinned**, `card-emphasis="Muted"`, `logo-style="Terracotta tile"`, `pipeline-style={{pipelineStyle}}` |
| 4 | `<dc-import name="Runs">` | `Runs` | — | runState / netPalette / runPhase / apiRetries / seedCardStyle |
| 5 | `<dc-import name="Explore">` | `Explore` | — | exploreState / netPalette |
| 6 | `<dc-import name="Settings">` | — | 60 | always mounted, `pointer-events:none` until it opens itself |
| 7 | `<dc-import name="System Banners">` | — | 80 | always mounted, `pointer-events:none` |
| 8 | `.rot` rotate notice | — | 140 | shown when root width < 1160 |
| 9 | `.dsw` demo-state switcher | — | 200 | 5-tap on `.tb-mark` |

`.pg` visibility is CSS only (`opacity/visibility/pointer-events`, 220 ms crossfade) — **pages are never unmounted**, so their timers, rAF loops and DOM state survive navigation. This is the single biggest fact for parity capture (see state-inventory.md).

### 5.2 Routing (`show(name, instant)`, entry 250–274)

There is **no router, no URL, no history**. `show()`:

1. flips `data-on` on `[data-pg]`;
2. dispatches `document` event `kl-page-shown {name}` (pages use this to gate their rAF loops);
3. after 140 ms walks every `.pc-root`, digs out its React fiber to find the `StreamableComponent`'s `logic`, and calls `logic._ssbInit()` — re-initialising smooth-scrollbar on newly visible scrollers;
4. paints `data-active` on the three `.tb-tab` buttons by **text content match** (`'Build' | 'Runs' | 'Explore'`).

Navigation triggers, all bubbling DOM CustomEvents caught on the shell root:

| Event | Source | Effect |
|---|---|---|
| click on `.tb-tab` | any page's top bar | `show(tabText)` — matched by `textContent`, not by id |
| `kl-login` | Login sign-in sequence | `show('Home')` |
| `kl-home` | pages' project-switcher *New project* / *All projects* | `show('Home')` |
| `kl-open-project {fresh}` | Home project rows / wizard Create | `_buildOv = fresh ? 'Before first run' : null; show('Build')` |
| `kl-logout` | account menu | `logout()` — 170 ms scale-down + fade of the current page, then `show('Login')`, cleanup at +320 ms |
| `kl-open-filters` | (legacy) | `show('Build')` |
| `kl-banner-inset {h}` (document) | System Banners | sets `--bn-inset` on the shell root; every page reserves that height under its top bar |

### 5.3 Theme sync

`syncTheme()` (entry 212–233) attaches a `MutationObserver` on `data-theme` of every `.pc-root` and mirrors the value to all the others, guarded by `this._themeLock`. Re-attached at 400/1200/2500 ms to catch late-mounting pages. Theme therefore has **no single source of truth** — whichever page's toggle you press becomes the master for that tick.

### 5.4 The hidden demo-state switcher

5 `pointerdown`s on `.tb-mark` within 2500 ms (entry 163–169) opens `.dsw`. It writes `this._ov = { build|run|explore: value }`, which `renderVals()` reads with `??` precedence:

```
build:   _ov.build   ?? _buildOv ?? props.buildState   ?? 'Has results'
run:     _ov.run     ??              props.runState    ?? 'Running'
explore: _ov.explore ??              props.exploreState?? 'Has results'
```

"Reset demo" clears `_ov` and `_buildOv` and returns to Build. Scope is deliberately limited — no connection/key-probe controls.

### 5.5 Locked variant defaults (verified against the entry's `data-props`)

`pipelineStyle='Flow chart (6d)'`, `networkPalette='Neutral ink'`, `collectionStyle='Cover rows'` — all three are the entry's declared defaults, plus hard-coded shell values `scheme='Warm paper'`, `bounce=false`, `runPhase='None'`, `logo-style='Terracotta tile'`, Build `layout='List'`, Settings `settings-style='Centered card'`. Page-level defaults: `buildState='Has results'`, `runState='Running'`, `exploreState='Has results'`, `homeState='Has projects'`, `connection='Online'`, `keyProbe='All valid'`, `apiRetries=false`, `viewport='Fill the window'`, `page='Build'`.

### 5.6 Responsive tiers (a product requirement — must survive the rewrite)

Every page self-measures with a `ResizeObserver` on its own `.pc-root` and sets `data-vp`:

```
clientWidth <= 1240 → "narrow"
clientWidth <= 1440 → "compact"
otherwise           → "full"
```

(Build: `Paper Card.dc.html:2385`; the same rule is repeated per page.) A `data-vp-ready="1"` flag is set two rAFs after the first measurement so drawer transitions do not fire on load. The demo entry adds a **landscape-only guard**: root width < 1160 ⇒ `data-rot="1"` ⇒ the `.rot` overlay (entry 178–182). iPad Pro 12.9 landscape (1366) lands in `compact`; portrait (1024) would be `narrow`, which the demo replaces with the rotate invite.

---

## 6. The run-replay simulation

### 6.1 The model (`run-mock.js`, 765 lines)

`window.KLRunMock = { PIPELINE, FILTERS, RUN37, RECO, RUN_LIBRARY, NEXT_RUN_NO, REPLAY, makeReplay, buildTimeline }`.

* `RUN37` is a real recorded run: per-step `found` counts and a six-slot `cuts` array `[year, citation, keyword, LLM-title, LLM-abstract, duplicate]`. Slots 0–4 sum to the step's `rej`; the duplicate pass is counted in the filter table but **not** in the run's rejected total.
* `TOKEN_MIX = { title: .1855, abs: .6023, db: 0, out: .2122 }` of 194 921 tokens.
* `RUN_LIBRARY` = the run list; `NEXT_RUN_NO = 38` — a user-started run becomes Run 38 and Run 37 is left untouched.
* `RECO` = topology-based Add-papers suggestions (two 5-paper groups: *Cites this corpus*, *Shared bibliography*), each with `rc/nc/hits/share/best` evidence fields and a `cites[]` id list.

### 6.2 The clock

```js
REPLAY = { totalMs: 60000, waveBase: 400, stepBase: 120, tickMs: 320,
           stageMix: [['fetch',.30],['meta',.10],['abs',.16],['basic',.10],['llmT',.20],['llmA',.14]],
           seedMix:  [['fetch',.50],['meta',.25],['abs',.25]] }
```

`buildTimeline(steps, cfg)` (132–184):

1. `waveSplit` groups consecutive steps that consume the same input into **waves** (the engine's parallel blocks).
2. Wave duration ∝ `waveBase + Σ found` — so FWD-07 (6 651 fetched) dominates, as it did in reality.
3. **Inside a wave, steps run strictly back-to-back, never overlapping** — execution is serial; the wave only *states* the parallel structure. Wave time is split between its steps ∝ `stepBase + found`.
4. Each step's duration is cut into stages by `stageMix` (or `seedMix` for the seed step).

`makeReplay(source, cfg).frameAt(t)` is a **pure function of elapsed ms** — it returns per-step phase (`queued`/`active`/`done`), stage progress, cumulative accepted/rejected, per-filter rejection counts + running cumulative, LLM call count and the four token buckets, `elapsed` (real seconds, `Math.round(t/1000)`) and an internal `engineSec` scaled to `source.elapsed`. Deliberate design note in the file: *"the on-screen clock ticks in REAL seconds: a demo compresses the WORK, never the wall clock the user watches."*

### 6.3 The driver (Runs page)

* `rplModel()` (`Runs.dc.html:2748`) lazily builds and caches `makeReplay(KLRunMock.RUN37)`.
* `startSeq(btn)` (2005–2035): guards against re-entry, `btnBusy('Starting…')`, resets `_elS`, decides `fresh` (a user-started run always replays from the top), `rnNewLiveRun()`, records `_rplT0 = Date.now()`, resets the paint cursors (`_rplAccPainted`, `_rplRejPainted`, `_arrN`, `_nvShown`, `_monSt`), rebuilds the canvas over the **whole** corpus so nodes can stream in, then `setInterval(rplTick, 320)`.
* `rplTick()` → `frameAt(Date.now() - _rplT0)` → `rplPaint(f)`; `f.finished` → `rplFinish(f)` (lands the run as completed, fires the success banner with a *Show in Explore* action).
* `rplStop()` tears the interval down and re-seeds the canvas from the accepted list — a stopped run only produced its prefix.
* A separate 1 s `_elT` ticks the `.rn-elapsed` label when `pageState='Running'` but no replay is active; **its seed is `this._elS = 758`** (`Runs.dc.html:2790`), i.e. the canned "Running" state starts at 12:38 elapsed.
* `phaseSet('starting'|'stopping')` drives the hero phase row; `stopping` runs its own 1 s `_phT` ticker over three stage captions (`Finishing the current batch… / Writing the checkpoint… / Saving partial results…`), advancing every 3 s.
* `_nvT` is a third interval that streams new nodes onto the citation network.

All of `_rplT`, `_nvT`, `_elT`, `_phT` are cleared in the page's teardown (`Runs.dc.html:3945–3947`).

### 6.4 The other "simulations"

| Flow | Where | Shape |
|---|---|---|
| Build search | `Paper Card.dc.html:2876–2895` | Enter or a 600 ms input debounce ⇒ `data-searching="1"` + skeletons ⇒ 850 ms ⇒ results. A monotonic `_sqTok` cancels stale responses. Filters and sort are **local and instant**, only the query is "remote". |
| Build import | `sbImpGo/sbImpRev/sbImpAdd` (4665–4790) | dropzone → per-file parse rows → `KLImport.parse` → match-review sections → `Add N to seed set` |
| Explore agent thread | `Explore.dc.html:1692`, 4152 | a send matching `/\blit(?:erature)?\s+review\b/i` (or the `/literature-review` skill) plays the live flow: status lines settle → canvas opens the artifact → `streamHtml` streams the draft at 220 cps → thread lands summary + card + follow-up. Any other send gets a canned cited answer. |
| Login sign-in | `Login.dc.html:455–485` | validate → `data-state="Signing in"` → **1300 ms** → `Signed in` → **750 ms** → `data-exit="1"` + camera dive (620 ms tween) → **430 ms** → dispatch `kl-login` → **800 ms** → reset. Total ≈ 3.3 s. SSO buttons are pure theatre (1400 ms spinner, no event). |
| Settings key probe | `Settings.dc.html` | API-key chips verify on open and on Save; `keyProbe` prop picks the outcome (`All valid` / `Invalid Gemini key` / `Rate-limited S2` / `OpenAI unreachable`). |
| System Banners | `System Banners.dc.html` | document events `kl-banner` / `kl-banner-hide`; a persistent status strip publishes its height back as `kl-banner-inset {h}` |

---

## 7. The app-level event bus (complete inventory)

| Event | Dispatched on | Emitters → listeners |
|---|---|---|
| `kl-page-shown {name}` | `document` | shell → Runs, Explore, Login (gate rAF loops via `pgVis`) |
| `kl-open-settings` | bubbling | Build/Runs/Explore/Home account menu → Settings |
| `kl-logout` | bubbling | all four account menus → shell (+ Login re-arms its background) |
| `kl-login` | bubbling | Login → shell |
| `kl-home` | bubbling | Build/Runs/Explore/Home project switcher → shell |
| `kl-open-project {fresh}` | bubbling | Home → shell |
| `kl-project {name}` | `document` | any page's project switcher → all pages (`applyProject`) |
| `kl-paper-row-ready` | `document` | paper-row.js → Build/Runs/Explore/Home (deferred wiring) |
| `kl-run-mock-ready` | `document` | run-mock.js → Runs |
| `kl-run-activity {state,label}` | `document` | Runs → Build/Explore top-bar Runs dot (`running` = pulsing, `finished` = static) |
| `kl-filter-groups` | `document` | Build ↔ Runs, mirrors `localStorage['kl-filter-groups']` |
| `kl-goto-runs` | bubbling | Explore corpus chip → Runs (deep-link to a version) |
| `kl-groups` | `document` | Runs internal |
| `kl-banner` / `kl-banner-hide` | `document` | Runs, Home → System Banners |
| `kl-banner-inset {h}` | `document` | System Banners → shell (`--bn-inset`) |
| `kl-modal` | `document` | Runs → the one global `klModal` |
| `kl-open-filters` | bubbling | legacy → shell |

There is exactly **one** `localStorage` key in the whole app: `kl-filter-groups` (read/written in `Runs.dc.html:4470/4476`, read in `Paper Card.dc.html:4348` and `Runs.dc.html:5730`), access try/catch-guarded.
