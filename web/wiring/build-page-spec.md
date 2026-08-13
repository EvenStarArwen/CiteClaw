# build-page-spec.md — exhaustive spec of the Build screen

> 中文摘要：Build 页 = `Paper Card.dc.html`（5877 行）。本文按区域穷举每一个控件、抽屉、对话框、hover 态、空态，并给出每个 UI 状态的**到达路径**（这份路径清单就是 parity 交互脚本）。所有行号指向 `$UI/Paper Card.dc.html`；因 bundle 与 sibling 文件逐字节相同（见 demo-architecture.md §1.3），这些行号对「视觉真相源」同样成立。
>
> `$UI` = `/private/tmp/claude-501/-Users-arwen-Downloads/1080a093-8b62-4b36-a8d0-32beaa09e80a/scratchpad/ui_design/`

## 0. File map

| Range | Content |
|---|---|
| 1–8 | doctype / head |
| 9–1327 | `<helmet>`: Google-Fonts + KaTeX links, the `__resources` shim script, the `__klTouchAid` script, and **one 1287-line `<style>` block (15–1302)** carrying every token and every component rule |
| 1329–2360 | the template (`.pc-root`) |
| 1331–1506 | **top bar** (`<header>`) |
| 1508 | banner inset spacer (`height: var(--bn-inset, 0px)`) |
| 1510–2359 | `<main>` — 3 columns |
| 1511–2075 | `<aside class="pc-drawer">` holding **two** `.sidebar` copies |
| 1513–1792 | `.sidebar[data-style="list"]` ← **the one the demo renders** |
| 1793–2075 | `.sidebar[data-style="cards"]` ← never rendered (`layout="List"` is pinned) |
| 2076–2268 | `<section class="panel-canvas">` — Pipeline |
| 2269–2357 | `<section class="panel-config">` — step config |
| 2362 | `data-props` |
| 2363–5877 | `class Component extends DCLogic` |

Props the shell passes (demo entry line 80): `card-emphasis="Muted"`, `color-scheme="Warm paper"`, `logo-style="Terracotta tile"`, **`layout="List"`**, `page-state={{buildState}}`, `pipeline-style="Flow chart (6d)"`, `overscroll-bounce=false`, `hint-size="100%,100%"`.

Root attributes written by `componentDidMount` (2401–2431): `data-theme`, `data-handle="Diamond"`, `data-scheme="Recessed canvas"`, `data-color`, `data-logo`, `data-tabs="Quiet type"`, `data-seg="Recessed shade"`, `data-addbtn="Circle"`, `data-chip="Quiet mono"`, `data-state` (`pre`|`ready`), `data-first` (`0`|`1`), `data-vp` (`full`|`compact`|`narrow`), `data-vp-ready`, `data-srcpage` (`set`|`search`), `data-pstyle`.

---

## 1. Region A — top bar (1331–1506, height 58 px)

| # | Element | Selector | Lines | Behaviour |
|---|---|---|---|---|
| A1 | Logo tile | `.tb-mark` | 1335–1347 | Terracotta K tile. **5 `pointerdown`s within 2.5 s opens the demo-state switcher** (handled by the shell, not this page). |
| A2 | Wordmark | — | 1348 | `OmniKnowledge`, Newsreader 19px/500 |
| A3 | Divider | — | 1350 | 1×22 px rule |
| A4 | Project switcher button | `.tb-project` | 1352–1358 | Folder tile + `.tb-proj-name` (`AI agents for scientific discovery`) + chevron. `aria-haspopup`. Opens A5. Wired by `setupProjMenu` (4947–4971). |
| A5 | Project menu | `.tb-proj-menu` | 1359–1435 | 336 px popover. Rows: **New project** (`.tb-proj-new` → `kl-home`), header `Projects`, six `.tb-proj-item` (AI agents ✓ / Evolutionary multi-objective optimization / Hyper-parameter optimization / Combinatorial optimization / Search-based software engineering / Meta science), divider, **All projects** (`.tb-proj-all` → `kl-home`). Picking an item dispatches `kl-project {name}` on `document` and plays a 420 ms fade-through on `<main>`. |
| A6 | Tabs | `.tb-nav` / `.tb-tab` ×3 | 1436–1440 | `Build` (`data-active="1"` in this file) · `Runs` (carries `.tb-runs-dot`, hidden by default) · `Explore`. **The shell matches these by `textContent`.** |
| A7 | Runs activity dot | `.tb-runs-dot` | 1438 | Painted by `actPaint` (2367–2384) off `document` event `kl-run-activity`: `running` → visible + `sbPulse 1.4s` + `title="… is running"`; `finished` → visible, static, `title="… finished, not viewed yet"`; hidden while the Runs tab is itself active. |
| A8 | Download button | `.tb-download` in `.tb-dl-wrap` | 1443–1447 | `aria-haspopup`. Wired by `setupDownload` (5049–5082). |
| A9 | Download menu | `.tb-dl-menu` | 1448–1474 | 268 px. Head row `Download · Run 6 · yesterday` (`.tb-dl-head/.tb-dl-run/.tb-dl-date`); items **Accepted papers** (`data-dl` → `CSV · 214 papers`) and **Full bundle** (`CSVs · BibTeX · graph · stats`); **empty state `.tb-dl-none`**: "No runs yet / Results download per run. Start one with Run pipeline." |
| A10 | Run pipeline | `.tb-run` | 1475–1478 | Primary filled button, ▶ glyph, `min-width:138px`. |
| A11 | Theme toggle | `.pc-theme-toggle` | 1479–1483 | Moon/sun swap + **a `<span class="tt-label" style="display:none">`** written by `applyTheme` (3416) but never shown. |
| A12 | Account button | `.tb-user` in `.tb-user-wrap` | 1485–1487 | |
| A13 | Account menu | `.tb-user-menu` | 1488–1505 | 224 px. Identity row `Ada Lovelace`; **Settings** (`.tb-user-settings` → `kl-open-settings`); **Log out** (`.tb-user-logout` → `kl-logout`); on touch devices `klFsRow()` (5022–5044) clones the Settings row into an **Enter full screen** row (`.tb-user-fs`), only when `matchMedia('(hover:none)')` and not `navigator.standalone`. |

Hover states: `.tb-project`, `.tb-settings`, `.pc-theme-toggle`, `.sb-filter-btn`, `.sb-sort-btn` share a transition rule (431–433); `@media (hover:none)` neutralises the sticky paint (1277). Open popovers set `aria-expanded="true"` → a 10 % fg tint.

---

## 2. Region B — search sidebar (`.pc-drawer` → `.sidebar[data-style="list"]`, 1513–1792)

Width `var(--sbw)`: **360 px** at `full`, **320 px** at `compact`/`narrow` (helmet CSS 18). At `narrow` the whole `<aside>` is **moved into `.cfg-shost`** by `_vpPlace()` (3921–3928) and loses its border/radius/shadow/background (CSS 20–21) and its title row (CSS 22).

### 2.1 Panel header (1553–1585)

| # | Element | Selector | Lines |
|---|---|---|---|
| B1 | Title row | `.sb-head-row` | 1554–1557 | serif `Papers` + `.sb-count` (`42 papers`; becomes `No search yet` in first-run, `Searching…` while querying, `–` on error). Hidden at `narrow`. |
| B2 | Mode seg | `.fc-seg.sb-imode` | 1559 | `Search` / `Import`, `data-sm` on the sidebar. |
| B3 | Search field | `.sb-search` inside `.sb-qwrap` | 1560–1563 | placeholder `Search Semantic Scholar…`, magnifier glyph. |
| B4 | Filter button | `.sb-filter-btn` + `.sb-filter-label` | 1566–1571 | Label becomes `N filters` when any are active. Opens the filter drawer (C). |
| B5 | Sort button | `.sb-sort-btn` + `.sb-sort-label` | 1572–1577 | Default `Most cited`. |
| B6 | Sort menu | `.sb-sort-menu` / `.sb-opt` ×3 | 1578–1582 | `Most cited` ✓ / `Newest` / `Title A–Z`. Applying staggers the rows. |

### 2.2 List body (`.sb-list`, 1587–1791)

Mutually exclusive children, gated purely by CSS (helmet 352–367):

| # | State element | Lines | Shown when |
|---|---|---|---|
| B7 | `.sb-first` first-run invite | 1588–1595 | `.sidebar[data-first="1"]:not([data-searched="1"])`. Magnifier tile, *Find your seed papers*, copy, three chips `language models` / `reasoning` / `premise selection` (`.sb-first .fc-chips button` → fills the field and searches). |
| B8 | `.sb-imp` import flow | 1596–1611 | `.sidebar[data-sm="import"]` |
| B9 | `.sb-searching` skeletons | 1612–1632 | `.sidebar[data-searching="1"]` — a pulsing dot + `Searching Semantic Scholar…` + three skeleton cards (delays 0/.15/.3 s) |
| B10 | `.sb-empty` | 1633–1640 | `sidebar.dataset.state === 'empty'` — *No papers found* + `.sb-clear` **Clear search & filters** |
| B11 | `.sb-error` | 1641–1648 | `sidebar.dataset.state === 'error'` — *Couldn't load papers* + `.sb-retry` **Retry**. ⚠ **UNREACHABLE in the shipped demo** (see §7). |
| B12 | `.pr` paper rows | 1650–1789 (14 seeded) + `expandMock()` | the normal path |
| B13 | `.sb-scrim` ×2 | created in `wireSidebar` (2704–2721) | top/bottom gradient scrims that fade in past 4 px of scroll (deliberately **not** `mask-image` — masks blur text on fractional DPR) |

Row anatomy (`.pr`, e.g. 1650–1659): absolute `.rail` accent bar (opacity 0→1 on selection, width 4 px), venue dot + venue caps (`--muted2`, fixed faint — no tweak), year + `<b>` citations, serif `.pr-title` (2-line clamp), `.pr-authors`. Data attributes `data-venue/-year/-cites/-url/-abstract` drive local filtering.

The **star** is not in the markup: `wireRow` (2439–2524) appends `.pr-star` only if `prCardInner` (paper-row.js) has not already built one, forces `padding-right:46px`, and its visibility is CSS state (`.pr:hover .pr-star`, `[data-saved]`). **Any paper-card change must be verified by hovering** — at rest the star is invisible by design.

### 2.3 Row interactions (`wireRow`, 2439–2524)

| Interaction | Result |
|---|---|
| hover row | `data-hover="1"` → `background: var(--row-hover)`; star becomes visible |
| `mousedown` / `mouseup` | `filter: brightness(.97)` / none |
| click row | de-selects every `.pr` in the sidebar, sets `data-selected="1"`, rail to opacity 1; calls `row.__onSelect` (opens the abstract drill, §5) |
| `Escape` (document) | clears selection in every sidebar that is not `data-select="persist"`, clears `.cf-filter[data-sel]`, `closeConfig()` (2432–2437) |
| star `pointerdown` | svg scales to .8 |
| star click **in a sidebar** | one-way: `data-saved="1"`, accent bounce (420 ms spring), then the row **animates out** (440 ms, delay 120 ms: fade + `translateX(30px)` + height/padding collapse) and `seedAdd(row)` runs. Re-clicking a saved row is a no-op. |
| star click **outside a sidebar** | plain toggle + 380 ms bounce |

### 2.4 Search state machine (`searchNow`, 2876–2895)

```
idle ──input(600 ms debounce) or Enter──▶ [q === lastQ] ─▶ no-op
                                       └─ [q empty] ─▶ clear data-searching, apply()
                                       └─ else ─▶ data-searching="1", count="Searching…"
                                                   └─850 ms─▶ clear, (first-run: data-searched="1"), apply()
```

A monotonic `sidebar._sqTok` discards stale responses. **Filters and sort are local and instant; only the query is "remote".**

Query grammar (`exprMatch`, 2926–2932): OR-separated groups, `AND` ignored as a joiner, `"quoted phrase"` exact, `*` wildcard → regex, everything case-insensitive substring. Searching matches title + venue + author; the filter fields each match their own single field.

---

## 3. Region C — filter drawer (`.sb-filter-panel`, 1514–1551)

`position:absolute; inset:0; z-index:30` over the sidebar; `display:none` until opened by B4.

| # | Control | Selector | Lines | Detail |
|---|---|---|---|---|
| C1 | Close | `.sb-filter-close` | 1517 | ✕ |
| C2 | Year range | `.dr[data-key=year]` `data-min=2018 data-max=2024 data-step=1` | 1522–1527 | two `.dr-h` handles, `.dr-fill`, live label `.fl-year-val` (`2018–2024`). `touch-action:none` inline; `::after` extends the hit area to ≥32 px. |
| C3 | Citations range | `.dr[data-key=cites]` `data-min=0 data-max=20000 data-step=100` | 1532–1537 | label `.fl-cites-val` (`0–20k+`) |
| C4 | Venue | `.ff-input[data-key=venue]` | 1541 | placeholder `Annals OR arXiv` |
| C5 | Title contains | `.ff-input[data-key=title]` | 1542 | placeholder `"prime gaps" AND sieve` |
| C6 | Abstract contains | `.ff-input[data-key=abstract]` | 1543 | placeholder `(automorph* OR modular) AND L-function` |
| C7 | Author | `.ff-input[data-key=author]` | 1544 | placeholder `Maynard OR Tao` |
| C8 | Reset all | `.sb-filter-reset` | 1548 | clears all four inputs + both ranges |
| C9 | Done | `.sb-filter-apply` + `.sb-apply-count` | 1549 | `Done  N shown` — count updates live |

`activeCount()` (2757+) counts a non-empty text field or a moved range as 1; the count feeds B4's label.

---

## 4. Region D — import flow (`.sb-imp`, 1596–1611)

Reached by B2 → **Import**. Four sub-states inside one container:

| # | Sub-state | Selector | Lines | Content |
|---|---|---|---|---|
| D1 | Dropzone | `.sbi-drop` | 1597–1604 | dashed 1.5 px, upload glyph, *Drop your reference files*, copy naming `.bib/.ris/CSV/DOI list/PDFs/folders/.zip`, **Browse files** (`.sbi-browse` → hidden `.sbi-file[multiple]`), **Or try a sample refs.bib** (`.sbi-sample`) |
| D2 | Parse list | `.sbi-parse` | 1605 | one `KLImport.fileRowHtml` row per file: EXT badge, name, `Parsing…` spinner (or an error chip for unsupported formats) |
| D3 | Match review | `.sbi-rev` | 1606 | `KLImport.groups()` sections in the fixed order **Needs a decision → Couldn't match → Matched → Already in the corpus**, each collapsible with a count pill (error-tinted on *Couldn't match*) |
| D4 | Footer | `.sbi-foot` | 1607–1610 | `.sbi-foot-err` error line + `.sbi-addbtn` **Add N to seed set** |

Wiring: `sbImpWire` 4631, `sbImpState` 4658, `sbImpGo` 4665, `sbImpRev` 4687, `sbImpFoot` 4711, `sbImpCand` 4728 (the "N matches" popover, `KLImport.candPopHtml`), `sbImpAdd` 4760 → `seedImport(entries)` 4776 (title-deduped, marks the matching library rows as starred seeds).

Deterministic fixture: **`refs.bib` → 34 entries**, indices 3/14/25 unmatched, 5/22 ambiguous, 9 duplicate. Unmatched entries are reported and **never added**.

CSS gating (helmet 360–363): in import mode every non-`.sb-imp` child of `.sb-list` and the query row, pager, first-run invite, empty and error states are `display:none !important`.

---

## 5. Region E — abstract drill (built in JS, `setupAbstracts` 3082 / `wireAbstract` 3100–3285)

Not present in the template. On a row click a full-panel drill is generated (markup strings at 3200–3220):

`.ab-back` ← Back · `.ab-ic[data-ic=ven]` venue chip · serif `.ab-title` (22 px) · meta line 11.5 px (`.ab-year`, `.ab-cites`) · `.ab-authors` · action row **`.ab-link2` open** / **`.ab-copy` copy** (with `abCopyFlash` feedback, 3806–3817) / **`.ab-save`** · caps label `Abstract` (10.5 px uppercase) · `.ab-abstract` (13.5 px / 1.65, `--abstract-fg`, `text-wrap:pretty`). A collapsed variant (3251+) uses a 15 px serif title.

⚠ This is a **third, JS-built copy** of a component that exists as markup on Runs and Explore — see component-duplication.md §4.

---

## 6. Region F — pipeline canvas (`.panel-canvas`, 2076–2268)

### 6.1 Header (2077–2113, height 56 px)

| # | Element | Selector | Lines |
|---|---|---|---|
| F1 | Title | — | 2080 | serif `Pipeline` |
| F2 | Step count | `.plh-count` | 2081 | `7 steps` — hidden in first-run (CSS 355) |
| F3 | Presets button | `.plh-preset` | 2084–2088 | hidden in first-run |
| F4 | Presets menu | `.plh-preset-menu` | 2089–2110 | `Scout` (4 steps, *One hop from the seeds, screened once. Quick and cheap.*) · `Survey` (6 steps, *Adds a second reference walk and screen. A balanced default.*) · `Dragnet` (10 steps, *Sweeps three waves deep and reranks between hops. Leaves little behind.*) + footnote *Replaces the current pipeline. Undo brings it back.* Picking one opens a **confirm popover** (`cfgConfirm`) before applying; applying runs `plPresetFx()` (fade/stagger). |
| F5 | Undo | `[aria-label="Undo last pipeline change"]` | 2112 | hidden in first-run |

### 6.2 First-run overlay (`.pl-first`, 2114–2140)

`position:absolute; top:57px; inset-x:0; bottom:0; z-index:20`, `display:none` by default, `display:flex` when `.pc-root[data-first="1"]` (CSS 352–353). Contents: glyph tile, serif *Shape your pipeline*, copy, three `.pf-card[data-first-preset]` (Scout / Survey / Dragnet, 172 px each, **note the Survey and Dragnet blurbs are shortened here vs F4**), an `or` rule, `.pl-first-scratch` **Start from the seed set** and the line *Just the Seed Set on the canvas. Add each step yourself.*

Choosing anything sets `_plFirstDone = true` and `data-first="0"` permanently for the session (5838–5849). "Start from the seed set" additionally fires the toast *Starting from the seed set — add steps with the plus button* with an Undo.

### 6.3 Flow (`.pl-scroll` → `.pl-flow`, 2141–2259)

The markup holds a static 7-step example (`.pb-card[data-kind=database][data-code=SED-01]` at 2144, then `.pl-wire` SVG + `.pl-group` ×3 at 2154/2182/2221 + a closing wire). **It is replaced on mount**: `plInit()` (5199–5209) builds the model

```
pipe = [ SED-01, parallel([FWD],[BWD]), parallel([FWD],[BWD]), parallel([BWD],[FWD]) ]
```

(7 steps, each searcher seeded with `filters: 5`), selects the first FWD, and `plRender()` (5430) draws it. With `pipelineStyle='Flow chart (6d)'` the renderer is the `pl6*` family (`pl6Card` 5296, `pl6Par` 5315, `pl6Tail` 5337, `pl6Seq` 5348, `pl6Ink` 5364) producing `.v6d-node` / `.v6d-plus` / `.v6d-fan` / `.v6d-junc--merge` / `.v6d-tailadd` / `.v6d-tool`; the `'Panel (default)'` branch uses `plCardEl`/`plParEl`/`plTailEl` + `plWires`/`plWireSvg` SVG wiring. **Only the 6d branch is reachable in the demo.**

### 6.4 Canvas interactions (`setupPipeline`, 5782–5872)

| Trigger | Result |
|---|---|
| click a `.pb-card` / `.v6d-node` | cancels any open config draft, `_plSel = id`, re-render (selection paint) |
| click `.v6d-plus[data-id]` / `.pb-add-s` / `.pl-join-add` | `plMenu(anchor, 'after', id)` |
| click `.v6d-fan[data-id]` / `.pb-add-p` | `plMenu(anchor, 'along', id)` — run alongside |
| click `.v6d-junc--merge[data-id]` | `plMenu(anchor, 'after', id)` |
| click `.v6d-tailadd` / `.pl-tailadd` | `plMenu(anchor, 'merge' | 'tail', after)` |
| click `.v6d-tool[data-act=del]` | `plDelete(id)` |
| click anywhere outside `.pl-scroll` | `plHideMenu()` |
| `Escape` | `plHideMenu()` |
| `Delete` / `Backspace` (not in a field) | `plDelete(_plSel)` |
| `⌘Z` / `^Z` | `plUndo()` |
| drag a card | `plDragWire` — touch lift threshold 13 px, ghost rides 64 px above the finger, drop-line radius 160 px on touch; `plSlots`/`plMoveFn` compute trunk slots with neighbour fallbacks; `pointercancel` springs back |
| window resize | `plWires()` re-draws the SVG wiring |

`.pl-addmenu` (2260–2268) header `Add step` + `Forward Searcher` / `Backward Searcher` / `Diversified Reranker` (+ `db`, `sem`); when hand-configured originals exist it appends a **Copy an existing searcher** section listing them by code — copies are marked `step.reused` and are themselves excluded from the list (`plMenu` 5695).

The config panel's `[aria-label="Delete step"]` also calls `plDelete(_plSel)` (5837).

---

## 7. Region G — config panel (`.panel-config`, 2269–2357, width `var(--sbw)`)

| # | Element | Selector | Lines | Notes |
|---|---|---|---|---|
| G1 | First-run overlay | `.cfg-first` | 2271–2277 | *Nothing to configure yet / Choose a starting point on the canvas…* — `display:flex` only when `data-first="1"` |
| G2 | Header | `.cfg-head` | 2279–2288 | `.cfgh-title` (serif 20 px, e.g. `Backward Searcher`), a **Delete step** icon button (hover → `--error-bg`), `.cfgh-code` (`BWD-03`), `.cfgh-desc` |
| G3 | Narrow-tier source seg | `.cfg-src-wrap` / `.fc-seg.cfg-src-seg` | 2290–2295 | **`Seed set` (+`.cfg-src-n` count) / `Search`**. `display:none` except `[data-vp=narrow][data-src=1]:not([data-first=1])` (CSS 23). Clicking writes `data-srcpage` on the root (2391–2394). |
| G4 | Scroll body | `.cfg-scroll` | 2297–2353 | hidden at narrow when `data-first=1` or `data-srcpage=search` (CSS 25) |
| G5 | Parameters group | `.cfg-params` / `.cfg-params-body` | 2298–2304 | rendered per step type by `plRenderParams` (5122) off `plParamDefs` (5083) |
| G6 | Seed papers group | `.cfg-seed` / `.cfg-seed-count` / `.cfg-seed-list` | 2305–2311 | rendered by `seedRender` (4825–4903); rows are starless `.cs-row` clones with a hover trash |
| G7 | Filter-pipeline head | `.cfg-pipe-head` / `.cfg-count` | 2312–2315 | `Filter pipeline` + count pill with tooltip `Filters in this step` |
| G8 | Pipeline tools | `.cfg-fp-tools` | 2316–2333 | **Copy** / **Paste** (disabled until a clipboard exists) / **Apply to all** / **Undo** (`.cfg-undo`, disabled until history exists). Paste and Apply-to-all both go through `cfgConfirm` popovers; every action fires a `pcToast` (3958–3973), Paste/Apply-to-all with an Undo affordance. |
| G9 | Filter rows | `.cfg-pipe` → `.cf-filter` ×5 | 2334–2339 | seeded: `year` (2023–2026) · `citation` (β = 30 cites/yr of age) · `keyword` (Abstract, a long boolean query in `data-query`) · `llm` (Title) · `llm` (Abstract). Each row = `.cf-rail` + `.cf-node`(icon + `.cf-num`) + `.cf-body`(`.cf-type`/`.cf-val`) + `.cf-tools`(**up** / **down** / **copy**). |
| G10 | Add filter | `.cfg-add-wrap` → `.cfg-add` + `.cfg-add-menu` | 2340–2351 | seven options: **Year range · Citation count · Keyword match · Author · Venue · LLM classifier · Similarity**. `cfgMenuSync` (3832) dims options whose function is already present, with explain-on-tap. The menu centres with `margin-left:-110px` (never a `translateX(-50%)` that `popIn` would knock out). |
| G11 | Drill host | `.cfg-shost` | 2355 | empty; receives the **whole search sidebar** at the narrow tier, and hosts focus-view editors |

### 7.1 Filter editing

* click a `.cf-filter` → `openConfig(row)` (4435) → `openDrill(row)` (4411) → `buildEditor(row)` (4313) renders a **borderless inline focus-view editor** into `.cfg-shost`, with a footer (`getFooter` 4512 / `showFooter` 4524) carrying a `.cfg-foot-status` chip.
* LLM filters get `buildLLM` (4202–4312) — a provider/model tree (`llmModels` 4106, `llmTree` 4112, `llmRule` 4122).
* Validation: `validateEditor` (4550–4582) + `fieldErr`/`fieldOk` (4530/4543); the status chip shows `N issues to fix` (`data-state="err"`) or `Applied ✓` (`data-state="ok"`, then auto-closes after 950 ms, 4595).
* `cancelConfig` (4190) drops the draft; `applyConfig` (4583) commits.
* Reorder: `.cf-tool[data-act=up|down]` → `moveFilter` (3767); **the arrows are hidden on touch** (`@media (hover:none)` hides `.cf-tool[data-act=up]/[data-act=down]`) and replaced by `wireCfDrag` — iOS-style 1:1 drag, 3 px threshold with a mouse, 240 ms hold on touch, neighbours slide via transforms, one FLIP commit on release, spring-back on `pointercancel`.
* `pushHistory`/`restorePipe`/`undo` (3876/3878/3886) keep ≤50 snapshots; `updateUndo` (3877) enables G8's Undo.

---

## 8. Complete reachable-state inventory (this is the parity interaction script)

Preconditions for every row: demo entry defaults, page = Build, `buildState = 'Has results'` unless stated. `⟨narrow⟩` = shrink the frame below 1241 px.

### 8.1 Baseline

| # | State | How to reach |
|---|---|---|
| S01 | **Build default** | load the demo (page defaults to Build) |
| S02 | Dark theme | click `.pc-theme-toggle` |
| S03 | Compact tier | frame width 1241–1440 (iPad Pro 12.9 landscape = 1366) |
| S04 | Narrow tier | frame width ≤1240 — sidebar moves into `.cfg-shost`, G3 seg appears, `.sb-head-row` hides |
| S05 | Rotate notice | frame width <1160 (shell overlay, covers the page) |

### 8.2 Top bar

| # | State | How to reach |
|---|---|---|
| S06 | Project menu open | click `.tb-project` |
| S07 | Project switched | S06 → click a non-current `.tb-proj-item` (420 ms fade-through) |
| S08 | Download menu open | click `.tb-download` |
| S09 | Download menu empty state | S08 with no runs → `.tb-dl-none` |
| S10 | Account menu open | click `.tb-user` |
| S11 | Account menu with **Enter full screen** | S10 on a touch device (`matchMedia('(hover:none)')`, not standalone) |
| S12 | Settings overlay | S10 → **Settings** |
| S13 | Logout → Login | S10 → **Log out** (page scales to .988 + fades over 170 ms, then Login) |
| S14 | Runs dot pulsing | start a run on Runs, return to Build |
| S15 | Runs dot static | let a run finish without visiting Runs |
| S16 | Demo-state switcher | 5 taps on `.tb-mark` within 2.5 s |

### 8.3 Sidebar — Search mode

| # | State | How to reach |
|---|---|---|
| S17 | Results list (14+ rows) | S01 |
| S18 | Row hover (+ star visible) | hover any `.pr` |
| S19 | Row selected | click a `.pr` (rail on, `--row-sel`) |
| S20 | Abstract drill | S19 (a row click opens it) |
| S21 | Abstract copied flash | S20 → `.ab-copy` |
| S22 | Star pressed / saved / row leaving | click `.pr-star` — 420 ms bounce then 440 ms slide-out |
| S23 | Seed set gains a paper | after S22, see `.cfg-seed` in region G |
| S24 | Searching skeletons | type in `.sb-search` and wait 600 ms, or press Enter (850 ms window) |
| S25 | Search results | 850 ms after S24 |
| S26 | Empty results | search a term matching nothing → `.sb-empty` |
| S27 | Cleared | S26 → **Clear search & filters** |
| S28 | Sort menu open | click `.sb-sort-btn` |
| S29 | Sorted by Newest / Title A–Z | S28 → pick (rows stagger) |

### 8.4 Sidebar — filter drawer

| # | State | How to reach |
|---|---|---|
| S30 | Filter drawer open | click `.sb-filter-btn` |
| S31 | Year range dragged | drag either `.dr-h[data-side]` on `[data-key=year]` |
| S32 | Citation range dragged | same on `[data-key=cites]` |
| S33 | Text filter typed | type in any `.ff-input` |
| S34 | Filter button reads `N filters` | S31–S33 then close |
| S35 | Reset | S30 → **Reset all** |
| S36 | Done with live count | S30 → **Done  N shown** |

### 8.5 Sidebar — Import mode

| # | State | How to reach |
|---|---|---|
| S37 | Import dropzone | `.sb-imode` → **Import** |
| S38 | Dropzone drag-over | drag a file over `.sbi-drop` |
| S39 | Parsing rows | S37 → **Browse files** (or drop) |
| S40 | Sample parse (deterministic) | S37 → **Or try a sample refs.bib** → 34 entries |
| S41 | Match review, all four sections | after S40: *Needs a decision* 2 · *Couldn't match* 3 · *Matched* 28 · *Already in the corpus* 1 |
| S42 | Section collapsed | click a `.ki-sec` header (label flips Hide/Show) |
| S43 | Candidate popover | click a `.ki-multi` **2 matches** pill |
| S44 | Candidate chosen | S43 → pick a record (row moves to *Matched*) |
| S45 | Entry unchecked | click a `.ki-ck` checkbox (footer count drops) |
| S46 | Footer error | uncheck everything → `.sbi-foot-err` |
| S47 | Added to seed set | **Add N to seed set** → seed count rises, matching library rows go starred |
| S48 | Unsupported file | drop e.g. `notes.docx` → row error *Unsupported format: use .bib, .ris, .csv, a DOI list, or PDFs* |

### 8.6 Pipeline canvas

| # | State | How to reach |
|---|---|---|
| S49 | Flow at rest (7 steps, 3 parallel waves) | S01 |
| S50 | Step selected | click a `.v6d-node` |
| S51 | Step hover tools | hover a node (`.v6d-tool`; always visible under `(hover:none)`) |
| S52 | Add menu — *Add step after* | click a `.v6d-plus` |
| S53 | Add menu — *Run alongside* | click a `.v6d-fan` |
| S54 | Add menu — *Add step* (tail) | click `.v6d-tailadd` |
| S55 | Add menu — *Merge branches into* | click `.v6d-tailadd[data-merge]` |
| S56 | Add menu with **Copy an existing searcher** | S52 after at least one searcher has been hand-configured |
| S57 | Step added | pick any option in S52–S56 |
| S58 | Step deleted | `.v6d-tool[data-act=del]`, or select + `Delete`/`Backspace`, or config header's Delete |
| S59 | Step dragged / reordered | press-drag a node (13 px touch threshold, ghost 64 px above the finger) |
| S60 | Drag cancelled | `pointercancel` mid-drag → spring back |
| S61 | Presets menu | click `.plh-preset` |
| S62 | Preset confirm popover | S61 → pick one |
| S63 | Preset applied | S62 → confirm (fade/stagger) |
| S64 | Undo | `.plh` Undo button, or `⌘Z`/`^Z` |

### 8.7 Config panel

| # | State | How to reach |
|---|---|---|
| S65 | Config for the selected step | S50 |
| S66 | Filter row hover tools | hover a `.cf-filter` |
| S67 | Filter focus-view editor | click a `.cf-filter` |
| S68 | LLM editor with model tree | click an `llm`-type `.cf-filter` |
| S69 | Editor validation error | clear a required field → `N issues to fix` |
| S70 | Editor applied | fix + apply → `Applied ✓`, auto-close after 950 ms |
| S71 | Editor cancelled | Cancel / `Escape` |
| S72 | Filter moved up/down | `.cf-tool[data-act=up|down]` (desktop) |
| S73 | Filter dragged | press-hold 240 ms on touch / 3 px with a mouse |
| S74 | Filter copied | `.cf-tool[data-act=copy]` |
| S75 | Filter deleted | delete from the row menu |
| S76 | Add-filter menu (7 options) | click `.cfg-add` |
| S77 | Add-filter menu with dimmed duplicates | S76 when that function already exists |
| S78 | Explain-on-tap for a dimmed option | S77 → tap a dimmed option |
| S79 | Filters copied | `.cfg-fp-tools` **Copy** → toast |
| S80 | Paste confirm | **Paste** with a non-empty step |
| S81 | Pasted + Undo toast | confirm S80 |
| S82 | Apply-to-all confirm | **Apply to all** |
| S83 | Applied to all + toast | confirm S82 |
| S84 | Pipeline Undo enabled/used | any of S79–S83 then `.cfg-undo` |
| S85 | Seed papers group populated | after S22/S47 |
| S86 | Seed paper removed | hover a `.cs-row` → trash |

### 8.8 Before-first-run (`buildState = 'Before first run'`)

Reach via the demo-state switcher (S16 → Build → **Before first run**) or Home → create a project (`kl-open-project {fresh:true}`).

| # | State | Detail |
|---|---|---|
| S87 | Sidebar first-run invite | `.sb-first` — *Find your seed papers* + three chips; search stays fully functional |
| S88 | Chip search | click a chip → fills the field and searches → invite is replaced by results, `data-searched="1"` |
| S89 | Canvas first-run | `.pl-first` — *Shape your pipeline*, 3 preset cards, **Start from the seed set** |
| S90 | Canvas header stripped | `.plh-count`, `.plh-preset-wrap` and Undo are hidden |
| S91 | Config first-run | `.cfg-first` — *Nothing to configure yet* |
| S92 | Seed set empty | `applyPage` stashes existing seeds into `_seedStash` and empties the list |
| S93 | Seeds restored | switch back to *Has results* — `_seedStash` is re-applied |
| S94 | First-run dismissed permanently | pick a preset or **Start from the seed set** → `_plFirstDone = true`, `data-first="0"` |
| S95 | Narrow + first-run | ⟨narrow⟩ + S87 — `.cfg-first` and `.cfg-head` hide, `.cfg-scroll` hides, `.cfg-shost` shows the sidebar |

### 8.9 Narrow-tier-specific

| # | State | How to reach |
|---|---|---|
| S96 | Sidebar inside the config column | ⟨narrow⟩ |
| S97 | `Seed set N | Search` seg | ⟨narrow⟩ with `data-src="1"` and not first-run |
| S98 | Seg on *Search* | S97 → **Search** (`data-srcpage="search"`, `.cfg-scroll` hides, `.cfg-shost` shows) |
| S99 | Seg on *Seed set* | S97 → **Seed set** |

### 8.10 Cross-page overlays that render above Build

| # | State | How to reach |
|---|---|---|
| S100 | Settings centred card | S12 (closes on ✕ / Cancel / scrim / `Escape` / Save) |
| S101 | Status strip (offline / server unreachable / restored) | shell `connection` prop — Build reserves the height via `--bn-inset` |
| S102 | Floating banner pill | `kl-banner` from another page |

---

## 9. Designed-but-unreachable / missing states found on Build

Recorded in full in `missing-states.md` as `MS-DIS-01`…`MS-DIS-10`; the two that also need a product decision are in `escalations.md` as `E-DIS-06` (`Run pipeline`) and `E-DIS-07` (the dead `cards` sidebar). Summarised here because they belong to this spec:

1. **`.sb-error` "Couldn't load papers" is unreachable.** `apply()` (2798–2807) only renders it when `sidebar.dataset.state === 'error'`, and the only writer is a `.sb-demo` switcher (3065–3078) **that does not exist in the shipped template** (`grep sb-demo` → 1 hit, in the JS only). So the demo can never show a search failure.
2. There is **no loading state for the pipeline or the config panel** — only the search list has skeletons.
3. **`Run pipeline` (A10) has no wired handler on Build.** `.tb-run` is queried on Runs (`Runs.dc.html:1852/2012/2042`) but nowhere in `Paper Card.dc.html` — clicking it on Build does nothing.
4. `.tb-download` items have no disabled/failed state; only the empty `.tb-dl-none`.
5. No offline/degraded state for the Semantic Scholar search or the import resolver — a failed import surfaces only through per-entry `Couldn't match`.
