# component-duplication.md — the copy-paste map

> 中文摘要：Claude Design 的通病是「组件被复制粘贴到每个 screen」。本文把 demo 里**所有**重复组件列出来，逐字比对副本，标出分歧；并对每处分歧说明 **iPad Demo 在哪个 screen 上实际渲染的是哪一版**。凡是逼迫我们做「哪一版才是正统」判断的分歧，同步登记到 `escalations.md`（标 D-01…D-08）。
>
> Method: byte/normalized comparison of the sibling `.dc.html` files (which are byte-identical to the compiled bundle — see demo-architecture.md §1.3). Normalized = whitespace collapsed; "IDENTICAL" below means the two copies are the same after that.

## 0. Scale of the problem

| Metric | Value |
|---|---|
| CSS class names appearing in more than one page template | **133** |
| Pages carrying a full top-bar copy | 4 (Build, Runs, Explore, Home) |
| Pages carrying a full papers-panel copy | 3 (Build with its own prefix, Runs, Explore) |
| Largest single-file duplication | Build's two `.sidebar` copies — **280 lines, only 3 of them different** |
| Shared code that is genuinely shared (one file, imported) | `paper-row.js`, `import-resolver.js`, `run-mock.js`, `stream-text.js`, the viz engines |

Only **five** things in this design system are actually shared rather than copied. Everything else in the table below is a copy.

---

## 1. Top bar (`<header>`, 58 px)

| Page | Lines | Length (normalized) |
|---|---|---|
| Build | `Paper Card.dc.html` 1331–1506 | 176 lines |
| Runs | `Runs.dc.html` 829–1008 | 180 lines |
| Explore | `Explore.dc.html` 279–454 | 176 lines |
| Home | `Home.dc.html` 227–270 | 44 lines (logo + theme + account only — no project switcher, no tabs, by design) |

### 1.1 Build vs Runs — 3 differences, all intentional

1. `data-active="1"` moves from the Build tab to the Runs tab.
2. Runs' `.tb-run` carries `data-when="pre idle" data-display="flex"` (state-gated visibility).
3. Runs adds a `.tb-stop` **Stop run** button (Runs 1009-region, absent on Build/Explore).

**Renders:** each page renders its own copy; only the active-tab paint is reconciled at runtime by the shell (`show()` re-paints `data-active` by `textContent`). ✅ No canonical conflict.

### 1.2 Build vs Explore — 4 differences

| # | What | Build / Runs | Explore |
|---|---|---|---|
| **D-01** | Project-switcher list | slot 1 = `AI agents for scientific discovery` **(current ✓)**, slot 2 = `Evolutionary multi-objective optimization` | slot 1 = **`Bounded gaps between primes`** (a project name that exists nowhere else in the app), slot 2 = `AI agents for scientific discovery` (current ✓) |
| **D-02** | Download menu | see §2 | see §2 |
| D-03 | Theme toggle | carries `<span class="tt-label" style="display:none">`, written by `applyTheme` (`Paper Card.dc.html:3416`, `Runs.dc.html:1935`) but never displayed | no `tt-label`; `applyTheme` does not reference it |
| D-04 | Download wrapper | `.tb-download` sits inside `<div class="tb-dl-wrap">` | same wrapper, one indentation level shallower (cosmetic only) |

**Renders:** the demo renders the Explore copy on the Explore screen and the Build/Runs copy elsewhere, so a user switching tabs sees the project list **re-order itself and gain/lose a project**. D-01 is a real product-visible inconsistency → escalation.

D-03 is dead markup: `tt-label` is `display:none` in both places that have it. Canonical choice = **drop it** (Explore/Home version), but it is invisible either way.

### 1.3 Account menu (`.tb-user-menu`)

**IDENTICAL across Build, Runs, Explore, Home** (2549 normalized chars). Contents: identity row `Ada Lovelace`, `Settings`, `Log out`. ✅ No divergence. The JS behind it, however, is not shared:

| Page | Method | Lines | Result |
|---|---|---|---|
| Build | `setupUserMenu` | 5001–5020 | **IDENTICAL** (sha cd585619) |
| Runs | `setupUserMenu` | 2631–2650 | **IDENTICAL** (sha cd585619) |
| Explore | inline in `componentDidMount` | 1507–1520 | rewritten by hand, same behaviour |
| Home | inline `setupUserMenu`-like block | 701–718 | rewritten by hand, same behaviour |

### 1.4 `klFsRow` (the *Enter full screen* row)

Four copies of the same 23-line method.

| Page | Lines | sha |
|---|---|---|
| Build | 5022–5044 | 77accde6 |
| Runs | 2652–2674 | 77accde6 |
| Home | 720–742 | 77accde6 |
| Explore | 3790–3812 | **f96821bd** |

**D-05:** Explore's copy differs by exactly one identifier — the local `const on` is renamed `const on2` (presumably to dodge a shadowing warning). Behaviourally identical. ✅ Cosmetic; canonical = the three-way-identical version.

### 1.5 Project switcher JS (`setupProjMenu`)

| Page | Lines | sha |
|---|---|---|
| Build | 4947–4971 | a8334f8f |
| Runs | 2577–2601 | a8334f8f |
| Explore | 1522–1547 | **aca2e2ec** — identical plus one added line `this.xpDlWire(root);` |

✅ Additive, not conflicting.

---

## 2. Download menu (`.tb-dl-menu`)

| Page | Lines | Normalized |
|---|---|---|
| Build | 1448–1474 | 3038 |
| Runs | 946–972 | 3038 — **IDENTICAL to Build** |
| Explore | 395–424 | **3732 — different component** |

**D-02 divergences (Explore vs Build/Runs):**

| Aspect | Build / Runs | Explore |
|---|---|---|
| Header row | `.tb-dl-head` + `.tb-dl-sep` + `.tb-dl-run` (`Run 6`) + `.tb-dl-date` (`yesterday`) | unclassed header: `Download` + `Run 37` + `v3` + right-aligned `what Explore reads` |
| First item | **Accepted papers** — `CSV · <span class="tb-dl-n">214</span> papers` | **Corpus papers** — `CSV&nbsp;&nbsp;354 papers` |
| Second item | — | **Literature review** — `Markdown&nbsp;&nbsp;the current draft` |
| Bundle item subtitle | `CSVs · BibTeX · graph · stats` | `CSVs&nbsp;&nbsp;BibTeX&nbsp;&nbsp;graph&nbsp;&nbsp;topics` |
| Empty state | `.tb-dl-none` — *No runs yet / Results download per run. Start one with Run pipeline.* | **absent** |
| Separator style | literal `·`-shaped separators in the source | `&nbsp;&nbsp;` (the project's own "no `·` separators" typography ban) |

**Renders:** Explore's screen shows the corpus-scoped menu; Build and Runs show the run-scoped menu. This is arguably intentional scoping (a corpus download vs a run download), but two things are not: the **separator style differs** (Explore follows the documented typography ban, Build/Runs violate it) and **only Build/Runs have an empty state**. → escalation D-02.

**D-06 (JS):** `setupDownload` differs on exactly one line —

```js
Build : menu.querySelector('.tb-dl-run').textContent = String(run.id).replace(/^RUN-0*/i, 'Run ');
Runs  : menu.querySelector('.tb-dl-run').textContent = this.rid(run.id);
// Runs.dc.html:3849 → rid(x) { return String(x??'').replace(/^RUN[-‑]0*/i, 'Run '); }
```

Build's hand-rolled copy **does not handle the non-breaking hyphen `‑`**, so a run id written with one renders as `RUN‑0006` on Build's download menu and `Run 6` on Runs'. A latent copy-paste bug, and a violation of the project's own rule that run ids display via `rid()`. → escalation D-06 (or just fix it: use one helper).

---

## 3. Papers panel + filter panel + sort menu

Three copies of the same panel with **two different class prefixes**:

| Copy | Page | Prefix | Lines |
|---|---|---|---|
| A | Build | `sb-*` | header 1553–1585, filter drawer 1514–1551, list 1587–1791 |
| B | Runs | `rp-*` | header 1019–1077, pager 1078, filter drawer 1079–1122 |
| C | Explore | `rp-*` | header 462–531, filter drawer 532–585 |

### 3.1 Build (`sb-*`) vs Runs/Explore (`rp-*`)

Same anatomy — title row, seg, search field, Filter button, Sort button + 3-option menu, list, empty state, filter drawer with two `.dr` ranges + text fields + Reset/Done — implemented with **completely different class names** and duplicated JS (`wireSidebar` on Build vs `rpWire*` on Runs/Explore). There is no textual overlap to diff; this is a full re-implementation.

**D-07:** two prefixes for one component. A rewrite must pick one. The `rp-*` version is the more evolved one (it carries the selection bar, pager, drill and change markers); the `sb-*` version carries the Search|Import seg and the first-run invite. Canonical choice needed. → escalation.

### 3.2 Runs vs Explore filter drawer — copy with divergences

`Runs.dc.html:1079–1122` vs `Explore.dc.html:532–585`. Identical structure; differences:

| # | Runs | Explore |
|---|---|---|
| root class | `rp-filter-panel` | `rp-filter-panel **lp-pap**` |
| header label | `Filters  <span class="rp-fp-tab">accepted</span>` (scoped to the current tab) | `Filters` (unscoped) |
| year range | `min=2022 max=2026`, label `2022–2026` | `min=2023 max=2026`, label `2023–2026` |
| citations range | `min=0 max=800 step=10`, label `0 – 800+` | `min=0 max=100 step=5`, label `0 – 100+` |
| extra switches | — | `.rp-cited-row` *Cited / In the review draft* (hidden by default) and `.rp-pdf-row` *PDF / Full text available* |
| pick label | `<label class="rp-pick-label">Step accepted</label>` | unclassed `<label>Step accepted</label>` |

Range bounds and the two switches are legitimately data-scoped. The **`rp-pick-label` class present on one copy and absent on the other** is pure copy-paste drift (Explore's JS cannot target it).

**Renders:** Runs shows the `accepted`-scoped header and 0–800 citations; Explore shows the plain header, 0–100 citations, and the two extra switches.

---

## 4. Abstract drill

**Three implementations of one component.**

| Copy | Page | Where | Form |
|---|---|---|---|
| A | Build | `wireAbstract` 3100–3285, markup built as strings at 3200–3220 and 3251+ | **generated in JS** |
| B | Runs | `Runs.dc.html` 1123–1150 (`.rp-drill` → `.ab-*`) | markup |
| C | Explore | `Explore.dc.html` 586–~620 (`.rp-drill` → `.ab-*` + extras) | markup |

### 4.1 Runs vs Explore — diffed

| # | Aspect | Runs | Explore |
|---|---|---|---|
| | z-index | `40` | `42` |
| **D-08a** | meta line font-size | **11.5 px** | **13 px** |
| **D-08b** | `Abstract` section label | `10.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--muted2)` (caps label) | `13px; font-weight:600; color:var(--fg)` (sentence-case heading) |
| | groups block | — | `.ab-groups` (topic/community chips) |
| | provenance row margin | `margin-top:10px` | `margin-top:14px` |
| | In-this-corpus module | — | `.cc-mod[data-cc]` with `.cc-sumbtn` **Summarize in chat**, `.cc-sub` *What other papers here say when they cite it* |

D-08a/b are **the same element styled two different ways in two screens**. This forces a canonical choice → escalation.

Build's JS copy (A) uses the **Runs** conventions (11.5 px meta, 10.5 px uppercase `Abstract` label) plus its own `.ab-save` button that neither markup copy has.

**Renders:** the Explore screen shows the 13 px / sentence-case version; Runs and Build show the 11.5 px / caps version.

---

## 5. Citation-network canvas chrome (`nv-*`)

Runs `1151–1215` vs Explore `779–900`. ~30 shared class names (`nv-tools`, `nv-lbl`, `nv-zoom`, `nv-fitbtn`, `nv-pbtn`, `nv-fbtn`, `nv-body`, `nv-canvas`, `nv-stats`, `nv-tip`, `om-tip`, `nv-legend`, `nv-leg-lo/-s/-hi`, `nv-hood`, `nv-hood-ic-in/-out/-txt/-n`, `nv-pop`, `nv-sl`, `sl-fill`, `sl-h`, `nv-sl-val`, `nv-sw`, `sw-p`, `sw-k`, `nv-seg`, `nv-dd`, `nv-dd-val`, `nv-num`, `nv-fcount`, `cfg-pop`, `fc-dd`, `fc-dd-btn`, `fc-dd-menu`, `fc-dd-opt`).

Diffed differences:

| # | Runs | Explore |
|---|---|---|
| `.nv-tools` | `data-when="run idle" data-display="flex"` (state-gated) | plain `display:none` toggled by JS |
| `.nv-body` | `data-when="run idle" data-display="block"` | plain |
| `.nv-legend` | 12 lines | 12 lines **wrapped in an extra `<span class="nv-leg-yr">`** (year-colour legend can be swapped out) |
| `.nv-hood` | no `z-index` | `z-index:20` |
| after `.nv-hood` | `.cfg-pop.nv-pop[data-pop=layout]` (ForceAtlas 2 sliders) | an extra `.cm-sub` button **View community subgraph** (with in/out glyphs) at the same anchor position |

Both pages then repeat the same slider/switch/segmented markup for the Layout / Style / Filters popovers. The **engines are genuinely shared** (`network-viz.js`) — only the chrome is duplicated.

⚠ Overlap hazard for the rewrite: on Explore, `.nv-hood` and `.cm-sub` are both `position:absolute; left:14px; bottom:12px` — they are mutually exclusive at runtime, not stacked. Any naive port must preserve that exclusivity.

---

## 6. Duplication *inside* Build: the two sidebars

`Paper Card.dc.html` ships **two `.sidebar` blocks** inside one `.pc-drawer`:

* `[data-style="list"]` lines **1513–1792** (280 lines)
* `[data-style="cards"]` lines **1793–2075** (283 lines)

A full diff yields **7 changed lines**, i.e. 3 real differences:

1. `data-style="list"` → `"cards"`
2. `.sb-list` gains `padding:10px; display:flex; flex-direction:column; gap:10px`
3. the trailing `</aside>` lives in the second copy

Lines 1514–1585 (the entire filter drawer + panel header + search + filter/sort row) are **byte-identical** between the two.

`applyRowStyle` (3404–3408) simply toggles `display` between them, and the shell pins `layout="List"`, so **the `cards` copy never renders in the demo**. It is 283 lines of dead UI that nonetheless gets wired (`setupSidebar` iterates `querySelectorAll('.sidebar')`, 2528) — every listener, ResizeObserver and scrim is installed twice.

**Renders:** `list` only. Recommendation for the rewrite: build one panel component; do **not** port the `cards` variant (it is not in the locked variant set and the product owner's truth file never shows it).

---

## 7. Shared inline `<script>` blocks (helmet)

| Block | Pages | Verdict |
|---|---|---|
| `__klTouchAid` touch helper | Build, Runs, Explore, Home, Login, Settings | **byte-identical, 1323 normalized chars, sha ad73234e** — 6 copies of one file's worth of code. Deduped at runtime only because the helmet manager keys scripts by their text. |
| `__resources` shim | Build, Explore, Home (sha 17f30278) | identical 3-way |
| `__resources` shim | Runs (sha 6d4fc3d5) | differs — additionally mounts `run-mock.js` |

Neither is a divergence; both are pure duplication that a rewrite should collapse to one module.

---

## 8. Duplicated CSS token blocks

Every page opens its `<style>` with its own `.pc-root { … }` token block. All seven are different lengths and different hashes:

| Page | Lines | Normalized chars |
|---|---|---|
| Build | 17–527 | 59 623 |
| Runs | 17–790 | 91 546 |
| Explore | 23–243 | 30 474 |
| Home | 18–192 | 24 401 |
| Login | 18–95 | 15 287 |
| Settings | 18–87 | 14 838 |
| System Banners | 18–100 | 13 099 |

These are not literal copies — each page carries the shared tokens **plus its own component rules**, all in one block. The shared prefix (colour schemes, `--fg/--muted/--card/--border/…`, `data-theme` overrides, `data-logo` variants, `data-seg` variants, the `@media (hover:none)` and `@media (pointer:coarse)` blocks) recurs in every file. `explorations-tokens.css` (57 KB) exists beside them but is **not loaded by any page**.

A rewrite should extract exactly one token sheet. Because the blocks are not identical, extraction must be done by **diffing the rules, not by taking any single page's block** — the risk is silently dropping a page-specific override.

---

## 9. Summary of divergences requiring a canonical choice

| ID | Component | Divergence | In escalations.md as |
|---|---|---|---|
| D-01 | Project switcher list | Explore lists `Bounded gaps between primes` and re-orders; Build/Runs list `Evolutionary multi-objective optimization` | `E-DIS-01` |
| D-02 | Download menu | Explore = corpus-scoped, different separators (`&nbsp;` vs `·`), no empty state; Build/Runs = run-scoped with empty state | `E-DIS-02` |
| D-06 | `.tb-dl-run` label | Build's inline regex vs Runs' `rid()` — Build misses `‑` | `E-DIS-03` |
| D-07 | Papers panel prefix | `sb-*` (Build) vs `rp-*` (Runs/Explore) for one component | `E-DIS-04` |
| D-08 | Abstract drill typography | meta 11.5 px + caps `Abstract` (Runs/Build) vs 13 px + sentence-case `Abstract` (Explore) | `E-DIS-05` |
| §6 | Build's never-rendered `cards` sidebar | 283 dead lines, wired twice | `E-DIS-07` |
| §5 | Explore's `.nv-hood` / `.cm-sub` sharing one absolute position | layout contract that only JS enforces | `E-DIS-08` |

Cosmetic-only, recorded but not escalated: D-03 (`tt-label` dead markup), D-04 (indentation), D-05 (`on` → `on2`), `rp-pick-label` class drift, `.nv-hood` z-index.

Not a duplication finding but escalated alongside these: `Run pipeline` (`.tb-run`) has no handler on Build or Explore — see `build-page-spec.md` §9 and `escalations.md` `E-DIS-06`.
