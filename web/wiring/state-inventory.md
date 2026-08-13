# state-inventory.md — cross-screen stateful & dynamic behaviour

> 中文摘要：对拍（parity capture）必须知道 demo 里**什么在动、什么会漂**。本文列出所有计时器、流式动画、随机源、持久化 key、跨页事件与登录直通，并给出每一项的「对拍处理建议」（冻结 / 等待 / 忽略 / seed）。
>
> Line references are to the sibling `.dc.html` / `.js` files in `$UI` (byte-identical to the compiled bundle).

---

## 1. The one structural fact everything else follows from

**Pages are never unmounted.** The shell mounts all seven screens at once and toggles `.pg[data-on]` (opacity/visibility/pointer-events, 220 ms). So:

* every page's `componentDidMount` runs **once, at boot**, for all seven screens;
* every timer, `ResizeObserver`, `MutationObserver` and rAF loop keeps running across navigation unless the page explicitly gates it;
* DOM state (selections, open popovers, scroll positions, draft edits) survives leaving and returning to a screen;
* there is **no route, no URL, no history, no reload path**. Killing and relaunching resets everything — that is the only reset.

Parity implication: a capture of screen X taken after visiting screen Y is not necessarily the same as a cold capture of X. **Capture order matters.** Prescribe it explicitly in the harness.

### 1.1 The one gate that exists

`show()` dispatches `document` event `kl-page-shown {name}`. Three pages listen and gate their canvas rAF loops:

| Page | Handler | Gates |
|---|---|---|
| Runs | `pgVis` `Runs.dc.html:4064–4068` | `nv.setVisible(on && data-state !== 'pre')` |
| Explore | `pgVis` `Explore.dc.html:3778–3787` | `nv.setVisible(on && ready && view==='network')`, `tm.setVisible(on && view==='topics' && ready)` |
| Login | `pgVis` `Login.dc.html:246` | `nv.setVisible(on)` |

`_pgOff` also gates the pages' internal `setVisible` calls (`Runs:4061`, `Explore:3025/3772`). Nothing else is gated — Runs' replay interval keeps ticking while you are on Build.

---

## 2. Timers, by page

Counts (whole file): `setInterval` / `setTimeout` / `requestAnimationFrame`

| Page | int | timeout | rAF |
|---|---|---|---|
| Runs | 5 | 63 | 19 |
| Explore | 4 | 34 | 17 |
| Build (Paper Card) | 1 | 39 | 21 |
| Home | 0 | 15 | 2 |
| Login | 1 | 10 | 1 |
| System Banners | 1 | 7 | 2 |
| Settings | 0 | 7 | 0 |
| Shell (demo entry) | 0 | 4 | 0 |

### 2.1 Runs — the replay engine (the biggest parity hazard)

| Handle | Period | Started by | Stopped by | Effect |
|---|---|---|---|---|
| `_rplT` | `KLRunMock.REPLAY.tickMs = **320 ms**` | `startSeq` (2005–2035) | `rplStop` (2793–2806), `data-state !== 'run'`, teardown 3945 | drives `frameAt(Date.now() - _rplT0)` → repaints steps, PRISMA counts, filter table, monitor dock, accepted/rejected lists, canvas |
| `_elT` | **1000 ms** | `applyPage('Running')` (2790) when no replay is live | mode change, teardown | ticks `.rn-elapsed`; **seeded at `_elS = 758`** ⇒ the canned Running state starts at `12:38` and climbs |
| `_phT` | **1000 ms** | `phaseSet('stopping')` (2028) | `phaseSet('')` | `stopping for Ns`, caption advances every 3 s through 3 stages |
| `_nvT` | (variable) | `nvStream` (4055) | 4049/4058, teardown | streams new nodes onto the citation canvas; **uses `Math.random()` at 4056** |
| smooth-scrollbar poll | 80 ms | `setupRubber` (1921) | on first success | only when `overscrollBounce` is true — **the demo passes `false`, so it never runs** |

**The replay is a pure function of wall-clock delta.** `frameAt(t)` is deterministic given `t`; `_rplT0 = Date.now()` is the only entropy. Total run length is exactly **60 000 ms**.

Parity handling: either (a) freeze by never starting a run and capturing only `Idle with history` / `Before first run`, or (b) drive the clock — inject `_rplT0` and call `frameAt(t)` at fixed offsets (e.g. 0 / 5 / 15 / 30 / 45 / 59.9 s). **Do not** capture the `Running` default state without pinning `_elS`, because `.rn-elapsed` climbs one second per second from 758.

### 2.2 Explore — streaming and the skill task card

| Handle | Period | Where | Notes |
|---|---|---|---|
| `streamHtml` rAF | **220 chars/s** default | `stream-text.js`, driven from `Explore.dc.html:4152` | word-by-word reveal of the review draft and the "In this corpus" summary; blocks fade in with a 220 ms `translateY(4px)` animation; a `.rv-caret` follows the cursor |
| `_ccIv` | interval, +6 papers per tick | 2353 | "In this corpus" scoring progress |
| `_ttIv` | interval | 4406 and 4449 | skill-task-card phase bars. **Both advance by a random step**: `k + step + Math.floor(Math.random()*step)` (4408) and `k + 2 + Math.floor(Math.random()*4)` (4451, the 42-abstract bar) |
| `_rvTick` | **30 000 ms** | 5620 | repaints the review draft's "edited N minutes ago" meta |

Also: a send is classified as the review flow by `/\blit(?:erature)?\s+review\b/i` **or** the `/literature-review` skill, and only when `data-state === 'ready'` (`Explore.dc.html:1692`).

Parity handling: the two `_ttIv` bars are **non-deterministic** — the progress numbers differ between two runs of the same interaction. Either stub `Math.random` to a fixed value in the harness, or exclude the task-card progress region from pixel comparison and assert only its terminal state.

### 2.3 Build

| Handle | Period | Where | Notes |
|---|---|---|---|
| search debounce `_sq` | **600 ms** | 2893 | input → search |
| search latency | **850 ms** | 2886 | fake round-trip; `_sqTok` cancels stale ones |
| import file settle | `340 + Math.random()*280` ms **per file** | 4683 | **non-deterministic** — the parse rows settle at random intervals |
| `applied ✓` auto-close | **950 ms** | 4595 | config editor |
| `vpAttach` retries | 200 / 800 / 2000 ms | 2387 | ResizeObserver re-attach |
| `_actRepaint` | `setTimeout(0)` + MutationObserver on `.pg[data-on]` | 2380–2384 | Runs activity dot |

### 2.4 Login

Fixed cascade, total ≈ 3.28 s + a 620 ms camera tween (`Login.dc.html:467–485`):

```
click Sign in → validate
  data-state="Signing in"      +0
  data-state="Signed in"       +1300 ms
  data-exit="1" + camTween(620 ms) +2050 ms
  dispatch kl-login             +2480 ms   ← the shell switches to Home here
  reset (state='', pw cleared, nv.fit) +3280 ms
```

`sso(button)` is pure theatre: 1400 ms spinner, no event, no navigation.
`_nvFitI` (339) and `_nvCapT` (386) manage the background graph's fit/calm.

### 2.5 System Banners

| Handle | Period | Where |
|---|---|---|
| per-pill `_ttl` | `o.ttl` ms (e.g. 3600 000 for the run-done pill, 4000 for *Connection restored*) | 163/190 |
| connection countdown | **1000 ms**, writes `.bn-cd` `Ns` | 234–239 |
| `connProbe` retry | 1100 ms, backoff `_back = min(12, _back+3)` s | 242–248 |

The strip publishes its height back to the shell as `kl-banner-inset {h}` → `--bn-inset`, and every page reserves that height (Build: `Paper Card.dc.html:1508`).

### 2.6 Canvas engines (always-on rAF)

`network-viz.js` and `topic-viz.js` each run their own `requestAnimationFrame` force-simulation loop. Four instances exist (Runs network, Explore network, Explore topic map, Login background); `setVisible()` gates them (§1.1). **They never fully settle to a fixed layout** — layouts are seeded and frozen from the embedding, but user edits re-heat the sim, and `network-viz.js` uses `Math.random()` for jitter (lines 149–150, 618, 695) and `topic-viz.js` at line 442.

Parity handling: canvases are **not pixel-comparable**. Assert structure (node count, selected node, legend labels, tool state) or mask the canvas rect.

---

## 3. Nondeterminism register

| Source | Sites | Parity treatment |
|---|---|---|
| `Math.random()` in force sims | `network-viz.js:149,150,618,695`; `topic-viz.js:442` | mask the canvas |
| `Math.random()` in skill-task progress | `Explore.dc.html:4408, 4451` | stub RNG or mask the bar |
| `Math.random()` in file-settle stagger | `Paper Card.dc.html:4683`; `Home.dc.html:953`; `Runs.dc.html:5359` | wait for the terminal state before capturing |
| `Math.random()` as an id | `Home.dc.html:1205` (`'p'+Date.now()`, `seed:`), `Runs.dc.html:4477` (`_klgId`) | ids never render; ignore |
| `Math.random()` in node streaming | `Runs.dc.html:4056` | mask the canvas |
| `Date.now()` | Runs ×8 (replay clock), Explore ×18, Home ×3 | inject a clock |
| `new Date().getFullYear()` | `Home.dc.html:886` | **will change on 1 Jan** — pin the date in the harness |

**Everything else is deterministic**, including the whole import mock (`import-resolver.js` uses a seeded LCG keyed on `hash(filename + ':' + i)`, and `refs.bib` is a hard-coded 34-entry fixture).

---

## 4. Persistence

**Exactly one `localStorage` key exists in the entire app.**

| Key | Written | Read | Payload | Guard |
|---|---|---|---|---|
| `kl-filter-groups` | `Runs.dc.html:4476` (`JSON.stringify(g)`) | `Runs.dc.html:4470`, `Runs.dc.html:5730`, `Paper Card.dc.html:4348` | filter-group definitions shared between Build and Runs | try/catch on both sides |

It is mirrored at runtime by the `document` event `kl-filter-groups` so both pages update without a reload.

There is **no** sessionStorage, no IndexedDB, no cookies, no service worker, no `window.name` persistence. Everything else is in-memory and dies with the tab.

Parity implication: clear `localStorage['kl-filter-groups']` before every capture run, or the second run diverges from the first.

---

## 5. Cross-screen event bus (what a capture on screen X can change on screen Y)

| Event | Target | Cross-screen side effect |
|---|---|---|
| `kl-project {name}` | `document` | **every** page's `applyProject` re-labels the top bar and plays a 420 ms fade-through on `<main>` — a project switch on Build changes Runs and Explore too |
| `kl-run-activity {state,label}` | `document` | paints the Runs dot on Build's and Explore's top bar; `window.__klRunAct` holds the last value so a late-mounting page can catch up |
| `kl-filter-groups` | `document` | Build ↔ Runs filter-group sync |
| `kl-banner` / `kl-banner-hide` | `document` | the banner layer sits above **all** pages |
| `kl-banner-inset {h}` | `document` | changes `--bn-inset` on the shell root ⇒ every page's content shifts down |
| `kl-goto-runs` | bubbling | Explore's corpus chip deep-links Runs to a specific version — changes Runs' viewed run |
| `kl-open-settings` | bubbling | opens the shared Settings overlay above whatever page is showing |
| `kl-page-shown {name}` | `document` | gates the canvas loops (§1.1) |
| `kl-modal` | `document` | the one global `klModal` (used by Runs for restore/delete confirmations) |
| theme `MutationObserver` | shell `syncTheme` (entry 212–233) | toggling the theme on any page mirrors `data-theme` onto every other `.pc-root`; **no single source of truth**, guarded only by `_themeLock` |

---

## 6. Login passthrough

* The demo boots to **Build**, not Login. Login is reachable only via account menu → **Log out** (`kl-logout`).
* Sign-in performs **client-side validation only**: email must match `/^\S+@\S+\.\S+$/`, password must be non-empty (`Login.dc.html:459–464`). There is no credential check — any well-formed pair succeeds.
* Success dispatches `kl-login`, which the shell maps to **Home** (entry line 152), not back to Build.
* `kl-logout` re-arms Login's background sim (`Login.dc.html:213`: `document.addEventListener('kl-logout', … setTimeout(arrive, 160))`).
* SSO buttons do nothing (1400 ms spinner, `Login.dc.html:488–493`).
* This matches decisions-ledger Q2/Q16: **登录页本期不接，随便输都能进**. The parity harness should treat Login as a decorative screen: capture its two states (`idle`, `Signing in`) and assert only that a valid submit reaches Home.

---

## 7. Page-state props (the only "external" state)

Injected by the shell, overridable by the hidden demo-state switcher (5 taps on the logo):

| Prop | Values | Default | Reaches |
|---|---|---|---|
| `buildState` | `Before first run` \| `Has results` | `Has results` | Build `data-state`, `data-first`, sidebar `data-first`, seed stash |
| `runState` | `Before first run` \| `Idle with history` \| `Running` | `Running` | Runs `data-state` = `pre`/`idle`/`run` |
| `exploreState` | `Before first run` \| `Has results` | `Has results` | Explore `pageState` |
| `homeState` | `Has projects` \| `First visit` | `Has projects` | Home empty states |
| `connection` | `Online` \| `Offline` \| `Server unreachable` \| `Restored` | `Online` | System Banners |
| `apiRetries` | boolean | `false` | Runs retry ticker |
| `keyProbe` | `All valid` \| `Invalid Gemini key` \| `Rate-limited S2` \| `OpenAI unreachable` | `All valid` | Settings key chips |
| `runPhase` | hard-coded `'None'` by the shell | — | Runs hero phase row |
| `viewport` | `Fill the window` \| 6 fixed frames | `Fill the window` | shell root width/height |

Locked variants (do not flip): `pipelineStyle='Flow chart (6d)'`, `networkPalette='Neutral ink'`, `collectionStyle='Cover rows'`, plus shell-pinned `scheme='Warm paper'`, `logoStyle='Terracotta tile'`, Build `layout='List'`, Settings `settingsStyle='Centered card'`, `overscrollBounce=false`.

**Note:** `_buildOv` is a one-shot override set by `kl-open-project {fresh:true}` and cleared whenever `props.buildState` changes (entry 210). A capture that arrives at Build via Home's wizard is in `Before first run` even though the prop says `Has results`.

---

## 8. Responsive state (a product requirement)

Every page owns a `ResizeObserver` on its own `.pc-root` and writes `data-vp`:

```
clientWidth <= 1240 → narrow
clientWidth <= 1440 → compact
otherwise           → full
```

Plus `data-vp-ready="1"` two rAFs after the first measurement (suppresses load-time drawer transitions), and the shell's own `data-rot="1"` when the root is narrower than 1160 px.

Structural DOM moves at `narrow` (not just CSS):

| Page | Move |
|---|---|
| Build | `_vpPlace()` (`Paper Card.dc.html:3921–3928`) moves the whole `<aside class="pc-drawer">` into `.cfg-shost` and back |
| Runs | the papers panel folds into the right panel as Runs/Accepted/Rejected tabs; the Monitor dock becomes a compact `.mon-mini` 3-row block |
| Home | `_libPlace()` moves the Library panel into the Projects panel behind a seg (only under `libraryLayout='Tabs'`, which the demo does not use) |
| Explore | narrow tier is **deferred** — the Agent panel simply hides (`.xp-navpanel` rule) |

Capture baselines per decisions-ledger Q15: **1600×900** (⇒ `full`) and **iPad Pro 12.9 = 1366×1024** (⇒ `compact`). A third capture at ≤1240 is needed to lock the `narrow` DOM moves, and one below 1160 to catch the rotate overlay.

---

## 9. Recommended parity capture protocol

1. Fresh tab, `localStorage.removeItem('kl-filter-groups')`, fixed viewport, fixed `prefers-color-scheme`.
2. Stub `Math.random` → constant and `Date.now` → a fixed epoch **before** the bundle's boot script runs.
3. Wait for `kl-paper-row-ready` **and** `kl-run-mock-ready`, then two rAFs, then `data-vp-ready="1"` on every `.pc-root`.
4. Never capture the Runs `Running` default without pinning `_elS`; prefer driving `frameAt(t)` at fixed offsets.
5. Mask all four canvas rects.
6. Capture screens in a **fixed order** and reload between orders if you need cold state — there is no in-app reset other than the demo-state switcher's *Reset demo*, which only resets the three page-state props and returns to Build.
7. For any interaction that ends in a stagger/settle animation (import parse, sort, preset apply), assert the terminal state, not the animation.
