# `web/parity` — pixel parity harness

The quality gate for the OmniKnowledge UI rewrite. It renders the **frozen design
demo** and the **rebuilt app** under identical, reproducible conditions and tells
you, per screen and per viewport, whether they are the same picture.

The load-bearing property is **determinism**: capturing the original demo twice
must produce byte-identical PNGs. If that ever stops being true, every diff this
harness reports becomes noise. There is a command that asserts it (`sanity`), and
it currently passes 6/6 byte-identical.

```
web/
  design-reference/   frozen verbatim copy of the design hand-off  (never edit)
  parity/             this harness
  app/                the rewrite            (captured as target B)
  wiring/             analysis docs, escalations, missing-states register
```

---

## Quick start

```bash
cd web/parity
pnpm install
pnpm exec playwright install chromium     # one-time, ~270 MB

node bin/parity.mjs sanity                # determinism acceptance test — must PASS
node bin/parity.mjs serve                 # browse the demo at http://127.0.0.1:4173
```

Requires Node (developed on v25.9) and pnpm (falls back to npm fine).

---

## Commands

### `serve`
Static file server for `web/design-reference`, so the demo is exercised over HTTP
exactly as a deployment would be.

```bash
node bin/parity.mjs serve [--port 4173] [--root <dir>]
```

### `capture`
Render every screen in a screens config at a set of viewports.

```bash
# the demo (auto-serves web/design-reference)
node bin/parity.mjs capture --screens screens/design-demo.json \
     --out runs/demo --viewports all --filmstrip

# the rewrite (point --target at its dev server)
node bin/parity.mjs capture --screens screens/design-demo.json \
     --target http://localhost:5173 --out runs/app --viewports all
```

| flag | meaning |
|---|---|
| `--target` | `design` (default, serves `web/design-reference`) or any base URL |
| `--screens` | screens config, default `screens/design-demo.json` |
| `--out` | output run directory |
| `--viewports` | `baseline` \| `sweep` \| `all` \| comma-separated names |
| `--filmstrip` | also write a PNG after every action-script step |
| `--headed` | watch it happen |

Output: `<out>/<screen>/<viewport>.png` plus `<out>/run.json` (per-shot SHA256,
pixel-stability result, determinism settings, CDN-mirror fingerprint, console
errors, external requests). Exit code 1 if any capture never reached pixel
stability.

### `diff`
Compare two capture runs.

```bash
node bin/parity.mjs diff --a runs/demo --b runs/app --out runs/diff-demo-vs-app
```

Produces, per screen: a **diff heatmap**, a **side-by-side triptych**
(A | heatmap | B) and `index.html`, a self-contained report listing pass/fail
with both the anti-aliasing-tolerant pixel count and the **raw** count. Exit
code 1 on any fail/missing/extra.

| flag | default | meaning |
|---|---|---|
| `--threshold` | `0.1` | pixelmatch per-pixel colour tolerance (absorbs AA/subpixel drift) |
| `--include-aa` | off | count anti-aliased pixels as differences |
| `--max-diff-ratio` | `0` | pass gate: fraction of the frame allowed to differ |

`--max-diff-ratio 0` means "byte-for-byte". Relax it deliberately and record why
— a silently-relaxed gate is the usual way a parity harness stops working.

### `script`
Run one JSON action script against one target and dump a screenshot filmstrip.
The debugging tool for authoring scripts.

```bash
node bin/parity.mjs script --script scripts/nav-build.json \
     --out runs/strip --viewport desktop-1600x900 [--headed]
```

### `sanity`  ← the acceptance test
Captures the **original demo twice**, in two independently launched browsers,
and diffs the runs. Anything other than 100% byte-identical is a harness bug.

```bash
node bin/parity.mjs sanity [--viewports all]
```

### `verify-reference`
Re-hashes `web/design-reference` against its `manifest.json`. Run it before
trusting a parity result; the reference is supposed to be immutable.

```bash
node bin/parity.mjs verify-reference
```

---

## Viewports

Defined once in `src/config.mjs`. Responsive behaviour is an explicit product
requirement, so the sweep is not optional.

| set | name | size |
|---|---|---|
| baseline | `desktop-1600x900` | 1600 × 900 |
| baseline | `ipad-pro-12.9-landscape-1366x1024` | 1366 × 1024 |
| sweep | `ipad-portrait-1024x1366` | 1024 × 1366 |
| sweep | `ipad-11-1194x834` | 1194 × 834 |
| sweep | `w900-900x1200` | 900 × 1200 |
| sweep | `w768-768x1024` | 768 × 1024 |

**The three portrait/narrow sweep viewports do not show the app.** The demo
replaces the whole UI with its designed *"Rotate to landscape"* gate (`.rot`)
below landscape proportions. That gate is itself part of the baseline — the
rewrite has to reproduce it. See `web/wiring/missing-states.md`.

---

## Action scripts

A screen is reached by replaying a JSON script, so the *same* script drives the
demo and the rewrite.

```json
{ "action": "click", "selector": ".tb-tab", "text": "Build", "note": "top nav" }
```

| key | meaning |
|---|---|
| `action` | `goto` `click` `hover` `type` `fill` `press` `scroll` `waitFor` `waitForText` `wait` `settle` `moveMouse` `blur` `shot` |
| `selector` | CSS selector |
| `text` | narrows `selector` to the element with this text (exact match first, then substring) |
| `nth` | pick the nth match instead of the first |
| `shot: false` | do not screenshot after this step (every step screenshots by default) |
| `skipIfVisible` | skip the step when this selector is visible — still screenshots |
| `note` | free text, kept in the filmstrip metadata |

`wait` advances the **virtual** clock, so a wait is exactly as long on a fast
laptop as on a loaded CI box.

`skipIfVisible` is how one script covers a screen that legitimately renders
differently at some viewports: `scripts/nav-build.json` marks the app-chrome
steps `skipIfVisible: ".rot"` so the portrait viewports capture the rotate gate
instead of timing out on an unclickable nav.

Shipped script: **`scripts/nav-build.json`** — the recorded navigation to the
Build/pipeline screen. It clicks the Build tab like a user rather than relying on
the demo's default page prop, so it remains meaningful against the rewrite. It
then parks the cursor and drops focus, because a baseline must not have a hover
or focus ring baked into it.

---

## How determinism is achieved

The demo animates, streams replayed pipeline text, and seeds some layout from
`Math.random()`. Six layers, all in `src/determinism.mjs`, and **all applied
identically to every target** — the harness never gives one side a setting the
other does not get:

1. **Seeded PRNG.** An init script replaces `Math.random`, `crypto.getRandomValues`
   and `crypto.randomUUID` with a mulberry32 stream before any page script runs.
2. **Frozen clock.** Playwright's clock API installs fake `Date`, `performance`,
   `setTimeout/Interval` and `requestAnimationFrame`, then **pauses immediately at
   a fixed epoch**. Pausing at install is essential — a merely *installed* fake
   clock still ticks with wall time, so a slow machine would hand the page more
   page-time than a fast one.
3. **Fixed virtual warm-up.** After load, exactly `12 000` virtual milliseconds
   are burned in 250 ms steps, each interleaved with a few real milliseconds so
   genuinely async work (bundle unpack, font decode) can progress. Every target
   receives the same page-time budget and lands on the same fake timestamp.
4. **Font readiness.** `document.fonts.ready` before any pixel is read.
5. **Pixel stability gate.** Screenshots are taken repeatedly until two
   consecutive frames are byte-identical. A screen that never stabilises is
   reported as `UNSTABLE` and fails the run rather than being baked into a
   baseline.
6. **`animations: 'disabled'` at screenshot time** plus `caret: 'hide'` and
   `scale: 'css'`. This is a capture-time lever, not an edit to the product's CSS.
   Note `reducedMotion` is deliberately **not** forced — that would change the
   rendered appearance, which this project is not allowed to do.

Chromium is also launched with a fixed sRGB colour profile, LCD-text and
subpixel-positioning disabled, and a forced device scale factor of 1.

### The CDN mirror

The demo is *almost* self-contained — its sibling `.js`/`.dc.html` files, the
Google Fonts CSS (with woff2 payloads as `data:` URIs) and React 18.3.1 UMD are
all inlined into the bundle. But **six scripts are not**, and are still fetched
live from public CDNs by `Paper Card.dc.html`, `Runs.dc.html` and
`Explore.dc.html`:

- `smooth-scrollbar@8.8.4` + its `overscroll` plugin (cdnjs)
- `marked@12.0.2`, `katex@0.16.11`, `turndown@7.2.0`, `turndown-plugin-gfm@1.0.2` (jsDelivr)

Left alone this breaks determinism twice over: every capture depends on live
network latency, and the two smooth-scrollbar files **race** — the plugin
frequently wins and throws `Cannot read properties of undefined (reading
'ScrollbarPlugin')`, which the demo swallows into a hidden toast.

`src/cdn-mirror.mjs` therefore intercepts those URLs and fulfils them from
version-pinned npm copies, holding each dependent script until its dependency's
global exists. Any *other* external request is aborted and recorded in
`run.json` under `unmirroredExternalRequests` — a capture must never silently
depend on the internet. With the mirror in place the demo loads with **zero
console errors**.

This mirrors the demo's runtime behaviour; it does not change its appearance.

---

## Baseline gallery

`baseline/design-demo/` is the committed first baseline: the demo's **Build /
pipeline screen** at both baseline viewports plus the full responsive sweep,
with a per-step filmstrip under `<viewport>.steps/`.

| viewport | SHA256 (first 12) | bytes |
|---|---|---|
| `desktop-1600x900` | `d280afe268e5` | 196 721 |
| `ipad-pro-12.9-landscape-1366x1024` | `85ac559e5038` | 209 947 |
| `ipad-portrait-1024x1366` | `3835a2a21061` | 23 114 |
| `ipad-11-1194x834` | `f6f44d69089e` | 103 842 |
| `w900-900x1200` | `27c6f468600d` | 21 571 |
| `w768-768x1024` | `5dec48c908f8` | 20 255 |

Regenerate with:

```bash
node bin/parity.mjs capture --screens screens/design-demo.json \
     --out baseline/design-demo --viewports all --filmstrip --label design-demo-baseline
```

Gate the rewrite against it with:

```bash
node bin/parity.mjs capture --screens screens/design-demo.json \
     --target http://localhost:5173 --out runs/app
node bin/parity.mjs diff --a baseline/design-demo --b runs/app --out runs/gate
```

---

## Extending it

**Add a screen**: add an entry to `screens/design-demo.json` and, if it needs
interaction, an action script under `scripts/`.

**Add a viewport**: add it to `SWEEP_VIEWPORTS` in `src/config.mjs`; it is picked
up by name everywhere.

**Point at the rewrite**: `--target http://localhost:5173`. If `web/app` uses
different class names, the action scripts need a per-target selector map — the
scripts intentionally lean on the demo's own `.tb-tab` / `[data-pg]` hooks, and
`scripts/nav-build.json` records that dependency in its `notes`.

## Layout

```
bin/parity.mjs            CLI
src/config.mjs            viewports, determinism constants, diff defaults
src/server.mjs            static server for web/design-reference
src/determinism.mjs       PRNG seeding, clock control, settle + stability gate
src/cdn-mirror.mjs        offline, order-deterministic mirror of the 6 CDN scripts
src/actions.mjs           JSON action-script runner
src/capture.mjs           capture command
src/diff.mjs              pixelmatch diff, heatmaps, triptychs
src/report.mjs            self-contained HTML report
src/reference-manifest.mjs  build/verify the design-reference SHA256 manifest
screens/design-demo.json  screen list
scripts/nav-build.json    recorded navigation to the Build screen
baseline/design-demo/     committed baseline gallery
runs/                     scratch output (gitignored)
```
