# `web/app/parity` — this app's side of the parity gate

`web/parity/` owns the harness, the frozen demo and the committed baseline. This
directory owns the two things that are properties of the **rewrite**: where its
screens live, and which of their states we claim to have matched.

Nothing here is imported by the app.

## Files

| file | what |
|---|---|
| `screens-app.json` | the same screen list as `web/parity/screens/design-demo.json`, repointed at this app's `/parity/<id>` routes. Reuses the harness's own `scripts/nav-build.json` unchanged. |
| `states-build.json` | the Build states that are reachable without a backend, numbered against `web/wiring/build-page-spec.md` §8. |
| `run-state-parity.mjs` | replays `states-build.json` against the demo *and* the rewrite and compares the settled screenshots. |
| `runs/` | scratch output. Gitignored. |

## The two gates

**1. Default screen, six viewports** — the harness's own commands:

```bash
cd web/app && pnpm dev                  # http://localhost:5273  (or pnpm preview -> :5274)

cd ../parity
node bin/parity.mjs capture --screens ../app/parity/screens-app.json \
     --target http://localhost:5273 --out runs/app-build --viewports all
node bin/parity.mjs diff --a baseline/design-demo --b runs/app-build --out runs/gate-build
```

Expected: 6/6 PASS, and the SHA256s in the capture log equal the ones in
`web/parity/README.md`'s baseline table — the rewrite is byte-identical, not
merely within threshold. Three of the six viewports are the demo's designed
"Rotate to landscape" gate; that is the demo's behaviour, not a shortcut.

**2. Reachable states** — this directory's runner:

```bash
cd web/app
node parity/run-state-parity.mjs http://localhost:5273/parity/build
```

Expected: 22/22, of which 19 are byte-identical. The three that are not
(`S06`/`S08`/`S10`, the top-bar popovers) report 0 pixels over the harness's
0.1 threshold with a sub-perceptual raw delta; see `H-BLD-06`.

## Before trusting either result

```bash
cd web/app   && node scripts/verify-transplants.mjs   # the copies are still the demo's bytes
cd ../parity && node bin/parity.mjs verify-reference  # the reference has not drifted
node bin/parity.mjs sanity                            # the harness is still deterministic
```

## Adding a screen

Add it to `src/parity/registry.ts`, add a `screens` entry here pointing at
`/parity/<id>`, and add its states to a `states-<id>.json`. Keep using the
harness's action scripts where they exist — the rewrite deliberately exposes the
demo's own `.tb-tab` / `[data-pg][data-on]` hooks so no per-target selector map
is needed.

## A caveat about clicks

`run-state-parity.mjs` clicks through the DOM rather than through Playwright's
locator API. Playwright's actionability check calls `scrollIntoViewIfNeeded`,
and `.sidebar` has a latent 14px horizontal overflow (`scrollWidth` 372 vs
`clientWidth` 358 — in the demo too, see `E-BLD-03`) that the sort menu pokes
through. Auto-scrolling it shifts the whole panel and produced a 18 000-pixel
"difference" that existed in neither target. If you add steps here, keep them
side-effect-symmetric or you will measure the automation.
