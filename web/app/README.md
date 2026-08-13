# web/app — OmniKnowledge frontend

Vite + React 18 + TypeScript. This is the rewrite of the **KnowledgeLab iPad
Demo** wired to the CiteClaw backend. It is a new tree: nothing here touches
`web/live`, `web/backend`, `web/public`, `web/frontend`, or `src/`.

## Run it

```bash
cd web/app
pnpm install
pnpm dev        # http://localhost:5273
pnpm build      # tsc -b && vite build  ->  dist/
pnpm preview    # http://localhost:5274  (serves dist/)
pnpm typecheck  # tsc -b, no emit
```

### Ports are fixed

| script | port | why |
| --- | --- | --- |
| `pnpm dev` | **5273** | the parity harness addresses it by number |
| `pnpm preview` | **5274** | ditto, for built output |

Both use `strictPort`, so a busy port fails loudly instead of silently moving
to 5274/5275 and breaking the harness. 5173 is deliberately avoided — the old
`web/frontend` already owns the Vite default.

## Routes

| route | what |
| --- | --- |
| `/` | the app. Empty shell for now: the demo's root box and its rotate invite, nothing else. |
| `/parity` | index of screens registered in `src/parity/registry.ts`. |
| `/parity/<id>` | mounts exactly one rewritten screen, no chrome, for diffing against the demo. `?viewport=desktop\|ipadPro129\|fluid`. |

## Layout

```
index.html                    demo <head>: viewport metas + the 4 font stylesheets
src/
  main.tsx                    entry. The two CSS imports at the top are ORDER-SENSITIVE.
  App.tsx                     router
  components/DemoViewport.tsx the demo shell's root box + rotate invite, transcribed
  routes/                     HomeRoute, ParityRoute
  parity/registry.ts          where rewrite agents register a finished screen
  styles/                     VERBATIM CSS foundations — see styles/README.md
  design-fonts/manifest.ts    per-screen transcript of the demo's font <link>s
  design-data/                VERBATIM demo data modules + PROVENANCE.json
  data/                       the DataSource boundary
```

## The two rules that are easy to break

**1. Verbatim means verbatim.** Everything in `src/styles/*.css` and
`src/design-data/*.js` is a byte-for-byte copy out of the read-only design
workspace. sha256s are recorded in `src/design-data/PROVENANCE.json`. Do not
reformat, lint, minify, or "clean up" these files — the diff against the source
is the parity check. Type them from the outside with the sibling `.d.ts` files
instead.

**2. Screens talk to `DataSource`, never to `design-data`.** `src/data/types.ts`
is the only seam between screens and where bytes come from.
`designDataSource.ts` serves the demo's own data so that visual parity is a
question about markup and CSS alone; the CiteClaw API adapter will be a sibling
file implementing the same interface. A screen that imports from
`../design-data` directly has to be rewritten when the API lands — that is the
whole cost this boundary exists to avoid.

## Font loading is not interchangeable

`index.html` carries four Google Fonts stylesheets, not one. The demo mounts all
seven screens into one document and support.js hoists every screen's `<link>`
into the shared `<head>`, so the demo's live font environment is that union.
Two of the four declare Hanken Grotesk as a variable `400..700` face and two
declare fixed `400;500;600` faces; the screens do request `font-weight:700`
(Runs 38×, Explore 42×, Build 13×). Drop a link and rendered weights change.
Per-screen transcript: `src/design-fonts/manifest.ts`.

Fonts are still loaded from `fonts.googleapis.com`, exactly as the demo does.
Self-hosting is tracked in `web/wiring/hardening-todo.md`.
