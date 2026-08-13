# CSS foundations — provenance and load order

Everything here except `demo-screens.css` and `demo-style-hover.css` is a
**verbatim copy** of a design asset. Never reformat, minify, prettify or "tidy"
one; a byte diff against `web/design-reference` is the parity check, and
`scripts/verify-transplants.mjs` runs it.

| file | source | bytes | loaded? |
| --- | --- | --- | --- |
| `ipad-demo-shell.css` | inner text of the `<helmet><style>` block in the iPad Demo's shell template (`script[type="__bundler/template"]`), extracted byte-for-byte | 1 897 | yes, from `main.tsx` |
| `demo-screen-css/0n-*.css` | inner text of each screen's `<helmet><style>`, from `web/design-reference/embedded-sources/*.dc.html`, byte-for-byte | 16 264 / 25 572 / 109 080 / 93 861 / 31 652 / 15 799 / 10 775 | yes, via `demo-screens.css` |
| `demo-screens.css` | not a copy — the seven above, `@import`ed in the demo's own head order | — | yes, from each screen |
| `demo-style-hover.css` | GENERATED from the templates' `style-hover` directives | — | yes, from each screen |
| `explorations-tokens.css` | `ui_design/explorations-tokens.css`, copied byte-for-byte | 58 296 | **no — see below** |

sha256s live in `../design-data/PROVENANCE.json`. `node scripts/verify-transplants.mjs`
re-checks every verbatim copy against `web/design-reference`.

## Load order is the contract

1. `main.tsx` imports `ipad-demo-shell.css` — the demo's shell block, which sits
   before every screen block in the demo's head.
2. Each screen imports `demo-screens.css`, which brings in all seven screens'
   blocks **in the demo's mount order**. That order is load-bearing and is not a
   style preference: later blocks override earlier ones for every screen. Read
   the header of `demo-screens.css` before touching it — the measurements are
   there, and the consequence is filed as `E-BLD-01`.

## `explorations-tokens.css` is not loaded

It is not loaded by the iPad Demo either — no screen references it — so any rule
in it that the demo's own stylesheets do not also carry becomes an invented
override. Measured on the Build pilot: its `.cf-num` and
`.cfg-pipe[data-number="Column"] .cf-idx-col` rules declare
`font-family: ui-monospace, monospace`, the demo's copies of those two rules do
not declare font-family at all, so nothing overrode it and five row numbers
rendered in the wrong typeface. `main.tsx` therefore no longer imports it.

The file stays here because it may yet be regenerated *from* the demo and become
the one token sheet the project needs (`component-duplication.md` §8). Until
that is decided (`E-BLD-02`, superseding the open question in `E-SCAF-01`), no
screen may import it.

## Where the tokens file disagrees with the demo (measured, 2026-08-13)

411 of the tokens file's 450 non-blank lines also appear in some demo screen, so
it is a good starting point for the eventual extracted sheet — but it is not
identical:

- demo: `html,body{ height:100%; margin:0; background:#ece5d8; }`
  tokens: `body{ margin:0; background:#ece5d8; }` (no `height:100%`)
- demo `.pc-root{…}` opens with `--pin:16px;--pin-sc:calc(var(--pin) - var(--kl-sbw, 8px));`
  — the tokens file has neither variable
- demo: `.pc-root button, .pc-root input, .pc-root textarea, .pc-root select{ font-family:inherit; }`
  — absent from the tokens file
- the `Fern & ink` / `Fern & ochre` colour variants carry a longer `--v-*` set
  in the tokens file than in the demo

…plus the `font-family` divergence above, which is the one that actually bit.
Do not "fix" either file to match the other — logged in
`web/wiring/escalations.md` (`E-SCAF-01`, `E-BLD-02`).

## Fonts are not here

Font loading is in `index.html`, self-hosted from `public/design-fonts/`, and
transcribed per screen in `../design-fonts/manifest.ts`. Read the comment in
`index.html` before touching any `<link>`: the four stylesheets are not
interchangeable and their order decides which face serves which weight.
