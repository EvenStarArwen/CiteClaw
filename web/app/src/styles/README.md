# CSS foundations — provenance and load order

Both files here are **verbatim copies**. Never reformat, minify, prettify, or
"tidy" them; a diff against the design workspace is the parity check.

| file | source | bytes |
| --- | --- | --- |
| `explorations-tokens.css` | `ui_design/explorations-tokens.css`, copied byte-for-byte | 58 296 |
| `ipad-demo-shell.css` | inner text of the `<helmet><style>` block in the iPad Demo's shell template (`script[type="__bundler/template"]`), extracted byte-for-byte | 1 897 |

sha256s live in `../design-data/PROVENANCE.json`.

## Load order is the contract

`main.tsx` imports these two, in this order, before any route module:

1. `explorations-tokens.css`
2. `ipad-demo-shell.css`

Screen CSS that rewrite agents bring over per-screen is imported from the
screen component, i.e. **after** both — so at equal specificity the screen's own
rules win. That is deliberate and must be preserved: where the two sources
disagree, the iPad Demo is the source of visual truth.

## Where they disagree (measured, 2026-08-13)

`explorations-tokens.css` is **not** loaded by the iPad Demo. The demo carries
its own `<style>` per screen; support.js's helmet manager hoists all seven into
one document head. 411 of the tokens file's 450 non-blank lines also appear in
some demo screen, so it is a good baseline — but it is not identical:

- demo: `html,body{ height:100%; margin:0; background:#ece5d8; }`
  tokens: `body{ margin:0; background:#ece5d8; }` (no `height:100%`)
- demo `.pc-root{…}` opens with `--pin:16px;--pin-sc:calc(var(--pin) - var(--kl-sbw, 8px));`
  — the tokens file has neither variable
- demo: `.pc-root button, .pc-root input, .pc-root textarea, .pc-root select{ font-family:inherit; }`
  — absent from the tokens file
- the `Fern & ink` / `Fern & ochre` colour variants carry a longer `--v-*` set
  in the tokens file than in the demo

Every demo screen's `<style>` contains a **complete** `.pc-root` token block, so
a screen that brings its own CSS over is self-sufficient and the divergences
above resolve in the demo's favour automatically. Do not "fix" either file to
match the other — logged in `web/wiring/escalations.md`.

## Fonts are not here

Font loading is in `index.html` and transcribed per screen in
`../design-fonts/manifest.ts`. Read the comment in `index.html` before touching
any `<link>`: the four Google Fonts stylesheets are not interchangeable.
