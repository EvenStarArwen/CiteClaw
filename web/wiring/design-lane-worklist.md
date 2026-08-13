# Design-lane worklist

Proposals that need a design agent to draw them **and** product sign-off before
anything is built. Nothing here has been implemented. Append entries, never
rewrite them; keep them dated.

Status vocabulary: `proposed` → `with design` → `signed off` → `built` / `dropped`.

---

## DL-01 — file-type icons for import

- **Date:** 2026-08-13
- **Raised by:** import 系统测试准备 lane
- **Status:** `proposed`
- **Surfaces:** Home wizard step 2 (`.hw-idrop` / `.hw-iparse`), Build seed
  sidebar (`.sbi-*`), Runs Add-papers panel (`.ra-*`) — all three render the
  same file row from `import-resolver.js::fileRowHtml`.

**What the demo does today.** There are no per-format icons anywhere in the
import flow. Two visuals carry all format information:

1. **The per-file badge.** `fileRowHtml` draws a 34 px chip
   (`--icon-bg` fill, `--border` hairline, 6 px radius, 9.5 px / 700 weight,
   `.04em` tracking) containing an uppercase **text** label from
   `var EXTC = { bib:'BIB', ris:'RIS', csv:'CSV', txt:'TXT', pdf:'PDF', zip:'ZIP' }`,
   falling back to `f.ext.toUpperCase()` and finally to `'?'`. On a per-file
   error the label's ink switches to the error tint; the chip itself does not.
2. **The dropzone glyph.** One generic upload-arrow SVG (a 46 px `--icon-bg`
   rounded square in Home, 42 px in Runs), identical for every format.

So format identity is **partial**: it is present, but as typography rather
than iconography, and the fallback for an unknown extension (`'?'`) is the same
neutral chip as a supported one.

**Why it may be worth a design pass.**

- The chip is the only place the user learns *what the system thought the file
  was*. When extension-vs-content disagree (`adversarial/wrong-ext-*.{bib,ris,txt}`)
  that distinction becomes load-bearing, and a text label reads as a fact
  rather than as an inference.
- `'?'` for an unknown extension sits in the same visual weight as `BIB`,
  immediately above an error message. A distinct unknown/error mark would
  separate "we don't know what this is" from "we know, and it's a BibTeX file".
- The format list is about to grow (`.json` is under discussion — see
  `escalations.md` E-IMP-01; `.enw` / `.xml` / `.rdf` get reject-with-guidance
  messages). More labels means more pressure on a 34 px chip.

**Explicitly NOT a recommendation to add icons.** The text chip is coherent
with the rest of the system (`§ Iconography & motion`, `§ Type minimums`), and
`Newsreader`-era restraint may well be the right answer. The ask is only that
design decides deliberately rather than by omission.

**Questions for the design agent / product owner:**

1. Should the file row keep the text chip, gain a glyph, or gain a glyph **and**
   the text?
2. Does the unknown / unsupported case deserve its own mark, distinct from `?`?
3. Does the dropzone deserve per-format affordances at all, or does one upload
   glyph plus the body copy carry it?
4. If glyphs are added: `bib ris csv txt pdf zip` today, plus `json enw xml rdf`
   later — is that a set worth drawing, or a set worth refusing?

**Blocked on:** design proposal + product sign-off. No implementation until then.
