# PaperCard

The one card used by every paper list in the CiteClaw web UI — **Build seeds**,
**Run accepted / rejected**, and **Explore**. It is a self-contained UI element:
everything visual lives in this folder, so the card can be redesigned here in
isolation without touching the 5,000-line app stylesheet.

```
ui-elements/paper-card/
  paper-card.css     all styling + the --pcard-* parameter tokens
  paper-card.jsx     the <PaperCard> React component (no icon dependency)
  demo-data.js       sample paper records
  demo.html          live showcase — every state + a parameter editor
  README.md          this file
```

## View the demo

The demo is served by the running web app (it uses the *same* files the app
loads, so what you see is what ships):

```
http://localhost:<port>/elements/paper-card/demo.html
```

Or serve this folder standalone:

```
cd ui-elements/paper-card && python3 -m http.server 8999
# → http://localhost:8999/demo.html
```

(Open it over http, not `file://` — Babel fetches `paper-card.jsx` at runtime.)

## Anatomy

A flex row of an optional **lead** column and a **body**. The lead (star / × /
reject mark) sits in its own column, so the title and every meta line share one
left rail. The body stacks fixed rows:

```
┌──────────────────────────────────────┐
│ ☆   Title …………………………………  ⟨trail⟩ │   title (2 lines, reserved)
│     Venue ……………………………………………      │   venue   (own row)
│     Authors ………………………………………………   │   authors (own row, clips)
│     Year · N cites ……………………………      │   stats   (own row)
│     ⟨footer⟩                          │   e.g. reject reason
└──────────────────────────────────────┘
```

Every row reserves its height, so a missing field renders `—` rather than
shrinking the card. All cards in a list are therefore the same height.

## Component API

```jsx
<PaperCard
  title="…" venue="…" authors="…" year={2023} cites={283}
  lead={<button className="pcard-starbtn">…</button>}   // optional slot
  trail={<span className="pcard-depth">d1</span>}        // optional slot
  footer={<div className="pcard-reason">…</div>}         // optional slot
  titlePrefix={<span className="pcard-seed-dot" />}      // optional
  rows={[authorsNode, statsNode]}   // optional — replace the 2 lower rows
  selected fresh offnet             // state flags
  onClick={fn} tooltip="…" className="…"
/>
```

| Prop | Type | Notes |
|------|------|-------|
| `title` | string | clamped to `--pcard-title-lines` (2) lines |
| `venue` | string | own row; `—` when empty |
| `authors` | string | own row; `—` when empty |
| `year`, `cites` | number | combined into the `year · N cites` row |
| `rows` | node[] | overrides the authors + stats rows (venue stays). Use for author-mode, an off-network glyph, etc. |
| `titlePrefix` | node | rendered before the title (e.g. the seed dot) |
| `lead` / `trail` / `footer` | node | slot content — the caller owns behaviour + glyph |
| `selected` | bool | black-invert selected state |
| `fresh` | bool | one-shot flash (newly-accepted row) |
| `offnet` | bool | dimmed, off-network (Explore) |

The card has **no icon dependency** — the caller passes rendered glyphs into the
slots. In the app those come from the shared `Icon` component; in the demo from
inline SVGs.

## Parameters (`--pcard-*` tokens)

Edit these at the top of `paper-card.css` to restyle every card at once. Geometry
and type are literals the card owns; colours defer to the app's `--cc-*` theme
tokens with a standalone fallback.

| Token | Default | Controls |
|-------|---------|----------|
| `--pcard-pad-y` / `--pcard-pad-x` | 12px | internal padding |
| `--pcard-radius` | 16px | corner roundness |
| `--pcard-lead-gap` | 9px | lead → body gap |
| `--pcard-row-gap` | 4px | gap between text rows |
| `--pcard-gap` | 4px | gap between cards (via `.pcard-list`) |
| `--pcard-title-lines` | 2 | lines the title reserves + clips to |
| `--pcard-title-size` / `-weight` / `-lh` | 12px / 600 / 1.35 | title type |
| `--pcard-meta-size` / `-lh` | 10.5px / 1.4 | meta type |
| `--pcard-ink-title` / `-venue` / `-meta` | `--cc-ink-1/2/3` | text colour ramp |
| `--pcard-hover-bg` | `#eceae5` | hover surface |
| `--pcard-sel-bg` / `-ink` / `-venue` | `#0a0a0a` / white | selected surface + text |
| `--pcard-fresh-bg` | green wash | flash colour |

A `:root[data-theme="dark"]` block re-points the state surfaces so the card
follows the app's dark theme.

## Decoration kit

Styled here so *all* card looks live in this folder; the lists only supply the
glyph + handler:

`pcard-starbtn` · `pcard-star` · `pcard-removebtn` · `pcard-reject-mark` ·
`pcard-depth` · `pcard-cat` · `pcard-reason` · `pcard-seed-dot` · `pcard-offnet-ic`

## How the web app loads it

`web/live/backend/server.py` mounts this store at `/elements` and, when it
assembles the page, links `paper-card.css` and concatenates `paper-card.jsx`
into the Babel block **before** the list components that use `<PaperCard>`. The
app no longer defines any card markup or `.pcard*` CSS of its own — it all comes
from here.
