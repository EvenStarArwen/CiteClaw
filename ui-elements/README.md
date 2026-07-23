# CiteClaw UI elements store

Standalone, reusable UI elements for the CiteClaw web app. Each element is a
self-contained folder that owns **everything** about one piece of UI — its CSS
(with a `--<name>-*` parameter token surface), its React component, sample data,
and a `demo.html` showcase. The web app *loads* elements from here rather than
hosting their markup and styles inline, so an element can be redesigned in its
own small sub-codebase without touching the main app.

## Elements

| Element | Folder | Used by |
|---------|--------|---------|
| **PaperCard** | [`paper-card/`](paper-card/) | Build seeds · Run accepted / rejected · Explore |

_(more to come — this is the first.)_

## Conventions

Each element folder contains:

```
<element>/
  <element>.css     styling + a --<element>-* token surface (the parameters)
  <element>.jsx     the React component, exposed on window.<Component>
  demo-data.js      sample data (window.<ELEMENT>_DEMO)
  demo.html         standalone showcase: every state + a live parameter editor
  README.md         anatomy, component API, token table, how the app loads it
```

Rules that keep elements portable:

- **One parameter surface.** All tunables are `--<element>-*` custom properties
  at the top of the CSS. Geometry / type are literals the element owns; theme
  colours defer to the app's `--cc-*` tokens with a literal fallback
  (`var(--cc-ink-1, #1e2735)`) so the element follows the app theme when
  embedded and still renders standalone in its demo.
- **No hard dependencies.** Components take glyphs / controls as slot props
  rather than importing the app's icon set, so they compile in both the app's
  Babel block and the standalone demo.
- **Same runtime as the app.** Demos load React / ReactDOM / Babel from the same
  unpkg versions the app uses.

## How the app loads elements

`web/live/backend/server.py` mounts this directory at `/elements`
(`StaticFiles`), links each element's CSS in the page `<head>`, and concatenates
each element's `.jsx` into the single Babel block **before** the components that
use it (function declarations are global, so the component is defined by the time
a list renders it). To view any element's showcase while the app runs:

```
http://localhost:<port>/elements/<element>/demo.html
```

## Adding an element

1. `mkdir ui-elements/<name>/` and add the five files above.
2. Register it in `server.py`: add its CSS to the `<head>` links and its `.jsx`
   to the concatenation list (`ELEMENT_JSX`).
3. Replace the app's inline markup/CSS with the component + its classes, and
   delete the now-duplicated styles from `web/live/static/app.css`.
