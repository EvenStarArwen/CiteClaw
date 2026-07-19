"""Assemble the CiteClaw design-system bundle for claude.ai/design.

Takes DOM captures harvested from the running app (see the capture flow in
the webapp notes: a CORS sidecar receives ``document.querySelector(sel)
.outerHTML`` posts, screenshots come from the browser tooling) and builds a
project folder of self-contained preview cards:

  README.html + docs/         the API-compatibility contract (the ONE rule)
  assets/app.css              the real stylesheet — the design's edit target
  foundations/                token swatches + type scale, parsed from app.css
  components/*.html           one @dsCard per captured component
  screens/*.html              full-page screenshots for composition context
  src/jsx/*.jsx               the actual component sources (markup reference)

Usage:
    python web/design_bundle.py <captures_dir> <screenshots_dir> <out_dir>

Every card reproduces the app's real root attributes (mono palette et al.)
so app.css applies exactly as in production. Scripts are stripped from all
captured HTML — cards are static by construction.
"""

from __future__ import annotations

import base64
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "web" / "live" / "static"
JSX_DIRS = [STATIC / "jsx", REPO / "web" / "public" / "static" / "jsx"]

# Captured from document.documentElement — the palette contract app.css keys on.
ROOT_ATTRS = (
    'lang="en" data-theme="light" data-palette="mono" data-mono-radius="pillowy" '
    'data-mono-primary="outline" data-mono-kind="tag-mono" data-mono-leaf="card" '
    'data-mono-seed="inline" data-mono-seedfill="orange-chip" data-mono-compline="gray" '
    'data-mono-runpill="plain" data-mono-trail="black" data-mono-acc="black" '
    'data-mono-rundot="green" data-mono-canvas="paper" data-mono-accent="cobalt"'
)

FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
         '&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">')

# component file -> (title, group, stage css)
_PANEL = "width: 340px; height: 720px; display: flex; flex-direction: column;"
COMPONENTS = {
    "build_topbar": ("Top bar — Build", "Chrome", "width: 1400px;"),
    "run_topbar": ("Top bar — running", "Chrome", "width: 1400px;"),
    "build_statusbar": ("Status bar", "Chrome", "width: 1400px;"),
    "run_statusbar": ("Status bar — running", "Chrome", "width: 1400px;"),
    "build_seeds_panel": ("Seeds panel", "Build", _PANEL),
    "build_seed_card": ("Seed paper card", "Build", "width: 300px;"),
    "build_config_panel": ("Configure-step panel (filter pipeline)", "Build", _PANEL),
    "build_pipeline_canvas": ("Pipeline canvas", "Build", "width: 980px; height: 760px; overflow: auto;"),
    "run_progress_panel": ("Pipeline progress panel — running", "Run", _PANEL),
    "run_done_progress_panel": ("Pipeline progress panel — finished", "Run", _PANEL),
    "run_step_detail": ("Step detail (branch roadmap)", "Run", _PANEL),
    "run_dashboard": ("Run dashboard — overview", "Run", "width: 950px;"),
    "run_done_dashboard": ("Run dashboard — finished", "Run", "width: 950px;"),
    "run_accepted_panel": ("Accepted papers panel", "Run", _PANEL),
    "run_net_toolbar": ("Graph toolbar", "Graph", "width: 420px;"),
    "run_net_legend": ("Graph legend", "Graph", "width: 620px;"),
    "run_graph_settings_pop": ("Graph settings popover", "Graph", "width: 300px;"),
    "explore_list_panel": ("Explore papers panel (+ filters open)", "Explore", _PANEL),
    "explore_detail_panel": ("Paper detail panel", "Explore", _PANEL),
    "explore_net_toolbar": ("Graph toolbar — Explore", "Graph", "width: 460px;"),
    "overlay_settings_modal": ("Settings modal", "Overlays", "width: 620px;"),
    "overlay_cap_dialog": ("Paper-cap dialog", "Overlays", "width: 560px;"),
}

SCREENS = {
    "build": ("Build workspace", 0),
    "run_midrun": ("Run — streaming mid-run", 1),
    "run_step_detail": ("Run — step detail page", 2),
    "run_finished": ("Run — finished", 3),
    "explore_selected": ("Explore — paper selected + filters", 5),
    "settings_modal": ("Settings modal in context", 6),
    "cap_dialog": ("Paper-cap dialog in context", 9),
}

_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.S | re.I)


def card(title: str, group: str, body: str, stage: str = "", extra_head: str = "") -> str:
    return f"""<!-- @dsCard group="{group}" -->
<!doctype html>
<html {ROOT_ATTRS}>
<head>
<meta charset="utf-8">
<title>{title}</title>
{FONTS}
<link rel="stylesheet" href="../assets/app.css">
{extra_head}
<style>
  body {{ background: var(--cc-bg); margin: 0; padding: 20px; }}
  .ds-stage {{ {stage} }}
</style>
</head>
<body>
<div class="ds-stage">
{body}
</div>
</body>
</html>
"""


def build_foundations(css: str, out: Path) -> None:
    root = re.search(r":root\s*{(.*?)}", css, re.S)
    tokens = re.findall(r"(--cc-[a-z0-9-]+)\s*:\s*([^;]+);", root.group(1) if root else "")
    colorish, other = [], []
    for name, val in tokens:
        v = val.strip()
        (colorish if re.match(r"^(#|rgb|hsl|oklch)", v) else other).append((name, v))
    sw = "".join(
        f'<div class="sw"><div class="chip" style="background:var({n})"></div>'
        f'<code>{n}</code><span>{v}</span></div>' for n, v in colorish)
    rows = "".join(f"<div class='tok'><code>{n}</code><span>{v}</span></div>" for n, v in other)
    body = f"""
<style>
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 10px; }}
 .sw {{ display: flex; align-items: center; gap: 9px; font: 12px Inter, sans-serif; }}
 .chip {{ width: 30px; height: 30px; border-radius: 7px; border: 1px solid var(--cc-rule); flex: none; }}
 .sw code {{ font: 11px "IBM Plex Mono", monospace; }}
 .sw span, .tok span {{ color: var(--cc-ink-3); font-size: 11px; }}
 .tok {{ display: flex; justify-content: space-between; gap: 12px; font: 11px "IBM Plex Mono", monospace;
         padding: 3px 0; border-bottom: 1px solid var(--cc-rule); }}
 h2 {{ font: 600 13px Inter, sans-serif; margin: 18px 0 10px; }}
</style>
<h2>Color tokens (light / :root)</h2><div class="grid">{sw}</div>
<h2>Other tokens</h2>{rows}
"""
    (out / "foundations").mkdir(parents=True, exist_ok=True)
    (out / "foundations" / "tokens.html").write_text(
        card("Design tokens", "Foundations", body, "max-width: 1100px;"))

    type_body = """
<style> .t { margin: 10px 0; } .t small { color: var(--cc-ink-3); font: 10px "IBM Plex Mono", monospace; } </style>
<div class="t"><small>14px 700 — brand</small><div style="font:700 14px Inter">CiteClaw</div></div>
<div class="t"><small>13px 600 — section titles</small><div style="font:600 13px Inter">Configure step</div></div>
<div class="t"><small>13px 400 — body</small><div style="font:400 13px Inter">Accepted papers stream in here as the run progresses.</div></div>
<div class="t"><small>12px 500 — labels</small><div style="font:500 12px Inter">Default screening model</div></div>
<div class="t"><small>11px mono — meta/ids</small><div style="font:500 11px 'IBM Plex Mono'">FWD-02 · 8 filters · S2 CACHE 73%</div></div>
"""
    (out / "foundations" / "type.html").write_text(
        card("Type scale", "Foundations", type_body, "max-width: 700px;"))


def main(captures: Path, shots: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    (out / "components").mkdir()
    (out / "screens").mkdir()
    (out / "src" / "jsx").mkdir(parents=True)
    (out / "docs").mkdir()

    css = (STATIC / "app.css").read_text()
    (out / "assets" / "app.css").write_text(css)
    build_foundations(css, out)

    for name, (title, group, stage) in COMPONENTS.items():
        f = captures / f"{name}.html"
        if not f.exists():
            print(f"  !! missing capture {name}")
            continue
        html = _SCRIPT_RE.sub("", f.read_text())
        (out / "components" / f"{name}.html").write_text(card(title, group, html, stage))

    jpgs = sorted(shots.glob("*.jpg"))
    for name, (title, idx) in SCREENS.items():
        if idx >= len(jpgs):
            print(f"  !! missing screenshot #{idx} for {name}")
            continue
        b64 = base64.b64encode(jpgs[idx].read_bytes()).decode()
        body = (f'<img alt="{title}" style="width: 100%; border: 1px solid var(--cc-rule); '
                f'border-radius: 10px;" src="data:image/jpeg;base64,{b64}">')
        (out / "screens" / f"{name}.html").write_text(
            card(title, "Screens", body, "max-width: 1300px;"))

    for d in JSX_DIRS:
        for f in sorted(d.glob("*.jsx")):
            (out / "src" / "jsx" / f.name).write_text(f.read_text())

    docs = REPO / "web" / "design_docs"
    shutil.copy2(docs / "README.html", out / "README.html")
    contract_md = (docs / "api-contract.md").read_text()
    (out / "docs" / "api-contract.md").write_text(contract_md)
    import html as _html
    (out / "docs" / "api-contract.html").write_text(card(
        "API compatibility contract", "Docs",
        f"<pre style='white-space: pre-wrap; font: 12px/1.6 \"IBM Plex Mono\", monospace;'>"
        f"{_html.escape(contract_md)}</pre>", "max-width: 820px;"))

    print(f"bundle at {out}")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    n = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"{n} files, {total/1e6:.1f} MB")
    big = [(p.relative_to(out), p.stat().st_size) for p in out.rglob("*")
           if p.is_file() and p.stat().st_size > 250_000]
    for p, s in big:
        print(f"  >250KB: {p} ({s/1000:.0f} KB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
