"""Pipeline preview: show the configured pipeline as an ASCII flow chart
before running, and prompt for user confirmation via an arrow-key
selectable menu (Claude Code style)."""
from __future__ import annotations

from typing import Any

from citeclaw.preview.flow_box import render
from citeclaw.preview.model import extract
from citeclaw.preview.prompt import confirm


def render_pipeline(pipeline: list[Any], *, width: int = 100) -> str:
    """Render a built pipeline (list of Step objects) as a boxed-flow ASCII diagram."""
    nodes = extract(pipeline)
    return render(nodes, width=width)


def confirm_proceed(prompt: str = "Proceed with this pipeline?") -> bool:
    """Arrow-key Yes/No confirmation. See ``citeclaw.preview.prompt.confirm``."""
    return confirm(prompt)


__all__ = ["render_pipeline", "confirm_proceed", "extract"]
