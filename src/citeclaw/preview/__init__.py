"""Pipeline preview: show the configured pipeline as an ASCII flow chart
before running, and prompt for user confirmation."""
from __future__ import annotations

import sys
from typing import Any

from citeclaw.preview.flow_box import render
from citeclaw.preview.model import extract


def render_pipeline(pipeline: list[Any], *, width: int = 100) -> str:
    """Render a built pipeline (list of Step objects) as a boxed-flow ASCII diagram."""
    nodes = extract(pipeline)
    return render(nodes, width=width)


def confirm_proceed(prompt: str = "Proceed with this pipeline? [Y/n] ") -> bool:
    """Ask the user to confirm. Treat enter/y/yes as proceed; anything else as abort.

    Returns ``True`` if the user wants to proceed, ``False`` otherwise.
    Non-interactive stdin (e.g. CI piping) returns ``True`` so unattended
    runs aren't blocked — pass ``--no-preview`` to skip the diagram
    entirely in that case.
    """
    if not sys.stdin.isatty():
        return True
    try:
        reply = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return reply in ("", "y", "yes")


__all__ = ["render_pipeline", "confirm_proceed", "extract"]
