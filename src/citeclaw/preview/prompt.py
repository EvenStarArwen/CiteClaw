"""Arrow-key Y/N selector for the pipeline-preview confirmation.

Renders a small inline menu like::

    Proceed with this pipeline?
    ❯ Yes — proceed
      No  — abort

and lets the user move the ``❯`` indicator with the up/down arrow keys
(or k/j vim-style), Enter to confirm. ``y`` / ``n`` are typed-letter
shortcuts that pick the matching option directly.

Implementation is stdlib-only (termios + tty + ANSI escapes); needs a
real tty for arrow-key reads, falls back to plain ``input("[Y/n] ")``
when stdin is piped/redirected. Non-Unix platforms (Windows) also fall
back to the text prompt — adding ``msvcrt.getch`` for a real arrow-key
mode there is a future task.
"""
from __future__ import annotations

import sys
from typing import Optional

try:
    import termios
    import tty
    _HAVE_TERMIOS = True
except ImportError:  # pragma: no cover - Windows fallback path
    _HAVE_TERMIOS = False


# ANSI escape sequences.
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_CLEAR_LINE = "\x1b[2K"
_CR = "\r"
# Style: bold cyan for the selected row, dim for the rest.
_SEL = "\x1b[1;36m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def _read_key(stream) -> str:
    """Read one keypress (or arrow escape sequence) from ``stream``.

    Returns a friendly string label: ``UP``, ``DOWN``, ``ENTER``, ``Y``,
    ``N``, ``J``, ``K``, ``ESC``, ``INT`` (Ctrl+C), or the raw character
    for anything else. Arrow keys arrive as the ESC ``[A`` / ``[B``
    three-byte sequence on Unix terminals.
    """
    ch = stream.read(1)
    if ch == "\x1b":
        try:
            seq = stream.read(2)
        except Exception:
            return "ESC"
        if seq == "[A":
            return "UP"
        if seq == "[B":
            return "DOWN"
        if seq == "[C":
            return "RIGHT"
        if seq == "[D":
            return "LEFT"
        return "ESC"
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch == "\x03":
        return "INT"
    if ch in ("y", "Y"):
        return "Y"
    if ch in ("n", "N"):
        return "N"
    if ch in ("j", "J"):
        return "J"
    if ch in ("k", "K"):
        return "K"
    return ch


def _render_menu(options: list[str], cur: int, file) -> None:
    """Write the menu rows to ``file`` (newline after each)."""
    for i, opt in enumerate(options):
        if i == cur:
            file.write(f"  {_SEL}❯ {opt}{_RESET}\n")
        else:
            file.write(f"  {_DIM}  {opt}{_RESET}\n")
    file.flush()


def select(
    options: list[str],
    *,
    default_idx: int = 0,
    prompt: str = "",
) -> Optional[int]:
    """Arrow-key selectable menu. Returns the chosen index, or ``None`` on cancel.

    Returns ``default_idx`` immediately when stdin isn't a real tty, so
    non-interactive runs (CI / piped stdin) don't hang.
    """
    n = len(options)
    if n == 0:
        return None

    if not (sys.stdin.isatty() and _HAVE_TERMIOS):
        return default_idx

    if prompt:
        sys.stdout.write(prompt + "\n")

    _render_menu(options, default_idx, sys.stdout)
    sys.stdout.write(_HIDE_CURSOR)
    sys.stdout.flush()

    cur = default_idx
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    cancelled = False
    try:
        tty.setcbreak(fd)
        while True:
            key = _read_key(sys.stdin)
            if key in ("UP", "K"):
                cur = (cur - 1) % n
            elif key in ("DOWN", "J"):
                cur = (cur + 1) % n
            elif key == "ENTER":
                break
            elif key == "Y" and n >= 1:
                cur = 0
                break
            elif key == "N" and n >= 2:
                cur = 1
                break
            elif key in ("INT", "ESC"):
                cancelled = True
                break
            else:
                continue

            # Move cursor up ``n`` lines and redraw the menu in place.
            sys.stdout.write(f"\x1b[{n}A" + _CR)
            for i, opt in enumerate(options):
                sys.stdout.write(_CLEAR_LINE + _CR)
                if i == cur:
                    sys.stdout.write(f"  {_SEL}❯ {opt}{_RESET}\n")
                else:
                    sys.stdout.write(f"  {_DIM}  {opt}{_RESET}\n")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    if cancelled:
        return None
    return cur


def confirm(prompt: str = "Proceed?") -> bool:
    """Yes/No arrow-key confirmation. Returns ``True`` on Yes, ``False`` on No / cancel.

    Non-interactive stdin auto-confirms (returns True) so piped runs
    don't hang; cancel with Ctrl+C / ESC to return False.
    """
    idx = select(
        ["Yes — proceed", "No  — abort"],
        default_idx=0,
        prompt=prompt,
    )
    return idx == 0
