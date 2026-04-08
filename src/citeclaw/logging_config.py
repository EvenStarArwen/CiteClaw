"""Structured logging setup — console + file output."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger("citeclaw")
    root.setLevel(logging.DEBUG)  # allow all levels through; handlers filter independently
    root.propagate = False

    # Console: short format (time only, no logger name)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # File: full format (date+time, level, logger name)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_dir / "citeclaw.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_fmt)
        root.addHandler(fh)
