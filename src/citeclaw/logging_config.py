"""Idempotent ``"citeclaw"`` logger setup — console + optional file output.

Every module under :mod:`citeclaw` writes to a ``"citeclaw.<submodule>"``
sub-logger. :func:`setup_logging` configures the shared parent logger
once per process so handlers don't get duplicated when the CLI re-enters
through ``--continue-from`` or when a long-lived web server reuses a
worker.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

CITECLAW_LOGGER = "citeclaw"
"""Root logger name shared by every ``citeclaw.*`` submodule."""

_CONSOLE_FORMAT = "[%(asctime)s] %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"
_FILE_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"
_LOG_FILENAME = "citeclaw.log"

_CONFIGURED = False


def setup_logging(
    log_dir: Path | None = None, level: int = logging.INFO,
) -> None:
    """Configure the ``"citeclaw"`` logger; idempotent across repeated calls.

    Adds two handlers to the parent logger when first called:

    * stderr console handler at ``level`` with a short ``HH:MM:SS``
      timestamp (no logger name, to keep the dashboard readable);
    * file handler at ``DEBUG`` writing to ``<log_dir>/citeclaw.log``
      with the full ``YYYY-MM-DD HH:MM:SS LEVEL logger.name`` prefix
      — only when ``log_dir`` is provided.

    The parent logger itself is set to ``DEBUG`` so handlers can filter
    independently. ``propagate`` is disabled so messages don't bubble
    up to the root logger and double-print under pytest.

    Calling :func:`setup_logging` a second time is a no-op (handlers
    are not re-attached). Tests that need to reconfigure must reset
    :data:`_CONFIGURED` directly.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger(CITECLAW_LOGGER)
    root.setLevel(logging.DEBUG)
    root.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_CONSOLE_DATEFMT))
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / _LOG_FILENAME, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATEFMT))
        root.addHandler(fh)
