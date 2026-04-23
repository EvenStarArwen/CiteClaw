"""Main path analysis — extract the most-cited knowledge trajectory
through a CiteClaw citation network.

Canonical algorithms following Hummon & Doreian (1989), Batagelj
(2003), Liu & Lu (2012), and Liu, Lu & Ho (2019). Invoked via the
``citeclaw mainpath`` subcommand (see :mod:`citeclaw.__main__`); the
programmatic entry point is :func:`run_mpa`.

Three orthogonal axes compose to define one MPA run:

==============  =====================================  ===============
Axis            Choices                                Default
==============  =====================================  ===============
weight          ``spc`` / ``splc`` / ``spnp``           ``spc``
search          ``local-forward`` / ``local-backward``  ``key-route``
                / ``global`` / ``key-route`` /
                ``multi-local``
cycle           ``shrink`` / ``preprint``               ``shrink``
==============  =====================================  ===============

Each axis has its own module-level registry so new variants can be
added without touching the runner.
"""

from __future__ import annotations

from citeclaw.mainpath.base import CyclePolicyTrace, MainPathResult
from citeclaw.mainpath.cycles import CYCLE_REGISTRY
from citeclaw.mainpath.runner import run_mpa
from citeclaw.mainpath.search import SEARCH_REGISTRY
from citeclaw.mainpath.weights import WEIGHT_REGISTRY

__all__ = [
    "CyclePolicyTrace",
    "MainPathResult",
    "run_mpa",
    "WEIGHT_REGISTRY",
    "SEARCH_REGISTRY",
    "CYCLE_REGISTRY",
]
