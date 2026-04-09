"""Publisher recipe registry + per-DOI dispatch.

Each recipe knows how to claim a DOI (by prefix) and how to fetch the
corresponding PDF. Multiple recipes can match the same DOI — the
``find_recipes`` helper returns ALL of them in registry order so the
fetcher can try them as a chain (cheap HTTP first, expensive browser
fallback second).

Registry order:

  1. **HTTP recipes** (no browser, free) — tried first
  2. **Browser recipes** (need Chromium + persistent profile) — tried
     second when the HTTP options fail or don't match

For overlapping prefixes (e.g. 10.1073/ matches both
``HighwireOpenAccessRecipe`` HTTP and ``PNASRecipe`` browser), the
HTTP one comes first so OA-window papers stay cheap; only the
recent paywalled ones fall through to the browser fallback.
"""

from __future__ import annotations

from pdfclaw.publishers.acm import ACMRecipe
from pdfclaw.publishers.acs import ACSRecipe
from pdfclaw.publishers.arxiv import ArxivRecipe
from pdfclaw.publishers.base import FetchResult, Recipe
from pdfclaw.publishers.biorxiv import BiorxivRecipe
from pdfclaw.publishers.elife import ELifeRecipe
from pdfclaw.publishers.elsevier import ElsevierRecipe
from pdfclaw.publishers.embo import EmboRecipe
from pdfclaw.publishers.highwire_oa import HighwireOpenAccessRecipe
from pdfclaw.publishers.ieee import IEEERecipe
from pdfclaw.publishers.iop import IOPRecipe
from pdfclaw.publishers.mdpi import MDPIRecipe
from pdfclaw.publishers.nature import NatureRecipe
from pdfclaw.publishers.oxford import OxfordRecipe
from pdfclaw.publishers.pnas import PNASRecipe
from pdfclaw.publishers.rsc import RSCRecipe
from pdfclaw.publishers.science import ScienceRecipe
from pdfclaw.publishers.springer import SpringerRecipe
from pdfclaw.publishers.wiley import WileyRecipe

__all__ = [
    "ACMRecipe",
    "ACSRecipe",
    "ArxivRecipe",
    "BiorxivRecipe",
    "ELifeRecipe",
    "ElsevierRecipe",
    "EmboRecipe",
    "FetchResult",
    "HighwireOpenAccessRecipe",
    "IEEERecipe",
    "IOPRecipe",
    "MDPIRecipe",
    "NatureRecipe",
    "OxfordRecipe",
    "PNASRecipe",
    "RSCRecipe",
    "Recipe",
    "ScienceRecipe",
    "SpringerRecipe",
    "WileyRecipe",
    "build_default_registry",
    "find_recipe",
    "find_recipes",
]


def build_default_registry() -> list[Recipe]:
    """Return the default ordered recipe list. HTTP recipes go first."""
    return [
        # === HTTP-only recipes (free, fast, no browser) ===
        ArxivRecipe(),                  # 10.48550/  — arXiv direct PDF URL
        HighwireOpenAccessRecipe(),     # 10.1186, 10.3389, 10.1371, 10.3390, 10.7554, 10.1073 (OA window)

        # === Browser recipes (need Chromium; some need SSO profile) ===
        # Open access but JS / Cloudflare / Accept-header blocked from httpx
        BiorxivRecipe(),                # 10.1101/   — Cloudflare cha­llenge zone
        MDPIRecipe(),                   # 10.3390/   — JS-rendered, CDN gates direct PDF URL
        ELifeRecipe(),                  # 10.7554/   — 406 on direct PDF URL
        # Paywalled — need SSO profile
        NatureRecipe(),                 # 10.1038/   — Nature Portfolio (Springer Nature)
        ElsevierRecipe(),               # 10.1016/   — Elsevier ScienceDirect (incl. Cell Press)
        ScienceRecipe(),                # 10.1126/   — AAAS Science family
        WileyRecipe(),                  # 10.1002/   — Wiley Online Library
        ACSRecipe(),                    # 10.1021/   — American Chemical Society
        OxfordRecipe(),                 # 10.1093/   — Oxford Academic (Bioinformatics, NAR, ...)
        RSCRecipe(),                    # 10.1039/   — Royal Society of Chemistry
        IEEERecipe(),                   # 10.1109/   — IEEE Xplore
        ACMRecipe(),                    # 10.1145/   — ACM Digital Library
        SpringerRecipe(),               # 10.1007/   — Springer (non-Nature)
        IOPRecipe(),                    # 10.1088/   — IOP Publishing (Phys. Med. Biol., ...)
        EmboRecipe(),                   # 10.15252/  — EMBO Press (now hosted on Wiley)
        PNASRecipe(),                   # 10.1073/   — PNAS paywalled fallback (HighwireOA tries OA first)
    ]


def find_recipe(doi: str, registry: list[Recipe]) -> Recipe | None:
    """First recipe in registry order whose ``matches(doi)`` is True, or None."""
    for recipe in registry:
        if recipe.matches(doi):
            return recipe
    return None


def find_recipes(doi: str, registry: list[Recipe]) -> list[Recipe]:
    """All recipes that claim the DOI, in registry order.

    Used by the fetcher to build a per-paper fallback chain — try the
    cheapest matching recipe first, then walk through the more
    expensive ones if it fails.
    """
    return [r for r in registry if r.matches(doi)]
