"""Springer (excluding Springer Nature/Nature Portfolio) — 10.1007/...

Covers Springer's classical book/journal lineup that doesn't go
through nature.com. Examples: Bioinformatics-adjacent journals,
Genome Biology and Evolution, Springer LNCS proceedings, etc.
Hosted at link.springer.com.

Note that 10.1186/ (BMC / Springer Open) is fully open access and
handled by HighwireOpenAccessRecipe — Springer.com proper sometimes
needs institutional auth, sometimes not (depends on the title).
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class SpringerRecipe(BrowserRecipeBase):
    name = "springer_browser"
    DOI_PREFIX = "10.1007/"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        'a[data-track-action="Pdf download"]',
        'a[data-test="pdf-link"]',
        'a.c-pdf-download__link',
        'a[href*=".pdf"][data-track-action*="ownload"]',
        'a:has-text("Download PDF")',
    ]
