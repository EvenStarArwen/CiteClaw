"""Royal Society of Chemistry (RSC) — 10.1039/...

Covers Chemical Science, Chem. Soc. Rev., Nanoscale, Soft Matter,
etc. Hosted at pubs.rsc.org.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class RSCRecipe(BrowserRecipeBase):
    name = "rsc_browser"
    DOI_PREFIX = "10.1039/"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        'a[title="Link to landing page article PDF"]',
        'a.btn__primary[href*=".pdf"]',
        'a[href$=".pdf"]',
        'a:has-text("PDF")',
    ]
