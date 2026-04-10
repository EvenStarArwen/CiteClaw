"""Wiley Online Library recipe — 10.1002/...

Covers Plant Biotech Journal, Plant Cell & Environment, Advanced
Materials, Angewandte Chemie, etc. — anything hosted at
onlinelibrary.wiley.com.

The download link is straightforward but Wiley sometimes returns an
"epdf" page first (an inline PDF viewer); the actual download anchor
on that page is what we want. Both selectors are listed.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class WileyRecipe(BrowserRecipeBase):
    name = "wiley_browser"
    DOI_PREFIX = "10.1002/"
    PDF_URL_TEMPLATE = "https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        # Direct PDF link in the article header (full text article view)
        'a.coolBar__ctrl[href*="/epdf/"]',
        'a.pdf-download[href*="/pdfdirect/"]',
        'a[href*="/doi/pdf/"]',
        'a[href*="/pdfdirect/"]',
        'a:has-text("PDF")',
        'a[aria-label="Download PDF"]',
    ]
