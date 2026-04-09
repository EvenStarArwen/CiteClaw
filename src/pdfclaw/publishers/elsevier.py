"""Elsevier ScienceDirect (incl. Cell Press) — 10.1016/...

Covers all Elsevier-published journals: Cell Press (Cell, Mol Cell,
Cell Reports, Cell Systems, ...), classical Elsevier (Plant Cell,
Journal of Molecular Biology, ...), Lancet, etc. They share one
ScienceDirect frontend so the selectors are identical.

ScienceDirect is more hostile to scraping than Nature: bot detection
is aggressive, and the PDF download is gated behind a "View PDF" -> JS
click. A real Chrome profile (not headless) handles all of this.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class ElsevierRecipe(BrowserRecipeBase):
    name = "elsevier_browser"
    DOI_PREFIX = "10.1016/"
    EXTRA_WAIT_MS = 2_000  # ScienceDirect needs an extra beat for the JS-rendered toolbar
    DOWNLOAD_SELECTORS = [
        # Newer ScienceDirect "View PDF" button (most common today)
        'a:has-text("View PDF")',
        # Older "Download PDF" anchor
        'a:has-text("Download PDF")',
        # Cell Press direct PDF link in the action toolbar
        'a.article-tools__item__link[href*=".pdf"]',
        'a.download-pdf-link',
        'a[aria-label*="Download"][aria-label*="PDF"]',
        'a[type="application/pdf"]',
    ]
