"""American Chemical Society (ACS) recipe — 10.1021/...

Covers all ACS journals: J. Am. Chem. Soc., Nano Letters, ACS Central
Science, etc. Hosted at pubs.acs.org.

The PDF link is a direct anchor with ``/doi/pdf/`` in the href. ACS
also has a ``data-mrf-class`` attribute on their canonical action
buttons.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class ACSRecipe(BrowserRecipeBase):
    name = "acs_browser"
    DOI_PREFIX = "10.1021/"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        'a[title="PDF"]',
        'a[href*="/doi/pdf/"]',
        'a[href*="/doi/epdf/"]',
        'a:has-text("Download PDF")',
        'a:has-text("PDF")',
    ]
