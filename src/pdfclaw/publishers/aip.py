"""American Institute of Physics (AIP) — 10.1063/...

Covers Journal of Chemical Physics, Applied Physics Letters, Review
of Scientific Instruments, etc. Hosted at pubs.aip.org on Atypon
Literatum (same backend as science.org).
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class AIPRecipe(BrowserRecipeBase):
    name = "aip_browser"
    DOI_PREFIX = "10.1063/"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        'a[data-test="download-pdf"]',
        'a[href*="/doi/pdf/"]',
        'a:has-text("Download PDF")',
        'a:has-text("PDF")',
    ]
