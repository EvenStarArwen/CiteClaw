"""Taylor & Francis Online — 10.1080/...

Covers mAbs, Bioinformatics-adjacent journals, etc. Hosted at
www.tandfonline.com. Many T&F articles need institutional auth
for the PDF.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class TaylorFrancisRecipe(BrowserRecipeBase):
    name = "tandf_browser"
    DOI_PREFIX = "10.1080/"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        'a[href*="/doi/pdf/"]',
        'a[href*="/doi/epdf/"]',
        'a.show-pdf',
        'a:has-text("Download PDF")',
        'a:has-text("PDF")',
    ]
