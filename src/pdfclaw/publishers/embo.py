"""EMBO Press — 10.15252/...

Covers The EMBO Journal, EMBO Reports, Molecular Systems Biology
(fully OA), EMBO Molecular Medicine. Embo papers used to live at
embopress.org but moved to Wiley's onlinelibrary.wiley.com platform
in 2023; the 10.15252 prefix still works as a DOI redirect.

Many EMBO articles are now hosted via Wiley's frontend, so the
selector list overlaps with WileyRecipe — but the registry will hit
this recipe first because it matches on the more specific 10.15252
prefix.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class EmboRecipe(BrowserRecipeBase):
    name = "embo_browser"
    DOI_PREFIX = "10.15252/"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        # Wiley-hosted EMBO articles
        'a.coolBar__ctrl[href*="/epdf/"]',
        'a[href*="/doi/pdf/"]',
        'a[href*="/pdfdirect/"]',
        # Legacy embopress.org pattern
        'a.article-pdfLink',
        'a:has-text("Download PDF")',
        'a:has-text("PDF")',
    ]
