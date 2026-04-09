"""Nature Portfolio (Springer Nature) recipe — 10.1038/...

Covers Nature, Nature Plants, Nature Methods, Nature Communications,
Nature Biotechnology, Scientific Reports, Nature Chemistry, etc. All
of these run on the same Oscar Platform stack so the same selectors
apply.

The Highwire-compliant ``data-track-action="download pdf"`` anchor has
been the canonical click target since at least 2020. If Nature ever
re-skins the article page, the secondary selectors below should still
catch it.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class NatureRecipe(BrowserRecipeBase):
    name = "nature_browser"
    DOI_PREFIX = "10.1038/"
    DOWNLOAD_SELECTORS = [
        'a[data-track-action="download pdf"]',
        'a[data-test="download-pdf"]',
        'a.c-pdf-download__link',
        'a:has-text("Download PDF")',
    ]
