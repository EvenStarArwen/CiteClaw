"""Cold Spring Harbor Laboratory Press / RNA Society — 10.1261/...

Covers RNA, Genes & Development, Genome Research, Cold Spring Harbor
Perspectives, etc. — all hosted at the various ``*.cshlp.org`` and
``rnajournal.cshlp.org`` subdomains. CSHL is a Highwire-based stack
so the standard download anchor selectors apply.

Many CSHL articles enter open access ~6 months after publication.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class CSHLRecipe(BrowserRecipeBase):
    name = "cshl_browser"
    DOI_PREFIX = "10.1261/"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        'a.article-dl-pdf-link',
        'a[href$=".full.pdf"]',
        'a[href*="/content/"][href*=".pdf"]',
        'a:has-text("Download PDF")',
        'a:has-text("Full Text PDF")',
    ]
