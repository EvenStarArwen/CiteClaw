"""Oxford University Press / Oxford Academic — 10.1093/...

Covers Bioinformatics, Briefings in Bioinformatics, Nucleic Acids
Research (NAR is fully OA), Genetics, Plant Cell Physiology, etc.
Hosted at academic.oup.com.

NAR articles are open access and the download usually works without
institutional auth; Bioinformatics and BiB are subscription-based.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class OxfordRecipe(BrowserRecipeBase):
    name = "oxford_browser"
    DOI_PREFIX = "10.1093/"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        'a[data-track-action="ArticleHeader|Download PDF"]',
        'a[href*=".pdf"][data-track-action*="PDF"]',
        'a:has-text("Download PDF")',
        'a:has-text("PDF")',
    ]
