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
        # OUP's canonical PDF link — the href contains the actual PDF path
        # e.g. /bioinformatics/article-pdf/33/21/3387/50315453/xxx.pdf
        'a.article-pdfLink',
        'a.pdf[href*="article-pdf"]',
        'a[href*="article-pdf"][href$=".pdf"]',
        'a[data-track-action="ArticleHeader|Download PDF"]',
        'a:has-text("PDF")',
    ]
