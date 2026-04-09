"""ACM Digital Library — 10.1145/...

Covers Communications of the ACM, ACM TOG (SIGGRAPH), ACM TOIS,
KDD / WWW / NeurIPS-equivalent CS conference proceedings, etc.
Hosted at dl.acm.org.

Many recent ACM papers are open access (the ACM Open initiative);
older ones need subscription. The PDF anchor is the same in both cases.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class ACMRecipe(BrowserRecipeBase):
    name = "acm_browser"
    DOI_PREFIX = "10.1145/"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        'a[title="PDF"]',
        'a.btn--primary[href*=".pdf"]',
        'a[href*="/doi/pdf/"]',
        'a:has-text("PDF")',
        'a:has-text("Download PDF")',
    ]
