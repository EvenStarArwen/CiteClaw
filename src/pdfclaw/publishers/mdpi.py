"""MDPI recipe — 10.3390/...

MDPI is fully open access but their site is JavaScript-rendered and
they don't expose ``citation_pdf_url`` meta tags consistently. Direct
HTTP fetches of guessed PDF URLs (``/<journal>/<vol>/<issue>/<id>/pdf``)
get 403 from their CDN. The reliable way is to drive a real browser:
navigate to the article landing page, click the "Download PDF" link.

No auth required.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import DEFAULT_SSO_HOSTS, BrowserRecipeBase


class MDPIRecipe(BrowserRecipeBase):
    name = "mdpi_browser"
    DOI_PREFIX = "10.3390/"
    EXTRA_WAIT_MS = 1_500
    # MDPI is OA — strip out SSO substrings that might cause false positives
    SSO_HOSTS = tuple(h for h in DEFAULT_SSO_HOSTS if h not in ("login.", "/login"))
    DOWNLOAD_SELECTORS = [
        # MDPI standard "Download PDF" button in the article toolbar
        'a[href$="/pdf"]',
        'a.UD_ArticlePDF',
        'a.button[href*="/pdf"]',
        'a:has-text("Download PDF")',
    ]
