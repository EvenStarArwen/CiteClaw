"""eLife recipe — 10.7554/...

eLife is fully open access but they don't expose ``citation_pdf_url``
in a way our HTTP recipe can pick up, AND their direct PDF URLs
(``elifesciences.org/articles/<id>.pdf``) return 406 Not Acceptable
because of strict ``Accept`` header negotiation. A real browser
navigates without trouble.

DOI pattern:
  10.7554/eLife.<5-digit-id>
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import DEFAULT_SSO_HOSTS, BrowserRecipeBase


class ELifeRecipe(BrowserRecipeBase):
    name = "elife_browser"
    DOI_PREFIX = "10.7554/"
    EXTRA_WAIT_MS = 1_500
    SSO_HOSTS = tuple(h for h in DEFAULT_SSO_HOSTS if h not in ("login.", "/login"))
    DOWNLOAD_SELECTORS = [
        # eLife's article header download anchor
        'a[href$=".pdf"][data-download-type="article"]',
        'a[href$=".pdf"]',
        'a:has-text("Download")',
        'a:has-text("PDF")',
    ]
