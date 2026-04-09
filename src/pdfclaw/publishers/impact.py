"""Impact Journals (Oncotarget, Aging, Genes & Cancer) — 10.18632/...

Hosted at oncotarget.com / aging-us.com. All Impact Journals titles
are nominally open access but their direct PDF URLs are gated behind
the article landing page; a real browser visit is the simplest path.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import DEFAULT_SSO_HOSTS, BrowserRecipeBase


class ImpactJournalsRecipe(BrowserRecipeBase):
    name = "impact_browser"
    DOI_PREFIX = "10.18632/"
    EXTRA_WAIT_MS = 1_500
    SSO_HOSTS = tuple(h for h in DEFAULT_SSO_HOSTS if h not in ("login.", "/login"))
    DOWNLOAD_SELECTORS = [
        'a[href*="/article/"][href*="/pdf"]',
        'a[href$=".pdf"]',
        'a:has-text("Download PDF")',
        'a:has-text("Full Text PDF")',
        'a:has-text("PDF")',
    ]
