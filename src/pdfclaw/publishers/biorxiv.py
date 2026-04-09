"""bioRxiv / medRxiv recipe — needs a real browser for Cloudflare.

Despite being open-access, bioRxiv (and its sibling medRxiv) sit
behind a Cloudflare ``cf-mitigated: challenge`` zone that 403s every
direct httpx request — even for the canonical Highwire
``citation_pdf_url`` PDFs. The only reliable way to fetch their PDFs is
to drive a real Chromium that auto-solves the JS challenge.

The good news is that bioRxiv requires no institutional auth — once
Chromium has cleared the challenge cookies (``cf_clearance``), the
"Download PDF" button works without ever signing in. So this recipe
shares the browser pass with the paywalled publishers but never trips
the SSO-detection branch.

DOI patterns:
  * bioRxiv:  10.1101/<id>
  * medRxiv:  10.1101/<id> (yes, same prefix — they share the
              ``10.1101/`` Crossref namespace)
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import DEFAULT_SSO_HOSTS, BrowserRecipeBase


class BiorxivRecipe(BrowserRecipeBase):
    name = "biorxiv_browser"
    DOI_PREFIX = "10.1101/"
    EXTRA_WAIT_MS = 3_000  # Let cf_clearance cookie settle
    # bioRxiv DOIs hit Cloudflare; their landing page never has SSO,
    # so override the default SSO host list to avoid false positives.
    SSO_HOSTS = tuple(h for h in DEFAULT_SSO_HOSTS if h not in ("login.", "/login"))
    DOWNLOAD_SELECTORS = [
        # bioRxiv direct download link in the article action toolbar
        'a.article-dl-pdf-link',
        'a.article-dl-pdf',
        'a:has-text("Download PDF")',
        # Highwire fallback — bioRxiv is technically a Highwire site
        'a[data-icon-position][href*=".full.pdf"]',
        'a[href*=".full.pdf"]',
    ]
