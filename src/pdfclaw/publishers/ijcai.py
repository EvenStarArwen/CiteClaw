"""IJCAI proceedings — 10.24963/...

The International Joint Conference on Artificial Intelligence
proceedings live at ``www.ijcai.org/proceedings/<year>/<paper_id>``
and are fully open access. The PDF is at the same URL with ``.pdf``
appended. The DOI redirect lands on the article page; the download
link is a direct anchor.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import DEFAULT_SSO_HOSTS, BrowserRecipeBase


class IJCAIRecipe(BrowserRecipeBase):
    name = "ijcai_browser"
    DOI_PREFIX = "10.24963/"
    EXTRA_WAIT_MS = 1_000
    SSO_HOSTS = tuple(h for h in DEFAULT_SSO_HOSTS if h not in ("login.", "/login"))
    DOWNLOAD_SELECTORS = [
        'a[href*="/proceedings/"][href$=".pdf"]',
        'a[href$=".pdf"]',
        'a:has-text("PDF")',
    ]
