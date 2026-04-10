"""Science / AAAS recipe — 10.1126/...

Covers Science, Science Advances, Science Translational Medicine,
Science Robotics, Science Immunology, Science Signaling. All hosted
at science.org.

Science.org runs a heavily-customised Atypon platform. The PDF link
sits in the top-right action toolbar. Some flagship articles are
fully open access; others require institutional auth.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class ScienceRecipe(BrowserRecipeBase):
    name = "science_browser"
    DOI_PREFIX = "10.1126/"
    PDF_URL_TEMPLATE = "https://www.science.org/doi/pdf/{doi}"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        # Science.org canonical PDF anchor
        'a[data-test="download-pdf"]',
        'a[href*="/doi/pdf/"]',
        'a:has-text("Download PDF")',
        'a:has-text("PDF")',
        'a[aria-label*="PDF"]',
    ]
