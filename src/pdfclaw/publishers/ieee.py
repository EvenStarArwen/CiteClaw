"""IEEE Xplore — 10.1109/...

Covers IEEE Trans. on Pattern Analysis and Machine Intelligence
(TPAMI), IEEE/ACM Trans. on Computational Biology and Bioinformatics,
IEEE Access, conference proceedings (CVPR / ICCV / etc. for CS),
and so on. Hosted at ieeexplore.ieee.org.

IEEE Xplore is one of the most hostile sites in academia for
automated access — they aggressively detect bots, gate PDFs behind
multiple iframes, and have their own subscription enforcement layer
on top of institutional auth. The selectors here are best-effort;
IEEE papers may need manual fallback if the click flow breaks.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class IEEERecipe(BrowserRecipeBase):
    name = "ieee_browser"
    DOI_PREFIX = "10.1109/"
    EXTRA_WAIT_MS = 3_000  # IEEE has a slow JS-rendered toolbar
    DOWNLOAD_SELECTORS = [
        'a[aria-label*="PDF"]',
        'xpl-pdf-viewer-button',
        'a.stats-document-lh-action-downloadPdf_2',
        'a:has-text("PDF")',
        'button:has-text("PDF")',
    ]
