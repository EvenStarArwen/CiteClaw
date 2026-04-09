"""IOP Publishing — 10.1088/...

Institute of Physics journals (Phys. Med. Biol., J. Neural Eng.,
Machine Learning: Science and Technology, etc.). Hosted at
iopscience.iop.org.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class IOPRecipe(BrowserRecipeBase):
    name = "iop_browser"
    DOI_PREFIX = "10.1088/"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        'a.btn-pdf',
        'a[href*="/article/"][href*="/pdf"]',
        'a:has-text("Download PDF")',
        'a:has-text("PDF")',
    ]
