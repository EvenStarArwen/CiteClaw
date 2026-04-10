"""PNAS (Proceedings of the National Academy of Sciences) — 10.1073/...

PNAS articles older than 6 months are fully open access via the NIH
PMC mirror; newer ones are subscription. PNAS uses Atypon Literatum
(same backend as science.org) so the selectors are similar.

The HighwireOpenAccessRecipe also matches 10.1073/ — PNAS does expose
``citation_pdf_url`` for OA papers — so for the OA-window cases the
HTTP recipe handles it. This browser recipe is the fallback for
recent paywalled papers (which need institutional auth).

The order in build_default_registry() is HighwireOpenAccessRecipe
BEFORE PNASRecipe, so the HTTP path is tried first.
"""

from __future__ import annotations

from pdfclaw.publishers._browser_base import BrowserRecipeBase


class PNASRecipe(BrowserRecipeBase):
    name = "pnas_browser"
    DOI_PREFIX = "10.1073/"
    PDF_URL_TEMPLATE = "https://www.pnas.org/doi/pdf/{doi}"
    EXTRA_WAIT_MS = 1_500
    DOWNLOAD_SELECTORS = [
        'a[data-test="download-pdf"]',
        'a[href*="/doi/pdf/"]',
        'a[href*="/doi/epdf/"]',
        'a:has-text("Download PDF")',
        'a:has-text("PDF")',
    ]
