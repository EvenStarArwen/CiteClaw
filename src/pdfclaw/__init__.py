"""pdfclaw — institutional-auth PDF fetcher for CiteClaw collections.

Standalone sister package to ``citeclaw``. Drives a real Chrome instance
(via Playwright + a persistent user-data-dir) so it can reuse your
institutional SSO cookies and download paywalled PDFs the same way you
would by hand. No LLM in the loop — every publisher recipe is
deterministic Playwright code, so the only cost is your existing
ScraperAPI-equivalent (your own Chrome) and Modal compute is irrelevant.

The package never imports from ``citeclaw``; the two share the same
project root and pyproject only because that's where the wheel target
lives.
"""

__version__ = "0.1.0"
