"""Recipe Protocol + ``FetchResult`` dataclass.

A "recipe" is a per-publisher fetch strategy. The contract is:

  * ``matches(doi: str) -> bool`` — claims the DOI based on its prefix
    (publisher fingerprint). Cheap, no I/O.
  * ``needs_browser: bool`` — class attribute. False = HTTP-only,
    True = needs a Playwright ``Page`` (the fetcher will boot one).
  * ``fetch(doi, *, browser_page=None, http=None) -> FetchResult`` —
    actually downloads the PDF bytes. Recipes raise nothing — every
    failure is captured in ``FetchResult.status`` so the orchestrator
    can keep going.

Recipes are deliberately stateless. Concurrency is the orchestrator's
job, and Playwright pages are passed in as parameters so the same recipe
can be reused across many papers without reopening the browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import httpx
    from playwright.sync_api import Page


# Status code vocabulary the orchestrator dispatches on.
# Keep these in sync with the docstring on Fetcher in fetcher.py.
STATUS_OK = "ok"                       # PDF bytes captured successfully
STATUS_AUTH = "auth_required"          # SSO redirect — user needs to log in once
STATUS_BLOCKED = "blocked"             # Cloudflare / bot detection
STATUS_NOT_FOUND = "not_found"         # 404 / DOI doesn't resolve at this publisher
STATUS_NOT_PDF = "not_pdf"             # Got bytes but they aren't a PDF
STATUS_ERROR = "error"                 # Anything else (timeout, transport, JS crash)


@dataclass
class FetchResult:
    """Result of one recipe fetch attempt."""

    paper_id: str
    doi: str
    status: str                # one of STATUS_*
    pdf_bytes: bytes | None = None
    fetched_via: str = ""      # recipe name, e.g. "nature_recipe"
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and self.pdf_bytes is not None


class Recipe(Protocol):
    """Per-publisher fetch strategy. Stateless."""

    name: str
    needs_browser: bool

    def matches(self, doi: str) -> bool: ...

    def fetch(
        self,
        paper_id: str,
        doi: str,
        *,
        browser_page: "Page | None" = None,
        http: "httpx.Client | None" = None,
    ) -> FetchResult: ...
