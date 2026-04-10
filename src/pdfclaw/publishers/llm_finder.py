"""LLM-guided PDF link finder — universal fallback for unknown publishers.

When all hardcoded recipes fail (no PDF_URL_TEMPLATE, selectors don't
match, etc.), this recipe loads the article page in the browser,
extracts all candidate links, and asks an LLM to pick the one that
downloads the main article PDF.

The LLM sees a numbered list of links (href + text + attributes) and
replies with a single number. If the chosen link returns HTML instead
of PDF (common: publisher opens an inline reader), we navigate to that
page and ask the LLM again — up to MAX_ROUNDS times.

Uses the Modal Gemma-4-31B endpoint (or any OpenAI-compatible endpoint)
configured via environment variables. Gated: if the env vars aren't
set, the recipe returns NOT_FOUND immediately.

Env vars:
  PDFCLAW_LLM_BASE_URL   — e.g. https://you--citeclaw-vllm-gemma-serve.modal.run/v1
  PDFCLAW_LLM_API_KEY    — bearer token for the endpoint
  PDFCLAW_LLM_MODEL      — model name (default: google/gemma-4-31B-it)

Or it falls back to CITECLAW_VLLM_API_KEY + a hardcoded base_url if
the PDFCLAW_ vars aren't set.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from pdfclaw.publishers.base import (
    STATUS_AUTH,
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_NOT_PDF,
    STATUS_OK,
    FetchResult,
)

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, Page

log = logging.getLogger("pdfclaw.llm_finder")

MAX_ROUNDS = 2       # article page → reader page at most
MAX_CANDIDATES = 20   # don't overwhelm the LLM

SYSTEM_PROMPT = """\
You help download PDFs of academic papers. You will see a numbered list \
of links from a webpage. Pick the link that downloads the MAIN article \
PDF (not supplementary material, not figures, not references).

Rules:
- Reply with ONLY the number (e.g. "3"), nothing else
- If the page is a PDF reader/viewer, pick the "Download" or "Save" button
- If no link leads to the main PDF, reply "none"
"""

USER_TEMPLATE = """\
Paper DOI: {doi}
Page URL: {url}
Page title: {title}

Candidate links:
{candidates}

Which number is the main article PDF download?"""


def _get_llm_config() -> tuple[str, str, str] | None:
    """Return (base_url, api_key, model) or None if not configured."""
    base_url = os.environ.get("PDFCLAW_LLM_BASE_URL") or ""
    api_key = (
        os.environ.get("PDFCLAW_LLM_API_KEY")
        or os.environ.get("CITECLAW_VLLM_API_KEY")
        or ""
    )
    model = os.environ.get("PDFCLAW_LLM_MODEL") or "google/gemma-4-31B-it"

    if not api_key:
        return None
    if not base_url:
        # Try to detect from common CiteClaw env patterns
        base_url = os.environ.get("CITECLAW_VLLM_BASE_URL") or ""
    if not base_url:
        return None
    return base_url, api_key, model


def _extract_candidates(page: "Page") -> list[dict]:
    """Extract all links that look PDF/download-related from the page."""
    try:
        raw = page.evaluate(
            """() => Array.from(document.querySelectorAll('a, button')).map(el => ({
                tag: el.tagName.toLowerCase(),
                href: el.getAttribute('href') || '',
                text: (el.innerText || '').trim().slice(0, 120),
                cls: el.getAttribute('class') || '',
                aria: el.getAttribute('aria-label') || '',
                title: el.getAttribute('title') || '',
            })).filter(a =>
                /pdf/i.test(a.href) || /pdf/i.test(a.text) ||
                /pdf/i.test(a.aria) || /pdf/i.test(a.title) ||
                /download/i.test(a.text) || /download/i.test(a.aria) ||
                /download/i.test(a.title) || /full.text/i.test(a.text) ||
                /save/i.test(a.text)
            )"""
        )
    except Exception:  # noqa: BLE001
        return []
    return (raw or [])[:MAX_CANDIDATES]


def _ask_llm(
    base_url: str, api_key: str, model: str,
    doi: str, page_url: str, page_title: str,
    candidates: list[dict],
) -> int | None:
    """Ask the LLM which candidate to pick. Returns index or None."""
    lines = []
    for i, c in enumerate(candidates):
        parts = [f'[{i}]']
        if c.get("href"):
            parts.append(f'href="{c["href"][:120]}"')
        if c.get("text"):
            parts.append(f'text="{c["text"]}"')
        if c.get("cls"):
            parts.append(f'class="{c["cls"][:60]}"')
        if c.get("aria"):
            parts.append(f'aria="{c["aria"][:60]}"')
        lines.append(" ".join(parts))

    prompt = USER_TEMPLATE.format(
        doi=doi,
        url=page_url,
        title=page_title[:100],
        candidates="\n".join(lines),
    )

    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 10,
                "temperature": 0,
            },
            timeout=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM call failed: %s", exc)
        return None

    if resp.status_code != 200:
        log.warning("LLM returned HTTP %d", resp.status_code)
        return None

    try:
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return None

    if text.lower() == "none":
        return None

    m = re.search(r"\d+", text)
    if m:
        idx = int(m.group())
        if 0 <= idx < len(candidates):
            return idx
    return None


class LLMPdfFinderRecipe:
    """Universal PDF finder powered by LLM. Fallback for unknown publishers."""

    name = "llm_finder"
    needs_browser = True

    def matches(self, doi: str) -> bool:
        # Matches any DOI — but only when LLM is configured
        return doi.lower().startswith("10.")

    def fetch(
        self,
        paper_id: str,
        doi: str,
        *,
        browser_page: "Page | None" = None,
        http=None,  # noqa: ARG002
    ) -> FetchResult:
        config = _get_llm_config()
        if config is None:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                fetched_via=self.name,
                error=(
                    "LLM finder disabled: set PDFCLAW_LLM_BASE_URL + "
                    "PDFCLAW_LLM_API_KEY (or CITECLAW_VLLM_API_KEY) to enable"
                ),
            )

        if browser_page is None:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error="needs browser_page",
            )

        base_url, api_key, model = config
        page = browser_page
        context: BrowserContext = page.context

        # Navigate to the article via DOI redirect
        try:
            page.goto("about:blank", wait_until="commit", timeout=5_000)
        except Exception:  # noqa: BLE001
            pass

        try:
            page.goto(
                f"https://doi.org/{doi}",
                wait_until="domcontentloaded",
                timeout=45_000,
            )
            page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:  # noqa: BLE001
            pass

        for round_num in range(MAX_ROUNDS):
            candidates = _extract_candidates(page)
            if not candidates:
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                    fetched_via=self.name,
                    error=f"Round {round_num}: no PDF-related links found on page",
                )

            chosen_idx = _ask_llm(
                base_url, api_key, model,
                doi, page.url, page.title(),
                candidates,
            )
            if chosen_idx is None:
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                    fetched_via=self.name,
                    error=f"Round {round_num}: LLM said 'none' or failed to respond",
                )

            chosen = candidates[chosen_idx]
            href = chosen.get("href", "")
            if not href:
                continue

            # Make absolute
            if href.startswith("/"):
                parts = urlparse(page.url)
                href = f"{parts.scheme}://{parts.netloc}{href}"
            elif not href.startswith("http"):
                href = f"https://{href}"

            log.info(
                "LLM picked [%d] %s (text=%r) on round %d",
                chosen_idx, href[:80], chosen.get("text", "")[:40], round_num,
            )

            # Try fetching the PDF
            try:
                api_resp = context.request.get(href, timeout=30_000)
            except Exception as exc:  # noqa: BLE001
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                    fetched_via=self.name,
                    error=f"Round {round_num}: fetch {href} failed: {exc}",
                )

            if not api_resp.ok:
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                    fetched_via=self.name,
                    error=f"Round {round_num}: HTTP {api_resp.status} for {href}",
                )

            body = api_resp.body()
            if body.startswith(b"%PDF-"):
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_OK,
                    pdf_bytes=body, fetched_via=self.name,
                    extra={
                        "href": href,
                        "llm_round": round_num,
                        "llm_choice": chosen_idx,
                        "n_bytes": len(body),
                    },
                )

            # Got HTML — probably a reader page. Navigate there and retry.
            if b"<html" in body[:1000].lower():
                log.info(
                    "Round %d: got HTML, navigating to reader page for round %d",
                    round_num, round_num + 1,
                )
                try:
                    page.goto(href, wait_until="domcontentloaded", timeout=30_000)
                    page.wait_for_load_state("networkidle", timeout=10_000)
                except Exception:  # noqa: BLE001
                    pass
                continue

            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_NOT_PDF,
                fetched_via=self.name,
                error=f"Round {round_num}: response isn't PDF (first bytes: {body[:8]!r})",
            )

        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_ERROR,
            fetched_via=self.name,
            error=f"Exhausted {MAX_ROUNDS} rounds without finding a PDF",
        )
