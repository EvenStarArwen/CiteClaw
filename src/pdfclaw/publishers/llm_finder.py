"""LLM-guided PDF finder — reasoning agent for unknown publishers.

A mini-agent that drives the browser to find and download a paper's
PDF. Unlike the hardcoded recipes, this uses an LLM (Modal Gemma or
any OpenAI-compatible endpoint) to REASON about what to do:

  1. Look at the page's links
  2. Decide which one to try (fetch or navigate)
  3. Check the result
  4. If it failed, REASON about why and try something else
  5. If the publisher's server is broken (404/500), recognize it
     and give up gracefully

The action space is tiny — pick a link, fetch it, check if it's a PDF
— so even a 31B model handles it well. Each turn costs ~500 input
tokens + ~100 output tokens (with reasoning).

Max 5 turns per paper. Most papers need 1-2.

Env vars:
  PDFCLAW_LLM_BASE_URL   — OpenAI-compatible endpoint
  PDFCLAW_LLM_API_KEY or CITECLAW_VLLM_API_KEY — bearer token
  PDFCLAW_LLM_MODEL       — model name (default: google/gemma-4-31B-it)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from pdfclaw.publishers.base import (
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_NOT_PDF,
    STATUS_OK,
    FetchResult,
)

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, Page

log = logging.getLogger("pdfclaw.llm_finder")

MAX_TURNS = 5
MAX_CANDIDATES = 25

SYSTEM_PROMPT = """\
You are an agent that downloads PDFs of academic papers from publisher websites. Your ONLY goal is to get the main article PDF. Everything else (cookie banners, login prompts, subscription nags, navigation menus) is an obstacle to dismiss or ignore.

You can take ONE action per turn:
- {"action": "click", "target": N, "reasoning": "..."} — CLICK link/button N in the browser (triggers JavaScript, best for download buttons)
- {"action": "fetch", "target": N, "reasoning": "..."} — HTTP GET link N's URL directly (fast, but no JavaScript)
- {"action": "navigate", "target": N, "reasoning": "..."} — open link N's page in the browser to see new links
- {"action": "give_up", "reasoning": "..."} — the PDF is genuinely unavailable

Decision rules:
- For download buttons/icons → use CLICK (it triggers the publisher's JavaScript download handler)
- For direct PDF URLs (.pdf in href) → use FETCH (faster than click)
- For reader/viewer pages (/reader/, /view/, /epdf/) → use NAVIGATE to see the page, then CLICK the download button inside
- If FETCH returns 403 → try CLICK on the same link instead (the browser has cookies that direct HTTP doesn't)
- If CLICK or FETCH returns 404/500 or "Failed to load" → GIVE UP
- Ignore cookie consent, privacy policy, ads, social media, reference links
- Pick the MAIN article PDF, not supplementary material or figures

Respond with ONLY a JSON object."""

USER_TEMPLATE = """\
Paper DOI: {doi}
Paper title: {title}
Current page: {url}
Page title: {page_title}

{history}

Available links on this page:
{candidates}

What should we do next? Respond with a JSON object."""


def _get_llm_config() -> tuple[str, str, str] | None:
    """Detect LLM config from env vars. Priority:
    1. Explicit PDFCLAW_LLM_* vars (any OpenAI-compatible endpoint)
    2. GEMINI_API_KEY (auto-configures Google's OpenAI-compatible endpoint)
    3. CITECLAW_VLLM_* vars (Modal Gemma)
    """
    # Option 1: explicit config
    base_url = os.environ.get("PDFCLAW_LLM_BASE_URL") or ""
    api_key = os.environ.get("PDFCLAW_LLM_API_KEY") or ""
    model = os.environ.get("PDFCLAW_LLM_MODEL") or ""
    if base_url and api_key:
        return base_url, api_key, model or "google/gemma-4-31B-it"

    # Option 2: Gemini (auto-detect from GEMINI_API_KEY)
    gemini_key = os.environ.get("GEMINI_API_KEY") or ""
    if gemini_key:
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            gemini_key,
            model or "gemini-3.1-flash-lite-preview",
        )

    # Option 3: Modal Gemma (CITECLAW_VLLM_* vars)
    vllm_key = os.environ.get("CITECLAW_VLLM_API_KEY") or ""
    vllm_base = os.environ.get("PDFCLAW_LLM_BASE_URL") or os.environ.get("CITECLAW_VLLM_BASE_URL") or ""
    if vllm_key and vllm_base:
        return vllm_base, vllm_key, model or "google/gemma-4-31B-it"

    return None


def _extract_pdf_from_viewer(page: "Page") -> bytes | None:
    """If Chrome opened a PDF in its built-in viewer, extract the bytes.

    When the page URL looks like a PDF URL (contains /pdf/ or ends in
    .pdf), use JavaScript fetch() from WITHIN the page to download the
    same URL. Because this fetch runs in the page's JS context, it has
    the full session cookies and auth — unlike context.request.get().
    """
    url = page.url.lower()
    if "/pdf" not in url and not url.endswith(".pdf"):
        return None

    log.info("Attempting to extract PDF from Chrome viewer at %s", page.url[:80])
    try:
        # JavaScript fetch from within the page — has all cookies
        raw = page.evaluate("""
            async () => {
                try {
                    const resp = await fetch(window.location.href, {
                        credentials: 'include'
                    });
                    if (!resp.ok) return null;
                    const buf = await resp.arrayBuffer();
                    // Convert to base64 for transfer to Python
                    const bytes = new Uint8Array(buf);
                    let binary = '';
                    for (let i = 0; i < bytes.length; i++) {
                        binary += String.fromCharCode(bytes[i]);
                    }
                    return btoa(binary);
                } catch (e) {
                    return null;
                }
            }
        """)
        if raw:
            import base64  # noqa: PLC0415
            body = base64.b64decode(raw)
            if len(body) > 1000:  # sanity check
                log.info("Extracted %d bytes from Chrome PDF viewer", len(body))
                return body
    except Exception as exc:  # noqa: BLE001
        log.debug("PDF viewer extraction failed: %s", exc)
    return None


def _dismiss_cookie_banners(page: "Page") -> None:
    """Try to close cookie consent banners that block the real content."""
    cookie_selectors = [
        'button:has-text("Accept All")',
        'button:has-text("Accept all")',
        'button:has-text("Accept Cookies")',
        'button:has-text("Accept")',
        'button:has-text("Agree")',
        'button:has-text("I agree")',
        'button:has-text("OK")',
        'button:has-text("Got it")',
        'button:has-text("Allow all")',
        '#onetrust-accept-btn-handler',
        '.cookie-accept',
        '[data-testid="cookie-accept"]',
        'button.accept-cookies',
    ]
    for sel in cookie_selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click(timeout=2_000)
                page.wait_for_timeout(500)
                return
        except Exception:  # noqa: BLE001
            continue


def _extract_candidates(page: "Page") -> list[dict]:
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
                /download/i.test(a.title) || /full.?text/i.test(a.text) ||
                /save/i.test(a.text) || /reader/i.test(a.href) ||
                /view/i.test(a.href) || /stamp/i.test(a.href)
            )"""
        )
    except Exception:  # noqa: BLE001
        return []
    return (raw or [])[:MAX_CANDIDATES]


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        parts = [f"[{i}]"]
        if c.get("href"):
            parts.append(f'href="{c["href"][:150]}"')
        if c.get("text"):
            parts.append(f'text="{c["text"]}"')
        if c.get("aria"):
            parts.append(f'aria-label="{c["aria"][:60]}"')
        lines.append(" ".join(parts))
    return "\n".join(lines) if lines else "(no relevant links found)"


def _call_llm(
    base_url: str, api_key: str, model: str,
    system: str, user: str,
) -> dict | None:
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
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 256,
                "temperature": 0,
            },
            timeout=60.0,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM call failed: %s", exc)
        return None

    if resp.status_code != 200:
        log.warning("LLM returned HTTP %d: %s", resp.status_code, resp.text[:200])
        return None

    try:
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return None

    # Extract JSON from the response (model might wrap it in markdown)
    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not json_match:
        log.warning("LLM response has no JSON: %s", raw[:200])
        return None

    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        log.warning("LLM JSON parse failed: %s", json_match.group()[:200])
        return None


class LLMPdfFinderRecipe:
    """Universal PDF finder powered by LLM reasoning."""

    name = "llm_finder"
    needs_browser = True

    def matches(self, doi: str) -> bool:
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
                error="LLM finder disabled: set PDFCLAW_LLM_BASE_URL + API key",
            )

        if browser_page is None:
            return FetchResult(
                paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                fetched_via=self.name, error="needs browser_page",
            )

        base_url, api_key, model = config
        page = browser_page
        context: BrowserContext = page.context

        # Navigate to article
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

        # Get paper title from the collection (passed via paper_id context)
        paper_title = page.title() or doi

        history_lines: list[str] = []

        for turn in range(MAX_TURNS):
            _dismiss_cookie_banners(page)
            candidates = _extract_candidates(page)

            history_str = ""
            if history_lines:
                history_str = "Previous attempts:\n" + "\n".join(history_lines)
            else:
                history_str = "This is your first turn. No previous attempts."

            prompt = USER_TEMPLATE.format(
                doi=doi,
                title=paper_title[:120],
                url=page.url,
                page_title=page.title()[:100],
                history=history_str,
                candidates=_format_candidates(candidates),
            )

            decision = _call_llm(base_url, api_key, model, SYSTEM_PROMPT, prompt)
            if decision is None:
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_ERROR,
                    fetched_via=self.name,
                    error=f"Turn {turn}: LLM failed to respond",
                )

            action = decision.get("action", "").lower()
            reasoning = decision.get("reasoning", "")
            target = decision.get("target")

            log.info(
                "LLM turn %d: action=%s target=%s reasoning=%s",
                turn, action, target, reasoning[:80],
            )

            if action == "give_up":
                return FetchResult(
                    paper_id=paper_id, doi=doi, status=STATUS_NOT_FOUND,
                    fetched_via=self.name,
                    error=f"LLM gave up after {turn} turns: {reasoning}",
                    extra={"llm_reasoning": reasoning, "turns": turn},
                )

            if target is None or not isinstance(target, int):
                history_lines.append(
                    f"- Turn {turn}: invalid response (action={action}, target={target})"
                )
                continue

            if target < 0 or target >= len(candidates):
                history_lines.append(
                    f"- Turn {turn}: target {target} out of range (0-{len(candidates)-1})"
                )
                continue

            chosen = candidates[target]
            href = chosen.get("href", "")
            if not href:
                history_lines.append(f"- Turn {turn}: link [{target}] has no href")
                continue

            # Make absolute
            if href.startswith("/"):
                parts = urlparse(page.url)
                href = f"{parts.scheme}://{parts.netloc}{href}"
            elif not href.startswith("http"):
                href = f"https://{href}"

            if action == "click":
                # Actually CLICK the element in the browser — triggers JS
                history_lines.append(
                    f'- Turn {turn}: CLICKED [{target}] (text={chosen.get("text", "")[:40]})'
                )
                try:
                    # Build a selector that matches this specific element
                    click_sel = None
                    if chosen.get("href"):
                        click_sel = f'{chosen["tag"]}[href="{chosen["href"]}"]'
                    elif chosen.get("text"):
                        click_sel = f'{chosen["tag"]}:has-text("{chosen["text"][:30]}")'

                    if click_sel and page.locator(click_sel).count() > 0:
                        # Try 1: click + expect_download (works when
                        # the click triggers a real file download)
                        download_ok = False
                        try:
                            import tempfile  # noqa: PLC0415
                            with page.expect_download(timeout=15_000) as dl_info:
                                page.locator(click_sel).first.click(timeout=5_000)
                            download = dl_info.value
                            tmpdir = Path(tempfile.mkdtemp(prefix="pdfclaw_llm_"))
                            tmp_pdf = tmpdir / "tmp.pdf"
                            try:
                                download.save_as(str(tmp_pdf))
                                body = tmp_pdf.read_bytes()
                            finally:
                                if tmp_pdf.exists():
                                    tmp_pdf.unlink()
                                tmpdir.rmdir()
                            if b"%PDF-" in body[:1024]:
                                pdf_start = body.index(b"%PDF-")
                                body = body[pdf_start:] if pdf_start > 0 else body
                                download_ok = True
                        except Exception:  # noqa: BLE001
                            # expect_download timed out — the click
                            # probably opened Chrome's PDF viewer
                            # instead of triggering a download.
                            pass

                        if not download_ok:
                            # Try 2: the click may have navigated to a
                            # PDF URL that Chrome opened in its viewer.
                            # Use in-page JavaScript fetch() to grab the
                            # bytes — this has the page's full auth.
                            body = _extract_pdf_from_viewer(page)

                        if body and b"%PDF-" in body[:1024]:
                            pdf_start = body.index(b"%PDF-")
                            body = body[pdf_start:] if pdf_start > 0 else body
                            return FetchResult(
                                paper_id=paper_id, doi=doi, status=STATUS_OK,
                                pdf_bytes=body, fetched_via=self.name,
                                extra={"method": "click+viewer_extract",
                                       "llm_turns": turn + 1,
                                       "n_bytes": len(body)},
                            )
                        history_lines[-1] += f" → click didn't produce PDF (viewer extract also failed)"
                    else:
                        history_lines[-1] += " → selector not found on page"
                except Exception as exc:  # noqa: BLE001
                    history_lines[-1] += f" → click/download failed: {exc}"
                continue

            if action == "navigate":
                history_lines.append(
                    f'- Turn {turn}: NAVIGATED to [{target}] ({href[:80]})'
                )
                try:
                    page.goto(href, wait_until="domcontentloaded", timeout=30_000)
                    page.wait_for_load_state("networkidle", timeout=10_000)
                except Exception as exc:  # noqa: BLE001
                    history_lines[-1] += f" → navigation failed: {exc}"
                else:
                    history_lines[-1] += f" → landed on: {page.url[:80]}"
                continue

            if action == "fetch":
                try:
                    api_resp = context.request.get(
                        href, timeout=30_000,
                        headers={"Referer": page.url},
                    )
                except Exception as exc:  # noqa: BLE001
                    history_lines.append(
                        f"- Turn {turn}: FETCHED [{target}] ({href[:80]}) → network error: {exc}"
                    )
                    continue

                status = api_resp.status
                body = api_resp.body() if api_resp.ok else b""

                if api_resp.ok and body.startswith(b"%PDF-"):
                    return FetchResult(
                        paper_id=paper_id, doi=doi, status=STATUS_OK,
                        pdf_bytes=body, fetched_via=self.name,
                        extra={
                            "href": href,
                            "llm_turns": turn + 1,
                            "llm_reasoning": reasoning,
                            "n_bytes": len(body),
                        },
                    )

                # Describe what happened for the next turn
                if not api_resp.ok:
                    desc = f"HTTP {status}"
                elif b"<html" in body[:1000].lower():
                    # Extract a snippet of visible text from the HTML
                    text_snippet = ""
                    try:
                        text_snippet = re.sub(
                            r"<[^>]+>", " ",
                            body[:2000].decode("utf-8", errors="ignore"),
                        )
                        text_snippet = re.sub(r"\s+", " ", text_snippet).strip()[:200]
                    except Exception:  # noqa: BLE001
                        pass
                    desc = f"got HTML page (not PDF). Page text: '{text_snippet}'"
                else:
                    desc = f"got non-PDF content (first bytes: {body[:16]!r})"

                history_lines.append(
                    f"- Turn {turn}: FETCHED [{target}] ({href[:80]}) → {desc}"
                )
                continue

            # Unknown action
            history_lines.append(f"- Turn {turn}: unknown action '{action}'")

        return FetchResult(
            paper_id=paper_id, doi=doi, status=STATUS_ERROR,
            fetched_via=self.name,
            error=f"Exhausted {MAX_TURNS} turns. History: {'; '.join(history_lines)}",
            extra={"history": history_lines},
        )
