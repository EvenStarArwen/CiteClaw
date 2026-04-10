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
You are an agent that downloads PDFs of academic papers from publisher websites.

You can take ONE action per turn:
- {"action": "fetch", "target": N, "reasoning": "..."} — HTTP GET link N's URL and check if it returns a PDF
- {"action": "navigate", "target": N, "reasoning": "..."} — go to link N's page to see more links (use when you think the link leads to a reader/viewer page, not a direct PDF)
- {"action": "give_up", "reasoning": "..."} — the PDF is genuinely unavailable (server error, broken page, etc.)

Important:
- FETCH tries to download the URL directly. If it returns a PDF file, we're done.
- NAVIGATE opens the page in the browser so you can see its links in the next turn.
- Choose FETCH for links that look like direct PDF URLs (contain .pdf, /pdf/, pdfdirect, etc.)
- Choose NAVIGATE for links that look like they open a reader/viewer (contain /reader/, /view/, stamp.jsp, etc.)
- If a previous FETCH returned an error (403, 404, 500) or HTML instead of PDF, reason about WHY and try a different approach.
- If the publisher's server is genuinely broken (e.g. "Failed to load PDF document"), GIVE UP — it's not your fault.
- Always pick the MAIN article PDF, not supplementary material or figures.

Respond with ONLY a JSON object, no other text."""

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
            model or "gemini-2.0-flash-lite",
        )

    # Option 3: Modal Gemma (CITECLAW_VLLM_* vars)
    vllm_key = os.environ.get("CITECLAW_VLLM_API_KEY") or ""
    vllm_base = os.environ.get("PDFCLAW_LLM_BASE_URL") or os.environ.get("CITECLAW_VLLM_BASE_URL") or ""
    if vllm_key and vllm_base:
        return vllm_base, vllm_key, model or "google/gemma-4-31B-it"

    return None


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
                    api_resp = context.request.get(href, timeout=30_000)
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
