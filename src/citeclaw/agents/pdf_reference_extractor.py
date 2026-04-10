"""LLM agent that reads a paper's full text and extracts topic-relevant references.

Public API
----------
- :func:`split_references` — heuristic split of raw PDF text into
  (body, reference_list).
- :func:`extract_pdf_references` — one LLM call → list of
  :class:`ExtractedReference`.

Both are pure functions (no side-effects, no state); the caller
(:class:`~citeclaw.steps.expand_by_pdf.ExpandByPDF`) owns the LLM
client, the cache, and the budget.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from citeclaw.prompts.pdf_extraction import (
    SYSTEM,
    USER_TEMPLATE,
    pdf_extraction_schema,
)

if TYPE_CHECKING:
    from citeclaw.clients.llm.base import LLMClient

log = logging.getLogger("citeclaw.agents.pdf_reference_extractor")

# ---------------------------------------------------------------------------
# Reference-list splitting
# ---------------------------------------------------------------------------

_REF_HEADING_RE = re.compile(
    r"^\s*(?:References|Bibliography|Works\s+Cited|Literature\s+Cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _try_salvage_json(text: str) -> dict | None:
    """Attempt to parse truncated JSON by closing brackets.

    When the LLM hits the token limit mid-output, the JSON is truncated.
    We try progressively closing brackets to recover as many complete
    reference entries as possible.
    """
    if not text or "{" not in text:
        return None
    # Strategy: find the last complete '}' that closes a reference entry,
    # then close the array and root object.
    for suffix in ("]}", "]}"):
        # Find last complete reference entry.
        last_close = text.rfind("}")
        while last_close > 0:
            candidate = text[:last_close + 1] + suffix
            try:
                decoded = json.loads(candidate)
                if isinstance(decoded, dict) and "relevant_references" in decoded:
                    log.info(
                        "Salvaged %d references from truncated JSON",
                        len(decoded["relevant_references"]),
                    )
                    return decoded
            except (json.JSONDecodeError, ValueError):
                pass
            last_close = text.rfind("}", 0, last_close)
    return None


def split_references(text: str) -> tuple[str, str]:
    """Split paper text into ``(body, reference_list)``.

    Finds the **last** occurrence of a references-section heading and
    splits there.  Only trusts the heading if it appears in the latter
    half of the document (avoids false positives from an abstract that
    mentions the word "References").

    Returns ``(full_text, "")`` when no heading is found — the LLM
    then works with inline citations only.
    """
    matches = list(_REF_HEADING_RE.finditer(text))
    if not matches:
        return text, ""
    last = matches[-1]
    # Only trust a heading in the latter 60% of the document.
    # Some papers (e.g. those with long appendices) have the reference
    # section as early as 40% of the total text.
    if last.start() < len(text) * 0.4:
        return text, ""
    body = text[: last.start()].rstrip()
    refs = text[last.end() :].lstrip()
    return body, refs


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Mention:
    """One inline citation of a reference in the paper body."""

    quote: str
    relevance: str = ""


@dataclass
class ExtractedReference:
    """A reference the LLM judged relevant to the topic."""

    citation_marker: str  # e.g. "[23]"
    reference_text: str  # full bib entry
    title: str  # extracted title for S2 search
    mentions: list[Mention] = field(default_factory=list)
    relevance_explanation: str = ""


@dataclass
class PdfExtractionResult:
    """Output of one :func:`extract_pdf_references` call."""

    references: list[ExtractedReference] = field(default_factory=list)
    raw_reasoning: str = ""
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

# Default budget: at ~1.3 tokens/char, 40 K chars ≈ 30 K input tokens.
# With a 128 K context window and ``reasoning_effort: high``, the model
# may use 30–60 K tokens for thinking, leaving plenty for the JSON
# output (~3 K tokens for 10–20 references).  For smaller context
# windows, users can override with ``max_input_chars`` in the YAML step
# config (e.g. 14000 for 32 K context).
_DEFAULT_MAX_INPUT_CHARS = 40_000


def _truncate_for_context(
    body: str,
    refs: str,
    max_chars: int,
) -> tuple[str, str]:
    """Ensure ``body + refs`` fits within *max_chars*.

    Allocation strategy: give the reference list up to 30% of the
    budget (capped at 8 K chars — that's ~120 references, more than
    enough for the LLM to resolve markers). The body gets the
    remainder, truncated from the end — the introduction and
    related-work sections at the beginning carry the most citation
    context.
    """
    # Cap the reference list — papers with huge appendices can have
    # 30–40 K chars after the "References" heading.
    ref_cap = min(max_chars * 3 // 10, 8_000)
    ref_budget = min(len(refs), ref_cap)
    refs_out = refs[:ref_budget]
    if len(refs) > ref_budget:
        refs_out += "\n\n[... reference list truncated ...]"

    body_budget = max_chars - len(refs_out)
    if len(body) > body_budget:
        body_out = body[:body_budget] + "\n\n[... body text truncated ...]"
    else:
        body_out = body
    return body_out, refs_out


def extract_pdf_references(
    full_text: str,
    paper_title: str,
    topic_description: str,
    llm: "LLMClient",
    *,
    max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS,
) -> PdfExtractionResult:
    """Run one LLM call to extract topic-relevant references.

    Parameters
    ----------
    full_text:
        Raw parsed PDF text (body + reference section).
    paper_title:
        Title of the paper being analysed.
    topic_description:
        The user's topic string from ``config.yaml``.
    llm:
        Pre-built :class:`LLMClient` instance.
    max_input_chars:
        Total character budget for the body + reference list inside the
        prompt.  Defaults to 24 000 (~18 K tokens), leaving room for
        system prompt, schema, reasoning, and output in a 32 K context
        window.

    Returns
    -------
    PdfExtractionResult
        Parsed references and metadata.  On LLM failure the result is
        empty (never raises).
    """
    body, refs = split_references(full_text)
    if not refs:
        # No structured reference list found — give the LLM the full
        # text and let it identify references inline.
        refs = "(No separate reference list found — references may be inline.)"

    body, refs = _truncate_for_context(body, refs, max_input_chars)

    user_prompt = USER_TEMPLATE.format(
        topic_description=topic_description,
        reference_list=refs,
        paper_title=paper_title or "(untitled)",
        body_text=body,
    )

    try:
        resp = llm.call(
            SYSTEM,
            user_prompt,
            category="pdf_reference_extraction",
            response_schema=pdf_extraction_schema(),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("PDF extraction LLM call failed: %s", exc)
        return PdfExtractionResult()

    raw_reasoning = getattr(resp, "reasoning_content", "") or ""

    try:
        decoded = json.loads(resp.text)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Try to salvage truncated JSON — the LLM may have hit the
        # token limit mid-output.  Look for complete reference entries.
        decoded = _try_salvage_json(resp.text or "")
        if decoded is None:
            log.warning(
                "PDF extraction: LLM returned invalid JSON (len=%d): %.120s",
                len(resp.text) if resp.text else 0,
                resp.text or "",
            )
            return PdfExtractionResult(raw_reasoning=raw_reasoning)

    if not isinstance(decoded, dict):
        return PdfExtractionResult(raw_reasoning=raw_reasoning)

    extracted: list[ExtractedReference] = []
    for item in decoded.get("relevant_references", []):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue  # Can't resolve without a title.
        mentions = []
        for m in item.get("mentions", []):
            if isinstance(m, dict) and m.get("quote"):
                mentions.append(
                    Mention(
                        quote=str(m["quote"]),
                        relevance=str(m.get("relevance", "")),
                    )
                )
        extracted.append(
            ExtractedReference(
                citation_marker=str(item.get("citation_marker", "")),
                reference_text=str(item.get("reference_text", "")),
                title=title,
                mentions=mentions,
                relevance_explanation=str(item.get("relevance_explanation", "")),
            )
        )

    return PdfExtractionResult(
        references=extracted,
        raw_reasoning=raw_reasoning,
    )
