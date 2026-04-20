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

log = logging.getLogger("citeclaw.steps._pdf_reference_extractor")

# ---------------------------------------------------------------------------
# Reference-list splitting
# ---------------------------------------------------------------------------

_REF_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"References?(?:\s+(?:and\s+Notes|Cited))?"
    r"|Bibliography"
    r"|Works\s+Cited"
    r"|Literature\s+Cited"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Fallback: detect numbered bibliography entries (e.g. "1. Author, ...")
# near the end of a document when no heading is found.
_NUMBERED_BIB_RE = re.compile(
    r"(?:^|\n)\s*(?:\d{1,3}[\.\)]\s+[A-Z]|\[\d{1,3}\]\s+[A-Z])",
)
_MIN_BIB_ENTRIES = 5  # need at least this many numbered entries to trust the heuristic


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


def _find_numbered_bib_start(text: str, search_from: float = 0.40) -> int | None:
    """Find the start of a numbered bibliography in the tail of *text*.

    Looks for a cluster of consecutive numbered entries (``1. Author``,
    ``[1] Author``) starting from *search_from* fraction of the document.
    Returns the character offset of the first entry, or ``None`` if
    fewer than :data:`_MIN_BIB_ENTRIES` are found.
    """
    threshold = int(len(text) * search_from)
    tail = text[threshold:]
    hits = list(_NUMBERED_BIB_RE.finditer(tail))
    if len(hits) < _MIN_BIB_ENTRIES:
        return None
    return threshold + hits[0].start()


def _refs_start_number(refs: str) -> int:
    """Return the first reference number found in *refs*, or 0."""
    m = re.match(r"\s*(?:\[?(\d+)[\]\.\)])", refs)
    return int(m.group(1)) if m else 0


def _looks_like_bibliography(text_after: str, max_chars: int = 3000) -> bool:
    """Check if *text_after* looks like a bibliography section.

    Used to validate a candidate reference-section heading when it sits
    earlier than expected (e.g. before the 40 % mark): papers with long
    appendices can push references into the middle of the document.
    Returns ``True`` if the first ``max_chars`` after the heading contain
    either a cluster of numbered entries or several author-year style
    entries (e.g. ``Smith, J., Jones, K.``).
    """
    sample = text_after[:max_chars]
    # Numbered entries: "1. Author..." or "[1] Author..."
    numbered = len(_NUMBERED_BIB_RE.findall(sample))
    if numbered >= _MIN_BIB_ENTRIES:
        return True
    # Author-year entries: lines starting with "Lastname, F." pattern.
    author_entries = len(re.findall(
        r"(?:^|\n)\s*[A-Z][a-zA-Z\-']+,\s+[A-Z]\.?",
        sample,
    ))
    return author_entries >= _MIN_BIB_ENTRIES


def split_references(text: str) -> tuple[str, str]:
    """Split paper text into ``(body, reference_list)``.

    **Strategy 1** — heading match: finds a references-section heading
    (``References``, ``References and Notes``, ``Bibliography``, …).
    Accepts headings at ≥ 40 % of the document, or at ≥ 15 % when the
    content immediately after looks like a bibliography (catches
    Nature-style papers with huge supplementary material where the
    main refs live earlier — e.g. Seurat v4 has refs at 20 % and 80 %
    supplementary after).

    **Strategy 2** — numbered-entry fallback: if no heading is found,
    detects a cluster of numbered bibliography entries (``1. Author``,
    ``[1] Author``) near the end.

    Returns ``(full_text, "")`` only when both strategies fail.
    """
    matches = list(_REF_HEADING_RE.finditer(text))
    # Prefer headings at ≥ 40 %, but also accept ≥ 15 % when followed
    # by bibliography-like content (papers with huge supplementary
    # material push the main reference list into the first third of
    # the document — Seurat v4's refs are at 20 % of the doc).
    for last in reversed(matches):
        pos_fraction = last.start() / max(len(text), 1)
        if pos_fraction >= 0.40:
            pass  # trusted position
        elif pos_fraction >= 0.15 and _looks_like_bibliography(text[last.end():]):
            log.debug(
                "split_references: accepted early heading at %.0f%% after content validation",
                pos_fraction * 100,
            )
        else:
            continue

        body = text[: last.start()].rstrip()
        refs = text[last.end() :].lstrip()
        # Sanity check: if the refs start at a suspiciously high
        # number (e.g. ≥ 10), we likely found a supplementary-
        # materials heading instead of the main bibliography.
        start_num = _refs_start_number(refs)
        if start_num >= 10:
            earlier = _find_numbered_bib_start(text, search_from=0.35)
            if earlier is not None and earlier < last.start():
                log.debug(
                    "split_references: heading split started at ref #%d; "
                    "found earlier bibliography at char %d",
                    start_num, earlier,
                )
                body = text[:earlier].rstrip()
                refs = text[earlier:].lstrip()
        return body, refs

    # Fallback: detect numbered bibliography entries near the end.
    bib_start = _find_numbered_bib_start(text)
    if bib_start is not None:
        body = text[:bib_start].rstrip()
        refs = text[bib_start:].lstrip()
        log.debug("split_references: no heading found, used numbered-entry fallback at char %d", bib_start)
        return body, refs

    return text, ""


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

# Default budget: at ~1.3 tokens/char, 80 K chars ≈ 60 K input tokens.
# With a 128 K context window and ``reasoning_effort: high`` the model
# typically spends 5–15 K tokens on thinking for extraction tasks
# (empirically measured on Gemma 4 31B over dozens of bio papers; the
# output JSON fits in ~3–5 K tokens), leaving a comfortable margin
# for long papers. Prior default was 40 K, which cut off the training-
# data section of Nature-style papers (body + main refs + methods +
# extended refs concatenated > 40 K chars), causing low recall on
# foundational papers like AlphaFold / ESM. For smaller context
# windows, users can override with ``max_input_chars`` in the YAML
# step config (e.g. 14 000 for 32 K context).
_DEFAULT_MAX_INPUT_CHARS = 80_000


def _truncate_body_middle_out(body: str, budget: int) -> str:
    """Truncate *body* from the middle, keeping head and tail.

    For reference extraction the most valuable regions are:

    * **Head** — introduction and related-work sections contain the
      densest citation context.
    * **Tail** — often contains the reference list when heading
      detection failed, plus late-paper discussion that references
      prior work.

    The middle (methods, detailed results) is the least important for
    identifying which references exist and why they were cited.
    """
    if len(body) <= budget:
        return body
    # 60 % head, 40 % tail (intro + related-work tend to be longer
    # than the trailing discussion/references).
    head_budget = int(budget * 0.6)
    tail_budget = budget - head_budget
    marker = "\n\n[... middle of paper truncated ...]\n\n"
    head_budget -= len(marker) // 2
    tail_budget -= len(marker) // 2
    return body[:head_budget] + marker + body[-tail_budget:]


def _truncate_for_context(
    body: str,
    refs: str,
    max_chars: int,
) -> tuple[str, str]:
    """Ensure ``body + refs`` fits within *max_chars*.

    When a reference section was successfully split off, it gets
    priority: up to 40 % of the budget.  The body is truncated from
    the end — the introduction and related-work at the top carry the
    most citation context.

    When ``refs`` is empty or just a placeholder (split failed), the
    body is truncated from the **middle** instead, preserving both
    the head (intro / related work) and the tail (likely containing
    the unsplit reference list).
    """
    refs_is_placeholder = not refs or refs.startswith("(")
    ref_cap = max_chars * 2 // 5  # 40% of total budget
    ref_budget = min(len(refs), ref_cap)
    refs_out = refs[:ref_budget]
    if len(refs) > ref_budget:
        refs_out += "\n\n[... reference list truncated ...]"

    body_budget = max_chars - len(refs_out)
    if len(body) <= body_budget:
        return body, refs_out

    if refs_is_placeholder:
        # Split failed — the reference list is buried at the end of body.
        # Truncate from the middle to preserve both head and tail.
        body_out = _truncate_body_middle_out(body, body_budget)
    else:
        # Normal case — refs split succeeded; trim body from the end.
        body_out = body[:body_budget] + "\n\n[... body text truncated ...]"
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

    # Clean up response text: strip markdown code fences that the model
    # may wrap around the JSON when structured output is not enforced.
    resp_text = (resp.text or "").strip()
    if resp_text.startswith("```"):
        # Remove opening fence (```json or ```) and closing fence
        lines = resp_text.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        resp_text = "\n".join(lines).strip()

    try:
        decoded = json.loads(resp_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Try to salvage truncated JSON — the LLM may have hit the
        # token limit mid-output.  Look for complete reference entries.
        decoded = _try_salvage_json(resp_text)
        if decoded is None:
            log.warning(
                "PDF extraction: LLM returned invalid JSON (len=%d): %.120s",
                len(resp_text),
                resp_text,
            )
            return PdfExtractionResult(raw_reasoning=raw_reasoning)

    if not isinstance(decoded, dict):
        return PdfExtractionResult(raw_reasoning=raw_reasoning)

    extracted: list[ExtractedReference] = []
    for item in decoded.get("relevant_references", []):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        # Empty title is tolerated — some bibliography styles emit
        # ``Author, Journal Vol, Pages (Year)`` with no distinct title
        # (SignalP 5.0, old Nature papers). The S2 resolution step
        # (``ExpandByPDF._resolve_reference``) will fall back to the
        # full ``reference_text`` when the title is empty, so the
        # agent's extraction still carries useful information.
        mentions = []
        for m in item.get("mentions", []):
            if isinstance(m, dict) and m.get("quote"):
                mentions.append(
                    Mention(
                        quote=str(m["quote"]),
                        relevance=str(m.get("relevance", "")),
                    )
                )
        # Tolerate schema-variant field names the model may use when
        # structured output is bypassed (e.g. "reference" vs
        # "reference_text", "explanation" vs "relevance_explanation").
        ref_text = (
            item.get("reference_text")
            or item.get("reference")
            or item.get("ref_text")
            or ""
        )
        rel_expl = (
            item.get("relevance_explanation")
            or item.get("explanation")
            or item.get("relevance")
            or ""
        )
        extracted.append(
            ExtractedReference(
                citation_marker=str(item.get("citation_marker", "")),
                reference_text=str(ref_text),
                title=title,
                mentions=mentions,
                relevance_explanation=str(rel_expl),
            )
        )

    return PdfExtractionResult(
        references=extracted,
        raw_reasoning=raw_reasoning,
    )
