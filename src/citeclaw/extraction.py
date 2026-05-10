"""Generic LLM extraction: paper text + instruction → structured JSON.

A small, single-call wrapper around :class:`LLMClient` for the use case
"give me a piece of information from this paper, formatted as JSON."

The reference-extraction layer in
:mod:`citeclaw.steps._pdf_reference_extractor` is a specialisation of
this — it provides its own system prompt, user template, and JSON
schema, then post-processes the output dict into typed dataclasses.
The same pattern works for any other extraction task: dataset names,
model architectures, evaluation metrics, …; the caller chooses how
strict to be by supplying (or omitting) a JSON schema.

Public API
----------
- :func:`extract_from_text` — one LLM call, parsed JSON output.
- :class:`ExtractionResult` — return value with parsed output, raw
  text, reasoning trace, and a failure flag.

The function is pure: no I/O, no caching, no budget tracking beyond
what the supplied :class:`LLMClient` already does.  Caching is opt-in
at the client level (wrap the client in
:class:`citeclaw.clients.llm.caching.CachingLLMClient` if you want
identical-prompt deduplication).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citeclaw.clients.llm.base import LLMClient

log = logging.getLogger("citeclaw.extraction")

# Default per-call text budget. At ~1.3 tokens/char this is ~60K tokens
# of paper body, leaving comfortable margin in a 128K-context model
# (Grok 4.1 Fast, Gemini 3, Gemma 4). Long appendices and supplementary
# material get truncated middle-out so the head (intro / methods) and
# the tail (results / refs) survive — see :func:`_truncate_middle_out`.
DEFAULT_MAX_INPUT_CHARS = 80_000

DEFAULT_SYSTEM = (
    "You are an expert academic literature analyst. The user will give "
    "you the text of a research paper and an instruction asking for "
    "specific information from it. Read the paper carefully, then return "
    "the requested information as a single JSON object.\n\n"
    "Rules:\n"
    "1. Output ONLY a JSON object — no markdown fences, no commentary, "
    "no preamble.\n"
    "2. Quote verbatim where possible. Do not paraphrase the paper's "
    "claims, do not invent values that aren't in the text.\n"
    "3. If the requested information is not present in the paper, "
    "return an empty/null value for that field rather than guessing.\n"
    "4. When the instruction implies a list or table, return an array; "
    "when it implies a single value, return the bare value or a small "
    "object — let the instruction's shape drive the output shape."
)

DEFAULT_USER_TEMPLATE = (
    "## Instruction\n{instruction}\n\n"
    "## Paper{title_clause}\n{text}\n\n"
    "## Output\n"
    "Return a JSON object only. No markdown fences, no commentary."
)


@dataclass
class ExtractionResult:
    """Result of one :func:`extract_from_text` call."""

    output: Any = None
    """Parsed JSON value (dict / list / scalar) or ``None`` on parse failure."""

    raw_text: str = ""
    """Raw assistant text from the LLM, post fence-stripping."""

    raw_reasoning: str = ""
    """Reasoning trace, when the provider exposes one."""

    extraction_failed: bool = False
    """True when the LLM call raised, returned empty, or emitted unparseable JSON."""

    error: str = ""
    """Short failure description when :attr:`extraction_failed` is True."""


def extract_from_text(
    text: str = "",
    instruction: str = "",
    *,
    llm: "LLMClient",
    schema: dict | None = None,
    paper_title: str = "",
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    system_prompt: str | None = None,
    user_template: str | None = None,
    user_prompt: str | None = None,
    category: str = "generic_extraction",
) -> ExtractionResult:
    """Extract structured information from a paper's text via an LLM.

    Parameters
    ----------
    text:
        The paper body (or any text). Truncated middle-out when over
        ``max_input_chars`` so that head and tail are both preserved.
        Ignored when ``user_prompt`` is provided.
    instruction:
        Free-form instruction telling the LLM what to extract. The
        clearer the schema-implied shape ("list of dataset names",
        "single integer", "object with keys X, Y, Z"), the more
        consistent the output. Ignored when ``user_prompt`` is provided.
    llm:
        Pre-built :class:`LLMClient`.
    schema:
        Optional JSON schema. When supplied, providers that support
        structured output enforce it (OpenAI ``response_format``,
        Gemini ``response_schema``, vLLM via xgrammar). When omitted,
        the LLM emits free-form JSON guided by the instruction; the
        caller relies on the prompt for shape.
    paper_title:
        Optional — included in the user prompt for context. Empty
        string omits the title clause entirely.
    max_input_chars:
        Truncation budget for the paper text portion of the prompt.
    system_prompt, user_template:
        Override the defaults to specialise this extractor for a
        specific task. The user template is ``str.format``-rendered with
        ``instruction``, ``text``, and ``title_clause`` keys.
    user_prompt:
        Pre-rendered user-message content. When provided, replaces the
        ``instruction`` + ``text`` template path entirely (no auto
        truncation, no template rendering). Use this when the caller
        needs full control over prompt structure (e.g. the reference
        extractor in :mod:`citeclaw.steps._pdf_reference_extractor`
        which puts the reference list before the body).
    category:
        Budget-tracker bucket key (passed through to ``llm.call``).

    Returns
    -------
    ExtractionResult
        Always returned (never raises). On any failure, ``output`` is
        ``None``, ``extraction_failed`` is True, and ``error`` carries
        a one-line description.
    """
    sys_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM

    if user_prompt is not None:
        user = user_prompt
    else:
        text = _truncate_middle_out(text, max_input_chars)
        template = user_template if user_template is not None else DEFAULT_USER_TEMPLATE
        title_clause = f"\nTitle: {paper_title}" if paper_title else ""
        user = template.format(
            instruction=instruction,
            text=text,
            title_clause=title_clause,
        )

    try:
        resp = llm.call(
            sys_prompt,
            user,
            category=category,
            response_schema=schema,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("extract_from_text: LLM call failed: %s", exc)
        return ExtractionResult(extraction_failed=True, error=f"LLM error: {exc}")

    raw_text = (resp.text or "").strip()
    raw_reasoning = getattr(resp, "reasoning_content", "") or ""

    if not raw_text:
        return ExtractionResult(
            raw_text=raw_text,
            raw_reasoning=raw_reasoning,
            extraction_failed=True,
            error="empty LLM response",
        )

    cleaned = _strip_markdown_fences(raw_text)

    try:
        decoded = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        salvaged = _try_salvage_json(cleaned)
        if salvaged is None:
            log.warning(
                "extract_from_text: invalid JSON (len=%d): %.120s",
                len(cleaned), cleaned,
            )
            return ExtractionResult(
                raw_text=raw_text,
                raw_reasoning=raw_reasoning,
                extraction_failed=True,
                error="invalid JSON",
            )
        decoded = salvaged

    return ExtractionResult(
        output=decoded,
        raw_text=raw_text,
        raw_reasoning=raw_reasoning,
    )


def _truncate_middle_out(text: str, budget: int) -> str:
    """Keep head (60%) and tail (40%) when ``text`` exceeds ``budget``.

    For paper extraction the most valuable regions are the head
    (abstract / intro / related work) and the tail (results / discussion
    / references). The middle (long methods, dense supplementary tables)
    is the least informative for most extraction tasks.
    """
    if len(text) <= budget:
        return text
    head_share = int(budget * 0.6)
    tail_share = budget - head_share
    marker = "\n\n[... middle of paper truncated ...]\n\n"
    head_share -= len(marker) // 2
    tail_share -= len(marker) // 2
    return text[:head_share] + marker + text[-tail_share:]


def _strip_markdown_fences(text: str) -> str:
    """Drop ```...``` fences if the model wrapped the JSON in them."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _try_salvage_json(text: str) -> Any | None:
    """Try to recover from JSON that was truncated by token-limit cutoff.

    Heuristic: walk the closing brackets right-to-left and try a few
    common terminator combinations. Returns the first parse that
    succeeds, or ``None``.
    """
    if not text or ("{" not in text and "[" not in text):
        return None
    candidates_terminators = ("", "]", "}", "]}", "}]", "\"]}", "\"}]", "}}")
    last_close = max(text.rfind("}"), text.rfind("]"))
    while last_close > 0:
        prefix = text[: last_close + 1]
        for term in candidates_terminators:
            try:
                return json.loads(prefix + term)
            except (json.JSONDecodeError, ValueError):
                continue
        next_close = max(
            text.rfind("}", 0, last_close),
            text.rfind("]", 0, last_close),
        )
        if next_close == last_close or next_close <= 0:
            break
        last_close = next_close
    return None
