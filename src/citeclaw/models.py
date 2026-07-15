"""Core data shapes, ID normalizers, and the domain exception hierarchy.

This module is the bottom of the dependency graph for the rest of the
package — every other module imports types from here, so it MUST stay
free of intra-package imports.

Sections, in order:

  * :class:`PaperSource` / :class:`LLMVerdict` — string constants for the
    enum-style fields on :class:`PaperRecord`.
  * :class:`PaperRecord` / :class:`ScreeningResult` — the two pydantic
    models the pipeline threads through every step.
  * :func:`normalize_openalex_id` — strict parser that rejects malformed
    IDs at the boundary instead of letting them sail through to the
    downstream API.
  * Domain exceptions — :class:`CiteClawError` plus subclasses.
    :class:`S2OutageError` deliberately subclasses :class:`BaseException`,
    not :class:`Exception`, so generic ``except Exception`` clauses don't
    swallow the outage signal; the CLI catches it explicitly.
"""

from __future__ import annotations

import enum
import re
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PaperSource:
    """String constants for the ``PaperRecord.source`` field.

    Plain attributes instead of an enum so new expansion modes can
    introduce their own source labels at runtime without a schema
    migration. Existing equality checks (``p.source ==
    PaperSource.BACKWARD``) keep working — the attributes are just
    strings.
    """

    SEED = "seed"
    BACKWARD = "backward"
    FORWARD = "forward"
    SEARCH = "search"
    SEMANTIC = "semantic"
    AUTHOR = "author"
    REINFORCED = "reinforced"
    PDF = "pdf"


class LLMVerdict(str, enum.Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    ACCEPT_SEED = "accept_seed"


# ---------------------------------------------------------------------------
# Paper record
# ---------------------------------------------------------------------------

class PaperRecord(BaseModel):
    """Single-paper record threaded through the entire pipeline.

    The same instance is reused as the unit of work for filters, expansion
    steps, clustering, reranking, dedup, and output. Fields fall into four
    groups: bibliographic identity (``paper_id`` + ``external_ids`` +
    ``aliases``), bibliographic content (``title`` / ``abstract`` /
    ``authors`` / ``venue`` / ``year`` / ``fields_of_study`` /
    ``publication_types``), graph signal (``references`` /
    ``reference_edges`` / ``citation_count`` /
    ``influential_citation_count`` / ``depth``), and pipeline-state
    annotations (``source`` / ``llm_verdict`` / ``llm_reasoning`` /
    ``supporting_papers`` / ``expanded`` / ``pdf_url`` / ``full_text``).
    """

    paper_id: str
    title: str = ""
    abstract: str | None = None
    year: int | None = None
    # ISO-8601 date string (``YYYY-MM-DD``, ``YYYY-MM``, or ``YYYY``) from
    # S2's ``publicationDate`` field.  ``None`` when S2 only knows the
    # year — many records have ``year`` populated but ``publicationDate``
    # missing, so consumers should fall back to ``year`` when needed.
    # Kept as a string rather than ``datetime.date`` so partial dates
    # (year-only, year-month) survive round-trips through JSON and the
    # SQLite cache without precision loss.
    publication_date: str | None = None
    venue: str | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    references: list[str] = Field(default_factory=list)
    depth: int = 0
    # Free-form string label — use :class:`PaperSource` constants for
    # the canonical values. New expansion modes can supply their own
    # string without a model migration.
    source: str = PaperSource.BACKWARD
    llm_verdict: LLMVerdict | None = None
    llm_reasoning: str | None = None
    supporting_papers: list[str] = Field(default_factory=list)
    expanded: bool = False
    pdf_url: str | None = None
    authors: list[dict] = Field(default_factory=list)
    reference_edges: dict[str, dict] = Field(default_factory=dict)
    # Semantic Scholar ``externalIds`` payload, e.g. ``{"DOI": "10.1/abc",
    # "ArXiv": "2301.00001", "PubMed": "12345"}``. Populated from S2 metadata
    # during fetch and consumed by the duplicate-detection step.
    external_ids: dict[str, str] = Field(default_factory=dict)
    # Alternate paper IDs folded into this record by the
    # ``MergeDuplicates`` step (e.g. the preprint ID of a peer-reviewed
    # paper). Empty in the common case.
    aliases: list[str] = Field(default_factory=list)
    # S2 ``fieldsOfStudy`` + ``s2FieldsOfStudy.category`` merged and
    # deduplicated; lets filters gate on subject area without an LLM.
    fields_of_study: list[str] = Field(default_factory=list)
    # S2 ``publicationTypes`` (e.g. ``["JournalArticle", "Review"]``) —
    # a cheap structured signal for distinguishing surveys / methods /
    # editorials.
    publication_types: list[str] = Field(default_factory=list)
    # Parsed body text from the open-access PDF, populated by
    # ``citeclaw.clients.pdf.PdfFetcher.prefetch`` for ``LLMFilter``
    # blocks whose scope is ``full_text``. None for closed-access
    # papers (the filter falls back to abstract content).
    full_text: str | None = None

    model_config = {"use_enum_values": True}

    @property
    def publication_month_ordinal(self) -> int | None:
        """Linear-with-time integer for ranking / colour-mapping.

        Gephi (and many other graph viewers) can't colour-rank by an
        ISO date string like ``"2025-04"`` — it treats the value as a
        category, not a continuum.  This property flattens the date to
        ``year * 12 + month`` so consumers get a single integer that
        increases monotonically with time:

        - ``"2024-04-29"`` → ``24292``  (2024*12 + 4)
        - ``"2024-04"``    → ``24292``  (day ignored)
        - ``"2024"``       → ``24289``  (month=1 fallback)
        - no date, only ``year=2024`` → ``24289``
        - no date and no year → ``None``

        Differences between values preserve the original month-interval
        gap (e.g. 24306 − 24301 = 5 = the months between 2025-01 and
        2025-06), so Gephi's colour gradient lines up with elapsed time
        regardless of the absolute baseline.
        """
        raw = (self.publication_date or "").strip()
        if raw:
            parts = raw.split("-")
            try:
                year = int(parts[0])
            except (ValueError, IndexError):
                year = None
            month = 1
            if len(parts) >= 2:
                try:
                    month = int(parts[1])
                except ValueError:
                    month = 1
            if year is not None:
                return year * 12 + month
        # Fall back to ``year`` when no publication_date — better than
        # ``None`` for the common partial-info case S2 returns.
        if self.year is not None:
            return self.year * 12 + 1
        return None


# ---------------------------------------------------------------------------
# LLM screening result
# ---------------------------------------------------------------------------

class ScreeningResult(BaseModel):
    """One LLM verdict on one paper, as returned by the screener.

    ``id`` echoes the paper id the LLM was asked about (the screener uses
    it to associate verdicts back to records when batching). ``verdict``
    is the raw string from the model — values are constrained to the
    members of :class:`LLMVerdict` by the screener's structured-output
    schema. ``reasoning`` is the model's free-text justification and may
    be empty when the prompt didn't request it. ``confidence`` is
    ``None`` for models that don't return calibrated scores.
    """

    id: str
    verdict: str
    reasoning: str = ""
    confidence: float | None = None


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

_OPENALEX_ID_RE = re.compile(r"^W\d+$")


def normalize_openalex_id(raw: Any) -> str | None:
    """Extract the short ``W\\d+`` form from a URL or bare ID.  Returns None if invalid."""
    if not raw or not isinstance(raw, str):
        return None
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    raw = raw.strip()
    if _OPENALEX_ID_RE.match(raw):
        return raw
    return None


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------

class CiteClawError(Exception):
    """Base exception for the citeclaw package."""


class SemanticScholarAPIError(CiteClawError):
    """Raised on non-retryable Semantic Scholar API errors."""


class S2OutageError(BaseException):
    """Raised when the S2 API has failed N consecutive calls in a row.

    Indicates a sustained outage (rate limit, networking, S2 down) rather
    than a single transient blip — every transient failure is already
    absorbed by tenacity's per-call retry.

    Subclasses :class:`BaseException` (NOT :class:`Exception`) so that the
    many ``except Exception`` clauses in expansion steps and S2 batch
    helpers (which exist to keep one bad paper from crashing the whole
    run) do not swallow this signal. The CLI entry point catches it
    explicitly, finalises the partial run, and exits 1.
    """


class BudgetExhaustedError(CiteClawError):
    """Raised when a safety cap (tokens or requests) is reached."""
