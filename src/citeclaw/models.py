"""Data models and domain exceptions."""

from __future__ import annotations

import enum
import re
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FilterResult(enum.Enum):
    SKIP = "skip"
    REJECT = "reject"
    PENDING_LLM = "pending_llm"


class PaperSource(str, enum.Enum):
    SEED = "seed"
    BACKWARD = "backward"
    FORWARD = "forward"


class LLMVerdict(str, enum.Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    ACCEPT_SEED = "accept_seed"


# ---------------------------------------------------------------------------
# Paper record
# ---------------------------------------------------------------------------

class PaperRecord(BaseModel):
    paper_id: str
    title: str = ""
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    references: list[str] = Field(default_factory=list)
    depth: int = 0
    source: PaperSource = PaperSource.BACKWARD
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
    # Alternate paper IDs that have been folded into this record by the
    # ``MergeDuplicates`` step (e.g. the preprint ID of a peer-reviewed
    # canonical paper). An empty list is the common case.
    aliases: list[str] = Field(default_factory=list)

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# LLM screening result
# ---------------------------------------------------------------------------

class ScreeningResult(BaseModel):
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


_S2_HEX_RE = re.compile(r"^[0-9a-f]{40}$")


def normalize_s2_id(raw: Any) -> str | None:
    """Normalize a Semantic Scholar paper ID."""
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    if _S2_HEX_RE.match(raw):
        return raw
    if any(raw.startswith(prefix) for prefix in ("CorpusId:", "DOI:", "ArXiv:", "PMID:", "MAG:", "ACL:")):
        return raw
    if raw.startswith("10."):
        return f"DOI:{raw}"
    return None


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------

class CiteClawError(Exception):
    """Base exception for the citeclaw package."""


class LLMParseError(CiteClawError):
    """Raised when the LLM returns output that cannot be parsed as JSON."""


class OpenAlexAPIError(CiteClawError):
    """Raised on non-retryable OpenAlex API errors."""


class SemanticScholarAPIError(CiteClawError):
    """Raised on non-retryable Semantic Scholar API errors."""


class BudgetExhaustedError(CiteClawError):
    """Raised when a safety cap (tokens or requests) is reached."""
