"""ExpandBackward step — for each paper in signal, fetch references and screen.

When ``pdf_references=True``, papers whose S2 reference list is empty
(not indexed, too new, or grey literature) get a PDF-based fallback:
the step fetches the paper's full text, extracts all reference titles
via a simple heuristic parser, resolves them through
``ctx.s2.search_match``, and feeds the resolved candidates into the
normal screening pipeline.  This complements the S2 API path without
replacing it — S2 references are always tried first.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block, record_rejections
from citeclaw.models import PaperRecord
from citeclaw.network import saturation_for_paper
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_backward")

# ---------------------------------------------------------------------------
# Lightweight reference-list parser (no LLM, no GROBID)
# ---------------------------------------------------------------------------

_REF_HEADING_RE = re.compile(
    r"^\s*(?:References|Bibliography|Works\s+Cited|Literature\s+Cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Matches common bibliography entry patterns:
#   [1] Author, ...  Title. Journal ...
#   1. Author, ...   Title. Journal ...
_REF_ENTRY_RE = re.compile(
    r"^\s*\[?\d{1,4}\]?\.?\s+",
    re.MULTILINE,
)


def _extract_all_ref_titles(text: str) -> list[str]:
    """Best-effort title extraction from a raw reference list.

    Heuristic: each numbered reference entry starts with ``[N]`` or
    ``N.``.  The title is typically the first sentence-like phrase after
    the author block (which ends with a period or colon after a year).
    We extract a candidate title by taking the text between the author
    block's terminating punctuation and the next period.

    This is intentionally rough — it's a fallback for the rare case
    where S2 has no reference data at all.
    """
    # Find the reference section.
    matches = list(_REF_HEADING_RE.finditer(text))
    if not matches:
        return []
    last = matches[-1]
    if last.start() < len(text) * 0.4:
        return []
    ref_section = text[last.end():]

    # Split into individual entries.
    entries = _REF_ENTRY_RE.split(ref_section)
    titles: list[str] = []
    for entry in entries:
        entry = entry.strip()
        if not entry or len(entry) < 20:
            continue
        title = _guess_title(entry)
        if title:
            titles.append(title)
    return titles


def _guess_title(entry: str) -> str | None:
    """Extract a plausible title from a single bibliography entry.

    Strategy: look for the pattern ``Author(s). Title. Journal/venue``
    and take the second sentence (the title).  Falls back to taking
    everything before the first period that's followed by a venue-like
    token (a capitalised word or a journal abbreviation).
    """
    # Common pattern: "Author, A., Author, B.: Title. Journal ..."
    # or "Author, A., Author, B. Title. Journal ..."
    # Try splitting on ". " and taking the first segment that looks
    # like a title (starts with a capital, > 15 chars, < 300 chars).
    parts = re.split(r"\.\s+", entry)
    for part in parts:
        part = part.strip()
        if len(part) < 15 or len(part) > 300:
            continue
        # Skip parts that look like author lists (contain ", " heavily).
        if part.count(",") > 3 and len(part) < 80:
            continue
        # Skip parts that look like journal names (short, all caps / mixed).
        if len(part) < 30 and part.count(" ") < 3:
            continue
        # Accept the first part that starts with a capital letter.
        if part[0].isupper():
            # Clean up trailing year/volume markers.
            cleaned = re.sub(r"\s*\(\d{4}\).*$", "", part).strip()
            if len(cleaned) >= 15:
                return cleaned
    return None


class ExpandBackward:
    name = "ExpandBackward"

    def __init__(
        self,
        *,
        screener=None,
        pdf_references: bool = False,
        pdf_model: str | None = None,
        headless: bool = True,
        openalex_references: bool = True,
    ) -> None:
        self.screener = screener
        self.pdf_references = pdf_references
        self.pdf_model = pdf_model
        self.headless = headless
        # OpenAlex reference fallback — used when S2 returns empty refs
        # AND the paper has a DOI in external_ids. Cheap (one OpenAlex
        # call + N single-DOI S2 resolves) and strictly improves recall
        # for fresh preprints S2 hasn't fully ingested. Defaults to True
        # because the network cost is small and the payoff is real; set
        # False to disable.
        self.openalex_references = openalex_references

    def run(self, signal: list[PaperRecord], ctx: Any) -> StepResult:
        if self.screener is None:
            return StepResult(signal=[], in_count=len(signal), stats={"reason": "no screener"})

        dash = ctx.dashboard
        dash.enable_outer_bar(total=len(signal), description="source papers")

        accepted: list[PaperRecord] = []
        pdf_fallback_count = 0
        openalex_fallback_count = 0

        for source in signal:
            if source.paper_id in ctx.expanded_backward:
                dash.advance_outer(1)
                continue
            ctx.expanded_backward.add(source.paper_id)

            dash.begin_phase("fetch refs", total=1)
            try:
                ref_records = ctx.s2.fetch_references(source.paper_id)
            except Exception as exc:
                log.warning("backward: failed for %s: %s", source.paper_id[:20], exc)
                dash.advance_outer(1)
                continue
            dash.tick_inner(1)

            # OpenAlex fallback: when S2 returns no references and the
            # paper has a DOI, consult OpenAlex's referenced_works. Tried
            # before the PDF fallback because it's O(refs) network calls
            # (cheap) vs O(PDF fetch + LLM parse) for the PDF path.
            if not ref_records and self.openalex_references:
                oa_refs = self._openalex_fallback(source, ctx)
                if oa_refs:
                    ref_records = oa_refs
                    openalex_fallback_count += 1

            # PDF fallback: when S2 AND OpenAlex both miss and the user
            # opted in, extract reference titles from the paper's PDF.
            if not ref_records and self.pdf_references:
                pdf_refs = self._pdf_fallback(source, ctx)
                if pdf_refs:
                    ref_records = pdf_refs
                    pdf_fallback_count += 1

            source.references = [r.paper_id for r in ref_records if r.paper_id]

            cands: list[PaperRecord] = []
            for r in ref_records:
                if not r.paper_id or r.paper_id in ctx.seen:
                    continue
                ctx.seen.add(r.paper_id)
                r.depth = source.depth + 1
                r.source = "backward"
                r.supporting_papers = [source.paper_id]
                cands.append(r)

            if not cands:
                dash.advance_outer(1)
                continue
            dash.note_candidates_seen(len(cands))

            dash.begin_phase("enrich · abstracts", total=1)
            ctx.s2.enrich_with_abstracts(cands)
            dash.tick_inner(1)

            fctx = FilterContext(ctx=ctx, source=source)
            passed, rejected = apply_block(cands, self.screener, fctx)
            record_rejections(rejected, fctx)
            for p in passed:
                p.llm_verdict = "accept"
                ctx.collection[p.paper_id] = p
                accepted.append(p)
                dash.paper_accepted(p, saturation=saturation_for_paper(p, ctx))

            dash.advance_outer(1)

        stats: dict[str, Any] = {"accepted": len(accepted)}
        if self.pdf_references:
            stats["pdf_fallback_used"] = pdf_fallback_count
        if self.openalex_references:
            stats["openalex_fallback_used"] = openalex_fallback_count
        return StepResult(
            signal=accepted, in_count=len(signal),
            stats=stats,
        )

    def _openalex_fallback(
        self,
        source: PaperRecord,
        ctx: Any,
    ) -> list[PaperRecord]:
        """Fetch references via OpenAlex for a paper S2 has no refs for.

        Only runs when the paper carries a DOI in ``external_ids``.
        OpenAlex's ``referenced_works`` is keyed by DOI (via the
        ``/works/doi:...`` path); we resolve each returned DOI back to
        an S2 ``PaperRecord`` via ``ctx.s2.fetch_metadata("DOI:...")``.
        Failures (network, missing work, non-DOI'd records) are silently
        skipped so a partially-broken OpenAlex response doesn't kill
        the run.
        """
        doi = (source.external_ids or {}).get("DOI")
        if not doi:
            return []
        try:
            from citeclaw.clients.openalex import OpenAlexClient
        except ImportError:  # pragma: no cover
            return []
        client = OpenAlexClient(ctx.config)
        try:
            ref_dois = client.fetch_references_by_doi(doi)
        except Exception as exc:  # noqa: BLE001 — best effort
            log.info("OpenAlex references lookup failed for %s: %s",
                     source.paper_id[:20], exc)
            client.close()
            return []
        finally:
            # ``client.close`` is safe to call twice; ``try/finally``
            # guarantees it runs even on the happy path.
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

        if not ref_dois:
            return []

        records: list[PaperRecord] = []
        for ref_doi in ref_dois:
            try:
                rec = ctx.s2.fetch_metadata(f"DOI:{ref_doi}")
            except Exception:  # noqa: BLE001 — skip unresolvable DOIs
                continue
            if rec is not None:
                records.append(rec)

        log.info(
            "openalex fallback: %d DOIs → %d resolved refs for %s",
            len(ref_dois), len(records), source.paper_id[:20],
        )
        return records

    def _pdf_fallback(
        self,
        source: PaperRecord,
        ctx: Any,
    ) -> list[PaperRecord]:
        """Extract references from the source paper's PDF.

        Returns a list of PaperRecords for each resolved reference.
        Uses the PdfClawBridge for PDF fetching and a lightweight
        heuristic parser for reference extraction (no LLM required).
        """
        from citeclaw.clients.pdfclaw_bridge import PdfClawBridge

        bridge = PdfClawBridge(ctx.cache, headless=self.headless)
        try:
            text = bridge.fetch_text(source)
        finally:
            bridge.close()

        if not text:
            log.debug(
                "pdf_references fallback: no text for %s", source.paper_id[:20],
            )
            return []

        titles = _extract_all_ref_titles(text)
        if not titles:
            return []

        log.info(
            "pdf_references fallback: extracted %d reference titles from %s",
            len(titles), source.paper_id[:20],
        )

        records: list[PaperRecord] = []
        for title in titles:
            try:
                match = ctx.s2.search_match(title)
            except Exception:  # noqa: BLE001
                continue
            if match is None:
                continue
            pid = match.get("paperId")
            if not pid:
                continue
            # Build a lightweight PaperRecord from the match.
            from citeclaw.clients.s2.converters import paper_to_record

            rec = paper_to_record(match)
            if rec:
                records.append(rec)

        log.info(
            "pdf_references fallback: resolved %d / %d titles for %s",
            len(records), len(titles), source.paper_id[:20],
        )
        return records
