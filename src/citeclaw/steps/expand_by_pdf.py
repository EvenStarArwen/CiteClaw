"""ExpandByPDF — discover papers by reading full-text PDFs with an LLM.

For each paper in the input signal the step:

  1. Fetches the full PDF text (cache → HTTP → pdfclaw browser recipes).
  2. Calls an LLM agent that reads the paper with the user's
     ``topic_description`` and returns a structured list of references
     that are relevant to the topic, each accompanied by verbatim
     quotes from the paper body.
  3. Resolves every extracted reference title to an S2 ``paperId`` via
     ``search_match``.
  4. Optionally screens the resolved candidates through the normal
     filter pipeline (``screen_expand_candidates``).
  5. Stores provenance metadata (quotes, relevance explanation) as edge
     attributes on the (source → discovered) citation edge.

Unlike ``ExpandForward`` / ``ExpandBackward`` where the citation graph
itself provides candidates, this step uses the *content* of the paper
to discover relevant prior work.  The LLM acts as a domain-aware reader
that can surface references a purely structural approach would miss —
papers mentioned in context as "the leading method" or "the dataset used
for pretraining" that may not share citation overlap with the source.

The ``screener`` is **optional**: when omitted, all resolved references
are accepted directly.  Users can add any combination of filter blocks
(or none) in their YAML config.

YAML example::

    - step: ExpandByPDF
      screener: forward_screener     # optional
      model: gemma-4-31b             # optional
      reasoning_effort: high         # optional
      max_papers: 20                 # cap papers to read (default: all)
      max_input_chars: 80000         # per-paper text budget (128K-ctx default)
      headless: true                 # browser mode for pdfclaw
"""

from __future__ import annotations

import logging
from typing import Any

from citeclaw.agents.pdf_reference_extractor import (
    ExtractedReference,
    extract_pdf_references,
)
from citeclaw.clients.llm.factory import build_llm_client
from citeclaw.clients.pdfclaw_bridge import PdfClawBridge
from citeclaw.filters.base import FilterContext
from citeclaw.filters.runner import apply_block, record_rejections
from citeclaw.models import PaperRecord
from citeclaw.steps._expand_helpers import (
    check_already_searched,
    fingerprint_signal,
    screen_expand_candidates,
)
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_by_pdf")


class ExpandByPDF:
    """Expand the collection by reading full-text PDFs with an LLM."""

    name = "ExpandByPDF"

    def __init__(
        self,
        *,
        screener: Any = None,
        topic_description: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = "high",
        max_papers: int | None = None,
        max_input_chars: int = 80_000,
        headless: bool = True,
    ) -> None:
        self.screener = screener
        self.topic_description = topic_description
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_papers = max_papers
        self.max_input_chars = max_input_chars
        self.headless = headless

    def run(self, signal: list[PaperRecord], ctx: Any) -> StepResult:
        # ----- idempotency ------------------------------------------------
        topic = (
            self.topic_description
            or getattr(ctx.config, "topic_description", "")
            or ""
        )
        fp = fingerprint_signal(
            self.name,
            signal,
            topic=topic,
            model=self.model or "",
            max_papers=self.max_papers or 0,
        )
        if (skip := check_already_searched(self.name, fp, ctx, len(signal))):
            return skip

        # ----- LLM client -------------------------------------------------
        llm = build_llm_client(
            ctx.config,
            ctx.budget,
            model=self.model or ctx.config.screening_model,
            reasoning_effort=self.reasoning_effort,
            cache=getattr(ctx, "cache", None),
        )

        # ----- PDF bridge --------------------------------------------------
        bridge = PdfClawBridge(
            ctx.cache,
            max_text_chars=self._text_chars_for_fetch(),
            headless=self.headless,
        )

        dash = ctx.dashboard
        papers_to_read = signal[: self.max_papers] if self.max_papers else signal
        dash.enable_outer_bar(total=len(papers_to_read), description="reading PDFs")

        # ----- per-paper extraction ----------------------------------------
        # raw_hits for screen_expand_candidates: [{paperId: str}, ...]
        raw_hits: list[dict[str, Any]] = []
        # provenance keyed by target paperId → list of mention dicts
        provenance: dict[str, list[dict[str, Any]]] = {}
        papers_read = 0
        papers_skipped = 0
        total_refs_extracted = 0

        try:
            for source in papers_to_read:
                dash.begin_phase("fetch PDF", total=1)
                text = bridge.fetch_text(source)
                dash.tick_inner(1)
                if not text:
                    papers_skipped += 1
                    dash.advance_outer(1)
                    continue

                dash.begin_phase("LLM extraction", total=1)
                result = extract_pdf_references(
                    text,
                    source.title,
                    topic,
                    llm,
                    max_input_chars=self.max_input_chars,
                )
                dash.tick_inner(1)
                papers_read += 1

                if not result.references:
                    dash.advance_outer(1)
                    continue

                # Resolve each extracted reference to an S2 paperId.
                dash.begin_phase(
                    "resolve refs",
                    total=len(result.references),
                )
                for ref in result.references:
                    resolved = self._resolve_reference(ref, ctx)
                    dash.tick_inner(1)
                    if resolved is None:
                        continue

                    pid = resolved
                    raw_hits.append({"paperId": pid})
                    total_refs_extracted += 1

                    # Build provenance record for this edge.
                    mentions = [
                        {"quote": m.quote, "relevance": m.relevance}
                        for m in ref.mentions
                    ]
                    prov = {
                        "source_paper_id": source.paper_id,
                        "citation_marker": ref.citation_marker,
                        "reference_text": ref.reference_text,
                        "relevance_explanation": ref.relevance_explanation,
                        "mentions": mentions,
                    }
                    provenance.setdefault(pid, []).append(prov)

                dash.advance_outer(1)

                # Polite delay between papers (browser-based fetches).
                bridge.sleep()

        finally:
            bridge.close()

        if not raw_hits:
            ctx.searched_signals.add(fp)
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={
                    "papers_read": papers_read,
                    "papers_skipped": papers_skipped,
                    "refs_extracted": 0,
                    "accepted": 0,
                },
            )

        # ----- screening / direct accept ----------------------------------
        if self.screener is not None:
            screened = screen_expand_candidates(
                raw_hits=raw_hits,
                source_label="pdf",
                screener=self.screener,
                ctx=ctx,
            )
            accepted = screened.passed
            base_stats = screened.base_stats
        else:
            # No screener — hydrate, dedup, accept all.
            accepted, base_stats = self._accept_all(raw_hits, ctx)

        # ----- provenance → edge_meta --------------------------------------
        for paper in accepted:
            prov_list = provenance.get(paper.paper_id, [])
            for prov in prov_list:
                src_id = prov["source_paper_id"]
                edge_key = (src_id, paper.paper_id)
                existing = ctx.edge_meta.get(edge_key, {})
                # Merge quotes into contexts.
                quotes = [
                    m["quote"]
                    for m in prov.get("mentions", [])
                    if m.get("quote")
                ]
                existing_contexts = existing.get("contexts", [])
                existing_contexts.extend(quotes)
                existing["contexts"] = existing_contexts
                # Store PDF-specific provenance.
                existing.setdefault("intents", []).append("pdf_extraction")
                existing["pdf_relevance"] = prov.get("relevance_explanation", "")
                existing["pdf_citation_marker"] = prov.get("citation_marker", "")
                ctx.edge_meta[edge_key] = existing

        # ----- done -------------------------------------------------------
        ctx.searched_signals.add(fp)

        stats: dict[str, Any] = {
            "papers_read": papers_read,
            "papers_skipped": papers_skipped,
            "refs_extracted": total_refs_extracted,
            **base_stats,
        }
        return StepResult(
            signal=accepted,
            in_count=len(signal),
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _text_chars_for_fetch(self) -> int:
        """PDF text budget — generous for fetching, the LLM agent truncates internally."""
        return max(self.max_input_chars * 4, 80_000)

    @staticmethod
    def _resolve_reference(
        ref: ExtractedReference,
        ctx: Any,
    ) -> str | None:
        """Resolve an extracted reference to an S2 paperId.

        Uses ``search_match`` with the title when available, falling
        back to the full ``reference_text`` when the bibliography
        style doesn't expose a distinct title (e.g. SignalP 5.0's
        ``Author, Journal Vol, Pages (Year)`` entries). Returns
        ``None`` if both paths fail — the reference is silently
        dropped (logged at DEBUG).
        """
        title = ref.title.strip()
        query = title if len(title) >= 8 else ""
        if not query:
            # Fall back to the full bibliography entry. S2's
            # search_match is text-based and can often recover a
            # paper from ``Author, Journal Vol, Pages (Year)``.
            query = ref.reference_text.strip()
        if not query or len(query) < 12:
            return None
        try:
            match = ctx.s2.search_match(query)
        except Exception as exc:  # noqa: BLE001
            log.debug("S2 search_match failed for %r: %s", query[:60], exc)
            return None
        if match is None:
            log.debug("S2 search_match: no result for %r", query[:60])
            return None
        pid = match.get("paperId")
        if not pid:
            return None
        return pid

    @staticmethod
    def _accept_all(
        raw_hits: list[dict[str, Any]],
        ctx: Any,
    ) -> tuple[list[PaperRecord], dict[str, Any]]:
        """Hydrate, dedup, and accept all candidates (no screener path)."""
        candidates = [
            {"paper_id": h["paperId"]}
            for h in raw_hits
            if isinstance(h.get("paperId"), str)
        ]
        # Deduplicate candidate list by paper_id.
        seen_pids: set[str] = set()
        deduped: list[dict[str, str]] = []
        for c in candidates:
            if c["paper_id"] not in seen_pids:
                seen_pids.add(c["paper_id"])
                deduped.append(c)

        hydrated: list[PaperRecord] = (
            ctx.s2.enrich_batch(deduped) if deduped else []
        )
        if hydrated:
            try:
                ctx.s2.enrich_with_abstracts(hydrated)
            except Exception:  # noqa: BLE001
                pass

        accepted: list[PaperRecord] = []
        for rec in hydrated:
            if not rec.paper_id or rec.paper_id in ctx.seen:
                continue
            rec.source = "pdf"
            rec.llm_verdict = "accept"
            ctx.seen.add(rec.paper_id)
            ctx.collection[rec.paper_id] = rec
            accepted.append(rec)

        return accepted, {
            "raw_hits": len(raw_hits),
            "hydrated": len(hydrated),
            "novel": len(accepted),
            "accepted": len(accepted),
            "rejected": 0,
        }
