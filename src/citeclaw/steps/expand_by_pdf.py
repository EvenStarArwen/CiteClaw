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

When chained **after** ``ExpandBackward`` in a pipeline, every paper
this step accepts is — by construction — a *new* candidate that the
citation-graph traversal missed, because ``ctx.seen`` dedups across
all expand steps.  The step's ``accepted`` count therefore directly
measures the marginal recall uplift contributed by PDF reading;
``run_pipeline`` surfaces this in the structured log as
``additional_via_pdf_augmentation``.

Per-paper work (fetch → LLM extract → S2 resolve) runs concurrently
through a :class:`ThreadPoolExecutor` sized by ``max_workers`` so the
slow blocking I/O (PDF download, GROBID parse, LLM completion) overlaps
across papers.  Browser access inside :class:`PdfClawBridge` is
serialised via an internal lock — only the publisher-recipe path that
needs Chrome runs one paper at a time.

YAML example::

    - step: ExpandByPDF
      screener: forward_screener     # optional
      model: gemma-4-31b             # optional
      reasoning_effort: high         # optional
      max_papers: 20                 # cap papers to read (default: all)
      max_input_chars: 80000         # per-paper text budget (128K-ctx default)
      headless: true                 # browser mode for pdfclaw
      parser: grobid                 # pdfclaw.parsers engine; default pymupdf
      parser_kwargs:                 # engine-specific kwargs (optional)
        do_ocr: false
      max_workers: 4                 # concurrent papers in flight (default: 4)
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from citeclaw.steps._pdf_reference_extractor import (
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


@dataclass
class _PaperOutcome:
    """Per-paper return value from one worker.

    Kept thread-local — accumulation into the step's shared state
    happens single-threaded on the main thread as futures complete.
    """

    source_id: str
    papers_read: int = 0       # 1 when the PDF parsed AND LLM ran
    papers_skipped: int = 0    # 1 when fetch_text returned None
    # ``(paper_id, provenance_dict)`` tuples — one entry per resolved ref.
    hits: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raw_reasoning: str = ""
    n_refs_extracted: int = 0  # before S2 resolution (raw LLM output count)


class ExpandByPDF:
    """Expand the collection by reading full-text PDFs with an LLM."""

    name = "ExpandByPDF"

    def __init__(
        self,
        *,
        screener: Any = None,
        topic_description: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = "medium",
        max_papers: int | None = None,
        max_input_chars: int = 80_000,
        headless: bool = True,
        parser: str = "pymupdf",
        parser_kwargs: dict | None = None,
        max_workers: int = 4,
    ) -> None:
        self.screener = screener
        self.topic_description = topic_description
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_papers = max_papers
        self.max_input_chars = max_input_chars
        self.headless = headless
        # Engine name from :mod:`pdfclaw.parsers`.  Default ``"pymupdf"``
        # keeps fast-path runs unchanged; production runs that care
        # about extraction quality set ``parser: docling`` (or
        # ``"grobid"``) in the YAML step config.
        self.parser = parser
        self.parser_kwargs = parser_kwargs or {}
        # ``max_workers`` sizes the per-paper ThreadPoolExecutor.  Each
        # worker does its own fetch → LLM → S2-resolve.  HTTP fetch +
        # GROBID/LLM calls are I/O-bound so threads (not processes) are
        # the right tool.  Browser access inside the shared
        # :class:`PdfClawBridge` is locked, so the "one Chrome window"
        # constraint is preserved.  ``max_workers=1`` keeps the legacy
        # sequential code path for debugging.
        self.max_workers = max(1, int(max_workers))

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
            parser=self.parser,
            parser_kwargs=self.parser_kwargs,
        )

        dash = ctx.dashboard
        papers_to_read = signal[: self.max_papers] if self.max_papers else signal
        dash.enable_outer_bar(total=len(papers_to_read), description="reading PDFs")

        # ----- per-paper extraction ----------------------------------------
        # ``raw_hits`` is the list fed to ``screen_expand_candidates``:
        # ``[{paperId: str}, ...]``.  ``provenance`` is keyed by target
        # paperId → list of mention dicts (used to build edge_meta after
        # screening).  Both are accumulated on the main thread as worker
        # futures complete — no locks needed because each future returns
        # a self-contained :class:`_PaperOutcome`.
        raw_hits: list[dict[str, Any]] = []
        provenance: dict[str, list[dict[str, Any]]] = {}
        papers_read = 0
        papers_skipped = 0
        total_refs_extracted = 0

        try:
            with ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="pdf-worker",
            ) as pool:
                futures = {
                    pool.submit(
                        self._process_one, source, bridge, llm, ctx, topic,
                    ): source
                    for source in papers_to_read
                }
                for fut in as_completed(futures):
                    source = futures[fut]
                    try:
                        outcome = fut.result()
                    except Exception as exc:  # noqa: BLE001 — never let one paper kill the run
                        log.warning(
                            "ExpandByPDF: worker for %s raised: %s",
                            source.paper_id[:20], exc,
                        )
                        papers_skipped += 1
                        dash.advance_outer(1)
                        continue

                    papers_read += outcome.papers_read
                    papers_skipped += outcome.papers_skipped
                    for pid, prov in outcome.hits:
                        raw_hits.append({"paperId": pid})
                        provenance.setdefault(pid, []).append(prov)
                        total_refs_extracted += 1

                    if outcome.raw_reasoning:
                        # Log the LLM reasoning trace at DEBUG so when the
                        # downstream verdict looks wrong, the operator can
                        # grep the run log and see *what* the model
                        # actually thought, not just the final JSON.
                        log.debug(
                            "ExpandByPDF reasoning for %s: %.500s",
                            outcome.source_id[:20], outcome.raw_reasoning,
                        )

                    dash.advance_outer(1)

        finally:
            bridge.close()

        if not raw_hits:
            ctx.searched_signals.add(fp)
            log.info(
                "ExpandByPDF: read=%d skipped=%d refs_extracted=0 "
                "additional_via_pdf_augmentation=0",
                papers_read, papers_skipped,
            )
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={
                    "papers_read": papers_read,
                    "papers_skipped": papers_skipped,
                    "refs_extracted": 0,
                    "accepted": 0,
                    "additional_via_pdf_augmentation": 0,
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

        # ``len(accepted)`` IS the marginal uplift the PDF reader gave
        # us: ``screen_expand_candidates`` only returns papers that are
        # new to ``ctx.seen`` (and pass the screener), so each accepted
        # paper is one that the prior citation-graph traversal missed.
        # Surfacing this under an explicit name makes the benefit easy
        # to read off the log without doing math against other stats.
        n_additional = len(accepted)
        log.info(
            "ExpandByPDF: read=%d skipped=%d refs_extracted=%d "
            "additional_via_pdf_augmentation=%d",
            papers_read, papers_skipped, total_refs_extracted, n_additional,
        )

        stats: dict[str, Any] = {
            "papers_read": papers_read,
            "papers_skipped": papers_skipped,
            "refs_extracted": total_refs_extracted,
            "additional_via_pdf_augmentation": n_additional,
            **base_stats,
        }
        return StepResult(
            signal=accepted,
            in_count=len(signal),
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Worker body
    # ------------------------------------------------------------------

    def _process_one(
        self,
        source: PaperRecord,
        bridge: PdfClawBridge,
        llm: Any,
        ctx: Any,
        topic: str,
    ) -> _PaperOutcome:
        """End-to-end work for a single source paper.

        Runs entirely on a worker thread:

          1. ``bridge.fetch_text`` (HTTP + lazy browser, browser-locked)
          2. ``extract_pdf_references`` (one LLM call, thread-safe client)
          3. ``_resolve_reference`` for each LLM-emitted reference
             (S2 search_match — internally rate-limited).

        Returns a self-contained :class:`_PaperOutcome` — no shared state
        is mutated here.  The main thread merges outcomes as futures
        complete.
        """
        text = bridge.fetch_text(source)
        if not text:
            return _PaperOutcome(source_id=source.paper_id, papers_skipped=1)

        result = extract_pdf_references(
            text,
            source.title,
            topic,
            llm,
            max_input_chars=self.max_input_chars,
        )

        outcome = _PaperOutcome(
            source_id=source.paper_id,
            papers_read=1,
            raw_reasoning=result.raw_reasoning,
            n_refs_extracted=len(result.references),
        )
        if not result.references:
            return outcome

        for ref in result.references:
            resolved = self._resolve_reference(ref, ctx)
            if resolved is None:
                continue
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
            outcome.hits.append((resolved, prov))
        return outcome

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _text_chars_for_fetch(self) -> int:
        """PDF text budget — generous for fetching, the LLM agent truncates internally."""
        return max(self.max_input_chars * 4, 80_000)

    # Strict DOI pattern matching :mod:`citeclaw.models`. ``\S+`` is the
    # suffix; we anchor with word boundaries so a trailing "." or ")"
    # from the reference text doesn't poison the match.
    _DOI_EXTRACT_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;)\]]+)")

    @classmethod
    def _extract_doi(cls, ref: ExtractedReference) -> str | None:
        """Pull a DOI out of the reference text (or title) if one is present.

        Checks both ``reference_text`` and ``title`` — some extractors
        put the DOI in the title field when the reference is fragmentary.
        Strips common trailing punctuation (``.``, ``)``, etc.) that
        aren't part of a DOI but often appear right after one in
        bibliography entries.
        """
        for source in (ref.reference_text, ref.title):
            if not source:
                continue
            m = cls._DOI_EXTRACT_RE.search(source)
            if m:
                doi = m.group(1).rstrip(".,;)]")
                if doi:
                    return doi
        return None

    @classmethod
    def _resolve_reference(
        cls,
        ref: ExtractedReference,
        ctx: Any,
    ) -> str | None:
        """Resolve an extracted reference to an S2 paperId.

        Cascade:

          1. Regex-extract a DOI from ``reference_text`` / ``title``.
             When found, try ``s2.fetch_metadata("DOI:<doi>")``. DOI
             lookups are structurally reliable — a cleanly-extracted
             DOI resolves the paper unambiguously, even if the title
             is missing or mangled.
          2. Fall back to ``s2.search_match`` on the title (≥ 8 chars)
             or the full reference text (≥ 12 chars). This is the
             historical path; useful when the reference has no DOI.

        Returns ``None`` when both paths fail — the reference is
        silently dropped (logged at DEBUG).
        """
        # Path 1: DOI-first.
        doi = cls._extract_doi(ref)
        if doi:
            try:
                record = ctx.s2.fetch_metadata(f"DOI:{doi}")
            except Exception as exc:  # noqa: BLE001
                log.debug("DOI lookup failed for %r: %s", doi, exc)
            else:
                pid = getattr(record, "paper_id", None)
                if pid:
                    return pid

        # Path 2: title/text search_match (existing behaviour).
        title = ref.title.strip()
        query = title if len(title) >= 8 else ""
        if not query:
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
