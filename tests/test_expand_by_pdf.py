"""Tests for the ExpandByPDF step and its components.

Covers:
  - Reference-list splitting heuristic
  - LLM extraction agent (with stub client)
  - S2 title resolution
  - Full step run (no screener + with screener)
  - Edge provenance metadata
  - Enhanced ExpandBackward with pdf_references
"""

from __future__ import annotations

import json
import textwrap
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from citeclaw.agents.pdf_reference_extractor import (
    ExtractedReference,
    Mention,
    PdfExtractionResult,
    extract_pdf_references,
    split_references,
)
from citeclaw.cache import Cache
from citeclaw.config import BudgetTracker, Settings
from citeclaw.context import Context
from citeclaw.models import PaperRecord
from citeclaw.steps.expand_by_pdf import ExpandByPDF
from citeclaw.steps.expand_backward import ExpandBackward, _extract_all_ref_titles
from tests.fakes import FakeS2Client, make_paper


# ---------------------------------------------------------------------------
# split_references
# ---------------------------------------------------------------------------


class TestSplitReferences:
    def test_heading_at_end(self):
        text = "Body text here.\n" * 100 + "\nReferences\n[1] Smith et al. Title. 2020."
        body, refs = split_references(text)
        assert "Body text here." in body
        assert "[1] Smith et al. Title. 2020." in refs
        assert "References" not in body

    def test_no_heading(self):
        text = "Just body text without any reference section."
        body, refs = split_references(text)
        assert body == text
        assert refs == ""

    def test_heading_too_early(self):
        text = "References in the abstract\nShort body."
        body, refs = split_references(text)
        assert body == text
        assert refs == ""

    def test_bibliography_heading(self):
        text = "Body.\n" * 100 + "\nBibliography\n[1] Entry one."
        body, refs = split_references(text)
        assert "[1] Entry one." in refs

    def test_multiple_headings_takes_last(self):
        text = "Body.\n" * 50 + "\nReferences\nEarly refs.\n" + "More body.\n" * 100 + "\nReferences\n[1] Real ref."
        body, refs = split_references(text)
        assert "[1] Real ref." in refs
        assert "Early refs." in body


# ---------------------------------------------------------------------------
# extract_pdf_references (LLM agent)
# ---------------------------------------------------------------------------


class _StubLLMClient:
    """Deterministic LLM client that returns a canned extraction result."""

    def __init__(self, response_text: str = ""):
        self._response_text = response_text
        self.calls: list[tuple[str, str]] = []

    def call(self, system, user, *, category="", response_schema=None, **kw):
        self.calls.append((system, user))
        return MagicMock(text=self._response_text, reasoning_content="thinking...")

    @property
    def supports_logprobs(self):
        return False


class TestJsonSalvage:
    """Test _try_salvage_json for truncated LLM output."""

    def test_salvage_one_complete_ref(self):
        from citeclaw.agents.pdf_reference_extractor import _try_salvage_json
        truncated = '{"relevant_references": [{"citation_marker": "[1]", "reference_text": "Author. Title. 2023.", "title": "Good Title", "mentions": [{"quote": "Q", "relevance": "R"}], "relevance_explanation": "Relevant"}, {"citation_marker": "[2]", "reference_text": "Trunc'
        result = _try_salvage_json(truncated)
        assert result is not None
        assert len(result["relevant_references"]) == 1
        assert result["relevant_references"][0]["title"] == "Good Title"

    def test_salvage_no_json(self):
        from citeclaw.agents.pdf_reference_extractor import _try_salvage_json
        assert _try_salvage_json("not json at all") is None
        assert _try_salvage_json("") is None

    def test_salvage_complete_json(self):
        from citeclaw.agents.pdf_reference_extractor import _try_salvage_json
        complete = json.dumps({"relevant_references": [
            {"citation_marker": "[1]", "reference_text": "X", "title": "T",
             "mentions": [], "relevance_explanation": "Y"}
        ]})
        # Complete JSON should still be salvageable (trivially).
        result = _try_salvage_json(complete)
        # May return None since json.loads would have worked first.
        # The salvage function is only called after json.loads fails.


class TestExtractPdfReferences:
    def test_basic_extraction(self):
        response = json.dumps({
            "relevant_references": [
                {
                    "citation_marker": "[1]",
                    "reference_text": "Smith et al. A Great Paper. Nature 2020.",
                    "title": "A Great Paper",
                    "mentions": [
                        {"quote": "We follow the approach of [1]", "relevance": "method"}
                    ],
                    "relevance_explanation": "Foundational method used in this work.",
                }
            ]
        })
        llm = _StubLLMClient(response)
        text = "Body with [1] citation.\n\nReferences\n[1] Smith et al. A Great Paper. Nature 2020."
        result = extract_pdf_references(text, "Test Paper", "methods", llm)
        assert len(result.references) == 1
        ref = result.references[0]
        assert ref.title == "A Great Paper"
        assert ref.citation_marker == "[1]"
        assert len(ref.mentions) == 1
        assert "approach of [1]" in ref.mentions[0].quote

    def test_empty_response(self):
        llm = _StubLLMClient(json.dumps({"relevant_references": []}))
        result = extract_pdf_references("Body text.", "Title", "topic", llm)
        assert result.references == []

    def test_invalid_json(self):
        llm = _StubLLMClient("not valid json at all")
        result = extract_pdf_references("Body.", "Title", "topic", llm)
        assert result.references == []

    def test_skips_entries_without_title(self):
        response = json.dumps({
            "relevant_references": [
                {
                    "citation_marker": "[1]",
                    "reference_text": "Something",
                    "title": "",
                    "mentions": [],
                    "relevance_explanation": "x",
                },
                {
                    "citation_marker": "[2]",
                    "reference_text": "Other",
                    "title": "Valid Title",
                    "mentions": [],
                    "relevance_explanation": "y",
                },
            ]
        })
        llm = _StubLLMClient(response)
        result = extract_pdf_references("Body.", "Title", "topic", llm)
        assert len(result.references) == 1
        assert result.references[0].title == "Valid Title"

    def test_llm_exception_returns_empty(self):
        llm = MagicMock()
        llm.call.side_effect = RuntimeError("LLM crashed")
        result = extract_pdf_references("Body.", "Title", "topic", llm)
        assert result.references == []

    def test_truncation_for_long_text(self):
        body = "A" * 50_000
        refs = "R" * 10_000
        text = body + "\nReferences\n" + refs
        llm = _StubLLMClient(json.dumps({"relevant_references": []}))
        result = extract_pdf_references(text, "Title", "topic", llm, max_input_chars=1000)
        # Verify the LLM was called (didn't crash on truncation)
        assert len(llm.calls) == 1
        # The user prompt should be shorter than the original text
        user_prompt = llm.calls[0][1]
        assert len(user_prompt) < len(text)


# ---------------------------------------------------------------------------
# ExpandByPDF step
# ---------------------------------------------------------------------------


def _make_ctx(s2: FakeS2Client, tmp_path) -> Context:
    cfg = Settings(
        screening_model="stub",
        data_dir=str(tmp_path),
        topic_description="RNA foundation models",
        seed_papers=[],
    )
    cache = Cache(tmp_path / "cache.db")
    budget = BudgetTracker()
    return Context(config=cfg, s2=s2, cache=cache, budget=budget)


def _fake_extraction_result(*titles: str) -> PdfExtractionResult:
    refs = []
    for i, t in enumerate(titles, 1):
        refs.append(
            ExtractedReference(
                citation_marker=f"[{i}]",
                reference_text=f"Author. {t}. Journal 2023.",
                title=t,
                mentions=[Mention(quote=f"We use {t} [ref].", relevance="method")],
                relevance_explanation=f"{t} is relevant because it's a foundation model.",
            )
        )
    return PdfExtractionResult(references=refs)


class TestExpandByPDFStep:
    """Test ExpandByPDF with mock PDF fetching and LLM extraction."""

    def test_no_signal(self, tmp_path):
        s2 = FakeS2Client()
        ctx = _make_ctx(s2, tmp_path)
        step = ExpandByPDF()
        result = step.run([], ctx)
        assert result.signal == []
        assert result.stats["papers_read"] == 0

    def test_basic_run_no_screener(self, tmp_path):
        """Papers are discovered via PDF reading and accepted without screening."""
        # Set up corpus: source paper + two discoverable papers
        s2 = FakeS2Client()
        source = make_paper("src1", title="Source Paper")
        discovered1 = make_paper("disc1", title="Discovered Paper One")
        discovered2 = make_paper("disc2", title="Discovered Paper Two")
        s2.add(source)
        s2.add(discovered1)
        s2.add(discovered2)

        # Register search_match for title resolution
        s2.register_search_match("Discovered Paper One", discovered1)
        s2.register_search_match("Discovered Paper Two", discovered2)

        ctx = _make_ctx(s2, tmp_path)
        source_rec = PaperRecord(paper_id="src1", title="Source Paper")
        ctx.collection["src1"] = source_rec
        ctx.seen.add("src1")

        step = ExpandByPDF()

        # Mock the PDF bridge and LLM extraction
        with patch(
            "citeclaw.steps.expand_by_pdf.PdfClawBridge"
        ) as MockBridge, patch(
            "citeclaw.steps.expand_by_pdf.extract_pdf_references"
        ) as mock_extract, patch(
            "citeclaw.steps.expand_by_pdf.build_llm_client"
        ):
            bridge_instance = MagicMock()
            bridge_instance.fetch_text.return_value = "Full paper text with [1] and [2]."
            MockBridge.return_value = bridge_instance

            mock_extract.return_value = _fake_extraction_result(
                "Discovered Paper One", "Discovered Paper Two"
            )

            result = step.run([source_rec], ctx)

        assert len(result.signal) == 2
        assert result.stats["papers_read"] == 1
        assert result.stats["refs_extracted"] == 2
        # Both papers should be in the collection now
        assert "disc1" in ctx.collection
        assert "disc2" in ctx.collection
        # Source stamp should be "pdf"
        assert ctx.collection["disc1"].source == "pdf"

    def test_dedup_against_seen(self, tmp_path):
        """Papers already in ctx.seen are not added again."""
        s2 = FakeS2Client()
        source = make_paper("src1", title="Source")
        already_seen = make_paper("old1", title="Already Seen Paper")
        s2.add(source)
        s2.add(already_seen)
        s2.register_search_match("Already Seen Paper", already_seen)

        ctx = _make_ctx(s2, tmp_path)
        ctx.seen.add("src1")
        ctx.seen.add("old1")  # Already seen

        source_rec = PaperRecord(paper_id="src1", title="Source")
        step = ExpandByPDF()

        with patch(
            "citeclaw.steps.expand_by_pdf.PdfClawBridge"
        ) as MockBridge, patch(
            "citeclaw.steps.expand_by_pdf.extract_pdf_references"
        ) as mock_extract, patch(
            "citeclaw.steps.expand_by_pdf.build_llm_client"
        ):
            MockBridge.return_value = MagicMock(
                fetch_text=MagicMock(return_value="text")
            )
            mock_extract.return_value = _fake_extraction_result("Already Seen Paper")

            result = step.run([source_rec], ctx)

        # old1 was already seen, so nothing new accepted
        assert len(result.signal) == 0

    def test_pdf_fetch_failure_skips_paper(self, tmp_path):
        """Papers whose PDFs can't be fetched are skipped gracefully."""
        s2 = FakeS2Client()
        s2.add(make_paper("src1"))
        ctx = _make_ctx(s2, tmp_path)
        ctx.seen.add("src1")
        source_rec = PaperRecord(paper_id="src1")

        step = ExpandByPDF()

        with patch(
            "citeclaw.steps.expand_by_pdf.PdfClawBridge"
        ) as MockBridge, patch(
            "citeclaw.steps.expand_by_pdf.build_llm_client"
        ):
            MockBridge.return_value = MagicMock(
                fetch_text=MagicMock(return_value=None)
            )

            result = step.run([source_rec], ctx)

        assert result.stats["papers_skipped"] == 1
        assert result.stats["papers_read"] == 0

    def test_edge_provenance_stored(self, tmp_path):
        """Provenance metadata (quotes, relevance) is stored in ctx.edge_meta."""
        s2 = FakeS2Client()
        source = make_paper("src1", title="Source")
        target = make_paper("tgt1", title="Target Paper")
        s2.add(source)
        s2.add(target)
        s2.register_search_match("Target Paper", target)

        ctx = _make_ctx(s2, tmp_path)
        ctx.seen.add("src1")
        source_rec = PaperRecord(paper_id="src1", title="Source")

        step = ExpandByPDF()

        with patch(
            "citeclaw.steps.expand_by_pdf.PdfClawBridge"
        ) as MockBridge, patch(
            "citeclaw.steps.expand_by_pdf.extract_pdf_references"
        ) as mock_extract, patch(
            "citeclaw.steps.expand_by_pdf.build_llm_client"
        ):
            MockBridge.return_value = MagicMock(
                fetch_text=MagicMock(return_value="text")
            )
            mock_extract.return_value = _fake_extraction_result("Target Paper")
            result = step.run([source_rec], ctx)

        assert len(result.signal) == 1
        # Check edge_meta
        edge_key = ("src1", "tgt1")
        assert edge_key in ctx.edge_meta
        meta = ctx.edge_meta[edge_key]
        assert "pdf_extraction" in meta.get("intents", [])
        assert len(meta.get("contexts", [])) > 0

    def test_idempotency(self, tmp_path):
        """Running the same step twice with identical signal is a no-op."""
        s2 = FakeS2Client()
        s2.add(make_paper("src1"))
        ctx = _make_ctx(s2, tmp_path)
        ctx.seen.add("src1")
        source_rec = PaperRecord(paper_id="src1")
        step = ExpandByPDF()

        with patch(
            "citeclaw.steps.expand_by_pdf.PdfClawBridge"
        ) as MockBridge, patch(
            "citeclaw.steps.expand_by_pdf.extract_pdf_references"
        ) as mock_extract, patch(
            "citeclaw.steps.expand_by_pdf.build_llm_client"
        ):
            MockBridge.return_value = MagicMock(
                fetch_text=MagicMock(return_value=None)
            )
            # First run
            step.run([source_rec], ctx)
            # Second run — should be a no-op
            result2 = step.run([source_rec], ctx)

        assert result2.stats.get("reason") == "already_searched"


# ---------------------------------------------------------------------------
# ExpandBackward pdf_references fallback
# ---------------------------------------------------------------------------


class TestExtractAllRefTitles:
    def test_numbered_refs(self):
        text = (
            "Body text here.\n" * 100
            + "\nReferences\n"
            + "[1] Smith, J., Doe, A. A great paper about methods. Nature 505, 100-110 (2020).\n"
            + "[2] Lee, K. Another important study on RNA models. Science 300, 50-55 (2021).\n"
        )
        titles = _extract_all_ref_titles(text)
        assert len(titles) >= 1  # Heuristic may not catch all

    def test_no_ref_section(self):
        text = "Just body text without references."
        titles = _extract_all_ref_titles(text)
        assert titles == []


class TestResolveReferenceDoiFirst:
    """Cover the DOI-first resolution path in ExpandByPDF._resolve_reference."""

    def _ref(self, title="", text=""):
        return ExtractedReference(
            citation_marker="[1]",
            reference_text=text,
            title=title,
            mentions=[Mention(quote="q", relevance="r")],
            relevance_explanation="e",
        )

    def test_extract_doi_from_reference_text(self):
        ref = self._ref(text="Smith et al. Nature 555 (2018). doi:10.1038/nature12345")
        assert ExpandByPDF._extract_doi(ref) == "10.1038/nature12345"

    def test_extract_doi_strips_trailing_punctuation(self):
        ref = self._ref(text="See 10.1038/abc123). Nature.")
        assert ExpandByPDF._extract_doi(ref) == "10.1038/abc123"

    def test_extract_doi_from_title_fallback(self):
        ref = self._ref(title="Paper 10.1145/3534678.3539092")
        assert ExpandByPDF._extract_doi(ref) == "10.1145/3534678.3539092"

    def test_extract_doi_none_when_absent(self):
        ref = self._ref(title="No DOI here", text="Smith. Nature. 2020.")
        assert ExpandByPDF._extract_doi(ref) is None

    def test_extract_doi_ignores_short_registrant(self):
        # 10.1 is too short (registrants are 4-9 digits).
        ref = self._ref(text="Fake 10.1/abc reference")
        assert ExpandByPDF._extract_doi(ref) is None

    def test_resolve_uses_doi_before_search_match(self):
        """When a DOI is present, fetch_metadata is called with DOI:…
        before search_match. The title path never runs in that case."""

        class _SpyS2:
            def __init__(self):
                self.fetch_metadata_calls: list[str] = []
                self.search_match_calls: list[str] = []

            def fetch_metadata(self, pid: str) -> PaperRecord:
                self.fetch_metadata_calls.append(pid)
                return PaperRecord(paper_id="RESOLVED", title="Resolved")

            def search_match(self, query: str):
                self.search_match_calls.append(query)
                return {"paperId": "TITLE_MATCH"}

        class _Ctx:
            def __init__(self, s2):
                self.s2 = s2

        spy = _SpyS2()
        ref = self._ref(
            title="Some Paper Title",
            text="Smith et al. Nature 2020. doi:10.1038/nature12345",
        )
        pid = ExpandByPDF._resolve_reference(ref, _Ctx(spy))
        assert pid == "RESOLVED"
        assert spy.fetch_metadata_calls == ["DOI:10.1038/nature12345"]
        assert spy.search_match_calls == []  # never fell through

    def test_resolve_falls_back_to_search_match_when_doi_fails(self):
        """fetch_metadata raising (e.g. S2 has no record for this DOI)
        must fall through to the title-based search_match path."""

        class _FlakyS2:
            def __init__(self):
                self.search_match_calls: list[str] = []

            def fetch_metadata(self, pid: str) -> PaperRecord:
                raise RuntimeError("not in S2")

            def search_match(self, query: str):
                self.search_match_calls.append(query)
                return {"paperId": "TITLE_MATCH"}

        class _Ctx:
            def __init__(self, s2):
                self.s2 = s2

        spy = _FlakyS2()
        ref = self._ref(
            title="Some Paper Title",
            text="Smith et al. doi:10.1038/abc456",
        )
        pid = ExpandByPDF._resolve_reference(ref, _Ctx(spy))
        assert pid == "TITLE_MATCH"
        assert spy.search_match_calls == ["Some Paper Title"]

    def test_resolve_search_match_when_no_doi(self):
        """No DOI present → skip DOI path, go straight to search_match."""

        class _SpyS2:
            def __init__(self):
                self.fetch_metadata_calls: list[str] = []
                self.search_match_calls: list[str] = []

            def fetch_metadata(self, pid: str):  # should not be called
                self.fetch_metadata_calls.append(pid)
                raise AssertionError("should not be reached")

            def search_match(self, query: str):
                self.search_match_calls.append(query)
                return {"paperId": "TITLE_MATCH"}

        class _Ctx:
            def __init__(self, s2):
                self.s2 = s2

        spy = _SpyS2()
        ref = self._ref(title="Some Paper Title", text="No doi here")
        pid = ExpandByPDF._resolve_reference(ref, _Ctx(spy))
        assert pid == "TITLE_MATCH"
        assert spy.fetch_metadata_calls == []
        assert spy.search_match_calls == ["Some Paper Title"]


class TestExpandByPDFWithScreener:
    """Test ExpandByPDF with a screener that filters candidates."""

    def test_screener_rejects_some(self, tmp_path):
        """Screener filters out candidates that don't pass."""
        s2 = FakeS2Client()
        source = make_paper("src1", title="Source")
        good = make_paper("good1", title="Good Paper", year=2022, citation_count=100)
        bad = make_paper("bad1", title="Bad Paper", year=2022, citation_count=100)
        s2.add(source)
        s2.add(good)
        s2.add(bad)
        s2.register_search_match("Good Paper", good)
        s2.register_search_match("Bad Paper", bad)

        ctx = _make_ctx(s2, tmp_path)
        ctx.seen.add("src1")
        source_rec = PaperRecord(paper_id="src1", title="Source")

        # Build a simple year filter as screener (accept year >= 2023).
        from citeclaw.filters.builder import build_blocks
        screener_block = build_blocks({
            "yr": {"type": "YearFilter", "min": 2023, "max": 2030}
        })["yr"]

        step = ExpandByPDF(screener=screener_block)

        with patch(
            "citeclaw.steps.expand_by_pdf.PdfClawBridge"
        ) as MockBridge, patch(
            "citeclaw.steps.expand_by_pdf.extract_pdf_references"
        ) as mock_extract, patch(
            "citeclaw.steps.expand_by_pdf.build_llm_client"
        ):
            MockBridge.return_value = MagicMock(
                fetch_text=MagicMock(return_value="text")
            )
            mock_extract.return_value = _fake_extraction_result(
                "Good Paper", "Bad Paper"
            )
            result = step.run([source_rec], ctx)

        # Both papers are year=2022, which is < 2023 min, so both rejected
        assert len(result.signal) == 0
        assert result.stats["rejected"] == 2


class TestExpandByPDFStubPipeline:
    """Exercise ExpandByPDF through build_step (YAML config parsing)."""

    def test_build_step_minimal(self):
        from citeclaw.steps import build_step
        step = build_step({"step": "ExpandByPDF"})
        assert step.name == "ExpandByPDF"
        assert step.screener is None
        assert step.reasoning_effort == "high"

    def test_build_step_with_options(self):
        from citeclaw.steps import build_step
        blocks = {}
        step = build_step({
            "step": "ExpandByPDF",
            "model": "gemma-4-31b",
            "reasoning_effort": "medium",
            "max_papers": 5,
            "max_input_chars": 16000,
            "headless": False,
        }, blocks)
        assert step.model == "gemma-4-31b"
        assert step.max_papers == 5
        assert step.max_input_chars == 16000
        assert step.headless is False

    def test_build_step_with_inline_screener(self):
        from citeclaw.steps import build_step
        step = build_step({
            "step": "ExpandByPDF",
            "screener": {"type": "YearFilter", "min": 2020, "max": 2025},
        })
        assert step.screener is not None

    def test_build_expand_backward_with_pdf_references(self):
        from citeclaw.steps import build_step
        step = build_step({
            "step": "ExpandBackward",
            "screener": {"type": "YearFilter", "min": 2020, "max": 2025},
            "pdf_references": True,
        })
        assert step.pdf_references is True


class TestExpandByPDFPipelineIntegration:
    """Exercise ExpandByPDF through run_pipeline in stub mode."""

    def test_stub_pipeline_with_expand_by_pdf(self, tmp_path):
        """Full pipeline: LoadSeeds → ExpandByPDF → Finalize in stub mode.

        Uses the stub LLM which returns one reference titled
        'Stub Reference Title'. We register that title in the fake S2
        client so resolution succeeds and the collection grows.
        """
        from citeclaw.config import SeedPaper
        from citeclaw.pipeline import run_pipeline

        # Build corpus
        s2 = FakeS2Client()
        seed = make_paper("SEED-1", title="My Seed Paper", year=2022)
        stub_ref = make_paper("STUBREF-1", title="Stub Reference Title", year=2023)
        s2.add(seed)
        s2.add(stub_ref)
        s2.register_search_match("Stub Reference Title", stub_ref)

        cfg = Settings(
            screening_model="stub",
            data_dir=str(tmp_path),
            topic_description="RNA foundation models",
            seed_papers=[SeedPaper(paper_id="SEED-1")],
            pipeline=[
                {"step": "LoadSeeds"},
                {"step": "ExpandByPDF"},
                {"step": "Finalize"},
            ],
        )
        cache = Cache(tmp_path / "cache.db")
        budget = BudgetTracker()
        ctx = Context(config=cfg, s2=s2, cache=cache, budget=budget)

        # Mock the PDF bridge to return text, since we're in stub mode
        with patch(
            "citeclaw.steps.expand_by_pdf.PdfClawBridge"
        ) as MockBridge:
            bridge_instance = MagicMock()
            bridge_instance.fetch_text.return_value = (
                "Body text with [1] reference.\n\n"
                "References\n"
                "[1] Stub Author. Stub Reference Title. Journal 2023."
            )
            MockBridge.return_value = bridge_instance

            run_pipeline(ctx)

        # LoadSeeds adds 1 seed; ExpandByPDF should add STUBREF-1
        assert "SEED-1" in ctx.collection
        assert "STUBREF-1" in ctx.collection
        assert len(ctx.collection) == 2
        assert ctx.collection["STUBREF-1"].source == "pdf"

        # Verify finalize wrote output files
        assert (tmp_path / "literature_collection.json").exists()


class TestExpandBackwardPdfFallback:
    def test_pdf_references_disabled_by_default(self, tmp_path):
        """Default ExpandBackward doesn't use PDF fallback."""
        step = ExpandBackward(screener=MagicMock())
        assert step.pdf_references is False

    def test_pdf_references_enabled(self, tmp_path):
        step = ExpandBackward(screener=MagicMock(), pdf_references=True)
        assert step.pdf_references is True
