"""Tests for :mod:`citeclaw.models` — PaperRecord, ID normalizers, exceptions."""

from __future__ import annotations

import pytest

from citeclaw.models import (
    BudgetExhaustedError,
    CiteClawError,
    FilterResult,
    LLMParseError,
    LLMVerdict,
    OpenAlexAPIError,
    PaperRecord,
    PaperSource,
    ScreeningResult,
    SemanticScholarAPIError,
    normalize_openalex_id,
    normalize_s2_id,
)


class TestPaperRecord:
    def test_defaults(self):
        p = PaperRecord(paper_id="x")
        assert p.paper_id == "x"
        assert p.title == ""
        assert p.abstract is None
        assert p.year is None
        assert p.venue is None
        assert p.citation_count is None
        assert p.references == []
        assert p.depth == 0
        assert p.source == PaperSource.BACKWARD
        assert p.llm_verdict is None
        assert p.supporting_papers == []
        assert p.expanded is False
        # Dedup-related fields default to empty containers.
        assert p.external_ids == {}
        assert p.aliases == []

    def test_external_ids_and_aliases(self):
        p = PaperRecord(
            paper_id="SSHex",
            external_ids={"DOI": "10.1/abc", "ArXiv": "2301.00001"},
            aliases=["ArXiv:2301.00001"],
        )
        assert p.external_ids["DOI"] == "10.1/abc"
        assert p.external_ids["ArXiv"] == "2301.00001"
        assert p.aliases == ["ArXiv:2301.00001"]

    def test_full_construction(self):
        p = PaperRecord(
            paper_id="DOI:10.1/abc",
            title="Hello",
            abstract="World",
            year=2024,
            venue="Nature",
            citation_count=12,
            influential_citation_count=2,
            references=["R1", "R2"],
            depth=3,
            source="forward",
            supporting_papers=["S1"],
            pdf_url="http://example.com/p.pdf",
            authors=[{"authorId": "A1", "name": "Alice"}],
        )
        assert p.year == 2024
        assert p.references == ["R1", "R2"]
        # ``use_enum_values=True`` coerces assigned strings but *not* the
        # defaulted enum; assigning a string value works either way.
        p.source = "seed"
        assert p.source == "seed"

    def test_model_dump_stringifies_enums(self):
        p = PaperRecord(paper_id="x")
        dumped = p.model_dump()
        # When the enum default is in place the raw enum appears — but once we
        # assign an enum value explicitly it serializes to its string form.
        p.source = PaperSource.FORWARD.value
        p.llm_verdict = LLMVerdict.ACCEPT.value
        dumped = p.model_dump()
        assert dumped["source"] == "forward"
        assert dumped["llm_verdict"] == "accept"


class TestNormalizeS2Id:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"),
            ("DOI:10.1/abc", "DOI:10.1/abc"),
            ("CorpusId:12345", "CorpusId:12345"),
            ("ArXiv:2201.00001", "ArXiv:2201.00001"),
            ("PMID:999", "PMID:999"),
            ("MAG:1", "MAG:1"),
            ("ACL:p21", "ACL:p21"),
            ("10.1234/foo", "DOI:10.1234/foo"),
        ],
    )
    def test_valid(self, raw, expected):
        assert normalize_s2_id(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", 42, {"x": 1}, "garbage"])
    def test_invalid(self, raw):
        assert normalize_s2_id(raw) is None

    def test_whitespace_stripped(self):
        assert normalize_s2_id("  DOI:10.1/abc  ") == "DOI:10.1/abc"


class TestNormalizeOpenAlexId:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("W123456", "W123456"),
            ("https://openalex.org/W999", "W999"),
            (" W42 ", "W42"),
        ],
    )
    def test_valid(self, raw, expected):
        assert normalize_openalex_id(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "W", "Wabc", 123, "X123"])
    def test_invalid(self, raw):
        assert normalize_openalex_id(raw) is None


class TestScreeningResult:
    def test_defaults(self):
        r = ScreeningResult(id="x", verdict="accept")
        assert r.id == "x"
        assert r.verdict == "accept"
        assert r.reasoning == ""
        assert r.confidence is None


class TestEnums:
    def test_filter_result(self):
        assert FilterResult.SKIP.value == "skip"
        assert FilterResult.REJECT.value == "reject"
        assert FilterResult.PENDING_LLM.value == "pending_llm"

    def test_paper_source(self):
        assert PaperSource.SEED.value == "seed"
        assert PaperSource.FORWARD.value == "forward"
        assert PaperSource.BACKWARD.value == "backward"

    def test_llm_verdict(self):
        assert LLMVerdict.ACCEPT.value == "accept"
        assert LLMVerdict.REJECT.value == "reject"
        assert LLMVerdict.ACCEPT_SEED.value == "accept_seed"


class TestExceptions:
    def test_hierarchy(self):
        for cls in (
            LLMParseError,
            OpenAlexAPIError,
            SemanticScholarAPIError,
            BudgetExhaustedError,
        ):
            assert issubclass(cls, CiteClawError)
            assert issubclass(cls, Exception)
