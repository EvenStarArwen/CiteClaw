"""Tests for :mod:`citeclaw.models` — PaperRecord, ID normalizers, exceptions."""

from __future__ import annotations

import pytest

from citeclaw.models import (
    BudgetExhaustedError,
    CiteClawError,
    LLMVerdict,
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
        # PA-06: subject area + publication type fields
        assert p.fields_of_study == []
        assert p.publication_types == []

    def test_fields_of_study_and_publication_types(self):
        """Direct construction sets the new fields."""
        p = PaperRecord(
            paper_id="x",
            fields_of_study=["Computer Science", "Biology"],
            publication_types=["JournalArticle", "Review"],
        )
        assert p.fields_of_study == ["Computer Science", "Biology"]
        assert p.publication_types == ["JournalArticle", "Review"]

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
        # PA-07: source is now a plain ``str`` field; PaperSource constants
        # are themselves strings, so direct assignment Just Works.
        p.source = PaperSource.FORWARD
        p.llm_verdict = LLMVerdict.ACCEPT.value
        dumped = p.model_dump()
        assert dumped["source"] == "forward"
        assert dumped["llm_verdict"] == "accept"


class TestNormalizeS2Id:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"),
            ("DOI:10.1234/abc", "DOI:10.1234/abc"),
            ("DOI:10.1038/s41586-021-03819-2", "DOI:10.1038/s41586-021-03819-2"),
            ("DOI:10.48550/arXiv.2301.00001", "DOI:10.48550/arXiv.2301.00001"),
            ("CorpusId:12345", "CorpusId:12345"),
            ("ArXiv:2201.00001", "ArXiv:2201.00001"),
            ("PMID:999", "PMID:999"),
            ("MAG:1", "MAG:1"),
            ("ACL:p21", "ACL:p21"),
            ("10.1234/foo", "DOI:10.1234/foo"),
            ("10.1145/3534678.3539092", "DOI:10.1145/3534678.3539092"),
        ],
    )
    def test_valid(self, raw, expected):
        assert normalize_s2_id(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", 42, {"x": 1}, "garbage"])
    def test_invalid(self, raw):
        assert normalize_s2_id(raw) is None

    @pytest.mark.parametrize(
        "raw",
        [
            "DOI:10.1/abc",            # registrant too short (< 4 digits)
            "DOI:10.12/abc",           # ditto
            "DOI:10.1234",             # missing suffix after /
            "DOI:10.1234/",            # empty suffix
            "DOI:foo/bar",             # non-numeric registrant
            "10.1/abc",                # bare, too-short registrant
            "10.12",                   # bare, no suffix
            "10.invalid/abc",          # non-numeric
        ],
    )
    def test_malformed_doi_rejected(self, raw):
        """Strict validation: malformed DOIs must return None instead of
        sneaking through to S2 and failing opaquely downstream."""
        assert normalize_s2_id(raw) is None

    def test_whitespace_stripped(self):
        assert normalize_s2_id("  DOI:10.1234/abc  ") == "DOI:10.1234/abc"


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
    def test_paper_source(self):
        # PA-07: PaperSource is a constants namespace, not an enum.
        assert PaperSource.SEED == "seed"
        assert PaperSource.FORWARD == "forward"
        assert PaperSource.BACKWARD == "backward"
        # New expansion-family sources also live here.
        assert PaperSource.SEARCH == "search"
        assert PaperSource.SEMANTIC == "semantic"
        assert PaperSource.AUTHOR == "author"
        assert PaperSource.REINFORCED == "reinforced"

    def test_llm_verdict(self):
        assert LLMVerdict.ACCEPT.value == "accept"
        assert LLMVerdict.REJECT.value == "reject"
        assert LLMVerdict.ACCEPT_SEED.value == "accept_seed"


class TestExceptions:
    def test_hierarchy(self):
        for cls in (
            SemanticScholarAPIError,
            BudgetExhaustedError,
        ):
            assert issubclass(cls, CiteClawError)
            assert issubclass(cls, Exception)


# ---------------------------------------------------------------------------
# PA-06: paper_to_record extracts fields_of_study + publication_types
# from the various S2 response shapes.
# ---------------------------------------------------------------------------


class TestPaperToRecordCitationSignals:
    """Pin the paper_to_record mapping for citation-signal fields.

    ``influential_citation_count`` is paid for in every S2 fetch (it's
    in PAPER_FIELDS) but easy to accidentally drop from the converter.
    The assert below catches that.
    """

    def test_influential_citation_count_mapped(self):
        from citeclaw.clients.s2.converters import paper_to_record

        rec = paper_to_record({
            "paperId": "p1",
            "citationCount": 120,
            "influentialCitationCount": 8,
        })
        assert rec is not None
        assert rec.citation_count == 120
        assert rec.influential_citation_count == 8

    def test_influential_citation_count_none_when_missing(self):
        from citeclaw.clients.s2.converters import paper_to_record

        rec = paper_to_record({"paperId": "p1", "citationCount": 10})
        assert rec is not None
        assert rec.influential_citation_count is None


class TestPaperToRecordSubjectFields:
    def test_legacy_fields_of_study_string_list(self):
        from citeclaw.clients.s2.converters import paper_to_record

        rec = paper_to_record({
            "paperId": "p1",
            "fieldsOfStudy": ["Computer Science", "Biology"],
        })
        assert rec is not None
        assert rec.fields_of_study == ["Computer Science", "Biology"]

    def test_s2_fields_of_study_dict_list(self):
        from citeclaw.clients.s2.converters import paper_to_record

        rec = paper_to_record({
            "paperId": "p1",
            "s2FieldsOfStudy": [
                {"category": "Medicine", "source": "external"},
                {"category": "Genetics", "source": "s2-fos-model"},
            ],
        })
        assert rec is not None
        assert rec.fields_of_study == ["Medicine", "Genetics"]

    def test_merges_legacy_and_s2_lists(self):
        from citeclaw.clients.s2.converters import paper_to_record

        rec = paper_to_record({
            "paperId": "p1",
            "fieldsOfStudy": ["Computer Science"],
            "s2FieldsOfStudy": [
                {"category": "Computer Science", "source": "s2"},  # dup
                {"category": "Mathematics", "source": "s2"},
            ],
        })
        assert rec is not None
        # Legacy comes first, dup is dropped, s2-only categories appended.
        assert rec.fields_of_study == ["Computer Science", "Mathematics"]

    def test_publication_types_passthrough(self):
        from citeclaw.clients.s2.converters import paper_to_record

        rec = paper_to_record({
            "paperId": "p1",
            "publicationTypes": ["JournalArticle", "Review"],
        })
        assert rec is not None
        assert rec.publication_types == ["JournalArticle", "Review"]

    def test_missing_subject_fields_default_to_empty(self):
        from citeclaw.clients.s2.converters import paper_to_record

        rec = paper_to_record({"paperId": "p1"})
        assert rec is not None
        assert rec.fields_of_study == []
        assert rec.publication_types == []

    def test_null_fields_of_study_does_not_explode(self):
        """S2 sometimes returns ``null`` for omitted list fields."""
        from citeclaw.clients.s2.converters import paper_to_record

        rec = paper_to_record({
            "paperId": "p1",
            "fieldsOfStudy": None,
            "s2FieldsOfStudy": None,
            "publicationTypes": None,
        })
        assert rec is not None
        assert rec.fields_of_study == []
        assert rec.publication_types == []

    def test_skips_malformed_s2_entries(self):
        """An ``s2FieldsOfStudy`` entry without a string ``category`` is dropped."""
        from citeclaw.clients.s2.converters import paper_to_record

        rec = paper_to_record({
            "paperId": "p1",
            "s2FieldsOfStudy": [
                {"category": "Biology"},
                {"source": "external"},  # no category — drop
                "not-a-dict",            # bad shape — drop
                {"category": 42},        # non-string category — drop
            ],
        })
        assert rec is not None
        assert rec.fields_of_study == ["Biology"]

    def test_skips_non_string_publication_types(self):
        from citeclaw.clients.s2.converters import paper_to_record

        rec = paper_to_record({
            "paperId": "p1",
            "publicationTypes": ["JournalArticle", None, 42, "Review"],
        })
        assert rec is not None
        assert rec.publication_types == ["JournalArticle", "Review"]
