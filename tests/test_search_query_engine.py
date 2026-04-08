"""Tests for :func:`citeclaw.search.query_engine.apply_local_query`.

The function is pure, so every test is a deterministic input/output
check over a small handful of hand-built ``PaperRecord`` instances.
"""

from __future__ import annotations

import pytest

from citeclaw.models import PaperRecord
from citeclaw.search.query_engine import apply_local_query


def _p(
    pid: str,
    *,
    year: int | None = 2020,
    venue: str | None = "Nature",
    citation_count: int | None = 50,
    title: str = "",
    abstract: str | None = None,
    fields_of_study: list[str] | None = None,
    publication_types: list[str] | None = None,
) -> PaperRecord:
    return PaperRecord(
        paper_id=pid,
        title=title or f"Title {pid}",
        abstract=abstract,
        year=year,
        venue=venue,
        citation_count=citation_count,
        fields_of_study=fields_of_study or [],
        publication_types=publication_types or [],
    )


class TestNoPredicates:
    def test_no_predicates_returns_input_unchanged(self):
        ps = [_p("a"), _p("b"), _p("c")]
        assert apply_local_query(ps) == ps

    def test_empty_input_returns_empty(self):
        assert apply_local_query([]) == []

    def test_no_predicates_does_not_mutate_input(self):
        ps = [_p("a"), _p("b")]
        out = apply_local_query(ps)
        assert out is not ps  # new list
        assert out == ps


class TestYearRange:
    def test_year_min_filters_old_papers(self):
        ps = [_p("a", year=2018), _p("b", year=2020), _p("c", year=2025)]
        out = apply_local_query(ps, year_min=2020)
        assert [p.paper_id for p in out] == ["b", "c"]

    def test_year_max_filters_new_papers(self):
        ps = [_p("a", year=2018), _p("b", year=2020), _p("c", year=2025)]
        out = apply_local_query(ps, year_max=2020)
        assert [p.paper_id for p in out] == ["a", "b"]

    def test_year_range_inclusive_on_both_ends(self):
        ps = [_p("a", year=2019), _p("b", year=2020), _p("c", year=2021), _p("d", year=2022)]
        out = apply_local_query(ps, year_min=2020, year_max=2021)
        assert [p.paper_id for p in out] == ["b", "c"]

    def test_year_min_strict_on_missing_year(self):
        """Strict on missing metadata: ``year is None`` → reject."""
        ps = [_p("a", year=None), _p("b", year=2020)]
        out = apply_local_query(ps, year_min=2020)
        assert [p.paper_id for p in out] == ["b"]

    def test_year_max_strict_on_missing_year(self):
        ps = [_p("a", year=None), _p("b", year=2020)]
        out = apply_local_query(ps, year_max=2025)
        assert [p.paper_id for p in out] == ["b"]


class TestMinCitations:
    def test_filters_below_threshold(self):
        ps = [_p("a", citation_count=10), _p("b", citation_count=100)]
        out = apply_local_query(ps, min_citations=50)
        assert [p.paper_id for p in out] == ["b"]

    def test_strict_on_missing_citation_count(self):
        ps = [_p("a", citation_count=None), _p("b", citation_count=100)]
        out = apply_local_query(ps, min_citations=50)
        assert [p.paper_id for p in out] == ["b"]

    def test_threshold_inclusive(self):
        ps = [_p("a", citation_count=49), _p("b", citation_count=50)]
        out = apply_local_query(ps, min_citations=50)
        assert [p.paper_id for p in out] == ["b"]


class TestVenueRegex:
    def test_case_insensitive_match(self):
        ps = [_p("a", venue="Nature"), _p("b", venue="science")]
        out = apply_local_query(ps, venue_regex="NATURE")
        assert [p.paper_id for p in out] == ["a"]

    def test_substring_match_via_search(self):
        """``re.search`` semantics — partial matches are allowed."""
        ps = [_p("a", venue="Nature Methods"), _p("b", venue="Cell")]
        out = apply_local_query(ps, venue_regex="nature")
        assert [p.paper_id for p in out] == ["a"]

    def test_alternation(self):
        ps = [_p("a", venue="Nature"), _p("b", venue="Science"), _p("c", venue="Cell")]
        out = apply_local_query(ps, venue_regex="nature|science")
        assert {p.paper_id for p in out} == {"a", "b"}

    def test_strict_on_missing_venue(self):
        ps = [_p("a", venue=None), _p("b", venue="Nature")]
        out = apply_local_query(ps, venue_regex="nature")
        assert [p.paper_id for p in out] == ["b"]


class TestTitleRegex:
    def test_case_insensitive_match(self):
        ps = [
            _p("a", title="Attention Is All You Need"),
            _p("b", title="Random Forests"),
        ]
        out = apply_local_query(ps, title_regex="attention")
        assert [p.paper_id for p in out] == ["a"]

    def test_empty_title_does_not_match_nontrivial_regex(self):
        ps = [_p("a", title=""), _p("b", title="Hello world")]
        out = apply_local_query(ps, title_regex="hello")
        assert [p.paper_id for p in out] == ["b"]


class TestAbstractRegex:
    def test_lenient_on_missing_abstract(self):
        """LENIENT: ``abstract is None`` → pass (S2 often lacks abstracts)."""
        ps = [_p("a", abstract=None), _p("b", abstract="contains topic")]
        out = apply_local_query(ps, abstract_regex="topic")
        # Both pass: "a" because lenient, "b" because matches.
        assert {p.paper_id for p in out} == {"a", "b"}

    def test_present_but_mismatched_abstract_is_rejected(self):
        ps = [
            _p("a", abstract="totally unrelated content"),
            _p("b", abstract="some topic discussion"),
        ]
        out = apply_local_query(ps, abstract_regex="topic")
        assert [p.paper_id for p in out] == ["b"]

    def test_case_insensitive_match(self):
        ps = [_p("a", abstract="DEEP LEARNING is great")]
        out = apply_local_query(ps, abstract_regex="deep learning")
        assert [p.paper_id for p in out] == ["a"]


class TestFieldsOfStudyAny:
    def test_any_intersection(self):
        ps = [
            _p("a", fields_of_study=["Biology"]),
            _p("b", fields_of_study=["Computer Science"]),
            _p("c", fields_of_study=["Biology", "Medicine"]),
        ]
        out = apply_local_query(ps, fields_of_study_any=["Biology", "Chemistry"])
        assert {p.paper_id for p in out} == {"a", "c"}

    def test_strict_on_empty_list_paper_field(self):
        """A paper with no fields_of_study cannot intersect any wanted set."""
        ps = [_p("a", fields_of_study=[]), _p("b", fields_of_study=["Biology"])]
        out = apply_local_query(ps, fields_of_study_any=["Biology"])
        assert [p.paper_id for p in out] == ["b"]


class TestPublicationTypesAny:
    def test_any_intersection(self):
        ps = [
            _p("a", publication_types=["JournalArticle"]),
            _p("b", publication_types=["Review"]),
            _p("c", publication_types=["Editorial"]),
        ]
        out = apply_local_query(
            ps, publication_types_any=["JournalArticle", "Review"],
        )
        assert {p.paper_id for p in out} == {"a", "b"}

    def test_strict_on_empty_paper_field(self):
        ps = [_p("a", publication_types=[]), _p("b", publication_types=["Review"])]
        out = apply_local_query(ps, publication_types_any=["Review"])
        assert [p.paper_id for p in out] == ["b"]


class TestCombinedPredicates:
    def test_and_combination_year_and_venue_and_citations(self):
        ps = [
            _p("a", year=2018, venue="Nature", citation_count=100),
            _p("b", year=2021, venue="Nature", citation_count=100),  # passes
            _p("c", year=2021, venue="Cell", citation_count=100),    # wrong venue
            _p("d", year=2021, venue="Nature", citation_count=10),   # too few cites
        ]
        out = apply_local_query(
            ps, year_min=2020, venue_regex="nature", min_citations=50,
        )
        assert [p.paper_id for p in out] == ["b"]

    def test_full_combination(self):
        """Every predicate at once — sanity-check the interaction."""
        ps = [
            _p(
                "match",
                year=2022, venue="Nature", citation_count=200,
                title="Deep Learning for Genomics",
                abstract="We propose a deep learning method.",
                fields_of_study=["Biology", "Computer Science"],
                publication_types=["JournalArticle"],
            ),
            _p(
                "wrong_field",
                year=2022, venue="Nature", citation_count=200,
                title="Deep Learning for Genomics",
                abstract="We propose a deep learning method.",
                fields_of_study=["Physics"],
                publication_types=["JournalArticle"],
            ),
        ]
        out = apply_local_query(
            ps,
            year_min=2020,
            venue_regex="nature",
            min_citations=100,
            title_regex="deep learning",
            abstract_regex="deep learning",
            fields_of_study_any=["Biology"],
            publication_types_any=["JournalArticle"],
        )
        assert [p.paper_id for p in out] == ["match"]
