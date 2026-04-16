"""Tests for :func:`citeclaw.filters.builder.build_blocks`."""

from __future__ import annotations

import pytest

from citeclaw.filters.atoms.citation import CitationFilter
from citeclaw.filters.atoms.keyword import (
    AbstractKeywordFilter,
    TitleKeywordFilter,
    VenueKeywordFilter,
)
from citeclaw.filters.atoms.llm_query import LLMFilter
from citeclaw.filters.atoms.year import YearFilter
from citeclaw.filters.blocks.any_block import Any_
from citeclaw.filters.blocks.not_block import Not_
from citeclaw.filters.blocks.route import Route
from citeclaw.filters.blocks.sequential import Sequential
from citeclaw.filters.blocks.similarity import SimilarityFilter
from citeclaw.filters.builder import build_blocks
from citeclaw.filters.measures.cit_sim import CitSimMeasure
from citeclaw.filters.measures.ref_sim import RefSimMeasure
from citeclaw.filters.measures.semantic_sim import SemanticSimMeasure


class TestAtomBlocks:
    def test_year_filter(self):
        built = build_blocks({"y": {"type": "YearFilter", "min": 2020, "max": 2024}})
        f = built["y"]
        assert isinstance(f, YearFilter)
        assert f.name == "y"

    def test_citation_filter(self):
        built = build_blocks({"c": {"type": "CitationFilter", "beta": 10.0}})
        f = built["c"]
        assert isinstance(f, CitationFilter)

    def test_llm_filter(self):
        built = build_blocks(
            {"l": {"type": "LLMFilter", "scope": "title", "prompt": "is relevant"}}
        )
        f = built["l"]
        assert isinstance(f, LLMFilter)
        assert f.scope == "title"
        assert f.prompt == "is relevant"
        # New default fields
        assert f.model is None
        assert f.reasoning_effort is None
        assert f.votes == 1
        assert f.min_accepts == 1

    def test_llm_filter_with_overrides(self):
        built = build_blocks({
            "l": {
                "type": "LLMFilter",
                "scope": "title_abstract",
                "prompt": "is relevant",
                "model": "gemini-2.5-flash",
                "reasoning_effort": "low",
                "votes": 5,
                "min_accepts": 3,
            }
        })
        f = built["l"]
        assert f.model == "gemini-2.5-flash"
        assert f.reasoning_effort == "low"
        assert f.votes == 5
        assert f.min_accepts == 3

    def test_unknown_atom_raises(self):
        with pytest.raises(ValueError, match="Unknown block type"):
            build_blocks({"x": {"type": "MysteryFilter"}})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            build_blocks({"x": {}})

    def test_title_keyword_filter_simple(self):
        built = build_blocks(
            {"tk": {"type": "TitleKeywordFilter", "keyword": "deep learning"}}
        )
        f = built["tk"]
        assert isinstance(f, TitleKeywordFilter)
        assert f.name == "tk"
        assert f.keyword == "deep learning"
        assert f.case_sensitive is False
        assert f.match == "substring"

    def test_venue_keyword_filter_allowlist(self):
        built = build_blocks(
            {
                "vk": {
                    "type": "VenueKeywordFilter",
                    "formula": "nature | science | cell",
                    "keywords": {
                        "nature": "Nature",
                        "science": "Science",
                        "cell": "Cell",
                    },
                    "match": "starts_with",
                }
            }
        )
        f = built["vk"]
        assert isinstance(f, VenueKeywordFilter)
        assert f.match == "starts_with"
        assert set(f.keywords.keys()) == {"nature", "science", "cell"}

    def test_abstract_keyword_filter_formula(self):
        built = build_blocks(
            {
                "ak": {
                    "type": "AbstractKeywordFilter",
                    "formula": "(ml | dl) & !survey",
                    "keywords": {
                        "ml": "machine learning",
                        "dl": "deep learning",
                        "survey": "survey",
                    },
                    "case_sensitive": True,
                    "match": "whole_word",
                }
            }
        )
        f = built["ak"]
        assert isinstance(f, AbstractKeywordFilter)
        assert f.formula_expr == "(ml | dl) & !survey"
        assert f.case_sensitive is True
        assert f.match == "whole_word"
        assert set(f.keywords.keys()) == {"ml", "dl", "survey"}


class TestCompositeBlocks:
    def test_sequential_with_refs(self):
        built = build_blocks({
            "y": {"type": "YearFilter", "min": 2020},
            "seq": {"type": "Sequential", "layers": ["y"]},
        })
        s = built["seq"]
        assert isinstance(s, Sequential)
        assert s.layers[0] is built["y"]

    def test_sequential_with_inline(self):
        built = build_blocks({
            "seq": {
                "type": "Sequential",
                "layers": [
                    {"type": "YearFilter", "min": 2020},
                    {"type": "CitationFilter", "beta": 5},
                ],
            },
        })
        s = built["seq"]
        assert len(s.layers) == 2
        assert isinstance(s.layers[0], YearFilter)
        assert isinstance(s.layers[1], CitationFilter)

    def test_any_block(self):
        built = build_blocks({
            "y1": {"type": "YearFilter", "min": 2020},
            "y2": {"type": "YearFilter", "min": 2025},
            "anyb": {"type": "Any", "layers": ["y1", "y2"]},
        })
        a = built["anyb"]
        assert isinstance(a, Any_)

    def test_not_block(self):
        built = build_blocks({
            "inner": {"type": "YearFilter", "min": 2020},
            "n": {"type": "Not", "layer": "inner"},
        })
        n = built["n"]
        assert isinstance(n, Not_)
        assert n.layer is built["inner"]

    def test_not_requires_singular_layer(self):
        with pytest.raises(ValueError, match="requires 'layer:'"):
            build_blocks({"n": {"type": "Not", "layers": []}})


class TestRouteBlock:
    def test_route_with_predicate_and_default(self):
        built = build_blocks({
            "cit_strict": {"type": "CitationFilter", "beta": 60},
            "cit_loose": {"type": "CitationFilter", "beta": 20},
            "r": {
                "type": "Route",
                "routes": [
                    {"if": {"venue_in": ["arXiv", "bioRxiv"]}, "pass_to": "cit_strict"},
                    {"default": "cit_loose"},
                ],
            },
        })
        r = built["r"]
        assert isinstance(r, Route)
        assert len(r.cases) == 2
        assert r.cases[0].is_default is False
        assert r.cases[0].target is built["cit_strict"]
        assert r.cases[1].is_default is True

    def test_route_cit_at_least_predicate(self):
        built = build_blocks({
            "target": {"type": "YearFilter", "min": 2020},
            "r": {
                "type": "Route",
                "routes": [
                    {"if": {"cit_at_least": 100}, "pass_to": "target"},
                    {"if": {"year_at_least": 2023}, "pass_to": "target"},
                ],
            },
        })
        assert len(built["r"].cases) == 2

    def test_route_unknown_predicate(self):
        with pytest.raises(ValueError, match="Unknown predicate"):
            build_blocks({
                "t": {"type": "YearFilter", "min": 2020},
                "r": {
                    "type": "Route",
                    "routes": [{"if": {"weird_thing": 1}, "pass_to": "t"}],
                },
            })

    def test_route_venue_preset_predicate(self):
        from citeclaw.filters.atoms.predicates import VenuePreset

        built = build_blocks({
            "cit_strict": {"type": "CitationFilter", "beta": 60},
            "cit_loose": {"type": "CitationFilter", "beta": 20},
            "r": {
                "type": "Route",
                "routes": [
                    {"if": {"venue_preset": ["nature", "science", "cell"]},
                     "pass_to": "cit_loose"},
                    {"if": {"venue_preset": ["preprint"]},
                     "pass_to": "cit_strict"},
                    {"default": "cit_loose"},
                ],
            },
        })
        r = built["r"]
        assert len(r.cases) == 3
        assert isinstance(r.cases[0].predicate, VenuePreset)
        assert isinstance(r.cases[1].predicate, VenuePreset)


class TestSimilarityBuilder:
    def test_all_measure_types(self):
        built = build_blocks({
            "sim": {
                "type": "SimilarityFilter",
                "threshold": 0.1,
                "on_no_data": "reject",
                "measures": [
                    {"type": "RefSim"},
                    {"type": "CitSim", "pass_if_cited_at_least": 500},
                    {"type": "SemanticSim", "embedder": "s2"},
                ],
            }
        })
        f = built["sim"]
        assert isinstance(f, SimilarityFilter)
        assert f.on_no_data == "reject"
        assert f.threshold == 0.1
        assert len(f.measures) == 3
        assert isinstance(f.measures[0], RefSimMeasure)
        assert isinstance(f.measures[1], CitSimMeasure)
        assert isinstance(f.measures[2], SemanticSimMeasure)

    def test_missing_measure_type(self):
        with pytest.raises(ValueError):
            build_blocks({
                "sim": {
                    "type": "SimilarityFilter",
                    "threshold": 0.1,
                    "measures": [{"type": "NotAMeasure"}],
                }
            })

    def test_measure_not_dict_raises(self):
        with pytest.raises(ValueError):
            build_blocks({
                "sim": {
                    "type": "SimilarityFilter",
                    "threshold": 0.1,
                    "measures": ["RefSim"],  # must be dict
                }
            })


class TestResolution:
    def test_cycle_detection(self):
        with pytest.raises(ValueError, match="[Cc]yclic"):
            build_blocks({
                "a": {"type": "Sequential", "layers": ["b"]},
                "b": {"type": "Sequential", "layers": ["a"]},
            })

    def test_unknown_ref_raises(self):
        with pytest.raises(KeyError, match="not defined"):
            build_blocks({
                "a": {"type": "Sequential", "layers": ["missing"]},
            })

    def test_forward_reference_resolved(self):
        """``build_blocks`` iterates in dict order but resolves lazily, so a
        block defined later can be referenced earlier."""
        built = build_blocks({
            "outer": {"type": "Sequential", "layers": ["inner"]},
            "inner": {"type": "YearFilter", "min": 2020},
        })
        assert built["outer"].layers[0] is built["inner"]
