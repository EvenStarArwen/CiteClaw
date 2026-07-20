"""Tests for the live Web UI's config translation helpers."""

from __future__ import annotations

import pytest

from web.live.backend.config_translate import (
    TranslationError,
    _compile_keyword_expression,
    _translate_filter,
)


class TestKeywordExpression:
    def test_phrases_and_bare_words(self):
        formula, keywords = _compile_keyword_expression(
            '("large language model" | LLM) & "scientific discovery"'
        )
        assert keywords == {
            "k1": "large language model",
            "k2": "LLM",
            "k3": "scientific discovery",
        }
        assert formula.split() == ["(", "k1", "|", "k2", ")", "&", "k3"]

    def test_negation_and_duplicate_terms(self):
        formula, keywords = _compile_keyword_expression('agent & !survey & agent')
        # duplicate term reuses one name
        assert keywords == {"k1": "agent", "k2": "survey"}
        assert formula.split() == ["k1", "&", "!", "k2", "&", "k1"]

    def test_wildcard_terms_pass_through(self):
        formula, keywords = _compile_keyword_expression(
            '(discover* OR "large language model*") AND agent*'
        )
        assert keywords == {
            "k1": "discover*",
            "k2": "large language model*",
            "k3": "agent*",
        }
        assert formula.split() == ["(", "k1", "|", "k2", ")", "&", "k3"]

    def test_word_operators_and_or_not(self):
        formula, keywords = _compile_keyword_expression("research AND agent OR NOT survey")
        assert keywords == {"k1": "research", "k2": "agent", "k3": "survey"}
        assert formula.split() == ["k1", "&", "k2", "|", "!", "k3"]

    def test_word_operators_case_insensitive_and_mixed_with_glyphs(self):
        formula, _ = _compile_keyword_expression("a and b | c")
        assert formula.split() == ["k1", "&", "k2", "|", "k3"]

    def test_lone_star_rejected(self):
        with pytest.raises(TranslationError, match="wildcard must be attached"):
            _compile_keyword_expression("agent * lab")

    def test_unbalanced_parens_rejected(self):
        with pytest.raises(TranslationError):
            _compile_keyword_expression("(agent | lab")

    def test_empty_rejected(self):
        with pytest.raises(TranslationError):
            _compile_keyword_expression("   ")

    def test_filter_node_uses_expression(self):
        node = {
            "kind": "AbstractKeywordFilter",
            "params": {"match": "substring", "expression": 'lab & "ai scientist"'},
        }
        out = _translate_filter(node)
        assert out["type"] == "AbstractKeywordFilter"
        assert out["keywords"] == {"k1": "lab", "k2": "ai scientist"}
        assert out["formula"].split() == ["k1", "&", "k2"]

    def test_legacy_named_form_still_translates(self):
        node = {
            "kind": "TitleKeywordFilter",
            "params": {"match": "substring", "formula": "a | b",
                       "keywords": {"a": "x", "b": "y"}},
        }
        out = _translate_filter(node)
        assert out["formula"] == "a | b"
        assert out["keywords"] == {"a": "x", "b": "y"}


class TestMergedKeywordFilter:
    """One 'Keyword' filter kind with a scope selector routes to the three
    scope-specific core classes (the legacy kinds still translate too)."""

    def test_scope_title_routes_to_title_class(self):
        node = {"kind": "KeywordFilter",
                "params": {"scope": "title", "match": "whole_word",
                           "expression": "agent* AND LLM"}}
        out = _translate_filter(node)
        assert out["type"] == "TitleKeywordFilter"
        assert out["match"] == "whole_word"
        assert out["keywords"] == {"k1": "agent*", "k2": "LLM"}
        assert out["formula"].split() == ["k1", "&", "k2"]

    def test_scope_abstract_and_venue(self):
        abs_out = _translate_filter({"kind": "KeywordFilter",
            "params": {"scope": "abstract", "expression": "lab*"}})
        assert abs_out["type"] == "AbstractKeywordFilter"
        ven_out = _translate_filter({"kind": "KeywordFilter",
            "params": {"scope": "venue", "match": "starts_with", "expression": '"Nature"'}})
        assert ven_out["type"] == "VenueKeywordFilter"

    def test_default_scope_is_abstract(self):
        out = _translate_filter({"kind": "KeywordFilter", "params": {"expression": "agent"}})
        assert out["type"] == "AbstractKeywordFilter"

    def test_unknown_scope_rejected(self):
        with pytest.raises(TranslationError, match="unknown scope"):
            _translate_filter({"kind": "KeywordFilter", "params": {"scope": "body", "expression": "x"}})


class TestLLMBlankCriterion:
    """A blank LLM criterion silently passes every paper (the model treats an
    empty 'Criterion' as matching everything). Translation must refuse it."""

    def test_formula_with_empty_query_rejected(self):
        node = {"kind": "LLMFilter",
                "params": {"scope": "title", "formula": "q1", "queries": {"q1": ""}}}
        with pytest.raises(TranslationError, match="no criterion"):
            _translate_filter(node)

    def test_formula_with_whitespace_query_rejected(self):
        node = {"kind": "LLMFilter",
                "params": {"scope": "title", "formula": "q1", "queries": {"q1": "   "}}}
        with pytest.raises(TranslationError, match="no criterion"):
            _translate_filter(node)

    def test_formula_missing_referenced_query_rejected(self):
        # formula references q1 but queries only defines q2 (q1 blank/absent)
        node = {"kind": "LLMFilter",
                "params": {"scope": "title", "formula": "q1 | q2",
                           "queries": {"q1": "is relevant", "q2": ""}}}
        with pytest.raises(TranslationError, match="no criterion"):
            _translate_filter(node)

    def test_blank_prompt_rejected(self):
        node = {"kind": "LLMFilter", "params": {"scope": "title", "prompt": "   "}}
        with pytest.raises(TranslationError, match="no criterion"):
            _translate_filter(node)

    def test_filled_query_ok(self):
        node = {"kind": "LLMFilter",
                "params": {"scope": "title", "formula": "q1", "queries": {"q1": "is relevant"}}}
        out = _translate_filter(node)
        assert out["type"] == "LLMFilter"
        assert out["formula"] == "q1"
        assert out["queries"] == {"q1": "is relevant"}

    def test_filled_prompt_ok(self):
        node = {"kind": "LLMFilter", "params": {"scope": "title", "prompt": "is relevant"}}
        out = _translate_filter(node)
        assert out["prompt"] == "is relevant"
