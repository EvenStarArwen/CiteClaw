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

    def test_wildcard_rejected(self):
        with pytest.raises(TranslationError, match="wildcards"):
            _compile_keyword_expression("agent*")

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
