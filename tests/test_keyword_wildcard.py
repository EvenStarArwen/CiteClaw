"""Trailing-``*`` word-prefix wildcards in keyword filters (title/abstract/venue)."""

from __future__ import annotations

from citeclaw.filters.atoms.keyword import AbstractKeywordFilter, TitleKeywordFilter
from citeclaw.filters.base import FilterContext
from citeclaw.models import PaperRecord


def _fctx(ctx):
    return FilterContext(ctx=ctx)


def _abs(text):
    return PaperRecord(paper_id="p", title="t", abstract=text)


class TestKeywordWildcard:
    def test_prefix_matches_inflections(self, ctx):
        f = AbstractKeywordFilter(keyword="discover*")
        assert f.check(_abs("a new discovery"), _fctx(ctx)).passed
        assert f.check(_abs("they discovered it"), _fctx(ctx)).passed
        assert f.check(_abs("we discover things"), _fctx(ctx)).passed

    def test_prefix_respects_word_boundary(self, ctx):
        f = AbstractKeywordFilter(keyword="discover*")
        # 'rediscover' has no word boundary before 'discover' -> no match
        assert not f.check(_abs("we rediscover old ideas"), _fctx(ctx)).passed
        assert not f.check(_abs("wholly unrelated content"), _fctx(ctx)).passed

    def test_phrase_wildcard(self, ctx):
        f = AbstractKeywordFilter(keyword="large language model*")
        assert f.check(_abs("we train large language models"), _fctx(ctx)).passed
        assert f.check(_abs("a large language model"), _fctx(ctx)).passed
        assert not f.check(_abs("small vision networks"), _fctx(ctx)).passed

    def test_wildcard_in_formula_mixes_with_plain(self, ctx):
        f = AbstractKeywordFilter(
            formula="agent & llm",
            keywords={"agent": "agent*", "llm": "LLM"},
            match="whole_word",
        )
        assert f.check(_abs("an agentic LLM pipeline"), _fctx(ctx)).passed
        # agentic matches (prefix) but LLM absent -> reject
        assert not f.check(_abs("an agentic pipeline"), _fctx(ctx)).passed

    def test_whole_word_plain_term_is_not_substring(self, ctx):
        # A bare term (no star) under whole_word must not match inside words:
        # 'AI' should hit 'AI system' but not 'brain'/'domain'.
        f = AbstractKeywordFilter(formula="ai", keywords={"ai": "AI"}, match="whole_word")
        assert f.check(_abs("an AI scientist"), _fctx(ctx)).passed
        assert not f.check(_abs("the brain and the domain"), _fctx(ctx)).passed

    def test_lone_star_does_not_match_everything(self, ctx):
        # The len>1 guard means a stray '*' is a literal, never a match-all.
        f = TitleKeywordFilter(keyword="*")
        assert not f.check(PaperRecord(paper_id="p", title="anything at all"), _fctx(ctx)).passed
