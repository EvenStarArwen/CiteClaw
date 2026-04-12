"""Twenty pipeline configs for live smoke tests.

Each function receives a :class:`SmokeTestTopic` and returns a dict::

    {
        "blocks":           dict,       # named filter blocks
        "pipeline":         list[dict], # step sequence
        "max_papers_total": int,
        "needs_llm":        bool,       # assert llm_calls > 0
        "needs_cluster":    bool,       # assert ctx.clusters non-empty
    }
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.smoke_topics import SmokeTestTopic

# ---------------------------------------------------------------------------
# Group A — individual steps (max_papers_total=15)
# ---------------------------------------------------------------------------


def pipeline_forward_only(topic: SmokeTestTopic) -> dict:
    """1. LoadSeeds → ExpandForward(10) → Finalize"""
    return {
        "blocks": {
            "pass_all": {"type": "YearFilter", "min": 1900, "max": 2100},
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "pass_all"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": False,
        "needs_cluster": False,
    }


def pipeline_backward_only(topic: SmokeTestTopic) -> dict:
    """2. LoadSeeds → ExpandBackward → Finalize"""
    return {
        "blocks": {
            "pass_all": {"type": "YearFilter", "min": 1900, "max": 2100},
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandBackward", "screener": "pass_all"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": False,
        "needs_cluster": False,
    }


def pipeline_search_only(topic: SmokeTestTopic) -> dict:
    """3. LoadSeeds → ExpandBySearch(1 iter) → Finalize

    S2 bulk search ignores the limit param (returns up to 1000/page),
    so we cap at 1 iteration and use an aggressive screener to keep
    the collection small enough for fast finalize.
    """
    return {
        "blocks": {
            "search_screener": {
                "type": "Sequential",
                "layers": ["year_gate", "cit_gate"],
            },
            "year_gate": {"type": "YearFilter", "min": 2023, "max": 2026},
            "cit_gate": {"type": "CitationFilter", "beta": 20},
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {
                "step": "ExpandBySearch",
                "screener": "search_screener",
                "agent": {
                    "max_iterations": 1,
                    "target_count": 10,
                    "search_limit_per_iter": 10,
                    "reasoning_effort": None,
                },
            },
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": True,
        "needs_cluster": False,
    }


def pipeline_semantics_only(topic: SmokeTestTopic) -> dict:
    """4. LoadSeeds → ExpandBySemantics(10) → Finalize"""
    return {
        "blocks": {
            "pass_all": {"type": "YearFilter", "min": 1900, "max": 2100},
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandBySemantics", "limit": 10, "max_anchor_papers": 3, "screener": "pass_all"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": False,
        "needs_cluster": False,
    }


def pipeline_author_only(topic: SmokeTestTopic) -> dict:
    """5. LoadSeeds → ExpandByAuthor(top_k=2) → Finalize"""
    return {
        "blocks": {
            "pass_all": {"type": "YearFilter", "min": 1900, "max": 2100},
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandByAuthor", "top_k_authors": 2, "papers_per_author": 5, "screener": "pass_all"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": False,
        "needs_cluster": False,
    }


def pipeline_rerank_only(topic: SmokeTestTopic) -> dict:
    """6. LoadSeeds → ExpandForward(10) → Rerank(citation, k=5) → Finalize"""
    return {
        "blocks": {
            "pass_all": {"type": "YearFilter", "min": 1900, "max": 2100},
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "pass_all"},
            {"step": "Rerank", "metric": "citation", "k": 5},
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": False,
        "needs_cluster": False,
    }


def pipeline_cluster_walktrap_and_topics(topic: SmokeTestTopic) -> dict:
    """7. LoadSeeds → ExpandForward(10) → Cluster(walktrap) → Cluster(topic_model) → Finalize"""
    return {
        "blocks": {
            "pass_all": {"type": "YearFilter", "min": 1900, "max": 2100},
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "pass_all"},
            {
                "step": "Cluster",
                "store_as": "walktrap_clusters",
                "algorithm": {"type": "walktrap"},
                "naming": {"mode": "tfidf"},
            },
            {
                "step": "Cluster",
                "store_as": "topic_clusters",
                "algorithm": {"type": "topic_model", "min_cluster_size": 3},
                "naming": {"mode": "tfidf"},
            },
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": False,
        "needs_cluster": True,
    }


# ---------------------------------------------------------------------------
# Group B — filter blocks (max_papers_total=15)
# ---------------------------------------------------------------------------


def pipeline_year_filter(topic: SmokeTestTopic) -> dict:
    """8. ExpandForward with YearFilter(min=2020)"""
    return {
        "blocks": {
            "year_only": {"type": "YearFilter", "min": 2020},
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "year_only"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": False,
        "needs_cluster": False,
    }


def pipeline_citation_filter(topic: SmokeTestTopic) -> dict:
    """9. ExpandForward with CitationFilter(beta=1)"""
    return {
        "blocks": {
            "cit_only": {"type": "CitationFilter", "beta": 1},
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "cit_only"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": False,
        "needs_cluster": False,
    }


def pipeline_llm_title(topic: SmokeTestTopic) -> dict:
    """10. ExpandForward with LLMFilter(scope=title)"""
    return {
        "blocks": {
            "llm_title": {
                "type": "LLMFilter",
                "scope": "title",
                "prompt": "The paper is directly relevant to the research topic",
            },
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "llm_title"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": True,
        "needs_cluster": False,
    }


def pipeline_llm_title_abstract(topic: SmokeTestTopic) -> dict:
    """11. ExpandForward with LLMFilter(scope=title_abstract)"""
    return {
        "blocks": {
            "llm_ta": {
                "type": "LLMFilter",
                "scope": "title_abstract",
                "prompt": "The paper is directly relevant to the research topic",
            },
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "llm_ta"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": True,
        "needs_cluster": False,
    }


def pipeline_llm_formula(topic: SmokeTestTopic) -> dict:
    """12. ExpandForward with LLMFilter formula mode"""
    return {
        "blocks": {
            "llm_formula": {
                "type": "LLMFilter",
                "scope": "title_abstract",
                "formula": "(q_method | q_empirical) & !q_survey",
                "queries": {
                    "q_method": "The paper proposes or applies a novel computational method",
                    "q_empirical": "The paper presents significant empirical results or benchmarks",
                    "q_survey": "The paper is primarily a survey, review, or tutorial",
                },
            },
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "llm_formula"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": True,
        "needs_cluster": False,
    }


def pipeline_similarity_filter(topic: SmokeTestTopic) -> dict:
    """13. ExpandForward with SimilarityFilter(SemanticSim)"""
    return {
        "blocks": {
            "sim_screener": {
                "type": "SimilarityFilter",
                "threshold": 0.01,
                "measures": [{"type": "SemanticSim", "embedder": "s2"}],
                "on_no_data": "pass",
            },
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "sim_screener"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 15,
        "needs_llm": False,
        "needs_cluster": False,
    }


# ---------------------------------------------------------------------------
# Group C — complex filter composition (max_papers_total=50)
# ---------------------------------------------------------------------------


def pipeline_sequential_filters(topic: SmokeTestTopic) -> dict:
    """14. Sequential[YearFilter, CitationFilter, LLMFilter]"""
    return {
        "blocks": {
            "year": {"type": "YearFilter", "min": 2018, "max": 2026},
            "cit": {"type": "CitationFilter", "beta": 1},
            "llm": {
                "type": "LLMFilter",
                "scope": "title",
                "prompt": "The paper is relevant to the research topic",
            },
            "seq_screener": {
                "type": "Sequential",
                "layers": ["year", "cit", "llm"],
            },
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "seq_screener"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 50,
        "needs_llm": True,
        "needs_cluster": False,
    }


def pipeline_any_block(topic: SmokeTestTopic) -> dict:
    """15. Any[LLMFilter(strict), CitationFilter(generous)]"""
    return {
        "blocks": {
            "strict_llm": {
                "type": "LLMFilter",
                "scope": "title_abstract",
                "prompt": "The paper makes a direct methodological contribution",
            },
            "generous_cit": {"type": "CitationFilter", "beta": 0.5},
            "any_screener": {
                "type": "Any",
                "layers": ["strict_llm", "generous_cit"],
            },
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "any_screener"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 50,
        "needs_llm": True,
        "needs_cluster": False,
    }


def pipeline_not_block(topic: SmokeTestTopic) -> dict:
    """16. Sequential[YearFilter, Not(LLMFilter("is a survey"))]"""
    return {
        "blocks": {
            "year": {"type": "YearFilter", "min": 2015, "max": 2026},
            "is_survey": {
                "type": "LLMFilter",
                "scope": "title",
                "prompt": "The paper is a survey, review, or meta-analysis",
            },
            "not_survey": {"type": "Not", "layer": "is_survey"},
            "not_screener": {
                "type": "Sequential",
                "layers": ["year", "not_survey"],
            },
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "not_screener"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 50,
        "needs_llm": True,
        "needs_cluster": False,
    }


def pipeline_route_block(topic: SmokeTestTopic) -> dict:
    """17. Route[venue_in arXiv → strict_cit, default → loose_cit]"""
    return {
        "blocks": {
            "strict_cit": {"type": "CitationFilter", "beta": 10},
            "loose_cit": {"type": "CitationFilter", "beta": 1},
            "route_screener": {
                "type": "Route",
                "routes": [
                    {
                        "if": {"venue_in": ["ArXiv", "arXiv", "bioRxiv", "medRxiv"]},
                        "pass_to": "strict_cit",
                    },
                    {"default": "loose_cit"},
                ],
            },
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "route_screener"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 50,
        "needs_llm": False,
        "needs_cluster": False,
    }


# ---------------------------------------------------------------------------
# Group D — composition patterns (max_papers_total=80)
# ---------------------------------------------------------------------------


def pipeline_parallel_forward_backward(topic: SmokeTestTopic) -> dict:
    """18. Parallel[[ExpandForward], [ExpandBackward]]"""
    return {
        "blocks": {
            "year": {"type": "YearFilter", "min": 2015, "max": 2026},
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {
                "step": "Parallel",
                "branches": [
                    [{"step": "ExpandForward", "max_citations": 10, "screener": "year"}],
                    [{"step": "ExpandBackward", "screener": "year"}],
                ],
            },
            {"step": "Finalize"},
        ],
        "max_papers_total": 80,
        "needs_llm": False,
        "needs_cluster": False,
    }


def pipeline_series_expand_rescreen(topic: SmokeTestTopic) -> dict:
    """19. ExpandForward → ReScreen(LLM) → ExpandBackward"""
    return {
        "blocks": {
            "lenient": {"type": "YearFilter", "min": 2015, "max": 2026},
            "strict_llm": {
                "type": "LLMFilter",
                "scope": "title",
                "prompt": "The paper directly proposes or evaluates a novel method",
            },
        },
        "pipeline": [
            {"step": "LoadSeeds"},
            {"step": "ExpandForward", "max_citations": 10, "screener": "lenient"},
            {"step": "ReScreen", "screener": "strict_llm"},
            {"step": "ExpandBackward", "screener": "lenient"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 80,
        "needs_llm": True,
        "needs_cluster": False,
    }


def pipeline_full_kitchen_sink(topic: SmokeTestTopic) -> dict:
    """20. Full pipeline: ResolveSeeds → LoadSeeds → Parallel → ReScreen → Search → Cluster → Rerank → Finalize"""
    return {
        "blocks": {
            "year": {"type": "YearFilter", "min": 2015, "max": 2026},
            "cit": {"type": "CitationFilter", "beta": 1},
            "llm_relevance": {
                "type": "LLMFilter",
                "scope": "title",
                "prompt": "The paper is relevant to the research topic",
            },
            "basic_screener": {
                "type": "Sequential",
                "layers": ["year", "cit"],
            },
            "strict_screener": {
                "type": "Sequential",
                "layers": ["year", "cit", "llm_relevance"],
            },
        },
        "pipeline": [
            {"step": "ResolveSeeds"},
            {"step": "LoadSeeds"},
            {
                "step": "Parallel",
                "branches": [
                    [
                        {"step": "ExpandForward", "max_citations": 30, "screener": "basic_screener"},
                        {"step": "Rerank", "metric": "citation", "k": 10},
                    ],
                    [
                        {"step": "ExpandBackward", "screener": "basic_screener"},
                    ],
                ],
            },
            {"step": "ReScreen", "screener": "strict_screener"},
            {
                "step": "ExpandBySearch",
                "screener": "strict_screener",
                "agent": {
                    "max_iterations": 1,
                    "target_count": 10,
                    "search_limit_per_iter": 10,
                    "reasoning_effort": None,
                },
            },
            {
                "step": "Cluster",
                "store_as": "topics",
                "algorithm": {"type": "walktrap"},
                "naming": {"mode": "tfidf"},
            },
            {"step": "Rerank", "metric": "citation", "k": 20, "diversity": {"cluster": "topics"}},
            {"step": "MergeDuplicates"},
            {"step": "Finalize"},
        ],
        "max_papers_total": 80,
        "needs_llm": True,
        "needs_cluster": True,
    }


# Ordered list consumed by the parametrized test matrix.
PIPELINES = [
    # Group A — individual steps
    pipeline_forward_only,
    pipeline_backward_only,
    pipeline_search_only,
    pipeline_semantics_only,
    pipeline_author_only,
    pipeline_rerank_only,
    pipeline_cluster_walktrap_and_topics,
    # Group B — filter blocks
    pipeline_year_filter,
    pipeline_citation_filter,
    pipeline_llm_title,
    pipeline_llm_title_abstract,
    pipeline_llm_formula,
    pipeline_similarity_filter,
    # Group C — complex filter composition
    pipeline_sequential_filters,
    pipeline_any_block,
    pipeline_not_block,
    pipeline_route_block,
    # Group D — composition patterns
    pipeline_parallel_forward_backward,
    pipeline_series_expand_rescreen,
    pipeline_full_kitchen_sink,
]
