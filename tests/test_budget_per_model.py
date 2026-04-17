"""Tests for per-model USD cost tracking in :class:`BudgetTracker`.

The BudgetTracker has long tracked per-category tokens. This module
covers the per-model layer added for multi-model runs (e.g. cheap
screening model + capable search agent), where single-model price
lookups misreport cost.
"""

from __future__ import annotations

from citeclaw.config import BudgetTracker, MODEL_PRICING


def test_record_llm_without_model_stays_out_of_by_model():
    """Legacy call sites that don't pass ``model=`` should not appear
    in the per-model breakdown (but still aggregate in totals)."""
    b = BudgetTracker()
    b.record_llm(100, 50, "screening")
    assert b.llm_total_tokens == 150
    assert b.cost_by_model() == {}
    assert b.total_cost_usd() == 0.0


def test_record_llm_with_model_accumulates():
    """Per-model bucket accumulates across calls on the same model."""
    b = BudgetTracker()
    b.record_llm(1000, 500, "screening", model="grok-4-fast-reasoning")
    b.record_llm(2000, 800, "screening", model="grok-4-fast-reasoning", reasoning_tokens=300)
    by_model = b.cost_by_model()
    assert "grok-4-fast-reasoning" in by_model
    entry = by_model["grok-4-fast-reasoning"]
    assert entry["input"] == 3000
    assert entry["output"] == 1300
    assert entry["reasoning"] == 300
    assert entry["calls"] == 2
    # Cost math: Grok-4-fast-reasoning is (0.20, 0.50, 0.50) per million.
    in_per_m, out_per_m, reason_per_m = MODEL_PRICING["grok-4-fast-reasoning"]
    expected = (
        3000 * in_per_m / 1_000_000
        + 1300 * out_per_m / 1_000_000
        + 300 * reason_per_m / 1_000_000
    )
    assert abs(entry["cost_usd"] - expected) < 1e-9


def test_mixed_models_priced_independently():
    """A run with two different models prices each one with its own
    per-million rates — NOT a single aggregate model."""
    b = BudgetTracker()
    # Gemma via Modal: treated as stub pricing (free) via GENERIC fallback
    # — pick an explicit catalogue entry so this test is hermetic.
    b.record_llm(10_000, 2_000, "cat_a", model="grok-4-fast-reasoning")
    b.record_llm(10_000, 2_000, "cat_b", model="gemini-3-flash")
    by_model = b.cost_by_model()
    assert set(by_model.keys()) == {"grok-4-fast-reasoning", "gemini-3-flash"}
    # Sorted desc by cost — Gemini 3 Flash ($0.30/$2.50 per M) costs more
    # than Grok-4-fast-reasoning ($0.20/$0.50 per M) on identical token
    # counts because its output rate dominates.
    models_in_order = list(by_model.keys())
    assert models_in_order[0] == "gemini-3-flash"
    assert by_model["gemini-3-flash"]["cost_usd"] > by_model["grok-4-fast-reasoning"]["cost_usd"]


def test_to_dict_exposes_by_model_and_total_cost():
    """The JSON surface surfaces both per-model breakdown and an
    aggregate USD estimate so callers don't have to sum."""
    b = BudgetTracker()
    b.record_llm(100_000, 50_000, "x", model="gemini-3-flash")
    d = b.to_dict()
    assert "by_model" in d["llm"]
    assert "total_cost_usd_est" in d["llm"]
    assert d["llm"]["total_cost_usd_est"] > 0
    assert "gemini-3-flash" in d["llm"]["by_model"]


def test_detailed_summary_mentions_model_and_usd():
    """Human-readable summary includes the per-model USD table when
    any record_llm call passed a model."""
    b = BudgetTracker()
    b.record_llm(100_000, 50_000, "x", model="gemini-3-flash")
    s = b.detailed_summary()
    assert "by model" in s
    assert "gemini-3-flash" in s
    assert "$" in s
