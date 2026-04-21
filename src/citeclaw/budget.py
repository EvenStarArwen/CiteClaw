"""Per-run budget accounting for LLM tokens, USD, and S2 requests.

Two parallel views of LLM usage are kept:

* **Per-category** — every ``record_llm`` call lands in a category bucket
  (``"screening"``, ``"annotate"``, ``"other"``, ...). The dashboard
  renders the category breakdown so users see where their tokens went.
* **Per-model** — when a caller passes ``model=`` (the resolved YAML
  alias), counts also accumulate into a per-model bucket. This is the
  only way :meth:`BudgetTracker.cost_by_model` can price a multi-model
  run accurately, since a single ``MODEL_PRICING`` lookup can't cover
  e.g. a cheap screening model plus a more capable search agent.

S2 requests have their own pair of buckets — ``_s2_api`` (live calls
that count against the rate limit) and ``_s2_cache`` (read-through
hits that don't). Cache hits are tracked separately so the
dashboard can report cache effectiveness.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citeclaw.config import Settings


_TOKENS_PER_MILLION = 1_000_000


# Per-million-token prices in USD: (input, output, reasoning).
#
# No provider exposes a per-call USD field — only token counts in
# ``response.usage``. Real cost is either (a) this local table x token
# counts, or (b) the provider's billing dashboard (delayed, aggregated,
# requires admin keys). We need live in-run estimates, so (a).
#
# Reasoning-token billing: every major provider currently bills the
# thinking trace at the output rate. The third tuple slot stays equal
# to ``output`` until a provider splits them — at which point only
# this table changes.
#
# Verified: 2026-01. Bump the date when re-checking the pricing pages.
# Unknown models fall back to ``GENERIC`` (mid-range Gemini Flash).

MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    # Gemini 3
    "gemini-3-pro":                          (1.25, 10.00, 10.00),
    "gemini-3-flash":                        (0.30,  2.50,  2.50),
    "gemini-3-flash-lite":                   (0.10,  0.40,  0.40),
    "gemini-3.1-flash-lite-preview":         (0.10,  0.40,  0.40),
    "gemini-3.1-pro-preview":                (1.25, 10.00, 10.00),
    # Gemini 2.5
    "gemini-2.5-pro":                        (1.25, 10.00, 10.00),
    "gemini-2.5-flash":                      (0.30,  2.50,  2.50),
    "gemini-2.5-flash-lite":                 (0.10,  0.40,  0.40),
    # OpenAI o-series
    "o3":                                    (2.00,  8.00,  8.00),
    "o3-mini":                               (1.10,  4.40,  4.40),
    "o4-mini":                               (1.10,  4.40,  4.40),
    # OpenAI gpt
    "gpt-5":                                 (1.25, 10.00, 10.00),
    "gpt-5-mini":                            (0.25,  2.00,  2.00),
    "gpt-4.1":                               (2.00,  8.00,  8.00),
    "gpt-4.1-mini":                          (0.40,  1.60,  1.60),
    # xAI Grok
    "grok-4":                                (3.00, 15.00, 15.00),
    "grok-4-reasoning":                      (3.00, 15.00, 15.00),
    "grok-4-fast":                           (0.20,  0.50,  0.50),
    "grok-4-fast-reasoning":                 (0.20,  0.50,  0.50),
    "grok-4.20":                             (3.00, 15.00, 15.00),
    "grok-4.20-reasoning":                   (3.00, 15.00, 15.00),
    "grok-4.20-0309":                        (3.00, 15.00, 15.00),
    "grok-4.20-0309-reasoning":              (3.00, 15.00, 15.00),
    "grok-3":                                (2.00, 15.00, 15.00),
    # Anthropic
    "claude-opus-4-6":                       (15.00, 75.00, 75.00),
    "claude-sonnet-4-6":                     (3.00, 15.00, 15.00),
    "claude-haiku-4-5":                      (1.00,  5.00,  5.00),
    "claude-opus-4":                         (15.00, 75.00, 75.00),
    "claude-sonnet-4":                       (3.00, 15.00, 15.00),
    "claude-haiku-4":                        (0.80,  4.00,  4.00),
    # Stub
    "stub":                                  (0.0,   0.0,   0.0),
    # Self-hosted via Modal/vLLM — billed by GPU-hours, not tokens.
    "gemma-4-31b":                           (0.0,   0.0,   0.0),
    "gemma-4":                               (0.0,   0.0,   0.0),
    # Catch-all so unknown models still produce a non-zero estimate.
    "GENERIC":                               (0.30,  2.50,  2.50),
}


def lookup_pricing(model_name: str | None) -> tuple[float, float, float]:
    """Return ``(input_per_M, output_per_M, reasoning_per_M)`` for ``model_name``.

    Falls back to ``GENERIC`` if the exact name is missing; tries
    longest-prefix matches first so e.g. ``gemini-3-flash-lite-preview-09``
    resolves to ``gemini-3-flash-lite``.
    """
    if not model_name:
        return MODEL_PRICING["GENERIC"]
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    for key in sorted(MODEL_PRICING.keys(), key=len, reverse=True):
        if key in {"GENERIC", "stub"}:
            continue
        if model_name.startswith(key):
            return MODEL_PRICING[key]
    return MODEL_PRICING["GENERIC"]


def _cost_usd(
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    pricing: tuple[float, float, float],
) -> float:
    """Apply ``(in_per_M, out_per_M, reason_per_M)`` to a token triple."""
    in_per_m, out_per_m, reason_per_m = pricing
    return (
        input_tokens * in_per_m
        + output_tokens * out_per_m
        + reasoning_tokens * reason_per_m
    ) / _TOKENS_PER_MILLION


class BudgetTracker:
    """Thread-safe per-run accumulator for LLM and S2 usage.

    Public surface is the ``record_*`` mutators plus the read-only
    ``llm_*`` / ``s2_*`` properties and the formatting / cost methods.
    All bucket dicts are private; callers should never reach into them.
    The lock guarantees increments are atomic across worker threads
    (the LLM screener parallelises calls, the S2 client batches across
    a worker pool).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._llm_tokens: dict[str, int] = {}
        self._llm_input_tokens: dict[str, int] = {}
        self._llm_output_tokens: dict[str, int] = {}
        self._llm_calls: dict[str, int] = {}
        self._llm_reasoning_tokens: dict[str, int] = {}
        # Cache hits don't go through ``record_llm`` (they pay nothing),
        # so this counter is the only signal that the prompt cache is
        # actually saving calls.
        self._llm_cache_hits: dict[str, int] = {}
        # Per-model bucket parallel to the per-category maps above. Used
        # by :meth:`cost_by_model` so a mixed-model run prices each
        # alias against its own row in MODEL_PRICING. Clients that
        # don't pass ``model=`` only contribute to the per-category maps.
        self._llm_by_model: dict[str, dict[str, int]] = {}
        self._s2_api: dict[str, int] = {}
        self._s2_cache: dict[str, int] = {}

    # --- aggregate read-only views ---------------------------------------

    @property
    def llm_total_tokens(self) -> int:
        return sum(self._llm_tokens.values())

    @property
    def llm_input_tokens(self) -> int:
        return sum(self._llm_input_tokens.values())

    @property
    def llm_output_tokens(self) -> int:
        return sum(self._llm_output_tokens.values())

    @property
    def llm_reasoning_tokens(self) -> int:
        return sum(self._llm_reasoning_tokens.values())

    @property
    def llm_calls(self) -> int:
        return sum(self._llm_calls.values())

    @property
    def s2_requests(self) -> int:
        return sum(self._s2_api.values())

    @property
    def s2_cache_hits(self) -> int:
        return sum(self._s2_cache.values())

    @property
    def llm_cache_hits(self) -> int:
        return sum(self._llm_cache_hits.values())

    # --- mutators --------------------------------------------------------

    def record_llm_cache_hit(self, category: str = "other") -> None:
        with self._lock:
            self._llm_cache_hits[category] = (
                self._llm_cache_hits.get(category, 0) + 1
            )

    def record_llm(
        self,
        input_tokens: int,
        output_tokens: int,
        category: str = "other",
        *,
        reasoning_tokens: int = 0,
        model: str | None = None,
    ) -> None:
        """Record one LLM call's token usage.

        ``model`` is the resolved YAML alias (NOT the
        ``served_model_name``). When passed, the counts are also
        accumulated into a per-model bucket so :meth:`cost_by_model`
        prices each model independently — required for accurate USD
        tracking in multi-model runs.
        """
        total = input_tokens + output_tokens
        with self._lock:
            self._llm_tokens[category] = self._llm_tokens.get(category, 0) + total
            self._llm_input_tokens[category] = self._llm_input_tokens.get(category, 0) + input_tokens
            self._llm_output_tokens[category] = self._llm_output_tokens.get(category, 0) + output_tokens
            self._llm_calls[category] = self._llm_calls.get(category, 0) + 1
            if reasoning_tokens:
                self._llm_reasoning_tokens[category] = (
                    self._llm_reasoning_tokens.get(category, 0) + reasoning_tokens
                )
            if model:
                bucket = self._llm_by_model.setdefault(
                    model,
                    {"input": 0, "output": 0, "reasoning": 0, "calls": 0},
                )
                bucket["input"] += input_tokens
                bucket["output"] += output_tokens
                bucket["reasoning"] += reasoning_tokens
                bucket["calls"] += 1

    def record_s2(self, req_type: str = "other", *, cached: bool = False) -> None:
        """Bump the api or cache-hit counter for ``req_type``."""
        with self._lock:
            target = self._s2_cache if cached else self._s2_api
            target[req_type] = target.get(req_type, 0) + 1

    # --- exhaustion checks -----------------------------------------------

    def is_exhausted(self, config: "Settings") -> bool:
        """True iff either token or S2-request budget has been consumed."""
        if self.llm_total_tokens >= config.max_llm_tokens:
            return True
        if self.s2_requests >= config.max_s2_requests:
            return True
        return False

    def exhausted(self) -> bool:
        """Legacy no-op kept for backwards compatibility — always returns False.

        Real budget gating goes through :meth:`is_exhausted`, which
        consults the run's :class:`Settings` for the actual ceilings.
        """
        return False

    # --- formatted summaries / breakdowns --------------------------------

    def summary(self) -> str:
        return (
            f"LLM: {self.llm_total_tokens / _TOKENS_PER_MILLION:.3f}M tokens "
            f"({self.llm_calls} calls) | "
            f"S2: {self.s2_requests:,} api + {sum(self._s2_cache.values()):,} cached"
        )

    def cost_estimate(self, model_name: str | None) -> float:
        """USD estimate from token counts × :data:`MODEL_PRICING`.

        Single-model approximation; for multi-model runs prefer
        :meth:`cost_by_model` which prices each alias separately.
        """
        return _cost_usd(
            self.llm_input_tokens,
            self.llm_output_tokens,
            self.llm_reasoning_tokens,
            lookup_pricing(model_name),
        )

    def cost_by_model(self) -> dict[str, dict[str, float | int]]:
        """Per-model USD estimate, sorted by cost descending.

        Returns ``{model: {"input", "output", "reasoning", "calls",
        "cost_usd"}}``. Empty when no client has passed ``model=`` to
        :meth:`record_llm`.
        """
        out: dict[str, dict[str, float | int]] = {}
        for model, counts in self._llm_by_model.items():
            cost = _cost_usd(
                counts["input"], counts["output"], counts["reasoning"],
                lookup_pricing(model),
            )
            out[model] = {**counts, "cost_usd": cost}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["cost_usd"]))

    def total_cost_usd(self) -> float:
        return sum(bucket["cost_usd"] for bucket in self.cost_by_model().values())

    def cost_breakdown(
        self, model_name: str | None,
    ) -> dict[str, dict[str, float | int]]:
        """Per-category cost breakdown for the run-end summary.

        Uses one model's pricing for all categories — accurate when
        every filter shares a model, approximate otherwise.
        """
        pricing = lookup_pricing(model_name)
        out: dict[str, dict[str, float | int]] = {}
        for cat in self._llm_tokens:
            in_t = self._llm_input_tokens.get(cat, 0)
            out_t = self._llm_output_tokens.get(cat, 0)
            reason_t = self._llm_reasoning_tokens.get(cat, 0)
            out[cat] = {
                "input": in_t,
                "output": out_t,
                "reason": reason_t,
                "calls": self._llm_calls.get(cat, 0),
                "cost_usd": _cost_usd(in_t, out_t, reason_t, pricing),
            }
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["cost_usd"]))

    def detailed_summary(self) -> str:
        lines = ["Budget breakdown:"]
        lines.append("  LLM usage by category:")
        for cat in sorted(self._llm_tokens.keys()):
            tokens = self._llm_tokens[cat]
            calls = self._llm_calls.get(cat, 0)
            lines.append(
                f"    {cat:<30s}  {tokens / _TOKENS_PER_MILLION:.3f}M tokens  ({calls} calls)"
            )
        lines.append(
            f"    {'TOTAL':<30s}  "
            f"{self.llm_total_tokens / _TOKENS_PER_MILLION:.3f}M tokens  "
            f"({self.llm_calls} calls)"
        )
        by_model = self.cost_by_model()
        if by_model:
            lines.append("  LLM usage by model (USD estimates):")
            total_usd = 0.0
            for model, b in by_model.items():
                tokens = b["input"] + b["output"]
                cost = b["cost_usd"]
                total_usd += cost
                lines.append(
                    f"    {model:<30s}  {tokens / _TOKENS_PER_MILLION:.3f}M tokens  "
                    f"({b['calls']} calls)  ${cost:.4f}"
                )
            lines.append(f"    {'TOTAL USD (est.)':<30s}  ${total_usd:.4f}")
        lines.append("  S2 usage:")
        all_types = sorted(set(self._s2_api.keys()) | set(self._s2_cache.keys()))
        for t in all_types:
            api = self._s2_api.get(t, 0)
            cache = self._s2_cache.get(t, 0)
            lines.append(f"    {t:<30s}  {api:>5} api  {cache:>5} cached")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        by_model = self.cost_by_model()
        return {
            "llm": {
                "total_tokens": self.llm_total_tokens,
                "total_tokens_millions": round(
                    self.llm_total_tokens / _TOKENS_PER_MILLION, 3,
                ),
                "total_reasoning_tokens": self.llm_reasoning_tokens,
                "total_calls": self.llm_calls,
                "total_cost_usd_est": round(self.total_cost_usd(), 6),
                "by_category": {
                    cat: {
                        "tokens": self._llm_tokens[cat],
                        "calls": self._llm_calls.get(cat, 0),
                        "reasoning_tokens": self._llm_reasoning_tokens.get(cat, 0),
                    }
                    for cat in self._llm_tokens
                },
                "by_model": by_model,
            },
            "s2": {
                "total_api_requests": sum(self._s2_api.values()),
                "total_cache_hits": sum(self._s2_cache.values()),
                "by_type": {
                    t: {"api": self._s2_api.get(t, 0), "cached": self._s2_cache.get(t, 0)}
                    for t in sorted(set(self._s2_api.keys()) | set(self._s2_cache.keys()))
                },
            },
        }
