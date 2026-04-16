"""Iterative meta-LLM search agent.

The agent designs targeted literature-database queries from a topic
description and a sample of papers already in the collection, then
iteratively refines its query based on what each search returned. Each
iteration consists of one LLM call (the agent designs a query, with a
mandatory ``thinking`` scratchpad placed first in the response schema)
followed by one ``s2.search_bulk`` call. The cumulative hit set is
de-duplicated by ``paperId`` across iterations.

This module exposes:

- :class:`AgentConfig` — knobs the caller dials in (iteration cap, token
  cap, target collection size, per-iter search width, etc.). Defaults
  match the architectural decisions captured in the roadmap: four
  iterations, 200k LLM tokens, target 200 papers, 500 results per
  S2 search call, 20-paper sample for the per-turn observation summary,
  and ``reasoning_effort="high"`` so capable models stack native
  reasoning tokens on top of the schema's ``thinking`` field.
- :class:`AgentTurn` — what one iteration produced.
- :class:`SearchAgentResult` — the agent's final report.
- :func:`run_iterative_search` — the loop body itself.

The loop body never builds its own LLM client: callers (typically
``ExpandBySearch.run`` in Phase C) build one once via
:func:`citeclaw.clients.llm.factory.build_llm_client` and pass it in,
so the same client instance is reused across all iterations and the
budget bookkeeping stays consistent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from citeclaw.models import PaperRecord
from citeclaw.prompts.search_refine import (
    RESPONSE_SCHEMA,
    SYSTEM,
    USER_TEMPLATE,
)

if TYPE_CHECKING:
    from citeclaw.clients.llm.base import LLMClient
    from citeclaw.context import Context


@dataclass
class AgentConfig:
    """Caller-tunable knobs for one search-agent run.

    All defaults match the values fixed during the design discussion
    (see roadmap "Architectural decisions" item 2). The two-level
    reasoning split is: ``max_iterations`` controls the outer loop
    where each turn sees the prior transcript, and
    ``reasoning_effort`` is forwarded to the underlying LLM client so
    capable models layer native chain-of-thought tokens on top of the
    response schema's ``thinking`` field.
    """

    max_iterations: int = 4
    max_llm_tokens: int = 200_000
    target_count: int = 200
    search_limit_per_iter: int = 500
    summarize_sample: int = 20
    model: str | None = None
    reasoning_effort: str | None = "high"


@dataclass
class AgentTurn:
    """One pass through the agent's outer loop.

    Created after each LLM call + S2 search round-trip and appended to
    :attr:`SearchAgentResult.transcript`. The fields after ``thinking``
    summarise the result set the agent observed *for this turn*, so
    PB-04's transcript renderer can show the agent's later self what
    each prior query actually returned (year range, unique venues,
    sample titles) without re-quoting the full hit list.

    PH-02: ``n_novel`` is the count of results from THIS turn that were
    not already in the cumulative hit set before this turn ran. It is
    the saturation signal — when ``n_novel`` drops to single digits the
    agent is wasting iterations on dupes and should mark satisfied or
    pivot to a substantively different query rather than broaden the
    same query incrementally.

    PH-03: ``total_in_corpus`` is what S2 reports as the total matching
    paper count for this query (vs ``n_results`` which is what we
    actually fetched, capped by ``search_limit_per_iter``). When
    ``total_in_corpus > n_results`` the agent is only seeing the first
    page of a much larger result set and MUST narrow with filters
    rather than synonyms — broadening into a 5000-paper corpus just
    surfaces a different first-1000 page of mostly-noise.

    PH-08: ``raw_reasoning`` is the model's NATIVE reasoning trace from
    the LLM client's reasoning_content field (vLLM gemma4 parser /
    OpenAI o-series / Gemini thinking parts). This is separate from
    ``thinking`` (which is the model's first-pass scratchpad that
    lives INSIDE the structured JSON response). Captured for diagnosis
    only — the prompt-design loop reads it to understand why the
    agent made a particular query / decision. Empty string when the
    provider doesn't expose native reasoning.
    """

    iteration: int
    thinking: str
    query: dict
    n_results: int
    n_novel: int
    total_in_corpus: int
    unique_venues: list[str]
    year_range: tuple[int | None, int | None]
    sample_titles: list[str]
    decision: str
    reasoning: str
    raw_reasoning: str = ""


@dataclass
class SearchAgentResult:
    """Final report from one ``run_iterative_search`` call.

    ``hits`` is the cumulative dedup'd hit set across all iterations;
    ``transcript`` preserves every turn in iteration order;
    ``final_decision`` records which lifecycle state broke the loop
    (``satisfied`` / ``abort`` / ``max_iterations`` / ``budget``);
    ``tokens_used`` and ``s2_requests_used`` are the deltas this run
    drew from the shared :class:`citeclaw.config.BudgetTracker`.
    """

    hits: list[dict] = field(default_factory=list)
    transcript: list[AgentTurn] = field(default_factory=list)
    final_decision: str = ""
    tokens_used: int = 0
    s2_requests_used: int = 0


# ---------------------------------------------------------------------------
# Loop body
# ---------------------------------------------------------------------------


def _render_anchor_papers(anchor_papers: list[PaperRecord]) -> str:
    """Render the anchor-papers block for the user prompt.

    Empty input falls back to the canonical bootstrap message so capable
    LLMs can still produce a sensible initial query from the topic
    description alone.
    """
    if not anchor_papers:
        return "(No anchor papers — bootstrap from topic description alone.)"
    lines: list[str] = []
    for i, p in enumerate(anchor_papers, start=1):
        title = p.title or "<no title>"
        if p.year is not None:
            lines.append(f"{i}. {title} ({p.year})")
        else:
            lines.append(f"{i}. {title}")
    return "\n".join(lines)


def _render_transcript(turns: list[AgentTurn]) -> str:
    """Render the transcript section for the next iteration's user prompt.

    Each turn is rendered with the labeled lines required by the spec
    (``Thinking:``, ``Query:``, ``Observed:``, ``Sample titles:``,
    ``Decision:``). The ``Query:`` line embeds the JSON-quoted ``"query":``
    key explicitly via ``json.dumps({"query": turn.query})`` so the stub
    responder's iteration counter (which counts ``"query":`` substrings)
    advances by exactly one per completed turn — that's the contract
    PB-02 established with this module.
    """
    if not turns:
        return "(No prior turns yet — this is the first iteration.)"
    blocks: list[str] = []
    for turn in turns:
        # Wrap the query in a single-key envelope so the rendered text
        # contains the literal substring `"query":` exactly once. The
        # double-Query labeling (`Query: {"query": ...}`) is intentional
        # — it lets a human reader see both the section heading and the
        # raw JSON the agent emitted, while keeping the iteration-count
        # contract with the stub responder.
        query_envelope = json.dumps({"query": turn.query})
        ymin, ymax = turn.year_range
        if ymin is None and ymax is None:
            year_str = "unknown"
        elif ymin == ymax:
            year_str = str(ymin)
        else:
            year_str = f"{ymin}-{ymax}"
        venues_str = ", ".join(turn.unique_venues[:5]) or "(none)"
        titles_str = "; ".join(turn.sample_titles[:5]) or "(none)"
        # Compose the observation line. When the corpus has more papers
        # than we fetched, surface that explicitly so the agent knows it's
        # only seeing a page and must narrow with filters rather than
        # broaden with synonyms.
        if turn.total_in_corpus > turn.n_results:
            count_phrase = (
                f"{turn.n_results} fetched of {turn.total_in_corpus} "
                f"matching in the S2 corpus (PARTIAL — narrow with "
                f"filters to see the rest)"
            )
        else:
            count_phrase = f"{turn.n_results} matching in the S2 corpus"
        block = (
            f"Turn {turn.iteration}:\n"
            f"  Thinking: {turn.thinking}\n"
            f"  Query: {query_envelope}\n"
            f"  Observed: {count_phrase}, "
            f"{turn.n_novel} NEW since previous turns, "
            f"years {year_str}, venues {venues_str}\n"
            f"  Sample titles: {titles_str}\n"
            f"  Decision: {turn.decision} — {turn.reasoning}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _summarize_results(
    new_papers: list[dict[str, Any]],
    ctx: "Context",
    sample_size: int,
) -> tuple[list[str], tuple[int | None, int | None], list[str]]:
    """Hydrate up to ``sample_size`` papers and return
    ``(unique_venues, (ymin, ymax), sample_titles)`` for the agent's
    self-observation summary."""
    if sample_size <= 0 or not new_papers:
        return [], (None, None), []
    sample = new_papers[:sample_size]
    candidates = [
        {"paper_id": p.get("paperId")}
        for p in sample
        if isinstance(p, dict) and p.get("paperId")
    ]
    if not candidates:
        return [], (None, None), []
    enriched = ctx.s2.enrich_batch(candidates)

    unique_venues: list[str] = []
    seen_venues: set[str] = set()
    years: list[int] = []
    sample_titles: list[str] = []
    for rec in enriched:
        venue = getattr(rec, "venue", None)
        if venue and venue not in seen_venues:
            seen_venues.add(venue)
            unique_venues.append(venue)
        year = getattr(rec, "year", None)
        if isinstance(year, int):
            years.append(year)
        title = getattr(rec, "title", None)
        if title:
            sample_titles.append(title)
    year_range: tuple[int | None, int | None] = (
        (min(years), max(years)) if years else (None, None)
    )
    return unique_venues, year_range, sample_titles


def run_iterative_search(
    topic_description: str,
    anchor_papers: list[PaperRecord],
    llm_client: "LLMClient",
    ctx: "Context",
    config: AgentConfig,
) -> SearchAgentResult:
    """Run the iterative meta-LLM search agent.

    Per iteration:

    1. Render the user prompt by formatting :data:`USER_TEMPLATE` with
       the topic, anchor block, and rendered transcript-so-far.
    2. Call ``llm_client.call`` with ``category="meta_search_agent"`` and
       ``response_schema=RESPONSE_SCHEMA`` so capable models constrain
       their output to the four-field shape (``thinking`` first).
    3. Parse the JSON response and extract ``thinking`` / ``query`` /
       ``agent_decision`` / ``reasoning``.
    4. Issue ``ctx.s2.search_bulk`` with the agent's query (text +
       optional filters/sort), capped at
       ``config.search_limit_per_iter``.
    5. Dedup the new hits into the cumulative hit set by ``paperId``.
    6. Sample up to ``config.summarize_sample`` of the new hits, hydrate
       them via ``ctx.s2.enrich_batch``, and summarize unique venues,
       year range, and titles for the next turn's observation line.
    7. Append a fresh :class:`AgentTurn` to the transcript.
    8. Break on ``agent_decision`` ∈ {``satisfied``, ``abort``}, on
       ``max_iterations``, or on the ``max_llm_tokens`` cap (delta from
       this run's starting LLM token total).

    The cumulative hit set, transcript, exit reason, and budget deltas
    are returned in a :class:`SearchAgentResult`. The function never
    builds its own LLM client — callers pass one in, so the same
    instance (and its budget bookkeeping) is reused across iterations.
    """
    start_tokens = ctx.budget.llm_total_tokens
    start_s2_requests = ctx.budget.s2_requests
    anchor_block = _render_anchor_papers(anchor_papers)
    transcript_turns: list[AgentTurn] = []
    cumulative_hits: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    final_decision = ""

    # Dashboard hook: drive a "phase A → phase B → phase C" inner bar
    # per iteration so the live panel never stares blank during the
    # multi-second LLM + S2 round-trip. Falls back to a no-op
    # NullDashboard when the run is non-interactive (tests / CI / piped).
    dash = getattr(ctx, "dashboard", None)
    if dash is None:
        from citeclaw.progress import NullDashboard
        dash = NullDashboard()

    for iteration in range(1, config.max_iterations + 1):
        # PH-08 v2 deterministic saturation guardrail (safety net for
        # the prompt's MANDATORY rule C). The prompt tells the model to
        # set should_stop=true when the corpus is exhausted (n_novel < 5
        # in two turns), but the model's optimism sometimes overrides
        # the rule and it tries one more pivot. The guardrail is a
        # STRICTER deterministic backstop:
        #
        #   if the last two completed turns BOTH had n_results > 0
        #   AND BOTH had n_novel == 0
        #   → forced stop, no LLM call, no S2 call
        #
        # The ``n_results > 0`` clause is critical: it distinguishes
        # genuine saturation (the agent found the same papers twice in a
        # row) from a broken query (zero results from a syntax error or
        # bad filter). PH-08 testing surfaced cases where the agent used
        # an invalid fieldsOfStudy abbreviation and got 0 results for two
        # turns straight; without this clause the guardrail fired and
        # killed the run before the agent could correct the typo.
        #
        # The threshold uses ``n_novel == 0`` (not <5) because:
        #   - The prompt's <5 rule is a soft signal the model should act
        #     on; the guardrail catches only the unambiguous "literally
        #     zero new papers for two turns straight" case.
        #   - <5 would false-positive on small test corpora and on
        #     legitimate niche topics where each turn finds 1-3 new
        #     relevant papers — those are still valuable progress.
        #
        # The check needs at least 2 prior turns (no-op for iter 1 and
        # iter 2). When the guardrail fires, ``final_decision`` records
        # a distinct ``"saturated_guardrail"`` so callers can tell this
        # apart from a model-driven satisfied stop.
        if len(transcript_turns) >= 2:
            last = transcript_turns[-1]
            prev = transcript_turns[-2]
            if (
                last.n_results > 0
                and prev.n_results > 0
                and last.n_novel == 0
                and prev.n_novel == 0
            ):
                final_decision = "saturated_guardrail"
                break

        transcript_text = _render_transcript(transcript_turns)
        # PH-08 v2: target_count is no longer surfaced to the model.
        # The v1 prompt embedded it as "Target: {target_count} papers"
        # and the model anchored on it as a hard requirement, refusing
        # to mark satisfied even when the corpus was clearly saturated.
        # AgentConfig.target_count still exists for callers that want
        # to set it, but it now ONLY influences the abort safety net,
        # not the LLM's stop decision.
        user_prompt = USER_TEMPLATE.format(
            topic_description=topic_description,
            anchor_papers_block=anchor_block,
            transcript=transcript_text,
            iteration=iteration,
            max_iterations=config.max_iterations,
        )
        dash.begin_phase(
            f"search iter {iteration}/{config.max_iterations} · designing query",
            total=3,
        )
        resp = llm_client.call(
            SYSTEM,
            user_prompt,
            category="meta_search_agent",
            response_schema=RESPONSE_SCHEMA,
        )
        dash.tick_inner(1)
        # PH-08: capture native reasoning_content (the gemma4 parser
        # exposes this when the request set skip_special_tokens=False).
        # Stored on the AgentTurn for diagnosis; not fed back into the
        # next iteration's prompt (the in-JSON ``thinking`` field already
        # serves that role).
        raw_reasoning = getattr(resp, "reasoning_content", "") or ""
        try:
            decoded = json.loads(resp.text)
        except (json.JSONDecodeError, TypeError, ValueError):
            decoded = {}
        if not isinstance(decoded, dict):
            decoded = {}

        # PH-08 v2 schema: ``evaluate`` replaces ``thinking`` (Reflexion-style
        # explicit reflection on the prior turn) and ``should_stop: bool``
        # replaces ``agent_decision: str``. Both old field names are still
        # read for backward compat with the v1 stub responder + tests; the
        # new fields take precedence when present.
        thinking = str(decoded.get("evaluate") or decoded.get("thinking") or "")
        raw_query = decoded.get("query")
        query: dict[str, Any] = raw_query if isinstance(raw_query, dict) else {}
        reasoning = str(decoded.get("reasoning", ""))
        # Stop signal: prefer the new ``should_stop: bool``; fall back to
        # the v1 string ``agent_decision == "satisfied"`` so the stub
        # responder + existing tests still terminate properly.
        v2_should_stop = decoded.get("should_stop")
        if isinstance(v2_should_stop, bool):
            should_stop = v2_should_stop
            agent_decision = "satisfied" if should_stop else "refine"
        else:
            agent_decision = str(decoded.get("agent_decision", ""))
            should_stop = agent_decision == "satisfied"

        query_text = str(query.get("text", ""))
        query_filters = query.get("filters") if isinstance(query.get("filters"), dict) else None
        raw_sort = query.get("sort")
        query_sort = raw_sort if isinstance(raw_sort, str) else None

        # Surface the actual query and filters above the live region so
        # the user can watch the agent's reasoning unfold instead of
        # staring at a frozen panel for 30+ seconds per turn.
        filter_str = (
            " · filters=" + ",".join(f"{k}={v}" for k, v in (query_filters or {}).items())
            if query_filters
            else ""
        )
        dash.note(
            f"search iter {iteration}/{config.max_iterations} · "
            f'query="{query_text[:80]}"{filter_str}'
        )
        dash.begin_phase(
            f"search iter {iteration}/{config.max_iterations} · S2 bulk search",
            total=3,
        )
        dash.tick_inner(1)

        search_payload = ctx.s2.search_bulk(
            query_text,
            filters=query_filters,
            sort=query_sort,
            limit=config.search_limit_per_iter,
        )
        dash.tick_inner(1)
        new_papers: list[dict[str, Any]] = []
        total_in_corpus = 0
        if isinstance(search_payload, dict):
            data = search_payload.get("data")
            if isinstance(data, list):
                new_papers = [p for p in data if isinstance(p, dict)]
            # ``total`` is S2's count of all matching papers in the corpus,
            # not just the ones it returned. Surfacing this in the
            # transcript lets the agent distinguish "we fetched all 30
            # matches" from "we fetched the first 1000 of 5000 — narrow
            # with filters now". When the field is missing or non-int we
            # fall back to the actual returned count.
            raw_total = search_payload.get("total")
            if isinstance(raw_total, int) and raw_total >= 0:
                total_in_corpus = raw_total
            else:
                total_in_corpus = len(new_papers)
        else:
            total_in_corpus = len(new_papers)

        n_novel_this_turn = 0
        for paper in new_papers:
            pid = paper.get("paperId")
            if isinstance(pid, str) and pid and pid not in seen_ids:
                seen_ids.add(pid)
                cumulative_hits.append(paper)
                n_novel_this_turn += 1

        dash.begin_phase(
            f"search iter {iteration}/{config.max_iterations} · summarising hits",
            total=3,
        )
        unique_venues, year_range, sample_titles = _summarize_results(
            new_papers, ctx, config.summarize_sample,
        )
        dash.tick_inner(1)

        transcript_turns.append(
            AgentTurn(
                iteration=iteration,
                thinking=thinking,
                query=query,
                n_results=len(new_papers),
                n_novel=n_novel_this_turn,
                total_in_corpus=total_in_corpus,
                unique_venues=unique_venues,
                year_range=year_range,
                sample_titles=sample_titles,
                decision=agent_decision,
                reasoning=reasoning,
                raw_reasoning=raw_reasoning,
            )
        )

        # One-line per-turn summary above the live region — gives the
        # user a running log of what the agent saw at each step.
        decision_label = agent_decision or ("stop" if should_stop else "refine")
        partial_marker = " (PARTIAL)" if total_in_corpus > len(new_papers) else ""
        dash.note(
            f"search iter {iteration}/{config.max_iterations} · "
            f"{len(new_papers)} hits{partial_marker} · "
            f"{n_novel_this_turn} new · → {decision_label}"
        )

        if should_stop:
            final_decision = "satisfied"
            break
        if agent_decision == "abort":
            final_decision = "abort"
            break
        if ctx.budget.llm_total_tokens - start_tokens >= config.max_llm_tokens:
            final_decision = "budget"
            break
    else:
        final_decision = "max_iterations"

    return SearchAgentResult(
        hits=cumulative_hits,
        transcript=transcript_turns,
        final_decision=final_decision,
        tokens_used=ctx.budget.llm_total_tokens - start_tokens,
        s2_requests_used=ctx.budget.s2_requests - start_s2_requests,
    )
