"""Batched LLM dispatch for LLMFilter atoms.

Extracted from the legacy ``QueryScreener``. Given a list of papers and an
``LLMFilter`` (scope + prompt), returns ``{paper_id: bool}`` verdicts.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from citeclaw.clients.llm import build_llm_client
from citeclaw.prompts.screening import SYSTEM as _SYSTEM
from citeclaw.prompts.screening import USER_TEMPLATE as _USER
from citeclaw.prompts.screening import VENUE_SYSTEM as _VENUE_SYSTEM
from citeclaw.screening.schemas import screening_json_schema

if TYPE_CHECKING:
    from citeclaw.context import Context
    from citeclaw.filters.atoms.llm_query import LLMFilter
    from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.llm_runner")


def _parse(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _parse_matches(raw: str, n: int) -> list[bool] | None:
    """Parse an LLM response into a list of ``n`` booleans.

    Accepts both the structured-output shape ``{"results": [{"index": N,
    "match": bool}, ...]}`` (emitted when the provider honors
    ``response_format``/``response_schema``) *and* the legacy flat-array
    shape ``[{"index": N, "match": bool}, ...]`` so mock responses and
    legacy providers keep working. Returns ``None`` on parse failure.
    """
    try:
        data = _parse(raw)
    except Exception:
        return None
    # Accept both shapes.
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        return None
    out = [False] * n
    for r in data:
        if not isinstance(r, dict):
            continue
        idx = r.get("index")
        m = r.get("match")
        if isinstance(idx, (int, float)) and isinstance(m, bool):
            i = int(idx)
            if 1 <= i <= n:
                out[i - 1] = m
    return out


# Legacy module-level cache retained only so existing tests that reach in
# and poison it keep working; new code should use ``_client_for``.
_LLM_CLIENTS: dict[int, object] = {}


def _client_for(ctx: "Context", llm_filter: "LLMFilter | None" = None):
    """Return a cached LLMClient for the given filter's resolved (model, reasoning).

    The cache lives on the context itself (``ctx._llm_client_cache``) — no
    module-level stale-id risk. A filter with no overrides keys on
    ``(None, None)``, giving every default-config filter a single shared
    client. Filters that override ``model`` and/or ``reasoning_effort``
    get their own cached client, built lazily on first use.
    """
    model = getattr(llm_filter, "model", None) if llm_filter is not None else None
    reasoning = getattr(llm_filter, "reasoning_effort", None) if llm_filter is not None else None
    # Legacy poison path: if a test has set ``_LLM_CLIENTS[id(ctx)]`` directly,
    # honor it so existing tests keep working.
    legacy = _LLM_CLIENTS.get(id(ctx))
    if legacy is not None and model is None and reasoning is None:
        return legacy
    cache: dict[tuple[str | None, str | None], object] = ctx.__dict__.setdefault(
        "_llm_client_cache", {},
    )
    key = (model, reasoning)
    c = cache.get(key)
    if c is None:
        c = build_llm_client(
            ctx.config, ctx.budget, model=model, reasoning_effort=reasoning,
            cache=getattr(ctx, "cache", None),
        )
        cache[key] = c
    return c


def _format_exc(exc: BaseException) -> str:
    """Render an exception for the failure log, unwrapping ``tenacity.RetryError``.

    Tenacity wraps the last underlying exception in a ``RetryError``, so the
    raw ``str(exc)`` is just ``RetryError[<Future at 0x... state=finished raised ClientError>]``
    — useless for diagnosis. Reach into ``last_attempt`` to surface the real
    type + message instead.
    """
    try:
        from tenacity import RetryError
    except ImportError:  # pragma: no cover - tenacity is a hard dep
        RetryError = ()  # type: ignore[assignment]
    if isinstance(exc, RetryError) and exc.last_attempt is not None:
        try:
            inner = exc.last_attempt.exception()
        except Exception:
            inner = None
        if inner is not None:
            return f"{type(inner).__name__}: {str(inner)[:600]}"
    return f"{type(exc).__name__}: {str(exc)[:600]}"


def _run_one_batch(client, llm_filter: "LLMFilter", contents: list[str], ids: list[str]) -> dict[str, bool]:
    label = "Venues" if llm_filter.scope == "venue" else "Items"
    system = _VENUE_SYSTEM if llm_filter.scope == "venue" else _SYSTEM
    block = "\n".join(f"{i}. {c}" for i, c in enumerate(contents, 1))
    user = _USER.format(criterion=llm_filter.prompt, label=label, n=len(contents), block=block)
    try:
        resp = client.call(
            system, user,
            category=f"llm_{llm_filter.name}",
            response_schema=screening_json_schema(),
        )
        matches = _parse_matches(resp.text, len(contents))
        if matches is None:
            log.warning("LLM JSON parse failed for %s; defaulting %d to False", llm_filter.name, len(contents))
            matches = [False] * len(contents)
    except TypeError:
        # Legacy client that doesn't accept ``response_schema`` — fall back
        # to the bare ``call(system, user, ...)`` signature so mock clients
        # in tests keep working without forcing every fake to grow the new
        # kwarg.
        try:
            resp = client.call(system, user, category=f"llm_{llm_filter.name}")
            matches = _parse_matches(resp.text, len(contents))
            if matches is None:
                log.warning("LLM JSON parse failed for %s; defaulting %d to False", llm_filter.name, len(contents))
                matches = [False] * len(contents)
        except Exception as exc:
            log.warning("LLM call failed for %s: %s", llm_filter.name, _format_exc(exc))
            matches = [False] * len(contents)
    except Exception as exc:
        log.warning("LLM call failed for %s: %s", llm_filter.name, exc)
        matches = [False] * len(contents)
    return {pid: m for pid, m in zip(ids, matches)}


def _warn_once_on_votes(ctx: "Context", llm_filter: "LLMFilter") -> None:
    """Emit a one-shot budget warning the first time a voting filter runs.

    Voting multiplies LLM spend ``votes``× per paper — users should see this
    clearly in the logs without having to audit every filter invocation.
    """
    if llm_filter.votes <= 1:
        return
    warned: set[str] = ctx.__dict__.setdefault("_voting_warned", set())
    if llm_filter.name in warned:
        return
    warned.add(llm_filter.name)
    log.warning(
        "LLMFilter %r runs %d votes per paper (min_accepts=%d); "
        "LLM spend for this filter scales %d×",
        llm_filter.name, llm_filter.votes, llm_filter.min_accepts, llm_filter.votes,
    )


def _warn_once_on_formula(ctx: "Context", llm_filter: "LLMFilter", n_queries: int) -> None:
    """Emit a one-shot budget warning the first time a formula filter runs.

    Formula mode multiplies LLM spend by the number of distinct sub-queries
    (times ``votes`` if voting is also enabled). Users should see the
    combined multiplier clearly in the logs.
    """
    warned: set[str] = ctx.__dict__.setdefault("_formula_warned", set())
    if llm_filter.name in warned:
        return
    warned.add(llm_filter.name)
    multiplier = n_queries * max(1, llm_filter.votes)
    log.warning(
        "LLMFilter %r runs %d sub-queries × %d votes = %d× LLM calls per paper",
        llm_filter.name, n_queries, max(1, llm_filter.votes), multiplier,
    )


def dispatch_batch(
    papers: list["PaperRecord"],
    llm_filter: "LLMFilter",
    ctx: "Context",
) -> dict[str, bool]:
    """Run ``llm_filter`` over ``papers`` with batching + concurrency.

    Returns ``{paper_id: bool}`` where the boolean reflects the aggregate
    voting outcome: ``True`` iff the paper collected at least
    ``llm_filter.min_accepts`` accepts across ``llm_filter.votes`` independent
    runs. Failed votes (exceptions in the LLM call or unparseable JSON) count
    as reject votes — the safe default.

    In **formula mode** (when ``llm_filter._formula`` is set), each named
    sub-query runs independently through the same simple path as a
    single-prompt filter; the final per-paper verdict is the Boolean
    evaluation of ``llm_filter._formula`` over those sub-query results.
    """
    if not papers:
        return {}

    if getattr(llm_filter, "_formula", None) is not None:
        return _dispatch_formula(papers, llm_filter, ctx)

    return _dispatch_simple(papers, llm_filter, ctx)


def _dispatch_formula(
    papers: list["PaperRecord"],
    llm_filter: "LLMFilter",
    ctx: "Context",
) -> dict[str, bool]:
    """Run each named sub-query independently and combine via the Boolean formula."""
    from citeclaw.filters.atoms.llm_query import LLMFilter  # local import to avoid cycles

    formula = llm_filter._formula
    # Only run the queries the formula actually references — extras are ignored.
    names = sorted(formula.query_names())
    _warn_once_on_formula(ctx, llm_filter, len(names))

    sub_verdicts: dict[str, dict[str, bool]] = {}
    for qname in names:
        sub_prompt = llm_filter.queries[qname]
        sub_filter = LLMFilter(
            name=f"{llm_filter.name}::{qname}",
            scope=llm_filter.scope,
            prompt=sub_prompt,
            model=llm_filter.model,
            reasoning_effort=llm_filter.reasoning_effort,
            votes=llm_filter.votes,
            min_accepts=llm_filter.min_accepts,
        )
        sub_verdicts[qname] = _dispatch_simple(papers, sub_filter, ctx)

    out: dict[str, bool] = {}
    for p in papers:
        values = {
            qname: bool(verdicts.get(p.paper_id, False))
            for qname, verdicts in sub_verdicts.items()
        }
        out[p.paper_id] = formula.evaluate(values)
    return out


def _prefetch_full_text_if_needed(
    papers: list["PaperRecord"],
    llm_filter: "LLMFilter",
    ctx: "Context",
) -> None:
    """For ``scope: full_text`` filters, populate ``paper.full_text`` from
    cached or freshly-fetched PDF bodies BEFORE the LLM dispatch loop.

    The fetcher lives on the Context as a singleton (one per run) so
    multiple full_text filters in the same pipeline share the same
    httpx connection pool and the same in-process cache.
    """
    if llm_filter.scope != "full_text":
        return
    fetcher = ctx.__dict__.get("_pdf_fetcher")
    if fetcher is None:
        from citeclaw.clients.pdf import PdfFetcher
        fetcher = PdfFetcher(ctx.cache)
        ctx.__dict__["_pdf_fetcher"] = fetcher
    # Only prefetch papers that haven't been hydrated yet — re-running
    # the same filter on the same papers should be a no-op.
    needs_fetch = [p for p in papers if getattr(p, "full_text", None) is None]
    if not needs_fetch:
        return
    text_by_id = fetcher.prefetch(needs_fetch, max_workers=4)
    for p in needs_fetch:
        text = text_by_id.get(p.paper_id)
        if text is not None:
            p.full_text = text


def _dispatch_simple(
    papers: list["PaperRecord"],
    llm_filter: "LLMFilter",
    ctx: "Context",
) -> dict[str, bool]:
    """Single-prompt dispatch path with voting. Used directly by
    :func:`dispatch_batch` in the non-formula case, and also by the formula
    dispatcher for each named sub-query.
    """
    cfg = ctx.config
    client = _client_for(ctx, llm_filter)
    votes = max(1, llm_filter.votes)
    min_accepts = max(1, llm_filter.min_accepts)
    _warn_once_on_votes(ctx, llm_filter)
    # PH-06: prefetch PDF bodies for full_text scope so content_for has
    # something to read. No-op for every other scope.
    _prefetch_full_text_if_needed(papers, llm_filter, ctx)

    # ------------------------------------------------------------------
    # Venue scope: dedup by venue string. Cache stores ``list[bool]`` of
    # length ``votes`` — one entry per independent vote.
    # ------------------------------------------------------------------
    if llm_filter.scope == "venue":
        venue_cache: dict[str, dict[str, list[bool]]] = ctx.__dict__.setdefault(
            "_venue_llm_cache", {},
        )
        cache = venue_cache.setdefault(llm_filter.name, {})

        unique: list[str] = []
        for p in papers:
            v = p.venue or ""
            if v and v not in cache and v not in unique:
                unique.append(v)
        if unique:
            batch_size = cfg.llm_batch_size
            batches = [unique[i:i + batch_size] for i in range(0, len(unique), batch_size)]
            # Pre-seed: each unique venue gets an empty list that we append
            # vote results to in-order as futures complete.
            tallies: dict[str, list[bool]] = {v: [] for v in unique}
            # Submit votes × batches tasks. Tag each future with its vote
            # index so we can merge results back into per-venue lists.
            with ThreadPoolExecutor(max_workers=cfg.llm_concurrency) as pool:
                futures = []
                for _ in range(votes):
                    for b in batches:
                        futures.append(pool.submit(_run_one_batch, client, llm_filter, b, b))
                for fut in as_completed(futures):
                    result = fut.result()
                    for venue, verdict in result.items():
                        tallies[venue].append(bool(verdict))
            # Write tallies into the persistent cache.
            for venue, results in tallies.items():
                cache[venue] = results

        out: dict[str, bool] = {}
        for p in papers:
            v = p.venue or ""
            if not v:
                out[p.paper_id] = False
                continue
            results = cache.get(v, [])
            accepts = sum(1 for r in results if r)
            out[p.paper_id] = accepts >= min_accepts
        # Venue scope dedups by venue, not paper, so we can't drip-feed
        # the inner bar per future the way the title path does. Tick to
        # completion in one shot once the venue cache is populated.
        dash = getattr(ctx, "dashboard", None)
        if dash is not None:
            dash.tick_inner(len(papers))
        return out

    # ------------------------------------------------------------------
    # Title / title_abstract scope. Run ``votes`` independent passes and
    # tally accepts per paper.
    # ------------------------------------------------------------------
    contents = [llm_filter.content_for(p) for p in papers]
    ids = [p.paper_id for p in papers]
    # PH-06: full_text scope packs ~25K tokens of body per paper, so
    # the usual ``llm_batch_size`` (default 20) blows past every model's
    # context window. Force batch_size=1 — each paper gets its own LLM
    # call. Concurrency (``llm_concurrency``) still parallelises across
    # papers, so throughput stays reasonable; only the per-call payload
    # shrinks. All other scopes (title, title_abstract, venue) keep
    # using the configured batch size since their per-paper payloads
    # are tiny.
    batch_size = 1 if llm_filter.scope == "full_text" else cfg.llm_batch_size
    batches = [
        (contents[i:i + batch_size], ids[i:i + batch_size])
        for i in range(0, len(papers), batch_size)
    ]
    tally: dict[str, int] = {pid: 0 for pid in ids}

    # Dashboard: apply_block already called begin_phase(layer.name, total=len(papers))
    # before invoking us. If voting > 1, the real number of batch results we
    # will see is votes × len(papers), so retotal accordingly. Tick per
    # future as batches complete so the bar moves smoothly.
    dash = getattr(ctx, "dashboard", None)
    if dash is not None and votes > 1:
        dash.retotal_phase(votes * len(papers))

    with ThreadPoolExecutor(max_workers=cfg.llm_concurrency) as pool:
        futures = []
        for _ in range(votes):
            for c, i in batches:
                futures.append(pool.submit(_run_one_batch, client, llm_filter, c, i))
        # Merge-in-consumer: increment tallies in the main thread as each
        # future completes. No locks needed — dict increment is only touched
        # from this one thread.
        for fut in as_completed(futures):
            result = fut.result()
            if dash is not None:
                dash.tick_inner(len(result))
            for pid, verdict in result.items():
                if verdict:
                    tally[pid] = tally.get(pid, 0) + 1
    return {pid: tally.get(pid, 0) >= min_accepts for pid in ids}
