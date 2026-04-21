"""ResolveSeeds — pre-pipeline seed expansion.

Reads ``ctx.config.seed_papers`` (a mix of ``{paper_id: ...}``- and
``{title: ...}``-style entries) and writes a flat list of resolved S2
paper IDs to ``ctx.resolved_seed_ids`` for ``LoadSeeds`` to consume.

Per-entry resolution:

  * ``{paper_id: ...}`` — kept as-is (the user already named the
    canonical S2 ID).
  * ``{title: ...}`` — looked up via ``ctx.s2.search_match(title)`` and
    converted to its primary ``paperId``.

When ``include_siblings=True``, every primary paper is then expanded
to its preprint / published siblings via TWO complementary lookups:

  1. **external_ids walk**: fetch the primary's metadata, walk its
     ``external_ids`` dict (DOI / ArXiv), and try to resolve each
     external ID through ``ctx.s2.fetch_metadata`` as a *separate* S2
     paper. Catches the cases where S2 has separate records keyed by
     ``DOI:10.x`` and ``ARXIV:2301.x`` for the same paper.

  2. **title round-trip**: take the primary's title and feed it back
     into ``ctx.s2.search_match``. If the best-match returns a
     *different* ``paper_id`` with overlapping authors, treat it as a
     sibling (typically a preprint↔published pair that S2 records as
     two separate papers without cross-linking them via external_ids).
     This catches the case the user motivated: "我给了正式的doi, 那或许
     可以读title然后又拿这个title返回去搜preprint".

The motivation is the same in both cases: S2 sometimes only has
citation/reference data on one of the preprint/published pair, and
loading both maximises graph coverage before ``MergeDuplicates``
collapses them later in the pipeline. The author-overlap check on the
title round-trip filters out unrelated papers that happen to share a
similar title.

The step never adds anything to ``ctx.collection`` directly — that's
``LoadSeeds``'s job. ResolveSeeds is purely a preprocessor.
"""

from __future__ import annotations

import logging

from citeclaw.models import PaperRecord
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.resolve_seeds")


# Mapping from PaperRecord.external_ids keys → the prefix the S2
# /paper/{id} endpoint expects. Only DOI and ArXiv are surfaced
# because those are the two preprint↔published pair patterns we
# actually see in practice.
_SIBLING_PREFIX_BY_KEY = {
    "DOI": "DOI:",
    "ARXIV": "ARXIV:",
    "ARXIVID": "ARXIV:",
}


class ResolveSeeds:
    """Resolve mixed-shape ``seed_papers`` entries to canonical S2 IDs."""

    name = "ResolveSeeds"

    def __init__(self, *, include_siblings: bool = False) -> None:
        self.include_siblings = include_siblings

    def _expand_siblings(
        self,
        primary_id: str,
        primary_rec: PaperRecord,
        ctx,
        seen: set[str],
    ) -> list[str]:
        """Return new sibling paper_ids found via the primary's
        ``external_ids``. Skips siblings that resolve to the primary
        itself or to a paper already in ``seen``."""
        out: list[str] = []
        for raw_key, raw_val in (primary_rec.external_ids or {}).items():
            key = (raw_key or "").strip().upper()
            value = (raw_val or "").strip()
            if not value:
                continue
            prefix = _SIBLING_PREFIX_BY_KEY.get(key)
            if not prefix:
                continue
            sibling_query = f"{prefix}{value}"
            try:
                sib_rec = ctx.s2.fetch_metadata(sibling_query)
            except Exception as exc:  # noqa: BLE001 — sibling miss is fine
                log.info(
                    "ResolveSeeds: sibling lookup %s failed: %s",
                    sibling_query, exc,
                )
                continue
            sib_id = (sib_rec.paper_id or "").strip()
            if not sib_id or sib_id == primary_id:
                continue
            if sib_id in seen:
                continue
            seen.add(sib_id)
            out.append(sib_id)
        return out

    def _expand_via_title_roundtrip(
        self,
        primary_id: str,
        primary_rec: PaperRecord,
        ctx,
        seen: set[str],
    ) -> list[str]:
        """Find a sibling preprint↔published pair via title search.

        Pulls the primary's title and feeds it back into
        ``ctx.s2.search_match``. If the best-match returns a different
        ``paper_id`` with overlapping authors, treat it as a sibling.

        Anti-noise filters:
          * Skip titles shorter than 30 characters (too generic to
            match cleanly — search_match will return the wrong paper).
          * Require at least one shared ``authorId`` between the
            primary and the matched record. Without this filter, a
            short title could match a totally different paper.
        """
        title = (primary_rec.title or "").strip()
        if len(title) < 30:
            return []
        try:
            match = ctx.s2.search_match(title)
        except Exception as exc:  # noqa: BLE001 — sibling miss is fine
            log.info(
                "ResolveSeeds: title roundtrip for %s failed: %s",
                primary_id, exc,
            )
            return []
        if not isinstance(match, dict):
            return []
        matched_id = match.get("paperId")
        if not isinstance(matched_id, str) or not matched_id:
            return []
        if matched_id == primary_id or matched_id in seen:
            return []

        # Author-overlap check: filter out unrelated papers that
        # happen to share a similar title. ``primary_aids`` may be
        # empty if S2 didn't return author IDs for the primary; in
        # that case we accept the match (no info → don't reject).
        primary_aids: set[str] = {
            (a.get("authorId") or "").strip()
            for a in (primary_rec.authors or [])
            if isinstance(a, dict) and a.get("authorId")
        }
        matched_aids: set[str] = set()
        for a in match.get("authors") or []:
            if isinstance(a, dict) and a.get("authorId"):
                matched_aids.add(str(a["authorId"]))
        if primary_aids and matched_aids and not (primary_aids & matched_aids):
            log.info(
                "ResolveSeeds: title roundtrip %s for %s rejected "
                "(no author overlap)",
                matched_id, primary_id,
            )
            return []

        log.info(
            "ResolveSeeds: title roundtrip found sibling %s for %s",
            matched_id, primary_id,
        )
        seen.add(matched_id)
        return [matched_id]

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        """Walk ``ctx.config.seed_papers`` and write resolved IDs to ctx.

        Two-step per-entry logic: (1) pick the primary paper_id by
        direct lookup or title-match; (2) when ``include_siblings`` is
        true, expand via the external_ids walk + title round-trip. The
        signal is passed through unchanged — ResolveSeeds is purely a
        preprocessor that populates ``ctx.resolved_seed_ids``;
        :class:`LoadSeeds` consumes that list to fetch metadata and
        seed ``ctx.collection``.
        """
        cfg = ctx.config
        resolved: list[str] = []
        seen: set[str] = set()
        primaries_resolved = 0
        siblings_added = 0
        unresolved_titles = 0

        for sp in cfg.seed_papers:
            # Step 1: pick the primary paper_id (direct or via title match).
            primary_id: str | None = None
            if sp.paper_id and sp.paper_id.strip():
                primary_id = sp.paper_id.strip()
            elif sp.title:
                try:
                    match = ctx.s2.search_match(sp.title)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "ResolveSeeds: search_match(%s) failed: %s",
                        sp.title[:60], exc,
                    )
                    unresolved_titles += 1
                    continue
                if not isinstance(match, dict):
                    log.warning(
                        "ResolveSeeds: no S2 match for title %r", sp.title[:60],
                    )
                    unresolved_titles += 1
                    continue
                pid = match.get("paperId")
                if not isinstance(pid, str) or not pid:
                    log.warning(
                        "ResolveSeeds: search_match returned no paperId for %r",
                        sp.title[:60],
                    )
                    unresolved_titles += 1
                    continue
                primary_id = pid
            else:
                log.warning(
                    "ResolveSeeds: skipping seed with neither paper_id nor title",
                )
                continue

            if primary_id not in seen:
                seen.add(primary_id)
                resolved.append(primary_id)
                primaries_resolved += 1

            # Step 2: optionally expand siblings via external_ids walk
            # AND title roundtrip. Both run when include_siblings=True.
            if not self.include_siblings:
                continue
            try:
                primary_rec = ctx.s2.fetch_metadata(primary_id)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "ResolveSeeds: fetch_metadata(%s) failed for sibling expansion: %s",
                    primary_id, exc,
                )
                continue
            new_via_extids = self._expand_siblings(
                primary_id, primary_rec, ctx, seen,
            )
            new_via_title = self._expand_via_title_roundtrip(
                primary_id, primary_rec, ctx, seen,
            )
            resolved.extend(new_via_extids)
            resolved.extend(new_via_title)
            siblings_added += len(new_via_extids) + len(new_via_title)

        ctx.resolved_seed_ids = resolved
        return StepResult(
            signal=signal,
            in_count=len(signal),
            stats={
                "input_seeds": len(cfg.seed_papers),
                "primaries_resolved": primaries_resolved,
                "siblings_added": siblings_added,
                "unresolved_titles": unresolved_titles,
                "total_resolved": len(resolved),
                "include_siblings": self.include_siblings,
            },
        )
