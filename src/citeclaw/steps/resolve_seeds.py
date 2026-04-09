"""ResolveSeeds — pre-pipeline seed expansion (PC-04).

Reads ``ctx.config.seed_papers`` (which may now mix
``{paper_id: ...}``- and ``{title: ...}``-style entries thanks to
PC-04's schema relaxation) and writes a flat list of resolved S2 paper
IDs to ``ctx.resolved_seed_ids`` for ``LoadSeeds`` to consume.

Per-entry resolution:

  * ``{paper_id: ...}`` — kept as-is (the user already named the
    canonical S2 ID).
  * ``{title: ...}`` — looked up via ``ctx.s2.search_match(title)`` and
    converted to its primary ``paperId``.

When ``include_siblings=True``, every primary paper is then expanded
to its preprint / published siblings: the step fetches the primary's
metadata, walks its ``external_ids`` dict (DOI / ArXiv), and tries to
resolve each external ID through ``ctx.s2.fetch_metadata`` as a
*separate* S2 paper. If the lookup returns a different ``paper_id``,
the sibling is added to the result list. The motivation is that S2
sometimes only has citation/reference data on one of the
preprint/published pair, and loading both maximises graph coverage
before ``MergeDuplicates`` collapses them later in the pipeline.

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

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
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

            # Step 2: optionally expand siblings via external_ids.
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
            new_siblings = self._expand_siblings(primary_id, primary_rec, ctx, seen)
            resolved.extend(new_siblings)
            siblings_added += len(new_siblings)

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
