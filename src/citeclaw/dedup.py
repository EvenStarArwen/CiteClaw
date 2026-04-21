"""Detect and merge preprint-vs-published duplicate papers.

Scientific papers regularly appear twice in a CiteClaw collection: once as
a preprint (arXiv, bioRxiv, medRxiv) and once as the peer-reviewed
version in a conference or journal. These are the *same work* and
double-counting them corrupts every downstream signal (rerank scores,
PageRank, citation counts, graph topology).

The detector fuses four signals to catch duplicates:

1. **Shared ``externalIds``** — two records listing the same DOI or the
   same ArXiv ID in ``paper.external_ids`` are almost certainly the same
   work. This is the highest-precision signal.
2. **Jaro-Winkler title similarity > 0.95** — catches the case where S2
   has stored the preprint and the published version with slightly
   different titles (punctuation, subtitle drops, etc.).
3. **Same first-author S2 ID + year window** — papers sharing the
   first author within a small year window (≤ ``year_window``) are
   candidates for further checks.
4. **SPECTER2 cosine > 0.98** — if embeddings are available, a near-1
   cosine is conclusive evidence that the two papers are the same work
   at the semantic level.

The detector uses *blocking*: candidates are grouped by first-author S2
ID (and separately by external IDs), so the pairwise comparison is
O(sum(k^2)) across blocks rather than O(N²) over the whole collection.

Once clusters are found, :func:`merge_cluster` folds every non-canonical
record into the canonical (peer-reviewed, if any) record:

- The canonical record absorbs each duplicate's paper_id into its
  ``aliases`` list.
- ``references`` and ``supporting_papers`` are unioned (preserving
  order).
- ``external_ids`` are merged.
- Metadata (abstract, venue, citation_count) is filled in from the
  duplicate only when the canonical is missing it.
- Every remaining record in the collection has references pointing at
  the absorbed IDs rewritten to the canonical.
- The duplicate record is removed from the collection.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

from citeclaw.filters.measures.semantic_sim import _cosine
from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.dedup")


# ---------------------------------------------------------------------------
# Jaro-Winkler similarity (pure Python, no external deps)
# ---------------------------------------------------------------------------


def jaro_similarity(a: str, b: str) -> float:
    """Standard Jaro similarity between two strings, in ``[0, 1]``.

    Returns 1.0 for identical strings (including the empty/empty case),
    0.0 when there are no matching characters.
    """
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    match_window = max(la, lb) // 2 - 1
    if match_window < 0:
        match_window = 0
    a_matches = [False] * la
    b_matches = [False] * lb
    matches = 0
    for i in range(la):
        lo = max(0, i - match_window)
        hi = min(i + match_window + 1, lb)
        for j in range(lo, hi):
            if b_matches[j]:
                continue
            if a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    # Count transpositions.
    transpositions = 0
    k = 0
    for i in range(la):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    return (
        matches / la
        + matches / lb
        + (matches - transpositions) / matches
    ) / 3.0


def jaro_winkler_similarity(a: str, b: str, *, prefix_weight: float = 0.1) -> float:
    """Jaro-Winkler similarity with a prefix boost (default 0.1, max 4 chars)."""
    j = jaro_similarity(a, b)
    # Common prefix, capped at 4 chars as in the standard JW definition.
    prefix = 0
    for i in range(min(len(a), len(b), 4)):
        if a[i] == b[i]:
            prefix += 1
        else:
            break
    return j + prefix * prefix_weight * (1.0 - j)


# ---------------------------------------------------------------------------
# Title normalisation — strips noise that doesn't affect identity
# ---------------------------------------------------------------------------


def normalize_title(title: str) -> str:
    """Lowercase, collapse whitespace, strip non-alphanumerics."""
    if not title:
        return ""
    out_chars: list[str] = []
    prev_space = False
    for ch in title.lower():
        if ch.isalnum():
            out_chars.append(ch)
            prev_space = False
        elif ch.isspace():
            if not prev_space and out_chars:
                out_chars.append(" ")
                prev_space = True
    s = "".join(out_chars).strip()
    return s


# ---------------------------------------------------------------------------
# Canonical selection within a cluster
# ---------------------------------------------------------------------------


_PREPRINT_VENUES = {"arxiv", "biorxiv", "medrxiv", "chemrxiv", "ssrn"}


def _is_preprint_venue(venue: str | None) -> bool:
    """True when ``venue`` contains any of the known preprint-server names.

    Substring match is intentional — S2 venue strings can include
    extra decoration (``"arXiv 2301.00001"``, ``"bioRxiv preprint"``)
    so exact-match would miss legitimate preprint records.
    """
    if not venue:
        return False
    v = venue.lower()
    return any(p in v for p in _PREPRINT_VENUES)


def pick_canonical(cluster: list[PaperRecord]) -> PaperRecord:
    """Pick the canonical record in a cluster of duplicates.

    Preference order:
      1. Non-preprint venue (peer-reviewed).
      2. Higher citation_count (treat ``None`` as 0).
      3. Longer abstract (more complete metadata).
      4. Lexicographically smaller ``paper_id`` (deterministic tiebreak).
    """
    def _key(p: PaperRecord) -> tuple:
        preprint = _is_preprint_venue(p.venue)
        return (
            0 if not preprint else 1,  # peer-reviewed first
            -(p.citation_count or 0),
            -len(p.abstract or ""),
            p.paper_id,
        )

    return min(cluster, key=_key)


# ---------------------------------------------------------------------------
# Cluster detection — blocking + pairwise signal fusion
# ---------------------------------------------------------------------------


def _first_author_id(rec: PaperRecord) -> str | None:
    """Return the first author's S2 ``authorId``, or ``None`` if unknown.

    Skips name-only author entries (S2 occasionally returns
    ``{"name": "J. Smith"}`` without an authorId when the author
    hasn't been disambiguated). The returned id is the blocking key
    used by :func:`detect_duplicate_clusters` to narrow pairwise
    comparisons.
    """
    for a in rec.authors or []:
        if isinstance(a, dict):
            aid = a.get("authorId")
            if aid:
                return str(aid)
    return None


_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)
_ARXIV_URL_PREFIXES = (
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
    "arxiv:",
)


def _normalise_external_value(namespace: str, value: str) -> str:
    """Strip URL/namespace prefixes from an external-id value.

    S2's native ``externalIds`` dict returns bare IDs, but values that
    originate in PDF extraction or Crossref can include ``https://doi.org/``
    wrappers. Without stripping, two papers with the same DOI but different
    storage formats would fail to match in the external-ID union-find
    pass.
    """
    v = (value or "").strip()
    if not v:
        return ""
    lower = v.lower()
    ns = namespace.lower()
    if ns == "doi":
        for prefix in _DOI_URL_PREFIXES:
            if lower.startswith(prefix):
                v = v[len(prefix):]
                break
    elif ns == "arxiv":
        for prefix in _ARXIV_URL_PREFIXES:
            if lower.startswith(prefix):
                v = v[len(prefix):]
                break
    return v.strip()


def _external_keys(rec: PaperRecord) -> list[str]:
    """Extract identity keys from external_ids.

    Each key is provider-scoped (e.g. ``DOI:10.1/abc``, ``ArXiv:2301.00001``)
    so a shared DOI doesn't collide with an identical-looking PMID. Values
    are normalised via :func:`_normalise_external_value` so URL-prefixed
    forms cluster with bare forms.
    """
    out: list[str] = []
    for k, v in (rec.external_ids or {}).items():
        if v:
            normalised = _normalise_external_value(k, str(v))
            if normalised:
                out.append(f"{k}:{normalised}".lower())
    return out


def _pair_is_duplicate(
    a: PaperRecord,
    b: PaperRecord,
    *,
    title_threshold: float,
    semantic_threshold: float,
    year_window: int,
    embeddings: dict[str, list[float] | None] | None,
) -> bool:
    """Return True if the pair looks like a preprint/published duplicate.

    Fuses the four signals described in the module docstring. A single
    high-confidence signal (external-id match, or semantic cosine > 0.98)
    is enough on its own; title similarity and first-author match are
    combined with the year window as a softer check.
    """
    # Signal 1 — shared external id (DOI / ArXiv / PubMed / ...)
    keys_a = set(_external_keys(a))
    keys_b = set(_external_keys(b))
    if keys_a & keys_b:
        return True

    # Signal 4 — SPECTER2 cosine > threshold (strong alone)
    if embeddings is not None and semantic_threshold <= 1.0:
        va = embeddings.get(a.paper_id)
        vb = embeddings.get(b.paper_id)
        if va and vb:
            cos = _cosine(va, vb)
            if cos is not None and cos >= semantic_threshold:
                return True

    # Signals 2 & 3 combined: title similarity + first-author + year window
    norm_a = normalize_title(a.title)
    norm_b = normalize_title(b.title)
    if norm_a and norm_b:
        title_sim = jaro_winkler_similarity(norm_a, norm_b)
        if title_sim >= title_threshold:
            # Year window check (if both years are known).
            if a.year is not None and b.year is not None:
                if abs(a.year - b.year) > year_window:
                    return False
            # First-author agreement (if both sides advertise one).
            fa_a = _first_author_id(a)
            fa_b = _first_author_id(b)
            if fa_a and fa_b and fa_a != fa_b:
                return False
            return True
    return False


def detect_duplicate_clusters(
    collection: dict[str, PaperRecord],
    *,
    title_threshold: float = 0.95,
    semantic_threshold: float = 0.98,
    year_window: int = 1,
    embeddings: dict[str, list[float] | None] | None = None,
) -> list[list[str]]:
    """Find clusters of likely-duplicate records in ``collection``.

    Returns a list of clusters; each cluster is a list of paper IDs with
    at least 2 members. Singletons are omitted. The algorithm uses
    union-find over candidate pairs generated by two blocking passes:

      * **External-id block**: papers sharing any DOI/ArXiv/PMID key are
        unioned directly (no further checks needed).
      * **First-author block**: for each author, pairwise check is run
        over the subset of the collection authored by that person. This
        keeps the total work O(sum k²) instead of O(N²).
    """
    ids = list(collection.keys())
    parent: dict[str, str] = {pid: pid for pid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # ----- Block 1: external IDs -----
    by_ext: dict[str, list[str]] = defaultdict(list)
    for pid, rec in collection.items():
        for key in _external_keys(rec):
            by_ext[key].append(pid)
    for pids in by_ext.values():
        if len(pids) < 2:
            continue
        first = pids[0]
        for other in pids[1:]:
            union(first, other)

    # ----- Block 2: first-author (only among pairs not already unioned) -----
    by_author: dict[str, list[str]] = defaultdict(list)
    for pid, rec in collection.items():
        aid = _first_author_id(rec)
        if aid:
            by_author[aid].append(pid)
    # Also consider a catch-all "unknown author" block — smaller collections
    # often have incomplete metadata and we don't want to miss duplicates
    # there. Iterate only when the block is small enough to pair-check.
    unknown_block = [pid for pid, rec in collection.items() if not _first_author_id(rec)]
    if unknown_block and len(unknown_block) <= 500:
        by_author["__unknown__"] = unknown_block

    for aid, pids in by_author.items():
        if len(pids) < 2:
            continue
        for i in range(len(pids)):
            a_rec = collection[pids[i]]
            for j in range(i + 1, len(pids)):
                b_rec = collection[pids[j]]
                if find(pids[i]) == find(pids[j]):
                    continue  # already in the same cluster
                if _pair_is_duplicate(
                    a_rec, b_rec,
                    title_threshold=title_threshold,
                    semantic_threshold=semantic_threshold,
                    year_window=year_window,
                    embeddings=embeddings,
                ):
                    union(pids[i], pids[j])

    # Collect components.
    components: dict[str, list[str]] = defaultdict(list)
    for pid in ids:
        components[find(pid)].append(pid)
    return [sorted(group) for group in components.values() if len(group) > 1]


# ---------------------------------------------------------------------------
# Cluster merge — destructive on the collection
# ---------------------------------------------------------------------------


def _merge_references_inplace(
    canonical: PaperRecord,
    duplicate: PaperRecord,
) -> None:
    """Union reference lists, preserving canonical order first."""
    seen = set(canonical.references)
    for r in duplicate.references:
        if r and r not in seen:
            canonical.references.append(r)
            seen.add(r)
    seen_sp = set(canonical.supporting_papers)
    for s in duplicate.supporting_papers:
        if s and s not in seen_sp:
            canonical.supporting_papers.append(s)
            seen_sp.add(s)


def _fill_missing_metadata(canonical: PaperRecord, duplicate: PaperRecord) -> None:
    """Fill in canonical metadata from the duplicate when canonical lacks it."""
    if not canonical.abstract and duplicate.abstract:
        canonical.abstract = duplicate.abstract
    if not canonical.venue and duplicate.venue:
        canonical.venue = duplicate.venue
    if canonical.year is None and duplicate.year is not None:
        canonical.year = duplicate.year
    if canonical.citation_count is None and duplicate.citation_count is not None:
        canonical.citation_count = duplicate.citation_count
    elif (
        duplicate.citation_count is not None
        and canonical.citation_count is not None
        and duplicate.citation_count > canonical.citation_count
    ):
        # Keep the higher count so rerank isn't penalised by a stale value.
        canonical.citation_count = duplicate.citation_count
    if not canonical.pdf_url and duplicate.pdf_url:
        canonical.pdf_url = duplicate.pdf_url


def merge_cluster(
    collection: dict[str, PaperRecord],
    cluster: Iterable[str],
    *,
    alias_map: dict[str, str],
) -> str | None:
    """Fold every non-canonical record in ``cluster`` into the canonical one.

    Returns the canonical paper_id, or ``None`` if the cluster had fewer
    than 2 surviving records (e.g. every id was already absorbed into
    another cluster). Destructive: removes non-canonical records from
    ``collection`` and updates ``alias_map`` so downstream graph
    construction can rewrite edge targets.
    """
    records = [collection[pid] for pid in cluster if pid in collection]
    if len(records) < 2:
        return records[0].paper_id if records else None

    canonical = pick_canonical(records)
    canonical_id = canonical.paper_id
    for rec in records:
        if rec.paper_id == canonical_id:
            continue
        _fill_missing_metadata(canonical, rec)
        _merge_references_inplace(canonical, rec)
        # Merge aliases (transitive): if the duplicate had its own aliases,
        # absorb them too.
        if rec.paper_id not in canonical.aliases:
            canonical.aliases.append(rec.paper_id)
        for a in rec.aliases:
            if a and a not in canonical.aliases and a != canonical_id:
                canonical.aliases.append(a)
        # Merge external_ids.
        for k, v in (rec.external_ids or {}).items():
            canonical.external_ids.setdefault(k, v)
        alias_map[rec.paper_id] = canonical_id
        del collection[rec.paper_id]

    # Rewrite references in every *other* remaining record that pointed at an
    # absorbed ID so downstream graph edges don't dangle.
    for rec in collection.values():
        if not rec.references:
            continue
        new_refs: list[str] = []
        seen: set[str] = set()
        changed = False
        for r in rec.references:
            new_r = alias_map.get(r, r)
            if new_r != r:
                changed = True
            if new_r in seen:
                changed = True
                continue
            seen.add(new_r)
            new_refs.append(new_r)
        if changed:
            rec.references = new_refs
    return canonical_id


__all__ = [
    "jaro_similarity",
    "jaro_winkler_similarity",
    "normalize_title",
    "pick_canonical",
    "detect_duplicate_clusters",
    "merge_cluster",
]
