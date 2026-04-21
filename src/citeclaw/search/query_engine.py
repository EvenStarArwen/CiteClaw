"""Pure local query engine — AND-ed predicate filtering over PaperRecords.

The S2 ``/paper/search/bulk`` endpoint can express a useful but
limited subset of predicates: year ranges, venue names, citation
floors, fields-of-study tags, publication types. It cannot express:

  - regex matches over venue / title / abstract,
  - abstract text search at all (S2 only matches title-field tokens),
  - the union of two unrelated criteria,
  - "at least one of these tags" without sending them as the OR query.

The expansion family handles those cases by fetching a slightly-too-broad
superset from S2 and then trimming the result with :func:`apply_local_query`
before LLM screening. This module is pure: no S2, no LLM, no Context —
just a function from a list to a list.
"""

from __future__ import annotations

import re

from citeclaw.models import PaperRecord


def apply_local_query(
    papers: list[PaperRecord],
    *,
    venue_regex: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    min_citations: int | None = None,
    fields_of_study_any: list[str] | None = None,
    publication_types_any: list[str] | None = None,
    abstract_regex: str | None = None,
    title_regex: str | None = None,
) -> list[PaperRecord]:
    """Return the subset of ``papers`` matching every supplied predicate.

    Predicates default to ``None`` (skip). Each non-None predicate is
    AND-ed: a paper must match all of them to survive.

    **Strictness on missing metadata.** If a predicate is set and the
    paper's value is missing (``None`` for scalar fields, empty list
    for list fields), the paper is REJECTED — except for
    ``abstract_regex``, which is LENIENT (papers with ``abstract is
    None`` always pass). The lenient carve-out exists because S2
    frequently returns no abstract for a substantial fraction of its
    corpus, and dropping all of those papers would be a much harsher
    filter than the user intended.

    Regexes use :func:`re.search` semantics with ``re.IGNORECASE`` so
    callers don't need to anchor or worry about case.
    """
    venue_re = re.compile(venue_regex, re.IGNORECASE) if venue_regex else None
    title_re = re.compile(title_regex, re.IGNORECASE) if title_regex else None
    abstract_re = re.compile(abstract_regex, re.IGNORECASE) if abstract_regex else None
    fos_wanted = set(fields_of_study_any) if fields_of_study_any is not None else None
    types_wanted = (
        set(publication_types_any) if publication_types_any is not None else None
    )

    def matches(p: PaperRecord) -> bool:
        if year_min is not None and (p.year is None or p.year < year_min):
            return False
        if year_max is not None and (p.year is None or p.year > year_max):
            return False
        if min_citations is not None and (
            p.citation_count is None or p.citation_count < min_citations
        ):
            return False
        if venue_re is not None and (not p.venue or not venue_re.search(p.venue)):
            return False
        # ``title`` is always a string (default ""), so an empty title
        # simply fails to match a non-trivial regex — no None branch.
        if title_re is not None and not title_re.search(p.title):
            return False
        # LENIENT: missing abstract → pass; only present-but-mismatched rejects.
        if (
            abstract_re is not None
            and p.abstract is not None
            and not abstract_re.search(p.abstract)
        ):
            return False
        if fos_wanted is not None and fos_wanted.isdisjoint(p.fields_of_study):
            return False
        if types_wanted is not None and types_wanted.isdisjoint(p.publication_types):
            return False
        return True

    return [p for p in papers if matches(p)]
