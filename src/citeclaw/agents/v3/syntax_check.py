"""Pre-S2 syntax checker for V3 worker queries.

When a worker's proposed query has a Lucene-syntax error that S2 would
reject (400) or silently mis-interpret, catching it BEFORE the API call
lets us hand the worker a specific, actionable error message and get a
rewrite without burning an S2 request and an iteration.

Empirical rules, derived from direct S2 probes:

  - Suffix wildcard (``word*``) supported. Prefix wildcard (``*word``)
    also supported. But a single-letter stem (``a*``) returns HTTP 400.
  - Quoted-phrase proximity (``"A B"~N``) supported for any non-negative
    integer N. Unquoted ``word~N`` is a fuzzy-match (not proximity) and
    almost never what the worker wants on a single term.
  - A top-level query with ONLY negative clauses (``-X``) returns 400 —
    there has to be at least one mandatory (``+``) clause.
  - Unmatched parens or quotes yield parse errors.
  - Empty ``()`` or ``||`` groups are rejected.

Returns a list of :class:`SyntaxIssue`. Callers treat ``severity='error'``
as "must rewrite before S2 call"; ``severity='warning'`` is advisory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from citeclaw.agents.v3.query_translate import (
    _tokenize_top_level,
    decompose_query,
    parse_or_alternatives,
)


Severity = Literal["error", "warning"]


@dataclass
class SyntaxIssue:
    severity: Severity
    code: str
    message: str
    hint: str
    location: str = ""

    def render(self) -> str:
        tag = "ERROR" if self.severity == "error" else "warn"
        loc = f"  at `{self.location}`" if self.location else ""
        return f"[{tag}] [{self.code}] {self.message}{loc}\n        hint: {self.hint}"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_balanced_parens(q: str) -> list[SyntaxIssue]:
    depth = 0
    in_quote = False
    for c in q:
        if c == '"':
            in_quote = not in_quote
        elif in_quote:
            continue
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return [SyntaxIssue(
                    severity="error",
                    code="paren_unmatched_close",
                    message="closing ')' without a matching '('",
                    hint="remove the extra ')' or add the missing '('",
                )]
    if depth > 0:
        return [SyntaxIssue(
            severity="error",
            code="paren_unmatched_open",
            message=f"{depth} unclosed '(' — every '(' needs a matching ')'",
            hint="add the missing ')' to close each group",
        )]
    return []


def _check_balanced_quotes(q: str) -> list[SyntaxIssue]:
    if q.count('"') % 2 != 0:
        return [SyntaxIssue(
            severity="error",
            code="quote_unmatched",
            message='odd number of " characters — a quoted phrase was not closed',
            hint='every "..." must have a matching closing "',
        )]
    return []


def _check_short_wildcard(q: str) -> list[SyntaxIssue]:
    issues: list[SyntaxIssue] = []
    # Suffix wildcards with stem < 2 chars: S2 returns 400.
    for m in re.finditer(r"(?<![A-Za-z])([A-Za-z]{0,1})\*", q):
        stem = m.group(1)
        issues.append(SyntaxIssue(
            severity="error",
            code="wildcard_stem_too_short",
            message=f"wildcard '{stem}*' has a stem shorter than 2 characters",
            hint="extend the stem to at least 2 characters (S2 rejects 1-letter wildcards with HTTP 400)",
            location=f"{stem}*",
        ))
    # Prefix wildcards — S2 accepts but very slow; just warn.
    for m in re.finditer(r"(?<!\w)\*[A-Za-z]", q):
        issues.append(SyntaxIssue(
            severity="warning",
            code="wildcard_prefix",
            message=f"prefix wildcard '{m.group(0)}' is slow and rarely what you want",
            hint="prefer a suffix wildcard (word*) or spell out the variants explicitly in an OR group",
            location=m.group(0),
        ))
    return issues


def _check_wildcard_in_phrase(q: str) -> list[SyntaxIssue]:
    """Flag ``*`` inside a quoted phrase — Lucene phrase queries don't
    expand wildcards, so ``"edit* LNP"`` searches for the literal
    string "edit* LNP" (with the asterisk character), not
    "editing LNP" / "editor LNP".
    """
    issues: list[SyntaxIssue] = []
    for m in re.finditer(r'"([^"]+)"', q):
        phrase = m.group(1)
        if "*" in phrase:
            issues.append(SyntaxIssue(
                severity="error",
                code="wildcard_in_phrase",
                message=(
                    f'wildcard "*" inside quoted phrase "{phrase}" — phrase '
                    f"queries match the literal asterisk, they do not expand"
                ),
                hint=(
                    'either drop the quotes (use AND: edit* AND LNP) or drop '
                    'the wildcard inside the phrase'
                ),
                location=m.group(0),
            ))
    return issues


def _check_operator_word_in_phrase(q: str) -> list[SyntaxIssue]:
    """Flag AND / OR / NOT appearing as a word inside a quoted phrase.
    Probably the worker intended a Boolean combination but left the
    operators inside the quotes, so the query looks for the literal
    sequence of tokens (almost never useful).
    """
    issues: list[SyntaxIssue] = []
    for m in re.finditer(r'"([^"]+)"', q):
        phrase = m.group(1)
        if re.search(r"\b(AND|OR|NOT)\b", phrase):
            issues.append(SyntaxIssue(
                severity="warning",
                code="operator_word_in_phrase",
                message=(
                    f'quoted phrase "{phrase}" contains AND/OR/NOT as a '
                    f"literal word — almost certainly you meant a Boolean"
                ),
                hint=(
                    "if you meant a Boolean combination, move the operator "
                    "outside the quotes (e.g. \"prime editing\" AND \"pegRNA\")"
                ),
                location=m.group(0),
            ))
    return issues


def _check_proximity_on_single_word(q: str) -> list[SyntaxIssue]:
    issues: list[SyntaxIssue] = []
    # Proximity on a single-word quoted phrase is meaningless.
    for m in re.finditer(r'"([^"]+)"~\d+', q):
        phrase = m.group(1).strip()
        if phrase and " " not in phrase:
            issues.append(SyntaxIssue(
                severity="warning",
                code="proximity_single_word",
                message=f'"{phrase}"~N is a proximity query on a single word — the ~N has no effect',
                hint='drop the ~N, or use a phrase with 2+ words inside the quotes',
                location=m.group(0),
            ))
    # Unquoted `word~N` is a FUZZY match, almost never what the worker wants.
    for m in re.finditer(r"(?<!\")(?<!\w)([A-Za-z][A-Za-z0-9-]+)~\d+(?!\w)", q):
        issues.append(SyntaxIssue(
            severity="warning",
            code="fuzzy_term",
            message=f"'{m.group(0)}' is a FUZZY match (Levenshtein distance), not proximity",
            hint='proximity requires a quoted phrase: "A B"~N. If you meant fuzzy, drop the ~N and use the actual terms.',
            location=m.group(0),
        ))
    return issues


def _check_empty_groups(q: str) -> list[SyntaxIssue]:
    issues: list[SyntaxIssue] = []
    if re.search(r"\(\s*\)", q):
        issues.append(SyntaxIssue(
            severity="error",
            code="empty_group",
            message="empty parenthesised group '()'",
            hint="either remove the empty group or put at least one term in it",
        ))
    # "||" or "| |" means an empty OR alternative
    if re.search(r"\|\s*\|", q):
        issues.append(SyntaxIssue(
            severity="error",
            code="empty_or_alternative",
            message="empty OR alternative (two consecutive | with nothing between)",
            hint="remove one of the | operators or fill in the missing term",
        ))
    return issues


def _check_mandatory_clause(q: str) -> list[SyntaxIssue]:
    """S2 returns 400 if the query has only negative (-) clauses at top level."""
    clauses = _tokenize_top_level(q)
    if not clauses:
        return []
    any_mandatory = any(c.startswith("+") or (c and c[0] not in "+-") for c in clauses)
    if not any_mandatory:
        return [SyntaxIssue(
            severity="error",
            code="no_mandatory_clause",
            message="query contains only NOT clauses — S2 requires at least one positive term",
            hint="add a positive (AND) clause that describes what papers MUST contain, not just what they must avoid",
        )]
    return []


def _check_subsumed_phrases(q: str) -> list[SyntaxIssue]:
    """Flag OR groups where one alternative is a substring of another."""
    issues: list[SyntaxIssue] = []
    for clause in decompose_query(q):
        alts = parse_or_alternatives(clause)
        if len(alts) < 2:
            continue
        norms = [(a, _normalise_for_subsume(a)) for a in alts]
        for i, (a_i, n_i) in enumerate(norms):
            for j, (a_j, n_j) in enumerate(norms):
                if i == j or not n_i or not n_j or n_i == n_j:
                    continue
                if n_i in n_j and n_i != n_j:
                    issues.append(SyntaxIssue(
                        severity="warning",
                        code="or_subsumed",
                        message=(
                            f"OR group has a redundant alternative: "
                            f"'{a_j}' matches a superset of '{a_i}', so keeping '{a_j}' adds no papers"
                        ),
                        hint=f"drop '{a_j}' from the OR group",
                        location=clause,
                    ))
                    break  # report once per pair
    return issues


def _normalise_for_subsume(alt: str) -> str:
    """Lower-case + strip quotes + collapse whitespace — used to compare
    alternatives for substring subsumption."""
    s = alt.strip()
    if s and s[0] in "+-":
        s = s[1:].strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    # Drop proximity modifier
    s = re.sub(r"~\d+$", "", s)
    # Drop trailing wildcard — for subsumption comparison only.
    s = s.rstrip("*").strip()
    return re.sub(r"\s+", " ", s.lower())


def _check_ambiguous_short_acronym(q: str) -> list[SyntaxIssue]:
    """Advisory: a short (<=3-char) bare acronym in an OR group has a
    high risk of colliding with an unrelated field (PE / RT / ESM / NN)
    unless it's paired with a full-phrase disambiguator in the SAME group."""
    issues: list[SyntaxIssue] = []
    for clause in decompose_query(q):
        alts = parse_or_alternatives(clause)
        if not alts:
            continue
        # Is any alt a full-phrase (multi-word quoted or unquoted)?
        has_full_phrase = False
        for a in alts:
            stripped = _normalise_for_subsume(a)
            if " " in stripped and len(stripped) >= 6:
                has_full_phrase = True
                break
        for a in alts:
            s = _normalise_for_subsume(a)
            # short (≤3 char) all-letters token = acronym candidate
            if re.fullmatch(r"[a-z]{1,3}", s) and not has_full_phrase:
                issues.append(SyntaxIssue(
                    severity="warning",
                    code="bare_short_acronym",
                    message=(
                        f"'{a}' is a short bare acronym with no full-phrase disambiguator "
                        f"in the same OR group — high risk of matching unrelated fields"
                    ),
                    hint=(
                        f"either remove '{a}', or add its full expansion to the OR group "
                        f"(e.g. \"<full phrase>\" OR {a}), or gate it with an AND facet that "
                        f"forces the right domain"
                    ),
                    location=clause,
                ))
    return issues


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def syntax_check(lucene_query: str) -> list[SyntaxIssue]:
    """Run every syntax check and return all issues found.

    Caller decides what to do; the worker driver treats any
    ``severity='error'`` issue as a blocker and hands the rendered
    messages back to the LLM for a rewrite.
    """
    q = (lucene_query or "").strip()
    if not q:
        return [SyntaxIssue(
            severity="error",
            code="empty_query",
            message="query is empty",
            hint="write at least one term",
        )]
    issues: list[SyntaxIssue] = []
    # Structural checks first — if parens/quotes are unmatched, nothing
    # downstream is trustworthy.
    issues.extend(_check_balanced_parens(q))
    issues.extend(_check_balanced_quotes(q))
    if any(i.severity == "error" for i in issues):
        return issues
    issues.extend(_check_short_wildcard(q))
    issues.extend(_check_wildcard_in_phrase(q))
    issues.extend(_check_empty_groups(q))
    issues.extend(_check_mandatory_clause(q))
    issues.extend(_check_proximity_on_single_word(q))
    issues.extend(_check_operator_word_in_phrase(q))
    issues.extend(_check_subsumed_phrases(q))
    issues.extend(_check_ambiguous_short_acronym(q))
    return issues


def render_issues(issues: list[SyntaxIssue]) -> str:
    if not issues:
        return "(no syntax issues)"
    return "\n".join(i.render() for i in issues)


def has_blocking_error(issues: list[SyntaxIssue]) -> bool:
    return any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    examples = [
        "+protein +language +model*",
        "+a* +language",                               # wildcard too short
        "+\"ProtBERT\" +\"protein",                    # unmatched quote
        "+(A | B) +(",                                 # unmatched paren
        "-plant -animal",                              # no mandatory
        "+\"prime editing\"~5 +cancer",                # single-word proximity
        "+protein~3 +cancer",                          # fuzzy (should warn)
        "+(\"prime editing\" | \"prime editing stuff\")",  # subsumption
        "+(PE | ESM | protein)",                       # bare short acronyms
        "+()",                                          # empty group
    ]
    for q in examples:
        print(f"\nQuery: {q}")
        issues = syntax_check(q)
        print(render_issues(issues) if issues else "  (OK)")
