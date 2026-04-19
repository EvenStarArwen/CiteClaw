"""Natural-language Boolean ↔ Lucene translator.

LLMs write boolean expressions more reliably with AND/OR/NOT words than
with the ``+``/``|``/``-`` symbols S2's Lucene parser expects. V3 now:

  - accepts AND/OR/NOT from workers and translates to ``+``/``|``/``-``
    before hitting S2
  - translates back to AND/OR/NOT when surfacing historic queries or
    query-tree clause labels to the worker

Robustness rules applied during translation (all the corner cases we've
observed in the wild, from Grok / OpenAI / Gemma outputs):

  1. Quoted strings are PROTECTED — any AND/OR/NOT inside a ``"..."`` is
     a literal, not an operator. ``"prime AND editing"`` stays as a
     three-word phrase, not ``prime + editing``.
  2. Redundant outermost parens are stripped. ``((X) AND (Y))`` →
     ``(X) AND (Y)`` before operator substitution, so ``decompose_query``
     can break the expression into its genuine top-level clauses.
  3. ``+-`` (from ``AND NOT``) collapses to ``-``.
  4. Whitespace around ``(`` and ``)`` is normalised.
"""

from __future__ import annotations

import re


_WORD_AND = re.compile(r"\s*\bAND\b\s*", re.IGNORECASE)
_WORD_OR = re.compile(r"\s*\bOR\b\s*", re.IGNORECASE)
_WORD_NOT = re.compile(r"\bNOT\s+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_top_level_pipe(s: str) -> bool:
    """True if there's a ``|`` outside any nested parens or quotes."""
    depth = 0
    in_quote = False
    for c in s:
        if c == '"':
            in_quote = not in_quote
        elif in_quote:
            continue
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "|" and depth == 0:
            return True
    return False


def _strip_redundant_outer_parens(q: str) -> str:
    """Peel pair(s) of enclosing parens that wrap the ENTIRE expression.

    ``((A AND B))`` → ``A AND B``. A pair is "enclosing" only when the
    first ``(`` matches the very last ``)`` — i.e. the string enclosed
    by that pair IS the whole expression, not a prefix. ``(A)(B)``
    does NOT get stripped because the first ``)`` closes before the end.
    Leading whitespace is stripped first so ``   (X)`` is handled too.
    """
    q = q.strip()
    while len(q) >= 2 and q.startswith("(") and q.endswith(")"):
        # Verify the outermost '(' matches the final ')'.
        depth = 0
        outer_matches = True
        for i, c in enumerate(q):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and i < len(q) - 1:
                    outer_matches = False
                    break
        if not outer_matches:
            break
        q = q[1:-1].strip()
    return q


def _protect_quoted(q: str) -> tuple[str, dict[str, str]]:
    """Replace every ``"..."`` with a placeholder so regex substitutions
    don't chew up AND / OR / NOT that appear INSIDE a quoted phrase.

    Returns ``(quoted-free string, placeholder → original dict)``.
    Call :func:`_restore_quoted` to swap them back.
    """
    placeholders: dict[str, str] = {}

    def _sub(m: re.Match) -> str:
        key = f"\x00Q{len(placeholders):04d}\x00"
        placeholders[key] = m.group(0)
        return key

    return re.sub(r'"[^"]*"', _sub, q), placeholders


def _restore_quoted(q: str, placeholders: dict[str, str]) -> str:
    for key, original in placeholders.items():
        q = q.replace(key, original)
    return q


# ---------------------------------------------------------------------------
# NL → Lucene
# ---------------------------------------------------------------------------


def to_lucene(nl_query: str) -> str:
    if not nl_query:
        return ""
    q = nl_query.strip()
    q = _strip_redundant_outer_parens(q)
    if not q:
        return ""

    # Protect quoted strings so operators inside them stay literal.
    q, placeholders = _protect_quoted(q)

    # NOT first (so later substitutions don't consume the word following AND/OR).
    q = _WORD_NOT.sub(" -", q)
    # OR is an infix.
    q = _WORD_OR.sub(" | ", q)
    # AND is a prefix operator in Lucene (+ on the right-hand operand).
    q = _WORD_AND.sub(" +", q)
    q = re.sub(r"\s+", " ", q).strip()

    # Restore quotes before prefixing and collapsing.
    q = _restore_quoted(q, placeholders)

    # Ensure the leftmost top-level operand is prefixed. S2 treats
    # un-prefixed top-level operands as SHOULD (optional), which
    # silently degrades an AND chain to an OR-ish mess.
    if q and q[0] not in "+-":
        q = "+" + q

    # `AND NOT` → `+-` → `-`
    q = re.sub(r"\+\s*-", "-", q)

    # Tidy parens + collapse spaces.
    q = re.sub(r"\(\s+", "(", q)
    q = re.sub(r"\s+\)", ")", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


# ---------------------------------------------------------------------------
# Lucene → NL
# ---------------------------------------------------------------------------


_LUCENE_PIPE = re.compile(r"\s*\|\s*")
_LUCENE_PLUS = re.compile(r"(^|\s|\()\+")
_LUCENE_DASH = re.compile(r"(^|\s|\()-(?=\S)")


def to_natural(lucene: str) -> str:
    if not lucene:
        return ""
    q = lucene.strip()

    # Protect quoted strings.
    q, placeholders = _protect_quoted(q)

    q = _LUCENE_PIPE.sub(" OR ", q)
    # `+X` / `+(...)` → `AND X` / `AND (...)` (except the leading one).
    q = _LUCENE_PLUS.sub(r"\1AND ", q)
    # Strip the leading AND (to_lucene always adds one).
    q = q.strip()
    if q.startswith("AND "):
        q = q[4:]
    # Same clean-up for AND right after an opening paren.
    q = re.sub(r"\(\s*AND\s+", "(", q)
    # `-X` → `NOT X`
    q = _LUCENE_DASH.sub(r"\1NOT ", q)
    # Ensure AND precedes NOT when NOT follows another operand.
    q = re.sub(r"(\S)\s+NOT\s+", r"\1 AND NOT ", q)
    q = re.sub(r"\s+", " ", q).strip()

    q = _restore_quoted(q, placeholders)
    return q


# ---------------------------------------------------------------------------
# Top-level clause decomposition (for query_tree)
# ---------------------------------------------------------------------------


def decompose_query(query: str) -> list[str]:
    """Split a Lucene query into top-level clauses.

    A clause is a quoted string or a parenthesised group, optionally
    prefixed with ``+`` or ``-``. Before tokenising we iteratively peel:

    * ``+(WHOLE)`` / ``-(WHOLE)`` when the parens wrap the whole expression
      (Grok and OpenAI's habit of wrapping the entire query in one outer
      group). After peeling, every un-prefixed top-level clause gets a
      ``+`` back so each facet is still recognised as mandatory — the
      outer group was providing that semantics.
    * Plain ``(WHOLE)`` redundant wraps (``((X))``).

    Bare top-level ``|`` infixes (rare, but possible when the LLM tried
    to write a top-level disjunction) are suppressed from the returned
    list because they aren't clauses themselves.
    """
    q = (query or "").strip()
    if not q:
        return []

    peeled_outer_plus = False
    while True:
        # Case 1: +( WHOLE ) or -( WHOLE ) where WHOLE is an implicit-AND
        # of multiple clauses. Do NOT peel when WHOLE is a single OR group
        # (top-level | and no separating whitespace between clauses): in
        # that case +(...) is the canonical form of "must contain one of
        # ..." and peeling would break the semantics into multiple MUSTs.
        if len(q) >= 3 and q[0] in "+-" and q[1] == "(" and q[-1] == ")":
            depth = 0
            matches = True
            for i in range(1, len(q)):
                if q[i] == "(":
                    depth += 1
                elif q[i] == ")":
                    depth -= 1
                    if depth == 0 and i < len(q) - 1:
                        matches = False
                        break
            if matches:
                inner = q[2:-1].strip()
                if _has_top_level_pipe(inner):
                    break  # single OR group — keep the +(…) wrap as one clause
                if q[0] == "+":
                    peeled_outer_plus = True
                q = inner
                continue
        # Case 2: plain ( WHOLE ) wrap
        new = _strip_redundant_outer_parens(q)
        if new != q:
            q = new
            continue
        break

    tokens = [t for t in _tokenize_top_level(q) if t.strip() and t.strip() != "|"]
    if peeled_outer_plus and len(tokens) > 1:
        # The stripped outer `+` lent mandatory semantics to every enclosed
        # top-level clause; re-apply `+` so the displayed tree is consistent
        # and each clause counts the right thing when probed independently.
        tokens = [
            ("+" + t) if t and t[0] not in "+-" else t
            for t in tokens
        ]
    return tokens


def parse_or_alternatives(clause: str) -> list[str]:
    """For a clause like ``+(A | B | "C D")`` return ``['A', 'B', '"C D"']``.

    Used by the query-tree builder to break an OR facet into its
    individual alternatives for per-term in-memory counting. Returns
    an empty list when the clause is not an OR group (e.g. ``+"A B"``
    or ``+foo``).

    Splits on top-level ``|`` only — does not descend into nested parens
    or quoted strings.
    """
    q = (clause or "").strip()
    if not q:
        return []
    if q[0] in "+-":
        q = q[1:].strip()
    if not (q.startswith("(") and q.endswith(")")):
        return []
    # Peel one layer of parens.
    depth = 0
    matches = True
    for i, c in enumerate(q):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and i < len(q) - 1:
                matches = False
                break
    if not matches:
        return []
    inner = q[1:-1].strip()
    if not inner:
        return []
    parts: list[str] = []
    current: list[str] = []
    in_quote = False
    depth = 0
    for c in inner:
        if c == '"':
            in_quote = not in_quote
            current.append(c)
        elif in_quote:
            current.append(c)
        elif c == "(":
            depth += 1
            current.append(c)
        elif c == ")":
            depth -= 1
            current.append(c)
        elif c == "|" and depth == 0:
            s = "".join(current).strip()
            if s:
                parts.append(s)
            current = []
        else:
            current.append(c)
    s = "".join(current).strip()
    if s:
        parts.append(s)
    return parts


def term_matches(term: str, text_lower: str) -> bool:
    """Does ``term`` (a single OR alternative) match anywhere in ``text_lower``?

    Approximates S2 Lucene semantics in Python so we can compute per-
    alternative counts over already-fetched papers without additional
    S2 queries. ``text_lower`` is expected to be the lower-cased
    concatenation of the paper's title and abstract.

    Handles:
      - ``"phrase"`` — case-insensitive substring match
      - ``"phrase"~N`` — proximity, approximated as phrase substring
        (strict proximity would need token positions; N slop is usually
        just a few positions so the approximation is close)
      - ``word`` — case-insensitive word-boundary match
      - ``word*`` — word-boundary + arbitrary suffix
      - Any ``+`` / ``-`` prefix is stripped before matching
    """
    if not term or not text_lower:
        return False
    t = term.strip()
    if t and t[0] in "+-":
        t = t[1:].strip()
    if not t:
        return False
    # "phrase"~N — take the inner phrase.
    m = re.fullmatch(r'"([^"]+)"~\d+', t)
    if m:
        return m.group(1).lower() in text_lower
    # "phrase"
    m = re.fullmatch(r'"([^"]+)"', t)
    if m:
        return m.group(1).lower() in text_lower
    # word* (suffix wildcard)
    if t.endswith("*"):
        stem = t[:-1]
        if len(stem) < 2:
            # Too permissive to be useful; treat as always-match.
            return True
        pattern = rf"\b{re.escape(stem.lower())}\w*\b"
        return bool(re.search(pattern, text_lower))
    # Bare word / multi-word unquoted — just do a substring match if it
    # has whitespace, else a word-boundary match.
    tl = t.lower()
    if re.search(r"\s", tl):
        return tl in text_lower
    pattern = rf"\b{re.escape(tl)}\b"
    return bool(re.search(pattern, text_lower))


def _tokenize_top_level(query: str) -> list[str]:
    """Walk the query and emit top-level clauses (with their +/- prefix).

    Does NOT touch the inside of a parenthesised group or quoted string.
    """
    clauses: list[str] = []
    i = 0
    n = len(query)
    while i < n:
        while i < n and query[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        if query[i] in "+-":
            i += 1
        while i < n and query[i].isspace():
            i += 1
        if i >= n:
            break
        ch = query[i]
        if ch == '"':
            i += 1
            while i < n and query[i] != '"':
                i += 1
            if i < n:
                i += 1  # consume closing quote
            # Consume optional proximity modifier ~N immediately following
            # the closing quote (e.g. "A B"~5).
            if i < n and query[i] == "~":
                i += 1
                while i < n and query[i].isdigit():
                    i += 1
        elif ch == "(":
            depth = 1
            i += 1
            while i < n and depth > 0:
                if query[i] == "(":
                    depth += 1
                elif query[i] == ")":
                    depth -= 1
                elif query[i] == '"':
                    # Skip past a quoted string inside the group.
                    i += 1
                    while i < n and query[i] != '"':
                        i += 1
                i += 1
        else:
            # Bare token (or `|` infix). Advance to next whitespace.
            while i < n and not query[i].isspace():
                i += 1
        # If we landed on an infix `|` operator, it's NOT a clause boundary.
        # Everything on the same side of each `|` belongs to one OR group,
        # but OR groups at top level are already enclosed in parens. A bare
        # top-level `|` would be unusual — just emit it as a "clause" so
        # the caller can see it if it happens.
        clauses.append(query[start:i])
    # Collapse runs of whitespace clauses.
    return [c.strip() for c in clauses if c.strip()]


# ---------------------------------------------------------------------------
# Quick sanity checks
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    cases = [
        # (input, comment)
        ('((A OR B) AND (C OR D OR E))', "outer-paren-wrap (Grok habit)"),
        ('(A OR B) AND (C OR D OR E)', "no outer paren"),
        ('"prime AND editing" AND (X OR Y)', "AND literal inside quote"),
        ('protein AND language AND model*', "three-facet with wildcard"),
        ('("A" OR "B") AND "C" AND NOT "D"', "with NOT"),
        ('(A OR (B AND C))', "nested inside OR group"),
        ('((((X))))', "redundant outer parens"),
        ('A', "single term"),
    ]
    for c, comment in cases:
        lu = to_lucene(c)
        bk = to_natural(lu)
        clauses = decompose_query(lu)
        print(f"NL   : {c}        # {comment}")
        print(f"LUC  : {lu}")
        print(f"BK   : {bk}")
        print(f"CLS  : {clauses}")
        print()
