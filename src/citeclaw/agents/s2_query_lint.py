"""Rule-based lint for Semantic Scholar bulk-search query text.

S2's Lucene parser is forgiving *syntactically* (it will accept almost
any string) but substantively fragile: many "reasonable-looking"
Lucene expressions return ``total=0`` because the operators are only
honoured between quoted phrases, or because over-constrained
intersections have no hits in the corpus. When the agent sees 0, it
cannot tell whether its query was bad or whether the topic is
genuinely unpopulated — both look identical in the S2 response.

This module is the worker's pre-flight lint: it runs BEFORE
``check_query_size`` calls S2 and rejects queries that almost
certainly won't work, with a teaching hint the agent can use to
rewrite. Runs are cheap (pure Python regex + counting), deterministic,
and scoped to ExpandBySearch — nothing else in the project imports it.

The lint is intentionally ADVISORY on borderline cases (warn, don't
reject) and HARD only on cases that are almost certain to return 0 or
trip S2's own query parser. When in doubt, let the query through and
trust the downstream screener.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class LintResult:
    """Outcome of linting one query string.

    ``ok=True`` means the query is safe to submit; ``ok=False`` means
    the dispatcher should reject the call with the ``message`` as
    ``error`` and ``hint`` as the corresponding suggestion.
    """

    ok: bool
    message: str = ""
    hint: str = ""


# Matches a single balanced quoted phrase.
_QUOTED_PHRASE = re.compile(r'"([^"]*)"')
# Matches a bare operator without surrounding quotes.
_BARE_OP_BETWEEN_TOKENS = re.compile(r"(?<![\"\s(])\s*[+|\-]\s*(?![\"\s)])")


def lint_s2_query(query: str) -> LintResult:
    """Run a battery of checks over ``query``. Return :class:`LintResult`.

    Rules (in order of severity):

    1. **Empty / whitespace-only**: reject — nothing to search.
    2. **Unbalanced parentheses**: reject — S2's parser will error.
    3. **Unbalanced quotes (odd number of ")**: reject.
    4. **Too many ``+`` clauses** (4 or more mandatory phrases): reject.
       Empirically returns 0 hits almost always; the "+" operator AND-s
       and 3+ clauses are rarely all simultaneously true in the corpus.
    5. **Operators between bare keywords** (e.g. ``DARTS +neural``):
       reject — S2 treats ``+neural`` as a literal token, not an
       operator. Quote the operand: ``"DARTS" +"neural"``.
    6. **No quoted phrases at all when using operators**: reject —
       ``foo | bar -baz`` is treated as bag-of-words.

    Intentionally NOT rejected (tunable / situational):
    - Single-word queries ("RNA", "transformer"): may match millions
      but still valid.
    - Queries with no operators at all: plain bag-of-words is valid.
    - Queries with ``-"noise_phrase"`` exclusions: valid Lucene.
    """
    if not isinstance(query, str):
        return LintResult(
            ok=False,
            message="query is not a string",
            hint="pass the query text as a JSON string",
        )
    text = query.strip()
    if not text:
        return LintResult(
            ok=False,
            message="empty query",
            hint="provide a 2–3 word quoted phrase that names the sub-topic",
        )

    # 2. Balanced parens.
    opens = text.count("(")
    closes = text.count(")")
    if opens != closes:
        return LintResult(
            ok=False,
            message=f"unbalanced parentheses ({opens} '(' vs {closes} ')')",
            hint="add/remove a matching paren",
        )

    # 3. Unbalanced quotes.
    if text.count('"') % 2 != 0:
        return LintResult(
            ok=False,
            message=f"unbalanced double quotes ({text.count('\"')} total; must be even)",
            hint="close every quoted phrase with a matching \"",
        )

    quoted_phrases = _QUOTED_PHRASE.findall(text)

    # 4. Too many '+' clauses (3+ is the empirical 0-hit tipping point).
    plus_clauses = len(re.findall(r'\+(?=")|\+(?=\()', text))
    if plus_clauses >= 3:
        return LintResult(
            ok=False,
            message=f"too many '+' clauses ({plus_clauses}); S2 intersections of 3+ phrases almost always return 0 hits",
            hint="keep at most 2 '+' mandatory phrases; use '|' OR to broaden",
        )

    # Also reject 3+ total mandatory phrases (quoted + explicit `+`
    # prefix considered). Example: ``"A" +"B" +"C" +"D"`` has 3 `+`
    # operators AND 4 total phrases — over-constrained.
    if len(quoted_phrases) >= 4 and plus_clauses >= 2:
        return LintResult(
            ok=False,
            message=(
                f"{len(quoted_phrases)} quoted phrases combined with {plus_clauses} '+' "
                f"operators creates an over-constrained AND of 4+ terms"
            ),
            hint="reduce to at most 3 quoted phrases, or convert some '+' to '|'",
        )

    # 5. Operators between bare (unquoted) tokens. We detect '+', '-',
    # or '|' that is NOT immediately preceded by '"' / whitespace / '('
    # AND not immediately followed by '"' / whitespace / ')'. Strip
    # quoted regions first to reduce false positives inside quoted text.
    stripped = _QUOTED_PHRASE.sub(" ", text)  # blank out quoted content
    for op, name in (("+", "plus"), ("|", "pipe"), ("-", "minus")):
        # Escape special regex chars where needed.
        op_esc = re.escape(op)
        if re.search(rf"(?<![\s(]){op_esc}(?![\s(\"])", stripped):
            return LintResult(
                ok=False,
                message=(
                    f"operator '{op}' attached to a bare unquoted token — "
                    f"S2 will treat this as a literal token, not an operator"
                ),
                hint=(
                    f'quote the operand: "term1" {op}"term2"  '
                    f'(not  term1 {op}term2  or  term1{op}term2)'
                ),
            )

    # 5b. Literal 'OR' / 'AND' / 'NOT' keywords used as pseudo-operators.
    # S2's Lucene parser does NOT honour these — they're treated as
    # bare tokens. Use '|', '+', '-' instead.
    if re.search(r"\b(OR|AND|NOT)\b", stripped):
        return LintResult(
            ok=False,
            message=(
                "query uses literal 'OR'/'AND'/'NOT' keywords; S2 treats "
                "them as tokens, not operators"
            ),
            hint=(
                "replace with the symbol operators: '|' for OR, '+' for AND, "
                "'-' for NOT, always between quoted phrases"
            ),
        )

    # 6. Uses operators but has no quoted phrases.
    uses_operators = any(op in text for op in ("+", "|"))
    if uses_operators and not quoted_phrases:
        return LintResult(
            ok=False,
            message="query uses '+'/'|' operators but contains no quoted phrases",
            hint='S2 operators only bind to quoted phrases. Either remove operators or quote every operand',
        )

    return LintResult(ok=True)


__all__ = ["LintResult", "lint_s2_query"]
