"""Natural-language Boolean ↔ Lucene translator.

LLMs (especially weaker ones) write boolean expressions more reliably
with the words ``AND`` / ``OR`` / ``NOT`` than with the symbolic
``+`` / ``|`` / ``-``. V3 now accepts the natural-language form from
workers and translates to Lucene before sending to S2. When showing a
historic query back to the worker (e.g. the iter-history block) we
translate the other way so the worker reads what it wrote.

Mapping:

  NL form                   Lucene form
  ---------------------     ---------------------
  A AND B                   +A +B
  A OR B                    (A | B)    — inside a group
  NOT A                     -A
  "A B"                     "A B"      — unchanged
  (…)                       (…)        — unchanged

Top-level AND operands must each be prefixed with ``+`` because the
default Lucene operator is OR-ish (SHOULD). We lean conservative: any
leading operand that is not already ``+``-/``-``-prefixed gets ``+``.
"""

from __future__ import annotations

import re


_WORD_AND = re.compile(r"\s*\bAND\b\s*", re.IGNORECASE)
_WORD_OR = re.compile(r"\s*\bOR\b\s*", re.IGNORECASE)
_WORD_NOT = re.compile(r"\bNOT\s+", re.IGNORECASE)


def to_lucene(nl_query: str) -> str:
    """Translate an AND/OR/NOT query into Lucene +/|/- form.

    Tolerant to mixed inputs — if the LLM happens to emit symbolic
    operators, they pass through untouched.
    """
    if not nl_query:
        return ""
    q = nl_query
    # NOT → - (prefix). Handle this FIRST so later AND/OR substitutions
    # don't eat the word "NOT" that follows AND/OR.
    q = _WORD_NOT.sub(" -", q)
    # OR → | (infix)
    q = _WORD_OR.sub(" | ", q)
    # AND → + (prefix on the right-hand operand — the left operand's
    # ``+`` is added below at the top level).
    q = _WORD_AND.sub(" +", q)
    q = re.sub(r"\s+", " ", q).strip()
    # Ensure the leftmost top-level operand is prefixed. S2 Lucene treats
    # unprefixed terms as SHOULD (optional), which makes AND chains
    # silently degrade to ORs. Only mandatory-prefix matters at top
    # level; inside parens the OR infix already handles it.
    if q and q[0] not in "+-":
        q = "+" + q
    # Collapse `+-` (which arises from "AND NOT") to just `-` — NOT is
    # already a prefix of the operand, the extra + is noise.
    q = re.sub(r"\+\s*-", "-", q)
    # Normalise stray spaces introduced by the substitutions.
    q = re.sub(r"\(\s+", "(", q)
    q = re.sub(r"\s+\)", ")", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


_LUCENE_PIPE = re.compile(r"\s*\|\s*")
_LUCENE_PLUS = re.compile(r"(^|\s|\()\+")
_LUCENE_DASH = re.compile(r"(^|\s|\()-(?=\S)")


def to_natural(lucene: str) -> str:
    """Translate a Lucene +/|/- query back into AND/OR/NOT form."""
    if not lucene:
        return ""
    q = lucene
    # | → OR
    q = _LUCENE_PIPE.sub(" OR ", q)
    # `+` prefix → AND (but drop the leading one — it's implicit)
    # First replace every `+` (after space or open-paren or at start) with AND.
    q = _LUCENE_PLUS.sub(r"\1AND ", q)
    # Strip leading AND (there's always one because to_lucene inserts it).
    q = q.strip()
    if q.startswith("AND "):
        q = q[4:]
    # Same for parenthesised groups that start with AND right after `(`.
    q = re.sub(r"\(\s*AND\s+", "(", q)
    # `-` prefix → NOT
    q = _LUCENE_DASH.sub(r"\1NOT ", q)
    # Ensure an explicit AND before NOT when it follows another operand
    # at the top level ("X NOT Y" → "X AND NOT Y").
    q = re.sub(r"(\S)\s+NOT\s+", r"\1 AND NOT ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


# ---------------------------------------------------------------------------
# Quick sanity checks — run as `python -m citeclaw.agents.v3.query_translate`
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    cases = [
        '("protein language model" OR "PLM") AND ("structure prediction" OR "folding")',
        'protein AND language AND model*',
        '("A" OR "B") AND "C" AND NOT "D"',
        'transformer AND ("self-attention" OR attention) AND NOT survey',
    ]
    for c in cases:
        lu = to_lucene(c)
        bk = to_natural(lu)
        print(f"NL  : {c}")
        print(f"LUC : {lu}")
        print(f"BK  : {bk}")
        print()
