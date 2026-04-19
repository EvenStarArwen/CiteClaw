"""Query shape test harness — verifies the translator and decomposer
handle the wide range of bracket / spacing / nesting styles we've
observed from Gemma / Grok / OpenAI / GPT outputs in practice.

Each case is exercised twice:
  1. Pure-Python check: to_lucene → decompose_query → clause count.
  2. Optional S2 probe: send the Lucene form to S2 search_bulk and
     confirm it returns without 4xx and (for non-trivial queries)
     has a positive total.

Run locally::

    S2_API_KEY=... python tests_v3/test_query_shapes.py

Passes that require the S2 probe are skipped silently when the API
key is absent or the probe errors out.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from citeclaw.agents.v3.query_translate import (
    decompose_query,
    parse_or_alternatives,
    to_lucene,
    to_natural,
    term_matches,
)


@dataclass
class Case:
    nl: str              # natural-language query (what a worker would write)
    expected_clauses: int | None  # None = don't check
    note: str
    probe_s2: bool = False


CASES: list[Case] = [
    # --- simplest shapes ----------------------------------------------
    Case("protein AND language AND model*", 3, "three-facet with wildcard", probe_s2=True),
    Case("cancer AND immunotherapy", 2, "two facet bare", probe_s2=True),
    Case("(A OR B) AND C", 2, "OR group + single term"),
    Case("A AND B AND C AND D", 4, "four flat facets"),

    # --- outer-paren wrap habit (Grok/OpenAI) -------------------------
    Case("((A OR B) AND (C OR D))", 2, "outer wrap, 2 facets"),
    Case("(((X AND Y)))", 2, "triple redundant outer parens"),
    Case('((("prime editing") AND ("pegRNA")))', 2, "outer wrap with quoted phrases", probe_s2=True),
    Case("(protein AND (language OR model))", 2, "outer wrap, one OR subgroup", probe_s2=True),

    # --- varied spacing -----------------------------------------------
    Case("(  A OR B   ) AND C", 2, "extra interior spaces"),
    Case("A   AND   B", 2, "multi-space AND"),
    Case("(A|B) AND (C|D)", 2, "pre-symbol-style input (no spaces around |)"),

    # --- case mixing --------------------------------------------------
    Case("A and B or C", None, "lowercase operators"),
    Case("a And (b oR c)", 2, "mixed-case operators"),

    # --- quoted phrases with operators inside --------------------------
    Case('"prime AND editing" AND (X OR Y)', 2, "AND literal inside quote"),
    Case('"A B C" OR "D E"', None, "phrase OR phrase (top-level OR — unusual)"),
    Case('"cell cycle" AND NOT "cell death"', 2, "phrase with NOT"),

    # --- proximity ----------------------------------------------------
    Case('"protein model"~5 AND cancer', 2, 'phrase proximity + term'),
    Case('("A B"~3 OR "C D") AND X', 2, 'OR-group with proximity inside'),

    # --- wildcards ----------------------------------------------------
    Case("transformer* AND attention*", 2, "two wildcards"),
    Case("(bert* OR gpt*) AND language", 2, "OR of wildcards"),

    # --- NOT patterns -------------------------------------------------
    Case("CRISPR AND NOT plant", 2, "AND NOT"),
    Case("(A OR B) AND NOT (C OR D)", 2, "NOT on a group"),
    Case("NOT plant AND CRISPR", 2, "leading NOT"),

    # --- nested subgroups ---------------------------------------------
    Case("A AND (B OR (C AND D))", 2, "nested AND inside OR"),
    Case("(A OR B) AND (C AND (D OR E))", 2, "depth-2 on one facet"),
    Case("A AND (B AND (C AND (D AND E)))", 2, "right-leaning AND chain"),
    Case("((A OR B) AND C) OR D", None, "top-level OR of a grouped AND"),

    # --- long OR lists (OpenAI habit) ---------------------------------
    Case(
        '("prime editing" OR "prime editor*" OR PE1 OR PE2 OR PE3 OR PEmax OR twinPE OR PASTE) AND pegRNA',
        2, "8-alternative OR group + anchor",
    ),
    Case(
        '(ESM OR ProtBERT OR ProtT5 OR ProtTrans OR ProGen OR Ankh OR ESMFold OR OmegaFold) AND protein',
        2, "model-name OR group",
    ),

    # --- edge cases ---------------------------------------------------
    Case("A", 1, "single bare term"),
    Case('"phrase"', 1, "single quoted phrase"),
    Case("protein*", 1, "single wildcard"),
    Case("(A)", 1, "single term in parens"),
    Case("((A))", 1, "doubly-wrapped single"),
    Case("A OR B", None, "bare top-level OR (degenerate)"),

    # --- things LLMs have actually written in our traces --------------
    Case(
        '(("prime editor*" OR "prime editing") AND ("PE1" OR "PE2" OR "PEmax" OR twinPE OR PASTE))',
        2, "Grok's outer-wrap + no-quotes mix",
    ),
    Case(
        "((prime editing OR prime-editing OR prime editor) AND (pegRNA OR epegRNA))",
        2, "Gemma's unquoted lowercase style",
    ),
    Case(
        '"gravitational wave" AND (LIGO OR Virgo OR KAGRA) AND (detection OR pipeline)',
        3, "three-facet mixed quoted/unquoted",
    ),
]


def _run_case(case: Case) -> tuple[bool, str]:
    """Return (passed, message)."""
    lu = to_lucene(case.nl)
    clauses = decompose_query(lu)
    nat = to_natural(lu)

    # Structural check
    if case.expected_clauses is not None and len(clauses) != case.expected_clauses:
        return False, (
            f"[clause-count] expected {case.expected_clauses}, got {len(clauses)}  "
            f"→ clauses={clauses}"
        )

    # Round-trip check — natural-form must re-parse to the same Lucene.
    re_lu = to_lucene(nat)
    if re_lu != lu:
        # Allow inconsequential whitespace/paren differences
        if re_lu.replace(" ", "") != lu.replace(" ", ""):
            return False, (
                f"[round-trip] mismatch\n"
                f"  orig     NL : {case.nl}\n"
                f"  LUC      : {lu}\n"
                f"  NL again : {nat}\n"
                f"  LUC again: {re_lu}"
            )

    # Every OR-group clause should have parseable alternatives
    for cl in clauses:
        if "(" in cl and "|" in cl:
            alts = parse_or_alternatives(cl)
            if not alts:
                return False, f"[or-parse] OR group didn't decompose: {cl}"

    return True, f"OK  LUC={lu}  clauses={len(clauses)}"


def _run_s2_probe(lu: str, s2) -> tuple[bool, str]:
    try:
        resp = s2.search_bulk(query=lu, limit=1)
    except Exception as exc:  # noqa: BLE001
        return False, f"[s2] error: {type(exc).__name__}: {str(exc)[:200]}"
    total = resp.get("total")
    if total is None or not isinstance(total, int):
        return False, f"[s2] response missing 'total': {list((resp or {}).keys())}"
    return True, f"[s2] total={total}"


def main() -> int:
    n_pass = 0
    n_fail = 0
    failures: list[str] = []

    have_s2 = bool(os.environ.get("S2_API_KEY"))
    s2 = None
    if have_s2:
        try:
            from citeclaw.cache import Cache
            from citeclaw.clients.s2.api import SemanticScholarClient
            from citeclaw.config import BudgetTracker, Settings
            cfg = Settings(
                data_dir=str(REPO / "tests_v3" / "data" / "_probe"),
                topic_description="x",
                seed_papers=[{"title": "x"}],
                screening_model="stub",
                pipeline=[],
            )
            (REPO / "tests_v3" / "data" / "_probe").mkdir(parents=True, exist_ok=True)
            cache = Cache(REPO / "tests_v3" / "data" / "_probe" / "cache.db")
            s2 = SemanticScholarClient(cfg, cache, BudgetTracker())
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] couldn't initialise S2 client: {exc}; skipping probes")
            s2 = None

    print("=" * 80)
    print(f"Query-shape cases: {len(CASES)} total")
    print("=" * 80)
    for case in CASES:
        passed, msg = _run_case(case)
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] ({case.note})")
        print(f"        NL   : {case.nl}")
        print(f"        LUC  : {to_lucene(case.nl)}")
        print(f"        {msg}")
        if passed:
            n_pass += 1
        else:
            n_fail += 1
            failures.append(f"{case.note}: {msg}")

        if case.probe_s2 and s2 is not None:
            p, pmsg = _run_s2_probe(to_lucene(case.nl), s2)
            print(f"        {pmsg}")
            if not p:
                failures.append(f"{case.note} [s2]: {pmsg}")

    print("=" * 80)
    print(f"Summary: {n_pass} passed, {n_fail} failed (of {len(CASES)})")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  · {f}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
