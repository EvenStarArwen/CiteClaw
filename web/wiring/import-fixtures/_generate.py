#!/usr/bin/env python3
"""Regenerate the OmniKnowledge import test corpus.

Every fixture under ``web/wiring/import-fixtures/`` is synthetic — no real
user library, no scraped file, no copyrighted abstract. Re-running this
script is idempotent: it rewrites the whole tree byte-for-byte.

    python3 web/wiring/import-fixtures/_generate.py

Naming convention (the automated import tests key off the prefix):

    ok-*        parses clean; every entry should reach match-review
    partial-*   parses, but some entries must be REPORTED as unmatched /
                skipped — never silently dropped
    edge-*      parses, but pins a decision the parser spec has to make
                explicit (encoding, delimiter, dedupe, fallback…)
    bad-*       must fail with a per-file error message; nothing added

See README.md for the per-file manifest and expected outcomes.
"""

from __future__ import annotations

import json
import os
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIRS = ["bibtex", "ris", "csv", "lists", "json", "pdf", "archives", "vendor",
        "adversarial"]

# ---------------------------------------------------------------- shared data

TITLES = [
    "Adaptive weight vectors for decomposition-based multi-objective optimization",
    "A survey of surrogate-assisted evolutionary algorithms for expensive problems",
    "Benchmarking many-objective optimizers on real-world engineering suites",
    "Constraint handling in decomposition frameworks: a unified view",
    "Pareto front estimation with learned scalarizing functions",
    "Neighborhood adaptation strategies for large-scale variable interaction",
    "On the convergence of reference-point methods under noisy evaluations",
    "Transfer optimization across related multi-objective tasks",
    "A knee-point driven algorithm for preference-based optimization",
    "Diversity maintenance beyond crowding distance: a systematic study",
    "Hybridizing local search with decomposition for combinatorial problems",
    "Scalable indicator-based selection with incremental hypervolume updates",
    "Dynamic multi-objective optimization with change-severity detection",
    "Weight vector generation on irregular Pareto fronts",
    "An empirical study of mating restriction in decomposition algorithms",
    "Bayesian preference elicitation for interactive optimization",
    "Robust optimization under decision-space perturbations",
    "Archive strategies for unbounded external populations",
    "Multi-task evolutionary optimization with shared representations",
    "Expensive constraint evaluation and feasibility-first ranking",
]

VENUES = [
    "IEEE Transactions on Evolutionary Computation",
    "Evolutionary Computation",
    "Swarm and Evolutionary Computation",
    "ACM Computing Surveys",
    "Applied Soft Computing",
    "Information Sciences",
    "Neurocomputing",
    "arXiv",
]

AUTHORS = [
    ("Zhang", "Qingfu"), ("Li", "Hui"), ("Deb", "Kalyanmoy"),
    ("Ishibuchi", "Hisao"), ("Coello", "Carlos"), ("Jin", "Yaochu"),
    ("Tan", "Kay Chen"), ("Wang", "Handing"), ("Sato", "Hiroyuki"),
    ("Chugh", "Tinkle"), ("Miettinen", "Kaisa"), ("Cheng", "Ran"),
]


def doi(i: int) -> str:
    return f"10.{1000 + (i % 8000)}/omk.{2014 + (i % 13)}.{i:05d}"


def arxiv(i: int) -> str:
    return f"{(i % 12) + 1:02d}{(i % 9) + 1:02d}.{10000 + (i % 89999)}"


def rec(i: int) -> dict:
    r = random.Random(i * 7919)
    a1 = AUTHORS[i % len(AUTHORS)]
    a2 = AUTHORS[(i * 3 + 5) % len(AUTHORS)]
    return {
        "key": f"omk{2014 + (i % 13)}{a1[0].lower()}{i:04d}",
        "title": TITLES[i % len(TITLES)],
        "authors": [a1, a2],
        "year": 2014 + (i % 13),
        "venue": VENUES[i % len(VENUES)],
        "doi": doi(i),
        "arxiv": arxiv(i),
        "pages": f"{r.randint(1, 400)}--{r.randint(401, 900)}",
        "volume": str(r.randint(1, 40)),
        "number": str(r.randint(1, 12)),
    }


def w(rel: str, data, encoding: str | None = "utf-8", newline: str = "\n") -> None:
    """Write ``data`` (str or bytes) to ``ROOT/rel``."""
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        p.write_bytes(data)
    else:
        if newline != "\n":
            data = data.replace("\n", newline)
        p.write_bytes(data.encode(encoding))


# --------------------------------------------------------------------- BibTeX

def bib_entry(e: dict, kind: str = "article") -> str:
    au = " and ".join(f"{s}, {g}" for s, g in e["authors"])
    field = "journal" if kind == "article" else "booktitle"
    return (
        f"@{kind}{{{e['key']},\n"
        f"  author    = {{{au}}},\n"
        f"  title     = {{{e['title']}}},\n"
        f"  {field:<9} = {{{e['venue']}}},\n"
        f"  year      = {{{e['year']}}},\n"
        f"  volume    = {{{e['volume']}}},\n"
        f"  number    = {{{e['number']}}},\n"
        f"  pages     = {{{e['pages']}}},\n"
        f"  doi       = {{{e['doi']}}}\n"
        f"}}\n"
    )


def gen_bibtex() -> None:
    # ok-basic: 12 clean entries, mixed article / inproceedings, DOI on each
    body = "".join(
        bib_entry(rec(i), "article" if i % 3 else "inproceedings")
        for i in range(12)
    )
    w("bibtex/ok-basic.bib", "% OmniKnowledge import fixture\n\n" + body)

    # ok-unicode-authors: non-ASCII names + LaTeX accent escapes + CJK
    uni = """% Unicode + LaTeX-escape torture test
@article{muller2021adaptive,
  author    = {M\\"{u}ller, J\\"{o}rg and Alvarez, Jos\\'{e} Mar\\'{i}a and Ko\\v{c}i, Pavel},
  title     = {Adaptive weight vectors under noisy evaluations},
  journal   = {Evolutionary Computation},
  year      = {2021},
  doi       = {10.1162/omk.2021.00012}
}

@article{huang2023zhongwen,
  author    = {黄, 明 and Ivanova, Екатерина and Þórsdóttir, Guðrún},
  title     = {多目标优化中的权重向量自适应 (Weight-vector adaptation in MOEA/D)},
  journal   = {Information Sciences},
  year      = {2023},
  doi       = {10.1016/omk.2023.00034}
}

@inproceedings{ozturk2019turkish,
  author    = {Öztürk, Şeyma and Nguyễn, Đức Anh and Στεφάνου, Μαρία},
  title     = {Diversity maintenance beyond crowding distance},
  booktitle = {GECCO '19: Proceedings of the Genetic and Evolutionary Computation Conference},
  year      = {2019},
  doi       = {10.1145/omk.2019.00087}
}

@article{emoji2024title,
  author    = {Sato, Hiroyuki},
  title     = {On the convergence of reference-point methods 🧪 (emoji in title)},
  journal   = {arXiv},
  year      = {2024},
  doi       = {10.48550/arXiv.2401.00099}
}
"""
    w("bibtex/ok-unicode-authors.bib", uni)

    # ok-nested-braces-crossref: @string macros, # concatenation, crossref,
    # deeply nested braces, and a title whose casing must survive.
    nested = """@string{tec  = {IEEE Transactions on Evolutionary Computation}}
@string{pub  = {IEEE}}

@proceedings{gecco19,
  title     = {Proceedings of the {Genetic and Evolutionary Computation Conference} ({GECCO} '19)},
  year      = {2019},
  publisher = pub,
  address   = {New York, NY, USA}
}

@inproceedings{knee19,
  author    = {Deb, Kalyanmoy and Cheng, Ran},
  title     = {A knee-point driven algorithm for {{Preference-Based}} optimization},
  crossref  = {gecco19},
  pages     = {101--112},
  doi       = {10.1145/omk.2019.00101}
}

@inproceedings{archive19,
  author    = {Zhang, Qingfu},
  title     = {Archive strategies for {{Unbounded}} external populations},
  crossref  = {gecco19},
  pages     = {113--124},
  doi       = {10.1145/omk.2019.00113}
}

@article{concat21,
  author    = {Ishibuchi, Hisao and Sato, Hiroyuki},
  title     = {Normalization pitfalls in many-objective benchmarking},
  journal   = tec # { (Early Access)},
  year      = {2021},
  note      = {Nested groups: {a {b {c}} d} must not terminate the field},
  doi       = {10.1109/omk.2021.00204}
}
"""
    w("bibtex/ok-nested-braces-crossref.bib", nested)

    # edge-bom: identical content to ok-basic, UTF-8 BOM prefixed
    w("bibtex/edge-bom-utf8.bib", b"\xef\xbb\xbf" + (ROOT / "bibtex/ok-basic.bib").read_bytes())

    # edge-latin1: legacy Zotero/Word export, cp1252 bytes, no BOM
    latin = (
        "@article{mueller1998legacy,\n"
        "  author  = {Müller, Jörg and Alvarez, José},\n"
        "  title   = {Pareto front estimation — a café study},\n"
        "  journal = {Evolutionary Computation},\n"
        "  year    = {1998},\n"
        "  doi     = {10.1162/omk.1998.00001}\n"
        "}\n"
    )
    w("bibtex/edge-latin1.bib", latin.encode("cp1252"))

    # edge-utf16: what Windows Notepad "Unicode" save produces
    w("bibtex/edge-utf16le.bib", (ROOT / "bibtex/ok-basic.bib").read_text("utf-8"),
      encoding="utf-16")

    # edge-crlf: Windows line endings throughout
    w("bibtex/edge-crlf.bib", (ROOT / "bibtex/ok-basic.bib").read_text("utf-8"),
      newline="\r\n")

    # edge-duplicate-keys: same citekey twice AND the same DOI on two keys
    e0, e1 = rec(0), rec(1)
    dup = bib_entry(e0) + bib_entry(e0)
    e1b = dict(e1)
    e1b["key"] = e1["key"] + "b"
    e1b["doi"] = e1["doi"]
    dup += bib_entry(e1) + bib_entry(e1b)
    w("bibtex/edge-duplicate-keys.bib", dup)

    # edge-no-doi: title/author only — must fall back to S2 title match
    nodoi = "".join(
        bib_entry(rec(i)).replace(f",\n  doi       = {{{doi(i)}}}", "")
        for i in range(20, 26)
    )
    w("bibtex/edge-no-doi.bib", nodoi)

    # partial-mixed-quality: half resolvable, half deliberately unresolvable
    part = "".join(bib_entry(rec(i)) for i in range(3))
    part += """@article{ghost2031,
  author  = {Nobody, A.},
  title   = {A paper that does not exist anywhere on Semantic Scholar 9f3c1a},
  journal = {Journal of Nonexistent Results},
  year    = {2031},
  doi     = {10.9999/omk.does.not.resolve}
}

@misc{titleonly,
  title   = {qqzzxx untitled fragment},
  year    = {2020}
}
"""
    w("bibtex/partial-mixed-quality.bib", part)

    # bad-unbalanced-braces: field never closes
    w("bibtex/bad-unbalanced-braces.bib",
      "@article{broken2020,\n"
      "  author = {Zhang, Qingfu},\n"
      "  title  = {A title that never closes its brace,\n"
      "  year   = {2020}\n"
      "}\n")

    # bad-truncated: file cut mid-entry (interrupted download)
    full = "".join(bib_entry(rec(i)) for i in range(6))
    w("bibtex/bad-truncated.bib", full[: int(len(full) * 0.62)])

    # bad-empty
    w("bibtex/bad-empty.bib", "")

    # bad-no-entries: syntactically fine, zero records
    w("bibtex/bad-no-entries.bib",
      "% Exported by a tool that wrote only the preamble.\n"
      "@string{tec = {IEEE Transactions on Evolutionary Computation}}\n"
      "@comment{jabref-meta: databaseType:bibtex;}\n")

    # bad-binary-as-bib: NUL bytes + random binary with a .bib name
    w("bibtex/bad-binary.bib",
      bytes([0x00, 0x01, 0x02, 0xff, 0xfe, 0x7f] * 200) + b"\x00@article{x,")

    # huge-3000: the size / streaming test
    huge = "".join(bib_entry(rec(i)) for i in range(3000))
    w("bibtex/edge-huge-3000.bib", huge)

    # unicode filename with spaces and parentheses
    w("bibtex/ok-参考文献 export (2026).bib", "".join(bib_entry(rec(i)) for i in range(5)))


# ------------------------------------------------------------------------ RIS

def ris_record(e: dict, ty: str = "JOUR") -> str:
    lines = [f"TY  - {ty}"]
    for s, g in e["authors"]:
        lines.append(f"AU  - {s}, {g}")
    lines += [
        f"TI  - {e['title']}",
        f"JO  - {e['venue']}",
        f"PY  - {e['year']}",
        f"VL  - {e['volume']}",
        f"IS  - {e['number']}",
        f"SP  - {e['pages'].split('--')[0]}",
        f"EP  - {e['pages'].split('--')[1]}",
        f"DO  - {e['doi']}",
        "ER  - ",
    ]
    return "\n".join(lines) + "\n\n"


def gen_ris() -> None:
    w("ris/ok-basic.ris", "".join(ris_record(rec(i)) for i in range(10)))

    # ok-folded-lines: abstracts and notes wrapped over continuation lines —
    # the classic RIS parser killer (a continuation line has no TAG marker).
    folded = """TY  - JOUR
AU  - Zhang, Qingfu
AU  - Li, Hui
TI  - Adaptive weight vectors for decomposition-based multi-objective
      optimization under noisy evaluations
AB  - This synthetic abstract deliberately wraps across several physical
      lines with a six-space hanging indent, which is how EndNote and
      Web of Science emit long fields. A parser that treats every line
      as "TAG  - value" will drop everything after the first line.

      It even contains a blank line inside the folded block, which is
      legal in some exporters and fatal in most naive parsers.
JO  - IEEE Transactions on Evolutionary Computation
PY  - 2019
DO  - 10.1109/omk.2019.00301
KW  - multi-objective
KW  - decomposition
ER  -

TY  - CONF
AU  - Deb, Kalyanmoy
TI  - A knee-point driven algorithm
	for preference-based optimization
T2  - GECCO '19
PY  - 2019
DO  - 10.1145/omk.2019.00302
ER  -
"""
    w("ris/ok-folded-lines.ris", folded)

    # edge-crlf-bom: BOM + CRLF + trailing spaces after ER
    w("ris/edge-crlf-bom.ris",
      b"\xef\xbb\xbf" + "".join(ris_record(rec(i)) for i in range(4))
      .replace("\n", "\r\n").replace("ER  - \r\n", "ER  -   \r\n").encode("utf-8"))

    # edge-no-blank-separator: records back to back with no blank line
    w("ris/edge-no-blank-separator.ris",
      "".join(ris_record(rec(i)) for i in range(5)).replace("\n\n", "\n"))

    # edge-types: every TY the corpus is likely to see, incl. non-paper ones
    types = ""
    for i, ty in enumerate(["JOUR", "CONF", "CPAPER", "BOOK", "CHAP", "THES",
                            "RPRT", "UNPB", "ELEC", "GEN", "DATA", "PAT"]):
        types += ris_record(rec(30 + i), ty)
    w("ris/edge-mixed-types.ris", types)

    # edge-mixed-encoding: valid UTF-8 file with two raw cp1252 bytes spliced in
    good = "".join(ris_record(rec(i)) for i in range(3)).encode("utf-8")
    bad = good.replace(b"Zhang", b"Z\xe9\xf1ng", 1)
    w("ris/edge-mixed-encoding.ris", bad)

    # partial-missing-doi: half the records carry no DO tag
    part = "".join(ris_record(rec(i)) for i in range(3))
    part += "".join(
        ris_record(rec(i)).replace(f"DO  - {doi(i)}\n", "") for i in range(40, 43)
    )
    w("ris/partial-missing-doi.ris", part)

    # bad-missing-er: last record never terminated
    w("ris/bad-missing-er.ris",
      ris_record(rec(0)) + ris_record(rec(1)).replace("ER  - \n", ""))

    # bad-tag-column: tags not in the "XX  - " 6-char form (3 spaces / no dash)
    w("ris/bad-tag-column.ris",
      "TY -JOUR\nAU   Zhang, Qingfu\nTI - Adaptive weight vectors\nER\n")

    # bad-empty
    w("ris/bad-empty.ris", "")

    # bad-html-saved-as-ris: publisher gave a login page instead of the export
    w("ris/bad-html-page.ris",
      "<!DOCTYPE html>\n<html><head><title>Sign in</title></head>\n"
      "<body><h1>Please sign in to download this citation.</h1></body></html>\n")


# ------------------------------------------------------------------------ CSV

def gen_csv() -> None:
    # ok-doi-column: the minimum viable CSV — one DOI column
    rows = ["Title,Author,Year,DOI"]
    for i in range(10):
        e = rec(i)
        rows.append(f'"{e["title"]}","{e["authors"][0][0]}, {e["authors"][0][1]}",{e["year"]},{e["doi"]}')
    w("csv/ok-doi-column.csv", "\n".join(rows) + "\n")

    # ok-zotero-export: Zotero's real header set (trimmed to the columns a
    # resolver would read) — note the DOI column sits at position 9.
    zh = ("Key,Item Type,Publication Year,Author,Title,Publication Title,ISBN,"
          "ISSN,DOI,Url,Abstract Note,Date,Pages,Issue,Volume,Publisher,"
          "Library Catalog,Extra,Manual Tags,Automatic Tags")
    zrows = [zh]
    for i in range(8):
        e = rec(i + 50)
        au = "; ".join(f"{s}, {g}" for s, g in e["authors"])
        zrows.append(
            f'ABCD{i:04d},journalArticle,{e["year"]},"{au}","{e["title"]}",'
            f'"{e["venue"]}",,1089-778X,{e["doi"]},https://doi.org/{e["doi"]},'
            f'"Synthetic abstract for fixture row {i}.",{e["year"]}-06-01,'
            f'{e["pages"].replace("--", "-")},{e["number"]},{e["volume"]},IEEE,'
            f'Zotero,,"moead; benchmark","optimization"'
        )
    w("csv/ok-zotero-export.csv", "\n".join(zrows) + "\n")

    # edge-semicolon-bom: Excel-in-Europe export — BOM, ';' delimiter, ',' decimals
    srows = ["Titel;Autor;Jahr;DOI"]
    for i in range(6):
        e = rec(i + 70)
        srows.append(f'"{e["title"]}";"{e["authors"][0][0]}";{e["year"]};{e["doi"]}')
    w("csv/edge-semicolon-bom.csv", b"\xef\xbb\xbf" + ("\n".join(srows) + "\n").encode("utf-8"))

    # edge-quoted-newlines: embedded newlines, commas, and doubled quotes
    q = ('Title,Author,Year,DOI\n'
         '"A title with, a comma and\n'
         'an embedded newline","Zhang, Qingfu",2020,10.1109/omk.2020.00001\n'
         '"A title with ""internal quotes"" in it","Li, Hui",2021,10.1109/omk.2021.00002\n'
         '"Trailing field with a lone \\ backslash","Deb, Kalyanmoy",2022,10.1109/omk.2022.00003\n')
    w("csv/edge-quoted-newlines.csv", q)

    # edge-no-doi-column: title/author only — must fall back to title match
    nrows = ["Title,Author,Year,Journal"]
    for i in range(6):
        e = rec(i + 90)
        nrows.append(f'"{e["title"]}","{e["authors"][0][0]}",{e["year"]},"{e["venue"]}"')
    w("csv/edge-no-doi-column.csv", "\n".join(nrows) + "\n")

    # edge-doi-column-aliases: header the resolver has to recognise loosely
    w("csv/edge-doi-column-aliases.csv",
      "paper title,doi_url,ArXiv ID,PMID\n"
      f'"{TITLES[0]}",https://doi.org/{doi(1)},{arxiv(1)},31234567\n'
      f'"{TITLES[1]}",doi:{doi(2)},arXiv:{arxiv(2)},\n'
      f'"{TITLES[2]}",{doi(3)},,29876543\n')

    # edge-tsv-as-csv: tab-delimited content under a .csv name
    trows = ["Title\tAuthor\tYear\tDOI"]
    for i in range(5):
        e = rec(i + 110)
        trows.append(f'{e["title"]}\t{e["authors"][0][0]}\t{e["year"]}\t{e["doi"]}')
    w("csv/edge-tsv-as-csv.csv", "\n".join(trows) + "\n")

    # bad-ragged: inconsistent column counts, header shorter than rows
    w("csv/bad-ragged.csv",
      "Title,DOI\n"
      f'"{TITLES[0]}",{doi(1)},extra,columns,nobody,asked,for\n'
      f'"{TITLES[1]}"\n'
      ",,\n")

    # bad-empty
    w("csv/bad-empty.csv", "")

    # bad-header-only
    w("csv/bad-header-only.csv", "Title,Author,Year,DOI\n")

    # bad-no-recognisable-columns
    w("csv/bad-no-recognisable-columns.csv",
      "col_a,col_b,col_c\n1,2,3\n4,5,6\n")


# ---------------------------------------------------------------- plain lists

def gen_lists() -> None:
    dois = "\n".join([
        "# One DOI per line; comments and blank lines are ignored.",
        doi(1),
        f"doi:{doi(2)}",
        f"DOI: {doi(3)}",
        f"https://doi.org/{doi(4)}",
        f"http://dx.doi.org/{doi(5)}",
        "",
        f"  {doi(6)}  ",
        f"{doi(7)}.",
    ])
    w("lists/ok-doi-list.txt", dois + "\n")

    arx = "\n".join([
        arxiv(1),
        f"arXiv:{arxiv(2)}",
        f"arXiv:{arxiv(3)}v2",
        f"https://arxiv.org/abs/{arxiv(4)}",
        f"https://arxiv.org/pdf/{arxiv(5)}v1.pdf",
        f"https://arxiv.org/abs/{arxiv(6)}v3",
        "cs.LG/0601001",
        "math.CO/0510001v2",
        f"10.48550/arXiv.{arxiv(7)}",
    ])
    w("lists/ok-arxiv-list.txt", arx + "\n")

    mixed = "\n".join([
        doi(11),
        f"arXiv:{arxiv(12)}",
        "https://www.semanticscholar.org/paper/649def34f8be52c8b66281af98ae884c09aef38b",
        "https://pubmed.ncbi.nlm.nih.gov/31234567/",
        "PMID: 29876543",
        "PMC7654321",
        f"https://www.nature.com/articles/s41586-021-{9000 + 12}-3",
        f"https://ieeexplore.ieee.org/document/{8000000 + 1234}",
        "https://openreview.net/forum?id=Sy2fzU9gl",
        "S2:5f2b8c1d9e",
    ])
    w("lists/ok-mixed-identifiers.txt", mixed + "\n")

    # ok-title-list: one title per line, no identifiers at all
    w("lists/ok-title-list.txt", "\n".join(TITLES[:8]) + "\n")

    # edge-crlf-bom
    w("lists/edge-crlf-bom.txt",
      b"\xef\xbb\xbf" + (dois + "\n").replace("\n", "\r\n").encode("utf-8"))

    # edge-comma-separated: everything on ONE line
    w("lists/edge-single-line-commas.txt",
      ", ".join(doi(i) for i in range(20, 32)) + "\n")

    # edge-word-paste: DOIs pasted out of Word with smart quotes and nbsp
    w("lists/edge-word-paste.txt",
      "“" + doi(41) + "” \n"
      "• " + doi(42) + " (accessed 2026)\n"
      "– " + doi(43) + " \n")

    # edge-duplicates: same DOI in three spellings
    w("lists/edge-duplicate-identifiers.txt",
      f"{doi(51)}\nhttps://doi.org/{doi(51)}\nDOI: {doi(51).upper()}\n{doi(52)}\n")

    # partial-some-bogus
    w("lists/partial-some-bogus.txt",
      f"{doi(61)}\n10.9999/definitely.not.a.real.doi\n{arxiv(62)}\n"
      "9999.99999\nnot-an-identifier-at-all\n")

    # bad-prose: a paragraph of text, no identifiers
    w("lists/bad-prose.txt",
      "These are my notes from the reading group. We talked about weight "
      "vectors and about whether crowding distance is still the right "
      "diversity measure. Nobody wrote down any DOIs.\n")

    # bad-empty / bad-whitespace-only
    w("lists/bad-empty.txt", "")
    w("lists/bad-whitespace-only.txt", "   \n\t\n\n   \n")

    # edge-huge: 20k identifiers
    w("lists/edge-huge-20k-dois.txt",
      "\n".join(doi(i) for i in range(20000)) + "\n")


# ----------------------------------------------------------------------- JSON

def gen_json() -> None:
    # CSL-JSON (Zotero "Export → CSL JSON", Pandoc, Better BibTeX)
    csl = []
    for i in range(8):
        e = rec(i + 130)
        csl.append({
            "id": e["key"],
            "type": "article-journal" if i % 2 else "paper-conference",
            "title": e["title"],
            "author": [{"family": s, "given": g} for s, g in e["authors"]],
            "issued": {"date-parts": [[e["year"], (i % 12) + 1]]},
            "container-title": e["venue"],
            "volume": e["volume"],
            "issue": e["number"],
            "page": e["pages"].replace("--", "-"),
            "DOI": e["doi"],
            "URL": f"https://doi.org/{e['doi']}",
        })
    w("json/ok-csl-json.json", json.dumps(csl, ensure_ascii=False, indent=2) + "\n")

    # S2-shaped array — exactly what /api/seeds/search returns today
    s2 = []
    for i in range(6):
        e = rec(i + 150)
        s2.append({
            "id": f"{i:040x}",
            "title": e["title"],
            "authors": f"{e['authors'][0][0]} et al.",
            "year": e["year"],
            "venue": e["venue"],
            "cites": 100 + i * 37,
            "externalIds": {"DOI": e["doi"], "ArXiv": e["arxiv"]},
        })
    w("json/ok-s2-seed-rows.json", json.dumps(s2, indent=2) + "\n")

    # CiteClaw's own literature_collection.json — the re-import-a-run case
    papers = []
    for i in range(6):
        e = rec(i + 170)
        papers.append({
            "paper_id": f"{i:040x}",
            "title": e["title"],
            "abstract": "Synthetic abstract for the import fixture corpus.",
            "year": e["year"],
            "publication_date": f"{e['year']}-06-01",
            "venue": e["venue"],
            "citation_count": 200 - i * 11,
            "influential_citation_count": 10 - i,
            "references": [],
            "depth": i % 3,
            "source": "seed" if i == 0 else "forward",
            "llm_verdict": "accept",
            "llm_reasoning": "",
            "supporting_papers": [],
            "expanded": False,
            "pdf_url": None,
            "authors": [f"{g} {s}" for s, g in e["authors"]],
            "external_ids": {"DOI": e["doi"], "ArXiv": e["arxiv"]},
        })
    w("json/ok-citeclaw-collection.json", json.dumps(
        {"summary": {"total_accepted": len(papers), "total_rejected": 0,
                     "total_seen": len(papers)},
         "papers": papers}, indent=2) + "\n")

    # edge-ndjson: one object per line (JSON Lines) under a .json name
    w("json/edge-ndjson.json",
      "\n".join(json.dumps(o) for o in s2) + "\n")

    # edge-nested-envelope: the array is buried under {"data": {"items": [...]}}
    w("json/edge-nested-envelope.json",
      json.dumps({"status": "ok", "data": {"items": csl[:3]}}, indent=2) + "\n")

    # bad-trailing-comma / bad-empty / bad-not-an-array
    w("json/bad-trailing-comma.json",
      '[\n  {"title": "A", "DOI": "10.1/a"},\n  {"title": "B", "DOI": "10.1/b"},\n]\n')
    w("json/bad-empty.json", "")
    w("json/bad-unrelated-schema.json",
      json.dumps({"version": 3, "settings": {"theme": "dark"},
                  "windows": [{"w": 1280, "h": 800}]}, indent=2) + "\n")


# ------------------------------------------------------------------------ PDF

def mkpdf(title: str, author: str, body_lines: list[str],
          *, with_info: bool = True) -> bytes:
    """Hand-rolled single-page PDF 1.4 with a correct xref table."""
    def esc(s: str) -> str:
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    text = "BT\n/F1 11 Tf\n72 720 Td\n14 TL\n"
    for ln in body_lines:
        text += f"({esc(ln)}) Tj\nT*\n"
    text += "ET\n"
    tb = text.encode("latin-1", "replace")

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(tb)).encode() + b" >>\nstream\n" + tb + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    if with_info:
        objs.append(
            b"<< /Title (" + esc(title).encode("latin-1", "replace") +
            b") /Author (" + esc(author).encode("latin-1", "replace") +
            b") /Producer (OmniKnowledge import fixture generator) >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + o + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    trailer = f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R"
    if with_info:
        trailer += f" /Info {len(objs)} 0 R"
    trailer += f" >>\nstartxref\n{xref_at}\n%%EOF\n"
    out += trailer.encode()
    return bytes(out)


def gen_pdf() -> None:
    e = rec(200)
    w("pdf/ok-single-with-doi.pdf", mkpdf(
        e["title"], f"{e['authors'][0][1]} {e['authors'][0][0]}",
        [e["title"],
         f"{e['authors'][0][1]} {e['authors'][0][0]}, {e['authors'][1][1]} {e['authors'][1][0]}",
         e["venue"] + f", {e['year']}",
         f"https://doi.org/{e['doi']}",
         "",
         "Abstract. Synthetic single-page PDF used by the OmniKnowledge",
         "import fixture corpus. It carries a DOI in the body text and in",
         "the /Info dictionary title, so both extraction paths have a target."]))

    for k in range(3):
        e = rec(210 + k)
        w(f"pdf/ok-batch-{k + 1}.pdf", mkpdf(
            e["title"], e["authors"][0][0],
            [e["title"], e["venue"] + f", {e['year']}", f"doi:{e['doi']}"]))

    # no /Info, no DOI anywhere — title match is the only path
    w("pdf/edge-no-metadata.pdf", mkpdf(
        "", "", ["Neighborhood adaptation strategies for large-scale",
                 "variable interaction", "", "1  Introduction",
                 "Synthetic body text with no identifier of any kind."],
        with_info=False))

    # scanned-look: a page with no extractable text objects at all
    w("pdf/edge-no-text-layer.pdf", mkpdf(
        "Scan of a printed paper", "Unknown", [], with_info=True))

    # bad-not-a-pdf: plain text wearing a .pdf extension
    w("pdf/bad-not-a-pdf.pdf",
      "This is a plain text file that someone renamed to .pdf.\n"
      "There is no %PDF header on the first line.\n")

    # bad-truncated-pdf: valid header, xref chopped off
    good = mkpdf("Truncated", "X", ["a", "b", "c"])
    w("pdf/bad-truncated.pdf", good[: int(len(good) * 0.55)])

    # bad-empty
    w("pdf/bad-empty.pdf", b"")


# -------------------------------------------------------------------- archives

def gen_archives() -> None:
    zp = ROOT / "archives"
    zp.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zp / "ok-pdfs.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for k in range(3):
            z.write(ROOT / f"pdf/ok-batch-{k + 1}.pdf", f"papers/paper-{k + 1}.pdf")

    with zipfile.ZipFile(zp / "ok-mixed-formats.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "bibtex/ok-basic.bib", "library/refs.bib")
        z.write(ROOT / "ris/ok-basic.ris", "library/refs.ris")
        z.write(ROOT / "lists/ok-doi-list.txt", "library/dois.txt")

    with zipfile.ZipFile(zp / "edge-nested-and-junk.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "bibtex/ok-basic.bib", "outer/inner/deeper/refs.bib")
        z.writestr("__MACOSX/._refs.bib", b"\x00\x05\x16\x07\x00\x02\x00\x00")
        z.writestr(".DS_Store", b"\x00\x00\x00\x01Bud1")
        z.writestr("outer/notes.md", "# not a bibliography\n")
        z.writestr("outer/empty/", b"")

    # zip-slip style path traversal name — must never be written to disk
    with zipfile.ZipFile(zp / "bad-path-traversal.zip", "w") as z:
        z.writestr("../../../../tmp/omk-import-escape.txt", "should never land\n")
        z.write(ROOT / "bibtex/ok-basic.bib", "refs.bib")

    # truncated archive
    data = (zp / "ok-mixed-formats.zip").read_bytes()
    (zp / "bad-truncated.zip").write_bytes(data[: int(len(data) * 0.6)])

    # empty archive (valid zip, zero members)
    with zipfile.ZipFile(zp / "bad-no-members.zip", "w"):
        pass

    # a "zip bomb lite": one small file that expands ~200x (not a real bomb,
    # just enough to pin an uncompressed-size guard in the parser spec)
    with zipfile.ZipFile(zp / "edge-high-compression.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("padding.txt", "A" * 4_000_000)


# ------------------------------------------------- vendor exports & adversarial

def gen_vendor() -> None:
    # Zotero BibTeX flavour: file= attachments, keywords, abstract, month macro
    zb = ""
    for i in range(4):
        e = rec(i + 230)
        au = " and ".join(f"{s}, {g}" for s, g in e["authors"])
        zb += (
            f"@article{{{e['key']},\n"
            f"  title = {{{e['title']}}},\n"
            f"  volume = {{{e['volume']}}},\n"
            f"  issn = {{1089-778X}},\n"
            f"  url = {{https://doi.org/{e['doi']}}},\n"
            f"  doi = {{{e['doi']}}},\n"
            f"  abstract = {{Synthetic abstract for the fixture corpus.}},\n"
            f"  number = {{{e['number']}}},\n"
            f"  journal = {{{e['venue']}}},\n"
            f"  author = {{{au}}},\n"
            f"  month = jun,\n"
            f"  year = {{{e['year']}}},\n"
            f"  keywords = {{multi-objective, decomposition}},\n"
            f"  pages = {{{e['pages']}}},\n"
            f"  file = {{Full Text PDF:/Users/someone/Zotero/storage/ABCD{i:04d}/"
            f"Zhang et al. - {e['year']} - paper.pdf:application/pdf}},\n"
            f"}}\n\n"
        )
    w("vendor/ok-zotero-betterbibtex.bib", zb)

    # Mendeley RIS flavour: L1 file paths, M3 doi, UR list
    mr = ""
    for i in range(4):
        e = rec(i + 250)
        mr += (
            "TY  - JOUR\n"
            f"T1  - {e['title']}\n"
            + "".join(f"A1  - {s}, {g}\n" for s, g in e["authors"]) +
            f"JF  - {e['venue']}\n"
            f"Y1  - {e['year']}///\n"
            f"VL  - {e['volume']}\n"
            f"IS  - {e['number']}\n"
            f"SP  - {e['pages'].split('--')[0]}\n"
            f"EP  - {e['pages'].split('--')[1]}\n"
            f"M3  - {e['doi']}\n"
            f"UR  - https://doi.org/{e['doi']}\n"
            f"L1  - file:///Users/someone/Mendeley/{e['key']}.pdf\n"
            "ER  - \n\n"
        )
    w("vendor/ok-mendeley.ris", mr)

    # EndNote native tagged export (.enw) — NOT RIS
    en = ""
    for i in range(3):
        e = rec(i + 270)
        en += (
            "%0 Journal Article\n"
            + "".join(f"%A {s}, {g}\n" for s, g in e["authors"]) +
            f"%T {e['title']}\n"
            f"%J {e['venue']}\n"
            f"%D {e['year']}\n"
            f"%V {e['volume']}\n"
            f"%N {e['number']}\n"
            f"%P {e['pages'].replace('--', '-')}\n"
            f"%R {e['doi']}\n\n"
        )
    w("vendor/bad-endnote-native.enw", en)

    # EndNote XML export
    ex = ['<?xml version="1.0" encoding="UTF-8"?>', "<xml><records>"]
    for i in range(3):
        e = rec(i + 280)
        ex.append(
            "<record><ref-type name=\"Journal Article\">17</ref-type>"
            "<contributors><authors>"
            + "".join(f"<author>{s}, {g}</author>" for s, g in e["authors"]) +
            "</authors></contributors>"
            f"<titles><title>{e['title']}</title>"
            f"<secondary-title>{e['venue']}</secondary-title></titles>"
            f"<dates><year>{e['year']}</year></dates>"
            f"<electronic-resource-num>{e['doi']}</electronic-resource-num>"
            "</record>"
        )
    ex.append("</records></xml>")
    w("vendor/bad-endnote-xml.xml", "\n".join(ex) + "\n")

    # Zotero RDF export
    w("vendor/bad-zotero-rdf.rdf",
      '<?xml version="1.0"?>\n'
      '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n'
      '         xmlns:bib="http://purl.org/net/biblio#"\n'
      '         xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
      '  <bib:Article rdf:about="urn:doi:10.1109/omk.2020.00001">\n'
      f'    <dc:title>{TITLES[0]}</dc:title>\n'
      '    <dc:date>2020</dc:date>\n'
      '  </bib:Article>\n'
      '</rdf:RDF>\n')


def gen_adversarial() -> None:
    # wrong extension, right content
    w("adversarial/wrong-ext-bibtex-named-ris.ris",
      (ROOT / "bibtex/ok-basic.bib").read_text("utf-8"))
    w("adversarial/wrong-ext-doilist-named-bib.bib",
      (ROOT / "lists/ok-doi-list.txt").read_text("utf-8"))
    w("adversarial/wrong-ext-csv-named-txt.txt",
      (ROOT / "csv/ok-doi-column.csv").read_text("utf-8"))
    w("adversarial/wrong-ext-pdf-named-bib.bib",
      (ROOT / "pdf/ok-single-with-doi.pdf").read_bytes())

    # no extension at all
    w("adversarial/no-extension-bibtex",
      (ROOT / "bibtex/ok-basic.bib").read_text("utf-8"))

    # double extension / uppercase extension
    w("adversarial/ok-uppercase-extension.BIB",
      (ROOT / "bibtex/ok-basic.bib").read_text("utf-8"))
    w("adversarial/edge-double-extension.bib.txt",
      (ROOT / "bibtex/ok-basic.bib").read_text("utf-8"))

    # genuinely unsupported types
    w("adversarial/bad-unsupported.docx",
      b"PK\x03\x04" + b"\x00" * 40 + b"[Content_Types].xml")
    w("adversarial/bad-unsupported.pages", b"PK\x03\x04" + b"\x00" * 24)
    w("adversarial/bad-unsupported.md", "# Reading list\n\n- Zhang 2007, MOEA/D\n")

    # HTML injection in metadata — the review row renders titles into innerHTML
    w("adversarial/edge-html-injection.bib",
      "@article{xss2026,\n"
      "  author = {<img src=x onerror=alert(1)>, A.},\n"
      "  title  = {<script>alert('xss')</script> & \"quoted\" &amp; entities},\n"
      "  journal = {<b>Venue</b>},\n"
      "  year   = {2026},\n"
      "  doi    = {10.1109/omk.2026.00001}\n"
      "}\n")

    # pathological single line (no newline in 400 KB)
    w("adversarial/edge-single-long-line.txt", "A" * 400_000)

    # very long field values inside a legal entry
    w("adversarial/edge-very-long-title.bib",
      "@article{longtitle,\n"
      "  author = {Zhang, Qingfu},\n"
      "  title  = {" + ("A very long title segment " * 400).strip() + "},\n"
      "  year   = {2025},\n"
      "  doi    = {10.1109/omk.2025.09999}\n"
      "}\n")

    # filename torture: spaces, unicode, emoji, %20, and a leading dash
    w("adversarial/-leading dash 参考 %20 refs 📄.bib",
      (ROOT / "bibtex/ok-basic.bib").read_text("utf-8"))


def main() -> None:
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    gen_bibtex()
    gen_ris()
    gen_csv()
    gen_lists()
    gen_json()
    gen_pdf()
    gen_archives()
    gen_vendor()
    gen_adversarial()

    total = 0
    count = 0
    for base, _dirs, files in os.walk(ROOT):
        for f in files:
            if f in ("_generate.py", "README.md"):
                continue
            total += (Path(base) / f).stat().st_size
            count += 1
    print(f"{count} fixtures, {total / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
