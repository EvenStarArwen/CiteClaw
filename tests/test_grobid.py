"""Tier 1 offline tests for :mod:`pdfclaw.parsers.grobid`.

GROBID is a network-bound HTTP client; every test in this file
either:

* exercises pure-Python TEI walkers (``_extract_body``,
  ``_extract_references``, ``_plain``) on hand-crafted TEI XML
  fixtures, or
* stubs :class:`GrobidParser._post` so the network path runs
  against a fake response.

No live GROBID server is required.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
import pytest

from pdfclaw.parsers import ParserError
from pdfclaw.parsers.grobid import (
    _ENV_PRIMARY,
    GrobidParser,
    _extract_body,
    _extract_references,
    _plain,
)


# ---------------------------------------------------------------------------
# Test fixtures — hand-rolled TEI shaped like Grobid's processFulltextDocument
# response. Kept minimal so the field semantics under test stay obvious.
# ---------------------------------------------------------------------------

_TEI_BODY_ONLY = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div>
        <head>Introduction</head>
        <p>The first paragraph.</p>
        <p>The second paragraph spans
           multiple lines.</p>
      </div>
      <div>
        <head>Methods</head>
        <p>Single methods paragraph.</p>
      </div>
    </body>
  </text>
</TEI>"""

_TEI_RAW_REFERENCES = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <back>
      <div>
        <listBibl>
          <biblStruct><note type="raw_reference">[1] Smith J. Foo. Nature 2020.</note></biblStruct>
          <biblStruct><note type="raw_reference">[2] Doe A. Bar. Cell 2021.</note></biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>"""

_TEI_STRUCTURED_REFERENCE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <back>
      <div>
        <listBibl>
          <biblStruct>
            <analytic>
              <author><persName><forename>Jane</forename><surname>Doe</surname></persName></author>
              <author><persName><forename>John</forename><surname>Roe</surname></persName></author>
              <title>The Structured Title</title>
            </analytic>
            <monogr>
              <title>Journal of Stuff</title>
              <imprint><date when="2022"/></imprint>
            </monogr>
            <idno type="DOI">10.1000/abcd</idno>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>"""

_TEI_FULL = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div><head>Intro</head><p>Body paragraph.</p></div>
    </body>
    <back>
      <div>
        <listBibl>
          <biblStruct><note type="raw_reference">[1] Ref A.</note></biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>"""


# ---------------------------------------------------------------------------
# GrobidParser._resolved_base_url — env-var fallback when ``base_url`` is
# unset
# ---------------------------------------------------------------------------


class TestResolvedBaseUrl:
    def test_uses_constructor_value_when_given(self, monkeypatch):
        monkeypatch.delenv(_ENV_PRIMARY, raising=False)
        p = GrobidParser(base_url="https://from-arg.example.com/")
        assert p._resolved_base_url() == "https://from-arg.example.com"

    def test_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv(_ENV_PRIMARY, "https://from-env.example.com/")
        p = GrobidParser()
        assert p._resolved_base_url() == "https://from-env.example.com"

    def test_returns_blank_when_neither(self, monkeypatch):
        monkeypatch.delenv(_ENV_PRIMARY, raising=False)
        monkeypatch.delenv("CITECLAW_GROBID_URL", raising=False)
        assert GrobidParser()._resolved_base_url() == ""

    def test_legacy_env_var_still_accepted(self, monkeypatch):
        monkeypatch.delenv(_ENV_PRIMARY, raising=False)
        monkeypatch.setenv("CITECLAW_GROBID_URL", "https://legacy.example.com")
        assert GrobidParser()._resolved_base_url() == "https://legacy.example.com"


# ---------------------------------------------------------------------------
# GrobidParser.parse — top-level orchestration + httpx fallback
# ---------------------------------------------------------------------------


class TestGrobidParserParse:
    def test_raises_when_no_url_configured(self, monkeypatch):
        monkeypatch.delenv(_ENV_PRIMARY, raising=False)
        monkeypatch.delenv("CITECLAW_GROBID_URL", raising=False)
        with pytest.raises(ParserError):
            GrobidParser().parse(b"%PDF-1.4 fake")

    def test_raises_on_http_error(self, monkeypatch):
        """Network failures bubble up as ParserError."""
        p = GrobidParser(base_url="https://grobid.example.com")
        monkeypatch.setattr(
            p, "_post",
            lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("simulated outage")),
        )
        with pytest.raises(ParserError):
            p.parse(b"%PDF-1.4 fake")

    def test_raises_on_malformed_xml(self, monkeypatch):
        """Malformed TEI raises ParserError rather than returning empty."""
        p = GrobidParser(base_url="https://grobid.example.com")
        monkeypatch.setattr(p, "_post", lambda *a, **kw: "<not-xml<<")
        with pytest.raises(ParserError):
            p.parse(b"%PDF-1.4 fake")

    def test_happy_path_returns_body_and_references(self, monkeypatch):
        """A clean TEI response yields populated body_text and references."""
        p = GrobidParser(base_url="https://grobid.example.com")
        monkeypatch.setattr(p, "_post", lambda *a, **kw: _TEI_FULL)
        result = p.parse(b"%PDF-1.4 fake")
        assert result.parser_used == "grobid"
        assert "Body paragraph." in result.body_text
        # References list is structured (not concatenated into body_text).
        assert result.references == ["[1] Ref A."]
        assert "References" not in result.body_text


# ---------------------------------------------------------------------------
# _extract_body — div / head / p flattening
# ---------------------------------------------------------------------------


class TestExtractBody:
    def test_collects_heads_and_paragraphs_in_order(self):
        root = ET.fromstring(_TEI_BODY_ONLY)
        lines = _extract_body(root)
        # Both section heads + 3 paragraphs across 2 divs
        assert lines == [
            "Introduction",
            "The first paragraph.",
            "The second paragraph spans multiple lines.",
            "Methods",
            "Single methods paragraph.",
        ]

    def test_returns_empty_list_when_no_body(self):
        root = ET.fromstring(
            '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text></text></TEI>'
        )
        assert _extract_body(root) == []


# ---------------------------------------------------------------------------
# _extract_references — raw + structured + numbering preservation
# ---------------------------------------------------------------------------


class TestExtractReferences:
    def test_prefers_raw_reference_text(self):
        root = ET.fromstring(_TEI_RAW_REFERENCES)
        refs = _extract_references(root)
        # Raw text preserved verbatim (whitespace collapsed) — sequential [N]
        # is NOT prepended when raw text already has its own numbering.
        assert refs == [
            "[1] Smith J. Foo. Nature 2020.",
            "[2] Doe A. Bar. Cell 2021.",
        ]

    def test_structured_fallback_assembles_authors_title_venue_year_doi(self):
        root = ET.fromstring(_TEI_STRUCTURED_REFERENCE)
        refs = _extract_references(root)
        # No raw_reference present → fallback assembles a [N] prefixed line.
        assert len(refs) == 1
        line = refs[0]
        assert line.startswith("[1]")
        assert "J. Doe" in line
        assert "J. Roe" in line
        assert "The Structured Title" in line
        assert "Journal of Stuff" in line
        assert "2022" in line
        assert "doi:10.1000/abcd" in line

    def test_returns_empty_list_when_no_listbibl(self):
        root = ET.fromstring(
            '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text></text></TEI>'
        )
        assert _extract_references(root) == []


# ---------------------------------------------------------------------------
# _plain — whitespace collapse + sub-element text + tails
# ---------------------------------------------------------------------------


class TestPlain:
    def test_collapses_whitespace(self):
        el = ET.fromstring("<x>foo    bar\n\n\nbaz</x>")
        assert _plain(el) == "foo bar baz"

    def test_returns_empty_on_empty_element(self):
        el = ET.fromstring("<x></x>")
        assert _plain(el) == ""

    def test_concatenates_sub_text_and_tails(self):
        # ``<x>A<y>B</y>C<z>D</z>E</x>`` — _plain joins (text, sub.text, sub.tail)
        # in document order; deeper nesting beyond one level isn't walked.
        el = ET.fromstring("<x>A<y>B</y>C<z>D</z>E</x>")
        out = _plain(el)
        # Per the implementation, expect the top-level text plus each sub's text + tail.
        assert "A" in out and "B" in out and "C" in out and "D" in out and "E" in out
