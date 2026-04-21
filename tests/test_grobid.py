"""Tier 1 offline tests for :mod:`citeclaw.clients.grobid`.

GROBID is a network-bound HTTP client; every test in this file
either:

* exercises pure-Python helpers (`_tei_to_text`, `_extract_body`,
  `_extract_references`, `_plain`, `grobid_url`) on hand-crafted
  TEI XML fixtures, or
* monkey-patches :class:`httpx.Client` so the network path runs
  against a fake response.

No live GROBID server is required.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
import pytest

from citeclaw.clients import grobid as grobid_mod
from citeclaw.clients.grobid import (
    ENV_GROBID_URL,
    _extract_body,
    _extract_references,
    _plain,
    _tei_to_text,
    grobid_url,
    parse_pdf_with_grobid,
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
</TEI>
"""

_TEI_RAW_REFERENCES = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <back>
      <div>
        <listBibl>
          <biblStruct>
            <note type="raw_reference">[1] Smith J. Foo. Nature 2020.</note>
          </biblStruct>
          <biblStruct>
            <note type="raw_reference">[2] Doe A. Bar. Cell 2021.</note>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""

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
              <imprint>
                <date when="2022-05-01"/>
              </imprint>
            </monogr>
            <idno type="DOI">10.1000/abcd</idno>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""

_TEI_FULL = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div><head>H</head><p>Body paragraph.</p></div>
    </body>
    <back>
      <div>
        <listBibl>
          <biblStruct>
            <note type="raw_reference">[1] Ref A.</note>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""


# ---------------------------------------------------------------------------
# grobid_url — env var presence / absence / trailing-slash strip
# ---------------------------------------------------------------------------


class TestGrobidUrl:
    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv(ENV_GROBID_URL, raising=False)
        assert grobid_url() is None

    def test_returns_none_when_blank(self, monkeypatch):
        monkeypatch.setenv(ENV_GROBID_URL, "   ")
        assert grobid_url() is None

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv(ENV_GROBID_URL, "https://grobid.example.com/")
        assert grobid_url() == "https://grobid.example.com"

    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv(ENV_GROBID_URL, "https://grobid.example.com")
        assert grobid_url() == "https://grobid.example.com"


# ---------------------------------------------------------------------------
# parse_pdf_with_grobid — top-level orchestration + httpx fallback
# ---------------------------------------------------------------------------


class TestParsePdfWithGrobid:
    def test_returns_none_when_no_url_configured(self, monkeypatch):
        monkeypatch.delenv(ENV_GROBID_URL, raising=False)
        assert parse_pdf_with_grobid(b"%PDF-1.4 fake") is None

    def test_returns_none_on_http_error(self, monkeypatch):
        """Network failures fall through to the PyMuPDF path — return None."""

        class _FakeClient:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def post(self, *a, **kw):
                raise httpx.ConnectError("simulated outage")

        monkeypatch.setattr(grobid_mod.httpx, "Client", _FakeClient)
        result = parse_pdf_with_grobid(b"%PDF-1.4 fake", base_url="https://grobid.example.com")
        assert result is None

    def test_returns_none_on_malformed_xml(self, monkeypatch):
        """Malformed TEI XML falls through to None."""

        class _FakeResponse:
            text = "<not-xml<<"

            def raise_for_status(self):
                pass

        class _FakeClient:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def post(self, *a, **kw):
                return _FakeResponse()

        monkeypatch.setattr(grobid_mod.httpx, "Client", _FakeClient)
        result = parse_pdf_with_grobid(b"%PDF-1.4 fake", base_url="https://grobid.example.com")
        assert result is None

    def test_happy_path_returns_body_plus_references(self, monkeypatch):
        """A clean TEI response yields body \\n\\n References \\n\\n list."""

        class _FakeResponse:
            text = _TEI_FULL

            def raise_for_status(self):
                pass

        class _FakeClient:
            def __init__(self, *a, **kw):
                self.posted = None

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def post(self, endpoint, *, files, data, **kw):
                self.posted = (endpoint, files, data)
                return _FakeResponse()

        monkeypatch.setattr(grobid_mod.httpx, "Client", _FakeClient)
        result = parse_pdf_with_grobid(
            b"%PDF-1.4 fake",
            base_url="https://grobid.example.com",
            include_raw_citations=True,
        )
        assert result is not None
        # Body text is present
        assert "Body paragraph." in result
        # References section is present + raw entry preserved
        assert "References" in result
        assert "[1] Ref A." in result


# ---------------------------------------------------------------------------
# _tei_to_text — body+refs concatenation
# ---------------------------------------------------------------------------


class TestTeiToText:
    def test_body_only_no_references_section(self):
        out = _tei_to_text(_TEI_BODY_ONLY)
        assert out is not None
        assert "Introduction" in out
        assert "The first paragraph." in out
        assert "Methods" in out
        # No refs in fixture → no "References" heading
        assert "References" not in out

    def test_returns_none_on_empty(self):
        empty_tei = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            '<text><body></body></text></TEI>'
        )
        assert _tei_to_text(empty_tei) is None

    def test_body_then_references_separator(self):
        out = _tei_to_text(_TEI_FULL)
        assert out is not None
        # Body comes first, then "References" heading, then refs
        body_pos = out.index("Body paragraph.")
        refs_pos = out.index("References")
        ref_entry_pos = out.index("[1] Ref A.")
        assert body_pos < refs_pos < ref_entry_pos


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
