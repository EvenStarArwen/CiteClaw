"""Tier 1 offline tests for :mod:`citeclaw.prompts.screening`.

These prompt constants drive every :class:`citeclaw.filters.atoms.llm_query.LLMFilter`
batch — silently breaking a placeholder name or stripping the
``"match"`` token would make every screening call return all-False
verdicts. Tests pin the constants, the placeholder set, and the
structural markers the stub responder + structured-output schemas
rely on.
"""

from __future__ import annotations

from citeclaw.prompts import screening


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


class TestSystemPrompts:
    def test_system_is_non_empty_str(self):
        assert isinstance(screening.SYSTEM, str)
        assert screening.SYSTEM.strip()

    def test_system_mentions_json(self):
        # All providers must produce JSON; spec says so explicitly.
        assert "JSON" in screening.SYSTEM or "json" in screening.SYSTEM

    def test_system_documents_yes_no_contract(self):
        # The "YES (match) / NO (no match)" framing is what makes the
        # response schema consistent across providers — pin it.
        sys = screening.SYSTEM
        assert "YES" in sys and "NO" in sys
        assert "match" in sys.lower()

    def test_venue_system_is_non_empty_str(self):
        assert isinstance(screening.VENUE_SYSTEM, str)
        assert screening.VENUE_SYSTEM.strip()

    def test_venue_system_describes_venue_identity_rule(self):
        # The "treat abbreviations and full names as the same venue"
        # contract is the entire reason VENUE_SYSTEM is separate from
        # SYSTEM — pin it.
        v = screening.VENUE_SYSTEM
        assert "venue" in v.lower()
        assert "abbrev" in v.lower() or "identity" in v.lower()


# ---------------------------------------------------------------------------
# USER_TEMPLATE — accepts {criterion, label, n, block}; n appears 3x
# ---------------------------------------------------------------------------


class TestUserTemplate:
    def test_four_placeholders_render(self):
        rendered = screening.USER_TEMPLATE.format(
            criterion="papers about machine learning",
            label="Items",
            n=3,
            block="1. Item one\n2. Item two\n3. Item three",
        )
        assert "papers about machine learning" in rendered
        assert "Items (3 items" in rendered
        assert "Item one" in rendered
        # The "exactly N results / exactly N items" repeat is what
        # constrains models to return one result per index.
        assert "exactly 3" in rendered

    def test_match_token_present(self):
        """Stub responder + downstream parser both look for the
        ``"match"`` token — it must appear in the rendered template."""
        rendered = screening.USER_TEMPLATE.format(
            criterion="X", label="Items", n=1, block="1. y",
        )
        # Either '"match": true' (in the schema example) or 'match'
        # (free-form) must appear so the stub responder shape detection
        # in citeclaw.clients.llm.stub.stub_respond('"match"' branch)
        # fires.
        assert '"match"' in rendered

    def test_results_envelope_documented(self):
        """Schema example uses ``{"results": [...]}`` so every provider
        emits the wrapped form."""
        rendered = screening.USER_TEMPLATE.format(
            criterion="X", label="Items", n=2, block="1. y\n2. z",
        )
        assert '"results"' in rendered

    def test_render_idempotent(self):
        a = screening.USER_TEMPLATE.format(
            criterion="X", label="Items", n=1, block="1. y",
        )
        b = screening.USER_TEMPLATE.format(
            criterion="X", label="Items", n=1, block="1. y",
        )
        assert a == b

    def test_label_swappable_to_venues(self):
        """The same template serves the venue scope (label="Venues")."""
        rendered = screening.USER_TEMPLATE.format(
            criterion="biology venues",
            label="Venues",
            n=2,
            block="1. Cell\n2. Nature",
        )
        assert "Venues (2 items" in rendered
        assert "Cell" in rendered
        assert "Nature" in rendered
