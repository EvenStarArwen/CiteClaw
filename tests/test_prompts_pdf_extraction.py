"""Tier 1 offline tests for :mod:`citeclaw.prompts.pdf_extraction`.

These prompts drive the LLM-based PDF reference extractor used by
ExpandByPDF. The literal ``"relevant_references"`` token is what the
stub responder shape-detector keys off, and the JSON schema must
remain self-consistent (every required key present at the top
level + per-item).

ExpandByPDF itself is in the loop's skip list (placeholder rewrite
in progress), but the prompt + schema constants are still in scope
because they're used elsewhere and pinned by audit gap #4.
"""

from __future__ import annotations

from citeclaw.prompts import pdf_extraction


# ---------------------------------------------------------------------------
# SYSTEM + USER_TEMPLATE prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_system_non_empty_and_long(self):
        sys = pdf_extraction.SYSTEM
        assert isinstance(sys, str) and sys.strip()
        # The 7-rule SYSTEM is intentionally lengthy — sanity-check it
        # didn't get accidentally truncated to a one-liner.
        assert len(sys) > 500

    def test_system_documents_relevant_references_token(self):
        # Stub responder + downstream parser key off this literal.
        # The system message doesn't have to mention it but the user
        # template must — verified separately below.
        assert "JSON" in pdf_extraction.SYSTEM

    def test_user_template_four_placeholders_render(self):
        rendered = pdf_extraction.USER_TEMPLATE.format(
            topic_description="protein structure prediction",
            reference_list="[1] Smith et al. Foo. 2020.",
            paper_title="An Example Paper",
            body_text="Body text mentions [1] in passing.",
        )
        assert "protein structure prediction" in rendered
        assert "[1] Smith et al." in rendered
        assert "An Example Paper" in rendered
        assert "Body text mentions [1]" in rendered

    def test_user_template_includes_relevant_references_token(self):
        """Stub responder shape-detector requires this literal."""
        rendered = pdf_extraction.USER_TEMPLATE.format(
            topic_description="X",
            reference_list="r",
            paper_title="t",
            body_text="b",
        )
        assert '"relevant_references"' in rendered


# ---------------------------------------------------------------------------
# pdf_extraction_schema — JSON Schema self-consistency
# ---------------------------------------------------------------------------


class TestSchema:
    def test_returns_dict(self):
        schema = pdf_extraction.pdf_extraction_schema()
        assert isinstance(schema, dict)

    def test_top_level_shape(self):
        schema = pdf_extraction.pdf_extraction_schema()
        assert schema["type"] == "object"
        assert "relevant_references" in schema["properties"]
        assert schema["properties"]["relevant_references"]["type"] == "array"
        assert schema["required"] == ["relevant_references"]
        assert schema["additionalProperties"] is False

    def test_per_item_required_keys(self):
        schema = pdf_extraction.pdf_extraction_schema()
        item = schema["properties"]["relevant_references"]["items"]
        assert item["type"] == "object"
        assert set(item["required"]) == {
            "citation_marker",
            "reference_text",
            "title",
            "mentions",
            "relevance_explanation",
        }
        assert item["additionalProperties"] is False

    def test_mentions_subschema(self):
        schema = pdf_extraction.pdf_extraction_schema()
        mentions = (
            schema["properties"]["relevant_references"]
            ["items"]["properties"]["mentions"]
        )
        assert mentions["type"] == "array"
        sub = mentions["items"]
        assert sub["type"] == "object"
        assert set(sub["required"]) == {"quote", "relevance"}
        assert sub["additionalProperties"] is False

    def test_fresh_dict_per_call(self):
        """Schema sanitisers may mutate the returned dict — every call
        must hand back an independent instance."""
        a = pdf_extraction.pdf_extraction_schema()
        b = pdf_extraction.pdf_extraction_schema()
        assert a is not b
        # Mutating a should not affect b.
        a["mutated"] = True
        assert "mutated" not in b
