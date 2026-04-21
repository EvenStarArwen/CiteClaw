"""Tier 1 offline tests for :mod:`citeclaw.prompts.topic_naming`.

These prompts are consumed by :mod:`citeclaw.cluster.representation`
to label clusters. The stub responder relies on the literal
``"topic_label"`` token + ``cluster_id=`` markers to detect shape;
this file pins both.
"""

from __future__ import annotations

import re

from citeclaw.prompts import topic_naming


# ---------------------------------------------------------------------------
# Single-cluster shape (legacy)
# ---------------------------------------------------------------------------


class TestSingleShape:
    def test_system_non_empty_mentions_topic_label(self):
        sys = topic_naming.SYSTEM
        assert isinstance(sys, str) and sys.strip()
        assert "topic_label" in sys
        assert "JSON" in sys

    def test_user_template_two_placeholders_render(self):
        rendered = topic_naming.USER_TEMPLATE.format(
            keywords="learning, neural, network",
            documents="1. Title A\n2. Title B",
        )
        assert "learning, neural, network" in rendered
        assert "Title A" in rendered
        # Stub-responder shape detection requires the literal string.
        assert "topic_label" in rendered
        assert "summary" in rendered


# ---------------------------------------------------------------------------
# Batched shape (BATCH_SYSTEM + BATCH_USER_TEMPLATE + CLUSTER_BLOCK_TEMPLATE)
# ---------------------------------------------------------------------------


class TestBatchShape:
    def test_batch_system_documents_results_envelope(self):
        sys = topic_naming.BATCH_SYSTEM
        assert isinstance(sys, str) and sys.strip()
        # Wire format the structured-output schema enforces.
        assert '"results"' in sys
        assert '"cluster_id"' in sys
        assert '"topic_label"' in sys
        assert '"summary"' in sys

    def test_batch_user_template_two_placeholders(self):
        rendered = topic_naming.BATCH_USER_TEMPLATE.format(
            n=3, clusters="<placeholder cluster blocks>",
        )
        assert "3 research-paper" in rendered
        assert "<placeholder cluster blocks>" in rendered

    def test_cluster_block_template_three_placeholders(self):
        rendered = topic_naming.CLUSTER_BLOCK_TEMPLATE.format(
            cid=42,
            keywords="ml, dl",
            documents="1. Foo paper",
        )
        # Stub responder regex looks for ``cluster_id=(-?\d+)`` — pin it.
        assert "cluster_id=42" in rendered
        assert "ml, dl" in rendered
        assert "Foo paper" in rendered

    def test_cluster_block_marker_matches_stub_regex(self):
        rendered = topic_naming.CLUSTER_BLOCK_TEMPLATE.format(
            cid=-1, keywords="x", documents="y",
        )
        m = re.search(r"cluster_id=(-?\d+)", rendered)
        assert m is not None
        assert m.group(1) == "-1"

    def test_full_batch_render_round_trip(self):
        blocks = "\n".join(
            topic_naming.CLUSTER_BLOCK_TEMPLATE.format(
                cid=i, keywords=f"kw{i}", documents=f"1. doc{i}",
            )
            for i in (0, 1)
        )
        rendered = topic_naming.BATCH_USER_TEMPLATE.format(n=2, clusters=blocks)
        assert "cluster_id=0" in rendered
        assert "cluster_id=1" in rendered
        assert "kw0" in rendered and "kw1" in rendered
