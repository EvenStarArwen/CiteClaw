"""Tier 1 offline tests for :mod:`citeclaw.prompts.annotation`.

The annotation prompts are pure string constants and ``str.format``
templates — easy to silently break by renaming a placeholder. Tests
here verify the constants exist, the templates accept their documented
fields, and the structural markers downstream code relies on (e.g.
the stub responder's ``Paper index=`` regex) are present.

No live LLM is exercised; these are fast contract tests.
"""

from __future__ import annotations

import re

from citeclaw.prompts import annotation


# ---------------------------------------------------------------------------
# Single-paper shape (legacy)
# ---------------------------------------------------------------------------


class TestSingleShape:
    def test_system_is_non_empty_str(self):
        assert isinstance(annotation.SYSTEM, str)
        assert annotation.SYSTEM.strip()
        # Spec says reply with ONLY the label text — make sure that contract
        # stays in the system message.
        assert "label" in annotation.SYSTEM.lower()

    def test_user_template_has_three_placeholders(self):
        # The format string must accept {instruction}, {title}, {abstract}.
        rendered = annotation.USER_TEMPLATE.format(
            instruction="describe the topic",
            title="Sample Title",
            abstract="Sample abstract.",
        )
        assert "describe the topic" in rendered
        assert "Sample Title" in rendered
        assert "Sample abstract." in rendered
        # Trailing "Label:" cue stays so the model knows what to emit.
        assert rendered.rstrip().endswith("Label:")

    def test_user_template_round_trip_idempotent(self):
        rendered_a = annotation.USER_TEMPLATE.format(
            instruction="X", title="Y", abstract="Z",
        )
        rendered_b = annotation.USER_TEMPLATE.format(
            instruction="X", title="Y", abstract="Z",
        )
        assert rendered_a == rendered_b


# ---------------------------------------------------------------------------
# Batched shape (BATCH_SYSTEM + BATCH_USER_TEMPLATE + PAPER_BLOCK_TEMPLATE)
# ---------------------------------------------------------------------------


class TestBatchShape:
    def test_batch_system_documents_structured_json(self):
        sys_msg = annotation.BATCH_SYSTEM
        assert isinstance(sys_msg, str)
        assert sys_msg.strip()
        # Spec says output JSON of {"results": [{"index": ..., "label": ...}]}
        assert '"results"' in sys_msg
        assert '"index"' in sys_msg
        assert '"label"' in sys_msg

    def test_batch_user_template_has_three_placeholders(self):
        rendered = annotation.BATCH_USER_TEMPLATE.format(
            instruction="describe topic",
            n=2,
            papers="dummy block",
        )
        assert "describe topic" in rendered
        assert "2 papers" in rendered
        assert "dummy block" in rendered

    def test_paper_block_template_has_three_placeholders(self):
        rendered = annotation.PAPER_BLOCK_TEMPLATE.format(
            idx=7,
            title="Sample Title",
            abstract="Sample abstract.",
        )
        # Stub responder regex looks for "Paper index=<int>" — pin it.
        assert "Paper index=7" in rendered
        assert "Sample Title" in rendered
        assert "Sample abstract." in rendered

    def test_paper_block_marker_matches_stub_regex(self):
        """The stub responder uses ``Paper index=(-?\\d+)`` — ensure the
        marker still appears in a rendered block."""
        rendered = annotation.PAPER_BLOCK_TEMPLATE.format(
            idx=42, title="t", abstract="a",
        )
        m = re.search(r"Paper index=(-?\d+)", rendered)
        assert m is not None
        assert m.group(1) == "42"

    def test_full_batch_render_round_trip(self):
        """Render a 2-paper batch end-to-end and check structure."""
        blocks = "\n".join(
            annotation.PAPER_BLOCK_TEMPLATE.format(
                idx=i, title=f"T{i}", abstract=f"A{i}",
            )
            for i in (1, 2)
        )
        rendered = annotation.BATCH_USER_TEMPLATE.format(
            instruction="label everything",
            n=2,
            papers=blocks,
        )
        # Both indices appear, both titles, both abstracts.
        assert "Paper index=1" in rendered
        assert "Paper index=2" in rendered
        assert "T1" in rendered and "T2" in rendered
        assert "A1" in rendered and "A2" in rendered
