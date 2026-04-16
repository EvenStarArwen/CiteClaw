"""Tests for citeclaw.filters.venue_presets (data + resolver)."""

from __future__ import annotations

import pytest

from citeclaw.filters.venue_presets import (
    CELL,
    NATURE,
    PREPRINT,
    SCIENCE,
    VENUE_PRESETS,
    resolve_presets,
)


class TestPresetContents:
    def test_known_presets_registered(self):
        assert set(VENUE_PRESETS) >= {"nature", "science", "cell", "preprint"}

    def test_flagships_in_family_lists(self):
        assert "Nature" in NATURE
        assert "Science" in SCIENCE
        assert "Cell" in CELL
        assert "arXiv" in PREPRINT

    def test_no_duplicates_within_preset(self):
        for name, lst in VENUE_PRESETS.items():
            assert len(lst) == len(set(lst)), f"{name!r} has duplicates"


class TestResolvePresets:
    def test_single_preset(self):
        out = resolve_presets(["science"])
        assert out == SCIENCE

    def test_multi_preset_dedups_preserving_order(self):
        out = resolve_presets(["nature", "nature", "science"])
        assert out.count("Nature") == 1
        assert "Science" in out
        assert out.index("Nature") < out.index("Science")

    def test_case_insensitive_preset_names(self):
        assert resolve_presets(["NATURE"]) == resolve_presets(["nature"])

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown venue preset"):
            resolve_presets(["not_a_real_preset"])
