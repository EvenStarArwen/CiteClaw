"""Tests for the pure-Python NMI / ARI implementations.

These exist so ``community_methods.csv`` carries agreement numbers on
installs without scikit-learn. The correctness bar is therefore "matches
scikit-learn", so where sklearn happens to be installed we compare
against it directly; the analytic edge cases run everywhere.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from citeclaw.cluster.agreement import (
    adjusted_rand,
    mutual_info,
    normalized_mutual_info,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CSV = (
    REPO_ROOT / "web" / "design-reference" / "uploads" / "run37" / "accepted.csv"
)

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class TestAnalyticCases:
    def test_identical_partitions(self):
        a = [0, 0, 1, 1, 2, 2]
        assert normalized_mutual_info(a, a) == pytest.approx(1.0)
        assert adjusted_rand(a, a) == pytest.approx(1.0)

    def test_relabelled_partition_is_still_identical(self):
        a = [0, 0, 1, 1, 2, 2]
        b = [5, 5, 9, 9, 3, 3]
        assert normalized_mutual_info(a, b) == pytest.approx(1.0)
        assert adjusted_rand(a, b) == pytest.approx(1.0)

    def test_one_side_trivial_scores_zero_nmi(self):
        a = [0, 0, 0, 0]
        b = [0, 1, 2, 3]
        assert normalized_mutual_info(a, b) == 0.0
        assert mutual_info(a, b) == pytest.approx(0.0)

    def test_both_sides_trivial_is_perfect_ari(self):
        assert adjusted_rand([0, 0, 0], [1, 1, 1]) == pytest.approx(1.0)

    def test_ari_can_go_negative_and_is_not_clamped(self):
        # Deliberately anti-correlated blocking.
        a = [0, 0, 1, 1]
        b = [0, 1, 0, 1]
        assert adjusted_rand(a, b) < 0.0

    def test_nmi_is_clamped_into_unit_interval(self):
        a = list(range(50))
        assert 0.0 <= normalized_mutual_info(a, a) <= 1.0

    def test_length_mismatch_raises(self):
        for fn in (mutual_info, normalized_mutual_info, adjusted_rand):
            with pytest.raises(ValueError, match="differ in length"):
                fn([0, 1], [0, 1, 2])

    def test_empty_and_singleton_inputs(self):
        assert normalized_mutual_info([], []) == 0.0
        assert mutual_info([], []) == 0.0
        assert adjusted_rand([], []) == 1.0
        assert adjusted_rand([3], [7]) == 1.0

    def test_symmetric(self):
        a = [0, 0, 1, 1, 2, 0, 1]
        b = [1, 0, 1, 2, 2, 0, 1]
        assert normalized_mutual_info(a, b) == pytest.approx(
            normalized_mutual_info(b, a),
        )
        assert adjusted_rand(a, b) == pytest.approx(adjusted_rand(b, a))


@pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
class TestMatchesSklearn:
    @pytest.mark.parametrize(
        "a, b",
        [
            ([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2]),
            ([-1, -1, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 2, 2]),
            ([0] * 10 + [1] * 5, [0] * 7 + [1] * 8),
            (list(range(20)), [i // 4 for i in range(20)]),
        ],
    )
    def test_parity(self, a, b):
        assert normalized_mutual_info(a, b) == pytest.approx(
            normalized_mutual_info_score(a, b), abs=1e-9,
        )
        assert adjusted_rand(a, b) == pytest.approx(
            adjusted_rand_score(a, b), abs=1e-9,
        )


@pytest.mark.skipif(
    not FIXTURE_CSV.exists(),
    reason="design-reference fixture not present in this checkout",
)
class TestRun37Fixture:
    """Reproduce the numbers shipped in run37's ``community_methods.csv``.

    The convention this pins down: agreement is computed over *all* 354
    papers with the topic model's ``-1`` noise bucket treated as an
    ordinary label. Dropping the noise papers instead moves leiden's NMI
    to 0.5026, so the choice is load-bearing, not cosmetic.
    """

    EXPECTED = {
        "louvain":           (0.4916, 0.2471),
        "leiden":            (0.4897, 0.2562),
        "walktrap":          (0.5731, 0.1805),
        "infomap":           (0.6060, 0.2713),
        "label_propagation": (0.4043, 0.0887),
    }

    @pytest.mark.parametrize("method", sorted(EXPECTED))
    def test_matches_shipped_metrics(self, method):
        with FIXTURE_CSV.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        topics = [int(r["topic"]) for r in rows]
        communities = [int(r[f"community_{method}"]) for r in rows]
        exp_nmi, exp_ari = self.EXPECTED[method]
        assert normalized_mutual_info(communities, topics) == pytest.approx(
            exp_nmi, abs=5e-5,
        )
        assert adjusted_rand(communities, topics) == pytest.approx(exp_ari, abs=5e-5)
