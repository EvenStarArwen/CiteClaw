"""Partition-agreement metrics — NMI and ARI, in pure Python.

The Explore screen's ⓘ "How this was computed" panel quotes how far a
community partition (citation space) drifts from the topic partition
(embedding space): normalized mutual information and the adjusted Rand
index. Those two numbers therefore have to be present in every run's
artifacts, including runs where the optional ``topic_model`` extras (and
with them scikit-learn) were never installed — so they are implemented
here rather than imported from ``sklearn.metrics``.

Both functions match scikit-learn's defaults on the design fixture
``web/design-reference/uploads/run37``:
``normalized_mutual_info`` uses the *arithmetic* mean of the two entropies
(``sklearn.metrics.normalized_mutual_info_score``'s ``average_method``
default) and ``adjusted_rand`` is the standard pair-counting formula
(``sklearn.metrics.adjusted_rand_score``). Pinned by
``tests/test_cluster_agreement.py``.

Noise handling is the caller's job: pass the label lists you want
compared. :mod:`citeclaw.output.groups` compares every paper both
partitions cover and treats the topic model's ``-1`` noise bucket as an
ordinary label — that is what the fixture's ``community_methods.csv``
did, and reproducing its numbers to four decimals was how this module
was validated. (Dropping the noise papers instead shifts leiden's NMI
0.4897 → 0.5026; the difference is real, so the convention is pinned
rather than left to taste.)
"""

from __future__ import annotations

from collections import Counter
from math import log


def _contingency(a: list[int], b: list[int]) -> dict[tuple[int, int], int]:
    """Joint counts of ``(a_label, b_label)`` pairs."""
    return Counter(zip(a, b))


def _entropy(labels: list[int]) -> float:
    """Shannon entropy (nats) of a label assignment."""
    n = len(labels)
    if n == 0:
        return 0.0
    return -sum((c / n) * log(c / n) for c in Counter(labels).values() if c > 0)


def mutual_info(a: list[int], b: list[int]) -> float:
    """Mutual information (nats) between two label assignments.

    Raises ``ValueError`` when the two lists differ in length — a
    mismatch means the caller aligned the partitions wrongly, which
    would otherwise surface as a quietly meaningless score.
    """
    if len(a) != len(b):
        raise ValueError(f"label lists differ in length: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    total = 0.0
    for (la, lb), nij in _contingency(a, b).items():
        if nij == 0:
            continue
        total += (nij / n) * log((nij * n) / (ca[la] * cb[lb]))
    return total


def normalized_mutual_info(a: list[int], b: list[int]) -> float:
    """NMI with arithmetic-mean normalisation, in ``[0, 1]``.

    Returns ``0.0`` when either side is degenerate (empty, or a single
    cluster covering everything) — there is no shared structure to
    measure and the normaliser would be zero. This mirrors
    scikit-learn's behaviour rather than raising, because a single-
    community run is a legitimate (if uninteresting) pipeline outcome.
    """
    if len(a) != len(b):
        raise ValueError(f"label lists differ in length: {len(a)} vs {len(b)}")
    if not a:
        return 0.0
    ha, hb = _entropy(a), _entropy(b)
    if ha <= 0.0 or hb <= 0.0:
        return 0.0
    nmi = mutual_info(a, b) / ((ha + hb) / 2.0)
    # Clamp: floating-point noise can push a perfect match to 1.0000000002.
    return max(0.0, min(1.0, nmi))


def adjusted_rand(a: list[int], b: list[int]) -> float:
    """Adjusted Rand index — chance-corrected, ``1.0`` for identical partitions.

    Can go slightly negative when agreement is worse than random, which
    is meaningful and is *not* clamped away. Returns ``1.0`` for the
    degenerate cases scikit-learn also calls perfect agreement (fewer
    than two items, or both partitions trivial in the same way).
    """
    if len(a) != len(b):
        raise ValueError(f"label lists differ in length: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return 1.0

    def _comb2(x: int) -> float:
        return x * (x - 1) / 2.0

    sum_ij = sum(_comb2(v) for v in _contingency(a, b).values())
    sum_a = sum(_comb2(v) for v in Counter(a).values())
    sum_b = sum(_comb2(v) for v in Counter(b).values())
    total = _comb2(n)
    expected = (sum_a * sum_b) / total
    maximum = (sum_a + sum_b) / 2.0
    if maximum == expected:
        # Both partitions are trivial (all-one-cluster or all-singletons)
        # in the same way — sklearn reports perfect agreement here.
        return 1.0
    return (sum_ij - expected) / (maximum - expected)
