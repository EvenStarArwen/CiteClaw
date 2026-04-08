"""Similarity measures registry."""

from __future__ import annotations

from citeclaw.filters.measures.base import SimilarityMeasure
from citeclaw.filters.measures.cit_sim import CitSimMeasure
from citeclaw.filters.measures.ref_sim import RefSimMeasure
from citeclaw.filters.measures.semantic_sim import SemanticSimMeasure

MEASURE_TYPES = {
    "RefSim":      RefSimMeasure,
    "CitSim":      CitSimMeasure,
    "SemanticSim": SemanticSimMeasure,
}

__all__ = [
    "SimilarityMeasure",
    "RefSimMeasure",
    "CitSimMeasure",
    "SemanticSimMeasure",
    "MEASURE_TYPES",
]
