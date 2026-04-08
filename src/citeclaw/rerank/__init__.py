"""Rerank package — metric scoring + cluster-aware diversity selection.

Clustering algorithms themselves live in :mod:`citeclaw.cluster`. This package
holds only the score-based ranking metrics and the stratified top-K
allocator that consumes a :class:`~citeclaw.cluster.base.ClusterResult`.
"""
