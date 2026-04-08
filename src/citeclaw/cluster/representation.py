"""Algorithm-agnostic cluster naming pipeline.

A clusterer's job (graph-based or embedding-based) is *only* to assign
papers to integer cluster ids. Everything human-readable about a cluster —
top keywords, label, summary, representative documents — is filled in here
by an algorithm-agnostic pipeline so the same code names a walktrap
community and a topic_model topic in exactly the same way.

Three pieces:

1. :func:`extract_keywords_ctfidf` — class-based TF-IDF (the BERTopic
   formula) over per-cluster super-documents. No LLM calls.
2. :func:`select_representative_papers` — pick the n papers per cluster
   closest to the cluster centroid in embedding space. Used as the
   "representative documents" input to the LLM naming prompt.
3. :func:`name_topics_via_llm` — concurrent LLM calls (one per cluster)
   that fill in ``ClusterMetadata.label`` and ``ClusterMetadata.summary``.
   Routes through the standard :class:`~citeclaw.clients.llm.base.LLMClient`
   protocol — no bespoke provider sniffing.
"""

from __future__ import annotations

import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from citeclaw.cluster.base import ClusterMetadata
from citeclaw.models import BudgetExhaustedError, PaperRecord
from citeclaw.prompts.topic_naming import SYSTEM as _NAMING_SYSTEM
from citeclaw.prompts.topic_naming import USER_TEMPLATE as _NAMING_USER

if TYPE_CHECKING:
    from citeclaw.clients.llm.base import LLMClient

log = logging.getLogger("citeclaw.cluster.representation")


# ---------------------------------------------------------------------------
# c-TF-IDF — class-based TF-IDF (BERTopic formula)
# ---------------------------------------------------------------------------


def _cluster_documents(
    membership: dict[str, int],
    papers: dict[str, PaperRecord],
) -> dict[int, str]:
    """Build per-cluster super-documents (one big string per cluster).

    Excludes the noise cluster (-1). Each super-document is the concatenation
    of every paper's title and abstract in that cluster, separated by spaces.
    """
    by_cluster: dict[int, list[str]] = {}
    for pid, cid in membership.items():
        if cid == -1:
            continue
        rec = papers.get(pid)
        if rec is None:
            continue
        text_parts = []
        if rec.title:
            text_parts.append(rec.title)
        if rec.abstract:
            text_parts.append(rec.abstract)
        if text_parts:
            by_cluster.setdefault(cid, []).append(" ".join(text_parts))
    return {cid: " ".join(docs) for cid, docs in by_cluster.items()}


def extract_keywords_ctfidf(
    membership: dict[str, int],
    papers: dict[str, PaperRecord],
    *,
    n_keywords: int = 10,
    stop_words: str = "english",
) -> dict[int, list[str]]:
    """Compute the top N c-TF-IDF keywords per cluster.

    Concatenates each cluster's papers' (title + abstract) into one big
    super-document, runs sklearn's CountVectorizer to get the term-frequency
    matrix, then applies the BERTopic c-TF-IDF formula:

        tf_norm[c, t] = freq[c, t] / sum_t freq[c, t]
        idf[t]        = log(1 + avg_freq_per_class / freq_across_all_classes[t])
        ctfidf[c, t]  = tf_norm[c, t] * idf[t]

    Returns ``{cluster_id: [keyword1, keyword2, ...]}`` for every non-noise
    cluster. Empty clusters and clusters with no usable text are silently
    skipped (their entry just doesn't appear).
    """
    cluster_docs = _cluster_documents(membership, papers)
    if not cluster_docs:
        return {}

    try:
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise RuntimeError(
            "extract_keywords_ctfidf requires scikit-learn. "
            "Install via the topic_model extras: pip install 'citeclaw[topic_model]'"
        ) from exc

    cluster_ids = sorted(cluster_docs.keys())
    docs = [cluster_docs[cid] for cid in cluster_ids]

    vectorizer = CountVectorizer(stop_words=stop_words)
    try:
        tf = vectorizer.fit_transform(docs)  # shape (n_clusters, vocab)
    except ValueError:
        # Empty vocabulary (e.g. all stop-words). Nothing to extract.
        log.warning("c-TF-IDF: empty vocabulary after stop-word removal")
        return {cid: [] for cid in cluster_ids}

    vocab = vectorizer.get_feature_names_out()
    tf_dense = tf.toarray()  # (n_clusters, n_terms)

    n_clusters, n_terms = tf_dense.shape

    # tf_norm[c, t] = freq[c, t] / sum_t freq[c, t]
    row_sums = tf_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid /0
    tf_norm = tf_dense / row_sums

    # idf[t] = log(1 + avg_freq_per_class / freq_across_all_classes[t])
    # avg_freq_per_class is the average length of a class super-document.
    avg_per_class = float(tf_dense.sum() / n_clusters) if n_clusters else 0.0
    freq_per_term = tf_dense.sum(axis=0)  # (n_terms,)
    # freq_per_term may have zeros if a CountVectorizer column is empty after
    # vocab pruning; clamp to avoid /0.
    freq_per_term_safe = freq_per_term.copy()
    freq_per_term_safe[freq_per_term_safe == 0] = 1.0
    idf = [math.log(1.0 + avg_per_class / float(f)) for f in freq_per_term_safe]

    out: dict[int, list[str]] = {}
    for row_idx, cid in enumerate(cluster_ids):
        scores = [tf_norm[row_idx, t] * idf[t] for t in range(n_terms)]
        # argsort descending; sklearn arrays support fancy indexing but we
        # stay in pure Python so this works without numpy on the path.
        ranked = sorted(range(n_terms), key=lambda t: scores[t], reverse=True)
        top = [str(vocab[t]) for t in ranked[:n_keywords] if scores[t] > 0]
        out[cid] = top
    return out


# ---------------------------------------------------------------------------
# Representative papers — pick centroid-closest n per cluster
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def select_representative_papers(
    membership: dict[str, int],
    embeddings: dict[str, list[float] | None],
    *,
    n: int = 5,
) -> dict[int, list[str]]:
    """Pick the n papers per cluster closest to that cluster's centroid.

    Centroids are computed in raw embedding space (no UMAP reduction — we
    want the most semantically representative papers, not the most visually
    central in the reduced view). Returns ``{cluster_id: [paper_id, ...]}``
    sorted by descending cosine similarity to the centroid.
    """
    by_cluster: dict[int, list[tuple[str, list[float]]]] = {}
    for pid, cid in membership.items():
        if cid == -1:
            continue
        v = embeddings.get(pid)
        if not v:
            continue
        by_cluster.setdefault(cid, []).append((pid, v))

    out: dict[int, list[str]] = {}
    for cid, items in by_cluster.items():
        if not items:
            continue
        dim = len(items[0][1])
        # Compute centroid as mean of vectors.
        centroid = [0.0] * dim
        for _, v in items:
            for i, x in enumerate(v):
                centroid[i] += x
        for i in range(dim):
            centroid[i] /= len(items)
        # Sort by descending cosine to centroid.
        scored = sorted(
            items,
            key=lambda pv: _cosine(pv[1], centroid),
            reverse=True,
        )
        out[cid] = [pid for pid, _ in scored[:n]]
    return out


# ---------------------------------------------------------------------------
# LLM-based naming — concurrent dispatch through LLMClient
# ---------------------------------------------------------------------------


def _parse_naming_response(text: str) -> tuple[str, str] | None:
    """Parse a topic-naming LLM response into (label, summary).

    Tolerates code-fenced JSON and slightly malformed responses. Returns
    None on parse failure so the caller can fall back to keyword-only naming.
    """
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    label = str(data.get("topic_label") or data.get("label") or "").strip()
    summary = str(data.get("summary") or "").strip()
    if not label and not summary:
        return None
    return label, summary


def _format_documents(papers: dict[str, PaperRecord], paper_ids: list[str]) -> str:
    """Format representative papers as bullet points for the LLM prompt."""
    lines: list[str] = []
    for pid in paper_ids:
        rec = papers.get(pid)
        if rec is None:
            continue
        title = (rec.title or "").strip() or "(untitled)"
        lines.append(f"- {title}")
    return "\n".join(lines) if lines else "(no representative documents)"


def name_topics_via_llm(
    metadata: dict[int, ClusterMetadata],
    papers: dict[str, PaperRecord],
    *,
    client: "LLMClient",
    max_workers: int = 4,
    category: str = "llm_topic_naming",
) -> None:
    """Mutate ``metadata`` in place: fill ``label`` and ``summary`` per cluster.

    For each cluster with at least one keyword (or representative paper),
    issues one LLM call via the unified ``LLMClient`` protocol with the
    topic-naming prompt and parses the JSON response. Failed calls log a
    warning and fall back to ``label = " ".join(keywords[:5])``.

    Concurrency: a small ThreadPoolExecutor (default ``max_workers=4``)
    fans out cluster naming since clusters are independent. Budget
    exhaustion stops the loop early — already-named clusters keep their
    labels.

    Each cluster only issues a call if it has at least one keyword OR at
    least one representative paper. Empty clusters are silently skipped.
    """
    if not metadata:
        return

    work: list[int] = []
    for cid, m in metadata.items():
        if cid == -1:
            continue
        if not m.keywords and not m.representative_papers:
            continue
        work.append(cid)
    if not work:
        return

    def _name_one(cid: int) -> tuple[int, str | None, str | None]:
        m = metadata[cid]
        keywords_str = ", ".join(m.keywords) if m.keywords else "(no keywords)"
        documents_str = _format_documents(papers, m.representative_papers)
        user = _NAMING_USER.format(keywords=keywords_str, documents=documents_str)
        try:
            resp = client.call(_NAMING_SYSTEM, user, category=category)
        except BudgetExhaustedError:
            return cid, None, None
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("topic naming LLM call failed for cluster %d: %s", cid, exc)
            return cid, None, None
        parsed = _parse_naming_response(resp.text or "")
        if parsed is None:
            return cid, None, None
        label, summary = parsed
        return cid, label, summary

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(_name_one, cid): cid for cid in work}
        for fut in as_completed(futures):
            cid, label, summary = fut.result()
            m = metadata[cid]
            if label is None and summary is None:
                # Fallback: build a label from the top keywords.
                if m.keywords:
                    m.label = " ".join(m.keywords[:5])
                continue
            if label:
                m.label = label
            if summary:
                m.summary = summary


__all__ = [
    "extract_keywords_ctfidf",
    "select_representative_papers",
    "name_topics_via_llm",
]
