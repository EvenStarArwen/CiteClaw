"""Thin OpenAlex client — abstract + references fallback for S2 gaps.

OpenAlex covers ~250M works including most arXiv preprints and carries
abstracts + structured ``referenced_works`` for many records S2 hasn't
fully processed yet. Used as a graceful fallback from the main S2 path:

* :meth:`OpenAlexClient.fetch_abstract_by_doi` — final fallback in
  :meth:`citeclaw.clients.s2.SemanticScholarClient.enrich_with_abstracts`
  when S2 returns no abstract. Reassembles OpenAlex's ``inverted_index``
  format back to plain text.
* :meth:`OpenAlexClient.fetch_references_by_doi` — cited-by list, used
  by ``ExpandBackward``'s arXiv fallback when S2 has no references.

Polite-pool access is free and unauthenticated; ``openalex_email`` in
:class:`citeclaw.config.Settings` enables polite-pool routing, and
``openalex_api_key`` unlocks the higher rate-limit pool. The per-second
cap comes from ``openalex_rps`` (default 5). Failures log but never
raise — this is a best-effort enrichment path and must not crash the
main pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from citeclaw.config import Settings

log = logging.getLogger("citeclaw.openalex")

_BASE_URL = "https://api.openalex.org"

# Prefixes a DOI may carry on the wire. Stripped (case-insensitively, in
# order) by :func:`_strip_doi_prefixes` to coerce any of these shapes to
# the bare ``10.xxxx/yyyy`` form that S2's DOI lookup expects.
_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:")


def _strip_doi_prefixes(s: str) -> str:
    """Return ``s`` with any leading DOI URL / ``doi:`` prefix removed.

    Returns the input unchanged when no prefix matches. Lowercase
    comparison so ``HTTPS://DOI.ORG/...`` is handled the same as the
    canonical form.
    """
    lower = s.lower()
    for prefix in _DOI_PREFIXES:
        if lower.startswith(prefix):
            return s[len(prefix):]
    return s


def _is_retryable(exc: BaseException) -> bool:
    """Match the S2 client's retry policy: transport errors + 429 + 5xx."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else 0
        if status == 429:
            return True
        if 500 <= status < 600:
            return True
    return False


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Rebuild plain text from OpenAlex's ``abstract_inverted_index``.

    The API doesn't return abstracts verbatim (licensing); it returns a
    word → [positions] mapping. We invert that mapping back to text.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    by_position: dict[int, str] = {}
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and pos >= 0:
                by_position[pos] = word
    if not by_position:
        return None
    max_pos = max(by_position)
    return " ".join(by_position.get(i, "") for i in range(max_pos + 1)).strip() or None


class OpenAlexClient:
    """Throttled HTTP wrapper around the OpenAlex works API.

    Rate-limiting is a simple request-spacing sleep to honour
    ``openalex_rps``. The retry policy mirrors the S2 client's
    (``retry_if_exception(_is_retryable)``) so transient failures don't
    escape to the caller.
    """

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._min_interval = 1.0 / max(0.1, config.openalex_rps)
        self._last_request_time = 0.0
        headers: dict[str, str] = {"Accept": "application/json"}
        # Polite-pool etiquette: email goes in the User-Agent so the
        # OpenAlex operators can reach out if our traffic causes issues.
        if config.openalex_email:
            headers["User-Agent"] = f"CiteClaw/1 (mailto:{config.openalex_email})"
        if config.openalex_api_key:
            # OpenAlex accepts the API key via an ``api_key`` query param
            # OR via the ``Authorization: Bearer`` header; header is
            # cleaner and avoids polluting cache keys downstream.
            headers["Authorization"] = f"Bearer {config.openalex_api_key}"
        self._http = httpx.Client(timeout=30, headers=headers)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(4),
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        self._throttle()
        resp = self._http.get(f"{_BASE_URL}{path}", params=params or {})
        # 404 is "not in OpenAlex" — clean ``None``, not an error.
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def fetch_work_by_doi(self, doi: str) -> dict[str, Any] | None:
        """Fetch the raw OpenAlex Work JSON for a DOI, or ``None`` on miss.

        OpenAlex's DOI endpoint accepts bare DOIs (``10.xxxx/yyy``) under
        the ``/works/doi:...`` path; we URL-encode the suffix via httpx's
        params handling. Failures of any kind return ``None`` so callers
        can treat this as a best-effort enrichment.
        """
        if not doi:
            return None
        clean = _strip_doi_prefixes(doi.strip())
        try:
            return self._get(f"/works/doi:{clean}")
        except (httpx.HTTPError, RetryError) as exc:
            log.info("OpenAlex fetch_work_by_doi(%s) failed: %s", clean[:60], exc)
            return None

    def fetch_abstract_by_doi(self, doi: str) -> str | None:
        """Return the abstract for a DOI via OpenAlex, or ``None`` on miss.

        Fallback for :meth:`SemanticScholarClient.enrich_with_abstracts`
        — OpenAlex frequently has abstracts S2 doesn't (older papers,
        fresh preprints, niche venues).
        """
        work = self.fetch_work_by_doi(doi)
        if not work:
            return None
        return _reconstruct_abstract(work.get("abstract_inverted_index"))

    def fetch_references_by_doi(self, doi: str) -> list[str]:
        """Return the list of referenced-work DOIs for a DOI.

        OpenAlex stores references as a list of OpenAlex Work IDs
        (``referenced_works``); we resolve each back to a DOI via a
        second call to ``/works/<id>`` only when we can't read the DOI
        off the initial Work payload directly. Typically the caller just
        needs DOIs to re-enter the S2 resolution path.
        """
        work = self.fetch_work_by_doi(doi)
        if not work:
            return []
        # OpenAlex stores ``referenced_works`` as a list of full OpenAlex
        # URLs like ``https://openalex.org/W1234567``. We need DOIs to
        # re-enter S2 via ``DOI:<doi>``, so resolve each.
        oa_ids = work.get("referenced_works") or []
        if not isinstance(oa_ids, list):
            return []
        dois: list[str] = []
        for oa_id in oa_ids:
            if not isinstance(oa_id, str):
                continue
            # Strip the URL prefix → bare W-id.
            short = oa_id.rsplit("/", 1)[-1] if "/" in oa_id else oa_id
            if not short.startswith("W"):
                continue
            try:
                ref_work = self._get(f"/works/{short}")
            except (httpx.HTTPError, RetryError) as exc:
                log.debug("OpenAlex ref lookup %s failed: %s", short, exc)
                continue
            if not ref_work:
                continue
            ref_doi = ref_work.get("doi")
            if isinstance(ref_doi, str):
                # OpenAlex DOIs come back as ``https://doi.org/10.xxx/yyy`` —
                # the helper coerces them to the bare ``10.xxx/yyy`` form S2
                # consumes.
                bare = _strip_doi_prefixes(ref_doi)
                if bare.startswith("10."):
                    dois.append(bare)
        return dois

    def close(self) -> None:
        self._http.close()
