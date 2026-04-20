"""Rate-limited, retrying HTTP layer for the S2 API.

Owns the httpx client, throttle, retry policy, and budget bookkeeping.
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

from citeclaw.budget import BudgetTracker

from citeclaw.config import Settings
from citeclaw.models import S2OutageError

log = logging.getLogger("citeclaw.s2.http")


def _is_retryable(exc: BaseException) -> bool:
    """Decide whether an exception is worth a retry.

    - Transport errors (connection reset, DNS, timeout) → retry.
    - HTTP 5xx → retry (transient server problem).
    - HTTP 429 → retry (rate limited; backoff handles the wait).
    - HTTP 4xx other than 429 → DO NOT retry. These are permanent
      client errors (bad sort key, bad params, missing auth) and
      retrying wastes 6 requests against the 1-rps budget while the
      caller waits for tenacity's exponential backoff to give up.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else 0
        if status == 429:
            return True
        if 500 <= status < 600:
            return True
        return False
    return False

BASE_URL = "https://api.semanticscholar.org/graph/v1"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
PAGE_SIZE = 100


def _retry_message(rs, kind: str) -> None:
    """Tenacity ``before_sleep`` callback that surfaces the retry to the
    active dashboard (if any) and falls back to the warning log.

    The dashboard receives a transient one-line banner; the file log
    keeps the legacy warning so postmortem debugging still has it.
    """
    # Compute next backoff so the user sees how long we'll wait.
    sleep = 0.0
    try:
        sleep = float(rs.next_action.sleep) if rs.next_action else 0.0
    except Exception:
        pass
    exc = ""
    try:
        e = rs.outcome.exception() if rs.outcome else None
        if e is not None:
            exc = f"{type(e).__name__}"
    except Exception:
        pass
    msg = (
        f"S2 {kind} retry {rs.attempt_number}/6"
        + (f" · {exc}" if exc else "")
        + (f" · backoff {sleep:.1f}s" if sleep else "")
    )
    # Always keep the warning so file logs retain it.
    log.warning(msg)
    try:
        from citeclaw.progress import get_active_dashboard
        dash = get_active_dashboard()
        if dash is not None:
            dash.set_retry_status(msg)
    except Exception:
        pass


class S2Http:
    """Throttled HTTP wrapper around the S2 graph API.

    Tracks consecutive retry-exhausted failures across every method
    (``get`` / ``get_url`` / ``post``); once they exceed
    ``Settings.s2_max_consecutive_failures``, the next failure is
    converted into an :class:`S2OutageError` so the pipeline runner can
    bail out cleanly instead of grinding through the full retry budget
    on every paper. Any successful call resets the counter.
    """

    def __init__(self, config: Settings, budget: BudgetTracker) -> None:
        self._config = config
        self._budget = budget
        self._min_interval = 1.0 / config.s2_rps
        self._last_request_time = 0.0
        self._consecutive_failures = 0
        self._max_consecutive_failures = max(0, config.s2_max_consecutive_failures)

        headers: dict[str, str] = {"Accept": "application/json"}
        if config.s2_api_key:
            headers["x-api-key"] = config.s2_api_key
        self._http = httpx.Client(timeout=60, headers=headers)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _clear_retry(self) -> None:
        """Clear the dashboard retry banner once the call succeeds."""
        try:
            from citeclaw.progress import get_active_dashboard
            dash = get_active_dashboard()
            if dash is not None:
                dash.clear_retry_status()
        except Exception:
            pass

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _record_failure(self, exc: BaseException) -> None:
        """Bump the consecutive-failure counter; raise outage if exceeded.

        Called from the outer wrapper when the inner retry-decorated
        helper has exhausted all attempts. Tenacity raises ``RetryError``
        when it gives up; transport errors that escape the retry guard
        (rare; would mean the predicate said "don't retry") count too.
        """
        self._consecutive_failures += 1
        if (
            self._max_consecutive_failures
            and self._consecutive_failures >= self._max_consecutive_failures
        ):
            raise S2OutageError(
                f"S2 API hit max retries on {self._consecutive_failures} "
                f"consecutive calls (limit={self._max_consecutive_failures}). "
                f"Last error: {type(exc).__name__}: {str(exc)[:200]}"
            ) from exc

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=lambda rs: _retry_message(rs, "request"),
    )
    def _get_with_retries(
        self, path: str, params: dict[str, Any] | None, req_type: str,
    ) -> dict[str, Any]:
        self._throttle()
        self._budget.record_s2(req_type)
        resp = self._http.get(f"{BASE_URL}{path}", params=params or {})
        resp.raise_for_status()
        self._clear_retry()
        return resp.json()

    def get(self, path: str, params: dict[str, Any] | None = None, *, req_type: str = "other") -> dict[str, Any]:
        try:
            result = self._get_with_retries(path, params, req_type)
        except (RetryError, httpx.HTTPError) as exc:
            self._record_failure(exc)
            raise
        self._record_success()
        return result

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=lambda rs: _retry_message(rs, "request"),
    )
    def _get_url_with_retries(
        self, url: str, params: dict[str, Any] | None, req_type: str,
    ) -> dict[str, Any]:
        self._throttle()
        self._budget.record_s2(req_type)
        resp = self._http.get(url, params=params or {})
        resp.raise_for_status()
        self._clear_retry()
        return resp.json()

    def get_url(
        self, url: str, params: dict[str, Any] | None = None, *, req_type: str = "other",
    ) -> dict[str, Any]:
        """Like :meth:`get` but accepts a full URL instead of a ``/graph/v1``
        relative path. Used by endpoints outside the graph API (e.g.
        ``/recommendations/v1``)."""
        try:
            result = self._get_url_with_retries(url, params, req_type)
        except (RetryError, httpx.HTTPError) as exc:
            self._record_failure(exc)
            raise
        self._record_success()
        return result

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=lambda rs: _retry_message(rs, "batch"),
    )
    def _post_with_retries(
        self, url: str, params: dict[str, Any] | None, json_body: Any, req_type: str,
    ) -> Any:
        self._throttle()
        self._budget.record_s2(req_type)
        resp = self._http.post(url, params=params or {}, json=json_body)
        resp.raise_for_status()
        self._clear_retry()
        return resp.json()

    def post(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        *,
        req_type: str = "batch",
    ) -> Any:
        try:
            result = self._post_with_retries(url, params, json_body, req_type)
        except (RetryError, httpx.HTTPError) as exc:
            self._record_failure(exc)
            raise
        self._record_success()
        return result

    def paginate(
        self,
        paper_id: str,
        edge: str,
        *,
        fields: str,
        max_items: int | None = None,
    ) -> list[dict[str, Any]]:
        """Paginate through ``/paper/{id}/references|citations``."""
        req_type = "references" if edge == "references" else "citations"
        results: list[dict[str, Any]] = []
        offset = 0
        while True:
            data = self.get(
                f"/paper/{paper_id}/{edge}",
                params={"fields": fields, "limit": PAGE_SIZE, "offset": offset},
                req_type=req_type,
            )
            batch = data.get("data", [])
            if not batch:
                break
            results.extend(batch)
            offset += len(batch)
            if max_items is not None and len(results) >= max_items:
                return results[:max_items]
            if len(batch) < PAGE_SIZE:
                break
        return results

    def close(self) -> None:
        self._http.close()
