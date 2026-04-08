"""Rate-limited, retrying HTTP layer for the S2 API.

Owns the httpx client, throttle, retry policy, and budget bookkeeping.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from citeclaw.config import BudgetTracker, Settings

log = logging.getLogger("citeclaw.s2.http")

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
    """Throttled HTTP wrapper around the S2 graph API."""

    def __init__(self, config: Settings, budget: BudgetTracker) -> None:
        self._config = config
        self._budget = budget
        self._min_interval = 1.0 / config.s2_rps
        self._last_request_time = 0.0

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

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=lambda rs: _retry_message(rs, "request"),
    )
    def get(self, path: str, params: dict[str, Any] | None = None, *, req_type: str = "other") -> dict[str, Any]:
        self._throttle()
        self._budget.record_s2(req_type)
        resp = self._http.get(f"{BASE_URL}{path}", params=params or {})
        resp.raise_for_status()
        self._clear_retry()
        return resp.json()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=lambda rs: _retry_message(rs, "request"),
    )
    def get_url(
        self, url: str, params: dict[str, Any] | None = None, *, req_type: str = "other",
    ) -> dict[str, Any]:
        """Like :meth:`get` but accepts a full URL instead of a ``/graph/v1``
        relative path. Used by endpoints outside the graph API (e.g.
        ``/recommendations/v1``)."""
        self._throttle()
        self._budget.record_s2(req_type)
        resp = self._http.get(url, params=params or {})
        resp.raise_for_status()
        self._clear_retry()
        return resp.json()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=lambda rs: _retry_message(rs, "batch"),
    )
    def post(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        *,
        req_type: str = "batch",
    ) -> Any:
        self._throttle()
        self._budget.record_s2(req_type)
        resp = self._http.post(url, params=params or {}, json=json_body)
        resp.raise_for_status()
        self._clear_retry()
        return resp.json()

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
