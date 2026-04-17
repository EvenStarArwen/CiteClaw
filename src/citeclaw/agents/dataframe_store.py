"""In-memory DataFrame storage for per-angle search results.

One ``DataFrameStore`` lives per :class:`~citeclaw.agents.state.WorkerState`
and holds the pandas DataFrames that ``fetch_results`` produces — one
DataFrame per angle (keyed by angle fingerprint's df_id derivative).
Lifetimes are strictly worker-local: when the worker calls ``done()``
the dispatcher clears the store, dropping every DataFrame the worker
built.

Why this module exists at all:

- The v2 agent design has inspection tools (``sample_titles``,
  ``year_distribution``, ``topic_model``, ``search_within_df``) that
  need random access into the full fetched result set, not just the
  paper id list the worker remembers. A pandas DataFrame keyed by
  paper_id with columns for metadata gives each tool cheap O(N)
  operations (sort, groupby, sample, filter) without reaching back
  to S2.

- Storing them out-of-band from ``WorkerState`` keeps the state
  dataclass lightweight (serialisable for postmortem logs) — the
  store has no serialisation contract; it is discarded at worker
  close.

- Cross-worker isolation: each worker gets a fresh store at dispatch
  time, so an angle's df_id from worker A can't leak into worker B.

The module is pure Python plus pandas; no citeclaw imports, so tests
can exercise it without spinning up a Context. Pandas is imported
locally at put-time so modules that don't use the store don't pay
the import cost.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger("citeclaw.agents.dataframe_store")


class DataFrameStoreError(Exception):
    """Raised when a store lookup hits a missing or expired df_id."""


class DataFrameStore:
    """Per-worker registry of named pandas DataFrames.

    Keys (``df_id``) are opaque strings the dispatcher generates from
    the worker id + angle fingerprint + turn counter. The store never
    generates ids itself — the caller is responsible for uniqueness.

    All methods are synchronous and non-thread-safe. The worker loop
    is sequential by design (one LLM call at a time, one tool
    dispatch at a time), so no locking is needed.
    """

    def __init__(self) -> None:
        self._store: dict[str, "pd.DataFrame"] = {}
        self._worker_id_index: dict[str, list[str]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, df_id: str) -> bool:
        return df_id in self._store

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def put(
        self,
        df_id: str,
        df: "pd.DataFrame",
        *,
        worker_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store ``df`` under ``df_id``.

        ``worker_id`` is optional — when provided, the store indexes
        this df_id under that worker so ``drop_all_for_worker`` can
        later sweep the worker's entire footprint in O(#owned) without
        a full scan. ``metadata`` is arbitrary and used by
        :meth:`metadata_for` callers that want to remember per-df
        notes (e.g. original query, n_fetched).

        Overwrites any prior df registered under the same df_id,
        matching the plain dict semantics of the v2 spec.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "DataFrameStore.put requires pandas. "
                "Install via the topic_model extras: pip install 'citeclaw[topic_model]'"
            ) from exc
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"DataFrameStore.put(df_id={df_id!r}): expected pd.DataFrame, "
                f"got {type(df).__name__}"
            )
        prior_owner = None
        if df_id in self._store and worker_id:
            # Remove the old back-index entry if the df is being
            # reassigned to a different worker (rare but defensible).
            for wid, ids in self._worker_id_index.items():
                if df_id in ids:
                    prior_owner = wid
                    break
        self._store[df_id] = df
        if worker_id:
            if prior_owner and prior_owner != worker_id:
                self._worker_id_index[prior_owner].remove(df_id)
            self._worker_id_index.setdefault(worker_id, [])
            if df_id not in self._worker_id_index[worker_id]:
                self._worker_id_index[worker_id].append(df_id)
        if metadata is not None:
            self._metadata[df_id] = dict(metadata)
        log.debug(
            "DataFrameStore put df_id=%s worker_id=%s rows=%d",
            df_id, worker_id or "-", len(df),
        )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get(self, df_id: str) -> "pd.DataFrame":
        """Return the DataFrame registered under ``df_id``.

        Raises :class:`DataFrameStoreError` on miss — the dispatcher
        translates this into the agent-visible
        ``{"error": "df_id not found", "hint": "..."}`` response so
        the LLM knows to call ``fetch_results`` first.
        """
        df = self._store.get(df_id)
        if df is None:
            raise DataFrameStoreError(
                f"df_id={df_id!r} not in store "
                f"(have {len(self._store)} df(s) registered)"
            )
        return df

    def metadata_for(self, df_id: str) -> dict[str, Any]:
        """Return stored metadata dict for ``df_id`` (empty dict if none)."""
        return dict(self._metadata.get(df_id, {}))

    def list_ids(self, *, worker_id: str | None = None) -> list[str]:
        """List df_ids, optionally scoped to one worker."""
        if worker_id is None:
            return list(self._store.keys())
        return list(self._worker_id_index.get(worker_id, []))

    # ------------------------------------------------------------------
    # Delete path
    # ------------------------------------------------------------------

    def drop(self, df_id: str) -> bool:
        """Remove ``df_id`` from the store. Returns True if it was present."""
        if df_id not in self._store:
            return False
        self._store.pop(df_id, None)
        self._metadata.pop(df_id, None)
        for wid, ids in list(self._worker_id_index.items()):
            if df_id in ids:
                ids.remove(df_id)
                if not ids:
                    self._worker_id_index.pop(wid, None)
        log.debug("DataFrameStore drop df_id=%s", df_id)
        return True

    def drop_all_for_worker(self, worker_id: str) -> int:
        """Remove every df owned by ``worker_id``. Returns count dropped.

        Called by the worker dispatcher at ``done()`` time to free
        the DataFrames the worker accumulated; they are never needed
        again once the worker's ``SubTopicResult`` has been handed
        back to the supervisor.
        """
        ids = self._worker_id_index.pop(worker_id, [])
        dropped = 0
        for df_id in list(ids):
            if df_id in self._store:
                self._store.pop(df_id, None)
                self._metadata.pop(df_id, None)
                dropped += 1
        if dropped:
            log.debug(
                "DataFrameStore drop_all_for_worker worker_id=%s dropped=%d",
                worker_id, dropped,
            )
        return dropped

    def clear(self) -> None:
        """Remove every registered df. Used for test isolation."""
        self._store.clear()
        self._metadata.clear()
        self._worker_id_index.clear()
