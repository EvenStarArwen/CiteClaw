"""orjson when available (ingest/serve containers), stdlib json fallback (tests)."""

from __future__ import annotations

try:
    import orjson as _orjson

    def loads(b):  # type: ignore[no-redef]
        return _orjson.loads(b)

    def dumps(obj) -> bytes:  # type: ignore[no-redef]
        return _orjson.dumps(obj)

except ImportError:  # pragma: no cover - exercised only where orjson is absent
    import json as _json

    def loads(b):
        if isinstance(b, (bytes, bytearray)):
            b = b.decode("utf-8")
        return _json.loads(b)

    def dumps(obj) -> bytes:
        return _json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
