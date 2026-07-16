#!/usr/bin/env python3
"""Launch the CiteClaw live WebUI.

    python web/live/run.py

Starts a local server that serves the CiteClaw design and runs real
searches, then opens your browser. Local only — nothing is exposed to the
internet. Stop with Ctrl-C.

Environment:
    CITECLAW_WEBUI_PORT       port (default 8787)
    CITECLAW_WEBUI_HOST       host (default 127.0.0.1)
    CITECLAW_WEBUI_NO_BROWSER set to skip auto-opening the browser
    CITECLAW_WEBUI_ALLOW_STUB set to allow the free offline 'stub' model
"""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent           # web/live
REPO = HERE.parents[1]                            # repo root
# make `backend` package and the citeclaw source importable
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    import uvicorn

    from backend.server import app  # noqa: E402

    host = os.environ.get("CITECLAW_WEBUI_HOST", "127.0.0.1")
    port = int(os.environ.get("CITECLAW_WEBUI_PORT", "8787"))
    url = f"http://{host}:{port}"

    if not os.environ.get("CITECLAW_WEBUI_NO_BROWSER"):
        def _open():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    print(f"\n  CiteClaw live WebUI  →  {url}\n  (local only · Ctrl-C to stop)\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
