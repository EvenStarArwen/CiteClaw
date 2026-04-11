"""Standalone Modal deployment: GROBID REST server for CiteClaw.

Sister to ``modal_vllm_server.py`` — an isolated, single-file Modal app
that runs a GROBID server (``lfoppiano/grobid:0.8.2-crf``) as a
web-tunneled REST service. Only depends on ``modal``; nothing in the
main CiteClaw package imports from this file.

GROBID is a Java/ML tool purpose-built for extracting structured data
from scientific PDFs: body text, sections, and — most importantly —
structured reference lists with cleanly parsed titles / authors / DOIs.
The **CRF-only** image is ~500 MB, runs on CPU, and delivers ~10 PDF/s
on a modern CPU with a bibliography-resolution F1 around 0.87.

Why the CRF image and not the ``0.8.2-full`` DL image?
- Our use case is ``/api/processFulltextDocument`` for body + refs.
- CRF is within a few F1 points of the DL model for reference parsing.
- No GPU needed → order-of-magnitude cheaper on Modal.
- ~500 MB vs. ~8 GB image — cold starts finish in seconds, not minutes.

============================================================================
Quickstart
============================================================================

One-time setup::

    pip install modal
    modal setup

Deploy::

    CITECLAW_GROBID_APP_NAME=citeclaw-grobid \
    modal deploy modal_grobid_server.py

Modal prints the public URL, e.g.::

    https://<you>--citeclaw-grobid-serve.modal.run

Point CiteClaw at it via the environment::

    export CITECLAW_GROBID_URL="https://<you>--citeclaw-grobid-serve.modal.run"

Stop::

    modal app stop citeclaw-grobid
"""

from __future__ import annotations

import os

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Image tag — ``lfoppiano/grobid:<version>-crf`` is a maintained, CPU-only,
# ~500 MB image. The full image (no ``-crf`` suffix) is ~8 GB and needs
# a GPU to be fast; unnecessary for our body-text + reference-extraction use.
GROBID_IMAGE_REF: str = os.environ.get(
    "CITECLAW_GROBID_IMAGE",
    "lfoppiano/grobid:0.8.2-crf",
)
APP_NAME: str = os.environ.get("CITECLAW_GROBID_APP_NAME", "citeclaw-grobid")
CPU_COUNT: float = float(os.environ.get("CITECLAW_GROBID_CPU", "8"))
MEMORY_MB: int = int(os.environ.get("CITECLAW_GROBID_MEMORY_MB", "8192"))
SCALEDOWN: int = int(os.environ.get("CITECLAW_GROBID_SCALEDOWN", "300"))
MAX_CONCURRENCY: int = int(os.environ.get("CITECLAW_GROBID_MAX_CONCURRENCY", "32"))
PORT = 8070

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

# Pull the pre-built GROBID image directly from Docker Hub. The image
# already has Java + GROBID installed and ships with an ENTRYPOINT that
# launches the server on port 8070, so we don't need to re-implement
# the startup command — just keep the container alive long enough for
# the web tunnel to pick up the port.
image = modal.Image.from_registry(
    GROBID_IMAGE_REF,
    # The GROBID image sets ``ENTRYPOINT=["./grobid-service/bin/grobid-service"]``.
    # Modal's web_server mechanism ignores ENTRYPOINT and runs the Python
    # function directly — we launch the server ourselves from Python below,
    # so the entrypoint doesn't need to be neutered.
    add_python="3.12",
)


@app.function(
    image=image,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    scaledown_window=SCALEDOWN,
    max_containers=1,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=MAX_CONCURRENCY)
@modal.web_server(port=PORT, startup_timeout=60 * 5)
def serve() -> None:
    """Launch the GROBID Java server.

    The lfoppiano/grobid image installs to ``/opt/grobid``. The upstream
    CMD is::

        ./grobid-service/bin/grobid-service server grobid-service/config/config.yaml

    run with ``WORKDIR=/opt/grobid`` and relative paths. We replicate
    that here, with a filesystem probe to locate the config under
    alternate names (recent GROBID versions have renamed ``config.yaml``
    to ``grobid.yaml`` in some distributions).
    """
    import glob
    import subprocess

    grobid_home = "/opt/grobid"
    bin_candidates = [
        f"{grobid_home}/grobid-service/bin/grobid-service",
        "grobid-service",
    ]
    entry: str | None = None
    for c in bin_candidates:
        if c.startswith("/") and os.path.exists(c):
            entry = c
            break
        if not c.startswith("/"):
            import shutil
            which = shutil.which(c)
            if which:
                entry = which
                break
    if entry is None:
        raise RuntimeError(
            "Could not find grobid-service entrypoint. Expected one of: "
            + ", ".join(bin_candidates)
        )

    # Probe for the Dropwizard config file. GROBID 0.8.x ships the
    # Dropwizard config as ``/opt/grobid/grobid-home/config/grobid.yaml``;
    # older 0.7.x releases kept it under ``grobid-service/config/``.
    # The glob fallback below handles other layouts.
    config_candidates = [
        f"{grobid_home}/grobid-home/config/grobid.yaml",
        f"{grobid_home}/grobid-service/config/config.yaml",
        f"{grobid_home}/grobid-service/config/grobid.yaml",
        f"{grobid_home}/grobid-service/config.yaml",
        f"{grobid_home}/grobid-service/grobid.yaml",
    ]
    config_path: str | None = None
    for c in config_candidates:
        if os.path.exists(c):
            config_path = c
            break
    if config_path is None:
        # Fall back to whatever we can find under grobid-home or /opt.
        for probe_root in (grobid_home, "/opt"):
            found = glob.glob(
                f"{probe_root}/**/grobid-service*/config/*.yaml",
                recursive=True,
            )
            print(f"[citeclaw-grobid] probe {probe_root}: {found}")
            if found:
                config_path = found[0]
                break
            found = glob.glob(
                f"{probe_root}/**/grobid.yaml", recursive=True,
            )
            print(f"[citeclaw-grobid] probe grobid.yaml {probe_root}: {found}")
            if found:
                config_path = found[0]
                break
        if config_path is None:
            import subprocess as _sp
            _sp.run(["ls", "-la", grobid_home], check=False)
            _sp.run(["ls", "-la", f"{grobid_home}/grobid-home"], check=False)
            raise RuntimeError(
                "Could not locate GROBID config yaml anywhere under "
                + grobid_home
            )

    print(
        f"[citeclaw-grobid] Launching {entry} "
        f"(cwd={grobid_home}, config={config_path}, port={PORT})"
    )
    subprocess.Popen(
        [entry, "server", config_path],
        cwd=grobid_home,
    )


@app.local_entrypoint()
def main() -> None:
    url = serve.get_web_url() if hasattr(serve, "get_web_url") else "(run `modal deploy` first)"
    print("=" * 72)
    print(f"CiteClaw GROBID server — Modal app: {APP_NAME}")
    print("=" * 72)
    print(f"  Image: {GROBID_IMAGE_REF}")
    print(f"  CPU:   {CPU_COUNT}")
    print(f"  Mem:   {MEMORY_MB} MB")
    print(f"  URL:   {url}")
    print()
    print("Export in your shell so CiteClaw picks it up:")
    print(f'  export CITECLAW_GROBID_URL="{url}"')
    print()
