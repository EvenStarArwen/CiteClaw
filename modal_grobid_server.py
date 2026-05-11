"""Standalone Modal deployment: GROBID REST server for CiteClaw.

Sister to ``modal_vllm_server.py`` — an isolated, single-file Modal app
that runs a GROBID server as a web-tunneled REST service. Only depends
on ``modal``; nothing in the main CiteClaw package imports from this
file.

GROBID is a Java/ML tool purpose-built for extracting structured data
from scientific PDFs: body text, sections, and — most importantly —
structured reference lists with cleanly parsed titles / authors / DOIs.

This file defaults to the **deep-learning** flavour of GROBID running
on an **L40S** GPU.  GROBID's DL pipeline (DeLFT/TensorFlow) is a few
F1 points better than CRF on reference parsing (~0.90 vs ~0.87) at the
cost of an order-of-magnitude higher hourly Modal bill (L40S ~ $1.95/h
vs an 8-CPU box at ~ $0.17/h).  Flip ``CITECLAW_GROBID_GPU=none`` plus
``CITECLAW_GROBID_IMAGE=lfoppiano/grobid:0.8.2-crf`` to fall back to the
CRF/CPU build when the cost is more pressing than the marginal recall.

============================================================================
Quickstart
============================================================================

One-time setup::

    pip install modal
    modal setup

Deploy (defaults: DL image, L40S, 1×GPU)::

    modal deploy modal_grobid_server.py

Or run the CPU-only CRF variant::

    CITECLAW_GROBID_IMAGE=lfoppiano/grobid:0.8.2-crf \
    CITECLAW_GROBID_GPU=none \
    modal deploy modal_grobid_server.py

Modal prints the public URL, e.g.::

    https://<you>--citeclaw-grobid-serve.modal.run

Point CiteClaw at it::

    export PDFCLAW_GROBID_URL="https://<you>--citeclaw-grobid-serve.modal.run"

Stop::

    modal app stop citeclaw-grobid
"""

from __future__ import annotations

import os

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default image: the official ``grobid/grobid:0.8.2`` is the full ~8 GB
# image with DeLFT models bundled — needed for the GPU-accelerated DL
# path.  Override with ``lfoppiano/grobid:0.8.2-crf`` for the lighter
# CRF/CPU variant.
GROBID_IMAGE_REF: str = os.environ.get(
    "CITECLAW_GROBID_IMAGE",
    "grobid/grobid:0.8.2",
)
APP_NAME: str = os.environ.get("CITECLAW_GROBID_APP_NAME", "citeclaw-grobid")
CPU_COUNT: float = float(os.environ.get("CITECLAW_GROBID_CPU", "8"))
MEMORY_MB: int = int(os.environ.get("CITECLAW_GROBID_MEMORY_MB", "16384"))
SCALEDOWN: int = int(os.environ.get("CITECLAW_GROBID_SCALEDOWN", "300"))
MAX_CONCURRENCY: int = int(os.environ.get("CITECLAW_GROBID_MAX_CONCURRENCY", "32"))
PORT = 8070

# ``L40S`` is a good default: 48 GB VRAM, well-priced on Modal, and
# DeLFT's models fit comfortably with room for batching.  Set
# ``CITECLAW_GROBID_GPU=none`` (or empty) to disable GPU entirely —
# combine with the CRF image for the cheap path.
GPU_TYPE: str = os.environ.get("CITECLAW_GROBID_GPU", "L40S")
GPU_COUNT: int = int(os.environ.get("CITECLAW_GROBID_GPU_COUNT", "1"))

# When the DL engine is enabled, GROBID needs a hint to bypass the
# default CRF wiring.  We do this by patching ``grobid.yaml`` at
# container start (see :func:`_patch_grobid_config_for_dl`).
ENGINE_OVERRIDE: str = os.environ.get(
    "CITECLAW_GROBID_ENGINE", "delft" if GPU_TYPE.lower() not in ("", "none") else "wapiti"
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

# Pull the pre-built GROBID image directly from Docker Hub. The image
# already has Java + GROBID installed and ships with an ENTRYPOINT that
# launches the server on port 8070, so we don't need to re-implement
# the startup command — just launch the binary ourselves so we can patch
# the config first.
#
# Both grobid/grobid:0.8.2 and lfoppiano/grobid:0.8.2-crf ship Python
# 3.11 with the DeLFT / scikit-learn deps GROBID needs, so we leave
# ``add_python`` off and let Modal use the image's bundled interpreter.
# (When ``add_python`` is set, Modal tries to symlink ``python3 →
# python`` which collides with the symlink the GROBID image already
# created.)
image = modal.Image.from_registry(GROBID_IMAGE_REF)


def _gpu_spec() -> str | None:
    """Translate the env-var pair into Modal's ``gpu=...`` argument.

    Returns ``None`` when the user has asked for CPU-only — Modal
    expects the kwarg absent in that case, not a string like ``"none"``.
    """
    g = (GPU_TYPE or "").strip()
    if not g or g.lower() == "none":
        return None
    return f"{g}:{GPU_COUNT}" if GPU_COUNT > 1 else g


@app.function(
    image=image,
    gpu=_gpu_spec(),
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

    The image installs to ``/opt/grobid``. The upstream CMD is::

        ./grobid-service/bin/grobid-service server grobid-service/config/config.yaml

    run with ``WORKDIR=/opt/grobid`` and relative paths. We replicate
    that here, with a filesystem probe to locate the config under
    alternate names (recent GROBID versions have renamed
    ``config.yaml`` to ``grobid.yaml`` in some distributions), then
    patch the engine setting to ``delft`` for GPU/DL deployments.
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

    # Patch the engine setting when the user has asked for DL inference.
    # GROBID ships with ``engine: "wapiti"`` (CRF) for every model;
    # flipping it to ``"delft"`` routes through the deep-learning
    # pipeline, which auto-detects an available GPU via TensorFlow.
    if ENGINE_OVERRIDE.lower() == "delft":
        _patch_grobid_config_for_dl(config_path)

    print(
        f"[citeclaw-grobid] Launching {entry} "
        f"(cwd={grobid_home}, config={config_path}, port={PORT}, "
        f"gpu={_gpu_spec()}, engine={ENGINE_OVERRIDE})"
    )
    subprocess.Popen(
        [entry, "server", config_path],
        cwd=grobid_home,
    )


def _patch_grobid_config_for_dl(config_path: str) -> None:
    """Swap ``engine: "wapiti"`` → ``engine: "delft"`` in GROBID config.

    This is the minimal change that lights up the DL inference path —
    DeLFT loads the bundled TensorFlow checkpoints for citation /
    header / fulltext models and auto-detects the GPU.  Done with
    ``sed`` so we avoid pulling PyYAML into the image just for this.
    """
    import subprocess

    print(f"[citeclaw-grobid] Patching {config_path}: wapiti → delft")
    # Match both quoted (``"wapiti"``) and bare (``wapiti``) forms, and
    # tolerate any indentation.  ``-E`` enables extended regex.
    result = subprocess.run(
        [
            "sed", "-i", "-E",
            r's/engine:\s*"?wapiti"?/engine: "delft"/g',
            config_path,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        print(
            f"[citeclaw-grobid] WARNING: config patch failed: {result.stderr!r}"
        )


@app.local_entrypoint()
def main() -> None:
    url = serve.get_web_url() if hasattr(serve, "get_web_url") else "(run `modal deploy` first)"
    print("=" * 72)
    print(f"CiteClaw GROBID server — Modal app: {APP_NAME}")
    print("=" * 72)
    print(f"  Image:  {GROBID_IMAGE_REF}")
    print(f"  GPU:    {_gpu_spec() or 'CPU-only'}")
    print(f"  Engine: {ENGINE_OVERRIDE}")
    print(f"  CPU:    {CPU_COUNT}")
    print(f"  Mem:    {MEMORY_MB} MB")
    print(f"  URL:    {url}")
    print()
    print("Export in your shell so CiteClaw picks it up:")
    print(f'  export PDFCLAW_GROBID_URL="{url}"')
    print()
