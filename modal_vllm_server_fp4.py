"""Modal deployment: vLLM-served Gemma 4 endpoints (FP4 default, BF16/FP8 via env).

Despite the ``_fp4`` filename, this file is the general-purpose Gemma 4
deploy — point ``CITECLAW_VLLM_FP4_MODEL`` at any vLLM-supported Gemma 4
checkpoint and adjust the GPU/quant defaults via env. The legacy
``modal_vllm_server.py`` only powers the BF16 v2 production deploy
(``citeclaw-vllm-gemma-v2``) and stays untouched.

Defaults
--------
* Model:       ``nvidia/Gemma-4-31B-IT-NVFP4``
* GPU:         ``H200`` (Modal's B200 capacity is no longer reliably available,
               so H200 is the default. NVFP4 falls back to Marlin weight-only
               dequant on Hopper — functional, ~30-50% slower decode than B200
               native FP4. Override to ``B200`` if/when capacity returns.)
* App name:    ``citeclaw-vllm-gemma-fp4``
* vLLM:        ``0.19.1`` (transformers 5.5.3 bundled — no FORCE_TRANSFORMERS)
* CUDA image:  ``nvidia/cuda:12.9.0-devel-ubuntu22.04``
* Context:     32768 (configurable; NVFP4 31B supports 256K natively)
* Concurrency: 1 container, max_inputs=32 (pick higher for throughput sweeps)
* Prefix cache: ON (must be opted-in via ``--enable-prefix-caching`` — vLLM's
  default for Gemma 4 is OFF)
* Reasoning parser: OFF (vLLM #38855 leaves it half-broken; client-side strip is more reliable)
* Tool-call parser: OFF (vLLM #39392 concurrent ``<pad>`` flood)

Configuration knobs (env vars)
------------------------------
=================================================  ============================
``CITECLAW_VLLM_FP4_APP_NAME``                     Modal app name
``CITECLAW_VLLM_FP4_MODEL``                        HF model id
``CITECLAW_VLLM_FP4_GPU``                          ``H200`` (default) / ``B200`` / ``H100`` / ...
``CITECLAW_VLLM_FP4_GPU_COUNT``                    Tensor-parallel size (default 1)
``CITECLAW_VLLM_FP4_API_KEY``                      Bearer token clients send
``CITECLAW_VLLM_FP4_MAX_MODEL_LEN``                Context length cap
``CITECLAW_VLLM_FP4_MAX_CONTAINERS``               Cap on container count Modal can scale to
``CITECLAW_VLLM_FP4_MAX_INPUTS``                   Per-container in-flight inputs (fan-out trigger)
``CITECLAW_VLLM_FP4_SCALEDOWN``                    Idle seconds before container shuts down
``CITECLAW_VLLM_FP4_VLLM_VERSION``                 vLLM pip version (default 0.19.1)
``CITECLAW_VLLM_FP4_CUDA_IMAGE``                   CUDA base image tag
``CITECLAW_VLLM_FP4_HF_SECRET``                    Modal secret name (must contain HF_TOKEN)
``CITECLAW_VLLM_FP4_PREFIX_CACHING``               ``1`` enable / ``0`` disable / ``""`` leave default
``CITECLAW_VLLM_FP4_REASONING_PARSER``             ``""`` off / ``gemma4``
``CITECLAW_VLLM_FP4_TOOL_CALL_PARSER``             ``""`` off / ``gemma4``
``CITECLAW_VLLM_FP4_FORCE_TRANSFORMERS``           pip target (e.g. ``git+...``); empty → none
``CITECLAW_VLLM_FP4_EXTRA_ARGS``                   Free-form extra ``vllm serve`` flags
=================================================  ============================

Quickstart — three common deploys
---------------------------------
NVFP4 on H200 (the current default; just deploy)::

    modal deploy modal_vllm_server_fp4.py

NVFP4 on B200 (when/if B200 capacity returns)::

    CITECLAW_VLLM_FP4_GPU=B200 \\
    modal deploy modal_vllm_server_fp4.py

BF16 on H200, separate app from the existing v2 production deploy::

    CITECLAW_VLLM_FP4_APP_NAME=citeclaw-vllm-gemma-bf16-v3 \\
    CITECLAW_VLLM_FP4_MODEL=google/gemma-4-31B-it \\
    CITECLAW_VLLM_FP4_GPU=H200 \\
    CITECLAW_VLLM_FP4_REASONING_PARSER=gemma4 \\
    modal deploy modal_vllm_server_fp4.py

FP8 (``RedHatAI/gemma-4-31B-it``, runs on H200)::

    CITECLAW_VLLM_FP4_APP_NAME=citeclaw-vllm-gemma-fp8 \\
    CITECLAW_VLLM_FP4_MODEL=RedHatAI/gemma-4-31B-it-FP8-block \\
    CITECLAW_VLLM_FP4_GPU=H200 \\
    modal deploy modal_vllm_server_fp4.py

Multi-container fan-out (8 H200, 96 simultaneous requests)::

    CITECLAW_VLLM_FP4_MAX_CONTAINERS=8 \\
    CITECLAW_VLLM_FP4_MAX_INPUTS=12 \\
    CITECLAW_VLLM_FP4_GPU=H200 \\
    modal deploy modal_vllm_server_fp4.py

After deploy::

    URL = https://<workspace>--<app-name>-serve.modal.run/v1
    KEY = $CITECLAW_VLLM_FP4_API_KEY (default: citeclaw-test-key)

Talk to it::

    from gemma4_client import Gemma4Client
    client = Gemma4Client.fp4()                     # or .bf16(), or pass base_url
    print(client.chat("Hello!", thinking="never"))

Why various flags are off by default
------------------------------------
* ``--quantization`` — vLLM auto-detects ``modelopt`` from config.json; passing
  it explicitly can cause "scheme mismatch" errors when the checkpoint
  declares compressed-tensors. Auto-detect is more robust.
* ``--reasoning-parser gemma4`` — vLLM #38855: parser does NOT populate
  ``message.reasoning_content`` for Gemma 4 (channel tokens stripped before
  the parser sees them). Client-side strip is more reliable. Also avoids
  the parser + xgrammar deadlock (#39130) and parser + tool-call deadlock.
  Override with ``CITECLAW_VLLM_FP4_REASONING_PARSER=gemma4`` if you want
  vLLM to try anyway.
* ``--tool-call-parser gemma4`` — vLLM #39392: ~40% of concurrent tool-call
  requests return 4096 ``<pad>`` tokens. Override with
  ``CITECLAW_VLLM_FP4_TOOL_CALL_PARSER=gemma4`` once the upstream lock
  fix lands.

Caveats / open issues to watch
------------------------------
* vLLM #39914 (engine hang at > ~4K-token prefill on Blackwell) is open
  as of April 2026. If your prompts are long, set
  ``CITECLAW_VLLM_FP4_EXTRA_ARGS="--enable-chunked-prefill"``.
* Long-context degradation past ~128K is documented; this deploy defaults
  to 32K to keep the cold start tight.
"""

from __future__ import annotations

import os

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME: str = os.environ.get(
    "CITECLAW_VLLM_FP4_MODEL", "nvidia/Gemma-4-31B-IT-NVFP4"
)
GPU_TYPE: str = os.environ.get("CITECLAW_VLLM_FP4_GPU", "H200")
GPU_COUNT: int = int(os.environ.get("CITECLAW_VLLM_FP4_GPU_COUNT", "1"))
API_KEY: str = os.environ.get("CITECLAW_VLLM_FP4_API_KEY", "citeclaw-test-key")
MAX_MODEL_LEN: int = int(os.environ.get("CITECLAW_VLLM_FP4_MAX_MODEL_LEN", "32768"))
SCALEDOWN: int = int(os.environ.get("CITECLAW_VLLM_FP4_SCALEDOWN", "300"))
MAX_CONTAINERS: int = int(os.environ.get("CITECLAW_VLLM_FP4_MAX_CONTAINERS", "1"))
MAX_INPUTS: int = int(os.environ.get("CITECLAW_VLLM_FP4_MAX_INPUTS", "32"))
HF_SECRET_NAME: str = os.environ.get(
    "CITECLAW_VLLM_FP4_HF_SECRET", "huggingface"
)
APP_NAME: str = os.environ.get(
    "CITECLAW_VLLM_FP4_APP_NAME", "citeclaw-vllm-gemma-fp4"
)
VLLM_VERSION: str = os.environ.get("CITECLAW_VLLM_FP4_VLLM_VERSION", "0.19.1")
EXTRA_VLLM_ARGS: str = os.environ.get("CITECLAW_VLLM_FP4_EXTRA_ARGS", "")

# CUDA base image. 12.9-devel matches vLLM 0.19.1 + the FP4 fixes in
# CUTLASS / flashinfer. Override to 12.8.0-devel if you specifically need
# to match the legacy BF16 deploy's environment, or to a 13.x tag for
# B300 / consumer Blackwell.
CUDA_IMAGE: str = os.environ.get(
    "CITECLAW_VLLM_FP4_CUDA_IMAGE", "nvidia/cuda:12.9.0-devel-ubuntu22.04"
)

# Prefix caching is OFF by default in vLLM for Gemma 4 (the engine logs it
# under "non-default args" when we pass --enable-prefix-caching). Reusing
# the KV blocks of common prefixes (system prompt + instruction) cuts
# prefill cost ~80% on workloads with shared prefixes. Almost always a win.
#   "1" → pass --enable-prefix-caching   (default; opt-in)
#   "0" → pass --no-enable-prefix-caching (force off)
#   ""  → leave at vLLM default          (currently equivalent to "0")
PREFIX_CACHING: str = os.environ.get("CITECLAW_VLLM_FP4_PREFIX_CACHING", "1")

# Reasoning parser. Default OFF: vLLM #38855 leaves Gemma4ReasoningParser
# half-broken — `message.reasoning_content` stays empty, all thinking
# leaks into `content`. The bundled gemma4_client strips client-side
# instead. Set to "gemma4" to enable anyway (e.g., for diagnostics or
# once the upstream fix lands).
REASONING_PARSER: str = os.environ.get("CITECLAW_VLLM_FP4_REASONING_PARSER", "")

# Tool-call parser. Default OFF: vLLM #39392 causes ~40% of concurrent
# tool-call requests to return 4096 `<pad>` tokens. Set to "gemma4" once
# the upstream lock fix merges, or if you need tool calls and can serialize
# them client-side.
TOOL_CALL_PARSER: str = os.environ.get("CITECLAW_VLLM_FP4_TOOL_CALL_PARSER", "")

# Force-install a transformers version AFTER vLLM, with --force-reinstall.
# vLLM 0.19.1 already pins transformers 5.5.3 which recognizes
# model_type=gemma4, so this knob is empty by default. Set to a pip target
# (e.g. "transformers==5.5.4" or "git+https://github.com/huggingface/transformers.git")
# only if you're pinning a different vLLM version that doesn't bundle a
# Gemma 4-aware transformers.
FORCE_TRANSFORMERS: str = os.environ.get(
    "CITECLAW_VLLM_FP4_FORCE_TRANSFORMERS", ""
)

PORT = 8000

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

# Reuse the same persistent HF cache as the BF16 deploy. HF stores by repo
# id, so caching `google/gemma-4-31B-it` and `nvidia/Gemma-4-31B-IT-NVFP4`
# in the same volume is fine — they don't collide.
hf_cache_vol = modal.Volume.from_name("citeclaw-hf-cache", create_if_missing=True)

_base_image = (
    # CUDA 12.9 devel (vs the legacy BF16 deploy's 12.8): newer CUTLASS gives
    # us the FP4 fixes that landed in vLLM 0.19.1 / 0.20. devel image is
    # required so flashinfer's JIT can find nvcc.
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("git")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]",
        # flashinfer-python provides the FP4 GEMM kernels and KV-cache fp8
        # kernel. vLLM 0.19+ defaults to FA4 for dense Gemma 4 on SM100, but
        # FP4 still goes through flashinfer.
        "flashinfer-python",
    )
)
if FORCE_TRANSFORMERS:
    _base_image = _base_image.run_commands(
        f"pip install --upgrade --force-reinstall {FORCE_TRANSFORMERS}",
    )

image = _base_image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Default V1 engine (it's the default in 0.20 but assert).
        "VLLM_USE_V1": "1",
        # Avoid NCCL hangs on some multi-GPU configs (no-op for TP=1).
        "NCCL_CUMEM_ENABLE": "0",
        # Help flashinfer's JIT find nvcc without probing.
        "CUDA_HOME": "/usr/local/cuda",
        # Propagate the resolved values into the container so that the
        # remote module re-import sees the same config the deploy
        # submitted (mirrors the BF16 file's pattern).
        "CITECLAW_VLLM_FP4_MODEL": MODEL_NAME,
        "CITECLAW_VLLM_FP4_GPU": GPU_TYPE,
        "CITECLAW_VLLM_FP4_GPU_COUNT": str(GPU_COUNT),
        "CITECLAW_VLLM_FP4_API_KEY": API_KEY,
        "CITECLAW_VLLM_FP4_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "CITECLAW_VLLM_FP4_SCALEDOWN": str(SCALEDOWN),
        "CITECLAW_VLLM_FP4_VLLM_VERSION": VLLM_VERSION,
        "CITECLAW_VLLM_FP4_MAX_CONTAINERS": str(MAX_CONTAINERS),
        "CITECLAW_VLLM_FP4_MAX_INPUTS": str(MAX_INPUTS),
        "CITECLAW_VLLM_FP4_HF_SECRET": HF_SECRET_NAME,
        "CITECLAW_VLLM_FP4_APP_NAME": APP_NAME,
        "CITECLAW_VLLM_FP4_EXTRA_ARGS": EXTRA_VLLM_ARGS,
        "CITECLAW_VLLM_FP4_CUDA_IMAGE": CUDA_IMAGE,
        "CITECLAW_VLLM_FP4_PREFIX_CACHING": PREFIX_CACHING,
        "CITECLAW_VLLM_FP4_REASONING_PARSER": REASONING_PARSER,
        "CITECLAW_VLLM_FP4_TOOL_CALL_PARSER": TOOL_CALL_PARSER,
        "CITECLAW_VLLM_FP4_FORCE_TRANSFORMERS": FORCE_TRANSFORMERS,
    }
)


# ---------------------------------------------------------------------------
# Web-server function
# ---------------------------------------------------------------------------

_GPU_SPEC = f"{GPU_TYPE}:{GPU_COUNT}" if GPU_COUNT > 1 else GPU_TYPE
_FUNCTION_SECRETS: list[modal.Secret] = (
    [modal.Secret.from_name(HF_SECRET_NAME)] if HF_SECRET_NAME else []
)


@app.function(
    image=image,
    gpu=_GPU_SPEC,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=_FUNCTION_SECRETS,
    scaledown_window=SCALEDOWN,
    max_containers=MAX_CONTAINERS,
    timeout=60 * 60,  # 1-hour container lifetime cap
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=PORT, startup_timeout=60 * 20)
def serve() -> None:
    """Launch ``vllm serve`` for the NVFP4 Gemma 4 31B checkpoint."""
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--api-key", API_KEY,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--trust-remote-code",
        "--dtype", "auto",
        "--kv-cache-dtype", "fp8",
        "--gpu-memory-utilization", "0.85",
    ]
    if PREFIX_CACHING == "1":
        cmd += ["--enable-prefix-caching"]
    elif PREFIX_CACHING == "0":
        cmd += ["--no-enable-prefix-caching"]
    # PREFIX_CACHING == "" (or any other) → leave at vLLM default.
    if REASONING_PARSER:
        cmd += ["--reasoning-parser", REASONING_PARSER]
    if TOOL_CALL_PARSER:
        cmd += ["--tool-call-parser", TOOL_CALL_PARSER]
    if GPU_COUNT > 1:
        cmd += ["--tensor-parallel-size", str(GPU_COUNT)]
    if EXTRA_VLLM_ARGS:
        import shlex
        cmd += shlex.split(EXTRA_VLLM_ARGS)

    print(
        f"[citeclaw-vllm-fp4] Launching: app={APP_NAME} "
        f"model={MODEL_NAME} gpu={_GPU_SPEC} max_len={MAX_MODEL_LEN}"
    )
    print(f"[citeclaw-vllm-fp4] Command: {' '.join(cmd)}")

    try:
        import vllm  # noqa: F401
        print(f"[citeclaw-vllm-fp4] vllm import OK (version={vllm.__version__})")
    except Exception as exc:
        print(f"[citeclaw-vllm-fp4] FATAL: import vllm failed: {exc!r}")
        raise

    subprocess.Popen(cmd)


# ---------------------------------------------------------------------------
# Local entrypoint: `modal run modal_vllm_server_fp4.py`
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    url = serve.get_web_url() if hasattr(serve, "get_web_url") else "(run modal deploy first)"
    print("=" * 72)
    print(f"CiteClaw vLLM FP4 server — Modal app: {APP_NAME}")
    print("=" * 72)
    print(f"  Model:         {MODEL_NAME}")
    print(f"  GPU:           {_GPU_SPEC}")
    print(f"  API key:       {API_KEY}")
    print(f"  Max context:   {MAX_MODEL_LEN}")
    print(f"  vLLM version:  {VLLM_VERSION}")
    print(f"  Scaledown:     {SCALEDOWN}s idle")
    print()
    print(f"  Endpoint URL:  {url}/v1")
    print()
    print("Wire into your CiteClaw config.yaml under models::")
    print()
    print('  gemma-4-31b-fp4:')
    print(f'    base_url: "{url}/v1"')
    print(f'    served_model_name: "{MODEL_NAME}"')
    print('    api_key_env: "CITECLAW_VLLM_API_KEY"')
    print('    thinking_budget: 4096   # tune per workload')
    print('    reasoning_parser: ""    # leave OFF — vLLM #38855')
    print()
