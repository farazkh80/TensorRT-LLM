#!/usr/bin/env bash
# Set up the K2.6 TokenSpeed MLA eval container.
#
# Run from host. Idempotent (kills + recreates the `tokenspeed-spike-k26`
# container; patch script is itself idempotent via its marker check).
#
# Prerequisite: K2.6 download is in /home/scratch.fkhoubsirat_coreai/hf-cache/
# (snapshot_download layout — see ../runbook.md Phase 1).
#
# Container layout when this script finishes:
#   /workspace/TensorRT-LLM          ← host TensorRT-LLM source (RW bind)
#   /workspace/tokenspeed            ← local tokenspeed clone (RO bind, for ref only)
#   /scratch                         ← coreai scratch mount (RW)
#   /scratch/hf-cache                ← HF snapshot of K2-Thinking-NVFP4
#   /scratch/runs/k2.6-spike         ← nsys / bench output
#   tokenspeed_mla.py at the package's installed site-packages path
#   apply_patches.py marker applied to utils.py, trtllm_gen.py, mla_decode.py

set -euo pipefail

CONTAINER="${CONTAINER:-tokenspeed-spike-k26}"
IMAGE="${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14}"
HOST_TRTLLM="${HOST_TRTLLM:-/home/farazkh_scratch/parallel/TensorRT-LLM}"
HOST_TOKENSPEED="${HOST_TOKENSPEED:-/home/farazkh_scratch/tokenspeed}"
HOST_SCRATCH="${HOST_SCRATCH:-/home/scratch.fkhoubsirat_coreai}"
SITE_ATTN="/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/attention_backend"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$THIS_DIR/../code" && pwd)"

echo "[$(date +%H:%M:%S)] tearing down old container if present"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "[$(date +%H:%M:%S)] launching $CONTAINER ($IMAGE)"
docker run -d --name "$CONTAINER" \
    --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$HOST_TRTLLM":/workspace/TensorRT-LLM \
    -v "$HOST_TOKENSPEED":/workspace/tokenspeed:ro \
    -v "$HOST_SCRATCH":/scratch \
    -e HF_HOME=/scratch/hf-cache \
    -e TRANSFORMERS_CACHE=/scratch/hf-cache \
    -e LD_LIBRARY_PATH=/usr/local/tensorrt/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    -w /scratch \
    "$IMAGE" sleep infinity >/dev/null
# NB: cwd intentionally /scratch (NOT /workspace) — the host TRT-LLM source is
# bind-mounted at /workspace/TensorRT-LLM, and /workspace on sys.path would
# shadow the container's installed `tensorrt_llm` package with the host's
# (newer than rc14) source, breaking `import tensorrt` due to ABI skew.
# Spike step 4 hit this same failure mode.

echo "[$(date +%H:%M:%S)] installing tokenspeed-mla inside container"
docker exec "$CONTAINER" pip install --quiet tokenspeed-mla

echo "[$(date +%H:%M:%S)] copying wrapper to container site-packages"
docker cp "$CODE_DIR/tokenspeed_mla.py" \
    "$CONTAINER:$SITE_ATTN/tokenspeed_mla.py"

echo "[$(date +%H:%M:%S)] applying inline patches (utils.py, trtllm_gen.py, mla_decode.py)"
docker cp "$THIS_DIR/apply_patches.py" \
    "$CONTAINER:/tmp/apply_patches.py"
docker exec "$CONTAINER" python /tmp/apply_patches.py

echo "[$(date +%H:%M:%S)] smoke test"
docker exec -i "$CONTAINER" python -u - <<'PY'
import os, sys, tensorrt_llm
from tensorrt_llm._torch.attention_backend.tokenspeed_mla import (
    is_tokenspeed_mla_available,
    tokenspeed_batch_decode_with_kv_cache_mla,
)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
print(f"trtllm:   {tensorrt_llm.__file__}")
print(f"selector: TOKENSPEED_MLA -> {get_attention_backend('TOKENSPEED_MLA').__name__}")
print(f"is_tokenspeed_mla_available: {is_tokenspeed_mla_available()}")
PY

echo "[$(date +%H:%M:%S)] OK; container $CONTAINER is ready"
echo "  shell:  docker exec -it $CONTAINER bash"
echo "  next:   ./verify_kernel_swap.sh"
