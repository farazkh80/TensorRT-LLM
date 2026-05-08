#!/bin/bash
# Copy edited / new files from the host source tree into the wheel-installed
# site-packages location inside the b12x_luke_runtime container, so the next
# import picks them up. Idempotent — safe to run after every edit.
#
# Usage:
#   bash .claude_docs/nemo-fp4-moe-b12x-mr/sync_b12x_luke_files.sh
set -euo pipefail

NAME=${RUNTIME_NAME:-b12x_luke_runtime}

# Discover where the wheel installed tensorrt_llm. The wheel install emits
# noisy stdout warnings ("🚨 Config not found for parakeet", "[TensorRT-LLM]
# TensorRT LLM version: ...") so we filter them out and keep only the last
# line which is our `print(...)` value.
SITE_PKG=$(docker exec "$NAME" python3 -c 'import os, tensorrt_llm; print(os.path.dirname(tensorrt_llm.__file__))' 2>/dev/null | grep -E '^/.*tensorrt_llm$' | tail -1)
if [ -z "$SITE_PKG" ]; then
    echo "[sync] ERROR: could not locate tensorrt_llm site-packages dir"; exit 1
fi
echo "[sync] site-packages tensorrt_llm at: $SITE_PKG"

# Files we touch on this branch. Add to this list when more files change.
FILES=(
    tensorrt_llm/_torch/modules/fused_moe/__init__.py
    tensorrt_llm/_torch/modules/fused_moe/create_moe.py
    tensorrt_llm/_torch/modules/fused_moe/fused_moe_flashinfer.py
    tensorrt_llm/_torch/modules/fused_moe/fused_moe_b12x_luke.py
    tensorrt_llm/llmapi/llm_args.py
)

for f in "${FILES[@]}"; do
    src=/workspace/TensorRT-LLM/$f
    dst_rel=${f#tensorrt_llm/}
    dst=$SITE_PKG/$dst_rel
    if docker exec "$NAME" test -e "$src"; then
        docker exec "$NAME" cp -v "$src" "$dst"
    else
        echo "[sync] skip (host file missing): $src"
    fi
done

# Drop any stale __pycache__ for these submodules.
docker exec "$NAME" bash -c "
    find $SITE_PKG/_torch/modules/fused_moe -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
    find $SITE_PKG/llmapi -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
"

echo "[sync] done."
