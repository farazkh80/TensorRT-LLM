#!/usr/bin/env bash
# Phase 3 follow-up: diagnose the actual dispatch decision in TrtllmAttention._run
# for K2.6 NVFP4 on B300. Settles whether the spike-patches path was
# unreachable (is_supported returns False) or reachable-but-ineffective.
#
# Applies diagnose_dispatch.py to the in-container trtllm.py, then runs
# minimal_generate.py once (no nsys, ~6 min total) with TLLM_DIAG=1 +
# TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 + TLLM_TOKENSPEED_MLA=1. Captures
# the one-shot stderr [DIAG] lines.
#
# Output goes to /scratch/runs/k2.6-spike/dispatch-diag/.

set -euo pipefail

CONTAINER="${CONTAINER:-tokenspeed-spike-k26}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/scratch/hf-cache/models--nvidia--Kimi-K2-Thinking-NVFP4}"
SNAPSHOT_GLOB="$MODEL_PATH/snapshots/*"
RUN_DIR="/scratch/runs/k2.6-spike/dispatch-diag"

echo "[diag] copying diagnose_dispatch.py to container"
docker cp "$THIS_DIR/diagnose_dispatch.py" "$CONTAINER":/tmp/diagnose_dispatch.py

echo "[diag] applying diagnostic patch (idempotent)"
docker exec "$CONTAINER" python /tmp/diagnose_dispatch.py

echo "[diag] running minimal_generate.py with TLLM_DIAG=1 (TS variant path)"
docker exec "$CONTAINER" bash -c "
    set -uo pipefail
    mkdir -p $RUN_DIR
    MODEL=\$(ls -d $SNAPSHOT_GLOB 2>/dev/null | head -1)
    LOG=$RUN_DIR/diag-stdout.log
    ERR=$RUN_DIR/diag-stderr.log
    echo \"[diag] model: \$MODEL\"
    TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 TLLM_TOKENSPEED_MLA=1 TLLM_DIAG=1 \
        python -u /workspace/TensorRT-LLM/.claude_docs/tokenspeed-kimik25/scripts/minimal_generate.py \
            --model_dir \"\$MODEL\" \
            --tp_size 4 \
            --max_tokens 32 \
            --attention_backend TRTLLM \
            > \"\$LOG\" 2> \"\$ERR\" || {
                echo \"  WARN: generate failed; tail of stderr:\" >&2
                tail -30 \"\$ERR\" >&2
            }
    echo
    echo \"=== [DIAG] lines from stderr ===\"
    grep -E '\\[DIAG\\]' \"\$ERR\" | sort -u
    echo
    echo \"=== minimal_generate stdout tail ===\"
    grep -E '^\\[minimal\\]' \"\$LOG\"
"
