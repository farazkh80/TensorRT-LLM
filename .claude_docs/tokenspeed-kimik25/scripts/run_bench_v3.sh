#!/usr/bin/env bash
# Phase 4 A/B re-run on the rebased build (post-PR-14291).
#
# Differences from run_bench_v2.sh:
#   - Output goes to RUN_DIR_BASE (default
#     /scratch/runs/k2.6-spike/phase4-rebased-bench) so Phase 4's
#     TS-only-on-rc14 logs in phase4-bench/ are preserved.
#   - No PYTHONPATH shadow needed: the editable install replaced the
#     container's rc14 tensorrt_llm Python tree with the rebased source,
#     and the rebuilt libtensorrt_llm.so is also in the source tree.
#   - Runs the four configs in a single batch instead of one-at-a-time.
#
# Each config produces /scratch/.../phase4-rebased-bench/<config>/{base,ts}.log
# plus a top-level summary.txt with the A/B table.
#
# Total wall-time estimate: 4 configs × 2 arms × ~7-10 min/arm ≈ 60-90 min.

set -uo pipefail

CONTAINER="${CONTAINER:-tokenspeed-spike-k26}"
MODEL_PATH="${MODEL_PATH:-/scratch/hf-cache-patched/k2.6-bf16kv}"
RUN_DIR_BASE="${RUN_DIR_BASE:-/scratch/runs/k2.6-spike/phase4-rebased-bench}"
RUN_DIR_BASE_HOST="${RUN_DIR_BASE_HOST:-/home/scratch.fkhoubsirat_coreai/runs/k2.6-spike/phase4-rebased-bench}"

# Per-config setup helper. CONCURRENCY / ISL / OSL / TP parsed from filename.
configs=(
    "bench-config:1:1024:1024:4"
    "bench-1k1k_tp8_conc1:1:1024:1024:8"
    "bench-8k1k_tp4_conc1:1:8192:1024:4"
    "bench-1k1k_tp4_conc16:16:1024:1024:4"
)

mkdir -p "$RUN_DIR_BASE_HOST"
chmod 777 "$RUN_DIR_BASE_HOST"
echo "[v3] container=$CONTAINER model=$MODEL_PATH run_dir_base=$RUN_DIR_BASE"

for entry in "${configs[@]}"; do
    IFS=":" read -r CONFIG_NAME CONC ISL OSL TP <<< "$entry"
    NUM_REQ=$((CONC * 16))

    RUN_DIR="$RUN_DIR_BASE/$CONFIG_NAME"
    RUN_DIR_HOST="$RUN_DIR_BASE_HOST/$CONFIG_NAME"
    SRC_CFG_HOST="/home/farazkh_scratch/parallel/TensorRT-LLM/.claude_docs/tokenspeed-kimik25/${CONFIG_NAME}.yml"

    mkdir -p "$RUN_DIR_HOST"
    chmod 777 "$RUN_DIR_HOST"

    echo
    echo "============================================================"
    echo "[v3] CONFIG=$CONFIG_NAME tp=$TP isl=$ISL osl=$OSL conc=$CONC num_req=$NUM_REQ"
    echo "[v3] RUN_DIR=$RUN_DIR_HOST"
    echo "============================================================"

    docker exec "$CONTAINER" bash -c "
        set -uo pipefail
        mkdir -p $RUN_DIR
        BASE_CFG=$RUN_DIR/_config_base.yml
        TS_CFG=$RUN_DIR/_config_ts.yml
        sed 's/^attn_backend:.*/attn_backend: TRTLLM/' /workspace/TensorRT-LLM/.claude_docs/tokenspeed-kimik25/${CONFIG_NAME}.yml > \$BASE_CFG
        sed 's/^attn_backend:.*/attn_backend: TOKENSPEED_MLA/' /workspace/TensorRT-LLM/.claude_docs/tokenspeed-kimik25/${CONFIG_NAME}.yml > \$TS_CFG

        for VARIANT in base ts; do
            if [[ \$VARIANT == base ]]; then
                CFG=\$BASE_CFG; BACKEND=TRTLLM
            else
                CFG=\$TS_CFG;   BACKEND=TOKENSPEED_MLA
            fi
            LOG=$RUN_DIR/\${VARIANT}.log
            echo \"[v3] running \$VARIANT (attn_backend=\$BACKEND) -> \$LOG\"
            python3 -u /workspace/TensorRT-LLM/.claude_docs/tokenspeed-kimik25/scripts/minimal_bench.py \
                --model_dir $MODEL_PATH \
                --tp_size $TP \
                --isl $ISL --osl $OSL \
                --num_requests $NUM_REQ --concurrency $CONC \
                --extra_llm_api_options \$CFG \
                > \$LOG 2>&1
            rc=\$?
            if [[ \$rc -ne 0 ]]; then
                echo \"  WARN: \$VARIANT exit=\$rc. tail of log:\"
                tail -20 \$LOG
            fi
        done
    "

    # Capture summary line for this config
    SUMMARY="$RUN_DIR_HOST/summary.txt"
    : > "$SUMMARY"
    for VARIANT in base ts; do
        LOG="$RUN_DIR_HOST/${VARIANT}.log"
        printf "%-4s  " "$VARIANT" >> "$SUMMARY"
        if grep -qE 'Per User Output|Token Throughput|Inter Token' "$LOG" 2>/dev/null; then
            grep -E 'Token Throughput|Inter Token|Per User Output Throughput median|Total wall|init time' "$LOG" \
                | tail -8 | tr '\n' '|' | sed 's/|/  /g' >> "$SUMMARY"
            echo >> "$SUMMARY"
        else
            # Look for NVRTC failure as a distinct signal
            if grep -q "NVRTC_ERROR_COMPILATION" "$LOG" 2>/dev/null; then
                echo "(NVRTC compilation failure — regression vs. PR #14291)" >> "$SUMMARY"
            else
                echo "(no perf summary — see $LOG)" >> "$SUMMARY"
            fi
        fi
    done
    echo "--- $CONFIG_NAME summary ---"
    cat "$SUMMARY"
done

echo
echo "============================================================"
echo "[v3] ALL DONE — final summary"
echo "============================================================"
for entry in "${configs[@]}"; do
    IFS=":" read -r CONFIG_NAME _ _ _ _ <<< "$entry"
    echo "## $CONFIG_NAME"
    cat "$RUN_DIR_BASE_HOST/$CONFIG_NAME/summary.txt" 2>&1
    echo
done
