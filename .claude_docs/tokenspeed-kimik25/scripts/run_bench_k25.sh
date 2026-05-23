#!/usr/bin/env bash
# Phase 5 K2.5 NVFP4 + EAGLE-3 A/B driver — trtllm-bench edition.
#
# Runs one config (bench-k25-mtp3.yml) with two arms:
#   base: attn_backend: TRTLLM
#   ts:   attn_backend: TOKENSPEED_MLA
#
# trtllm-bench is preferred over minimal_bench.py because:
#   - Native acceptance-rate reporting for spec-decode runs.
#   - Standard published-comparable metrics.
#
# If K2.5's HF tokenizer is broken (the same way K2.6's was), the
# `trtllm-bench prepare-dataset` step will fail; in that case re-run with
# USE_MINIMAL_BENCH=1 to fall back to minimal_bench.py + token-ID
# prompts (which bypasses AutoTokenizer).
#
# Output: /scratch/runs/k2.6-spike/phase5-k25-mtp3/{base,ts}.log

set -uo pipefail

CONTAINER="${CONTAINER:-tokenspeed-spike-k26}"
MODEL_PATH="${MODEL_PATH:-/scratch/hf-cache-patched/k2.5-bf16kv}"
MODEL_ID="${MODEL_ID:-nvidia/Kimi-K2.5-NVFP4}"
CFG_SRC="${CFG_SRC:-.claude_docs/tokenspeed-kimik25/bench-k25-mtp3.yml}"
RUN_DIR_HOST="/home/scratch.fkhoubsirat_coreai/runs/k2.6-spike/phase5-k25-mtp3"
RUN_DIR="/scratch/runs/k2.6-spike/phase5-k25-mtp3"

ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
NUM_REQ="${NUM_REQ:-32}"
CONC="${CONC:-2}"
TP="${TP:-4}"

USE_MINIMAL_BENCH="${USE_MINIMAL_BENCH:-0}"

mkdir -p "$RUN_DIR_HOST"
chmod 777 "$RUN_DIR_HOST"

echo "[v_k25] container=$CONTAINER model=$MODEL_PATH"
echo "[v_k25] run_dir=$RUN_DIR  ISL=$ISL OSL=$OSL num_req=$NUM_REQ conc=$CONC tp=$TP"
echo "[v_k25] use_minimal_bench=$USE_MINIMAL_BENCH"

# Pre-flight: model + EAGLE3 snapshot exist inside the container.
docker exec "$CONTAINER" bash -lc "
    set -e
    [[ -d $MODEL_PATH ]] || { echo 'ERROR: $MODEL_PATH not found'; exit 1; }
    EAGLE=\$(grep speculative_model_dir /workspace/TensorRT-LLM/$CFG_SRC | sed 's/.*: *//')
    [[ -d \"\$EAGLE\" ]] || { echo \"ERROR: \$EAGLE not found\"; exit 1; }
    echo '[v_k25] pre-flight OK: target + EAGLE3 draft present'
"

docker exec "$CONTAINER" bash -c "
    set -uo pipefail
    mkdir -p $RUN_DIR
    BASE_CFG=$RUN_DIR/_config_base.yml
    TS_CFG=$RUN_DIR/_config_ts.yml
    sed 's/^attn_backend:.*/attn_backend: TRTLLM/' /workspace/TensorRT-LLM/$CFG_SRC > \$BASE_CFG
    sed 's/^attn_backend:.*/attn_backend: TOKENSPEED_MLA/' /workspace/TensorRT-LLM/$CFG_SRC > \$TS_CFG

    # Tokenizer sanity gate + dataset build. If this fails, the
    # trtllm-bench path is dead.
    DATASET=$RUN_DIR/synth_${ISL}_${OSL}_${NUM_REQ}.json
    if [[ $USE_MINIMAL_BENCH -eq 0 ]]; then
        if [[ ! -s \$DATASET ]]; then
            echo '[v_k25] prepare-dataset (tokenizer sanity gate + dataset build)'
            python3 -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_PATH', trust_remote_code=True)\" \
                > $RUN_DIR/_tokenizer_check.log 2>&1
            rc=\$?
            if [[ \$rc -ne 0 ]]; then
                echo '  ERROR: K2.5 tokenizer is broken (same as K2.6 was).'
                echo '  Run with USE_MINIMAL_BENCH=1 to fall back.'
                tail -20 $RUN_DIR/_tokenizer_check.log
                exit 1
            fi
            echo '  tokenizer OK; building dataset...'
            trtllm-bench --model $MODEL_ID --model_path $MODEL_PATH \
                prepare-dataset --trust-remote-code --output \$DATASET \
                token-norm-dist \
                --num-requests $NUM_REQ \
                --input-mean $ISL --input-stdev 0 \
                --output-mean $OSL --output-stdev 0 \
                > $RUN_DIR/_prepare-dataset.log 2>&1
            rc=\$?
            if [[ \$rc -ne 0 || ! -s \$DATASET ]]; then
                echo '  ERROR: prepare-dataset failed (rc=\$rc, dataset size=$(stat -c %s \$DATASET 2>/dev/null || echo 0))'
                tail -25 $RUN_DIR/_prepare-dataset.log
                exit 1
            fi
            echo \"  dataset built: \$(wc -l <\$DATASET) lines, \$(stat -c %s \$DATASET) bytes\"
        fi
    fi

    for VARIANT in base ts; do
        if [[ \$VARIANT == base ]]; then
            CFG=\$BASE_CFG; BACKEND=TRTLLM
        else
            CFG=\$TS_CFG;   BACKEND=TOKENSPEED_MLA
        fi
        LOG=$RUN_DIR/\${VARIANT}.log
        echo \"[v_k25] running \$VARIANT (attn_backend=\$BACKEND) -> \$LOG\"

        if [[ $USE_MINIMAL_BENCH -eq 1 ]]; then
            python3 -u /workspace/TensorRT-LLM/.claude_docs/tokenspeed-kimik25/scripts/minimal_bench.py \
                --model_dir $MODEL_PATH \
                --tp_size $TP \
                --isl $ISL --osl $OSL \
                --num_requests $NUM_REQ --concurrency $CONC \
                --extra_llm_api_options \$CFG \
                > \$LOG 2>&1
        else
            # IMPORTANT: trtllm-bench's engine-sizing heuristic
            # ignores tensor_parallel_size from --extra_llm_api_options
            # YAML; --tp/--ep must be on the CLI.
            trtllm-bench --model $MODEL_ID --model_path $MODEL_PATH \
                throughput --dataset \$DATASET \
                --extra_llm_api_options \$CFG \
                --tp $TP --ep $TP \
                --kv_cache_free_gpu_mem_fraction 0.85 \
                --concurrency $CONC --num_requests $NUM_REQ --backend pytorch \
                > \$LOG 2>&1
        fi
        rc=\$?
        if [[ \$rc -ne 0 ]]; then
            echo \"  WARN: \$VARIANT exit=\$rc. tail of log:\"
            tail -25 \$LOG
        fi
    done
"

# Summary
echo
echo '=== Phase 5 K2.5 EAGLE-3 A/B summary ==='
for VARIANT in base ts; do
    LOG="$RUN_DIR_HOST/${VARIANT}.log"
    printf "  %-4s  " "$VARIANT"
    if grep -qE 'Per User Output|Token Throughput|Token Throughput \(' "$LOG" 2>/dev/null; then
        grep -E 'Token Throughput|Inter Token|Per User Output Throughput|Total wall|LLM init|Acceptance|Average accepted|TPOT' "$LOG" \
            | tail -10 | tr '\n' '|' | sed 's/|/  /g'
        echo
    elif grep -q "NVRTC_ERROR_COMPILATION" "$LOG" 2>/dev/null; then
        echo "(NVRTC compile failure — investigate)"
    elif grep -q "tokenizer is broken" "$LOG" 2>/dev/null; then
        echo "(tokenizer broken; re-run with USE_MINIMAL_BENCH=1)"
    else
        echo "(no perf summary — see $LOG)"
    fi
done
echo "Full logs in $RUN_DIR"
