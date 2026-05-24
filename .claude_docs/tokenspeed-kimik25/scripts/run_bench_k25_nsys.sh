#!/usr/bin/env bash
# Phase 5 follow-up: nsys pure-kernel A/B for K2.5 NVFP4 + EAGLE-3 mtp=3.
#
# Same config as bench-k25-mtp3.yml (TP=4, ISL=OSL=1024, BS=2, conc=2,
# max_draft_len=3), but:
#   - Reduced to num_requests=4 to keep nsys traces manageable (~2-3 GB each).
#   - Wrapped with `nsys profile` to capture MLA-decode kernel times.
#
# Goal: isolate forward-kernel time per arm, removing the EAGLE
# acceptance-rate confound that contaminated the perf-only Phase 5
# throughput numbers (TS +6.7% AL vs +4.2% throughput → kernel work
# may actually be ~2-3% slower under TS, hidden by higher AL).

set -uo pipefail

CONTAINER="${CONTAINER:-tokenspeed-spike-k26}"
MODEL_PATH="${MODEL_PATH:-/scratch/hf-cache-patched/k2.5-bf16kv}"
MODEL_ID="${MODEL_ID:-nvidia/Kimi-K2.5-NVFP4}"
CFG_SRC="${CFG_SRC:-.claude_docs/tokenspeed-kimik25/bench-k25-mtp3-tp8.yml}"

RUN_DIR_HOST="/home/scratch.fkhoubsirat_coreai/runs/k2.6-spike/phase5-k25-mtp3-nsys"
RUN_DIR="/scratch/runs/k2.6-spike/phase5-k25-mtp3-nsys"

ISL=1024
OSL=1024
NUM_REQ=4
CONC=2
TP=8

mkdir -p "$RUN_DIR_HOST"
chmod 777 "$RUN_DIR_HOST"

echo "[v_k25_nsys] container=$CONTAINER model=$MODEL_PATH"
echo "[v_k25_nsys] run_dir=$RUN_DIR  ISL=$ISL OSL=$OSL num_req=$NUM_REQ conc=$CONC tp=$TP"

# Pre-flight: reuse the dataset from the perf-only run if it exists; else build.
PERF_DATASET=/scratch/runs/k2.6-spike/phase5-k25-mtp3/synth_1024_1024_32.json

docker exec "$CONTAINER" bash -c "
    set -uo pipefail
    mkdir -p $RUN_DIR

    # Sidecar configs (base + ts), reusing the Phase 5 YAML.
    BASE_CFG=$RUN_DIR/_config_base.yml
    TS_CFG=$RUN_DIR/_config_ts.yml
    sed 's/^attn_backend:.*/attn_backend: TRTLLM/' /workspace/TensorRT-LLM/$CFG_SRC > \$BASE_CFG
    sed 's/^attn_backend:.*/attn_backend: TOKENSPEED_MLA/' /workspace/TensorRT-LLM/$CFG_SRC > \$TS_CFG

    # Reuse the perf-only dataset (32 reqs); --num_requests will cap.
    DATASET=$PERF_DATASET
    if [[ ! -s \$DATASET ]]; then
        echo 'ERROR: perf-only dataset not found at \$DATASET'
        exit 1
    fi
    echo \"[v_k25_nsys] reusing dataset: \$DATASET\"

    for VARIANT in base ts; do
        if [[ \$VARIANT == base ]]; then
            CFG=\$BASE_CFG; BACKEND=TRTLLM
        else
            CFG=\$TS_CFG;   BACKEND=TOKENSPEED_MLA
        fi
        NSYS_OUT=$RUN_DIR/\${VARIANT}.nsys-rep
        STDOUT_LOG=$RUN_DIR/\${VARIANT}.log

        echo \"[v_k25_nsys] running \$VARIANT (attn_backend=\$BACKEND) -> \$NSYS_OUT\"

        # nsys options:
        #   -t cuda,nvtx,osrt: trace CUDA API + NVTX (TRT-LLM emits these)
        #     + osrt for proc creation (so we see all MPI rank workers).
        #   -s none: no CPU sampling (we only care about GPU kernels).
        #   --capture-range=none: profile entire lifetime; we'll filter
        #     by NVTX/kernel patterns in analysis.
        #   --force-overwrite=true: replace prior trace.
        #   --output=...
        nsys profile \
            -t cuda,nvtx,osrt -s none \
            --capture-range=none \
            --force-overwrite=true \
            --output=\$NSYS_OUT \
            trtllm-bench --model $MODEL_ID --model_path $MODEL_PATH \
                throughput --dataset \$DATASET \
                --extra_llm_api_options \$CFG \
                --tp $TP \
                --concurrency $CONC --num_requests $NUM_REQ --backend pytorch \
                > \$STDOUT_LOG 2>&1
        rc=\$?
        if [[ \$rc -ne 0 ]]; then
            echo \"  WARN: \$VARIANT exit=\$rc. tail of log:\"
            tail -25 \$STDOUT_LOG
        fi

        # Extract kernel summary CSV
        echo \"[v_k25_nsys]   producing cudakernsum CSV...\"
        nsys stats --report cudakernsum --format csv \
            --output=$RUN_DIR/\${VARIANT}-kernsum \
            \$NSYS_OUT > $RUN_DIR/\${VARIANT}.kernsum.log 2>&1
    done
"

# Host-side: build the comparison table
echo
echo '=== Phase 5 nsys pure-kernel A/B summary ==='
for VARIANT in base ts; do
    CSV=\$(ls $RUN_DIR_HOST/\${VARIANT}-kernsum_cudakernsum.csv 2>/dev/null | head -1)
    if [[ -z "$CSV" || ! -s "$CSV" ]]; then
        CSV=$(ls "$RUN_DIR_HOST"/${VARIANT}*cudakernsum*csv 2>/dev/null | head -1)
    fi
    echo "--- $VARIANT  csv=$CSV ---"
    if [[ -n "$CSV" && -s "$CSV" ]]; then
        echo "MLA / FMHA / TokenSpeed kernels (top 20 by total time):"
        head -2 "$CSV"
        grep -iE "fmha|mla|tokenspeed|cute_dsl|sm103a|attention" "$CSV" 2>/dev/null | head -20
    else
        echo "(no kernsum CSV)"
    fi
done
echo
echo "Full nsys traces in $RUN_DIR_HOST/"
ls -la "$RUN_DIR_HOST/" 2>&1 | head -10
