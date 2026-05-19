#!/usr/bin/env bash
# Phase 3 (current-main edition) — verify TOKENSPEED_MLA backend swap.
#
# Replaces verify_kernel_swap.sh's env-var-based swap (which targeted the
# rc14 spike's run_mla_generation patch) with the proper backend-class
# swap landed in current main:
#
#   TensorRT-LLM/tensorrt_llm/_torch/attention_backend/
#     ├── tokenspeed_mla_attention.py  (the backend class)
#     ├── tokenspeed_mla.py            (the wrapper)
#     └── utils.py                     (the TOKENSPEED_MLA selector entry)
#
# Runs minimal_generate.py twice under nsys profile:
#   - base: --attention_backend TRTLLM
#   - ts:   --attention_backend TOKENSPEED_MLA
# Then diffs the kernel symbols. Three outcomes:
#
#   - swap fires: ts trace has CuTe DSL (TokenSpeed) MLA decode symbols
#     that the baseline doesn't; baseline-only symbols include trtllm-gen
#     ...ForGen kernels. Proceed to Phase 4 (run_bench.sh).
#
#   - swap silent: kernel sets identical → backend class didn't fire.
#     Investigate the dispatch gate (_tokenspeed_can_dispatch); most
#     likely a config mismatch (dtype, sinks, helix, sage, sparse).
#
#   - crash / OOM: K2.6-specific; debug.
#
# Both runs use PYTHONPATH=/workspace/TensorRT-LLM to shadow the rc14
# install with the current-main source tree (the host bind-mount). No
# editable install is required.

set -euo pipefail

CONTAINER="${CONTAINER:-tokenspeed-spike-k26}"
# Default to the production K2.6 NVFP4 snapshot. The patched BF16-KV
# sibling at /scratch/hf-cache-patched/k2.6-bf16kv/ is an alternative if
# the FP8-Q kernel-compile bug bites the baseline run.
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/scratch/hf-cache/models--nvidia--Kimi-K2-Thinking-NVFP4/snapshots/d427ad5a93317c6e3f18278da3d720ea4b5cc06a}"
RUN_DIR="/scratch/runs/k2.6-spike/phase3-verify-backend"
MAX_TOKENS="${MAX_TOKENS:-32}"
TP="${TP:-4}"

docker exec "$CONTAINER" bash -c "
    set -uo pipefail
    mkdir -p $RUN_DIR
    MODEL=$MODEL_SNAPSHOT
    if [[ ! -d \"\$MODEL\" ]]; then
        echo \"ERROR: model snapshot dir \$MODEL not found\" >&2
        exit 1
    fi
    echo \"[verify] using model: \$MODEL\"

    for VARIANT in base ts; do
        if [[ \$VARIANT == base ]]; then BACKEND=TRTLLM; else BACKEND=TOKENSPEED_MLA; fi
        OUT=$RUN_DIR/nsys-k26-\${VARIANT}-mtp1
        LOG=$RUN_DIR/nsys-k26-\${VARIANT}-stdout.log
        echo \"[verify] running \$VARIANT (--attention_backend \$BACKEND) -> \$OUT.nsys-rep\"

        PYTHONPATH=/workspace/TensorRT-LLM \
            nsys profile --force-overwrite=true -o \$OUT \
                --trace=cuda,nvtx \
                --cuda-trace-scope=system-wide \
                --wait=all \
                python -u /workspace/TensorRT-LLM/.claude_docs/tokenspeed-kimik25/scripts/minimal_generate.py \
                    --model_dir \"\$MODEL\" \
                    --tp_size $TP \
                    --max_tokens $MAX_TOKENS \
                    --attention_backend \$BACKEND \
                    > \"\$LOG\" 2>&1 || {
                echo \"  WARN: \$VARIANT run failed or partial; tail of log:\"
                tail -10 \"\$LOG\"
            }
        echo \"[verify]  \$VARIANT done; head of stdout:\"
        grep -E '\\[minimal\\]' \"\$LOG\" | head -6
    done

    echo
    echo \"=== Backend swap kernel-symbol diff ===\"
    BASE_FILE=$RUN_DIR/nsys-k26-base-mtp1.nsys-rep
    TS_FILE=$RUN_DIR/nsys-k26-ts-mtp1.nsys-rep
    if [[ ! -f \"\$BASE_FILE\" || ! -f \"\$TS_FILE\" ]]; then
        echo \"ERROR: nsys-rep file(s) missing — at least one variant failed.\" >&2
        ls -la $RUN_DIR/ >&2
        exit 1
    fi
    # Use Python csv module — C++ template kernel names contain commas
    # inside quoted fields, which trips awk -F','.
    BASE_KERNELS=\$(nsys stats --report cuda_gpu_kern_sum --format csv \
        \"\$BASE_FILE\" 2>/dev/null \
        | python3 -c \"
import csv, sys
rows = list(csv.reader(sys.stdin))
hdr = next((i for i, r in enumerate(rows) if r and r[0] == 'Time (%)'), None)
if hdr is None: sys.exit()
name_idx = rows[hdr].index('Name')
for r in rows[hdr+1:]:
    if r and len(r) > name_idx: print(r[name_idx])
\" | sort -u)
    TS_KERNELS=\$(nsys stats --report cuda_gpu_kern_sum --format csv \
        \"\$TS_FILE\" 2>/dev/null \
        | python3 -c \"
import csv, sys
rows = list(csv.reader(sys.stdin))
hdr = next((i for i, r in enumerate(rows) if r and r[0] == 'Time (%)'), None)
if hdr is None: sys.exit()
name_idx = rows[hdr].index('Name')
for r in rows[hdr+1:]:
    if r and len(r) > name_idx: print(r[name_idx])
\" | sort -u)
    if [[ \"\$BASE_KERNELS\" == \"\$TS_KERNELS\" ]]; then
        echo \"VERDICT: kernel sets IDENTICAL — backend swap did NOT fire.\"
        echo \"         Inspect the dispatch gate; likely config mismatch.\"
        exit 2
    else
        echo \"VERDICT: kernel sets DIFFER — swap likely fired. Differential symbols:\"
        echo \"--- baseline-only (TS-variant removed) ---\"
        comm -23 <(echo \"\$BASE_KERNELS\") <(echo \"\$TS_KERNELS\") | sed 's/^/  /' | head -20
        echo \"--- ts-only (added by variant) ---\"
        comm -13 <(echo \"\$BASE_KERNELS\") <(echo \"\$TS_KERNELS\") | sed 's/^/  /' | head -20
        echo
        echo \"Look for tokenspeed_mla / BlackwellMultiHeadLatentAttention* in the ts-only side.\"
        echo \"If present: proceed to Phase 4 (./run_bench.sh ../bench-config.yml).\"
    fi
"
