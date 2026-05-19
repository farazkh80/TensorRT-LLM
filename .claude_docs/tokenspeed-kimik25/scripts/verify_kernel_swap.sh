#!/usr/bin/env bash
# Phase 3 of the K2.6 plan (../runbook.md). THE GO/NO-GO GATE.
#
# Runs minimal_generate.py twice under nsys profile — once baseline
# (TLLM_TOKENSPEED_MLA=0), once variant (TLLM_TOKENSPEED_MLA=1) — and diffs
# the kernel name sets. Three outcomes:
#
#   - swap fires: variant has BlackwellMultiHeadLatentAttentionForward (or
#     similar tokenspeed_mla CuTe DSL kernel symbols) that the baseline
#     does not. Proceed to Phase 4 (run_bench.sh).
#
#   - swap silent: kernel sets identical → repeats the DSV3-Lite step 6
#     finding. K2.6 also bypasses run_mla_generation; spike patches are
#     dead code; need TokenSpeedMLAAttention backend class instead.
#
#   - crash / OOM: K2.6-specific; debug.
#
# NB: we use minimal_generate.py (not quickstart_advanced.py) because the
# NVIDIA Kimi-K2-Thinking-NVFP4 snapshot's custom tokenization_kimi.py
# silently hangs/fails at AutoTokenizer.from_pretrained even with
# trust_remote_code=True, leaving LLM.tokenizer=None and bombing
# _prepare_sampling_params. The minimal script uses skip_tokenizer_init=True
# + explicit end_id to bypass that path. We only need 32 generated tokens
# for nsys to capture MLA decode kernel symbols.
#
# nsys trace scope: cuda + nvtx only (NO osrt). osrt instrumentation
# slowed K2.6 first-load by ~10x in the initial attempt.
#
# Run time once cubins are warm: ~5-7 min per variant.

set -euo pipefail

CONTAINER="${CONTAINER:-tokenspeed-spike-k26}"
MODEL_PATH="${MODEL_PATH:-/scratch/hf-cache/models--nvidia--Kimi-K2-Thinking-NVFP4}"
SNAPSHOT_GLOB="$MODEL_PATH/snapshots/*"
RUN_DIR="/scratch/runs/k2.6-spike/phase3-verify"
MAX_TOKENS="${MAX_TOKENS:-32}"
# Phase 3 uses TP4 (matches bench-config.yml). Override for TP8 variants.
TP="${TP:-4}"

docker exec "$CONTAINER" bash -c "
    set -uo pipefail
    mkdir -p $RUN_DIR
    MODEL=\$(ls -d $SNAPSHOT_GLOB 2>/dev/null | head -1)
    if [[ -z \"\$MODEL\" ]]; then
        echo 'ERROR: no model snapshot found under $MODEL_PATH/snapshots/' >&2
        exit 1
    fi
    echo \"[verify] using model: \$MODEL\"

    for VARIANT in base ts; do
        if [[ \$VARIANT == base ]]; then TS=0; else TS=1; fi
        OUT=$RUN_DIR/nsys-k26-\${VARIANT}-mtp1
        LOG=$RUN_DIR/nsys-k26-\${VARIANT}-stdout.log
        echo \"[verify] running \$VARIANT (TLLM_TOKENSPEED_MLA=\$TS) -> \$OUT.nsys-rep\"

        TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 TLLM_TOKENSPEED_MLA=\$TS \
            nsys profile --force-overwrite=true -o \$OUT \
                --trace=cuda,nvtx \
                python -u /workspace/TensorRT-LLM/.claude_docs/tokenspeed-kimik25/scripts/minimal_generate.py \
                    --model_dir \"\$MODEL\" \
                    --tp_size $TP \
                    --max_tokens $MAX_TOKENS \
                    --attention_backend TRTLLM \
                    > \"\$LOG\" 2>&1 || {
                echo \"  WARN: \$VARIANT run failed or partial; tail of log:\"
                tail -10 \"\$LOG\"
            }
        echo \"[verify]  \$VARIANT done; head of stdout:\"
        grep -E '\\[minimal\\]' \"\$LOG\" | head -6
    done

    echo
    echo \"=== Phase 3 kernel-symbol diff ===\"
    BASE_FILE=$RUN_DIR/nsys-k26-base-mtp1.nsys-rep
    TS_FILE=$RUN_DIR/nsys-k26-ts-mtp1.nsys-rep
    if [[ ! -f \"\$BASE_FILE\" || ! -f \"\$TS_FILE\" ]]; then
        echo \"ERROR: nsys-rep file(s) missing — at least one variant failed.\" >&2
        ls -la $RUN_DIR/ >&2
        exit 1
    fi
    BASE_KERNELS=\$(nsys stats --report cuda_gpu_kern_sum --format csv \
        \"\$BASE_FILE\" 2>/dev/null \
        | awk -F',' 'NR>1 {print \$NF}' | sort -u)
    TS_KERNELS=\$(nsys stats --report cuda_gpu_kern_sum --format csv \
        \"\$TS_FILE\" 2>/dev/null \
        | awk -F',' 'NR>1 {print \$NF}' | sort -u)
    if [[ \"\$BASE_KERNELS\" == \"\$TS_KERNELS\" ]]; then
        echo \"VERDICT: kernel sets IDENTICAL — swap did NOT fire.\"
        echo \"         Same finding as DSV3-Lite spike step 6.\"
        echo \"         Action: skip Phase 4, escalate to TokenSpeedMLAAttention class.\"
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
