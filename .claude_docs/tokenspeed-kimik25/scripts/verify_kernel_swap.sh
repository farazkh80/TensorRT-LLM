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
# NB: snapshot dir overridable. The Phase-3-fixup default points at the
# *patched* sibling snapshot (kv_cache_quant_algo removed from
# hf_quant_config.json), which makes K2.6 NVFP4 load with BF16 KV instead
# of the default FP8 KV — required for the spike's run_mla_generation
# swap to be reachable. Override to /scratch/hf-cache/models--... and
# KV_DTYPE=auto for the original FP8 KV behavior.
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/scratch/hf-cache-patched/k2.6-bf16kv}"
RUN_DIR="/scratch/runs/k2.6-spike/phase3-verify"
MAX_TOKENS="${MAX_TOKENS:-32}"
# Phase 3 uses TP4 (matches bench-config.yml). Override for TP8 variants.
TP="${TP:-4}"
# KV cache dtype override. With the patched snapshot, "auto" is correct
# (no fp8 forcing in quant config). Valid: auto | fp8 | nvfp4.
KV_DTYPE="${KV_DTYPE:-auto}"

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
                    --kv_cache_dtype $KV_DTYPE \
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
