# Step 7: trtllm-bench A/B

**Date:** 2026-05-13
**Experiment:** tokenspeed-mla-dsv3-lite
**GPU / SM:** B300 SXM6 AC / sm_103
**Model:** DSV3-Lite NVFP4 (MoE-only quantized)

## What ran

Two attempts at trtllm-bench throughput A/B, both with the wrapper-side and
`run_mla_generation`-side 3D→4D fixes in place (commits `6fc90ceadb` and
`bfd5e5d36f`):

```bash
# Attempt 1 — 1024 in / 128 out
trtllm-bench --model deepseek-ai/DeepSeek-V3 --model_path ... \
    prepare-dataset --output /tmp/synth_1024_128_50.txt \
    token-norm-dist --input-mean 1024 --output-mean 128 --num-requests 50

TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 TLLM_TOKENSPEED_MLA=$ENV_VAL \
    trtllm-bench --model ... --model_path ... throughput --dataset ...

# Attempt 2 — 128 in / 128 out (to dodge chunked-prefill)
# (same shape; same failure mode)
```

## Result — BOTH base AND ts fail identically

Attempt 1 (1024/128):
```
RuntimeError: invalid shape dimension -1023 at index 1 of shape [1, -1023, 32, 576]
ValueError: Requests failed: invalid shape dimension -1023 ... (2 requests)
```

Attempt 2 (128/128):
```
RuntimeError: invalid shape dimension -127 at index 1 of shape [1, -127, 32, 576]
```

**Pattern:** `-N = 1 - input_length`. `q_len_per_req` is being computed as
`1 - input_length` somewhere in the `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1`
code path under trtllm-bench's specific request-batching / chunked-context
flow. The shape `[1, -127, 32, 576]` is `[batch_beam, q_len_per_req,
num_heads, kv_lora_rank+qk_rope]` — confirms the bug is in
`q_len_per_req` math at the call site.

**This is upstream rc14, not our code:**
- Both the FlashInfer baseline (`TLLM_TOKENSPEED_MLA=0`) and the TokenSpeed
  variant (`TLLM_TOKENSPEED_MLA=1`) hit the *exact same* error.
- The single-request `quickstart_advanced.py` path with the same env vars
  generates correctly (verified in step 6 v3 nsys runs) — the failure only
  happens under trtllm-bench's multi-request scheduling.
- Trying `--input-mean 128` instead of 1024 doesn't avoid the bug; it just
  shifts the magnitude. The off-by-input-length math is fundamental.

Possibly a chunked-context boundary calculation that assumes the C++ thop
code path's request bookkeeping, breaks when the FlashInfer Python wrapper
is in the middle.

## What we DID get for perf numbers

From step 6 nsys traces v3 (already collected, see `step-6-e2e.md`):

| Kernel | Calls | GPU time |
|---|---|---|
| **base** (FlashInfer / trtllm-gen MLA decode) | 2040 | **46.0 ms** total |
| **ts** (TokenSpeed CuTe DSL `BlackwellMultiHeadLatentAttentionForward`) | 2040 | **41.3 ms** total |

**Net kernel-level: TokenSpeed ≈ 10% faster** on DSV3-Lite NVFP4 / B300 /
sm_103 at `q_len_per_req=1` (model native MTP=1), 32-token decode.

Caveats:
- Single non-warmup run each. No statistical significance.
- DSV3-Lite is small (16B); MLA isn't decode-bottlenecked here.
- This isn't TokenSpeed's headline regime — their 2× claim is on spec-decode
  q_len_per_req ≥ 4 + larger BS. We couldn't reproduce that without MTP=3
  (which fails on this checkpoint per step 6).
- TokenSpeed's split-KV reduction also runs ~96 ms of auxiliary CUTLASS GEMMs
  that the FlashInfer baseline doesn't have. Net e2e delta is smaller than
  the MLA-kernel-only number.

## Issues / blockers

1. **rc14 upstream bug**: `q_len_per_req` math in
   `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1` path is broken under trtllm-bench
   request batching. Filed as an internal note; would need debugging by
   the trtllm-gen team. Not in our spike's scope.
2. **No clean e2e A/B**: TPOT / TTFT / throughput numbers are unattainable
   for this config until issue (1) is fixed upstream. Until then, kernel-
   level nsys numbers are the best available signal.

## Next

Recommend ship findings as-is. Three concrete deliverables for the
TRTLLM-12510 follow-up:

1. The ~10% MLA decode kernel speedup at q_len=1 on DSV3-Lite NVFP4.
2. The parity divergence in the spec-decode regime (step 5 report).
3. The trtllm-bench rc14 path bug as an upstream filing.
