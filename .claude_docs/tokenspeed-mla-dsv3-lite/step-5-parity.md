# Step 5: Parity unit test

**Date:** 2026-05-13
**Experiment:** tokenspeed-mla-dsv3-lite
**GPU / SM:** B300 SXM6 AC / sm_103

## What ran

```bash
docker exec tokenspeed-spike bash -c '
    cd /workspace
    python -u -m pytest -v -s --no-header --noconftest -p no:cacheprovider \
        /workspace/TensorRT-LLM/tests/unittest/_torch/attention/test_tokenspeed_mla.py
'
```

Iterations needed before pytest ran: (1) `pip install parameterized` for the conftest, (2) `--noconftest` to skip the wider TRT-LLM conftest chain (needs `mako`, `tritongraph`, etc.), (3) **monkey-patch `tokenspeed_mla/mla_decode.py` to allocate an LSE tensor** — see "Issues" below.

## Result

13 tests collected (12 parity × `num_heads ∈ {16, 32}` × dtype ∈ {bf16, fp16} × {bs1_qlen1, bs4_qlen1_varlen, bs8_qlen4_spec}, plus 1 sinks rejection):

| Cases | Outcome |
|---|---|
| 5 | **PASSED** — bf16 parity, bs1_qlen1 + bs4_qlen1_varlen × H16+H32 |
| 6 | **SKIPPED** — FlashInfer baseline missing on B300 (`Missing TRTLLM-GEN kernel` for fp16 + bs8_qlen4 split-KV) |
| 2 | **FAILED** — bf16 parity diff on `bs8_qlen4_spec`-H16 and -H32 |
| 1 | **PASSED** — `test_tokenspeed_mla_rejects_sinks` (wrapper correctly rejects `sinks=...`) |

Numerical diff on the 2 failures (both bf16, BS=8, q_len=4, varlen seq_lens 256–512, num_heads ∈ {16, 32}):
- Mismatched elements: **2459 / 262144 (0.9%)**
- Max absolute diff: **0.33** at index `(1, 1, 10, 14)` (tolerance 0.05)
- Max relative diff: **1166×** at index `(6, 0, 3, 436)` (tolerance 0.005)

This is the **MTP/spec-decode regime** — exactly where TokenSpeed's `fold_sq_factor` re-orders queries into the head axis. The reduction order differs from FlashInfer's reference, so a numerical divergence is expected; whether the 0.33 abs max-diff matters in practice depends on downstream softmax/sampling.

## Issues / blockers

- **tokenspeed-mla 0.1.2 bug**: the BF16/FP16 decode kernel reinterprets `lse` unconditionally (`mla_decode_fp16.py:419`), but the wrapper `tokenspeed_mla_decode()` passes `lse=None` (compile-time and runtime). Crashes with `AttributeError: 'NoneType' object has no attribute 'iterator'`.
  - **Workaround in this container only**: monkey-patched 3 sites in `/usr/local/lib/python3.12/dist-packages/tokenspeed_mla/mla_decode.py`:
    1. Added `lse_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (sym_batch, sym_seq_q, sym_heads), stride_order=(2,1,0), assumed_align=4)` after `o_fake`.
    2. Replaced compile-time `None,  # lse (disabled)` with `lse_fake`.
    3. Replaced runtime `None,  # lse (disabled)` with `torch.empty((B, q_len, H), dtype=torch.float32, device=query.device)`.
  - Worth filing upstream against `lightseekorg/tokenspeed` — same bug in 0.1.1 and 0.1.2 on PyPI and in the local clone.
- **FlashInfer 0.6.x kernel coverage gap on sm_103**: the trtllm-gen cubin set lacks MLA decode for several DSV3-Lite × split-KV combinations (especially fp16 and certain `numTokensPerPage=64` configs). Test skips gracefully; not blocking.

## Findings worth reporting to Albert Di / Julien

1. TokenSpeed BF16 MLA decode passes parity vs FlashInfer on plain decode shapes (BS=1, BS=4×varlen, q_len=1) at DSv2-Lite and DSv3-Lite num_heads.
2. **TokenSpeed diverges numerically from FlashInfer in the spec-decode regime** (BS=8, q_len=4). Max abs 0.33, 0.9% elements off. Same regime as the headline 2× perf claim. Needs root-cause: `fold_sq_factor` reorder, multi-CTA split-KV reduction order, or accumulator dtype.
3. **`tokenspeed-mla 0.1.2` has a bug in its high-level wrapper** when `return_lse` is not exposed; the kernel always expects an LSE tensor. Patch is local; worth upstreaming.

## Next

Step 6 — E2E run of `quickstart_advanced.py` on DSV3-Lite NVFP4 with `TLLM_TOKENSPEED_MLA={0,1}` under `nsys profile`, then invoke **perf-nsight-systems** to confirm `tokenspeed_mla_decode` actually appears in the trace. **Pause for user confirmation before running.**
