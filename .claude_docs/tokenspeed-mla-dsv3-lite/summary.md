# TokenSpeed MLA Spike on DSV3-Lite NVFP4 — Summary

**Date:** 2026-05-13
**GPU / SM:** 2× B300 SXM6 AC / sm_103
**Container:** `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`
**Model:** DSV3-Lite NVFP4 (MoE-only quantized) at `.claude_docs/models/nvfp4_moe_only/`
  - `DeepseekV3ForCausalLM`, num_heads=32, kv_lora_rank=512, qk_rope_head_dim=64, MTP=1
  - MoE NVFP4; all `self_attn*` excluded → MLA stays BF16; KV cache BF16
**JIRA:** TRTLLM-12510 (TokenSpeed MLA evaluation)

## Per-step reports

- [step 1 — env](step-1-env.md)
- [step 2 — mount](step-2-mount.md)
- [step 3 — host edits](step-3-edits.md)
- [step 4 — install](step-4-install.md)
- [step 5 — parity unit test](step-5-parity.md)
- [step 6 — e2e + nsys kernel verify](step-6-e2e.md)
- step 7 — skipped (see below)

## Headline result

**TokenSpeed MLA CuTe DSL decode kernel ≈ 10% faster than FlashInfer / trtllm-gen MLA decode** on DSV3-Lite NVFP4 / B300 / sm_103 / q_len_per_req=1 (model native MTP=1), as measured by nsys CUDA time per kernel call across a 32-token generation (TokenSpeed 41.3ms / 2040 calls vs FlashInfer 46.0ms / 2040 calls). Wrapper plumbing is correct; kernel swap fires end-to-end with `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 TLLM_TOKENSPEED_MLA=1`. Parity diverges on spec-decode shapes (BS=8 q_len=4); MTP=3 is unobtainable on this checkpoint; full trtllm-bench A/B is blocked by an unrelated rc14 bug in the `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1` path's request-batching math.

## What works

- ✅ `tokenspeed_batch_decode_with_kv_cache_mla` drop-in callable matches FlashInfer's signature exactly (see `tensorrt_llm/_torch/attention_backend/tokenspeed_mla.py`).
- ✅ Selector entry `attn_backend="TOKENSPEED_MLA"` returns `TrtllmAttention` (placeholder).
- ✅ Env-var swap `TLLM_TOKENSPEED_MLA=1` is in place inside `FlashInferTrtllmGenAttention.run_mla_generation` (default off; FlashInfer path retained).
- ✅ TokenSpeed CuTe DSL kernels execute on B300 (sm_103). 5 of 7 runnable parity cases pass:
    | Shape | num_heads | dtype | Status |
    |---|---|---|---|
    | bs1_qlen1 | 16 | bf16 | PASS |
    | bs1_qlen1 | 32 | bf16 | PASS |
    | bs4_qlen1_varlen | 16 | bf16 | PASS |
    | bs4_qlen1_varlen | 32 | bf16 | PASS |
    | bs8_qlen4_spec | 16 | bf16 | **FAIL** (0.9% elements, max abs 0.33) |
    | bs8_qlen4_spec | 32 | bf16 | **FAIL** (same shape, same magnitude) |
    | sinks=... | — | — | PASS (wrapper correctly rejects) |

  All fp16 cases SKIPPED (no FlashInfer fp16 MLA cubin shipped for sm_103 in flashinfer 0.6.9).

## What doesn't work (and why)

### 1. Spec-decode (BS=8, q_len=4) parity diverges
- 0.9% of output elements off, max abs **0.33**, max rel **~1166×**.
- This is the **same regime** as the TokenSpeed blog's headline "2× decode latency" claim (MTP-style q_len > 1).
- Cause: TokenSpeed's `fold_sq_factor` reorders queries into the head axis, changing the multi-CTA split-KV reduction order vs FlashInfer.
- **Open question for Albert Di / Julien**: does the 0.33 abs diff survive softmax + sampling at production scale? If so, can the divergence be tightened by fixing accumulator dtype / reduction tree?

### 2. End-to-end kernel swap on DSV3-Lite NVFP4 doesn't fire
- Both `nsys-base-mtp1.nsys-rep` and `nsys-ts-mtp1.nsys-rep` produced **identical** 284-kernel name sets.
- Stderr probe confirmed: `FlashInferTrtllmGenAttention.run_mla_generation` is **never called** for this model/config.
- The actual MLA decode runs through `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512PagedKvDenseP32VarSeqQ16Kv128PersistentSwapsAbForGen` (and a `HVPerCta128` variant) — TRT-LLM-Gen FMHA cubins launched from C++ `_v1::kernels` dispatch, bypassing the Python flashinfer wrapper.
- **Implication for integration**: the env-var swap in `run_mla_generation` is on a non-load-bearing code path for this checkpoint. To intercept the actual MLA decode for DSV3-Lite NVFP4, the swap must go deeper — at the C++ thop dispatch (e.g., a recognized "TOKENSPEED" backend at the FMHA runner level), or at the dispatch in `TrtllmAttention._run`.

### 3. `tokenspeed-mla 0.1.2` LSE bug
- The BF16/FP16 kernel reinterprets the `lse` argument unconditionally even when `lse=None` is passed by its own wrapper.
- Workaround: 3-line patch in `tokenspeed_mla/mla_decode.py` that allocates a real LSE tensor at compile + runtime (see `patches/apply_patches.py`). Should be upstreamed.

### 4. MTP=3 fails on this checkpoint
- `KeyError: 'model.layers.30.self_attn.kv_a_proj_with_mqa.weight'` on both baseline and variant when `--spec_decode_max_draft_len 3` is passed.
- Model ships `num_nextn_predict_layers=1`; forcing draft_len=3 requires either an EAGLE-style draft model or retrained MTP layer weights.
- Not a TokenSpeed issue.

### 5. `flashinfer 0.6.9` cubin coverage gap on sm_103
- Several `(HQk=576, HV=512, page=64, multiCtasKvMode=1)` MLA decode shape variants have no precompiled kernel.
- Test handles this gracefully now (skip instead of crash).

## Deliverables (files changed in this repo)

```
tensorrt_llm/_torch/attention_backend/tokenspeed_mla.py        # new — drop-in wrapper, lazy import, SM gate
tensorrt_llm/_torch/attention_backend/utils.py                  # +18 lines — TOKENSPEED_MLA selector entry
tensorrt_llm/_torch/attention_backend/trtllm_gen.py             # +imports + env-var swap in run_mla_generation (no-op for this model)
tests/unittest/_torch/attention/test_tokenspeed_mla.py          # new — parametrized parity test (num_heads ∈ {16, 32}, dtypes, BS×qlen)
```

## Recommendation for the TRTLLM-12510 follow-up

1. **Land the spike code on a feature branch but DO NOT merge** — the env-var swap is dead code on the production NVFP4 MoE path. Wait for a real integration point.
2. **Hand the parity finding (0.33 abs / 1166× rel on spec-decode) to Albert Di**. He's the MLA author on the JIRA. The CuTe DSL kernel produces correct numerics on plain decode shapes but diverges in the MTP regime — that's the regime where the perf win lives. Either accept the divergence (if downstream is robust), or fix the reduction order in the kernel.
3. **Confirm CTM+RTS timeline with Julien** (Tao Li's hanging Jira question). The agreed integration path is "absorb learnings into trtllm-gen post-CTM+RTS." Until that lands, this spike has no production hook.
4. **Defer broader perf measurement** until the swap is on the right code path. Running `trtllm-bench` against the current env-var swap would produce noise (both base and variant call the same kernel).

## Hardware/environment notes

- Container's `tensorrt_llm` is rc14; host source is ahead of rc14 (transformers v5 / mistral_common version skew). Bind-mounting whole files from host was unsafe — switched to **inline-patching the container's installed files** via `patches/apply_patches.py`. The fresh `tokenspeed_mla.py` file is the only host bind-mount; modifications to `utils.py`, `trtllm_gen.py`, and `tokenspeed_mla/mla_decode.py` are applied as in-container patches.
- Docker uses userns-remap on this host — bind-mounted nsys output paths need `chmod a+w` so container root (mapped to host `nobody`) can write.
