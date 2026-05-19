# Step 6: E2E + nsys kernel-swap verify

**Date:** 2026-05-13
**Experiment:** tokenspeed-mla-dsv3-lite
**GPU / SM:** B300 SXM6 AC / sm_103
**Model:** DSV3-Lite NVFP4 (MoE-only quantized) — see step-1-env.md

## What ran

```bash
# 4 nsys runs: {base,ts} × {mtp1,mtp3}
for VARIANT in base ts; do for MTP in 1 3; do
  TLLM_TOKENSPEED_MLA=${0 or 1} nsys profile -o nsys-$VARIANT-mtp$MTP --trace=cuda,nvtx,osrt \
    python quickstart_advanced.py --model_dir <NVFP4_MOE_ONLY> \
        --attention_backend TRTLLM --moe_backend AUTO --kv_cache_dtype auto \
        --max_tokens 32 --prompt "..."   [+ --spec_decode_algo MTP --spec_decode_max_draft_len 3 if MTP=3]
done; done

# Then: stderr-probe rerun to confirm whether run_mla_generation was hit
```

## Result

### 1. E2E generation works on DSV3-Lite NVFP4 (MTP=1, both base + ts)

- `base-mtp1`: "there lived a young prince named Alexander. He was known for his bravery and intelligence, and he had a special gift for solving puzzles and riddles."
- `ts-mtp1`:   "there lived a young prince named Alexander. He was known for his bravery and intelligence, and the people of the kingdom loved him dearly. However, one day,"
- Accept rate 1.00 in both. 32 tokens in ~1.3 s. Model loads in ~11 s.

### 2. MTP=3 fails on this checkpoint (both base and ts)

```
KeyError: 'model.layers.30.self_attn.kv_a_proj_with_mqa.weight'
RuntimeError: Executor worker returned error
```

Checkpoint ships with `num_nextn_predict_layers=1` (one MTP layer at index 30). Forcing `--spec_decode_max_draft_len 3` makes the loader expect additional MTP layer weights that aren't in the checkpoint. **Not a TokenSpeed issue** — same failure with FlashInfer baseline. Matches the JIRA gotcha (model trained with MTP=1; forcing MTP=3 requires retrained weights or a draft model).

### 3. Kernel-swap verification: NEGATIVE — env-var swap is on the wrong code path

**The smoking gun**: both `nsys-base-mtp1.nsys-rep` and `nsys-ts-mtp1.nsys-rep` have an **identical set of 284 unique kernel names**. No `tokenspeed_mla` or `BlackwellMultiHeadLatent*` symbols anywhere in the variant trace.

To confirm, I injected an `[SPIKE]` stderr-print at the top of the patched `run_mla_generation` and reran the variant. The print never fired during 8-token generation — **the function was never called.**

What's actually doing MLA decode in the trace:

```
1.4%   fmhaSm103aKernel_QkvBfloat16OBfloat16HQk192HV128SeparateQkvCausalVarSeqQ128Kv128PersistentContext      210 calls   ← MLA context (Q=128, H=192=qk_nope+qk_rope)
0.6%   fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512PagedKvDenseP32VarSeqQ16Kv128PersistentSwapsAbForGen   120 calls   ← MLA decode (Q=16, HQk=576=kv_lora+qk_rope, HV=512=kv_lora)
0.5%   fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta128PagedKvDenseP32VarSeqQ16Kv128StaticSwapsAbForGen   1920 calls
```

These are **precompiled TRT-LLM-Gen FMHA cubins**, dispatched from TRT-LLM's C++ thop attention layer — NOT from `flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla` (the Python entry point our patch lives next to).

**Root cause:** for DSV3-Lite NVFP4 + sm_103, `TrtllmAttention.forward` dispatches MLA decode through the C++ thop path (`tensorrt_llm._v1.kernels...` direct cubin launch), bypassing the Python `run_mla_generation` flashinfer wrapper entirely. The wrapper-level env-var swap only intercepts the *flashinfer-routed* MLA decode path, which this model/config doesn't take.

To actually swap kernels for this model, the swap needs to happen **inside the C++ thop dispatch** — either by adding TokenSpeed as a recognized backend at the C++ level, or by replacing the `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512*` runner with TokenSpeed's CuTe DSL kernel. That's a much bigger integration than the spike scope.

## Files in this folder

- `nsys-base-mtp1.nsys-rep` (72 MB) — FlashInfer/TRT-LLM-Gen MLA baseline trace
- `nsys-ts-mtp1.nsys-rep` (71 MB) — same baseline trace (TokenSpeed swap silently no-op'd)
- `nsys-base-mtp3.nsys-rep` (13 MB) — load + failure (KeyError MTP=3)
- `nsys-ts-mtp3.nsys-rep` (13 MB) — same failure
- `patches/apply_patches.py` — three patches for utils.py, trtllm_gen.py, tokenspeed_mla/mla_decode.py
- `patches/fix_probe.py` — clean stderr probe injection

## Issues / blockers

1. **Spike's env-var swap is on the wrong code path for this model+arch.** Discovered only after measurement; the wrapper-level swap matched flashinfer's Python API exactly but didn't intercept the TRT-LLM-Gen direct-cubin MLA path. Documented as a finding.
2. **No usable A/B perf data for this config.** Both base and variant ran the *same* baseline kernel, so any time difference would be noise.

## Next

Step 7 (`trtllm-bench` A/B) would also produce noise on this config for the same reason — the env-var swap doesn't fire. Two options:

a. **Skip step 7** for this model. Land the spike code + this finding as the deliverable; recommend Albert Di/Julien target the C++ thop MLA dispatch.
b. **Test on a different config** that *does* use the flashinfer-Python MLA path. Need to identify one — possibly DSV3 full or a model with explicit `attn_backend=FLASHINFER` rather than the TRTLLM C++ dispatch.

**Pause for user decision before proceeding.**
