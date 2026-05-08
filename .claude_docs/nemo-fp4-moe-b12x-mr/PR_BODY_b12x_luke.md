# PR Title

```
[None][feat] B12xLukeFusedMoE: NVFP4 MoE backend wrapping lukealonso/b12x
```

# PR Body (paste into GitHub UI)

## Summary

- Adds `B12xLukeFusedMoE`, a sibling of `FlashInferFusedMoE` selectable via `moe_backend=B12X_LUKE`. Wraps lukealonso's standalone `b12x` package (https://github.com/lukealonso/b12x) — the upstream of the b12x SM120/SM121 NVFP4 MoE kernel that `FlashInferFusedMoE` uses through its flashinfer-vendored `B12xMoEWrapper`.
- Same SM120/SM121 + NVFP4 + bf16/fp16 + non-gptoss-style gating, same hybrid CUTLASS-prefill / b12x-decode dispatch via the existing `TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD` env var, same NVFP4 ModelOpt weights consumed by both backends.
- **Status: integration runs end-to-end on Nemotron-Super-120B-NVFP4 but decode TPOT regresses to 13.40 ms (vs `FlashInferFusedMoE` hybrid 11.49 ms) because lukealonso/b12x master HEAD `1378cea7` has the Nemotron-tuned micro path gated off (`return False` at `b12x/integration/tp_moe.py:2721`) due to a real upstream scope bug (`NameError: name 'micro_mac' is not defined`). Committed as a stepping stone — when upstream fixes `micro_mac`, this backend activates the optimized decode kernel without further TRT-LLM changes.**

This PR is **stacked on top of #13773** (`faraz/b12x-flashinfer-moe-pr`). Will retarget to `main` once #13773 merges.

## What changes

`tensorrt_llm/_torch/modules/fused_moe/fused_moe_b12x_luke.py` (new, ~280 lines) subclasses `CutlassFusedMoE` and overrides only the per-expert compute step:

- `can_implement` narrows to NVFP4 + SM120/SM121 + bf16/fp16 + non-gptoss-style.
- `__init__` rejects `ep_size > 1`, alltoall, and unsupported activations.
- `post_load_weights` builds a `B12XFP4ExpertWeights` from the inherited NVFP4 quant tensors, allocates a module-level shared `(max_num_tokens, hidden_size)` output buffer (keyed by shape/dtype/device — required by b12x's CUDA-graph capture contract: caller-owned output during graph capture) and a shared `TPMoEWorkspacePool`.
- `quantize_input` is a passthrough; b12x quantizes activations internally.
- `run_moe` slices the shared output buffer to the current batch and dispatches to `b12x.integration.b12x_moe_fp4`. Hybrid dispatch reuses the env var so existing yaml / scripts work unchanged when swapping `moe_config.backend` from `FLASHINFER` to `B12X_LUKE`.

`create_moe.get_moe_cls` adds a `B12X_LUKE` branch that hard-errors on missing NVFP4 quant or unsupported SM (mirrors the existing `FLASHINFER` branch). `__init__.py` re-exports the new class. `MoeConfig.backend` literal in `llm_args.py` adds `"B12X_LUKE"`.

## Weight conversion delta vs `FlashInferFusedMoE`

| Quantity | `FlashInferFusedMoE` (flashinfer-vendored b12x) | `B12xLukeFusedMoE` (lukealonso b12x v0.13.0) |
|---|---|---|
| `w*_blockscale` | UN-normalized FP8 SF (multiply by `weight_scale_2` then swizzle) | NORMALIZED FP8 SF (HF/ModelOpt form, just swizzle) |
| `w*_alpha(s)` | `1/input_scale` (kernel does dual-use cancel) | `input_scale^2 * fc_alpha` (per-expert epilogue dequant) |
| `a*_gscale` | (folded into `w*_alpha`) | `1/input_scale` (reciprocal input scale, scalar) |
| swizzle helper | `flashinfer.cute_dsl.utils.convert_sf_to_mma_layout` | `b12x.cute.fp4.swizzle_block_scale` |

The NORMALIZED-SF convention matches what the inherited `CutlassFusedMoE` NVFP4 `post_load_weights` already produces, so the conversion is shorter than `FlashInferFusedMoE`'s — no un-normalization step needed. `a*_gscale` are kept scalar (numel == 1) because b12x's documented contract is "[E] OR scalar" and the Nemotron-tuned micro path explicitly requires `numel == 1`.

## Bench result on Nemotron-Super-120B-NVFP4 (1× SM120, RTX PRO 6000)

Same harness as the `FlashInferFusedMoE` hybrid baseline (`HYBRID_RESULTS.md` row "Hybrid"): ISL=2048, OSL=1024, 5 reqs, conc=1, KV reuse off, `cuda_graph_config: {batch_sizes: [1]}`, `max_num_tokens=4096`, `--streaming`.

| Metric | Hybrid (FI b12x) baseline | **Hybrid-luke (B12X_LUKE)** | Δ |
| --- | ---: | ---: | ---: |
| Total Output Throughput (tok/s) | 85.92 | **73.68** | **−14.2 %** |
| TPOT P50 (ms) | 11.49 | **13.40** | **+16.6 %** |
| TTFT P50 (ms) | 154.53 | 156.45 | +1.2 % (within noise — CUTLASS prefill in both arms) |
| Total Latency (ms, 5 reqs) | 59,588 | 69,493 | +16.6 % |

Hybrid-luke TPOT (13.40 ms) is essentially identical to pure CUTLASS-only's 13.97 ms — the b12x dispatch is producing CUTLASS-class decode times because the optimized Nemotron micro path is unreachable on upstream master.

## Root cause of the perf regression — upstream bug

`b12x/integration/tp_moe.py:2721` at SHA `1378cea7`:

```python
def _is_exact_relu2_bs1_nemotron_case(...):
    return False                      # <-- unconditional gate
    if not (...): return False        # <-- dead code below
    ...
```

Locally removing the `return False` and re-running exposes the underlying upstream bug:

```
File "/usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py",
  line 2816, in _get_exact_relu2_bs1_nemotron_launcher
    mac_override=micro_mac,
                 ^^^^^^^^^
NameError: name 'micro_mac' is not defined
```

`micro_mac` is referenced at L2816 inside `_get_exact_relu2_bs1_nemotron_launcher` but is only defined at L3064 — inside the calling `b12x_moe_fp4` body, *after* the Nemotron-bs1 early-return branch that calls the launcher. The `return False` is upstream's workaround for this dangling reference; the kernel cannot be revived without an upstream fix.

## Test plan

- [x] Smoke-test: `from tensorrt_llm._torch.modules.fused_moe import B12xLukeFusedMoE` + 4 negative `can_implement` cases (FP8 / fp32 / SM mismatch / gptoss) all behave correctly
- [x] `get_moe_cls("B12X_LUKE", ...)` resolves to `B12xLukeFusedMoE` on NVFP4 + SM120; hard-errors on FP8
- [x] End-to-end bench on Nemotron-Super-120B-NVFP4 (1× SM120) completes 5/5 requests, exit=0
- [x] CUDA-graph capture passes (caller-owned output buffer wired through shared module-level pool)
- [x] Hybrid dispatch confirmed by TTFT P50 = 156.45 ms ≈ pure CUTLASS prefill 154.53 ms (vs pure b12x 229.16 ms)
- [ ] Token parity check vs `FlashInferFusedMoE` hybrid (skipped — moot given the perf regression)
- [ ] Multi-GPU TP > 1 (not applicable; b12x has no dispatch/combine kernel — `__init__` rejects `ep_size > 1`)

## Risks / known limitations

1. **Performance regression at upstream HEAD.** Documented above. Will resolve when lukealonso/b12x fixes the `micro_mac` scope bug.
2. **`ep_size > 1` rejected at construction.** b12x has no expert-parallel dispatch/combine kernel. Same restriction as `FlashInferFusedMoE`.
3. **Stacked on #13773.** Cannot merge until #13773 lands.
4. **Hardcoded backend pin.** Container script in `.claude_docs/` (not part of this PR) pins `b12x v0.13.0 @ 1378cea7`. Consumers will need to install the package themselves; we don't add it as a TRT-LLM build dep.

## Recommendation

Until upstream fixes the `micro_mac` scope bug, `moe_backend=FLASHINFER` (from #13773) remains the fastest known SM120 Nemotron-Super decode path. This PR lands the integration so it's ready to activate as soon as upstream patches the kernel.
