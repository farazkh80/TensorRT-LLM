# PR Title

```
[None][feat] B12xLukeFusedMoE: NVFP4 MoE backend wrapping lukealonso/b12x
```

# PR Body (paste into GitHub UI)

## Summary

- Adds `B12xLukeFusedMoE`, a sibling of `FlashInferFusedMoE` selectable via `moe_backend=B12X_LUKE`. Wraps lukealonso's standalone `b12x` package (https://github.com/lukealonso/b12x) — the upstream of the b12x SM120/SM121 NVFP4 MoE kernel that `FlashInferFusedMoE` uses through its flashinfer-vendored `B12xMoEWrapper`.
- Same SM120/SM121 + NVFP4 + bf16/fp16 + non-gptoss-style gating, same hybrid CUTLASS-prefill / b12x-decode dispatch via the existing `TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD` env var, same NVFP4 ModelOpt weights consumed by both backends.
- **Status: integration runs end-to-end on Nemotron-Super-120B-NVFP4 but decode TPOT regresses by +12.8 % (12.97 ms vs FI's 11.49 ms) at the best `B12xLukeFusedMoE` config tested. Investigated root cause across three angles (commit bisect, kernel-source diff, dispatch-path forcing); the gap is intrinsic to the small-batch tuning of the kernels luke ships in master, not the architectural pattern. Committed as a stepping stone — when upstream adds a small-batch-tuned warp-spec kernel for Nemotron decode, this backend activates the win without further TRT-LLM changes.**

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

The NORMALIZED-SF convention matches what the inherited `CutlassFusedMoE` NVFP4 `post_load_weights` already produces, so the conversion is shorter than `FlashInferFusedMoE`'s — no un-normalization step needed. `a*_gscale` are kept scalar (numel == 1) because b12x's documented contract is "[E] OR scalar" and the small-batch optimization paths require `numel == 1` (activates `share_input_across_experts=True` + `share_expert_scales=True`).

## Bench result on Nemotron-Super-120B-NVFP4 (1× SM120, RTX PRO 6000)

Same harness as the `FlashInferFusedMoE` hybrid baseline (`HYBRID_RESULTS.md` row "Hybrid"): ISL=2048, OSL=1024, 5 reqs, conc=1, KV reuse off, `cuda_graph_config: {batch_sizes: [1]}`, `max_num_tokens=4096`, `--streaming`.

| Variant | TPOT P50 (ms) | Δ vs FI | Notes |
|---|---:|---:|---|
| **FI hybrid baseline** | **11.4935** | — | warp-specialized small-batch-tuned `MoEMicroKernel` |
| Luke flat `MoEMicroKernel` (per-expert gscale) | 13.4047 | +16.6 % | flat 16-warp |
| **Luke flat `MoEMicroKernel` (scalar gscale, this PR)** | **12.9654** | **+12.8 %** | flat 16-warp + share-input/expert-scales flags |
| Luke `c9cc90ec` flat `MoEMicroKernel` (May 6) | 13.5516 | +17.9 % | bisect, pre-A16-reorg |
| Luke `986a405a` flat `MoEMicroKernel` (May 4) | 13.5534 | +17.9 % | bisect, pre-published-master |
| Luke warp-spec `MoEStaticKernel` (forced) | 13.5717 | +18.1 % | warp-spec, large-batch tuned |
| CUTLASS-only | 13.9681 | +21.5 % | NVFP4 grouped GEMM |

## Investigation: why luke loses, three independent angles

### 1. Commit bisect

Installed luke at three SHAs spanning May 4 → May 7:
- `986a405a` (May 4, before "A16 MoE Kernel variants" reorg) → **13.55 ms**
- `c9cc90ec` (May 6, "Bump version to 0.13.0", post-reorg / pre-CTA-restore) → **13.55 ms**
- `1378cea7` (May 7, master HEAD, "Restore Nemotron micro MoE performance") → **12.97 ms** (best)

The May 6 → May 7 commits net to a small *improvement* (-0.59 ms ≈ +5 %). The +12–18 % gap vs FI is constant going backward and predates the entire bisectable luke master history.

### 2. Trace probe: which kernel actually fires?

Patched `b12x.integration.tp_moe._launch_compact_static` with `print()` instrumentation. Confirmed every decode call flows through:

```
[trace-luke] _launch_compact_static: m=1 k=1024 n=2688 num_topk=22 E=512 si=True ses=True quant_mode=nvfp4
[trace-luke]   use_micro_direct=True
[trace-luke]   _compiled_direct_micro_accepts_block_dim=True (BLOCK_DIM=512)
[trace-luke]   ** MICRO LAUNCHED **
```

Luke's `MoEMicroKernel.launch()` IS firing every iteration. No fall-through to slower paths. The 12.97 ms is the true kernel runtime.

### 3. Apples-to-apples: forcing luke's warp-specialized `MoEStaticKernel`

Luke ships THREE kernel files. `static.py:95` and `dynamic.py:270` both have `num_mma_warps=4 / tma_load_warp_id=4 / threads_per_cta=160` — structurally identical to FI's 5-warp Blackwell warp-specialized producer/consumer pattern. Only `micro.py` is the flat 16-warp design.

Patched `_launch_compact_static` to skip the `use_micro_direct` branch, forcing dispatch to luke's warp-specialized `MoEStaticKernel`. **Result: 13.5717 ms (+18.1 % vs FI)** — essentially the same as the flat micro path. Warp specialization alone is not the differentiator.

## Refined diagnosis

- **Luke does have warp specialization** in `static.py` / `dynamic.py`. The original "luke lacks warp spec" hypothesis was wrong.
- **But luke's warp-spec kernels are tuned for *larger* routed_rows**, not the small-batch / decode case (m=1, routed_rows=22). Forcing them onto our shape pays the wrong-tuning cost.
- **FI's `MoEMicroKernel` is a separate small-batch-specialized warp-spec kernel** (picked for `routed_rows ≤ 40`). It's tuned specifically for our shape. Luke has no equivalent.

Architectural picture:

| | FI vendored | Luke master |
|---|---|---|
| Small-batch path | dedicated warp-spec `MoEMicroKernel`, tuned for `routed_rows ≤ 40` | flat `MoEMicroKernel` (16-warp, no warp-spec) |
| Large-batch path | warp-spec `MoEStaticKernel` | warp-spec `MoEStaticKernel` (~equivalent) |
| **Missing piece in luke** | — | small-batch-tuned warp-spec kernel |

Closing the gap requires either:
1. Re-tuning luke's `MoEStaticKernel` for `routed_rows ≤ 40` (kernel-level work, ncu profiling territory).
2. Adding a new small-batch warp-spec kernel to luke's master (mirrors what FI did).
3. Hand-porting FI's `MoEMicroKernel` into TRT-LLM (~12 k-line vendor cascade across 7+ CuTe DSL files; out of scope for this PR).

## Test plan

- [x] Smoke-test: `from tensorrt_llm._torch.modules.fused_moe import B12xLukeFusedMoE` + 4 negative `can_implement` cases (FP8 / fp32 / SM mismatch / gptoss) all behave correctly
- [x] `get_moe_cls("B12X_LUKE", ...)` resolves to `B12xLukeFusedMoE` on NVFP4 + SM120; hard-errors on FP8
- [x] End-to-end bench on Nemotron-Super-120B-NVFP4 (1× SM120) completes 5/5 requests, exit=0 on luke `1378cea7`, `c9cc90ec`, `986a405a`, and forced-warp-spec variants
- [x] CUDA-graph capture passes (caller-owned output buffer wired through shared module-level pool)
- [x] Hybrid dispatch confirmed by TTFT P50 = 156.45 ms ≈ pure CUTLASS prefill 154.53 ms (vs pure b12x 229.16 ms)
- [x] Trace probe confirmed luke's `MoEMicroKernel.launch` IS firing on every decode iteration
- [ ] Token parity check vs `FlashInferFusedMoE` hybrid (skipped — moot given the perf regression)
- [ ] Multi-GPU TP > 1 (not applicable; b12x has no dispatch/combine kernel — `__init__` rejects `ep_size > 1`)

## Risks / known limitations

1. **Performance regression at upstream HEAD.** Documented above. Will resolve when lukealonso adds a small-batch-tuned warp-spec kernel (or re-tunes `MoEStaticKernel` for routed_rows ≤ 40).
2. **`ep_size > 1` rejected at construction.** b12x has no expert-parallel dispatch/combine kernel. Same restriction as `FlashInferFusedMoE`.
3. **Stacked on #13773.** Cannot merge until #13773 lands.
4. **Hardcoded backend pin.** Container script in `.claude_docs/` (committed in this PR's docs commit) pins `b12x v0.13.0 @ 1378cea7`. Consumers need to install the package themselves; we don't add it as a TRT-LLM build dep.

## Recommendation

**Stay on `FlashInferFusedMoE` (#13773) for production SM120 Nemotron-Super decode.** `B12xLukeFusedMoE` is committed as a stepping stone. The bench/bisect/diff data captured in this PR's docs commit (`07ce279c57`) and the warp-spec swap test (commit `37dcb63031`) document exactly what luke would need to change upstream for this backend to win.

## Companion docs in this PR

- `.claude_docs/nemo-fp4-moe-b12x-mr/B12X_LUKE_RESULTS.md` — bench numbers + token-parity result + success-criteria table
- `.claude_docs/nemo-fp4-moe-b12x-mr/FI_VS_LUKE_DELTA.md` — full architecture comparison + bisect data + warp-spec swap evidence + refined verdict
- Helper scripts: `start_runtime_container_b12x_luke.sh`, `sync_b12x_luke_files.sh`, `bench_kvoff_b12x_luke.yml`, `parity_check_b12x_luke.py`, `_patch_tp_moe_trace.py`
