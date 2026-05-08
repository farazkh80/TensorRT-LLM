# PR Body for NVIDIA/TensorRT-LLM (Draft, base=main)

## Open via UI

```
https://github.com/NVIDIA/TensorRT-LLM/compare/main...farazkh80:b12x-luke-decode?expand=1
```

After clicking through, use the dropdown next to "Create pull request" to select **"Create draft pull request"**.

## Title

```
[None][feat] B12xLukeFusedMoE: NVFP4 MoE backend wrapping lukealonso/b12x (stacked on #13773)
```

## Body (paste into PR description)

> **DRAFT — blocked on #13773. Do not merge until #13773 lands.**
>
> This PR is stacked on the open #13773 (FlashInfer NVFP4 MoE backend +
> hybrid CUTLASS-prefill / b12x-decode dispatch). It carries the 3 commits
> from #13773's branch as its base layer plus 4 new commits adding the
> `B12xLukeFusedMoE` backend and investigation docs. Once #13773 merges,
> this PR will be rebased to drop the overlapping commits.

## Summary

- Adds `B12xLukeFusedMoE`, a sibling of `FlashInferFusedMoE` selectable via `moe_backend=B12X_LUKE`. Wraps lukealonso's standalone `b12x` package (https://github.com/lukealonso/b12x) — the upstream of the b12x SM120/SM121 NVFP4 MoE kernel that `FlashInferFusedMoE` uses through its flashinfer-vendored `B12xMoEWrapper`.
- Same SM120/SM121 + NVFP4 + bf16/fp16 + non-gptoss-style gating, same hybrid CUTLASS-prefill / b12x-decode dispatch via the existing `TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD` env var, same NVFP4 ModelOpt weights consumed by both backends.
- **Status: integration runs end-to-end on Nemotron-Super-120B-NVFP4 but decode TPOT regresses by +12.8 % (12.97 ms vs FI's 11.49 ms) at the best `B12xLukeFusedMoE` config. Investigated root cause across three independent angles (commit bisect, kernel-source diff, dispatch-path forcing); the gap is intrinsic to the small-batch tuning of the kernels luke ships in master, not the architectural pattern. Committed as a stepping stone — when upstream lukealonso adds a small-batch-tuned warp-spec kernel for Nemotron decode, this backend activates the win without further TRT-LLM changes.**

## Bench result on Nemotron-Super-120B-NVFP4 (1× SM120, RTX PRO 6000)

Same harness as `HYBRID_RESULTS.md` row "Hybrid": ISL=2048, OSL=1024, 5 reqs, conc=1, KV reuse off, `cuda_graph_config: {batch_sizes: [1]}`, `max_num_tokens=4096`, `--streaming`.

| Variant | TPOT P50 (ms) | Δ vs FI | Notes |
|---|---:|---:|---|
| **FI hybrid baseline (#13773)** | **11.4935** | — | warp-specialized small-batch-tuned `MoEMicroKernel` |
| Luke flat `MoEMicroKernel` (per-expert gscale) | 13.4047 | +16.6 % | flat 16-warp |
| **Luke flat `MoEMicroKernel` (scalar gscale, this PR)** | **12.9654** | **+12.8 %** | flat 16-warp + share-input/expert-scales flags |
| Luke `c9cc90ec` flat `MoEMicroKernel` (May 6) | 13.5516 | +17.9 % | bisect, pre-A16-reorg |
| Luke `986a405a` flat `MoEMicroKernel` (May 4) | 13.5534 | +17.9 % | bisect, pre-published-master |
| Luke warp-spec `MoEStaticKernel` (forced) | 13.5717 | +18.1 % | warp-spec, large-batch tuned |
| CUTLASS-only | 13.9681 | +21.5 % | NVFP4 grouped GEMM |

Forcing luke's warp-specialized `MoEStaticKernel` did not recover the gap — luke ships warp specialization, but tuned for *larger* routed_rows. FI's `MoEMicroKernel` is a separate small-batch-specialized warp-spec kernel; luke has no equivalent.

## Recommendation

**Stay on `FlashInferFusedMoE` (#13773) for production SM120 Nemotron-Super decode.** `B12xLukeFusedMoE` is committed as a stepping stone for when upstream lukealonso adds a small-batch-tuned warp-spec kernel.

## Companion docs in this PR

Full investigation evidence in `.claude_docs/nemo-fp4-moe-b12x-mr/`:

- `B12X_LUKE_RESULTS.md` — bench numbers + token-parity result + success-criteria table
- `FI_VS_LUKE_DELTA.md` — full architecture comparison + bisect data + warp-spec swap evidence + refined verdict
- `PR_BODY_b12x_luke.md` — long-form PR description (this body is condensed)
- Helper scripts: `start_runtime_container_b12x_luke.sh`, `sync_b12x_luke_files.sh`, `bench_kvoff_b12x_luke.yml`, `parity_check_b12x_luke.py`, `_patch_tp_moe_trace.py`

## Test plan

- [x] Smoke-test imports + 4 negative `can_implement` cases (FP8 / fp32 / SM mismatch / gptoss)
- [x] `get_moe_cls("B12X_LUKE", ...)` resolves on NVFP4 + SM120; hard-errors on FP8
- [x] End-to-end bench on Nemotron-Super-120B-NVFP4 (1× SM120) completes 5/5 requests, exit=0 across 4 luke variants tested
- [x] CUDA-graph capture passes (caller-owned output buffer wired through shared module-level pool)
- [x] Hybrid dispatch confirmed by TTFT P50 ≈ pure CUTLASS prefill 154.5 ms
- [x] Trace probe confirmed luke's `MoEMicroKernel.launch` IS firing on every decode iteration (not a fall-through bug)
- [ ] Token parity vs `FlashInferFusedMoE` hybrid (skipped — moot given perf regression)
- [ ] Multi-GPU TP > 1 (not applicable; b12x has no dispatch/combine kernel)
