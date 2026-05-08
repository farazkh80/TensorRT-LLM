# lukealonso/b12x decode kernel — measured results

Branch: `b12x-luke-decode` cut from `b12x-hybrid` (commit `ac436984e5`).
Date: 2026-05-08.
Bench log: `/home/farazkh_scratch/logs/b12x_luke_hybrid_20260508_061327.log`.

## TL;DR — the new kernel is **slower**, not faster

| Metric | Hybrid (FI b12x decode) baseline | **Hybrid-luke (b12x.integration decode)** | Δ |
| --- | ---: | ---: | ---: |
| Total Output Throughput (tok/s) | 85.92 | **73.68** | **−14.2 %** |
| TPOT P50 (ms) | 11.49 | **13.40** | **+16.6 %** |
| TPOT MIN/MAX (ms) | 11.4884 / 11.5187 | 13.3982 / 13.4786 | +16.7 % / +17.0 % |
| TTFT P50 (ms) | 154.53 | 156.45 | +1.2 % (within noise — CUTLASS prefill in both arms) |
| TTFT MAX (ms) | 161.22 | 230.28 | first-request cold-start outlier (same shape as `HYBRID_RESULTS.md`'s autotuner-cold first request) |
| Total Latency (ms, 5 reqs) | 59,588 | 69,493 | +16.6 % |

Hybrid-luke TPOT (13.40 ms) is essentially **identical to pure CUTLASS-only's 13.97 ms TPOT** from `HYBRID_RESULTS.md` — i.e. the b12x dispatch is producing CUTLASS-class decode times, *not* the b12x-class decode times the older flashinfer-vendored `B12xMoEWrapper` achieves.

## Root cause — the Nemotron micro path is disabled in upstream

In `b12x/integration/tp_moe.py` at SHA `1378cea7` (commit "Restore Nemotron micro MoE performance", 2026-05-07, the most recent commit on `master`):

```python
def _is_exact_relu2_bs1_nemotron_case(
    *, activation, a, w1_fp4, a1_gscale, a2_gscale, w2_fp4, topk_weights, topk_ids,
) -> bool:
    return False                  # ← UNCONDITIONAL EARLY RETURN
    if not (                      # dead code below
        activation == "relu2"
        and a.shape[0] == 1
        and a.shape[1] == 1024
        and w1_fp4.shape == (512, 2688, 512)
        and w2_fp4.shape == (512, 1024, 1344)
        ...
```

Despite the commit title, the upstream `master` HEAD has the Nemotron-tuned `_launch_exact_relu2_bs1_nemotron` fast path **gated off** by an unconditional `return False`. The dispatcher at `b12x_moe_fp4` L3198 therefore never enters this branch and falls through to the generic static / dynamic kernel — which on Nemotron-Super-120B at m=1 produces ~CUTLASS-class TPOT.

Inspecting the prior commit `ff205cf0` ("Restore micro MoE CTA warp count", same day) shows the same `return False` already in place — so the gate is older than the recent "Restore" commits. Everything in those commits is changes to dead code paths or to the `dynamic` kernel that we end up hitting instead.

## Did the integration itself work?

Yes — every other layer of the bench is healthy:

- **`B12xLukeFusedMoE active: hidden=1024, intermediate=2688, experts=512, top_k=22, activation=relu2`** info-once fired at 13:17:08 → `post_load_weights` ran cleanly through the b12x weight conversion (`swizzle_block_scale` on normalized FP8 SF, `a1_gscale = 1/input_scale`, `w1_alphas = input_scale² · fc31_alpha`, `B12XFP4ExpertWeights` packing) for every MoE layer.
- **CUDA-graph capture** passed (the v1 failure was caller-owned-output-buffer; v2 fix in `B12xLukeFusedMoE.post_load_weights` allocates a shared `(max_num_tokens, hidden_size)` buffer keyed by `(shape, dtype, device)` and slices it per call — same pattern as the existing `_SHARED_MOE_OUTPUT_BUF` for FlashInfer).
- **Hybrid dispatch** confirmed by `TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD=64` exported and TTFT P50 = 156.45 ms ≈ pure CUTLASS prefill 154.53 ms (vs pure b12x 229.16 ms in `HYBRID_RESULTS.md`).
- **5/5 requests completed cleanly**, exit=0.

## Harness (identical to `HYBRID_RESULTS.md` row "Hybrid")

- GPU: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120, 97 GB), GPU 1.
- Model: `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (HF / ModelOpt NVFP4).
- TRT-LLM: `1.3.0rc14` wheel + source-tree overlay of fused_moe submodule + `llm_args.py`.
- FlashInfer: `0.6.8` @ `8a9970b4` (kept installed; unused on this run).
- **lukealonso/b12x: `0.13.0` @ `1378cea7` ("Restore Nemotron micro MoE performance", 2026-05-07).**
- cutlass-dsl 4.4.2 trio.
- Bench: `tensorrt_llm.commands.bench ... throughput --max_batch_size 1 --max_num_tokens 4096 --num_requests 5 --warmup 0 --concurrency 1 --streaming`.
- ISL=2048, OSL=1024, 5 reqs, conc=1.
- YAML (`bench_kvoff_b12x_luke.yml`): `kv_cache_config.enable_block_reuse=false`,
  `kv_cache_config.free_gpu_memory_fraction=0.6`, `enable_chunked_prefill=true`,
  `enable_iter_perf_stats=true`, **`moe_config.backend=B12X_LUKE`**.
- `TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD=64` → CUTLASS prefill, lukealonso-b12x decode.

## Per-request TPOT (ms)

|       | Hybrid (FI b12x) | **Hybrid-luke** | CUTLASS-only |
| ----- | ---------------: | --------------: | -----------: |
| MIN   | 11.4884          | 13.3982         | 13.9516      |
| **P50** | **11.4935**    | **13.4047**     | **13.9681**  |
| AVG   | 11.4973          | 13.4189         | 13.9918      |
| P90   | 11.5187          | 13.4786         | 14.0917      |

## Per-request TTFT (ms)

|       | Hybrid (FI b12x) | **Hybrid-luke** | CUTLASS-only |
| ----- | ---------------: | --------------: | -----------: |
| MIN   | 153.9739         | 155.7840        | 154.5613     |
| **P50** | **154.5287**   | **156.4523**    | **154.6690** |
| AVG   | 155.7434         | 170.9975        | 193.8085     |
| P90   | 161.2233         | 230.2775        | 349.7044     |

## Success criteria

| #   | Criterion                       | Target              | Measured | Verdict |
| --- | ------------------------------- | ------------------- | -------- | ------- |
| 1   | Token parity                    | First-token agreement preferred; FP4 noise OK | (not run — unblocked, but moot given perf regression) | _N/A_   |
| 2   | Decode TPOT improvement         | < 11.49 × 0.99 ≈ 11.37 ms | 13.40 ms | **FAIL (+16.6 % regression)** |
| 3   | TTFT no-regression              | ≤ 154.5 × 1.02 ≈ 157.6 ms | 156.45 ms | **PASS** |
| (s) | Throughput stretch              | ≥ 87 tok/s          | 73.68 tok/s | **FAIL (−14.2 %)** |

## Net

The integration with the standalone `lukealonso/b12x v0.13.0` package works end-to-end on Nemotron-Super-120B-NVFP4 (correctness path runs, CUDA graph captures, full hybrid dispatch fires), but the **decode kernel performance is a regression** vs the older flashinfer-vendored b12x — entirely because the Nemotron-tuned `_launch_exact_relu2_bs1_nemotron` micro path is gated off in upstream `master`.

## Follow-up probe — micro path is broken in upstream

To confirm whether the gated-off micro path would have given us the FI-vendored speedup if enabled, I patched out the `return False` in the container's installed `b12x/integration/tp_moe.py` and changed `B12xLukeFusedMoE.post_load_weights` to pass scalar `a1_gscale` / `a2_gscale` (`numel == 1`, required by the dead-code shape check). Re-bench at `b12x_luke_micro_20260508_062243.log`.

Result: **`NameError: name 'micro_mac' is not defined`**, raised on the very first MoE forward pass (CUDA graph capture):

```
File "/usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py",
  line 2816, in _get_exact_relu2_bs1_nemotron_launcher
    mac_override=micro_mac,
                 ^^^^^^^^^
NameError: name 'micro_mac' is not defined
```

`micro_mac` is referenced at L2816 inside `_get_exact_relu2_bs1_nemotron_launcher` but is only ever defined at L3064 — inside `b12x_moe_fp4` itself, *after* the Nemotron-bs1 early-return branch that calls the launcher. So the launcher tries to read a variable that exists only in the caller's enclosing scope.

This is the actual bug the `return False` was working around. The micro path is *broken* in upstream `master` HEAD at `1378cea7` and cannot be revived without an upstream patch.

The container's `tp_moe.py` was restored to the original upstream version after this probe; `B12xLukeFusedMoE.post_load_weights` was left with the scalar `a1_gscale` / `a2_gscale` form (b12x's documented contract is "[E] OR scalar"; scalar is preferred for our case since it's what the eventual fixed micro path will require, and it costs nothing for the static/dynamic path).

## Possible next steps (not pursued in this hack)

1. **Walk older commits** (pre-`a60bf0eb` "A16 MoE Kernel variants", 2026-05-06 — that commit was a major rework that almost certainly introduced the `micro_mac` scope bug). Earlier tags might have a working micro path; the trade-off is they may also lack other recent fixes our codepath needs.
2. **Re-derive the launcher contract** so we can pass `micro_mac` in ourselves (compute it on our side and inject it via a tiny monkey-patch on `_get_exact_relu2_bs1_nemotron_launcher`). Risky — there are likely other shared variables.
3. **File the bug upstream** to lukealonso/b12x asking for the `micro_mac` scope fix (or documentation for why the path is gated off).
4. **Stay on the existing FlashInfer-vendored b12x** for production. The current branch's `FlashInferFusedMoE` (commit `ac436984e5`) is the fastest-known Nemotron decode path on SM120 today.
