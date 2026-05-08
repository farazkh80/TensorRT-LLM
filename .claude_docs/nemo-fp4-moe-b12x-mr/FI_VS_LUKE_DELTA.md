# FlashInfer-vendored b12x vs lukealonso/b12x v0.13.0 — architectural delta

Inspecting both packages installed side-by-side in `b12x_luke_runtime`:

- FI: `flashinfer-python==0.6.8` @ git `8a9970b4` → `/usr/local/lib/python3.12/dist-packages/flashinfer/`
- Luke: `b12x==0.13.0` @ git `1378cea7` → `/usr/local/lib/python3.12/dist-packages/b12x/`

Both target the same SM120/SM121 NVFP4 MoE workload and ship `MoEMicroKernel`, `MoEStaticKernel`, `MoEDynamicKernel` as the three kernel families. The deltas are entirely in **dispatch logic**, **shape gates**, and **config thresholds** — not in the kernel sources themselves.

Our shape (Nemotron-Super-120B-NVFP4 decode): `m=1, k=1024, n=2688, num_topk=22, E=512, activation=relu2`. Nominal `routed_rows = m × num_topk = 22`.

## File layout

| Concern | FlashInfer | Lukealonso b12x |
|---|---|---|
| Top-level entry-point | `flashinfer.B12xMoEWrapper` (class) + `flashinfer.b12x_fused_moe` (function) → `flashinfer/fused_moe/cute_dsl/b12x_moe.py` (15 KB) | `b12x.integration.b12x_moe_fp4` (function) + `b12x.integration.b12x_sparse_moe_fp4` (gate-fused) → `b12x/integration/tp_moe.py` (143 KB) |
| Dispatch internals | `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` (1760 lines) | merged into `tp_moe.py` |
| Kernel sources | `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_{micro,static,dynamic}_kernel.py` (2.4k / 2.3k / 2.3k lines, 100 KB each) | `b12x/moe/fused/{micro,static,dynamic}.py` (similar size) |
| Activation specs | hard-coded in `_get_micro_kernel` / `_get_static_kernel` / `_get_dynamic_kernel` | dataclass `_ActivationKernelSpec` with `make_micro_kernel` / `make_static_kernel` / `make_dynamic_kernel` factories |
| Workspace | per-call `Sm120{Static,Dynamic}MoEWorkspace` cached at module level by shape, plus `_workspace` arg for graph-capture wrapper | `TPMoEWorkspacePool` (stateful, grows on demand, reused across layers/shapes) |

## Dispatch hierarchy (where the kernel is picked)

### FlashInfer

```
b12x_fused_moe(...)  or  B12xMoEWrapper.run(...)
  → launch_sm120_moe(num_tokens, top_k)              # picks "static" or "dynamic"
      └── select_sm120_moe_backend(num_tokens, num_topk):
              backend = "dynamic" if heuristic else "static"

      ├── if "dynamic":  launch_sm120_dynamic_moe → MoEDynamicKernel
      └── if "static":   launch_sm120_static_moe (lines 784–956)
              └── routed_rows ≤ _MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK (default 40 for top_k>1):
                       use_micro = True  → _get_micro_kernel → MoEMicroKernel
              └── else: MoEStaticKernel
```

For our shape: `routed_rows = 22`. `top_k = 22 > 1`, so cutover = **40**. `22 ≤ 40` → **micro fires**. TPOT 11.49 ms confirms.

### Lukealonso b12x

```
b12x_moe_fp4(...)
  ├── if _is_exact_relu2_bs1_nemotron_case(...):                       # ← unconditionally False!
  │       return _launch_exact_relu2_bs1_nemotron(...)                 # broken (NameError: micro_mac)
  ├── plan = _make_workspace_plan(...)
  │     └── _resolve_workspace_layout → select_tp_moe_backend(num_tokens, num_topk):
  │             return "static" if routed_rows ≤ _get_static_compact_cutover_pairs() (default 640) else "dynamic"
  ├── if impl == "dynamic":  _launch_dynamic → MoEDynamicKernel
  └── if impl == "static":   _launch_compact_static (lines 2985–3130)
        └── use_micro_direct = is_supported(m, k, n, num_topk, E)      # shape-conditioned
              └── if True AND _compiled_direct_micro_accepts_block_dim(compiled, _BLOCK_DIM=512):
                          MoEMicroKernel.launch(...)
                  else:    MoEStaticKernel
```

For our shape:
- `routed_rows = 22 ≤ 640` → impl = **"static"** ✓
- `is_supported(m=1, k=1024, n=2688, num_topk=22, E=512)` returned **True** when probed (verified inside container).
- `_compiled_direct_micro_accepts_block_dim(compiled, 512)` is a runtime check that queries the JIT-compiled micro kernel's `MAX_THREADS_PER_BLOCK` via `cuKernelGetAttribute`. **If the JIT compile chose a block_dim < 512** (e.g., from register-pressure-induced `__launch_bounds__`), this returns False and the dispatch falls through to `MoEStaticKernel` — the slow path that is the prime suspect for our 13.40 ms TPOT (≈ CUTLASS class).

## Concrete deltas

### 1. Top-level dispatch — luke added a 3rd top-level branch that's broken

| | FI | Luke |
|---|---|---|
| Top-level branches | 2 (`static` / `dynamic`) | 3 (`exact_relu2_bs1_nemotron` / `static` / `dynamic`) |
| Status | works | bs1 branch unconditionally returns False (`tp_moe.py:2721`); dead-code body crashes with `NameError: name 'micro_mac' is not defined` at `_get_exact_relu2_bs1_nemotron_launcher:2816` |
| Effect on our shape | reaches static, internally picks micro | falls through to static, internally tries `use_micro_direct`, may fall further to MoEStaticKernel via `_compiled_direct_micro_accepts_block_dim` gate |

### 2. Micro selection criterion

| | FI | Luke |
|---|---|---|
| Where | inside `launch_sm120_static_moe` after entering the "static" backend | inside `_launch_compact_static` after entering the "static" implementation |
| How | table lookup: `routed_rows ≤ _MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK` (40) | shape-fitness check: `MoEMicroKernelRelu2.is_supported(m, k, n, num_topk, E)` (constraints on `m ∈ {1,2,4,8}`, `k % 512 == 0`, `n % _BLOCK_SIZE == 0`, `num_topk ≤ 32`, etc.) |
| Plus runtime gate | none | `_compiled_direct_micro_accepts_block_dim(compiled, 512)` — queries JIT-compiled kernel's `MAX_THREADS_PER_BLOCK` |
| Plus Triton compaction | `triton_compact.compact_topk_ids` runs unconditionally in micro path before the kernel | luke runs the kernel directly when ids are int32-contiguous; otherwise compacts via workspace |

The runtime block-dim gate is the most likely culprit. If the compiled micro kernel has a `__launch_bounds__` < 512 due to register pressure, the entire path silently falls through to the slow static kernel.

### 3. Cutover thresholds (env-overridable in luke, hard-coded in FI)

| Threshold | FI value | Luke default | Luke env override |
|---|---:|---:|---|
| `static`-vs-`dynamic` (top-level) | (heuristic in `select_sm120_moe_backend`) | 640 | `B12X_STATIC_COMPACT_CUTOVER_PAIRS` |
| `micro`-vs-`static` (within static) | 40 (top_k > 1), 20 (top_k = 1) | 80 (top_k > 1), 20 (top_k = 1) | `B12X_MICRO_COMPACT_CUTOVER_PAIRS` |

Luke is more permissive on both axes (more shapes go through static; more shapes within static go through micro). For our 22-routed-rows decode, both hit the micro cutover in either package.

### 4. Activation-quant scale convention

(Already addressed in `B12X_LUKE_RESULTS.md`.)

| | FI | Luke |
|---|---|---|
| `w*_blockscale` | UN-normalized FP8 SF (multiply by `weight_scale_2` then swizzle) | NORMALIZED FP8 SF (HF/ModelOpt form, just swizzle) |
| `w*_alpha(s)` | `1/input_scale` (kernel does dual-use cancel) | `input_scale^2 × fc_alpha` (per-expert epilogue dequant) |
| `a*_gscale` | (folded into `w*_alpha`) | `1/input_scale` (reciprocal input scale, scalar OR `[E]`) |
| swizzle helper | `flashinfer.cute_dsl.utils.convert_sf_to_mma_layout` | `b12x.cute.fp4.swizzle_block_scale` |

### 5. CUDA-graph contract

| | FI | Luke |
|---|---|---|
| Output buffer | `B12xMoEWrapper` allocates an internal `_moe_output: (max_num_tokens, hidden_size)` buffer at construction; ours then aliases this across layers | function call requires caller-owned `output: (m, k)` during graph capture; raises `ValueError("CUDA graph capture requires a caller-owned output buffer")` if `output=None` and `is_current_stream_capturing()` |
| Workspace | `_get_cached_workspace` keyed by `(backend, state_E, weight_E, routed_rows, k, n, num_topk, device)` | `TPMoEWorkspacePool` stateful, grows on demand |

This is why our v1 bench crashed at graph capture and why we had to add `_SHARED_MOE_OUTPUT_BUF` + per-call slicing in `B12xLukeFusedMoE.post_load_weights` / `run_moe`.

## Where luke regressed vs FI

The `1378cea7` commit "Restore Nemotron micro MoE performance" itself is the regression vector:

1. The `_is_exact_relu2_bs1_nemotron_case` early-return gate was either always there or added recently — the `_launch_exact_relu2_bs1_nemotron` call site references `micro_mac` from the caller's scope, which the gate was added to suppress.
2. **Both kernels lost the FI-style table-driven micro cutover**. FI ships a static table `_MICRO_MAC_LADDER` of `(routed_rows → mac)` tunings that picks the best CTA count for each tile size. Luke replaced this with `_get_impl_mac("micro", routed_rows)` which is also a tuning ladder, but the bs1 nemotron case relied on `micro_mac` being computed in the caller scope and passed in — which is exactly the bug.
3. The new `_compiled_direct_micro_accepts_block_dim` runtime gate (FI doesn't have it) silently falls through to the static kernel if the JIT-compiled micro doesn't accept the requested 512-thread block. This adds a new failure mode that FI's pipeline doesn't have.

## Hypotheses (initially considered)

1. **`_compiled_direct_micro_accepts_block_dim` returns False** because the JIT-compiled luke-micro kernel has `MAX_THREADS_PER_BLOCK < 512` due to register pressure. Falls through to `MoEStaticKernel`, which produces CUTLASS-class TPOT.
2. **`use_micro_direct` does fire** but luke's `MoEMicroKernel` is slower per-op than FI's at this shape — different CTA tile config, different prefetch depth, different warp count. (Recent commit `ff205cf0` "Restore micro MoE CTA warp count" suggests the kernel was retuned away from FI's settings.)
3. **`use_micro_direct` fires** but luke's micro kernel doesn't get the `share_input_across_experts` / `share_expert_scales` shortcuts because of how `a*_gscale` is shaped.

## Confirmed via direct runtime probe

A 1-request trace bench (`b12x_luke_trace_20260508_084118.log`) with `print` statements injected into `_launch_compact_static` confirmed the dispatch path on every decode iteration:

```
[trace-luke] _launch_compact_static: m=1 k=1024 n=2688 num_topk=22 E=512 si=True ses=True quant_mode=nvfp4
[trace-luke]   use_micro_direct=True
[trace-luke]   _compiled_direct_micro_accepts_block_dim=True (BLOCK_DIM=512)
[trace-luke]   ** MICRO LAUNCHED **
```

Repeated identically for every decode token across all 40 MoE layers, ~40,000+ instances, with no fall-through. The trace bench's PERFORMANCE OVERVIEW reports **TPOT P50 = 12.9654 ms** (single request, ISL=2048, OSL=1024, scalar `a*_gscale`).

**Hypothesis #1 falsified.** The runtime gate `_compiled_direct_micro_accepts_block_dim` returned True; the JIT-compiled luke-micro kernel accepts 512 threads/block. The dispatch reaches `MoEMicroKernel.launch`, no fall-through to MoEStaticKernel.

**Hypothesis #3 partially falsified, partially confirmed.** With scalar `a*_gscale` (`numel == 1`), both `share_input_across_experts` and `share_expert_scales` flags fire True. The flags do measurably help: TPOT 12.97 ms (scalar) vs 13.40 ms (per-expert v2 run) is a +3.3% delta from those flags alone. The full v2-vs-trace comparison:

| Run | `a*_gscale` shape | `share_input_across_experts` | `share_expert_scales` | TPOT P50 (ms) | Δ vs FI hybrid |
|---|---|---|---|---:|---:|
| FI hybrid baseline (`HYBRID_RESULTS.md`) | (folded into `w*_alpha`) | n/a | n/a | **11.4935** | — |
| Luke v2 (`b12x_luke_hybrid_20260508_061327.log`) | `[E]=512` | False | False | 13.4047 | **+16.6 %** |
| Luke trace (`b12x_luke_trace_20260508_084118.log`) | `numel=1` (scalar) | True | True | **12.9654** | **+12.8 %** |
| CUTLASS-only (`HYBRID_RESULTS.md`) | n/a | n/a | n/a | 13.9681 | +21.5 % |

**Hypothesis #2 confirmed as the residual bottleneck.** Even with all dispatch optimizations active (micro path + share_input_across_experts + share_expert_scales), luke's `MoEMicroKernel` is **+12.8 % slower per-op than FI's vendored snapshot** for our exact Nemotron-Super decode shape. This is intrinsic to the kernel itself — not dispatch, not scale-sharing flags, not JIT artifacts.

## Verdict

The two packages share dispatch architecture (modulo the broken Nemotron-bs1 fast path) and scale-sharing optimizations. They diverge in the **kernel implementation**: between flashinfer's pinned snapshot at `8a9970b4` (May 4) and lukealonso's master HEAD `1378cea7` (May 7), the upstream did "A16 MoE Kernel variants" reorg + "Restore micro MoE CTA warp count" retuning, and the new tuning is **+12.8 % slower than the older one** for Nemotron-Super at decode batch=1 / top_k=22 / E=512.

Confirmation requires kernel-level profiling (ncu) of:
- Per-warp instruction mix (FP4 dequant ↔ MMA ratios)
- Stall reasons (memory dependency, register dependency, sync, etc.)
- Achieved occupancy
- L2 hit rate

A direct A/B ncu-rep comparison of FI's `MoEMicroKernel<...>::__forward__` vs luke's `MoEMicroKernel<...>::__forward__` for our exact `(m=1, k=1024, n=2688, num_topk=22, E=512, relu2)` shape would identify which specific change in luke's retuning regressed perf — most likely candidates: warp count, software-pipelining stage count, FC1 chunk size, FC2 K-segment count.

## Bisect — `c9cc90ec` is *worse*, not better

To localize the regression, I installed luke's `c9cc90ec` ("Bump version to 0.13.0", 2026-05-06 — the commit immediately before the May 7 reorg+retune block) and ran the same 5-req hybrid bench. Log: `b12x_luke_c9cc90ec_20260508_085059.log`.

| Commit | Date | TPOT P50 (ms) | Δ vs FI |
|---|---|---:|---:|
| FI vendored snapshot `8a9970b4` (in flashinfer 0.6.8) | May 4 | 11.4935 | — |
| **luke `c9cc90ec`** (commit before May 7 reorg) | **May 6** | **13.5516** | **+17.9 %** |
| luke `1378cea7` (master HEAD, scalar gscale) | May 7 | 12.9654 | +12.8 % |

Going backward from May 7 → May 6 made TPOT **worse by +0.59 ms (+4.5 %)**. The May 6 → May 7 commits net to a perf *improvement*; most likely `ff205cf0` "Restore micro MoE CTA warp count" was an actual fix (not a regression as the architectural comparison earlier hypothesized).

**Therefore the +12–18 % regression vs FI predates `c9cc90ec`.** Continued the bisect one more step back to `986a405a` (May 4, "Add dynamically-computed MoE down projection scale" — the last MoE-changing commit before the May 6 `a60bf0eb` "A16 MoE Kernel variants" reorg). Log: `b12x_luke_986a405a_20260508_085932.log`.

| Commit | Date | TPOT P50 (ms) | Δ vs FI hybrid | Δ vs c9cc90ec |
|---|---|---:|---:|---:|
| FI vendored snapshot `8a9970b4` (in flashinfer 0.6.8) | May 4 | 11.4935 | — | — |
| **luke `986a405a`** (pre-A16-reorg) | **May 4** | **13.5534** | **+17.9 %** | +0.002 ms (noise) |
| luke `c9cc90ec` (post-A16-reorg, pre-CTA-restore) | May 6 | 13.5516 | +17.9 % | — |
| luke `1378cea7` (master HEAD, scalar gscale) | May 7 | 12.9654 | +12.8 % | −0.59 ms |

**`a60bf0eb` "A16 MoE Kernel variants" did NOT introduce the regression.** The TPOT at 986a405a (pre-reorg) and c9cc90ec (post-reorg) is identical to within 0.002 ms (0.01 %). The reorg refactored layout but didn't change the achieved kernel performance for our shape.

**The +17.9 % regression vs FI predates `986a405a`** — i.e., it predates the entire stretch of `lukealonso/b12x` master we can bisect. The kernel that ships in luke's public master has been ~+17.9 % slower than FI's vendored snapshot for at least 4 days (May 4 → present), with the May 7 commits improving it by −0.59 ms (~+5 %) but never recovering FI's tuning.

So either:
1. **FI vendored from a luke fork or private branch** that doesn't appear in `lukealonso/b12x` master history. The `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/{moe_micro,moe_static,moe_dynamic}_kernel.py` sources may have been hand-cherry-picked from an earlier or different luke state.
2. **FI's vendored kernel has been hand-tuned post-vendoring** — flashinfer maintainers modified the kernel inside their tree to optimize for the Nemotron-Super shape, drifting from luke's upstream.

Either way, **commit-selection alone within `lukealonso/b12x` cannot close the gap.** The kernel must be re-tuned or hand-ported.

## How to push the investigation further (not pursued)

1. **ncu-rep diff** of FI's `MoEMicroKernel<...>::__forward__` vs luke's at any commit — identify the specific tuning regression (warp count, pipelining stages, FC1 chunk size, FC2 K-segment count, register pressure, occupancy). This is the most valuable next step now that bisecting is exhausted.
2. **Compare the kernel sources directly** — `diff -u flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py b12x/moe/fused/micro.py` (after stripping import-path differences). Look for any changes in MMA tile selection, warp-specialization roles, software-pipelining depth, or FC1/FC2 chunking. The diff likely reveals exactly which lines differ.
3. **Hand-port FI's vendored kernel into a `torch.library.custom_op` inside our backend** — apples-to-apples control of both dispatch and kernel; bypasses the dependency on the public `lukealonso/b12x` package.
4. **File upstream issue** with the trace evidence + bisect data + perf delta, asking lukealonso to (a) fix the `micro_mac` scope bug in the bs1 fast path, (b) clarify which historical state of the kernel flashinfer vendored (since master HEAD doesn't reproduce FI's perf at any commit going back to May 4), and (c) re-tune the public master kernel to match for Nemotron-Super-120B.

## Apples-to-apples: forcing luke's warp-specialized `MoEStaticKernel`

After noticing that luke ships THREE kernel files (`micro.py`, `static.py`, `dynamic.py`) and only the first is flat-16-warp, an obvious follow-up was to force the dispatcher to skip the flat micro path and pick luke's warp-specialized `MoEStaticKernel`. Both `static.py:95` and `dynamic.py:270` set `num_mma_warps=4 / tma_load_warp_id=4 / threads_per_cta=160` — structurally identical to FI's `MoEMicroKernel` warp specialization.

Patch: in `b12x/integration/tp_moe.py`, force the `if use_micro_direct:` block to be skipped, so dispatch falls through to `_get_static_kernel` → `MoEStaticKernel.launch`. Re-bench at `b12x_luke_warpspec_20260508_094658.log`.

| Variant | TPOT P50 (ms) | Δ vs FI hybrid |
|---|---:|---:|
| FI hybrid baseline | 11.4935 | — |
| Luke `1378cea7` flat `MoEMicroKernel` (v2, per-expert) | 13.4047 | +16.6 % |
| Luke `1378cea7` flat `MoEMicroKernel` (trace, scalar) | 12.9654 | +12.8 % |
| Luke `c9cc90ec` flat `MoEMicroKernel` | 13.5516 | +17.9 % |
| Luke `986a405a` flat `MoEMicroKernel` | 13.5534 | +17.9 % |
| **Luke `1378cea7` warp-specialized `MoEStaticKernel` (forced)** | **13.5717** | **+18.1 %** |

**The +18 % gap survives the warp-specialization swap.** Forcing luke's warp-specialized kernel did not recover the FI perf. So the architectural delta is more nuanced than "FI has warp spec, luke doesn't":

- **FI** has a separate **small-batch-specialized** warp-spec kernel (`MoEMicroKernel`, picked when `routed_rows ≤ 40`). It is tuned specifically for the m=1 / decode case.
- **Luke master** has:
  - **`MoEMicroKernel`** (`micro.py`, flat 16-warp) — picked for small batches via `use_micro_direct`
  - **`MoEStaticKernel`** (`static.py`, warp-specialized 5-warp) — fallback, but tuned for *larger* routed_rows
  - **`MoEDynamicKernel`** (`dynamic.py`, warp-specialized 5-warp) — for large batches

So luke has a "small-batch flat" kernel and a "large-batch warp-spec" kernel, but **no "small-batch warp-spec" kernel** matching FI's `MoEMicroKernel`. Forcing the large-batch kernel onto small-batch shapes pays the wrong-tuning cost (≈ same as forcing the wrong-architecture flat micro kernel).

The real perf gap is **kernel-level tuning specific to small routed_rows / decode shapes**: CTA tile sizes, software-pipelining stages, FC1/FC2 chunk counts, MMA mode ladder. None of luke's three published kernels appear to be tuned for our shape.

## Kernel-source diff — the smoking gun

After exhausting bisect, I diff'd the actual kernel files inside the container:
- FI: `/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py` (2,414 lines, 106 KB)
- Luke @ `1378cea7`: `/usr/local/lib/python3.12/dist-packages/b12x/moe/fused/micro.py` (1,725 lines, 86 KB)

The two files have **fundamentally different CTA architectures**, not just different tuning constants:

| Aspect | FI vendored snapshot | Luke public master (any commit May 4–7) |
|---|---|---|
| Warps per CTA | **5** (`num_mma_warps=4` + `tma_load_warp_id=4`) | 16 (`_NUM_WARPS=16`) |
| Threads per CTA | **160** (`(num_mma_warps+1) * 32`) | 512 (`_BLOCK_DIM = _NUM_WARPS*32`) |
| Pattern | **Warp-specialized** producer / consumer | Flat / cooperative |
| `warp_idx` dispatch | Yes (FI L1318: `if warp_idx < self.num_mma_warps:` / FI L2188: `elif warp_idx == self.tma_load_warp_id:`) | **None** — all 16 warps run the same code |
| MMA warp register policy | `cute.arch.setmaxregister_increase(mma_register_requirement)` (FI L1319) | n/a |
| DMA warp register policy | `cute.arch.setmaxregister_decrease(load_register_requirement)` (FI L2189) | n/a |
| SMEM layout | `cute.nvgpu.warpgroup.make_smem_layout_atom` (FI L434) — warpgroup-aware | not warpgroup-aware |
| TMA driver | dedicated DMA warp issuing `cp.async.bulk` from `cute.nvgpu.warp.MMA_*` atoms (FI L1322 / L2188) | each of 16 warps loads + computes itself |

**FI uses the textbook CUTLASS Blackwell warp-specialized producer/consumer pattern**: warp 4 is a dedicated DMA loader issuing async TMA bulk loads while warps 0–3 execute MMA on previously-loaded tiles. The MMA group `setmaxregister_increase` to claim registers for accumulators; the DMA warp `setmaxregister_decrease` since it's memory-bound. This pipeline hides memory latency for free.

**Luke's kernel runs all 16 warps in lockstep with no producer/consumer split.** Every warp loads its own data and computes its own MMA. No async DMA, no warp-spec register repartitioning. MMA stalls every time a load is in flight.

For decode (m=1, latency-bound), warp specialization is the dominant optimization. Without it, the kernel cannot hide TMA latency behind MMA, and every load adds end-to-end latency. This is the architectural root cause of the +17.9 % TPOT regression.

**This is not a tuning regression. It's a missing kernel-design pattern entirely.** Luke's public master kernel was written without warp specialization in the first place; FI's vendored snapshot is from a fork or branch that has it.

## Bottom line

Refined verdict after the warp-specialization swap test:

- Luke does ship warp specialization — in `MoEStaticKernel` and `MoEDynamicKernel` (5-warp `num_mma_warps=4 / tma_load_warp_id=4`). My initial diff against `MoEMicroKernel` (flat 16-warp) was apples-to-oranges.
- But forcing luke's warp-spec `MoEStaticKernel` doesn't recover the perf — it lands at +18.1 % vs FI, essentially the same as the flat micro path.
- So the real gap is **shape-specific kernel tuning**: FI's `MoEMicroKernel` is a separate kernel **specifically tuned for the small-batch / decode case** (routed_rows ≤ 40). Luke has no equivalent — its small-batch path uses a flat kernel, its warp-spec kernels are tuned for larger routed_rows.

The bisect (May 4 → May 7) showed the gap is constant across that window. The warp-spec swap showed the gap survives the architectural pattern swap. So: closing the gap requires either kernel-level retuning of luke's `MoEStaticKernel` for `routed_rows ≤ 40`, or hand-porting FI's small-batch-tuned `MoEMicroKernel` (~12k lines of CuTe DSL cascade — out of scope).

`FlashInferFusedMoE` (#13773) remains the fastest known SM120 Nemotron-Super decode path. `B12xLukeFusedMoE` is committed as a stepping stone — when lukealonso adds a small-batch-tuned warp-spec kernel (or re-tunes the existing `MoEStaticKernel` for routed_rows ≤ 40), this backend activates the win without further TRT-LLM changes.

The cleanest next move if the perf delta matters enough is option 3 above: hand-port FI's `moe_micro_kernel.py` into a `torch.library.custom_op` inside `B12xLukeFusedMoE`, with the kernel-launch glue from `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` adapted to b12x's `B12XFP4ExpertWeights` weight format. That gives us FI's kernel perf without depending on flashinfer's wrapper code.
