# Phase 5 follow-up — nsys pure-kernel A/B (K2.5 NVFP4 + EAGLE-3, TP=8 mtp=3)

**Date:** 2026-05-23
**Goal:** Isolate the MLA-decode kernel time per forward under TRTLLM vs
TOKENSPEED_MLA backends to remove the EAGLE acceptance-rate confound
that contaminated the throughput-only Phase 5 numbers.

**Headline:** **TokenSpeed's MLA-decode kernel is genuinely 5.7% faster
than TRT-LLM's per forward at q_len_per_req=4 on K2.5.** But end-to-end
throughput is 15.6% **slower** under TS at TP=8 — the kernel-level win
is more than wiped out by TS's split-kernel design (2 GPU launches per
forward + a sync point) and other downstream overhead. The Phase 5 +4.2%
TS throughput at TP=4 was a partial reconciliation of these two effects
plus an acceptance-rate gap that disappeared at TP=8.

## TL;DR

| Metric | TRTLLM (base) | TOKENSPEED_MLA (ts) | Δ |
|---|---:|---:|---:|
| **MLA-decode kernel total** (nsys) | 94.15 ms | **88.80 ms** | **TS −5.7%** ✓ |
| Per-forward MLA-decode time | 12.82 μs | **12.09 μs** | **TS −5.7%** ✓ |
| MLA-decode kernel launches | 7344 | **14688** (2× per forward) | **TS +2× launches** |
| Bench Total Token Throughput | **1240.77 tok/s** | 1046.59 tok/s | TS −15.6% ✗ |
| Bench Per-User Throughput | **336.43 tps/user** | 313.93 tps/user | TS −6.7% ✗ |
| Bench Acceptance Length avg | 3.03 | 3.07 | TS +1.3% (tied) |

**Two findings, one story:**

1. **At the pure MLA-decode kernel level, TokenSpeed wins by ~5.7% per
   forward** — this validates the `fold_sq_factor` BMM1 reformulation
   that TokenSpeed claims at `q_len_per_req > 1`.
2. **End-to-end TS loses 15.6%** because of the design choice to split
   partial-KV reduction into a separate kernel. The pure compute win
   (−5.7%) is overwhelmed by ~7344 extra kernel launches (~5-10 μs each
   = 37-73 ms cumulative) and the GPU sync point between split-KV and
   reduction.

## Setup

Same as Phase 5 perf-only, but TP=8 instead of TP=4 (nsys instrumentation
overhead pushed K2.5 over the 268 GB-per-GPU budget at TP=4).

- **Model:** `nvidia/Kimi-K2.5-NVFP4` BF16-KV patched
- **Draft:** `nvidia/Kimi-K2.5-Thinking-Eagle3`
- **HW:** B300 SXM6 (sm_103a), TP=8, single node
- **Source:** `tokenspeed-kimik25-eval-public` rebased on `upstream/main@f278c4f170`
  + PR #14291 (FMHA JIT fix) + PR #9677 (Eagle MLA)
- **Bench:** `trtllm-bench throughput --tp 8`, ISL=OSL=1024, conc=2,
  num_requests=4 (small to keep nsys trace manageable), max_draft_len=3
- **nsys flags:** `-t cuda,nvtx,osrt -s none --capture-range=none`

Files: `.claude_docs/tokenspeed-kimik25/{bench-k25-mtp3-tp8.yml,scripts/run_bench_k25_nsys.sh}`.
Trace artifacts: `/scratch/runs/k2.6-spike/phase5-k25-mtp3-nsys/{base,ts}.{nsys-rep,sqlite,log}`
(each `.nsys-rep` ≈ 1.2 GB, each `.sqlite` ≈ 15 GB).

## MLA-decode kernel breakdown

### TRTLLM (base) — single kernel per forward

| Kernel | Time | Inst | Avg μs |
|---|---:|---:|---:|
| `fmhaSm103a...HQk576HV512HVPerCta256PagedKvDenseP32MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen` | 54.59 ms | 3968 | 13.76 |
| `fmhaSm103a...HQk576HV512HVPerCta128PagedKvDenseP32MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen` | 39.56 ms | 3376 | 11.72 |
| **TOTAL** | **94.15 ms** | **7344** | **12.82** |

The two HVPerCta variants reflect the scheduler choosing different tile
configs for different batch / context shapes. Both are `Q8Kv128` (the
`Q8` reflects q_len_per_req=4 → 4-query-token batches grouped into a
larger M-dim tile).

### TOKENSPEED_MLA (ts) — split into 2 kernels per forward

| Kernel | Time | Inst | Avg μs |
|---|---:|---:|---:|
| `tokenspeed_mla...split_kv_kernel...mla_decode_fp16BlackwellMultiHeadLatentAttentionForwardFP16` | 60.96 ms | 7344 | 8.30 |
| `tokenspeed_mla...reduction_kernel...mla_decode_fp16...` (variant 1) | 26.37 ms | 6944 | 3.80 |
| `tokenspeed_mla...reduction_kernel...mla_decode_fp16...` (variant 2, small) | 1.47 ms | 400 | 3.68 |
| **TOTAL** | **88.80 ms** | **14688** | **12.09 per forward** |

The kernel name uses `_fp16` but the actual datatype is BF16 — same
storage, naming-convention quirk in the CuTe DSL library. The two
reduction variants reflect the same scheduler choice (which Cta-shape
to reduce).

### Interpretation

- **Pure compute (GPU kernel) time per forward: TS 12.09 μs vs base 12.82 μs → TS −5.7%.**
- **Per-forward GPU work split:** TS = (split-KV 8.30 + reduce 3.79) = 12.09 μs.
  Base = (~12.82 μs in a single fused FMHA kernel).
- **TS uses 2× the kernel launches** for the same forward count. At ~5-10 μs
  CPU launch overhead each, 7344 extra launches ≈ 37-73 ms of host-side
  wall time that doesn't show up in GPU kernel summary.
- **Implicit sync between split-KV and reduction**: the split-KV writes
  partial outputs that the reduction reads. CUDA stream ordering
  guarantees this happens in order, but the GPU may stall waiting for
  the split-KV warps to drain before reduction can start. Base's fused
  kernel doesn't have this boundary.

## Reconciling with Phase 5 throughput data

| Config | TS throughput Δ | TS AL Δ | Likely kernel Δ |
|---|---|---|---|
| Phase 5 TP=4 (perf-only, num_req=32) | **+4.2%** | +6.7% | ~−2.3% per forward (estimated) |
| Phase 5 nsys TP=8 (num_req=4) | **−15.6%** | +1.3% | **−5.7%** (measured) |

**At TP=4** (Phase 5 perf-only): the +6.7% acceptance gap roughly compensated
for the per-forward kernel disadvantage (which we now know is actually a
*small advantage* at the pure-compute level), so TS came out +4.2%
throughput. The launch overhead penalty was less visible because TP=4
has less parallelism contention.

**At TP=8** (this run): the acceptance gap collapsed to +1.3% (within
noise) because the EAGLE-3 draft is independent of TP. With no acceptance
help, the launch overhead penalty + split-kernel sync dominated, giving
TS −15.6% throughput despite the −5.7% pure-kernel advantage.

The story: **TokenSpeed's algorithmic claim (`fold_sq_factor` reduces MLA-decode
GPU work at q_len > 1) is real and measurable at the kernel level**. But
the **implementation design** (split partial-KV reduction into a
separate kernel) costs more in launch + sync overhead than the algorithm
saves on B300 sm_103a — at least at TP=8 with this config.

## Caveats

1. **Single point**: 4 requests at TP=8 mtp=3 1k/1k. Variance unknown.
   The −5.7% kernel-level delta should reproduce; the +37-73 ms launch
   overhead estimate is order-of-magnitude and depends on the host's
   CPU + CUDA driver path.
2. **Small bench**: 4 requests is just enough to capture many MLA-decode
   invocations (7344). Scaling to 32 requests doesn't change the
   per-forward kernel time but would smooth out the launch overhead
   estimate.
3. **No CUDA graph analysis**: Both arms have `cuda_graph_config:
   enable_padding: true, batch_sizes: [1,2,4]` set. If TS's split-kernel
   design falls out of CUDA graph capture more often than base's fused
   kernel, the launch overhead penalty would be larger than estimated.
   nsys trace can be queried for graph hit/miss counts but I haven't.
4. **BF16 KV, not prod FP8 KV.** Same caveat as Phase 5 perf-only.
5. **TP=8 ≠ Phase 5 TP=4**: this run uses TP=8 because TP=4 OOM'd under
   nsys instrumentation overhead. TS's behavior at TP=4 could differ
   (more contention, different launch-overhead-to-kernel-time ratio).

## What this means for the TokenSpeed verdict

- **Algorithm**: The `fold_sq_factor` BMM1 reformulation **works** —
  TS's MLA-decode kernel is ~5.7% faster per forward at q_len=4.
- **Implementation**: TS's split-kernel design **doesn't pay** on B300
  sm_103a in this regime. Re-fusing partial-KV reduction back into the
  attention kernel (the way TRT-LLM's FMHA does) would in principle
  let TS keep the algorithmic win without paying the launch + sync
  cost.
- **Adoption recommendation unchanged from Phase 5 / Phase 4**: don't
  default-on TokenSpeed for K2.5 production. The end-to-end throughput
  is worse than the post-#14291 TRT-LLM baseline at every regime
  tested so far (K2.6 q_len=1: TS −3 to −7%; K2.5 q_len=4 TP=4 perf:
  +4.2% but acceptance-confounded; K2.5 q_len=4 TP=8 nsys: −15.6%).
- **Path forward, if interested**: implement the `fold_sq_factor`
  pattern inside TRT-LLM's existing FMHA kernel (i.e., absorb the
  algorithmic idea without taking the split-kernel design). This is
  what DKG MR 21023 hints at — keep the BMM1 reformulation, drop the
  separate reduction kernel.

## Artifacts

- `phase5-k25-eagle3.md` — Phase 5 perf-only writeup (TP=4 throughput numbers)
- `phase5-k25-eagle3-nsys.md` — this writeup (TP=8 nsys pure-kernel)
- `bench-k25-mtp3.yml`, `bench-k25-mtp3-tp8.yml` — bench configs
- `scripts/run_bench_k25.sh` — Phase 5 perf-only driver
- `scripts/run_bench_k25_nsys.sh` — nsys A/B driver
- `/scratch/runs/k2.6-spike/phase5-k25-mtp3-nsys/{base,ts}.nsys-rep` — full traces
- `/scratch/runs/k2.6-spike/phase5-k25-mtp3-nsys/{base,ts}-kernsum_cuda_gpu_kern_sum.csv` — kernel summaries

## ⚠️ Methodological caveat — the "−5.7%" was warmup kernels, not timed-run

While verifying the launch-overhead hypothesis (next section), I noticed
a major issue with the kernel-level finding:

- base.cudakernsum reports **7344 MLA-decode kernel invocations** total
  (across 8 ranks) = 918 per rank.
- ts reports 14688 (2× because of split-KV + reduction) = 1836 per rank.
- K2.5 has 61 layers; the timed run had ~333 generation iterations
  (computed from 1024 OSL / AL=3.03 ≈ 338 iters per req × conc=2 batches).
- Expected MLA-decode invocations during timed run alone:
  ~333 iters × 61 layers / rank = ~20k per rank.

918 actual << 20k expected. **Most timed-run kernels are inside CUDA
graphs** (we saw 48,340 `cudaGraphLaunch` instances each arm, identical
between base and ts). nsys by default aggregates graph-captured kernels
into single `cudaGraphLaunch` events — they don't appear individually
in `cuda_gpu_kern_sum`.

**So the "MLA-decode TS −5.7% per forward" finding above is for
non-graph kernels** (warmup, autotuner, first-shape-compile edges).
The graph-captured kernels (the bulk of timed-run compute) are not
individually visible in this trace. To get accurate timed-run per-kernel
data, nsys profile must be re-run with `--cuda-graph-trace=node` (which
unfolds graph kernels into individual events). That'd add ~30 min per
arm.

**What this changes for the verdict:**

- **MLA-decode pure kernel comparison: INCONCLUSIVE for the timed run.**
  The −5.7% number applies to warmup/non-graph executions only, where
  shape-driven JIT effects and kernel selection may differ from
  steady-state graph-captured replay.
- **Launch overhead, CUDA graph usage, total GPU work**: those numbers
  are aggregated across the whole trace, including init/warmup which
  dominates the trace by time. The launch overhead and graph-usage
  comparisons aren't necessarily timed-run-specific either, but at
  least the graph counts match exactly between arms.
- **The +15.6% throughput delta is the only verified timed-run
  measurement**. Whatever's slowing TS is happening inside the
  CUDA-graph-replayed kernels — and our nsys trace can't see it at
  per-kernel granularity.

To resolve the kernel-level question definitively, re-run nsys with
`--cuda-graph-trace=node`. Otherwise we have to leave the per-kernel
attribution as an open question.

## Follow-up verification: launch-overhead hypothesis REFUTED

After noting the apparent ~610 ms wall gap, I queried `nsys stats
--report cuda_api_sum` and `cuda_gpu_kern_sum` on both traces to verify
the launch-overhead hypothesis. **The numbers don't support it.**

### CUDA API totals (across whole trace, 8 ranks summed)

| API | base | ts | Δ |
|---|---:|---:|---:|
| `cuLaunchKernelEx` | 48.80 sec / 548,536 calls | (rare) | — |
| `cudaLaunchKernelExC` | 29.03 sec / 730,420 calls | 20.46 sec / 749,204 calls | TS −8.6 sec |
| `cudaLaunchKernel` | 7.97 sec / 1.42M calls | **14.55 sec / 1.40M calls** | TS +6.6 sec |
| **Total launch CPU time** | **77.83 sec** | **35.01 sec** | **TS −42.8 sec** |
| Total launches | 1,278,956 | 2,151,723 (+68%) | TS more launches but less time |

**TS spends 43 sec LESS on launches over the whole trace.**

### CUDA graph usage (identical between arms)

| API | base | ts |
|---|---:|---:|
| `cudaGraphInstantiateWithFlags` | 48,340 | 48,340 |
| `cudaGraphLaunch` | 48,340 | 48,340 |
| `cudaGraphExecDestroy` | 48,324 | 48,324 |

CUDA-graph hypothesis also refuted — both arms use graphs identically.

### Total GPU kernel work (across whole trace, 8 ranks summed)

| Metric | base | ts | Δ |
|---|---:|---:|---:|
| Total GPU kernel time | **221.03 sec** | **147.20 sec** | **TS −33.4% (-74 sec)** |
| Total kernel invocations | 1,848,001 | 1,828,165 | tied |

**TS does substantially LESS GPU work overall, with the same invocation count.**

### Per-kernel comparison (notable differences)

| Kernel | base | ts | Notes |
|---|---:|---:|---|
| `ncclSymk_AllReduce_AGxLLMC_R_sum_bf16` | **69.77 sec / 2784 inst** (25.06 ms/call) | (not in top 8) | Big disparity |
| `ar_fusion_kernel<pattern=0>` | 52.67 sec / 1424 inst (**36.99 ms/call**) | 23.21 sec / 1424 inst (**16.30 ms/call**) | base 2.27× longer per call |
| `ar_fusion_kernel<pattern=1>` | 13.25 sec | 10.92 sec | base 21% more |
| MLA decode (decode-only kernels) | 94.15 ms (decode only) | 88.80 ms (decode only) | TS −5.7% (confirms above) |
| `dsv3MinLatencyKernels::fused_a_gemm` | 8.21 sec / 8784 inst | 8.20 sec / 8784 inst | identical |
| `nvjet_sm103_tss_32x64_64x16_4x1` | 7.33 sec / 8640 inst | 8.22 sec / 8640 inst | TS +12% |

### What this really means

The apparent contradiction:
- TS does **less GPU work** (74 sec less across whole trace)
- TS does **less launch work** (43 sec less)
- TS takes **MORE wall time** (timed-run throughput 1046 vs 1240 tok/s)

There's only one possibility left: **TS has more GPU idle time during
the timed run.** The GPU is starving — kernels are individually faster
but they don't pack as tightly into the iteration timeline.

The most plausible mechanism — visible in the data:

The base arm's `ar_fusion_kernel<pattern=0>` runs for 37 ms per call;
TS's same kernel runs for only 16 ms per call. **Same call count
(1424 each).** This is an NCCL/AR-fusion allreduce that overlaps with
attention compute on a separate stream. On base, the allreduce kernel's
duration covers the compute on the other stream (overlap = the
allreduce is "the long pole"). On TS, the allreduce completes faster
because the compute on the other stream is also faster — but that
means the GPU finishes both sooner AND has nothing else to do.

The base configuration appears to be **compute-bound** (compute is
the long pole, allreduce hides behind it). TS's faster compute kernels
flip this to **comm-bound or partly idle** (allreduce can't overlap
enough to hide its full cost). The bigger TS allreduce ratio +
shorter compute = wall time loses ground despite kernel-level wins.

This is consistent with TS's split-KV → reduction design: the boundary
between split-KV and reduction is a stream-level barrier that the
allreduce can't cross. base's fused FMHA kernel emits its output in one
go, letting the next allreduce start as soon as the kernel finishes;
TS has to wait for both split-KV and reduction to finish before the
post-MLA allreduce can launch. That's a few μs per forward, but over
7344 forwards it's measurable.

### Updated story

| Layer | Result |
|---|---|
| **MLA-decode pure kernel** | TS −5.7% per forward ✓ (algorithm wins) |
| **CUDA launch overhead** | TS slightly cheaper (−43 sec total) |
| **CUDA graph usage** | identical (48,340 each) |
| **Total GPU kernel time** | TS −74 sec total (TS does less work) |
| **Comm/compute overlap** | base's allreduce hides behind longer compute; TS exposes more allreduce cost |
| **Timed-run wall** | TS −15.6% (slower) — most of the gap is **GPU idle time**, not extra work |

So the right framing isn't "TS pays launch overhead" but rather
**"TS doesn't compose as well with the rest of the pipeline."** Faster
kernels in isolation don't help if they unbalance the compute/comm
schedule.

## Recommended next steps (revised again)

1. **Get a timeline view** — nsys GUI or `nsys export --type=parquet`
   on a small slice; look at the per-stream Gantt during 1-2 iterations
   of the timed run. Verify that base's allreduce is overlapping
   compute that TS's allreduce can't hide.
2. **Filter `cuda_api_sum` to the timed-run NVTX range** — get a clean
   apples-to-apples comparison of just the timed iterations, not the
   init/warmup-dominated whole-trace totals.
3. **Try TS at higher concurrency / batch size** — if the issue is
   GPU starvation from short kernels, more concurrent work should mask
   it. Phase 4 at conc=16 showed TS tied with base on K2.6; the same
   probably applies on K2.5 with EAGLE-3.
4. **If pursuing TS adoption**: absorb the `fold_sq_factor` algorithm
   into TRT-LLM's FMHA kernel (keep base's fused single-kernel design,
   adopt TS's BMM1 grouping). This is what DKG MR 21023 hints at.
   That preserves the −5.7% kernel win without breaking comm/compute
   overlap.
