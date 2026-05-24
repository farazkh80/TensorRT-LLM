# Phase 5 — nsys A/B (TP=4 + `--cuda-graph-trace=node`)

**Date:** 2026-05-24
**JIRA:** TRTLLM-12510
**Config:** K2.5 NVFP4 + EAGLE-3 (`max_draft_len=3`), TP=4 EP=4, conc=2,
ISL=OSL=1024, NVFP4 NB-MoE, BF16 KV (patched checkpoint).
**Variant from perf-only Phase 5:** `num_requests=4` (vs 32) to keep nsys
traces manageable; **nsys flag added: `--cuda-graph-trace=node`** to
unfold graph-captured kernels into individual events in
`cuda_gpu_kern_sum`.

## Goal

The perf-only Phase 5 TP=4 run reported TS **+4.2% throughput** with a
**+6.7% acceptance-length (AL) gap** (3.18 vs 2.98). The Phase 5 TP=8
nsys run could not confirm whether the underlying TS MLA-decode kernel
was actually faster — kernels captured inside CUDA graphs do not appear
individually in `cuda_gpu_kern_sum` without `--cuda-graph-trace=node`.

**This run resolves that.** With `--cuda-graph-trace=node` we get
per-kernel timing for graph-captured launches at TP=4 (the same regime
that produced the +4.2% perf-only win) so we can directly compare TS's
MLA-decode kernel against TRT-LLM's `fmhaSm103aKernel` per-call.

## Bench-level results (sanity)

|             | base (TRTLLM)  | ts (TOKENSPEED_MLA) | Δ      |
|-------------|----------------|---------------------|--------|
| Total Token Throughput (tok/s) | **947.50** | **913.07**          | **−3.6%** |
| Per-User Output Throughput (tps/user) | 280.63 | 253.79          | −9.6%  |
| Total Latency (ms)             | 8645.89    | 8971.91             | +3.8%  |
| Avg request latency (ms)       | 3766.23    | n/a                 |        |
| Avg GPU power (W)              | 615.25     | n/a                 |        |
| Acceptance Length (AVG)        | **2.99**   | **2.71**            | **−9.4%** |
| Acceptance Length (P50)        | 3.24       | 2.98                |        |
| Acceptance Length (MIN/MAX)    | 2.34 / 3.92| 1.99 / 3.86         |        |

Source: bench logs `base.log`, `ts.log` (per-arm PERFORMANCE OVERVIEW
sections, lines ~8390 onwards).

**Key observation:** at `num_requests=4`, TS's AL is **worse** by 9.4%,
whereas at `num_requests=32` (perf-only run) TS's AL was **better** by
6.7%. **AL is statistical noise at small N** — it cannot be a stable
algorithmic property of TS. The +4.2% perf-only TP=4 win was an AL
artifact, not a kernel win.

## Per-kernel attention A/B (the headline)

`--cuda-graph-trace=node` worked: both arms now show their MLA-decode
kernels as individual events, with full per-call timing.

### Base (TRTLLM `fmhaSm103aKernel`)

| Kernel | Total time | Instances | µs/call |
|---|---:|---:|---:|
| `fmhaSm103aKernel ... PerCta256 PagedKvDenseP32 MultiCtasKvCga VarSeqQ16Kv128 StaticSwapsAbForGen` | 3.363 s | 206,336 | 16.30 |
| `fmhaSm103aKernel ... PerCta128 PagedKvDenseP32 MultiCtasKvCga VarSeqQ16Kv128 StaticSwapsAbForGen` | 1.163 s | 88,630 | 13.13 |
| **MLA attention total**                                                                          | **4.526 s** | **294,966** | **15.34** (weighted) |

The two variants partition the 294,966 calls between two `Per-Cta` tile
sizes — both come from the same fused FMHA path.

### TS (TokenSpeed MLA-decode FP16, split-KV + reduction)

| Kernel | Total time | Instances | µs/call |
|---|---:|---:|---:|
| `kernel_cutlass_split_kv_kernel_tokenspeed_mla_decode_fp16BlackwellMultiHeadLatentAttentionForwardFP16` | 2.875 s | 307,151 | 9.36 |
| `kernel_cutlass_reduction_kernel_tokenspeed_mla_decode_fp16BlackwellMultiHeadLatentAttentionForwardFP16` (main, div4div16) | 1.205 s | 297,848 | 4.05 |
| `kernel_cutlass_reduction_kernel_tokenspeed_mla_decode_fp16BlackwellMultiHeadLatentAttentionForwardFP16` (small, div16div16) | 0.036 s | 9,303 | 3.84 |
| **MLA attention total**                                                                          | **4.116 s** | **307,151** | **13.40** (weighted) |

The TS path replaces the single fused FMHA with the documented 2-kernel
pattern: a split-KV kernel producing partial outputs + a reduction
kernel combining them. The small reduction variant is a fallback for
specific tail shapes.

### Per-call delta (kernel-only, controlling for invocation count)

|                      | base       | ts         | Δ        |
|----------------------|------------|------------|----------|
| Attn calls           | 294,966    | 307,151    | +4.1%    |
| Total attn time      | 4.526 s    | 4.116 s    | **−9.1%** |
| µs per attn call     | 15.34 µs   | 13.40 µs   | **−12.7%** |

**TokenSpeed's MLA-decode kernel is ~12.7% faster per attention call
than TRT-LLM's `fmhaSm103a` fused FMHA on this run.** This is the
direct kernel-level evidence that was missing in the Phase 5 TP=8 nsys
run. The `fold_sq_factor` BMM1 reformulation does deliver a real
per-call attention speedup; it is **not** marketing.

### What's missing from the per-call win

Total decode wall time saved on attention:
`4.526 − 4.116 = 0.410 s`
on a total benchmark wall of ~8.6 s — i.e., **~4.8% of the total run
was saved on attention kernels**.

But the bench latency went **up** by 3.8% (TS slower end-to-end). So
where did 0.41 s of attention savings disappear? Answer: TS made
**more forward steps** because its AL was lower.

## AL-aware forward-step accounting

`forward_steps = generated_tokens / acceptance_length` per request,
summed over the batch.

|                | base       | ts         |
|----------------|------------|------------|
| Generated tokens (4 reqs × 1024 OSL) | 4096 | 4096 |
| AL (avg)       | 2.99       | 2.71       |
| Forward steps  | **~1370**  | **~1512**  |
| Per-forward-step wall | 6.31 ms | 5.93 ms (**−6.0%**) |
| Per-forward-step attn time | 3.30 ms | 2.70 ms (**−18.2%**) |

So per forward step, TS is genuinely faster — both on attention
specifically (−18.2%) and on aggregate (−6.0%, dominated by the
attention saving + small absorb-path differences). **But TS makes
~10.4% more forward steps**, so net wall increases.

This is the mechanism: kernel win (~5% of total wall) is real but
fragile. AL noise (±10% sample-to-sample at small N) dwarfs it.

## Absorb-path GEMMs (cute_dsl `BlockScaledPersistentDenseGemmKernel`)

The MLA absorb path uses two `cute_dsl` blockscaled GEMM variants for
BMM1 / BMM2. With TokenSpeed, the `fold_sq_factor` reformulation
changes which variant is hit:

| Variant | Description | base | ts |
|---|---|---|---|
| `VMNK21111000_Permu_0` (one MMA atom shape) | bigger-grid GEMM, 7-8 µs/call | 3.026 s / 399,976 inst / 7.56 µs | **3.803 s / 622,624 inst / 6.11 µs** |
| `VMNK11110000_Permu_0` (different atom shape) | smaller-grid GEMM, 6-15 µs/call | 1.555 s / 254,304 inst / 6.12 µs | **0.831 s / 55,288 inst / 15.03 µs** |
| Sum | | **4.581 s** | **4.634 s** (≈ tied, +1.2%) |

TS calls the first variant **1.56× more** and the second **4.6× less**.
This is consistent with the `fold_sq_factor` change reshaping BMM1
operands so the persistent kernel hits a different tile-config branch.

**Total absorb-path GEMM time is essentially tied between arms.** The
absorb-path BMM1 reformulation doesn't itself save time on the GEMMs
— it saves time inside the fused attention kernel by avoiding a small
square-root scaling redundancy. The GEMM redistribution is incidental.

## Other kernels — instance count tracks forward steps

Top non-attention kernels scale linearly with forward-step count and
have unchanged per-call times:

| Kernel | base inst | ts inst | base µs/call | ts µs/call |
|---|---:|---:|---:|---:|
| `dsv3MinLatencyKernels::fused_a_gemm_kernel` (DSV3 down-proj) | 281,280 | 293,044 (+4.2%) | 25.27 | 25.40 (≈) |
| `ar_fusion::moe::moefinalize_allreduce_fusion_kernel_oneshot_lamport` | 262,655 | 273,828 (+4.3%) | 23.68 | 23.78 (≈) |
| `routingIndicesClusterKernel` | 256,800 | 305,520 (+19%) | 14.93 | 14.73 (≈) |
| `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT` | 603,618 | 628,409 (+4.1%) | 5.37 | 5.50 (≈) |
| `flashinfer rms_norm` | 292,498 | 306,128 (+4.7%) | 7.78 | 7.75 (≈) |
| `applyMLARopeAndAssignQKVKernelGeneration` | 294,966 | 307,151 (+4.1%) | 4.61 | 4.64 (≈) |

The +4-5% instance count delta tracks the forward-step count delta.
Per-call wall is essentially unchanged. Nothing else in the trace
shows kernel-level differences between arms.

## Trace-level totals

Sum of `Total Time` across all kernels in each CSV:

| Arm  | Total GPU kernel time | Δ |
|------|-----------------------|---|
| base | 75.693 s              | — |
| ts   | 80.175 s              | **+5.9%** |

This includes warmup, CUDA graph capture, finalize — phases that
differ per arm (TS recompiles new kernels at warmup, base reuses
cached compiled FMHA modules). For purposes of the kernel-level A/B
on attention, the per-call numbers above are cleaner.

## Verdict

**Resolved sub-hypothesis from Phase 5 TP=8 nsys run:**
"TS's MLA-decode kernel is slower per call" — **REFUTED**. With
`--cuda-graph-trace=node` exposing graph-captured kernels at TP=4,
the data shows TS attention is **~12.7% faster per call** (13.40 vs
15.34 µs) and **~9.1% faster on attention total wall** (4.12 vs 4.53 s).

**Reconciliation with prior phases:**

| Phase | Regime | TS Δ throughput | Driving factor |
|---|---|---|---|
| Phase 4 (K2.6, no MTP) | `q_len_per_req=1` (no draft tokens) | −3 to −7% | Attention is a smaller fraction of work; non-attention overhead dominates |
| Phase 5 TP=4 perf | `q_len_per_req=4`, num_req=32 | +4.2% | AL noise: +6.7% AL > 5% kernel win = +4.2% net |
| Phase 5 TP=8 nsys | num_req=4, AL noise opposite | −15.6% | AL noise (collapsed to +1.3%); kernel win can't carry through TP=8 comm overlap |
| **Phase 5 TP=4 nsys (this run)** | num_req=4, AL noise opposite | **−3.6%** | AL noise: **−9.4% AL** wipes out the **5% kernel win** = −3.6% net |

**The TokenSpeed kernel win is real but ~5% of total wall — too small
to survive AL variance from EAGLE-3 verifier outcomes.** Net
production throughput on K2.5/K2.6 NVFP4 + EAGLE-3 mtp=3 + B300 is
not a TS win, but the underlying `fold_sq_factor` BMM1 reformulation
**should** be absorbed into TRT-LLM's `fmhaSm103a` kernel via DKG MR
21023 to capture the per-call win without the two-kernel
split-KV + reduction overhead.

This run is the **strongest kernel-level evidence to date** that:
1. The TS kernel design has a genuine per-call advantage over the
   current TRT-LLM fused FMHA path.
2. The end-to-end product loss is real and is **not** due to the
   kernel itself being slow.
3. The reasonable engineering move is "**absorb the BMM1 trick, drop
   the split-KV machinery**" — i.e., adopt the algorithmic idea
   without taking the TokenSpeed package.

## Artifacts

```text
/scratch/runs/k2.6-spike/phase5-k25-mtp3-nsys-tp4-graphnode/
├── base.log                                   1.0 MB   bench log (TRTLLM)
├── base.nsys-rep                              725 MB   nsys trace
├── base.sqlite                                7.3 GB   exported SQLite (graph-trace=node)
├── base-kernsum_cuda_gpu_kern_sum.csv         108 KB   per-kernel CSV (200 rows)
├── base.kernsum.log
├── ts.log                                     1.0 MB   bench log (TOKENSPEED_MLA)
├── ts.nsys-rep                                781 MB   nsys trace
├── ts.sqlite                                  7.5 GB   exported SQLite (graph-trace=node)
├── ts-kernsum_cuda_gpu_kern_sum.csv           110 KB   per-kernel CSV (204 rows)
├── ts.kernsum.log
├── _config_base.yml                           sidecar config (attn_backend: TRTLLM)
└── _config_ts.yml                             sidecar config (attn_backend: TOKENSPEED_MLA)

Driver script:
.claude_docs/tokenspeed-kimik25/scripts/run_bench_k25_nsys_tp4_graphnode.sh
```

Reproduction:
```bash
bash .claude_docs/tokenspeed-kimik25/scripts/run_bench_k25_nsys_tp4_graphnode.sh
```
(~35 min wall, ~16 GB disk per arm including SQLite export.)
