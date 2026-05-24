# TokenSpeed MLA on Kimi K-series — Final Summary

**Date:** 2026-05-24
**JIRA:** TRTLLM-12510
**Branch:** `tokenspeed-kimik25-eval-public` rebased on `upstream/main@f278c4f170`
  + PR #14291 (FMHA JIT fix) + PR #9677 (Eagle MLA-based, `Eagle3MLAttention`)
**Hardware:** B300 SXM6 (sm_103a), single node, 8 GPUs

## Verdict in one line

**TokenSpeed's MLA-decode kernel does not produce an end-to-end perf
win on K2.5 / K2.6 against the post-#14291 TRT-LLM baseline at any
regime we tested on B300.** Adoption recommendation: **don't
default-on**. If the algorithmic idea (`fold_sq_factor` BMM1 reformulation)
is worth keeping, absorb it into TRT-LLM's fused FMHA kernel via DKG
MR 21023 patterns — don't take the TokenSpeed package.

**Kernel-level update (2026-05-24, Phase 5 nsys TP=4 with
`--cuda-graph-trace=node`):** TS's MLA-decode kernel IS measurably
faster per call (~12.7% per-call, ~9.1% on attention total wall on a
TP=4 num_req=4 sample). The end-to-end loss is from acceptance-length
volatility larger than the kernel saving, not from a slow kernel. The
algorithmic case for absorbing `fold_sq_factor` into the fused FMHA
path is therefore strengthened. See
`phase5-k25-eagle3-nsys-tp4-graphnode.md`.

## Result grid

| Phase | Model | Regime | TP | Bench | TS Δ throughput | Caveat |
|---|---|---|---|---|---|---|
| Phase 4 | K2.6 NVFP4 | `q_len_per_req=1` (no MTP) | 4 | 1k/1k conc=1, 16 reqs | **−3.6%** | clean |
| Phase 4 | K2.6 NVFP4 | `q_len_per_req=1` (no MTP) | 8 | 1k/1k conc=1, 16 reqs | **−7.1%** | clean |
| Phase 4 | K2.6 NVFP4 | `q_len_per_req=1` (no MTP) | 4 | 8k/1k conc=1, 16 reqs | **−3.4%** | clean |
| Phase 4 | K2.6 NVFP4 | `q_len_per_req=1` (no MTP) | 4 | 1k/1k conc=16, 256 reqs | **+0.6% (tied)** | clean |
| Phase 5 | K2.5 NVFP4 | `q_len_per_req=4` (EAGLE-3 mtp=3) | 4 | 1k/1k conc=2, 32 reqs | **+4.2%** | acceptance-rate confound (+6.7% AL gap) |
| Phase 5 | K2.5 NVFP4 | `q_len_per_req=4` (EAGLE-3 mtp=3) | 8 | 1k/1k conc=2, 4 reqs (nsys) | **−15.6%** | acceptance gap collapsed to +1.3% |
| Phase 5 nsys TP=4 graph-trace=node | K2.5 NVFP4 | `q_len_per_req=4` (EAGLE-3 mtp=3) | 4 | 1k/1k conc=2, 4 reqs | **−3.6%** | AL flipped to **−9.4%**; **TS attn kernel −12.7%/call, −9.1% on attn total wall** |

**The headline TokenSpeed "+10% decode-latency / ~9% min-latency
bs=1 / ~11% throughput at ~100 TPS/user" claims do not reproduce on
B300 sm_103a against the post-#14291 TRT-LLM FMHA path on K2.5/K2.6.**

## Why TS appeared to win at Phase 5 TP=4 but lost at Phase 5 TP=8

| Layer | Phase 5 perf (TP=4) | Phase 5 nsys (TP=8) |
|---|---|---|
| TS throughput Δ | +4.2% | −15.6% |
| AL Δ (TS over base) | +6.7% (3.18 vs 2.98) | +1.3% (3.07 vs 3.03, ≈ tied) |
| Estimated kernel Δ (excluding AL) | ~−2 to −3% per forward | not cleanly isolable |

The +4.2% TP=4 win was an **acceptance-rate artifact**, not a kernel
win. The TS MLA kernel's known parity divergence (max abs 0.33, max
rel ~1166× from the DSV3-Lite spike) produces slightly different
logits that happened to please the EAGLE-3 verifier on K2.5 at TP=4,
giving a +6.7% AL bonus that overcame the underlying kernel
disadvantage.

At TP=8 the AL gap collapsed to noise (+1.3%) and the real throughput
delta surfaced: TS is **−15.6% slower end-to-end**.

## What's actually happening per the nsys A/B (Phase 5 nsys verification)

Verified with `nsys stats --report cuda_api_sum`, `cuda_gpu_kern_sum`,
and graph-related API queries on both arms' traces:

| Sub-hypothesis | Status | Evidence |
|---|---|---|
| TS has more launch overhead | **REFUTED** | TS spends 43 sec LESS on `cudaLaunchKernel*` across whole trace |
| TS falls out of CUDA graph capture | **REFUTED** | Both arms have exactly 48,340 `cudaGraphLaunch` instances |
| TS's MLA-decode kernel is slower per call | **REFUTED (Phase 5 nsys TP=4 graph-trace=node)** | With graph-trace=node exposing per-kernel timings: TS attention is **−12.7% per call** (13.40 µs vs 15.34 µs) and **−9.1% on attention total wall** (4.12 vs 4.53 s); TS does ~4-5% more attn calls due to more forward steps from worse AL in this sample |
| TS has more GPU idle time during timed run | **PLAUSIBLE** | TS's faster kernels in isolation unbalance the compute/comm overlap pattern; allreduce kernels show 2.27× different per-call durations between arms (`ar_fusion_kernel<pattern=0>`: base 36.99 ms/call vs ts 16.30 ms/call) — consistent with base hiding allreduce behind longer compute, TS exposing it |

The TP=4 nsys run with `--cuda-graph-trace=node` (added 2026-05-24)
resolves the previous INCONCLUSIVE entry: **TokenSpeed's MLA-decode
kernel IS faster per call**, by ~12.7%. The end-to-end product still
isn't a win because the kernel saving (~5% of total wall, ~0.4 s on
this 8-9 s run) is smaller than AL variance from EAGLE-3 verifier
outcomes (±10% sample-to-sample at small N). See
`phase5-k25-eagle3-nsys-tp4-graphnode.md` for the full per-kernel diff.

## Cross-phase findings that hold up

1. **PR #14291 (FMHA JIT fix) closed the gap that the DSV3-Lite spike
   originally measured.** The pre-#14291 TRT-LLM FMHA path was the
   baseline TokenSpeed was beating by ~10% on DSV3-Lite. Once that
   fix landed, TRT-LLM's MLA-decode is competitive or better than
   TokenSpeed in every regime we tested on K-series.
2. **TokenSpeed-as-shipped does not improve K2.5/K2.6 production
   throughput on B300.** Even at the regime where the algorithm
   should shine (`q_len_per_req=4` with EAGLE-3 mtp=3), the
   end-to-end throughput is either tied or worse.
3. **The acceptance-rate gap at Phase 5 TP=4 is suspicious and was
   not portable to TP=8.** Likely a numerical-divergence artifact of
   the TS kernel rather than a stable algorithmic benefit. **Output
   sanity check (Phase 5 punted) is the open question for whether the
   TS kernel produces sensible text on K2.5.**

## What's still worth doing (if anyone picks this up)

1. **Output sanity check** — generate the same fixed-seed prompts on
   both arms at TP=4 mtp=3 and diff the text. If TS produces gibberish
   or low-quality output, the +6.7% AL gap was numerical noise, not a
   real win. Phase 5 was perf-only by user direction; this is the
   missing piece.
2. **Absorb `fold_sq_factor` into TRT-LLM's FMHA** — DKG MR 21023 has
   the pattern. Adopt the BMM1 grouping inside the existing fused
   `fmhaSm103a` kernel without taking the split-KV+reduction
   2-kernel design. This **preserves the algorithmic idea while
   avoiding the comm/compute overlap issue** that hurts TS-as-shipped.
3. **(For Fireworks / Together customers asking about TokenSpeed):**
   the open-source decode kernel (`mla_decode_fp8.py`) is at
   https://github.com/lightseekorg/tokenspeed/tree/main/tokenspeed-mla.
   The binary prefill kernel is **not in the repo** — only the open-source
   prefill variant is shipped (and per LightSeek's own README, "the
   open-source version is slightly slower than TensorRT-LLM's
   native implementation"). The performant prefill kernel uses
   NVIDIA-internal SASS knobs that aren't fit for open release. NVIDIA
   does not have privileged access to the closed binary either.

## Artifacts

```
.claude_docs/tokenspeed-kimik25/
├── final-summary.md              ← THIS FILE
├── understanding.md              ← original goal / risk register
├── runbook.md                    ← repro instructions
│
├── nvbug-draft-nvrtc-baseline.md ← Phase 4 NVRTC bug (archived; fixed by PR #14291)
├── nvrtc-rebase-verify.md        ← PR #14291 verification
│
├── phase4-summary.md             ← Phase 4 K2.6 TS-only (rc14 build, base crashed)
├── phase4-rebased-ab.md          ← Phase 4 K2.6 A/B (rebased build) — main K2.6 result
│
├── phase5-plan.md                ← Phase 5 plan (K2.5 EAGLE-3 setup)
├── phase5-k25-eagle3.md          ← Phase 5 perf-only K2.5 EAGLE-3 mtp=3 TP=4 — main K2.5 result
├── phase5-k25-eagle3-nsys.md     ← Phase 5 nsys A/B + kernel/graph hypothesis testing
├── phase5-k25-eagle3-nsys-tp4-graphnode.md  ← Phase 5 nsys TP=4 + cuda-graph-trace=node — per-kernel A/B
│
├── bench-config.yml                ← K2.6 TP4 1k/1k conc=1
├── bench-config_base.yml           ← K2.6 TP4 1k/1k conc=1 (TRTLLM sidecar)
├── bench-1k1k_tp8_conc1.yml        ← K2.6 TP8
├── bench-1k1k_tp4_conc16.yml       ← K2.6 conc=16
├── bench-8k1k_tp4_conc1.yml        ← K2.6 8k context
├── bench-k25-mtp3.yml              ← K2.5 EAGLE-3 mtp=3 TP=4
├── bench-k25-mtp3-tp8.yml          ← K2.5 EAGLE-3 mtp=3 TP=8 (nsys A/B)
│
└── scripts/
    ├── minimal_bench.py            ← K2.6 bench harness (bypasses K2.6 tokenizer)
    ├── run_bench_v2.sh             ← Phase 3 old driver
    ├── run_bench_v3.sh             ← Phase 4 K2.6 A/B driver
    ├── run_bench_k25.sh            ← Phase 5 K2.5 A/B driver (trtllm-bench)
    ├── run_bench_k25_nsys.sh       ← Phase 5 nsys A/B driver
    └── patch_k25_bf16kv.sh         ← K2.5 BF16-KV patch script

Runtime artifacts (~30 GB total):
/scratch/runs/k2.6-spike/
├── phase4-bench/                   ← old rc14 TS-only logs (Phase 4 pre-rebase)
├── phase4-rebased-bench/           ← Phase 4 rebased A/B logs (TRTLLM baseline runs)
├── phase5-k25-mtp3/                ← Phase 5 perf-only logs
└── phase5-k25-mtp3-nsys/           ← Phase 5 nsys A/B (incl 1.2 GB nsys-rep + 15 GB sqlite per arm)
```

## Phase summary table for the impatient

| Phase | What we did | Key finding | Status |
|---|---|---|---|
| Phase 3 | DSV3-Lite spike (companion repo) | TS +10% on FlashInfer/trtllm-gen pre-#14291 | done, archived |
| Phase 4 (K2.6) | Rebuilt rebased TRT-LLM + 4-config A/B | TS −3 to −7% at conc=1, tied at conc=16 | done |
| Phase 5 perf | K2.5 NVFP4 + EAGLE-3 mtp=3 TP=4 | TS +4.2% (acceptance confound, not kernel) | done |
| Phase 5 nsys | K2.5 EAGLE-3 mtp=3 TP=8, pure-kernel A/B | TS −15.6% when AL gap collapses; launch overhead + CUDA graphs are NOT the cause; timed-run kernel-level attribution requires `--cuda-graph-trace=node` | done (verdict clear despite per-kernel inconclusive) |
| Phase 5 nsys TP=4 graph-trace=node | K2.5 EAGLE-3 mtp=3 TP=4, per-kernel A/B | **TS attn kernel −12.7% per call** (13.40 vs 15.34 µs); **−9.1% on attn total wall** (4.12 vs 4.53 s). End-to-end −3.6% because AL=−9.4% (this sample's AL swung the opposite direction vs num_req=32). Absorb-path GEMM time tied (+1.2%). Trace-total GPU time +5.9% (warmup + more forward steps). | done — confirms kernel-level TS advantage exists but is too small to survive AL noise |
| Phase 5 output check | (not run, deferred per user) | open question on TS kernel output quality | OPEN |
