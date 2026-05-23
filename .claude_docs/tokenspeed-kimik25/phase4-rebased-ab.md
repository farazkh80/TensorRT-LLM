# Phase 4 A/B re-run on rebased TRT-LLM (post-PR-#14291)

**Date:** 2026-05-20
**Branch:** `tokenspeed-kimik25-eval-public` rebased onto
  `upstream/main@f278c4f170` + rebuilt `libtensorrt_llm.so` (build time ~47 min)
**Model:** `/scratch/hf-cache-patched/k2.6-bf16kv` (BF16-KV-patched K2.6 NVFP4)
**Hardware:** B300 SXM6 (sm_103a)
**Driver:** `.claude_docs/tokenspeed-kimik25/scripts/run_bench_v3.sh`
**Logs:** `/scratch/runs/k2.6-spike/phase4-rebased-bench/<config>/{base,ts}.log`

## TL;DR

| Config | TP / conc | TRTLLM | TOKENSPEED_MLA | TS/TRTLLM | ITL (TRTLLM / TS) | Per-user (TRTLLM / TS) |
|---|---|---:|---:|---:|---:|---:|
| `bench-config` | TP4 1k/1k conc=1 | **158.5** | 152.8 | **0.964** (−3.6%) | 6.31 / 6.54 ms | 158.65 / 153.00 |
| `bench-1k1k_tp8_conc1` | TP8 1k/1k conc=1 | **182.2** | 169.3 | **0.929** (−7.1%) | 5.49 / 5.91 ms | 182.26 / 169.16 |
| `bench-8k1k_tp4_conc1` | TP4 8k/1k conc=1 | **152.0** | 146.9 | **0.966** (−3.4%) | 6.58 / 6.81 ms | 152.42 / 146.93 |
| `bench-1k1k_tp4_conc16` | TP4 1k/1k conc=16 | 1239.3 | **1246.8** | **1.006** (+0.6%) | 0.81 / 0.80 ms | 77.37 / 77.74 |

**Headline:** On K2.6 against the post-#14291 TRTLLM backend, TokenSpeed
is **3–7% slower at conc=1** and **tied at conc=16** (within noise). The
DSV3-Lite spike's +10% TS advantage was against the *pre-#14291* FMHA path
on a smaller model. PR #14291's new FMHA JIT path closed the gap that
TokenSpeed previously exploited.

## What changed vs. Phase 4 (TS-only on old build)

Phase 4's TRTLLM baseline could not run (NVRTC crash). This re-run
produces the first clean apples-to-apples A/B.

| Config | Phase 4 TS (old) | Rebased TRTLLM | Rebased TS |
|---|---:|---:|---:|
| TP4 1k/1k conc=1 | 151.8 | **158.5** (+4.4%) | 152.8 (+0.7%) |
| TP8 1k/1k conc=1 | 169.4 | **182.2** (+7.6%) | 169.3 (−0.1%) |
| TP4 8k/1k conc=1 | 145.5 | **152.0** (+4.5%) | 146.9 (+1.0%) |
| TP4 1k/1k conc=16 | 1241.4 | **1239.3** (−0.2%) | 1246.8 (+0.4%) |

Observations:

1. **TokenSpeed-arm performance is stable** across rebuild: rebased TS ≈
   old-build TS at every config (within ±1%). No regression introduced by
   the rebase / rebuild on the TS side.
2. **Rebased TRTLLM is markedly faster than old-build TS** at conc=1
   (+4.4 – 7.6%) — this is where PR #14291's FMHA JIT improvements show
   up.
3. **At conc=16 the two arms converge** — both ~1240 tok/s aggregate,
   both ~77 tok/s per-user, both ~0.81 ms ITL. The TS dispatch advantage
   (a single fused CuTe kernel) and the TRTLLM batched path produce
   essentially the same throughput once the engine is fed enough work to
   amortize per-request overheads.

## What this means for the K2.6 TokenSpeed deployment story

- **Conservative `q_len_per_req=1` decode regime:** TokenSpeed offers no
  performance benefit on K2.6 today. It is in fact a small regression at
  conc=1.
- **Throughput regime (conc=16, ~100 TPS/user target):** TokenSpeed is
  within noise of TRTLLM — neither clearly better nor worse.
- **Headline 2× MTP / spec-decode regime:** **NOT MEASURED** here. K2.6's
  current checkpoint has `num_nextn_predict_layers=0`, so the
  `q_len_per_req=4` / `fold_sq_factor` regime where the DSV3-Lite spike
  showed the largest TS advantage is unreachable. Until a K2.6 variant
  with MTP weights exists (or Eagle is attached), the headline claim is
  not testable here.

## Method summary

- 4 configs, 2 arms each, sequential runs (couldn't parallelize because
  TP8 takes all 8 GPUs).
- For each `bench-*.yml`, `run_bench_v3.sh` sed's two sidecar configs
  (`attn_backend: TRTLLM` and `attn_backend: TOKENSPEED_MLA`) and runs
  `minimal_bench.py` with `num_requests = 16 * concurrency` (16 for
  conc=1, 256 for conc=16).
- `minimal_bench.py` uses `skip_tokenizer_init=True` + token-ID prompts
  to bypass K2.6's broken HF AutoTokenizer.
- LLM init for each config: ~250–280 s (model load + CUDA graph capture
  + FMHA NVRTC compile of unique shapes). No NVRTC failure on either arm
  in any of the 8 runs — confirms the PR #14291 fix.

## Caveats (still apply)

1. **No MTP / spec-decode regime measured.** As above.
2. **BF16 KV only.** K2.6 production FP8 KV is gated by the
   TokenSpeed-side `kv.dtype == q.dtype` assertion. Production FP8 KV vs
   either arm is not tested here.
3. **TTFT not measured.** `minimal_bench.py` is non-streaming.
4. **N=1 timed measurement per config.** No replication / variance bars.
   Differences <2% should be treated as within noise; the −3 to −7%
   conc=1 deltas are real but should be reproduced before publication.

## Artifacts

```
.claude_docs/tokenspeed-kimik25/
├── scripts/run_bench_v3.sh              # this run's driver
├── scripts/minimal_bench.py             # bench harness (unchanged)
├── bench-config.yml                     # TP4 1k/1k conc=1
├── bench-1k1k_tp8_conc1.yml             # TP8 1k/1k conc=1
├── bench-8k1k_tp4_conc1.yml             # TP4 8k/1k conc=1
├── bench-1k1k_tp4_conc16.yml            # TP4 1k/1k conc=16
├── phase4-summary.md                    # original Phase 4 (TS-only)
├── nvrtc-rebase-verify.md               # PR #14291 fix verification
├── nvbug-draft-nvrtc-baseline.md        # archived NVBugs draft
└── phase4-rebased-ab.md                 # this file

/scratch/runs/k2.6-spike/phase4-rebased-bench/
├── bench-config/{base,ts}.log + summary.txt
├── bench-1k1k_tp8_conc1/{base,ts}.log + summary.txt
├── bench-8k1k_tp4_conc1/{base,ts}.log + summary.txt
└── bench-1k1k_tp4_conc16/{base,ts}.log + summary.txt
```

## Recommended next steps

1. **Retire / not-default TokenSpeed for K2.6 conc=1 deployment.** It is
   a regression at the q_len_per_req=1 regime that K2.6 currently lives
   in.
2. **Re-test once a K2.6-with-MTP variant exists.** The `q_len_per_req
   > 1` regime is the only one where the headline TS advantage was ever
   measured (DSV3-Lite spike step 7). Until that exists, the K2.6
   TokenSpeed story is settled at "no benefit / mild regression."
3. **Optionally: nsys A/B at TP4 conc=1 1k/1k.** Confirm that the
   slowdown is in the MLA-decode kernel itself, not in surrounding
   plumbing — useful evidence if filing a TS bug or proposing a TS
   improvement.
4. **Add TTFT to `minimal_bench.py`.** The conc=16 measurement would
   benefit from a streaming-mode result; tied ITL doesn't tell us
   whether TS / TRTLLM differ on TTFT.
