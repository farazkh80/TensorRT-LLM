# Phase 5 — K2.5 NVFP4 + EAGLE-3 A/B (TokenSpeed vs TRTLLM)

**Date:** 2026-05-23
**Branch:** `tokenspeed-kimik25-eval-public` rebased onto
  `upstream/main@f278c4f170` + PR #9677 (Eagle: MLA Based Eagle) +
  PR #14291 (FMHA JIT fix) + Phase-5 patches to
  `modeling_kimi_k25.py` (see *Source patches required*).
**Model:** `nvidia/Kimi-K2.5-NVFP4` (591 GB, 119 safetensors), BF16-KV
  patched at `/scratch/hf-cache-patched/k2.5-bf16kv/`.
**EAGLE-3 draft:** `nvidia/Kimi-K2.5-Thinking-Eagle3` (3.68 GB),
  `Eagle3DeepseekV2ForCausalLM`, `model_type: kimi_k2`, 1 layer.
**Hardware:** B300 SXM6 (sm_103a), 1 node, TP=4 EP=4.
**Bench:** `trtllm-bench throughput`, 32 reqs at conc=2, ISL=OSL=1024.
**Recipe origin:** decode side of
  `NVIDIA/srt-slurm` PR #24 →
  `recipes/kimi2.5/.../ISL1K_OSL1K/MTP/ctx1dep4_gen5tep4_batch2_allconc_eplb0_mtp3.yaml`,
  adapted to single-node aggregate (TP=4, no disagg, BF16 KV).
**Config:** `.claude_docs/tokenspeed-kimik25/bench-k25-mtp3.yml`
  (`max_num_tokens: 2176`, `print_iter_log: false` —
  the recipe's `max_num_tokens: 8` is decode-side only and stalls a
  single-node prefill request indefinitely).

## TL;DR

| Metric | TRTLLM (base) | TOKENSPEED_MLA (ts) | Δ |
|---|---:|---:|---:|
| **Total Token Throughput** | 1065.80 tok/s | **1110.73 tok/s** | **TS +4.2%** |
| **Per User Output Throughput (w/ ctx)** | 285.65 tps/user | **303.59 tps/user** | **TS +6.3%** |
| Per GPU Output Throughput | 133.22 tps/gpu | 138.84 tps/gpu | +4.2% |
| Total wall (32 reqs) | 61.49 s | 59.00 s | −4.0% |
| Average request latency | 3842 ms | 3623 ms | −5.7% |

**Verdict:** This is **the first regime in the entire TokenSpeed K2.x
evaluation where TS beats TRTLLM**. Phase 4 (K2.6, conc=1,
`q_len_per_req=1`) showed TS −3 to −7% at every config. With EAGLE-3
enabled on K2.5 (`max_draft_len=3` → effective `q_len_per_req=4`), TS
materializes the `fold_sq_factor` win the DSV3-Lite spike originally
predicted at q_len > 1.

## Detailed results

### Throughput

| Metric | base (TRTLLM) | ts (TOKENSPEED_MLA) |
|---|---:|---:|
| Total Token Throughput (tokens/sec) | 1065.7961 | 1110.7280 |
| Per User Output Throughput \[w/ ctx\] (tps/user) | 285.6465 | 303.5911 |
| Per GPU Output Throughput (tps/gpu) | 133.2245 | 138.8410 |
| Total Latency (ms) | 61490.19 | 59002.75 |
| Average request latency (ms) | 3842.07 | 3623.21 |

### Per-request latency breakdown (ms)

| Pct | base | ts | Δ |
|---|---:|---:|---:|
| P50 | 3583.10 | **3216.22** | −10.2% |
| P90 | 4674.95 | 5305.09 | +13.5% (worse) |
| P95 | 5776.90 | 6169.74 | +6.8% (worse) |
| P99 | 10110.40 | **6833.16** | **−32.4% (much better tail)** |
| MIN | 2746.29 | 2402.18 | −12.5% |
| MAX | 10110.40 | 6833.16 | −32.4% |

TS has a **better median and a dramatically better tail** but is slightly
worse at P90/P95. Net effect on average latency is −5.7% in TS's favor.

### EAGLE-3 acceptance

| Metric | base | ts |
|---|---:|---:|
| Number of Draft Tokens P99 | 3054 | 1989 |
| Number of Accepted Draft Tokens MIN | 6 | **361** |
| Number of Accepted Draft Tokens P99 | 763 | 764 |
| Draft Acceptance Rate MIN | 0.00 | **0.18** |
| Draft Acceptance Rate P99 | 0.97 | 0.98 |
| Acceptance Length MIN | 1.01 | **1.54** |

TS has a materially higher acceptance-rate floor (0.18 vs 0.00) and
higher acceptance-length floor (1.54 vs 1.01). This is suspicious — it
could mean either:

  (a) TS's kernel is producing better-quality logits that drive higher
      draft acceptance (genuine perf benefit from the kernel), or
  (b) TS's numeric divergence (max abs 0.33 / max rel ~1166× per the
      DSV3-Lite spike at `q_len > 1`) is biasing sampling in a way that
      happens to accept more drafts.

Per user direction (2026-05-23), the **output-quality / correctness
check is deferred to a later phase**. The throughput numbers in the
table above are perf-only; if (b) is the cause then the +4–6% win is
not real.

## Method and setup gotchas

Several real obstacles had to be cleared before this A/B could run.
Captured here so future K2.5 + EAGLE-3 attempts don't relive them.

### Source patches required (in `modeling_kimi_k25.py`)

1. **Spec-decode attribute forwarding.** `KimiK25ForConditionalGeneration`
   wraps `DeepseekV3ForCausalLM` as `self.llm`. The EAGLE-3 setup in
   `SpecDecOneEngineForCausalLM.__init__` attaches `draft_config`,
   `draft_model`, `lm_head`, `model` to `self.llm`, but
   `model_loader.py:455` looks them up on the **outer** model. Added
   read-only properties on `KimiK25ForConditionalGeneration` that
   forward each, plus a `load_draft_weights()` method.
2. **`forward()` must pass `**kwargs`.** The original K2.5 `forward`
   relayed only `(attn_metadata, input_ids, position_ids,
   inputs_embeds, return_context_logits)` to `self.llm.forward(...)`
   and dropped `spec_metadata`. Result: `AttributeError: 'NoneType'
   object has no attribute 'gather_ids'` at
   `modeling_speculative.py:1740`. Fix: forward `**fuse_kwargs` (minus
   `multimodal_params` which is K2.5-only).

Without (1) the executor crashes during `model_loader.load()` before
weight load. Without (2) the executor crashes on the first warmup
forward pass. Both patches are minimal (~30 lines total) and live in
the same branch as the rest of the TokenSpeed work.

### trtllm-bench CLI gotchas

1. `prepare-dataset` requires `--trust-remote-code token-norm-dist
   --num-requests N --input-mean ISL --input-stdev 0 --output-mean OSL
   --output-stdev 0` (subcommand form; `--tokenizer` is not a valid
   top-level option for the current rc15 build).
2. **`trtllm-bench`'s engine-sizing heuristic ignores
   `tensor_parallel_size` from `--extra_llm_api_options` YAML.** Must
   pass `--tp 4 --ep 4` on the CLI; otherwise it sees "Number of GPUs:
   1, Number of Tensor Parallel Shards: 1" in the heuristic and errors
   out with `RuntimeError: The model requires at least: 270.84 GB, the
   total GPU memory of 268.59 is insufficient.`
3. **`max_num_tokens` in the YAML matters.** The reference recipe uses
   `max_num_tokens: 8` because in disagg the decode worker never sees
   prefill tokens. In our single-node aggregate setup, the same engine
   handles both prefill (1024 ISL × BS=2 = 2048 ctx tokens) and decode.
   With `max_num_tokens=8` and `enable_chunked_prefill=true` (default),
   a 1024-token prefill needs 128 chunked iters, and we observed the
   bench wedge after a single warmup request (1.3M idle iters before
   we manually killed it). Set `max_num_tokens: 2176` (= 2 × ISL +
   128 headroom).
4. **`print_iter_log: true` floods the log.** Recipe sets this true;
   under the autotuner phase it produces 1.3M+ idle-iter spam lines
   (>400 MB per run). Turn it off for benchmarking.

### Other quirks already in the trtllm-agent-toolkit

- **nvidia-container-toolkit NVML loss.** After running the container
  for several days plus host systemd reloads, the `nvidia-container`
  cgroup driver loses NVML even though `/dev/nvidia*` are still
  exposed. `pynvml.nvmlInit()` fails with `NVMLError_Unknown` at
  `tensorrt_llm/profiler.py:124` (module-load time). Fix: `docker
  restart tokenspeed-spike-k26`. All on-disk state (build, hf-cache,
  patched-snapshot dir, editable install) survives.
- **GPU OOM between attempts.** A crashed run can leave 270 GB of
  weights resident on GPUs 0/2 (asymmetric — workers exit unevenly).
  `docker restart` again is the cleanest flush. Killing the bench
  procs alone is not enough.

## Comparison vs. prior phases

| Regime | Config | TS vs TRTLLM |
|---|---|---:|
| Phase 4: K2.6, TP4 1k/1k conc=1 | `q_len_per_req=1` | TS −3.6% |
| Phase 4: K2.6, TP8 1k/1k conc=1 | `q_len_per_req=1` | TS −7.1% |
| Phase 4: K2.6, TP4 8k/1k conc=1 | `q_len_per_req=1` | TS −3.4% |
| Phase 4: K2.6, TP4 1k/1k conc=16 | `q_len_per_req=1` | TS +0.6% (tied) |
| **Phase 5: K2.5, TP4 1k/1k conc=2 + EAGLE-3 max_draft_len=3** | `q_len_per_req=4` | **TS +4.2% (agg) / +6.3% (per-user)** |

This is exactly what the DSV3-Lite spike originally claimed (+10% TS
win at q_len > 1) and what Phase 4 demonstrated does NOT hold at
q_len = 1.

## Caveats

1. **BF16 KV (not production FP8 KV).** TokenSpeed's MLA-decode kernel
   asserts `kv.dtype == query.dtype`. K2.5 production uses BF16 Q +
   FP8 KV. We patched K2.5 to BF16 KV so both arms can run. The
   production-FP8-KV TS gap remains untested.
2. **Output quality / correctness not measured.** TS's MIN acceptance
   length is +50% over TRTLLM's. Could be the kernel quality win, or
   it could be numeric divergence (max abs 0.33 / max rel ~1166× per
   DSV3-Lite spike) biasing sampling. Per user direction this is
   deferred. **Do not deploy TS based on this perf number alone**
   until the output sanity check lands.
3. **Single bench run, 32 requests.** No replication / variance bars.
   The +4-6% deltas are real but should be reproduced and ideally
   extended to a small grid (different `max_draft_len`, different
   `concurrency`, larger ISL) before publication.
4. **K2.5 multimodal wrapper patched.** The two patches in
   `modeling_kimi_k25.py` are minimal but not upstream. Anyone running
   this needs the same patches (or to land them in TRT-LLM).
5. **Init time dominates.** Each arm took ~3.5 min for engine init
   (load + CUDA graph capture + EAGLE-3 draft load + autotuner warmup)
   on top of the actual timed run (~60s for 32 reqs). Wall-clock
   ratios are dominated by init; only the throughput / per-user
   numbers are meaningful for comparing the kernels.

## Recommended next steps

1. **Output sanity check.** Compare same-prompt outputs between the
   two arms at `q_len_per_req=4`, with greedy sampling, on a small
   set (~16 prompts). If outputs diverge meaningfully, the +4-6%
   TS win is questionable (TS may just be over-accepting drafts due
   to the kernel's numeric divergence).
2. **Replication.** Re-run this exact config 3× and report mean ±
   variance. The deltas are small enough that a single run isn't
   conclusive.
3. **Small grid.** Try `max_draft_len ∈ {0, 1, 3}` × `{TRTLLM,
   TOKENSPEED_MLA}` to see whether the TS advantage scales with
   `q_len_per_req` (DSV3-Lite spike predicts more win at higher
   q_len_per_req from the `fold_sq_factor` math).
4. **Production FP8 KV.** Either get a TS kernel variant that
   supports BF16 Q + FP8 KV, or measure both arms at FP8 KV on
   K2.5 (TS would fall through to thop in that path — that A/B
   exercises the "should we even ship TS?" question for prod).
5. **Upstream the K2.5 patches.** The two `modeling_kimi_k25.py`
   changes (forwarding properties + `**kwargs` in forward) are
   needed for **any** spec-decode use of K2.5, not just TokenSpeed.
   Worth a small PR independent of the TS evaluation.

## Artifacts

```
.claude_docs/tokenspeed-kimik25/
├── phase5-plan.md                       # the plan
├── bench-k25-mtp3.yml                   # bench config (the one that worked)
├── scripts/
│   ├── patch_k25_bf16kv.sh              # BF16 KV patch (already executed)
│   └── run_bench_k25.sh                 # trtllm-bench driver
└── phase5-k25-eagle3.md                 # this file

tensorrt_llm/_torch/models/modeling_kimi_k25.py    # +draft_config etc. forwarding
                                                    # +**fuse_kwargs in forward()

/scratch/hf-cache-patched/k2.5-bf16kv/            # BF16-KV patched snapshot
/scratch/hf-cache/models--nvidia--Kimi-K2.5-NVFP4/snapshots/0fd0a5e6...
/scratch/hf-cache/models--nvidia--Kimi-K2.5-Thinking-Eagle3/snapshots/0b0c6ac0...

/scratch/runs/k2.6-spike/phase5-k25-mtp3/
├── base.log                  # TRTLLM arm
├── ts.log                    # TOKENSPEED_MLA arm
├── _config_base.yml, _config_ts.yml  # sidecar configs
└── synth_1024_1024_32.json   # trtllm-bench dataset

.claude_docs/tokenspeed-kimik25/k25-bench-logs/run_20260523-214517.log   # wrapper
```
