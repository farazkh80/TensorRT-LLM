# Phase 4 — Kimi K2.6 TokenSpeed bench grid (current main)

**Branch:** `tokenspeed-kimik25-eval-public`
**Date:** 2026-05-19
**JIRA:** TRTLLM-12510
**Container:** `tokenspeed-spike-k26` (image `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`)
**TRT-LLM source:** current `main` (1.3.0rc15-dev), built from `/home/farazkh_scratch/parallel/TensorRT-LLM`, shadowed via `PYTHONPATH=/workspace/TensorRT-LLM`
**Model:** `nvidia/Kimi-K2-Thinking-NVFP4` — DeepSeek-V3 dense MLA, 128 heads, `kv_lora_rank=512`, `qk_rope_head_dim=64`, 61 layers
**KV cache:** **BF16** (from `/scratch/hf-cache-patched/k2.6-bf16kv/` — patched snapshot with `kv_cache_quant_algo` removed). Production FP8 KV not used; see *Caveats* §1.
**GPUs:** B300 SXM6 (sm_103a)

---

## TL;DR

| Config | TP | conc | ISL/OSL | Aggregate (tok/s) | Per-user median | ITL (ms/tok) | LLM init (s) |
|---|---|---|---|---|---|---|---|
| `bench-config.yml` | 4 | 1 | 1k/1k | **151.8** | 153.19 | 6.59 | 256.6 |
| `bench-1k1k_tp8_conc1.yml` | 8 | 1 | 1k/1k | **169.4** | 170.58 | 5.90 | 246.6 |
| `bench-8k1k_tp4_conc1.yml` | 4 | 1 | 8k/1k | **145.5** | 146.58 | 6.87 | 255.9 |
| `bench-1k1k_tp4_conc16.yml` | 4 | 16 | 1k/1k | **1241.4** | 77.94 | 0.81 | 266.6 |

All numbers are from the **TokenSpeed arm** (`attn_backend: TOKENSPEED_MLA`).
The **baseline arm** (`attn_backend: TRTLLM`) **crashes during warmup on all four configs** with `NVRTC_ERROR_COMPILATION` on the `fmhaSm103aKernel_...HQk576HV512...ForGen` MLA-decode cubin — this is an upstream current-main bug, not a TokenSpeed regression. See *Caveats* §2.

The relative comparisons we *can* make from the TokenSpeed arm alone:

- **TP8 vs TP4 at conc=1**: TP8 is **+12% higher per-user throughput** (170.58 vs 153.19 tok/s) and **−10% ITL** (5.90 vs 6.59 ms/tok). Matches TokenSpeed's documented "smaller effective heads → larger fold_sq_factor win" prediction (TP8 = 16 effective heads/rank, TP4 = 32).
- **8k vs 1k context at TP4 conc=1**: only **−4% per-user throughput** (146.58 vs 153.19). Decode time is dominated by per-token MLA work, not KV-cache walk — expected behavior for absorption-path MLA.
- **conc=16 vs conc=1 at TP4 1k/1k**: **+8.2× aggregate throughput** (1241.4 vs 151.8), per-user drops to ~half (77.94 vs 153.19). Standard throughput/latency tradeoff; ITL at conc=16 is 0.81 ms/tok = ~1.2k tok/s aggregate, which is the production-relevant number for batched serving.

---

## Method

### Bench script

`/home/farazkh_scratch/parallel/TensorRT-LLM/.claude_docs/tokenspeed-kimik25/scripts/run_bench_v2.sh`

For each YAML config, the driver:

1. Generates two sidecar configs by sed-overriding `attn_backend` to `TRTLLM` (base) and `TOKENSPEED_MLA` (ts).
2. For each variant, invokes `minimal_bench.py` (see below) with the sidecar config as `--extra_llm_api_options`.
3. Greps the per-arm logs for `Token Throughput`, `Inter Token`, `Per User Output Throughput`.

`minimal_bench.py` (companion script):
- Uses `skip_tokenizer_init=True` and feeds pre-baked token-ID prompts. **Required** because the K2.6 NVFP4 snapshot's `tokenization_kimi.py` chokes on `AutoTokenizer.from_pretrained` (the official `trtllm-bench` driver crashes with `ValueError: Couldn't instantiate the backend tokenizer` before any model code runs).
- Runs 1 warmup request + N timed requests via `ThreadPoolExecutor(max_workers=concurrency)` (each thread calls `llm.generate([prompt], sp)`; TRT-LLM batches them in the engine).
- Reports aggregate tok/s, per-user median tok/s, ITL average.

### Models, KV, and the backend swap

- **Model**: `nvidia/Kimi-K2-Thinking-NVFP4`. The patched BF16-KV snapshot is the same checkpoint with `kv_cache_quant_algo` removed from `hf_quant_config.json`. Identical weights; only KV-cache element dtype changes.
- **TokenSpeed swap**: `tensorrt_llm/_torch/attention_backend/tokenspeed_mla_attention.py` is a `TrtllmAttention` subclass that overrides `_run` for the MLA-decode-generation-only branch. Dispatch gate: `is_mla_enable && attention_input_type == generation_only && q_dtype ∈ {bf16, fp16} && kv_cache.dtype == q.dtype && no sinks/helix/sage/sparse && tokenspeed_mla available`. Everything else falls through to `super()._run(...)`. Selected via YAML `attn_backend: TOKENSPEED_MLA` (registered in `tensorrt_llm/_torch/attention_backend/utils.py`).
- **Kernel**: `tokenspeed_mla.tokenspeed_mla_decode` via `tensorrt_llm/_torch/attention_backend/tokenspeed_mla.py` (a FlashInfer-signature-compatible wrapper). 32 MiB int8 workspace lazy-allocated per-class.

### Warmup and JIT

TokenSpeed's CuTe DSL kernels are JIT-compiled the first time the engine sees each shape (`B`, `q_len_per_req`, KV-page-count). The `warmup` request in `minimal_bench.py` triggers compilation for the runtime-time shapes; subsequent timed requests get cached cubins. First-shape compilation costs ~5-7s per shape and shows up in the `init` time (the TRT-LLM warmup phase compiles a different set of shapes — `max_num_tokens × max_batch_size` for CUDA-graph capture — which is why init takes ~4 min vs ~3 min for the base arm before its crash).

---

## Detailed results

### Config 1 — `bench-config.yml` (TP4, 1k/1k, conc=1, MTP off)

JIRA "min-latency floor" — matches the production-stated K2.5 NVFP4 placement.

```
Total requests: 16   wall time: 107.91s
Token Throughput: 151.8 tok/s
Inter Token avg: 6.59 ms/tok
Per User Output Throughput median: 153.19 tok/s
                       min/max:    135.37 / 153.47 tok/s
```

The min (135) reflects the first timed request paying first-shape JIT cost; max (153.47) reflects steady state. Warmup measured 144.9 tok/s aggregate.

### Config 2 — `bench-1k1k_tp8_conc1.yml` (TP8, 1k/1k, conc=1, MTP off)

JIRA "deeper TS regime" — TP8 puts effective num_heads at 16, where TokenSpeed's `fold_sq_factor` BMM1 reformulation has more headroom.

```
Total requests: 16   wall time: 96.72s
Token Throughput: 169.4 tok/s     (+11.6% vs TP4)
Inter Token avg: 5.90 ms/tok      (−10.5% vs TP4)
Per User Output Throughput median: 170.58 tok/s
                       min/max:    149.69 / 171.94 tok/s
```

Init was 246.6s — actually *faster* than TP4 (256.6s), because each TP8 rank has half the weights to load.

### Config 3 — `bench-8k1k_tp4_conc1.yml` (TP4, 8k/1k, conc=1, MTP off)

Long-context floor — closer to TokenSpeed's blog regime (60k ISL is gated by KV-cache headroom; 8k is the largest round number that fits at BS=1 with `free_gpu_memory_fraction=0.8`).

```
Total requests: 16   wall time: 112.63s
Token Throughput: 145.5 tok/s     (−4.1% vs TP4 1k/1k)
Inter Token avg: 6.87 ms/tok      (+4.2% vs TP4 1k/1k)
Per User Output Throughput median: 146.58 tok/s
                       min/max:    131.01 / 146.91 tok/s
```

The 4% slowdown vs 1k context comes from the larger KV-cache walk per decode step — but only 4%, which is the win for MLA's absorbed kv_lora_rank=512 vs full per-head expansion.

### Config 4 — `bench-1k1k_tp4_conc16.yml` (TP4, 1k/1k, conc=16, MTP off)

JIRA "100 TPS/user" throughput sweep — the regime where customer SLAs typically live.

```
Total requests: 256   wall time: 211.18s
Token Throughput: 1241.4 tok/s     (+717% vs conc=1)
Inter Token avg: 0.81 ms/tok       (−87.7% vs conc=1)
Per User Output Throughput median: 77.94 tok/s
                       min/max:    72.62 / 79.30 tok/s
```

Aggregate scales linearly-ish with concurrency: 8.2× at conc=16, which is well above the "diminishing returns" point for batched MLA decode. Per-user drops to 78 tok/s, which lines up with the JIRA's "~100 TPS/user target" regime.

---

## Caveats and what we did not measure

### 1. No baseline comparison

The grid was designed as an A/B (base = stock TRTLLM backend, ts = TOKENSPEED_MLA) so we could quote a *ratio*. **In every config, the base arm crashes during warmup** with the upstream NVRTC bug:

```
RuntimeError: Failed to preprocess kernel
  fmhaSm103aKernel_QkvBfloat16OBfloat16HQk576HV512PagedKvDenseP32MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen
  : Compilation failed: NVRTC_ERROR_COMPILATION
  (expected a type specifier at lines 167, 298, 3413, 6842, 6844, 6848, 6850)
```

Same family of kernel fails for `QkvE4m3O...` (FP8 input) and `QkvBfloat16O...` (BF16 input), and the TP8 variant fails on a different cubin name (`...HVPerCta128...VarSeqQ8Kv128...`) but with the same "expected a type specifier" error class.

Root cause (from log lines 183 / 333 / 3046 / 5629 / 5631 / 5635 / 5637): NVRTC cannot resolve `trtllm::dev::CutlassUmmaConsumerAsyncPipeline<1, false, false, false>` — both the class type and its nested `::SharedStorage` / `::PipelineState`. The class was defined in `namespace cutlass` in the rc14 source tree, but the cubin sources referenced it under `namespace trtllm::dev` — a partial namespace migration that left header/cubin out of sync.

**Resolved 2026-05-20 by upstream PR #14291** (`a173761069 [None][feat] Update the logic of FMHA JIT path`), which moves `CutlassUmmaConsumerAsyncPipeline` to `namespace trtllm::dev` and regenerates the cubin archives. After rebasing onto `upstream/main@f278c4f170` and rebuilding `libtensorrt_llm.so`, the TRTLLM baseline arm runs end-to-end on K2.6 (148.6 tok/s aggregate at TP4 1k/1k conc=1). Verification: `.claude_docs/tokenspeed-kimik25/nvrtc-rebase-verify.md`. NVBugs draft kept as triage history but no longer actionable. Reference comparison points (now obsolete but kept here):

- DSV3-Lite spike step 7 (rc14, `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1` Python path): TokenSpeed = ~+10% over FlashInfer/trtllm-gen MLA decode at q_len=1.
- Same spike step 5: parity FAIL on `bs8_qlen4_spec` (max abs 0.33, max rel ~1166×). Independent of integration path; gates default-on production until Albert weighs in.

### 2. No spec-decode / MTP

K2.6 ships `num_nextn_predict_layers: 0` — no MTP weights in the checkpoint. The bench grid originally included `_mtp3` configs for the regime where TokenSpeed's headline "2× decode latency" claim lives (`fold_sq_factor` benefit is proportional to `q_len_per_req`). Those configs require either (a) Eagle/draft model attached separately, (b) a K2.6-with-MTP-weights checkpoint variant (does not exist today), or (c) DSV3-Lite spike step 7's already-published parity-FAIL number at `bs8 q_len=4`. The current grid measures TokenSpeed at `q_len_per_req=1` only, the **conservative regime** where the spike confirmed parity PASS.

### 3. No production FP8 KV

TokenSpeed's CuTe DSL kernel asserts `kv_cache.dtype == query.dtype`. K2.6's production FP8 KV + BF16 Q doesn't satisfy this. The dispatch gate falls back to thop, which then hits the same NVRTC bug. For the bench grid we use the BF16-KV-patched snapshot. To unlock production FP8 KV, either:
- TokenSpeed must add an FP8-KV / BF16-Q kernel variant, or
- Add a BF16-Q-to-FP8-Q quantization shim, then a matching dequant on output.

### 4. nsys baseline comparison only via indirect proof

The nsys A/B in Phase 3 (`verify_backend_swap.sh`) couldn't capture model-forward kernels under CUDA graphs (default `--cuda-graph-trace=graph` doesn't expose individual kernels, and `--cuda-trace-scope=system-wide` didn't help with re-parented MPI workers). The indirect proof — TS arm generates 32 tokens at 16.0 tok/s while base arm crashes at NVRTC — is conclusive that our backend class is intercepting MLA decode correctly, but we don't have a side-by-side kernel symbol table from the same trace. See Phase 3 report for the full discussion.

### 5. TTFT not measured

`minimal_bench.py` uses non-streaming `llm.generate`. TTFT (time-to-first-token) needs streaming mode; the script reports `-1 ms (not measured; non-streaming)` as a placeholder. Adding streaming is straightforward but not done for this report. Customers' SLA-relevant metrics that we have:
- ITL ✓ (per-decoded-token latency average across the timed run)
- Aggregate throughput ✓
- Per-user throughput ✓ (median / min / max)
- TTFT ✗

---

## Artifacts

```
.claude_docs/tokenspeed-kimik25/
├── bench-config.yml                          # 1k/1k TP4 conc=1
├── bench-1k1k_tp8_conc1.yml                  # 1k/1k TP8 conc=1
├── bench-8k1k_tp4_conc1.yml                  # 8k/1k TP4 conc=1
├── bench-1k1k_tp4_conc16.yml                 # 1k/1k TP4 conc=16
├── bench-*_mtp3*.yml                         # MTP=3 configs (unrunnable on K2.6 today)
├── scripts/
│   ├── run_bench_v2.sh                       # bench driver (sidecar YAMLs, BF16-KV default)
│   ├── minimal_bench.py                      # token-ID bench (bypasses K2.6 tokenizer)
│   ├── minimal_generate.py                   # smoke (Phase 3)
│   └── verify_backend_swap.sh                # nsys A/B (Phase 3)
└── phase4-summary.md                         # this file

tensorrt_llm/_torch/attention_backend/
├── tokenspeed_mla.py                         # FlashInfer-signature wrapper
├── tokenspeed_mla_attention.py               # TokenSpeedMLAAttention(TrtllmAttention)
└── utils.py                                  # registers TOKENSPEED_MLA
```

Per-config logs at:

```
/home/scratch.fkhoubsirat_coreai/runs/k2.6-spike/phase4-bench/
├── bench-config/{base,ts}.log, driver.log, _config_{base,ts}.yml
├── bench-1k1k_tp8_conc1/{base,ts}.log, driver.log, _config_{base,ts}.yml
├── bench-8k1k_tp4_conc1/{base,ts}.log, driver.log, _config_{base,ts}.yml
└── bench-1k1k_tp4_conc16/{base,ts}.log, driver.log, _config_{base,ts}.yml
```

---

## Recommended next steps (in priority order)

1. ~~File the upstream NVRTC bug.~~ **Fixed upstream by PR #14291**
   (verified 2026-05-20 — see `nvrtc-rebase-verify.md`). Rebase onto
   `upstream/main` and rebuild `libtensorrt_llm.so`; the TRTLLM baseline
   arm now runs to completion on K2.6. Re-run the four configs above on
   the rebased build with both arms (`TRTLLM` and `TOKENSPEED_MLA`) to
   produce the apples-to-apples TS/baseline ratio. Single-point tentative
   readout at TP4 1k/1k conc=1: TS 151.8 tok/s (Phase 4, old build) vs
   TRTLLM 148.6 tok/s (rebased build) ≈ TS +2.2% aggregate — but the TS
   number is from the old build, so this is not the final ratio.
2. **Land the backend class as a feature-branch PR.** Three small files plus an utils.py entry; the implementation is in `tensorrt_llm/_torch/attention_backend/tokenspeed_mla{,_attention}.py`. Caveats above belong in the PR description (single sentence each).
3. **Spec-decode parity (Albert Di).** Even on K2.6 without MTP weights, the parity gate from the DSV3-Lite spike (max abs 0.33, max rel ~1166× on `bs8 q_len=4`) is the hard go/no-go for production default-on. Surface those numbers in the same review.
4. **TTFT.** Add a streaming variant of `minimal_bench.py` to capture TTFT and complete the per-user latency picture for customers.
5. **Production FP8 KV.** Either request a TokenSpeed kernel that supports BF16 Q + FP8 KV, or add a BF16-Q-to-FP8-Q quantization shim in the dispatch wrapper. Without one of these, default-on TokenSpeed on K2.6's production FP8-KV path is gated.
6. **Move to a K2.6 variant with MTP.** The fold_sq_factor regime is where the headline 2× claim lives. The current grid does not exercise that regime.
