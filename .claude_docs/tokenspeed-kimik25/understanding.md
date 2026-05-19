# TokenSpeed MLA → Kimi K2.5/K2.6 Integration — Understanding

**Date:** 2026-05-19
**Owner:** Faraz Khoubsirat
**Manager chain:** Rajeev Rao (assigned 2026-05-13 after Sharan Chetlur offered Rajeev's team)
**JIRA:** TRTLLM-12510 (Reporter: Jonas Li; Evaluator: Yuxian Qiu; original target assignee: Zhenhuan Chen)
**Companion spike:** `../tokenspeed-mla-dsv3-lite/` (DSV3-Lite NVFP4 / B300 sm_103, completed 2026-05-13)

## 1. Why this exists

TokenSpeed is an external inference engine (preview) co-developed by NVIDIA + AMD + Qwen Inference + Together AI + Mooncake + LongCat + FluentLLM. Their public blog (https://lightseek.org/blog/lightseek-tokenspeed.html) shows MLA decode beating TRT-LLM's MLA in the small-batch regime on Kimi-class models. Two notable customers — **Fireworks** and **Together** — serve Kimi K2.6 and have asked NVIDIA whether the kernel is available.

### Decision history (JIRA + email)

- **Initial JIRA verdict (Yuxian Qiu, Perkz Zheng, Julien Demouth):** Low priority. Speedup is narrowly applicable (dense MLA + TP4/TP8 + MTP>0). Of frontier models, only **Kimi K2.5 / K2.6** hit the regime. Recommended path: absorb the learnings into trtllm-gen **after CTM + RTS lands**, not adopt TokenSpeed as a new dependency.
- **Tao Li (2026-05-10):** asked Perkz for CTM+RTS timeline. No answer in JIRA yet.
- **External pressure (email, 2026-05-09 → 2026-05-13):** Laikh Tewari flagged that Fireworks asked for the binary; Albert Di clarified the **MLA decode kernel is fully open-sourced in TokenSpeed's repo** (`mla_decode_fp8.py`), only the prefill `.so` is closed. June Yang escalated: K2.5 is already a key InferenceX model, K2.6 shares the architecture, both customers will benefit. June restated the same concern from JIRA — maintenance overhead of adopting a binary kernel.
- **Final assignment (Rajeev Rao, 2026-05-13):** Faraz to evaluate the integration with maintainability and long-term support in mind, get leadership sign-off before implementation.

## 2. The benefit profile (from JIRA evaluation)

| Path | Applicability | Measured win |
|---|---|---|
| **Decode** (group `q_seqlen × num_heads` into BMM1 M; split partial-KV reduction into a separate kernel) | Dense MLA + TP4/TP8 (→ effective num_heads ≤ 32) + MTP>0. Sparse MLA (DSV3.2/V4) gets nothing. | Kimi K2.5/K2.6: ~9% min-latency win at bs=1, ~11% throughput around 100 TPS/user. |
| **Prefill** (AOT cute-DSL softmax, MR 20837) | Small-batch dense long-context chunked prefill. | 10–32% per-context-kernel vs trtllm-gen baseline. Shrinks vs cute-DSL baseline. In disagg, doesn't translate to E2E because prefill must over-feed decode. Aligns with Julien's "prefill not a priority" stance and the trtllm-gen→RTS-CTM migration. |

**Net:** decode is the load-bearing optimization. Prefill is deprioritized.

## 3. What's available from TokenSpeed

- **Open source:** `mla_decode_fp8.py` in https://github.com/lightseekorg/tokenspeed/tree/main/tokenspeed-mla. Decode is fully implemented in CuTe DSL.
- **Binary-only:** prefill kernel — distributed via a future `.whl` (delayed; key contributor traveling). Not relevant to the decode integration.
- **Upstream of TokenSpeed:** Albert Di's improvements were merged into DKG examples — **MR 21161 (prefill)** and **MR 21023 (decode)**. So the decode kernel patterns also exist as DKG / CuTe DSL examples we can study/copy without taking a runtime dependency on TokenSpeed.
- **Pip packages:** `tokenspeed-mla 0.1.2`, `tokenspeed-triton 3.7.10.post20260505` — both installable from PyPI (~89 MB).

## 4. What the DSV3-Lite spike already proved (2026-05-13)

Ran on 2× B300 sm_103 inside `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`. Full reports in `../tokenspeed-mla-dsv3-lite/`. The headline learnings that matter for K2.5/K2.6:

### A. Kernel-level perf is real

- TokenSpeed `BlackwellMultiHeadLatentAttentionForward` ≈ **10% faster** than the FlashInfer/trtllm-gen MLA decode on DSV3-Lite NVFP4 (q_len_per_req=1, 32-token decode, 2040 kernel calls: 41.3 ms vs 46.0 ms).
- This was measured via nsys with `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 TLLM_TOKENSPEED_MLA=1` once we forced the path. Single-run, no statistical reps yet.

### B. The dominant production code path bypasses Python wrappers

- For DSV3-Lite NVFP4 + sm_103 + default config, MLA decode flows through **C++ thop direct cubin dispatch** (`fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512PagedKvDenseP32VarSeqQ16Kv128PersistentSwapsAbForGen` and a `HVPerCta128` variant), **not** the `flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla` Python wrapper.
- Implication for Kimi K2.5: a swap inside `FlashInferTrtllmGenAttention.run_mla_generation` is dead code on the default path. Either (a) set `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1` to force the FlashInfer Python path, or (b) intercept earlier — at `TrtllmAttention._run` or at the C++ thop dispatch.
- The chosen design (see `../tokenspeed-mla-dsv3-lite/design-tokenspeed-attn-backend.md`) is a peer backend class `TokenSpeedMLAAttention(TrtllmAttention)` that overrides only `_run` for the MLA-decode-generation-only branch — modeled on PR #13773's `FlashInferNvfp4Sm12xFusedMoE(CutlassFusedMoE)` pattern.

### C. Spec-decode parity diverges (the regime where the headline 2× lives)

- Plain decode shapes (BS=1, BS=4×varlen, q_len=1, BF16, num_heads ∈ {16, 32}): **PASS** parity vs FlashInfer.
- Spec-decode shapes (BS=8, q_len=4): **FAIL** — 0.9% of elements off, max abs **0.33**, max rel **~1166×**. Suspected cause: `fold_sq_factor` reorders queries into the head axis, changing the multi-CTA split-KV reduction order. Open question for Albert Di: does this survive softmax + sampling at production scale, or do we need to fix the reduction tree / accumulator dtype?
- This is the **same regime** as TokenSpeed's headline "2× decode latency" claim. Kimi K2.5 with MTP>0 will land in this regime by design.

### D. Known bugs/blockers we hit

1. `tokenspeed-mla 0.1.2` LSE bug — the BF16/FP16 kernel reinterprets `lse` unconditionally even when the wrapper passes `lse=None`. 3-line patch in `mla_decode.py` (see `../tokenspeed-mla-dsv3-lite/patches/apply_patches.py`). Worth upstreaming to lightseekorg.
2. MTP=3 fails on the DSV3-Lite checkpoint (`KeyError: 'model.layers.30.self_attn.kv_a_proj_with_mqa.weight'`) — the model ships `num_nextn_predict_layers=1`. **Not a TokenSpeed issue**, but it blocked spec-decode A/B on this model. K2.5/K2.6 with native MTP should not hit this.
3. `flashinfer 0.6.9` cubin coverage gap on sm_103 for some `(HQk=576, HV=512, page=64, multiCtasKvMode=1)` shapes — test skips gracefully.
4. **rc14 upstream bug in the `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1` path under `trtllm-bench`:** `q_len_per_req` computed as `1 - input_length` under multi-request scheduling — crashes both base and ts identically. Blocks E2E `trtllm-bench` A/B on the FlashInfer Python path until upstream-fixed.

### E. Already-shipped artifacts (host source, not yet committed for K2.5 work)

- `tensorrt_llm/_torch/attention_backend/tokenspeed_mla.py` (new — drop-in wrapper matching FlashInfer's signature)
- `tensorrt_llm/_torch/attention_backend/utils.py` (+18 lines — `TOKENSPEED_MLA` selector entry, placeholder returning `TrtllmAttention`)
- `tensorrt_llm/_torch/attention_backend/trtllm_gen.py` (env-var swap in `run_mla_generation`; spike code, to be reverted in the real integration per the design doc)
- `tests/unittest/_torch/attention/test_tokenspeed_mla.py` (parametrized parity test: num_heads ∈ {16, 32}, dtypes, BS×qlen ∈ {(1,1),(4,1),(8,4)})

## 5. Open questions before Kimi K2.5/K2.6 integration

1. **CTM + RTS timeline** (Tao Li → Perkz, unanswered). The JIRA's preferred path is "absorb into trtllm-gen post-CTM+RTS." If CTM+RTS is months out and Fireworks/Together need this for K2.5 now, the staging plan is different.
2. **Spec-decode parity tolerance** (Albert Di). Is 0.33 max abs / 1166× max rel acceptable downstream of softmax + sampling on K2.5? If no, do we (a) fix the reduction order in the CuTe DSL kernel, (b) gate it off by default until upstream fix, or (c) require an accuracy-bench gate before each integration?
3. **Integration form** (Sharan, June). Three options on the table:
   - Python backend class `TokenSpeedMLAAttention(TrtllmAttention)` (the spike's design — short to land, intercepts at `_run`, but leaves the C++ thop path untouched).
   - Direct port of the CuTe DSL decode kernel into trtllm-gen's C++ FMHA dispatch (deeper, matches Julien/Perkz's "absorb into trtllm-gen post-CTM+RTS" plan).
   - Take a runtime dep on `tokenspeed-mla` PyPI wheel (rejected by Yuxian/June — maintenance overhead).
4. **Test model availability.** DSV3-Lite was a stand-in; we need a Kimi K2.5 checkpoint that lands in the dense-MLA TP4/TP8 regime with native MTP>0 to reproduce TokenSpeed's headline win.
5. **The trtllm-bench rc14 bug** must be fixed (or we use a different harness) before any clean E2E A/B for K2.5 numbers. May be already fixed in rc15+/main — needs verification.

## 6. Proposed next steps (assignee: me, draft for leadership sign-off)

1. **Reproduce the kernel-level win on a Kimi K2.5 checkpoint** (or DSV3-full if a K2.5 checkpoint isn't yet hostable on our B300 capacity), with `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 TLLM_TOKENSPEED_MLA=1` and the spike's env-var swap. Capture nsys decode-kernel time A/B. Estimated 1 day.
2. **Land the `TokenSpeedMLAAttention(TrtllmAttention)` backend class** on a feature branch (per `../tokenspeed-mla-dsv3-lite/design-tokenspeed-attn-backend.md`). Revert the `run_mla_generation` env-var swap so we don't have two integration points. Estimated 1 day.
3. **Sync with Albert Di** on the spec-decode parity divergence. Bring the unit-test diff numbers and the kernel symbol names. Decision needed before merge.
4. **Confirm CTM+RTS timeline with Julien.** If <1 quarter, gate the Python backend as a temporary measurement bridge with a clear deletion plan. If longer, justify it as a production-grade landing for K2.5.
5. **File upstream:** the `tokenspeed-mla 0.1.2` LSE bug (lightseekorg/tokenspeed) and the rc14 `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 × trtllm-bench` shape bug (internal).
6. **Bench config grid** (this folder's `bench-*.yml` files): one config per experiment, each adjacent pair differing by a single dial. Full grid in §7. Treat the grid as the planning artifact for steps 1 + 3 — each row maps to either a kernel-level win measurement or a parity-divergence data point Albert Di needs.

## 7. Experiment grid (bench configs)

Eleven configs live alongside this doc, each a one-experiment file consumed by `trtllm-bench --extra_llm_api_options` (or `trtllm-serve --disagg_config_file` for the disagg row). Every adjacent pair changes exactly one experimental dial — TP, concurrency, MTP, AttentionDP, ISL, dataset, serving mode, or host KV cache — so the leadership-sign-off story is a clean orthogonal sweep rather than eleven unrelated runs.

| File | Mode | TP (P/D) | ISL/OSL | conc | MTP | AttnDP | Dataset | Host KV | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| `bench-config.yml` | agg | 4 | 1k/1k | 1 | 1 | off | synth | — | Min-latency floor; matches JIRA ~9% bs=1 claim regime |
| `bench-1k1k_tp8_conc1.yml` | agg | 8 | 1k/1k | 1 | 1 | off | synth | — | TP scaling (effective heads 32→16) |
| `bench-1k1k_tp4_conc16.yml` | agg | 4 | 1k/1k | 16 | 1 | off | synth | — | Throughput, kernel-clean; JIRA ~11% @ 100 TPS/user regime |
| `bench-1k1k_tp4_conc16_mtp3.yml` | agg | 4 | 1k/1k | 16 | 3 | off | synth | — | MTP=3 short context; spec-decode parity gate |
| `bench-1k1k_tp4_conc16_attndp.yml` | agg | 4 | 1k/1k | 16 | 1 | on | synth | — | Production-realistic AttnDP variant |
| `bench-8k1k_tp4_conc1.yml` | agg | 4 | 8k/1k | 1 | 1 | off | synth | — | Long-context floor |
| `bench-8k1k_tp4_conc1_mtp3.yml` | agg | 4 | 8k/1k | 1 | 3 | off | synth | — | MTP=3 × long context |
| `bench-60k1k_tp8_conc1_mtp3.yml` | agg | 8 | 60k/1k | 1 | 3 | off | synth | — | Blog-regime infra-fit (single-turn) |
| `bench-60k1k_tp8_conc1_mtp3_multiturn.yml` | agg | 8 | 60k/1k | 1 | 3 | off | multi-turn | — | Blog-regime headline-match (KV reuse on) |
| `bench-60k1k_disagg_p4d4_conc1_mtp3.yml` | disagg | 4/4 | 60k/1k | 1 | 3 | off | multi-turn | — | Decode-isolated TS win (D-only) |
| `bench-60k1k_tp8_conc16_mtp3_cpukvoffload.yml` | agg | 8 | 60k/1k | 16 | 3 | off | multi-turn | 64 GiB | JIRA gap closure (BS=16/60k previously OOMed) |

Reading the grid:

- **Rows 1–5 (1k/1k):** isolate the kernel-level TS win along the cheap axes — TP, concurrency, MTP, AttnDP. These give the floor numbers and let us reject the integration cheaply if even the small-context win doesn't materialize.
- **Rows 6–7 (8k/1k):** scale context to the limit of what fits comfortably in our standard KV budget. Tests whether the small-context win grows with the BMM2-against-KV step's share of decode time, which is TS's claimed mechanism.
- **Rows 8–9 (60k/1k agg):** the blog-headline regime as close as our hardware can replicate it. Row 8 is the infra-fit test; row 9 adds the multi-turn dataset that the win-mechanism requires. If TS's headline transfers to K2.5 on our hardware, it shows up here.
- **Row 10 (60k disagg):** the cleanest decode-only isolation. Answers "if P keeps D fed, what does TS buy on D-side throughput?" — a question the agg grid can't answer.
- **Row 11 (60k + CPU offload):** closes the specific gap the JIRA called out — TokenSpeed's blog has BS=16/60k because they used CPU KV offload; TRT-LLM has the feature, we just hadn't enabled it. Without this row any "TS vs TRT-LLM at BS=16/60k" comparison is unfair.

All configs share the same A/B invocation pattern: re-run with `TLLM_TOKENSPEED_MLA=0` vs `=1` while `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1` is held on. The disagg row sets the TS env vars ONLY on the D-side worker process.

Risk register accumulates across rows (each numbered note is referenced by the relevant config's header):

| # | Risk | First raised in |
|---|---|---|
| 1 | Checkpoint MTP layer count (need ≥3 native for max_draft_len=3) | `bench-1k1k_tp4_conc16_mtp3.yml` |
| 2 | Spec-decode parity divergence (max abs 0.33 / max rel ~1166× from spike) | `bench-1k1k_tp4_conc16_mtp3.yml` |
| 3 | KV cache pressure with draft chain at long context | `bench-8k1k_tp4_conc1_mtp3.yml` |
| 4 | rc14 `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 × trtllm-bench` shape bug | `bench-60k1k_tp8_conc1_mtp3.yml` |
| 5 | Prefill cost dominates → synth dataset under-reports TS win | `bench-60k1k_tp8_conc1_mtp3.yml` |
| 6 | KV reuse validity (verify reuse stats > 0 with multi-turn dataset) | `bench-60k1k_tp8_conc1_mtp3_multiturn.yml` |
| 7 | P–D balance (D starvation masks TS win) | `bench-60k1k_disagg_p4d4_conc1_mtp3.yml` |
| 8 | KV transfer cost (TTFT impact from NIXL P→D transfer) | `bench-60k1k_disagg_p4d4_conc1_mtp3.yml` |
| 9 | Multi-process disagg launch infra | `bench-60k1k_disagg_p4d4_conc1_mtp3.yml` |
| 10 | Host RAM budget (~64 GiB × 8 ranks) | `bench-60k1k_tp8_conc16_mtp3_cpukvoffload.yml` |
| 11 | PCIe / NVLink-C2C bandwidth for KV offload churn | `bench-60k1k_tp8_conc16_mtp3_cpukvoffload.yml` |
| 12 | TS × host cache thrashing via priority eviction | `bench-60k1k_tp8_conc16_mtp3_cpukvoffload.yml` |

## 8. Phase 3 result on Kimi-K2-Thinking-NVFP4 (2026-05-19)

Per the hybrid plan, Phase 3 of `runbook.md` ran the spike's env-var swap
against the actual K2.6 target (`nvidia/Kimi-K2-Thinking-NVFP4`, TP4, B300
sm_103, 32-token decode) under nsys. Findings:

| Signal | Base (`TLLM_TOKENSPEED_MLA=0`) | Variant (`TLLM_TOKENSPEED_MLA=1`) |
|---|---|---|
| LLM init time | 293.2 s | 294.7 s |
| 32-token generate latency | 1.41 s (22.7 tok/s) | 1.36 s (23.6 tok/s) |
| Unique kernel symbols in `cuda_gpu_kern_sum` | **9** | **9 (identical set)** |
| `cudaLaunchKernel` count | 238,090 | 238,090 |
| MLA / FMHA / TokenSpeed kernel symbols visible in trace | 0 | **0** |

The 4 % latency delta is well within single-run noise. The kernel-name diff
is empty — both traces show identical kernel composition. This is the
**same finding as DSV3-Lite spike step 6**: the env-var swap inside
`FlashInferTrtllmGenAttention.run_mla_generation` is dead code on the
default K2.6 NVFP4 dispatch path, because TRT-LLM routes MLA decode through
the C++ thop direct-cubin path before the Python FlashInfer wrapper.

(nsys's `cuda_gpu_kern_sum` aggregates inside CUDA Graphs, so the actual
trtllm-gen FMHA cubin symbols don't appear in this report — they'd require
`nsys profile --cudagraph-trace=node`. The dispositive fact is that the
aggregated kernel sets are *identical* between base and variant, not that
attention kernels are missing from both.)

### Implication for the hybrid criterion

The hybrid criterion in `runbook.md` Phase 5 said: if the spike-patch path
reproduces the JIRA-claimed kernel-level win on K2.6, invest in the
`TokenSpeedMLAAttention(TrtllmAttention)` backend class. Phase 3 shows the
spike-patches *cannot reach* the MLA decode code path on K2.6, so the
hybrid criterion is exhausted with a "spike-measurement bridge gave us no
signal" result. The only remaining path to land a measurement on K2.6 is
the backend-class implementation that intercepts at `TrtllmAttention._run`
(the dispatch site upstream of the C++ thop call).

### Action required (paused for leadership sign-off)

Before implementing the backend class, this finding needs to land with:

- **Rajeev Rao** — confirm continued staffing now that the cheap spike path
  is exhausted; the next step is a ~1-day implementation per the design
  doc, not a sub-day measurement.
- **Sharan Chetlur / June Yang** — re-confirm the "absorb into trtllm-gen
  post-CTM+RTS" plan still holds (per JIRA TRTLLM-12510 + email thread).
  The Python backend class is the temporary measurement bridge in that
  plan; the eventual home is in trtllm-gen.
- **Albert Di** — needs the DSV3-Lite parity divergence answer (max abs
  0.33 / max rel ~1166× on spec-decode shapes) before any TS path can
  default-on for K2.6 MTP=3 production. This was open before Phase 3 and
  remains the gating question for actually merging the backend class.
- **Julien Demouth** — CTM+RTS timeline (Tao Li's hanging JIRA question);
  determines whether the Python backend class is a quick-win for ≤1 quarter
  or a longer-term production landing.

### Bottom line for leadership

- Kernel-level win is real on DSV3-Lite (~10% via direct kernel A/B in spike step 7).
- Customer pressure (Fireworks / Together on K2.6) is real.
- The spike's Python env-var swap can't reach K2.6's MLA decode path — Phase 3 confirmed this directly. Same as DSV3-Lite.
- The next viable step is the `TokenSpeedMLAAttention(TrtllmAttention)` backend class (~1 day implementation), per `../tokenspeed-mla-dsv3-lite/design-tokenspeed-attn-backend.md`. Code + grid + risk register are all staged in this directory; only blocker is leadership go-ahead.
- The spec-decode parity divergence remains the gate for default-on production. Until Albert weighs in, even the backend class is a "perf measurement bridge", not a default-on path.

## 9. Phase 3 trace artifacts

Stored at `/home/scratch.fkhoubsirat_coreai/runs/k2.6-spike/phase3-verify/`:

- `nsys-k26-base-mtp1.nsys-rep` (44 MB) — baseline trace
- `nsys-k26-base-mtp1.sqlite` (108 MB) — exported event DB
- `nsys-k26-base-stdout.log` (145 KB) — full base run output
- `nsys-k26-ts-mtp1.nsys-rep` (44 MB) — variant trace
- `nsys-k26-ts-mtp1.sqlite` (108 MB) — exported event DB
- `nsys-k26-ts-stdout.log` (140 KB) — full variant run output

The traces stay on `/home/scratch.fkhoubsirat_coreai` (not committed) for
follow-up analysis. If a deeper diff with `--cudagraph-trace=node` becomes
useful before the backend class lands, the traces are re-runnable via
`./scripts/verify_kernel_swap.sh` once we update the nsys invocation.

## References

- JIRA: https://jirasw.nvidia.com/browse/TRTLLM-12510
- Blog: https://lightseek.org/blog/lightseek-tokenspeed.html
- TokenSpeed repo: https://github.com/lightseekorg/tokenspeed/tree/main/tokenspeed-mla
- DKG decode MR: MR 21023; prefill MR: MR 21161
- MoE backend pattern to mirror: PR #13773
- Spike summary: `../tokenspeed-mla-dsv3-lite/summary.md`
- Spike design: `../tokenspeed-mla-dsv3-lite/design-tokenspeed-attn-backend.md`
- Slack thread: https://nvidia.slack.com/archives/C0AMFBS39E3/p1778090844363919
