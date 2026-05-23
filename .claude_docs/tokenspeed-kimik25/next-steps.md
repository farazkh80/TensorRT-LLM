# TokenSpeed MLA K2.5/K2.6 - Next Steps

## Current readout (post Phase 5, 2026-05-23)

**Headline:** TokenSpeed's value is **regime-dependent on K2.x**.

| Regime | Result |
|---|---|
| `q_len_per_req = 1` (Phase 4, K2.6 conc=1 across TP4/TP8 + 1k/8k ISL) | TS **−3 to −7%** (slower than rebased TRTLLM) |
| `q_len_per_req = 1`, conc=16 (Phase 4, K2.6 throughput sweep) | TS **+0.6%** (tied) |
| `q_len_per_req = 4` via EAGLE-3 (Phase 5, K2.5 NVFP4 + draft) | TS **+4.2%** aggregate / **+6.3%** per-user / **−32.4%** P99 latency |

So the DSV3-Lite spike's "TS wins at q_len > 1" prediction holds on
K2.5 with EAGLE-3, but the K2.6 q_len=1 production regime is a small
regression. Whether to adopt TS depends entirely on whether your
deployment uses spec-decode.

What we have:

- DSV3-Lite spike: kernel-level A/B showed TokenSpeed about 10% faster
  than FlashInfer/trtllm-gen MLA decode at `q_len_per_req=1`. This
  did NOT reproduce on K2.6 (Phase 4) — the rebased FMHA path (PR
  #14291) closed that gap.
- K2.5/K2.6 current-main: `TokenSpeedMLAAttention` backend is
  implemented and runnable via `attn_backend: TOKENSPEED_MLA`.
- K2.6 Phase 4 grid: full A/B numbers for four configs (TP4/TP8,
  1k/8k ISL, conc=1/16). See `phase4-rebased-ab.md`.
- K2.5 Phase 5 EAGLE-3: full A/B numbers for the MTP/spec-decode
  regime (max_draft_len=3). See `phase5-k25-eagle3.md`.
- K2.5 spec-decode support: two minimal patches to
  `tensorrt_llm/_torch/models/modeling_kimi_k25.py` to make the
  `KimiK25ForConditionalGeneration` multimodal wrapper forward the
  spec-decode attributes (`draft_config`, `draft_model`,
  `load_draft_weights`, `model`, `lm_head`) and pass `**kwargs`
  (incl. `spec_metadata`) to the wrapped DSV3 forward.

What we do not have:

- **Output-quality / correctness check for the Phase 5 TS win.** TS's
  draft-acceptance MIN (0.18 vs 0.00) and acceptance-length MIN (1.54
  vs 1.01) are materially higher than TRTLLM's, which could be a real
  kernel-quality win OR could be the kernel's numeric divergence (max
  abs 0.33 / max rel ~1166× per DSV3-Lite spike at q_len > 1) biasing
  sampling. **Do not adopt TS for spec-decode based on perf alone**
  until this is settled. (Per user direction, deferred from Phase 5.)
- No production-FP8-KV result. The TokenSpeed kernel requires
  `kv_cache.dtype == query.dtype`; K2.x production uses BF16 Q + FP8
  KV, so Phase 4/5 used BF16-KV-patched snapshots.
- No replication of Phase 5. Single bench run, 32 requests, no
  variance bars.
- No TTFT measurement. trtllm-bench used here is non-streaming.

## Results to carry forward

### K2.6 Phase 4 rebased A/B (TS vs TRTLLM, both arms ran):

| Config | TP / conc / ISL | TRTLLM | TOKENSPEED_MLA | TS Δ |
|---|---|---:|---:|---:|
| `bench-config.yml` | 4 / 1 / 1k/1k | **158.5** | 152.8 | **−3.6%** |
| `bench-1k1k_tp8_conc1.yml` | 8 / 1 / 1k/1k | **182.2** | 169.3 | **−7.1%** |
| `bench-8k1k_tp4_conc1.yml` | 4 / 1 / 8k/1k | **152.0** | 146.9 | **−3.4%** |
| `bench-1k1k_tp4_conc16.yml` | 4 / 16 / 1k/1k | 1239.3 | 1246.8 | +0.6% (tied) |

Numbers are aggregate tok/s. See `phase4-rebased-ab.md`.

### K2.5 Phase 5 EAGLE-3 A/B (max_draft_len=3, conc=2):

| Metric | TRTLLM | TOKENSPEED_MLA | Δ |
|---|---:|---:|---:|
| Total Token Throughput | 1065.80 | **1110.73** | **TS +4.2%** |
| Per User Output Throughput | 285.65 | **303.59** | **TS +6.3%** |
| P99 latency (ms) | 10110 | **6833** | **TS −32.4%** |

See `phase5-k25-eagle3.md`.

## Next steps

1. ~~File or fix the upstream current-main NVRTC baseline bug.~~ **Fixed
   upstream by PR #14291** (`a173761069 [None][feat] Update the logic of
   FMHA JIT path`), confirmed 2026-05-20 by rebasing onto
   `upstream/main@f278c4f170`, rebuilding `libtensorrt_llm.so`, and running
   the TRTLLM-baseline arm against `bench-config.yml` end-to-end (LLM init
   279.9 s, 148.6 tok/s aggregate, 6.73 ms/tok ITL). Confirms root-cause
   hypothesis #3 (rc14 cubins ahead of header on the
   `cutlass::` → `trtllm::dev::` namespace migration). Verification
   write-up: `.claude_docs/tokenspeed-kimik25/nvrtc-rebase-verify.md`.
   NVBugs draft kept as triage history but no longer actionable.
2. ~~Re-run the four Phase 4 configs on the rebased build.~~ **Done
   2026-05-20** — full A/B grid at `.claude_docs/tokenspeed-kimik25/phase4-rebased-ab.md`.
   Result: TokenSpeed is **3–7% slower than rebased TRTLLM at conc=1**
   (all three configs: TP4 1k/1k, TP8 1k/1k, TP4 8k/1k), and **tied at
   conc=16** (1239 vs 1247 tok/s, within noise). The DSV3-Lite spike's
   +10% TS advantage no longer holds against the post-#14291 FMHA JIT
   path. Rebased TS ≈ old-build TS (within ±1%) → no regression on the
   TS side; the gap closed because the TRTLLM baseline got faster.
3. ~~Get a K2.6/K2.5 checkpoint with MTP/spec-decode enabled.~~ **Done
   2026-05-23** via Phase 5 — used `nvidia/Kimi-K2.5-NVFP4` +
   `nvidia/Kimi-K2.5-Thinking-Eagle3` EAGLE-3 draft on B300. TS wins
   +4.2% aggregate / +6.3% per-user / −32.4% P99 latency at
   `max_draft_len=3`. See `phase5-k25-eagle3.md`.
4. **Output-quality / correctness check for Phase 5 TS.** TS's
   draft-acceptance MIN is 0.18 vs 0.00 for TRTLLM, and
   acceptance-length MIN is 1.54 vs 1.01. Could be a real kernel win,
   or could be the kernel's numeric divergence biasing sampling. Plan:
   greedy-sampling A/B on ~16 fixed prompts, diff outputs. Albert Di
   for parity guidance. Until this lands, **do not adopt TS for
   spec-decode based on perf alone**.
5. **Replicate Phase 5** — single-run 32-req smoke; rerun 3× and
   report mean ± variance before publishing.
6. **Small Phase 5 grid** — `max_draft_len ∈ {0, 1, 3}` × backend
   to verify the TS advantage scales with `q_len_per_req` as the
   `fold_sq_factor` math predicts.
7. **Upstream the two K2.5 patches** in `modeling_kimi_k25.py`
   (forwarding properties + `**kwargs` in forward). These are needed
   for any spec-decode use of K2.5, not just TokenSpeed. Worth a
   small PR independent of the TS evaluation.
8. Add a streaming path to capture TTFT alongside ITL/throughput.
9. Resolve the production FP8-KV gap. Either TokenSpeed adds a
   BF16-Q + FP8-KV kernel variant, or we add a quantization shim.
   Until then, K2.x production FP8 KV deployments cannot use TS.

## References

- **Phase 5 A/B on K2.5 + EAGLE-3 (current verdict — TS WINS at q_len > 1)**: `.claude_docs/tokenspeed-kimik25/phase5-k25-eagle3.md`
- Phase 5 plan: `.claude_docs/tokenspeed-kimik25/phase5-plan.md`
- Phase 4 A/B on K2.6 rebased build (TS LOSES at q_len = 1): `.claude_docs/tokenspeed-kimik25/phase4-rebased-ab.md`
- Original Phase 4 writeup (TS-only, base crashed): `.claude_docs/tokenspeed-kimik25/phase4-summary.md`
- NVRTC rebase verification (PR #14291 confirmed-fix): `.claude_docs/tokenspeed-kimik25/nvrtc-rebase-verify.md`
- NVBugs draft (NVRTC baseline blocker — archived, no longer actionable): `.claude_docs/tokenspeed-kimik25/nvbug-draft-nvrtc-baseline.md`
- Original goal and risk register: `.claude_docs/tokenspeed-kimik25/understanding.md`
- Runbook: `.claude_docs/tokenspeed-kimik25/runbook.md`
- Final handoff: `.claude/context-handoffs/2026-05-19_23-32_tokenspeed-mla-k26-phase4-final.md`
