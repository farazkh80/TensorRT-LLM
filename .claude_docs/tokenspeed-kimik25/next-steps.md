# TokenSpeed MLA K2.6 - Next Steps

## Current readout

The original goal was to compare TokenSpeed MLA decode against TRT-LLM's
FlashInfer/trtllm-gen MLA/FMHA baseline and verify whether the proposed
speedup reproduces on Kimi K2.6.

That goal is not complete yet.

What we have:

- DSV3-Lite spike: a clean kernel-level A/B showed TokenSpeed about 10%
  faster than FlashInfer/trtllm-gen MLA decode at `q_len_per_req=1`
  (`41.3 ms` vs `46.0 ms` over 2040 kernel calls).
- K2.6 current-main branch: `TokenSpeedMLAAttention` backend is implemented
  and runnable via `attn_backend: TOKENSPEED_MLA`.
- K2.6 Phase 4 grid: TokenSpeed absolute throughput numbers were collected
  for four runnable configs.

What we do not have:

- No valid K2.6 TokenSpeed-vs-TRTLLM ratio. The `TRTLLM` baseline crashed in
  every Phase 4 config during warmup with the upstream NVRTC failure in the
  `fmhaSm103aKernel_...ForGen` MLA-decode cubin family.
- No measurement of TokenSpeed's headline 2x decode-latency regime. The
  available K2.6 checkpoint has `num_nextn_predict_layers=0`, so native
  MTP/spec-decode is unavailable.
- No production-FP8-KV result. The TokenSpeed kernel requires
  `kv_cache.dtype == query.dtype`; K2.6 production uses BF16 Q + FP8 KV, so
  the Phase 4 run used the BF16-KV-patched snapshot.
- No TTFT measurement. `minimal_bench.py` is non-streaming and only reports
  throughput / ITL-style metrics.

## Results to carry forward

K2.6 TokenSpeed-only Phase 4 numbers on B300 SXM6, BF16 KV:

| Config | TP | Concurrency | ISL/OSL | Aggregate tok/s | Per-user median tok/s | ITL |
|---|---:|---:|---|---:|---:|---:|
| `bench-config.yml` | 4 | 1 | 1k/1k | 151.8 | 153.19 | 6.59 ms/token |
| `bench-1k1k_tp8_conc1.yml` | 8 | 1 | 1k/1k | 169.4 | 170.58 | 5.90 ms/token |
| `bench-8k1k_tp4_conc1.yml` | 4 | 1 | 8k/1k | 145.5 | 146.58 | 6.87 ms/token |
| `bench-1k1k_tp4_conc16.yml` | 4 | 16 | 1k/1k | 1241.4 | 77.94 | 0.81 ms/token |

These numbers prove the TokenSpeed backend can run on K2.6, but they do not
prove the proposed speedup over TRT-LLM because the TRT-LLM baseline did not
complete.

## Next steps

1. File or fix the upstream current-main NVRTC baseline bug:
   `fmhaSm103aKernel_...HQk576HV512...ForGen` and the TP8
   `...HVPerCta128...VarSeqQ8Kv128...` variant fail with
   `NVRTC_ERROR_COMPILATION`. Root cause from logs: NVRTC cannot resolve
   `trtllm::dev::CutlassUmmaConsumerAsyncPipeline<1,false,false,false>`
   (defined at `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/trtllmGen_fmha_export/trtllm/dev/CutlassPipeline.h:1467`)
   — likely a missing Jitify include or a CUTLASS-version mismatch on the
   underlying `cutlass::PipelineUmmaConsumerAsync` alias. Paste-ready
   NVBugs draft: `.claude_docs/tokenspeed-kimik25/nvbug-draft-nvrtc-baseline.md`
   (manual submit; NVBugs MCP does not support bug creation).
2. Once the baseline runs, rerun the four Phase 4 configs with both arms:
   `attn_backend: TRTLLM` and `attn_backend: TOKENSPEED_MLA`. The output
   needed is the actual TokenSpeed/TRTLLM ratio for aggregate tok/s,
   per-user tok/s, and ITL.
3. Add a streaming path to `minimal_bench.py` or use an equivalent harness
   so TTFT is measured alongside ITL and throughput.
4. Resolve the production FP8-KV gap. Either TokenSpeed needs a BF16-Q +
   FP8-KV kernel variant, or TRT-LLM needs a validated shim strategy. Until
   then, K2.6 production FP8 KV falls back away from TokenSpeed.
5. Resolve spec-decode parity with Albert Di before any default-on path.
   Existing DSV3-Lite parity data shows `bs8/q_len=4` divergence
   (`max abs 0.33`, `max rel ~1166x`), which is the same regime where
   TokenSpeed's headline 2x claim lives.
6. Get a K2.6/K2.5 checkpoint or attached draft/Eagle setup with MTP enabled
   if the goal is to test the headline 2x spec-decode regime rather than the
   conservative `q_len_per_req=1` decode path.

## References

- Full Phase 4 writeup: `.claude_docs/tokenspeed-kimik25/phase4-summary.md`
- NVBugs draft (NVRTC baseline blocker): `.claude_docs/tokenspeed-kimik25/nvbug-draft-nvrtc-baseline.md`
- Original goal and risk register: `.claude_docs/tokenspeed-kimik25/understanding.md`
- Runbook: `.claude_docs/tokenspeed-kimik25/runbook.md`
- Final handoff: `.claude/context-handoffs/2026-05-19_23-32_tokenspeed-mla-k26-phase4-final.md`
