# TokenSpeed MLA evaluation configs — Kimi K2.5

Eleven `trtllm-bench` / `trtllm-serve` configs scoping an A/B evaluation of
the [TokenSpeed MLA decode kernel](https://github.com/lightseekorg/tokenspeed/tree/main/tokenspeed-mla)
against the current TRT-LLM MLA path on Kimi K2.5 / K2-Thinking. The intent
is to land enough orthogonal data points to decide whether to absorb
TokenSpeed's decode-kernel optimizations into `trtllm-gen` (or wire the
CuTe DSL kernel in as an opt-in attention backend), without taking a runtime
dependency on the upstream TokenSpeed wheel.

## Context

TokenSpeed is an external inference engine that publishes a CuTe DSL MLA
decode kernel claiming a measurable win on Kimi-class models in the
small-batch regime, attributed to two mechanisms:

- **Grouping `q_seqlen × num_heads` into the BMM1 M dimension** — boosts
  tensor-core utilization when effective `num_heads` is small (≤32). This
  fires under dense MLA with TP4 / TP8 (where K2.5's 128 heads shard to
  16–32 per rank) combined with MTP ≥ 1 (which gives `q_len_per_req ≥ 2`).
- **Splitting the multi-CTA KV partial-result reduction** into a separate
  kernel. Dominant at long context, where BMM2-against-KV is the heaviest
  step of decode.

The kernel is open-source in the TokenSpeed repo (`mla_decode_fp8.py`);
the prefill side is closed-binary and out of scope here.

## How the grid is organized

| File | Mode | TP (P/D) | ISL/OSL | conc | MTP | AttnDP | Dataset | Host KV | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| `bench-config.yml` | agg | 4 | 1k/1k | 1 | 1 | off | synth | — | Min-latency floor |
| `bench-1k1k_tp8_conc1.yml` | agg | 8 | 1k/1k | 1 | 1 | off | synth | — | TP-driven win curve |
| `bench-1k1k_tp4_conc16.yml` | agg | 4 | 1k/1k | 16 | 1 | off | synth | — | Throughput, kernel-clean |
| `bench-1k1k_tp4_conc16_mtp3.yml` | agg | 4 | 1k/1k | 16 | 3 | off | synth | — | MTP=3 short context |
| `bench-1k1k_tp4_conc16_attndp.yml` | agg | 4 | 1k/1k | 16 | 1 | on | synth | — | Production-realistic AttnDP |
| `bench-8k1k_tp4_conc1.yml` | agg | 4 | 8k/1k | 1 | 1 | off | synth | — | Long-context floor |
| `bench-8k1k_tp4_conc1_mtp3.yml` | agg | 4 | 8k/1k | 1 | 3 | off | synth | — | MTP=3 × long context |
| `bench-60k1k_tp8_conc1_mtp3.yml` | agg | 8 | 60k/1k | 1 | 3 | off | synth | — | 60k regime infra-fit |
| `bench-60k1k_tp8_conc1_mtp3_multiturn.yml` | agg | 8 | 60k/1k | 1 | 3 | off | multi-turn | — | 60k + KV reuse |
| `bench-60k1k_disagg_p4d4_conc1_mtp3.yml` | disagg | 4/4 | 60k/1k | 1 | 3 | off | multi-turn | — | Decode-isolated (D-only) |
| `bench-60k1k_tp8_conc16_mtp3_cpukvoffload.yml` | agg | 8 | 60k/1k | 16 | 3 | off | multi-turn | 64 GiB | BS=16 / 60k via CPU KV offload |

Every adjacent pair changes exactly one experimental dial — TP, concurrency,
MTP, AttentionDP, ISL, dataset, serving mode, or host KV cache — so each
file's contribution to the overall A/B story is unambiguous.

## How to run

All configs except the disagg row are consumed by `trtllm-bench
--extra_llm_api_options`:

```bash
# Baseline (TRT-LLM's current MLA decode path):
TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 TLLM_TOKENSPEED_MLA=0 \
    trtllm-bench --model nvidia/Kimi-K2-Thinking-NVFP4 \
        throughput --dataset <synth_or_multiturn>.txt \
        --extra_llm_api_options bench-config.yml --concurrency <N>

# TokenSpeed variant (env-var-gated kernel swap):
TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 TLLM_TOKENSPEED_MLA=1 \
    trtllm-bench --model nvidia/Kimi-K2-Thinking-NVFP4 \
        throughput --dataset <synth_or_multiturn>.txt \
        --extra_llm_api_options bench-config.yml --concurrency <N>
```

`TLLM_TOKENSPEED_MLA=1` swaps the TokenSpeed CuTe DSL decode kernel in
place of the FlashInfer / trtllm-gen MLA decode call inside
`FlashInferTrtllmGenAttention.run_mla_generation`. `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1`
forces the FlashInfer Python MLA path, which is where the swap is wired;
without it, default dispatch routes through a C++ thop direct-cubin path
that bypasses the swap entirely.

The disagg row is a `trtllm-serve --disagg_config_file` orchestration
config and has its own invocation block in the file header.

## Known risks tracked across the grid

The configs collectively probe twelve risks, each referenced from the
config where it was first raised:

| # | Risk | First raised in |
|---|---|---|
| 1 | Checkpoint MTP layer count (need ≥3 native for `max_draft_len=3`) | `bench-1k1k_tp4_conc16_mtp3.yml` |
| 2 | Spec-decode parity divergence (see open question on `fold_sq_factor` reduction-order) | `bench-1k1k_tp4_conc16_mtp3.yml` |
| 3 | KV cache pressure with draft chain at long context | `bench-8k1k_tp4_conc1_mtp3.yml` |
| 4 | Possible upstream bug in `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 × trtllm-bench` request batching | `bench-60k1k_tp8_conc1_mtp3.yml` |
| 5 | Prefill cost dominance at long context → synthetic dataset under-reports decode-kernel win | `bench-60k1k_tp8_conc1_mtp3.yml` |
| 6 | KV reuse validity (verify reuse stats > 0 with multi-turn dataset) | `bench-60k1k_tp8_conc1_mtp3_multiturn.yml` |
| 7 | P–D balance in disagg (D starvation masks the win) | `bench-60k1k_disagg_p4d4_conc1_mtp3.yml` |
| 8 | KV transfer cost over NIXL/UCX (TTFT impact) | `bench-60k1k_disagg_p4d4_conc1_mtp3.yml` |
| 9 | Multi-process disagg launch infra | `bench-60k1k_disagg_p4d4_conc1_mtp3.yml` |
| 10 | Host RAM budget (~64 GiB × N ranks) | `bench-60k1k_tp8_conc16_mtp3_cpukvoffload.yml` |
| 11 | PCIe / NVLink-C2C bandwidth for KV offload churn | `bench-60k1k_tp8_conc16_mtp3_cpukvoffload.yml` |
| 12 | Host-cache thrashing via priority-eviction interaction | `bench-60k1k_tp8_conc16_mtp3_cpukvoffload.yml` |

## Open question this grid does NOT settle

Whether the TokenSpeed kernel's spec-decode (`q_len_per_req > 1`) output
matches the existing FlashInfer / trtllm-gen MLA decode bit-for-bit. An
earlier evaluation on a smaller-scale dense-MLA model produced a numerical
divergence in the `BS=8 / q_len=4` regime (the exact regime where the
`fold_sq_factor` reorder operates). Whether that divergence survives
softmax + sampling at K2.5 production scale needs a paired downstream
accuracy eval before any swap can be made default-on.
