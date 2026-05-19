# Step 1: Env

**Date:** 2026-05-13
**Experiment:** tokenspeed-mla-dsv3-lite
**GPU / SM:** B300 SXM6 AC / sm_103 (2× visible)
**Model:** DSV3-Lite NVFP4 (MoE-only quantized) at `./.claude_docs/models/nvfp4_moe_only/`
  - `DeepseekV3ForCausalLM`, hidden=2560, layers=30, num_heads=32
  - MLA dims: kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128
  - `num_nextn_predict_layers=1` → MTP=1 default; will A/B with MTP=3
  - `quant_algo=NVFP4` MoE only; ALL `self_attn*` excluded → MLA stays BF16; KV cache BF16

## What ran

- `nvidia-smi --query-gpu=name,compute_cap --format=csv`
- `docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`
- Inspected `.claude_docs/models/nvfp4_moe_only/{config.json,hf_quant_config.json}`

## Result

- B300 SM 10.3 confirmed — TokenSpeed MLA kernels supported.
- Image pulled (40.1 GB).
- Model ideal: MLA stays BF16, MTP=1 native, MoE NVFP4 does not touch attention.

## Issues / blockers

None.

## Next

Step 2 — `docker run` with `$PWD` mounted at `/workspace/TensorRT-LLM`.
