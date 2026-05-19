# Step 3: Edits on host

**Date:** 2026-05-13
**Experiment:** tokenspeed-mla-dsv3-lite
**GPU / SM:** B300 SXM6 AC / sm_103

## What ran

```bash
git status --short tensorrt_llm/_torch/attention_backend/ tests/unittest/_torch/attention/
python3 -m py_compile tensorrt_llm/_torch/attention_backend/{tokenspeed_mla,utils,trtllm_gen}.py
python3 -m py_compile tests/unittest/_torch/attention/test_tokenspeed_mla.py
```

## Result

Five edits, all Python, all `py_compile` clean:

| File | Change |
|---|---|
| `tensorrt_llm/_torch/attention_backend/tokenspeed_mla.py` | **new** — `is_tokenspeed_mla_available()` + `tokenspeed_batch_decode_with_kv_cache_mla(...)` drop-in (FlashInfer MLA signature) |
| `tensorrt_llm/_torch/attention_backend/utils.py` | **mod** — `get_attention_backend("TOKENSPEED_MLA")` registry entry |
| `tensorrt_llm/_torch/attention_backend/trtllm_gen.py` | **mod** — `TLLM_TOKENSPEED_MLA=1` env-var swap at `run_mla_generation`; default off |
| `tests/unittest/_torch/attention/test_tokenspeed_mla.py` | **new** — parity test parametrized over `num_heads ∈ {16, 32}`, `dtype ∈ {bf16, fp16}`, BS×qlen ∈ {(1,1),(4,1),(8,4)} |
| (rename) `.claude_docs/tokenspeed-mla-dsv3-lite/` | experiment folder |

## Issues / blockers

None on host. tokenspeed-mla wheel install happens in step 4 inside container.

## Next

Step 4 — `pip install -e .` + `pip install tokenspeed-mla` inside container. **Pause for user confirmation before running.**
