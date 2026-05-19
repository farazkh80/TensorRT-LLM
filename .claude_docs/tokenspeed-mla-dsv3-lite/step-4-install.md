# Step 4: Install

**Date:** 2026-05-13
**Experiment:** tokenspeed-mla-dsv3-lite
**GPU / SM:** B300 SXM6 AC / sm_103

## What ran

```bash
docker run -d --name tokenspeed-spike --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$PWD:/workspace/TensorRT-LLM" \
    -v /home/farazkh_scratch/tokenspeed:/workspace/tokenspeed:ro \
    -v "$PWD/tensorrt_llm/_torch/attention_backend/tokenspeed_mla.py:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/attention_backend/tokenspeed_mla.py:ro" \
    -v "$PWD/tensorrt_llm/_torch/attention_backend/utils.py:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/attention_backend/utils.py:ro" \
    -v "$PWD/tensorrt_llm/_torch/attention_backend/trtllm_gen.py:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/attention_backend/trtllm_gen.py:ro" \
    -w /workspace \
    nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14 \
    sleep infinity

docker exec tokenspeed-spike pip install -q tokenspeed-mla
docker exec tokenspeed-spike python -u -c '<smoke test>'
```

## Result

- Strategy pivot: `pip install -e .` against host source crashed on `ImportError: cannot import name 'ExaoneMoeConfig' from 'transformers'` (host branch is ahead of rc14; transformers v5 / mistral_common version skew). Switched to **bind-mounting only our 3 changed files** over the container's installed `tensorrt_llm/_torch/attention_backend/{tokenspeed_mla,utils,trtllm_gen}.py`. CWD set to `/workspace` so `import tensorrt_llm` resolves the installed package.
- `tokenspeed-mla 0.1.2` + `tokenspeed-triton 3.7.10.post20260505` installed from PyPI cleanly (89 MB).
- Smoke output:
  ```
  trtllm: /usr/local/lib/python3.12/dist-packages/tensorrt_llm/__init__.py
  selector(TOKENSPEED_MLA) -> TrtllmAttention
  tokenspeed_mla: /usr/local/lib/python3.12/dist-packages/tokenspeed_mla/__init__.py
  is_tokenspeed_mla_available: True
  ```
- Container `tokenspeed-spike` is detached; subsequent steps use `docker exec`.

## Issues / blockers

- Container's installed TRT-LLM (rc14) is older than our host branch; full source overlay via `PYTHONPATH` triggers dep cascades. Bind-mounting only the 3 patched files works because container's `trtllm_gen.py` shares the exact line layout (`run_mla_generation` at 1368, FlashInfer MLA at 1420).
- `modelopt` warning about transformers 4.57.3 — harmless for our path.

## Next

Step 5 — `pytest tests/unittest/_torch/attention/test_tokenspeed_mla.py -v` inside `tokenspeed-spike`.
