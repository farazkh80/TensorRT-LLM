# Step 2: Mount

**Date:** 2026-05-13
**Experiment:** tokenspeed-mla-dsv3-lite
**GPU / SM:** B300 SXM6 AC / sm_103

## What ran

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14
```

Planned `docker run` (executed at step 4):

```bash
docker run --rm -it --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$PWD:/workspace/TensorRT-LLM" \
    -w /workspace/TensorRT-LLM \
    nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14 bash
```

## Result

- Image present: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14` (40.1 GB)
- Run command prepared; container start is bundled with step 4 install so we do it in one go.
- Model checkpoint already inside `$PWD/.claude_docs/models/nvfp4_moe_only/` → visible inside container at `/workspace/TensorRT-LLM/.claude_docs/models/nvfp4_moe_only/`.

## Issues / blockers

None.

## Next

Step 3 — confirm host-side edits (already done) → Step 4 install (pause for confirmation).
