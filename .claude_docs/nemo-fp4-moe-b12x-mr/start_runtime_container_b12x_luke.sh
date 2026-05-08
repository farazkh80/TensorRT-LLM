#!/bin/bash
# Start a named persistent container for the lukealonso/b12x decode-kernel hack:
#   - Pre-built TRT-LLM wheel installed (build/tensorrt_llm-1.3.0rc14-...whl)
#   - flashinfer 0.6.8 @ commit 8a9970b4 (b12x API still installed for the
#     existing FlashInferFusedMoE backend; the new B12xLukeFusedMoE backend
#     does NOT use flashinfer's b12x — it uses lukealonso's standalone package)
#   - lukealonso/b12x @ pinned SHA (Apache-2.0, "Restore Nemotron micro MoE
#     performance", 2026-05-07)
#   - cutlass-dsl 4.4.2 trio (satisfies both flashinfer and lukealonso/b12x)
#
# Source-tree overlay strategy:
#   The source tree at /workspace/TensorRT-LLM has imports (e.g. cache_dit) that
#   the rc12 base container lacks. So we DON'T set PYTHONPATH wholesale. Instead
#   we install the wheel (which gives us a working, importable tensorrt_llm in
#   site-packages) and then `sync_b12x_luke_files.sh` copies only the files we
#   touched (fused_moe submodule + llm_args.py) onto the wheel install. Run that
#   helper script after every code edit on the host, before the next bench.
#
# Usage:
#   bash .claude_docs/nemo-fp4-moe-b12x-mr/start_runtime_container_b12x_luke.sh
#   bash .claude_docs/nemo-fp4-moe-b12x-mr/sync_b12x_luke_files.sh   # after edits
#   docker exec -it b12x_luke_runtime bash
set -euo pipefail

NAME=${RUNTIME_NAME:-b12x_luke_runtime}
WHEEL=/workspace/TensorRT-LLM/build/tensorrt_llm-1.3.0rc14-cp312-cp312-linux_x86_64.whl
B12X_LUKE_SHA=1378cea76d2c0ca0f4cc48835d9b9b41dd785cb4
FLASHINFER_SHA=8a9970b45a1e5bddace1f9d26b1b7a07a77ba504

# Idempotent: drop existing container of same name if any
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
    --name "$NAME" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size=8g \
    -v /home/farazkh_scratch/TensorRT-LLM:/workspace/TensorRT-LLM \
    -v /home/farazkh_scratch/.cache/huggingface:/root/.cache/huggingface \
    -v /home/farazkh_scratch/logs:/workspace/logs \
    -w /root \
    -e LD_LIBRARY_PATH=/usr/local/tensorrt/lib:/usr/local/cuda/lib64 \
    nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12 \
    bash -c "
        set -e
        echo '[runtime] installing pre-built TRT-LLM wheel (rc14)...'
        pip install --no-deps --force-reinstall --no-cache-dir '$WHEEL' 2>&1 | tail -3
        echo '[runtime] installing flashinfer @ b12x commit (no-deps)...'
        pip install --no-build-isolation --no-deps --force-reinstall --no-cache-dir \
            'git+https://github.com/flashinfer-ai/flashinfer.git@${FLASHINFER_SHA}' 2>&1 | tail -3
        echo '[runtime] installing lukealonso/b12x @ ${B12X_LUKE_SHA} (no-deps, no-build-isolation)...'
        pip install --no-build-isolation --no-deps --force-reinstall --no-cache-dir \
            'git+https://github.com/lukealonso/b12x.git@${B12X_LUKE_SHA}' 2>&1 | tail -3
        echo '[runtime] installing missing rc14 deps absent from rc12 base image (cache_dit)...'
        pip install --no-deps --no-cache-dir 'cache_dit' 2>&1 | tail -2
        echo '[runtime] cutlass-dsl 4.4.2 trio (required for both flashinfer b12x and lukealonso b12x)...'
        # Order matters: libs-base FIRST (.pth survives), libs-cu13 LAST (newer
        # libcute_dsl_runtime.so overwrites libs-base's older one — required for
        # sm_120a FP4 MMA encoding to pass ptxas).
        pip install --no-deps --force-reinstall --no-cache-dir 'nvidia-cutlass-dsl==4.4.2' 2>&1 | tail -2
        pip install --no-deps --force-reinstall --no-cache-dir 'nvidia-cutlass-dsl-libs-base==4.4.2' 2>&1 | tail -2
        pip install --no-deps --force-reinstall --no-cache-dir 'nvidia-cutlass-dsl-libs-cu13==4.4.2' 2>&1 | tail -2
        echo '[runtime] verifying imports...'
        python3 -c 'import tensorrt_llm; print(\"tensorrt_llm\", tensorrt_llm.__version__, \"@\", tensorrt_llm.__file__)'
        python3 -c 'from cutlass.cute.nvgpu.warp.mma import Field; print(\"cutlass Field:\", list(Field))'
        python3 -c 'from flashinfer import B12xMoEWrapper, b12x_fused_moe; print(\"flashinfer b12x: OK\")'
        python3 -c 'import b12x; print(\"lukealonso b12x:\", b12x.__file__)'
        python3 -c 'from b12x.integration import b12x_moe_fp4, b12x_sparse_moe_fp4, B12XFP4ExpertWeights, allocate_tp_moe_workspace_pool; print(\"b12x.integration: OK\")'
        echo '[runtime] ready. keeping container alive.'
        sleep infinity
    "

echo "Container '$NAME' started. Tail bootstrap log:"
echo "  docker logs -f $NAME"
echo ""
echo "Wait for ready then sync source edits into site-packages:"
echo "  docker logs $NAME 2>&1 | tail -1"
echo "  bash .claude_docs/nemo-fp4-moe-b12x-mr/sync_b12x_luke_files.sh"
echo ""
echo "Attach shell:"
echo "  docker exec -it $NAME bash"
echo ""
echo "Stop:"
echo "  docker rm -f $NAME"
