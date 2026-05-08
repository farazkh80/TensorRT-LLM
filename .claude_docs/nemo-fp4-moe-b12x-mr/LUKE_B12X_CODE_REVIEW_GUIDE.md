# `lukealonso/b12x` kernel review guide

Detailed file:line pointers for everything tested in the `b12x-luke-decode` investigation. Use this as a tour map when reviewing the bench results and the conclusions in `B12X_LUKE_RESULTS.md` / `FI_VS_LUKE_DELTA.md`.

All paths assume the runtime container `b12x_luke_runtime` (or the host source tree where indicated). Container is started by `start_runtime_container_b12x_luke.sh` in this directory and pins:

- TRT-LLM: rc14 wheel from `build/`, with source-tree overlay via `sync_b12x_luke_files.sh`
- flashinfer: `0.6.8` @ commit `8a9970b45a1e5bddace1f9d26b1b7a07a77ba504`
- **lukealonso/b12x: master commit `1378cea76d2c0ca0f4cc48835d9b9b41dd785cb4`** ("Restore Nemotron micro MoE performance", 2026-05-07)
- cutlass-dsl 4.4.2 trio

## TRT-LLM-side code we wrote

| Concern | Path | Key lines |
|---|---|---|
| New backend class | `tensorrt_llm/_torch/modules/fused_moe/fused_moe_b12x_luke.py` | full file, ~280 lines |
| `can_implement` SM/quant/dtype gates | same | L71–93 (`_SUPPORTED_SM_VERSIONS`, `can_implement`) |
| Construction-time rejects (`ep_size>1`, alltoall, gptoss) | same | L106–120 |
| Hybrid CUTLASS-prefill / b12x-decode env-var dispatch | same | L126–143 (`_prefill_via_cutlass_threshold`, `_route_to_cutlass`) |
| `post_load_weights` — luke weight conversion | same | L145–266 |
| `quantize_input` passthrough | same | L267–294 |
| `run_moe` — calls `b12x.integration.b12x_moe_fp4` | same | L296–354 |
| Backend wiring — `B12X_LUKE` branch | `tensorrt_llm/_torch/modules/fused_moe/create_moe.py` | L102–118 (mirrors existing `FLASHINFER` branch) |
| Re-export | `tensorrt_llm/_torch/modules/fused_moe/__init__.py` | L2 (import), L27 (`__all__`) |
| `MoeConfig.backend` literal | `tensorrt_llm/llmapi/llm_args.py` | L552–555 (added `"B12X_LUKE"`) |

The committed code lives on branch `b12x-luke-decode` (commit `374b483ab0`), pushed to `farazkh80/TensorRT-LLM`.

## luke kernel sources (inside container)

**Master HEAD `1378cea76d2c0ca0f4cc48835d9b9b41dd785cb4` — what gets installed by `start_runtime_container_b12x_luke.sh`:**

| File (in container) | Lines | Role | Warp specialization? |
|---|---:|---|---|
| `/usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py` | 3,940 | Top-level dispatcher, workspace plans, `_launch_compact_static`, `b12x_moe_fp4` | n/a (dispatcher) |
| `/usr/local/lib/python3.12/dist-packages/b12x/moe/fused/micro.py` | 1,725 | `MoEMicroKernel` family (relu2, silu) | **NO** — flat 16 warps (`_NUM_WARPS=16`, `_BLOCK_DIM=512`) |
| `/usr/local/lib/python3.12/dist-packages/b12x/moe/fused/static.py` | 1,789 | `MoEStaticKernel` family | **YES** — 5 warps (4 MMA + 1 TMA-load), `num_mma_warps=4` at L95 |
| `/usr/local/lib/python3.12/dist-packages/b12x/moe/fused/dynamic.py` | 1,692 | `MoEDynamicKernel` family | **YES** — 5 warps + `pipeline.CooperativeGroup`, `num_mma_warps=4` at L270 |
| `/usr/local/lib/python3.12/dist-packages/b12x/moe/fused/relu2.py` | small | Activation specs (`MoEMicroKernelRelu2`, `MoEStaticKernelRelu2`, etc.) | n/a |
| `/usr/local/lib/python3.12/dist-packages/b12x/moe/fused/silu.py` | small | Same for silu | n/a |
| `/usr/local/lib/python3.12/dist-packages/b12x/cute/fp4.py` | ~3,200 | `swizzle_block_scale`, `quantize_grouped_nvfp4_torch`, FP4 utility kernels | n/a |
| `/usr/local/lib/python3.12/dist-packages/b12x/integration/__init__.py` | 64 | Public API surface; re-exports `b12x_moe_fp4`, `B12XFP4ExpertWeights`, `TPMoEWorkspacePool`, etc. | n/a |

To inspect any of these on the host (without entering the container):
```bash
docker cp b12x_luke_runtime:/usr/local/lib/python3.12/dist-packages/b12x/moe/fused/micro.py /tmp/luke_micro.py
docker cp b12x_luke_runtime:/usr/local/lib/python3.12/dist-packages/b12x/moe/fused/static.py /tmp/luke_static.py
docker cp b12x_luke_runtime:/usr/local/lib/python3.12/dist-packages/b12x/moe/fused/dynamic.py /tmp/luke_dynamic.py
docker cp b12x_luke_runtime:/usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py /tmp/luke_tp_moe.py
```

## Warp-specialization markers — exact line numbers

The "smoking gun" for the architectural comparison.

### Luke's `static.py` (warp-specialized)

```
$ grep -n "num_mma_warps\|tma_load_warp_id\|threads_per_cta\|setmaxregister\|warpgroup\.make_smem_layout" /tmp/luke_static.py | head -8
95:        self.num_mma_warps = 4
96:        self.tma_load_warp_id = self.num_mma_warps
97:        self.num_threads_per_warp = 32
98:        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
104:            num_threads=self.num_mma_warps * self.num_threads_per_warp,
108:            num_threads=self.threads_per_cta,
130:        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
500:            block=[self.threads_per_cta, 1, 1],
```

### Luke's `dynamic.py` (warp-specialized)

```
$ grep -n "num_mma_warps\|tma_load_warp_id\|CooperativeGroup" /tmp/luke_dynamic.py | head -8
270:        self.num_mma_warps = 4
271:        self.tma_load_warp_id = self.num_mma_warps
272:        self.num_threads_per_warp = 32
273:        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
278:            barrier_id=1, num_threads=self.num_mma_warps * self.num_threads_per_warp,
281:            barrier_id=2, num_threads=self.threads_per_cta,
531:            block=[self.threads_per_cta, 1, 1],
657:        cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps)
```

### Luke's `micro.py` (flat — NO warp specialization)

```
$ grep -n "_NUM_WARPS\|_BLOCK_DIM\|num_mma_warps\|tma_load_warp" /tmp/luke_micro.py | head -8
49:_NUM_WARPS = 16
50:_BLOCK_DIM = _NUM_WARPS * 32
115:    rows_per_warp_fc1 = i_chunk // _NUM_WARPS
148:    rows_per_warp_fc1 = i_chunk // _NUM_WARPS
# (no num_mma_warps / tma_load_warp_id matches anywhere in micro.py)
```

### FI's `moe_micro_kernel.py` (warp-specialized — for comparison)

Inside container at `/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py`:

```
399:        self.num_mma_warps = 4
400:        self.tma_load_warp_id = self.num_mma_warps
401:        self.num_threads_per_warp = 32
402:        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp  # = 160
1318:        if warp_idx < self.num_mma_warps:                              # MMA warps 0-3
1319:            cute.arch.setmaxregister_increase(self.mma_register_requirement)
2188:        elif warp_idx == self.tma_load_warp_id:                        # DMA warp 4
2189:            cute.arch.setmaxregister_decrease(self.load_register_requirement)
```

## Dispatch logic — exact line numbers

### `_launch_compact_static` in luke's `tp_moe.py`

```
$ grep -n "def _launch_compact_static\|use_micro_direct\|_compiled_direct_micro_accepts_block_dim\|micro_cls.launch\|_get_static_kernel" /tmp/luke_tp_moe.py | head -10
2985:def _launch_compact_static(
3012:    use_micro_direct = quant_mode in {"nvfp4", "w4a16"} and micro_cls.is_supported(
3034:        if _compiled_direct_micro_accepts_block_dim(
3039:            micro_cls.launch(
3133:    compiled, mac = _get_static_kernel(    # fall-through to MoEStaticKernel
```

The dispatch decision for our shape (m=1, k=1024, n=2688, num_topk=22, E=512, relu2):

1. L3012: `use_micro_direct = True` (luke's micro `is_supported` returns True)
2. L3034: `_compiled_direct_micro_accepts_block_dim(compiled, 512) = True` (we probed this)
3. L3039: `MoEMicroKernel.launch(...)` fires — the **flat** kernel from `micro.py`
4. L3133 is unreachable for our shape unless we patch around the use_micro_direct branch

### Top-level `b12x_moe_fp4` dispatcher

```
$ grep -n "def b12x_moe_fp4\|_is_exact_relu2_bs1_nemotron_case\|_resolve_workspace_layout\|select_tp_moe_backend\|impl == \"static\"\|impl == \"dynamic\"" /tmp/luke_tp_moe.py | head -10
778:def select_tp_moe_backend(  # routed_rows ≤ 640 → "static", else "dynamic"
927:        implementation == "static"
2721:def _is_exact_relu2_bs1_nemotron_case(  # has `return False` gate at L2731 — unreachable
2747:def _get_exact_relu2_bs1_nemotron_launcher(  # has the `micro_mac` NameError at L2816
3160:def b12x_moe_fp4(
3296:    if impl == "static":
3347:    if impl == "dynamic":
3369:    else:                                # _launch_compact_static call
```

For our shape (`routed_rows = 22 ≤ 640`), `select_tp_moe_backend` returns `"static"`, so `b12x_moe_fp4` calls `_launch_compact_static` at L3369, which then internally takes the `use_micro_direct` shortcut to luke's flat `MoEMicroKernel`.

### Broken `_is_exact_relu2_bs1_nemotron_case` (gated off, has a real upstream bug)

```
$ sed -n '2721,2750p' /tmp/luke_tp_moe.py
def _is_exact_relu2_bs1_nemotron_case(
    *, activation, a, w1_fp4, a1_gscale, a2_gscale, w2_fp4, topk_weights, topk_ids,
) -> bool:
    return False                  # ← unconditional gate
    if not (                      # ← dead code below
        activation == "relu2"
        ...
```

Removing the gate exposes:

```
$ grep -n "mac_override=micro_mac\|micro_mac =" /tmp/luke_tp_moe.py
2816:        mac_override=micro_mac,                   # ← inside _get_exact_relu2_bs1_nemotron_launcher
3064:        micro_mac = min(_get_impl_mac("micro", routed_rows=routed_rows), micro_work_tiles)
```

L2816 references `micro_mac` from inside `_get_exact_relu2_bs1_nemotron_launcher`, but `micro_mac` is only defined at L3064 inside the *caller* `b12x_moe_fp4`. So the launcher reads an undefined name. We confirmed this empirically (run #3 in `B12X_LUKE_RESULTS.md`):

```
File "/usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py",
  line 2816, in _get_exact_relu2_bs1_nemotron_launcher
    mac_override=micro_mac,
                 ^^^^^^^^^
NameError: name 'micro_mac' is not defined
```

## Patches we applied (and reverted) for diagnostic experiments

All patches operate on `/usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py` inside the container. We always backed up to `tp_moe.py.orig` (and `tp_moe.py.bak2`) and restored after each test. The `tp_moe.py` in the container is currently at upstream-clean state.

### Patch 1 — remove `return False` gate to probe Nemotron-bs1 fast path (run #3)

```python
# /home/farazkh_scratch/TensorRT-LLM/.claude_docs/nemo-fp4-moe-b12x-mr/_patch_tp_moe_trace.py
# (idempotent; same script also installed the trace prints below)

# What it changed: at L2731 of tp_moe.py:
-    return False
+    pass  # was: return False (gate disabled by hack)
```

Result: `NameError: name 'micro_mac' is not defined` (see L2816 / L3064 reference above). Confirmed the gate is upstream's workaround for a real bug.

### Patch 2 — inject `[trace-luke]` prints in `_launch_compact_static` (run #4)

Same `_patch_tp_moe_trace.py` (Patches A–D). Added 5 print statements:

- Entry of `_launch_compact_static` — prints `m`, `k`, `n`, `num_topk`, `E`, `share_input_across_experts`, `share_expert_scales`, `quant_mode`
- After `use_micro_direct = ...` — prints whether True/False
- Before `_compiled_direct_micro_accepts_block_dim` call — prints its result
- After `micro_cls.launch(...)` returns — prints `** MICRO LAUNCHED **`

Output proved luke's flat `MoEMicroKernel` IS firing every decode iteration, with `use_micro_direct=True`, gate `True`, `share_input=True`, `share_scales=True`. No fall-through.

### Patch 3 — force `use_micro_direct=False` to route to warp-spec `MoEStaticKernel` (run #6)

```bash
sed -i 's|^    if use_micro_direct:$|    if False:  # forced to skip flat MoEMicroKernel; route to warp-spec MoEStaticKernel|' \
  /usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py
```

This skips the if-block at L3013–3050 of the patched file, so dispatch falls through to L3133 (`_get_static_kernel` → `MoEStaticKernel.launch`). Result: TPOT 13.5717 ms (+18.1% vs FI). Forcing the warp-spec kernel did NOT recover the gap — proves the remaining delta is shape-specific kernel tuning, not the warp-spec architectural pattern.

## Bench logs — which log proves what

All under `/home/farazkh_scratch/logs/` (host, not committed).

| Run | Log | Variant tested | TPOT P50 | Outcome |
|---|---|---|---:|---|
| 1 | `b12x_luke_hybrid_20260508_060402.log` | v1 (per-expert gscale + no shared output buffer) | n/a | failed CUDA-graph capture (`caller-owned output buffer`) |
| 2 | `b12x_luke_hybrid_20260508_061327.log` | v2 (per-expert gscale + shared output buffer) | 13.4047 ms | 5/5 reqs OK; +16.6 % vs FI |
| 3 | `b12x_luke_micro_20260508_062243.log` | Patch 1: `return False` removed → bs1 fast path attempt | n/a | crashed `NameError: micro_mac` |
| 4 | `b12x_luke_trace_20260508_084118.log` | Patch 2: trace prints; scalar gscale | 12.9654 ms | 1/1 req OK; confirms flat micro IS firing |
| 5 | `b12x_luke_c9cc90ec_20260508_085059.log` | bisect to luke `c9cc90ec` (May 6) | 13.5516 ms | gap unchanged |
| 5b | `b12x_luke_986a405a_20260508_085932.log` | bisect to luke `986a405a` (May 4) | 13.5534 ms | gap unchanged |
| 6 | `b12x_luke_warpspec_20260508_094658.log` | Patch 3: force warp-spec `MoEStaticKernel` | 13.5717 ms | gap unchanged; small-batch tuning is missing |

For each, the relevant lines in the log to look at:

```bash
grep -nE "B12xLukeFusedMoE active|Model init total|PERFORMANCE OVERVIEW|TPOT|TTFT|trace-luke|caller-owned|micro_mac|MICRO LAUNCHED|\[end\]" <log>
```

## Companion docs in this directory

| File | Purpose |
|---|---|
| `B12X_LUKE_RESULTS.md` | Bench numbers + token-parity result + success-criteria table + root-cause writeup. Start here. |
| `FI_VS_LUKE_DELTA.md` | Full architecture comparison + bisect data + warp-spec swap evidence + refined verdict. The "deep dive" doc. |
| `PR_BODY_b12x_luke.md` | Long-form PR body for the fork-side PR (`farazkh80:b12x-luke-decode → b12x-hybrid`). |
| `PR_BODY_b12x_luke_NVIDIA.md` | Condensed PR body for the NVIDIA-side Draft PR (`farazkh80:b12x-luke-decode → NVIDIA:main`). |
| **This file** | Code-pointer review guide — file:line citations for everything tested. |
| `_patch_tp_moe_trace.py` | The trace-probe patcher script (idempotent, reapplies the trace prints). |
| `start_runtime_container_b12x_luke.sh` | Container bootstrap. |
| `sync_b12x_luke_files.sh` | Source-tree → site-packages sync helper. |
| `bench_kvoff_b12x_luke.yml` | Bench config with `moe_config.backend=B12X_LUKE`. |
| `parity_check_b12x_luke.py` | Token-parity script with `--moe-backend FLASHINFER|B12X_LUKE` flag. |

## Reproducing any single experiment

```bash
# 1. Bring up container (idempotent — drops existing one of same name)
bash .claude_docs/nemo-fp4-moe-b12x-mr/start_runtime_container_b12x_luke.sh

# 2. Wait for "[runtime] ready" in `docker logs b12x_luke_runtime`

# 3. Sync the b12x_luke source files into site-packages
bash .claude_docs/nemo-fp4-moe-b12x-mr/sync_b12x_luke_files.sh

# 4. (Optional) Apply patches for diagnostic runs:
docker exec b12x_luke_runtime python3 \
  /workspace/TensorRT-LLM/.claude_docs/nemo-fp4-moe-b12x-mr/_patch_tp_moe_trace.py
# OR for warp-spec swap (run #6):
docker exec b12x_luke_runtime sed -i \
  's|^    if use_micro_direct:$|    if False:  # forced ...|' \
  /usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py
# OR install a different luke commit:
docker exec b12x_luke_runtime pip install --no-build-isolation --no-deps \
  --force-reinstall --no-cache-dir \
  'git+https://github.com/lukealonso/b12x.git@<commit-sha>'

# 5. Bench (single 5-req run, ~5 min)
TS=$(date +%Y%m%d_%H%M%S)
HOST_LOG=/home/farazkh_scratch/logs/b12x_luke_${TS}.log
docker exec -d \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD=64 \
  -e PYTHONUNBUFFERED=1 \
  b12x_luke_runtime bash -c "(
    cd /workspace/TensorRT-LLM &&
    python3 -m tensorrt_llm.commands.bench \
      --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
      --model_path /workspace/TensorRT-LLM/.claude_docs/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
      throughput \
      --dataset /workspace/TensorRT-LLM/.claude_docs/nemo-fp4-moe-b12x-mr/bench_dataset_5x.jsonl \
      --extra_llm_api_options /workspace/TensorRT-LLM/.claude_docs/nemo-fp4-moe-b12x-mr/bench_kvoff_b12x_luke.yml \
      --backend pytorch --max_batch_size 1 --max_num_tokens 4096 \
      --num_requests 5 --warmup 0 --concurrency 1 --streaming
  ) > /workspace/logs/b12x_luke_${TS}.log 2>&1"
```
