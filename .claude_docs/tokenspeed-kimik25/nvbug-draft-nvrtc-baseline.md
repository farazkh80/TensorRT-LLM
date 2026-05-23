# NVBugs draft — TRT-LLM MLA decode NVRTC compilation failure on sm_103a (B300)

> **STATUS (2026-05-20): FIXED UPSTREAM, draft archived — do not file.**
>
> Resolved by upstream PR #14291 (`a173761069 [None][feat] Update the logic
> of FMHA JIT path`) which moved `CutlassUmmaConsumerAsyncPipeline` from
> `namespace cutlass` to `namespace trtllm::dev` and refreshed the cubin
> archives to match. After rebasing onto `upstream/main@f278c4f170` and
> rebuilding `libtensorrt_llm.so`, the TRTLLM baseline arm runs
> end-to-end on K2.6 (TP4 1k/1k conc=1, 148.6 tok/s aggregate, 6.73 ms/tok
> ITL — no NVRTC failure). Confirms root-cause hypothesis #3 below. See
> `nvrtc-rebase-verify.md` for the verification run.
>
> The rest of this file is kept as triage history — the root-cause
> analysis and three-hypothesis ranking remain instructive but the bug
> itself no longer needs filing.

---

This is a paste-ready NVBugs draft. Field mapping follows the standard NVBugs web-form layout.

---

## Synopsis (one-liner)

`[TRT-LLM 1.3.0rc15 / sm_103a] NVRTC fails to compile fmhaSm103aKernel ...HQk576HV512... ForGen family — "expected a type specifier" on trtllm::dev::CutlassUmmaConsumerAsyncPipeline<1,false,false,false>; blocks K2.6 / DSV3-family MLA-decode on B300.`

## Module / Component

- **Module:** TensorRT-LLM
- **Sub-component:** trtllmGenKernels / fmha (sm_103a MLA-decode kernel family)
- **Owner suggestion:** owners of `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/trtllmGen_fmha_export/` and the JIT/NVRTC preprocessing path (`CudaRunner.cpp`).

## Severity / Priority suggestion

- **Severity:** 2 — High. Blocks any K2.6 / DSV3-family MLA-decode model from running on B300 with `attn_backend: TRTLLM` (the default) on current main. There is **no working alternative MLA-decode backend on current main for sm_103a** in the production FP8-KV configuration; the in-flight `TOKENSPEED_MLA` backend only works under the patched BF16-KV snapshot.
- **Priority:** P1 — gates the K2.6 TokenSpeed-vs-baseline measurement on B300 and any downstream K2.6/DSV3 perf/eval work on B300.
- **Found in version:** TRT-LLM `1.3.0rc15` (current `main`).
- **Regression?** Suspected (the same kernel family has shipped on prior sm_103a builds). Bisection not yet done — see *Next steps for the owner*.

## Environment

- **GPU:** B300 SXM6, sm_103a.
- **Container:** `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`.
- **TRT-LLM source under test:** current `main` at commit `630cfeb56738b164d5b9b7d3dae27b3172ba43e3` (`tokenspeed-kimik25-eval-public` branch is rebased onto this HEAD; the TRT-LLM-side change set on this branch is the new `TokenSpeedMLAAttention` backend, which is *not* in the failing code path — the crash is in the default `TRTLLM` backend).
- **TRT-LLM reported version:** `TensorRT LLM version: 1.3.0rc15` (from log: `[TensorRT-LLM] TensorRT LLM version: 1.3.0rc15`).
- **Python / Torch in container:** Python 3.12, PyTorch `2.11.0a0+eb65b36914.nv26.02`.
- **Shadowing:** The container's `/usr/local/lib/.../tensorrt_llm` is shadowed by the host source via `PYTHONPATH=/workspace/TensorRT-LLM` (Python only). C++ kernels including the failing JIT bundle come from the container's `libtensorrt_llm`.
- **Reproducibility:** 100% on all 4 attempted configs (TP4/TP8 × {1k, 8k} × {conc=1, conc=16}). Fails during executor warmup before any user request is served.

## What fails — exact NVRTC error

The default PyTorch backend (`attn_backend: TRTLLM`) crashes during model warmup with `RuntimeError: Failed to preprocess kernel ...: Compilation failed: NVRTC_ERROR_COMPILATION` on one of four sm_103a MLA-decode kernels depending on `q_len_per_req`/CGA-mode shape:

| Config (TP / ISL / conc) | Failing kernel symbol |
|---|---|
| TP4 1k/1k conc=1 | `fmhaSm103aKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta128PagedKvDenseP32MultiCtasKvCgaVarSeqQ16Kv128StaticSwapsAbForGen` |
| TP4 1k/1k conc=16 | (same as above) |
| TP4 8k/1k conc=1 | `fmhaSm103aKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta128PagedKvDenseP32MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen` (no `Cga` prefix) |
| TP8 1k/1k conc=1 | `fmhaSm103aKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta128PagedKvDenseP32MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen` (`Q8Kv128` instead of `Q16Kv128`) |

All three distinct kernels share the same root-cause NVRTC error class. Verbatim diagnostic (from TP4 1k/1k log, lines 211–241):

```
[E] [CudaRunner.cpp:474]: Failed to preprocess kernel
  fmhaSm103aKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta128PagedKvDenseP32MultiCtasKvCgaVarSeqQ16Kv128StaticSwapsAbForGen
  : Compilation failed: NVRTC_ERROR_COMPILATION

fmha...ForGen(183): error: expected a type specifier
  typename trtllm::dev::CutlassUmmaConsumerAsyncPipeline<1, false, false, false>::SharedStorage mBarriers;

fmha...ForGen(333): error: expected a type specifier
  trtllm::dev::CutlassUmmaConsumerAsyncPipeline<1, false, false, false> mPipeline;

fmha...ForGen(3046): error: expected a type specifier
  trtllm::dev::CutlassUmmaConsumerAsyncPipeline<1, false, false, false>::PipelineState
    tmemP0ProdState{int32_t{0}, int32_t{1}, int32_t{0}};

fmha...ForGen(5629): error: expected a type specifier
  trtllm::dev::CutlassUmmaConsumerAsyncPipeline<1, false, false, false>::PipelineState
    tmemP0ConsState{};

fmha...ForGen(5631): error: expected a type specifier
  trtllm::dev::CutlassUmmaConsumerAsyncPipeline<1, false, false, false>::PipelineState
    tmemP0ConsReleaseState{};

fmha...ForGen(5635): error: expected a type specifier
  trtllm::dev::CutlassUmmaConsumerAsyncPipeline<1, false, false, false>::PipelineState
    tmemP1ConsState{};

fmha...ForGen(5637): error: expected a type specifier
  trtllm::dev::CutlassUmmaConsumerAsyncPipeline<1, false, false, false>::PipelineState
    tmemP1ConsReleaseState{};

[TRT-LLM] [E] [executor] [RANK N] Failed to initialize executor on rank N:
  Failed to preprocess kernel ...: NVRTC_ERROR_COMPILATION
```

The diagnostic message is identical on all four ranks (and on TP8: identical on all eight ranks) — this is not a per-rank or per-shape race; it is a structural issue in the kernel source the JIT path is feeding to NVRTC.

## Root-cause hypothesis

The class actually exists in the source tree:

- **Definition:** `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/trtllmGen_fmha_export/trtllm/dev/CutlassPipeline.h:1467`

```cpp
template <int NumStages,
          bool UsesCpAsyncBarrierArrive = false,
          bool UsesFenceBeforeProdCommit = false,
          bool UsesUmmaProducerCommit = false,
          class AtomThrShapeMNK = cute::Shape<cute::_1, cute::_1, cute::_1>>
class CutlassUmmaConsumerAsyncPipeline {
  using Pipeline = cutlass::PipelineUmmaConsumerAsync<NumStages, AtomThrShapeMNK>;
  ...
public:
  using PipelineState  = typename Pipeline::PipelineState;
  using SharedStorage  = typename Pipeline::SharedStorage;
  ...
};
```

The four-arg use site (`<1, false, false, false>`) should match the primary template with `AtomThrShapeMNK` defaulted. NVRTC nevertheless reports "expected a type specifier" on **every** reference to the class — both the bare class type and its nested `::SharedStorage` / `::PipelineState` types — which means NVRTC's TU never resolved a primary template for `CutlassUmmaConsumerAsyncPipeline` in this scope.

Plausible causes, in order of likelihood:

1. **Missing include in the Jitify bundle.** The `.cu` kernel sources for `fmhaSm103aKernel_...ForGen` either do not `#include "trtllm/dev/CutlassPipeline.h"` directly, or rely on a transitive include that no longer exists in the export bundle. (Jitify bundles `__jitify_rel_inc@__jitify_I1@@__jitify_name@crt/host_defines.h` and friends — see log line 119 — so a header *is* being driven through; the question is whether `CutlassPipeline.h` is in that set.)
2. **CUTLASS version mismatch on `cutlass::PipelineUmmaConsumerAsync`.** The class aliases `using Pipeline = cutlass::PipelineUmmaConsumerAsync<NumStages, AtomThrShapeMNK>;`. If the CUTLASS in the container's `libcutlass`/headers ships a different template signature for `PipelineUmmaConsumerAsync` (e.g., extra/changed template params, or a removed primary template), the alias substitution silently fails in NVRTC; nested-type substitution then yields the cascade of "expected a type specifier" diagnostics on `::SharedStorage` and `::PipelineState`. This is consistent with the *symptom* (a working class definition appears unusable from NVRTC at every nested-type use site).
3. **Pre-baked cubin / source mismatch.** The TRT-LLM source under test is `1.3.0rc15-dev` while the container is `1.3.0rc14`. The `libtensorrt_llm` shared object (which owns the JIT path) is from rc14. If rc15-dev changed the `CutlassUmmaConsumerAsyncPipeline` template signature (e.g., added/removed a template parameter) but the cubin .cu source the rc14 `libtensorrt_llm` is generating still uses the old shape, NVRTC's view of the header (rc15 host paths) and the kernel source (rc14 cubin) will not match.

Hypothesis (3) is testable by running with a pure rc14 PyTorch tree (no PYTHONPATH shadow). I have not done that test — see *Next steps for the owner*.

## Steps to reproduce

1. Pull `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`. Run on a B300 SXM6 host with 4 or 8 GPUs visible.
2. Fetch the patched K2.6 BF16-KV snapshot (or any DSV3-family checkpoint with `kv_lora_rank=512, qk_rope_head_dim=64` and BF16 Q+KV). The patched snapshot is at `/scratch/hf-cache-patched/k2.6-bf16kv/` on the repro host.
3. Launch the engine with `attn_backend: TRTLLM` (default). Minimal repro via:
   ```bash
   python .claude_docs/tokenspeed-kimik25/scripts/minimal_bench.py \
     --model /scratch/hf-cache-patched/k2.6-bf16kv \
     --tp 4 --isl 1024 --osl 1024 \
     --num_req 1 --conc 1 \
     --extra_llm_api_options .claude_docs/tokenspeed-kimik25/bench-config.yml
   # bench-config.yml has attn_backend: TRTLLM
   ```
4. Crash occurs during warmup (`[TRT-LLM] [E] [executor] [RANK N] Failed to initialize executor on rank N`).

Repro is **100% deterministic** across 4 distinct (TP, ISL, conc) combinations. No user request is required — the failure happens in the executor's NVRTC warmup before the first token is generated.

## Workarounds known

- **None on current main for sm_103a MLA decode.**
- Setting `attn_backend: TOKENSPEED_MLA` (an in-flight feature branch on `tokenspeed-kimik25-eval-public`) avoids the failing kernel, but is gated by `q_dtype == kv_dtype` (no production FP8-KV) and only ships under the TokenSpeed feature branch; not a general workaround.
- TRT backend (legacy) not attempted — DSV3-family MLA is PyTorch-only in current TRT-LLM.

## What this blocks

- Any K2.6 or DSV3-family MLA-decode evaluation on B300 with the default backend.
- The TokenSpeed-vs-TRTLLM A/B grid on K2.6 (the baseline arm cannot run, so no TS/baseline ratio can be quoted).
- Phase 4 of TRTLLM-12510 (K2.6 TokenSpeed eval).

## Next steps for the owner

1. **Bisect.** The TRT-LLM main HEAD `630cfeb567` (May 19 2026) reproduces. The release container `1.3.0rc14` ships an older `libtensorrt_llm`. Test: (a) repro on pure `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14` *without* `PYTHONPATH` shadowing; (b) bisect through `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/trtllmGen_fmha_export/` over the rc14→rc15 window for changes to the `CutlassUmmaConsumerAsyncPipeline` template signature or its `cutlass::PipelineUmmaConsumerAsync` alias.
2. **Check Jitify include manifest.** Confirm that `trtllm/dev/CutlassPipeline.h` is included in the JIT compilation unit for `fmhaSm103aKernel_...HQk576HV512...ForGen` cubin sources. The error mode (every nested-type use becomes "expected a type specifier") is consistent with the primary template never being declared in NVRTC's TU.
3. **CUTLASS pin.** Verify the CUTLASS version (header *and* device-side template definitions) shipped in the container matches what the trtllmGen kernel sources were generated against. If `cutlass::PipelineUmmaConsumerAsync` template signature has drifted, the trtllm `using Pipeline = ...` alias breaks silently.

## Repro logs (artifacts)

All four base-arm logs available on the repro host:

```
/home/scratch.fkhoubsirat_coreai/runs/k2.6-spike/phase4-bench/
├── bench-config/base.log                  (TP4 1k/1k conc=1,  CgaVarSeqQ16Kv128)
├── bench-1k1k_tp8_conc1/base.log          (TP8 1k/1k conc=1,  CgaVarSeqQ8Kv128)
├── bench-8k1k_tp4_conc1/base.log          (TP4 8k/1k conc=1,  VarSeqQ16Kv128 no-Cga)
└── bench-1k1k_tp4_conc16/base.log         (TP4 1k/1k conc=16, CgaVarSeqQ16Kv128)
```

Each log has 8+ identical NVRTC diagnostic blocks (one per rank); the first block is at line ~116 of `bench-config/base.log`.

## Related work / references

- Internal: TRTLLM-12510 (K2.6 TokenSpeed eval).
- Branch under test: `tokenspeed-kimik25-eval-public` at commit `630cfeb567`.
- Phase 4 writeup with broader context (TokenSpeed-arm numbers, caveats):
  `.claude_docs/tokenspeed-kimik25/phase4-summary.md`.
- Next-steps tracker for the blocking item:
  `.claude_docs/tokenspeed-kimik25/next-steps.md` (item 1).
- DSV3-Lite spike that established the parity/perf reference on the FlashInfer/trtllm-gen MLA path (different sm/arch — does **not** exercise sm_103a, hence not affected): see `.claude_docs/tokenspeed-mla-dsv3-lite/` (if uploaded).

---

## Submitter notes

- File against TensorRT-LLM. Use the verbatim Synopsis above.
- Severity 2 / Priority P1 unless an internal triage owner downgrades after a workaround is found.
- If asked for "minimal" repro: the smallest is `tp=4, isl=1024, osl=1024, num_req=1, conc=1` with `attn_backend: TRTLLM` on the BF16-KV K2.6 snapshot — the crash is in warmup and does not depend on `num_req`.
