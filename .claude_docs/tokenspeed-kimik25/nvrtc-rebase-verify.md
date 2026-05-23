# NVRTC baseline blocker — verified fixed by upstream rebase

**Date:** 2026-05-20
**Branch:** `tokenspeed-kimik25-eval-public` rebased onto `upstream/main` at `f278c4f170`
**Backup of pre-rebase HEAD:** `backup/tokenspeed-kimik25-eval-public-20260520-154134`

## TL;DR

The Phase 4 baseline NVRTC blocker (`fmhaSm103aKernel_...HQk576HV512...ForGen`
family failing under NVRTC with "expected a type specifier" on
`trtllm::dev::CutlassUmmaConsumerAsyncPipeline<1,false,false,false>`) is
**fixed in upstream/main** by PR #14291
(`a173761069 [None][feat] Update the logic of FMHA JIT path`).

After rebasing onto upstream/main and rebuilding `libtensorrt_llm.so`, the
TRTLLM baseline arm now runs end-to-end on K2.6. The NVBugs draft does not
need to be filed.

## What PR #14291 changed

`cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/trtllmGen_fmha_export/trtllm/dev/CutlassPipeline.h`:

```diff
-namespace cutlass {
+namespace trtllm::dev {
```

Plus restructured headers to use new local pipeline/tile-scheduler headers
(`CutlassSm90Pipeline.h`, `CutlassSm100Pipeline.h`, etc.) instead of
referencing `cutlass::PipelineUmmaConsumerAsync` directly. Also refreshed
~3000 cubin `.tar.zst` archives + 4 `libTrtLlmGen*.a` static libs so the
embedded JIT kernel sources match the new namespace.

This confirms root-cause hypothesis **#3** in the NVBugs draft (rc14 cubins
referenced `trtllm::dev::CutlassUmmaConsumerAsyncPipeline` while rc14
`CutlassPipeline.h` still declared it as `cutlass::CutlassUmmaConsumerAsyncPipeline`
— the two were out of sync during the namespace migration).

## Verification run

**Setup:**
- Container: `tokenspeed-spike-k26` (image `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`)
- Source: rebased HEAD `f278c4f170 + 6 TokenSpeed commits`, built via
  `scripts/build_wheel.py --cuda_architectures="100-real;103-real" --use_ccache --install --no-venv -j 32`
- Build time: ~47 min (16:51 → 16:39 UTC, 2026-05-20). Editable install,
  installed `tensorrt_llm 1.3.0rc15`.
- Model: `/scratch/hf-cache-patched/k2.6-bf16kv` (BF16-KV-patched snapshot)
- Config: `.claude_docs/tokenspeed-kimik25/bench-config_base.yml` —
  bench-config.yml with `attn_backend: TRTLLM`
- Hardware: B300 SXM6 (sm_103a), TP=4

**Command:**
```
python3 -u .claude_docs/tokenspeed-kimik25/scripts/minimal_bench.py \
  --model_dir /scratch/hf-cache-patched/k2.6-bf16kv \
  --tp_size 4 \
  --isl 1024 --osl 256 \
  --num_requests 2 --concurrency 1 --warmup 1 \
  --extra_llm_api_options .claude_docs/tokenspeed-kimik25/bench-config_base.yml
```

**Result:**
- ✅ LLM init: 279.9 s (no NVRTC failure — pre-rebase would have crashed here)
- ✅ Warmup: 256 tokens / 2.09 s (122.6 tok/s aggregate)
- ✅ Timed run: 2 requests / 3.44 s
- **Aggregate throughput:** 148.6 tok/s
- **Per-user median:** 148.89 tok/s (min/max 142.66 / 155.11)
- **ITL avg:** 6.73 ms/tok

Log: `.claude_docs/tokenspeed-kimik25/repro-logs/base_postfix_164235.log`

## Implications

1. **NVBugs draft is no longer actionable.** Keep
   `.claude_docs/tokenspeed-kimik25/nvbug-draft-nvrtc-baseline.md` as
   archived root-cause documentation, but do not file. Add a banner that
   the blocker was resolved by PR #14291 upstream.

2. **Phase 4 A/B ratio now achievable.** The Phase 4 TokenSpeed-arm
   numbers were collected with the **rc14 container's libtensorrt_llm**
   shadowed via PYTHONPATH (Python only). The new build replaces
   libtensorrt_llm with the rebased version. To get a clean apples-to-apples
   TokenSpeed/TRTLLM ratio on K2.6, the four Phase 4 configs need to be
   re-run on this rebased build with **both arms** (`TRTLLM` and
   `TOKENSPEED_MLA`).

3. **Tentative ratio at TP4 1k/1k conc=1:** TokenSpeed (Phase 4, old build)
   151.8 tok/s vs TRTLLM (rebased build) 148.6 tok/s = **TS +2.2% aggregate,
   +2.4% per-user median, −2.1% ITL**. This is meaningfully smaller than
   the DSV3-Lite spike's ~+10% — but the underlying libtensorrt_llm
   versions differ, so the comparison is not yet clean. Re-run TS arm on
   the rebased build before publishing the ratio.

## What to do next

1. Run the four Phase 4 configs on the rebased build with both
   `attn_backend: TRTLLM` and `attn_backend: TOKENSPEED_MLA`. The
   `run_bench_v2.sh` driver already does this — the only change is that
   no PYTHONPATH shadow is needed (the source is now the install).
2. Update `phase4-summary.md` and `next-steps.md` to reflect that the
   ratio numbers are available.
3. Decide whether to retire the NVBugs draft (mark "fixed upstream by
   #14291") or keep it as triage history.
