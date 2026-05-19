# Design — `TokenSpeedMLAAttention` (Real Attention Backend, MoE-Pattern)

**Date:** 2026-05-13
**Authors:** Faraz Khoubsirat (with Claude assistance)
**Status:** Design draft, not yet implemented
**Context:** Follow-up to the TokenSpeed MLA spike (see [summary.md](summary.md)). The spike's env-var swap is plumbed into `FlashInferTrtllmGenAttention.run_mla_generation`, which DSV3-Lite NVFP4 + sm_103 + default config does not call. This doc proposes a real integration modeled on PR [#13773](https://github.com/NVIDIA/TensorRT-LLM/pull/13773)'s `FlashInferNvfp4Sm12xFusedMoE(CutlassFusedMoE)`.

## Goal & non-goals

**Goal:** Make `attn_backend="TOKENSPEED_MLA"` produce a measurable kernel swap and A/B-able perf delta on a real MLA model (DSV3-Lite NVFP4 as the bring-up target; Kimi K2.5 as the production target). The integration must follow the same registration / gating pattern already used for MoE backends so a future maintainer can read one file (and the MoE PR for context) and understand it.

**Non-goals:**
- Replacing the entire `TrtllmAttention` C++ path. We override the MLA decode launcher only and inherit everything else (block tables, KV cache, scheduling, sparse, spec-dec).
- Production merge to `main`. This is to produce measurement infrastructure for the TRTLLM-12510 follow-up (CTM+RTS port). Once Albert Di / Julien have a real C++-side TokenSpeed integration, this Python backend can be removed.

## Quick verification path (do this first — ~30 min, no new code)

Discovered while writing this doc: `tensorrt_llm/_torch/attention_backend/trtllm.py:31-32` defines an existing env var that selects between the C++ thop direct path and the FlashInfer trtllm-gen Python path:

```python
_TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION = (os.environ.get(
    "TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION", "0") == "1")
```

Default is **off** → C++ thop → `fmhaSm100fKernel_*` direct cubin launch → spike env-var swap never fires.

Setting it on routes attention through `FlashInferTrtllmGenAttention.run_mla_generation` (subject to `trtllm_gen.is_supported(...)` returning True at line 1351). That's where our spike's `TLLM_TOKENSPEED_MLA=1` swap lives.

**Action:** rerun the nsys A/B with **both** env vars set:
```bash
TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 TLLM_TOKENSPEED_MLA=1 \
    nsys profile -o nsys-ts-mtp1-trtllm-gen \
    python quickstart_advanced.py --attention_backend TRTLLM ... 
```

If `trtllm_gen.is_supported(...)` returns True for DSV3-Lite NVFP4, the spike code activates and we get real perf data. If `is_supported()` rejects this config (e.g. because of NVFP4 + MLA + multi-CTA combination), we fall through to the full design below.

## Class design — `TokenSpeedMLAAttention(TrtllmAttention)`

Inherits from `TrtllmAttention` so all the heavy lifting (metadata, block tables, KV cache, sparse, spec-dec) is reused. Overrides only the MLA decode dispatch.

```python
# tensorrt_llm/_torch/attention_backend/tokenspeed_mla_attention.py  (new file)
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from .trtllm import TrtllmAttention
from .tokenspeed_mla import (
    is_tokenspeed_mla_available,
    tokenspeed_batch_decode_with_kv_cache_mla,
)

class TokenSpeedMLAAttention(TrtllmAttention):
    """MLA decode through tokenspeed-mla CuTe DSL; everything else inherits.

    Mirrors PR #13773's FlashInferNvfp4Sm12xFusedMoE(CutlassFusedMoE) pattern:
    a peer of TrtllmAttention selected via the attn_backend registry, gated
    by can_implement() on arch + package + MLA enablement.
    """

    @classmethod
    def support_mla(cls) -> bool:
        return True

    @classmethod
    def can_implement(
        cls,
        *,
        is_mla_enable: bool,
        kv_cache_dtype: torch.dtype,
        q_len_per_req: int,
    ) -> Tuple[bool, str]:
        if not is_mla_enable:
            return False, "TokenSpeed MLA is decode-only and requires MLA"
        if not is_sm_100f(get_sm_version()):
            return False, "TokenSpeed MLA requires data-center Blackwell (SM 10.0/10.3)"
        if not is_tokenspeed_mla_available():
            return False, "tokenspeed_mla package not importable"
        if kv_cache_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            return False, f"TokenSpeed MLA does not accept {kv_cache_dtype} KV cache"
        if q_len_per_req < 2:
            # No fold_sq_factor benefit; fall back to parent so we don't pay
            # JIT-compile cost for no perf win.
            return False, "q_len_per_req < 2: no fold_sq_factor benefit"
        return True, ""

    def _run(self, q, k, v, output, output_sf, metadata, forward_args, *args, **kwargs):
        # Only intercept MLA decode tokens; defer everything else (context,
        # mixed-batch, non-MLA) to parent.
        if (self.is_mla_enable
                and forward_args.attention_input_type
                    == AttentionInputType.generation_only
                and metadata.num_generations > 0
                and self._tokenspeed_can_run(metadata, forward_args)):
            self._run_mla_decode_tokenspeed(
                q, output, metadata, forward_args)
            return
        return super()._run(q, k, v, output, output_sf, metadata,
                            forward_args, *args, **kwargs)
```

### `_run_mla_decode_tokenspeed` — the kernel-launch site

This is the ~40 lines that move from `FlashInferTrtllmGenAttention.run_mla_generation` (`trtllm_gen.py:1368-1437`) into the new backend. The shape is essentially the same as what our spike's env-var swap calls today; we just relocate it.

```python
def _run_mla_decode_tokenspeed(
        self, q, output, metadata, forward_args):
    """Launch tokenspeed-mla decode for the generation-only sub-batch."""
    # Block tables, KV cache buffers — same as FlashInferTrtllmGenAttention does.
    batch_beam = metadata.num_generations * metadata.beam_width
    block_tables = self._build_block_tables_for_mla(
        metadata, batch_beam)
    kv_cache = metadata.kv_cache_manager.get_buffers(
        self.layer_idx, kv_layout=DEFAULT_KV_LAYOUT)
    if kv_cache.ndim == 5:
        kv_cache = kv_cache.squeeze(2)

    mla_head_dim_qk = self.kv_lora_rank + self.qk_rope_head_dim
    q_len_per_req = q.shape[0] // batch_beam if batch_beam else 1
    query = q.view(batch_beam, q_len_per_req, self.num_heads, mla_head_dim_qk)

    bmm1_scale = 1.0 / (
        self.q_scaling
        * math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim))
    bmm2_scale = 1.0
    out_buf = output if q_len_per_req == 1 else None

    mla_out = tokenspeed_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=metadata.tokenspeed_workspace,  # see "Workspace" below
        qk_nope_head_dim=self.qk_nope_head_dim,
        kv_lora_rank=self.kv_lora_rank,
        qk_rope_head_dim=self.qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=metadata.sequence_lengths,
        max_seq_len=metadata.max_past_kv_length,
        out=out_buf,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=None,
    )
    if q_len_per_req > 1:
        output.copy_(mla_out.reshape_as(output))
```

### `_build_block_tables_for_mla` — duplicate or share?

The block-table builder exists in two places already (`FlashInferTrtllmGenAttention._build_block_tables`, plus the C++ thop equivalent). Cleanest move: hoist a shared helper to `attention_backend/_mla_helpers.py` and call from both. Acceptable spike-level move: inline-duplicate ~15 lines into `tokenspeed_mla_attention.py` and leave a TODO.

## Registry change

`tensorrt_llm/_torch/attention_backend/utils.py`, replace the spike's placeholder:

```python
elif backend_name == "TOKENSPEED_MLA":
    from .tokenspeed_mla_attention import TokenSpeedMLAAttention
    return TokenSpeedMLAAttention
```

(Today's spike entry returns `TrtllmAttention` with a logger warning — the placeholder.)

## What `forward()` callers see

Nothing changes. `TokenSpeedMLAAttention.forward()` is inherited from `TrtllmAttention.forward()`, which calls `self._run(...)`. Our override intercepts at `_run` for the MLA-decode-generation-only branch; everything else (context prefill, mixed-batch, non-MLA layers in hybrid models, sparse, sinks, spec-dec) falls through to parent.

## Workspace

`TrtllmAttention` already manages `attention_workspace` via `metadata.workspace`. TokenSpeed's wrapper accepts `params.workspace.view(-1, 4)` reinterpreted to int8 (the spike wrapper already does this). No new allocation needed; we re-use the existing workspace buffer.

## CLI surface

`examples/llm-api/quickstart_advanced.py`, extend `--attention_backend` choices:

```python
choices=['VANILLA', 'TRTLLM', 'FLASHINFER',
         'FLASHINFER_STAR_ATTENTION',
         'TOKENSPEED_MLA'],   # ← add
```

Users select with `--attention_backend TOKENSPEED_MLA`. Falls back to `TrtllmAttention` silently if `can_implement()` returns False (logged once).

## Test plan

Existing `tests/unittest/_torch/attention/test_tokenspeed_mla.py` parity test continues to be the unit-level guard. Add:

1. **Backend-selection test**: assert `get_attention_backend("TOKENSPEED_MLA")` returns `TokenSpeedMLAAttention` on supported arch, `TrtllmAttention` (with warning) on unsupported.
2. **End-to-end kernel verify on B300**: rerun the step-6 nsys sweep with `--attention_backend TOKENSPEED_MLA`. The variant trace MUST contain TokenSpeed CuTe DSL kernel symbols that the baseline trace does not. If still identical, the integration is wrong.
3. **A/B perf**: re-run `trtllm-bench` baseline vs variant on DSV3-Lite NVFP4 (BS=1..16, q_len_per_req=2 via MTP=1). Capture decode TPOT and total throughput. This is the missing data point from the spike.

## What gates `_tokenspeed_can_run` at runtime

`can_implement()` is the static gate; `_tokenspeed_can_run(metadata, forward_args)` is the per-call gate. Conditions:

- `metadata.num_generations > 0` and `forward_args.attention_input_type == generation_only`
- `q_len_per_req >= 2` (the MTP / spec-dec regime where `fold_sq_factor` wins)
- `forward_args.attention_sinks is None` (TokenSpeed rejects sinks per the wrapper)
- `metadata.is_chunked_prefill_for_mla_context(...)` is False (chunked context goes through parent)
- `metadata.has_cached_kv_for_mla_context(...)` is False (same)

If any fail, defer to parent.

## Risks

- **Parity divergence (already known)**: 2 spec-decode cases failed parity in step 5 (max abs 0.33 / max rel ~1166×). The integration must either (a) accept the divergence as bit-exact-not-required for inference quality, (b) be gated off by default until upstream tokenspeed-mla fixes the reduction order, or (c) match-test against downstream task accuracy on real prompts. **Pre-merge: must agree with Albert Di on acceptable tolerance.**
- **C++ thop dispatch may have side effects we miss**: TrtllmAttention's `_run()` ends in a thop call that updates KV cache + may log perf counters. By replacing the MLA-decode branch we must replicate ALL its side effects (KV append, counters). Mitigation: read `run_mla_generation` carefully; that's exactly what FlashInferTrtllmGenAttention already does — same shape.
- **`q_len_per_req >= 2` gate may be wrong for some models**: MTP=0 models would never use the backend. Decision: that's fine — there's no perf win for MTP=0 anyway.
- **`tokenspeed-mla 0.1.2` LSE kernel bug**: must be either upstream-fixed or patched at install time. Spike currently patches inside the container. For real integration, prefer pinning a future fixed release.
- **API stability tests**: adding a backend name to a registry typically doesn't break api_stability, but verify before merge.

## File-level diff vs. spike

| File | Spike | Real integration |
|---|---|---|
| `tensorrt_llm/_torch/attention_backend/tokenspeed_mla.py` | New (wrapper) | Unchanged — still the kernel-level drop-in |
| `tensorrt_llm/_torch/attention_backend/tokenspeed_mla_attention.py` | — | **New** — the backend class |
| `tensorrt_llm/_torch/attention_backend/utils.py` | Returns `TrtllmAttention` for `TOKENSPEED_MLA` | Returns `TokenSpeedMLAAttention` |
| `tensorrt_llm/_torch/attention_backend/trtllm_gen.py` | Env-var swap inside `run_mla_generation` | **Revert.** The swap was dead code; the real path is via `TokenSpeedMLAAttention._run`. |
| `examples/llm-api/quickstart_advanced.py` | (unchanged in spike) | Add `TOKENSPEED_MLA` to `--attention_backend` choices |
| `tests/unittest/_torch/attention/test_tokenspeed_mla.py` | Parity at kernel level | Add backend-selection assertion test |

The `trtllm_gen.py` revert is important — having two integration points (env-var swap AND backend class) for the same logical feature is the kind of duplication the trtllm-code-contribution skill warns against.

## Effort estimate

| Phase | Effort | Outputs |
|---|---|---|
| Quick verify (Option C, no new code) | ~30 min | Either: confirmation env-var swap fires + first perf number, OR confirmation `trtllm_gen.is_supported()` rejects DSV3-Lite NVFP4 (which we then handle in the full integration) |
| Full integration (Option A above) | ~1 day | New backend class, registry change, CLI choice, test assertion |
| Re-verify with nsys A/B + trtllm-bench | ~0.5 day | The perf number we missed in step 6/7 |
| Hand-off doc + PR description | ~1 hour | Production-ready PR text for upstream review (still draft until Albert Di accepts the parity divergence) |

**Total: ~2 days from go-ahead to a real, measured A/B result.**

## References

- PR #13773 (MoE backend pattern): https://github.com/NVIDIA/TensorRT-LLM/pull/13773
- Spike summary: [summary.md](summary.md)
- TrtllmAttention dispatch: `tensorrt_llm/_torch/attention_backend/trtllm.py:1204` (`_run`) and `:1351` (FlashInfer-vs-thop branch on `_TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION`)
- FlashInferTrtllmGenAttention MLA decode shape (to copy): `trtllm_gen.py:1368-1437`
- TokenSpeed kernel wrapper: `tokenspeed_mla.py` (committed in `a12c0c8fc6`)
