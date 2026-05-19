#!/usr/bin/env python3
"""Diagnostic patch for TrtllmAttentionWrapper.run() dispatch decision.

Wraps the `_TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION` if-branch in trtllm.py
(rc14, line 560 inside TrtllmAttentionWrapper.run()) with one-shot
stderr logging gated on the `TLLM_DIAG=1` env var. Tells us:

  - whether `_TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION` is True at runtime
  - what `trtllm_gen.is_supported(...)` returns (and the reject reason)
  - which branch fires (`trtllm_gen_attention` vs `thop.attention`)
  - whether `self.is_mla_enable` is True for this layer

NB: rc14's architecture differs from main — the dispatch lives in
TrtllmAttentionWrapper.run(), NOT in TrtllmAttention._run. The eventual
backend class will need to handle this (current main DOES have
TrtllmAttention._run, so build-from-source would simplify the override).

Idempotent via the `# DIAG PATCH` marker.

Run via:
    docker cp diagnose_dispatch.py "$CONTAINER":/tmp/diagnose_dispatch.py
    docker exec "$CONTAINER" python /tmp/diagnose_dispatch.py
"""

import sys
from pathlib import Path

TRTLLM_PY = Path(
    "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/attention_backend/trtllm.py"
)
MARKER = "# DIAG PATCH"


# Full if-condition as in rc14's TrtllmAttentionWrapper.run() — different
# from current main: `self.head_size` not `self.head_dim`, no `not use_sage_attn`,
# `self.tokens_per_block` not `metadata.tokens_per_block`, etc.
OLD = """        if _TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION and not helix_active and trtllm_gen.is_supported(
                q=q,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                out_dtype=output.dtype,
                mask_type=int(mask_type),
                has_alibi=(self.position_embedding_type == 4
                           or self.position_embedding_type == 5),
                is_padded=False,
                use_paged_kv_cache=(self.kv_cache_block_offsets is not None),
                tokens_per_block=self.tokens_per_block,
                beam_width=self.beam_width,
                position_shift_enabled=False,
                sink_token_length=self.sink_token_length,
                cross_attention=False,
                is_spec_decoding=self.is_spec_decoding_enabled,
                is_mla_enable=self.is_mla_enable,
                is_fused_qkv=is_fused_qkv,
                update_kv_cache=update_kv_cache,
                has_cross_kv=False,
                quant_config=self.quant_config,
                kv_cache_manager=self.kv_cache_manager,
                skip_softmax_threshold_scale_factor_prefill=self.
                skip_softmax_threshold_scale_factor_prefill,
                skip_softmax_threshold_scale_factor_decode=self.
                skip_softmax_threshold_scale_factor_decode,
        )[0]:
"""

NEW = """        # DIAG PATCH: capture is_supported result for one-shot logging.
        _diag_is_supported = trtllm_gen.is_supported(
                q=q,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                out_dtype=output.dtype,
                mask_type=int(mask_type),
                has_alibi=(self.position_embedding_type == 4
                           or self.position_embedding_type == 5),
                is_padded=False,
                use_paged_kv_cache=(self.kv_cache_block_offsets is not None),
                tokens_per_block=self.tokens_per_block,
                beam_width=self.beam_width,
                position_shift_enabled=False,
                sink_token_length=self.sink_token_length,
                cross_attention=False,
                is_spec_decoding=self.is_spec_decoding_enabled,
                is_mla_enable=self.is_mla_enable,
                is_fused_qkv=is_fused_qkv,
                update_kv_cache=update_kv_cache,
                has_cross_kv=False,
                quant_config=self.quant_config,
                kv_cache_manager=self.kv_cache_manager,
                skip_softmax_threshold_scale_factor_prefill=self.
                skip_softmax_threshold_scale_factor_prefill,
                skip_softmax_threshold_scale_factor_decode=self.
                skip_softmax_threshold_scale_factor_decode,
        )
        _diag_layer = getattr(self, "layer_idx", "?")
        if os.environ.get("TLLM_DIAG") == "1" and not getattr(self, "_diag_logged", False):
            print(f"[DIAG] layer={_diag_layer} _TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION={_TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION}", file=sys.stderr, flush=True)
            print(f"[DIAG] layer={_diag_layer} helix_active={helix_active}", file=sys.stderr, flush=True)
            print(f"[DIAG] layer={_diag_layer} is_supported={_diag_is_supported}", file=sys.stderr, flush=True)
            print(f"[DIAG] layer={_diag_layer} is_mla={self.is_mla_enable} is_fused_qkv={is_fused_qkv} update_kv_cache={update_kv_cache}", file=sys.stderr, flush=True)
            print(f"[DIAG] layer={_diag_layer} TLLM_TOKENSPEED_MLA={os.environ.get('TLLM_TOKENSPEED_MLA')}", file=sys.stderr, flush=True)
            self._diag_logged = True
        if _TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION and not helix_active and _diag_is_supported[0]:
            if os.environ.get("TLLM_DIAG") == "1" and not getattr(self, "_diag_branch_logged", False):
                print(f"[DIAG] layer={_diag_layer} BRANCH=trtllm_gen_attention (Python flashinfer path; SPIKE swap location)", file=sys.stderr, flush=True)
                self._diag_branch_logged = True
"""

ELSE_OLD = """        else:
            thop.attention(
                q,
                k,
                v,
                output,
                output_sf,
                self.workspace,
"""

ELSE_NEW = """        else:
            if os.environ.get("TLLM_DIAG") == "1" and not getattr(self, "_diag_branch_logged", False):
                _diag_layer = getattr(self, "layer_idx", "?")
                print(f"[DIAG] layer={_diag_layer} BRANCH=thop.attention (C++ thop direct-cubin path; SPIKE swap dead code)", file=sys.stderr, flush=True)
                self._diag_branch_logged = True
            thop.attention(
                q,
                k,
                v,
                output,
                output_sf,
                self.workspace,
"""


def main() -> None:
    src = TRTLLM_PY.read_text()
    if MARKER in src:
        print(f"[skip] {TRTLLM_PY} already diagnostically patched")
        return

    if src.count(OLD) != 1:
        sys.exit(f"[fail] if-condition anchor not found exactly once in {TRTLLM_PY}")
    if src.count(ELSE_OLD) != 1:
        sys.exit(f"[fail] else-branch anchor not found exactly once in {TRTLLM_PY}")

    # Ensure both `import os` and `import sys` are present in trtllm.py (rc14 doesn't
    # import them at top by default — they're only injected into trtllm_gen.py by the
    # SPIKE PATCH).
    new_imports = ""
    if "\nimport os\n" not in src and "\nimport os, " not in src:
        new_imports += "import os  # DIAG PATCH\n"
    if "\nimport sys\n" not in src and "\nimport sys, " not in src:
        new_imports += "import sys  # DIAG PATCH\n"
    if new_imports:
        # Insert after the first `import` line in the file.
        first_import_idx = src.find("\nimport ")
        if first_import_idx == -1:
            sys.exit("[fail] could not find an existing 'import' line to anchor injection")
        # Position right after the newline that precedes the existing import; we want
        # to insert before that line so our injected imports come first.
        inj_at = first_import_idx + 1  # past the leading \n
        src = src[:inj_at] + new_imports + src[inj_at:]

    src = src.replace(OLD, NEW)
    src = src.replace(ELSE_OLD, ELSE_NEW)

    TRTLLM_PY.write_text(src)
    print(f"[ok] diagnostic patch applied to {TRTLLM_PY}")
    print("    Re-run minimal_generate.py with TLLM_DIAG=1 to see dispatch decision.")


if __name__ == "__main__":
    main()
