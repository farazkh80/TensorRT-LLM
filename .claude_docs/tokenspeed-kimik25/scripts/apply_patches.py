#!/usr/bin/env python3
"""Inline-patches for the rc14 container.

Applied via docker exec at experiment start. Three patches:
  1. tensorrt_llm/_torch/attention_backend/utils.py
     - Add TOKENSPEED_MLA branch to get_attention_backend.
  2. tensorrt_llm/_torch/attention_backend/trtllm_gen.py
     - Add `import os`.
     - Wrap the FlashInfer MLA decode call with TLLM_TOKENSPEED_MLA env-var swap.
  3. tokenspeed_mla/mla_decode.py
     - Workaround for 0.1.2 bug where the BF16/FP16 kernel reinterprets
       `lse` unconditionally; allocate a real LSE tensor at compile + runtime.

Idempotent: re-running is a no-op (each patch checks for a marker).
"""

import sys
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages")
ATTN = SITE / "tensorrt_llm/_torch/attention_backend"
TS_MLA = SITE / "tokenspeed_mla"

MARKER = "# SPIKE PATCH"


def patch_utils_py() -> None:
    p = ATTN / "utils.py"
    s = p.read_text()
    if MARKER in s:
        print(f"[skip] {p} already patched")
        return
    old = (
        '    elif backend_name == "FLASHINFER_STAR_ATTENTION" and IS_FLASHINFER_AVAILABLE:\n'
        '        from .star_flashinfer import StarAttention\n'
        '        return StarAttention'
    )
    new = old + (
        '\n    elif backend_name == "TOKENSPEED_MLA":'
        '\n        # SPIKE PATCH: TokenSpeed MLA opt-in.'
        '\n        from .tokenspeed_mla import is_tokenspeed_mla_available'
        '\n        if not is_tokenspeed_mla_available():'
        '\n            from tensorrt_llm.logger import logger'
        '\n            logger.warning('
        '\n                "attn_backend=TOKENSPEED_MLA requested but tokenspeed-mla is "'
        '\n                "unavailable; falling back to TRTLLM.")'
        '\n        return TrtllmAttention'
    )
    if s.count(old) != 1:
        sys.exit(f"[fail] utils.py anchor missing or duplicated")
    p.write_text(s.replace(old, new))
    print(f"[ok]   {p}")


def patch_trtllm_gen_py() -> None:
    p = ATTN / "trtllm_gen.py"
    s = p.read_text()
    if MARKER in s:
        print(f"[skip] {p} already patched")
        return
    # 1. import os
    old1 = "import math\nfrom dataclasses import dataclass"
    new1 = "import math\nimport os  # SPIKE PATCH\nfrom dataclasses import dataclass"
    if s.count(old1) != 1:
        sys.exit("[fail] trtllm_gen.py import anchor missing")
    s = s.replace(old1, new1)

    # 2. env-var swap around FlashInfer MLA call
    old2 = (
        "        mla_out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(\n"
        "            query=query,\n"
        "            kv_cache=kv_cache,\n"
        "            workspace_buffer=params.workspace.view(-1, 4),\n"
        "            qk_nope_head_dim=params.qk_nope_head_dim,\n"
        "            kv_lora_rank=params.kv_lora_rank,\n"
        "            qk_rope_head_dim=params.qk_rope_head_dim,\n"
        "            block_tables=block_tables,\n"
        "            seq_lens=params.sequence_lengths,\n"
        "            max_seq_len=params.max_past_kv_length,\n"
        "            out=out_buf,\n"
        "            bmm1_scale=bmm1_scale,\n"
        "            bmm2_scale=bmm2_scale,\n"
        "            sinks=params.attention_sinks,\n"
        "        )"
    )
    new2 = (
        "        # SPIKE PATCH: TLLM_TOKENSPEED_MLA=1 swaps in tokenspeed-mla decode kernel.\n"
        "        _use_tokenspeed_mla = False\n"
        '        if os.environ.get("TLLM_TOKENSPEED_MLA") == "1":\n'
        "            from .tokenspeed_mla import (\n"
        "                is_tokenspeed_mla_available,\n"
        "                tokenspeed_batch_decode_with_kv_cache_mla,\n"
        "            )\n"
        "            _use_tokenspeed_mla = is_tokenspeed_mla_available()\n"
        "        if _use_tokenspeed_mla:\n"
        "            mla_out = tokenspeed_batch_decode_with_kv_cache_mla(\n"
        "                query=query, kv_cache=kv_cache,\n"
        "                workspace_buffer=params.workspace.view(-1, 4),\n"
        "                qk_nope_head_dim=params.qk_nope_head_dim,\n"
        "                kv_lora_rank=params.kv_lora_rank,\n"
        "                qk_rope_head_dim=params.qk_rope_head_dim,\n"
        "                block_tables=block_tables, seq_lens=params.sequence_lengths,\n"
        "                max_seq_len=params.max_past_kv_length, out=out_buf,\n"
        "                bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale,\n"
        "                sinks=params.attention_sinks,\n"
        "            )\n"
        "        else:\n"
        "            mla_out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(\n"
        "                query=query, kv_cache=kv_cache,\n"
        "                workspace_buffer=params.workspace.view(-1, 4),\n"
        "                qk_nope_head_dim=params.qk_nope_head_dim,\n"
        "                kv_lora_rank=params.kv_lora_rank,\n"
        "                qk_rope_head_dim=params.qk_rope_head_dim,\n"
        "                block_tables=block_tables, seq_lens=params.sequence_lengths,\n"
        "                max_seq_len=params.max_past_kv_length, out=out_buf,\n"
        "                bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale,\n"
        "                sinks=params.attention_sinks,\n"
        "            )"
    )
    if s.count(old2) != 1:
        sys.exit("[fail] trtllm_gen.py MLA call anchor missing")
    p.write_text(s.replace(old2, new2))
    print(f"[ok]   {p}")


def patch_tokenspeed_mla_decode() -> None:
    p = TS_MLA / "mla_decode.py"
    s = p.read_text()
    if MARKER in s:
        print(f"[skip] {p} already patched")
        return
    # Insert lse_fake after o_fake
    old1 = (
        "    # o: [batch_size, seq_len_q, num_heads, latent_dim] — contiguous\n"
        "    o_fake = cute.runtime.make_fake_compact_tensor(\n"
        "        cutlass_out_dtype,\n"
        "        (sym_batch, sym_seq_q, sym_heads, sym_latent),\n"
        "        stride_order=(3, 2, 1, 0),\n"
        "        assumed_align=16,\n"
        "    )"
    )
    new1 = old1 + (
        "\n    # SPIKE PATCH: tokenspeed-mla 0.1.2 BF16/FP16 kernel reinterprets lse\n"
        "    # unconditionally; provide real fake + runtime tensor.\n"
        "    lse_fake = cute.runtime.make_fake_compact_tensor(\n"
        "        cutlass.Float32,\n"
        "        (sym_batch, sym_seq_q, sym_heads),\n"
        "        stride_order=(2, 1, 0),\n"
        "        assumed_align=4,\n"
        "    )"
    )
    if s.count(old1) != 1:
        sys.exit("[fail] mla_decode.py o_fake anchor missing")
    s = s.replace(old1, new1)

    # Compile-time None -> lse_fake
    old2 = "        o_fake,\n        None,  # lse (disabled)\n        workspace_fake,"
    new2 = "        o_fake,\n        lse_fake,  # SPIKE PATCH\n        workspace_fake,"
    if s.count(old2) != 1:
        sys.exit("[fail] mla_decode.py compile-time lse anchor missing")
    s = s.replace(old2, new2)

    # Runtime None -> real tensor
    old3 = "            o_k,\n            None,  # lse (disabled)\n            workspace_bytes,"
    new3 = (
        "            o_k,\n"
        "            torch.empty((B, q_len, H), dtype=torch.float32, device=query.device),  # SPIKE PATCH\n"
        "            workspace_bytes,"
    )
    if s.count(old3) != 1:
        sys.exit("[fail] mla_decode.py runtime lse anchor missing")
    s = s.replace(old3, new3)

    p.write_text(s)
    print(f"[ok]   {p}")


if __name__ == "__main__":
    patch_utils_py()
    patch_trtllm_gen_py()
    patch_tokenspeed_mla_decode()
    print("all patches applied")
