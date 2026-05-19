# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This is the host copy of the spike's TokenSpeed MLA wrapper.
# Reference deployment path inside the container (bind-mount target):
#   /usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/attention_backend/tokenspeed_mla.py
#
# Designed to be FlashInfer-signature compatible so it can drop in for
# `flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla` at the call site in
# `tensorrt_llm/_torch/attention_backend/trtllm_gen.py:run_mla_generation`,
# under the TLLM_TOKENSPEED_MLA=1 env-var swap that `apply_patches.py` adds.
#
# Wrapped kernel: `tokenspeed_mla.tokenspeed_mla_decode` (CuTe DSL, Blackwell
# SM10x; details in tokenspeed-mla repo).
#
# Lazy-imports tokenspeed_mla so the host file is safe to ship even on hosts
# that don't have the package installed (selector entry in utils.py still
# falls back to TrtllmAttention with a warning).
"""TokenSpeed MLA wrapper module (drop-in for FlashInfer MLA decode)."""

from __future__ import annotations

import math
from typing import Optional

import torch


_TOKENSPEED_MLA_AVAILABLE: Optional[bool] = None


def is_tokenspeed_mla_available() -> bool:
    """Return True iff the tokenspeed_mla package is importable AND the active
    GPU is data-center Blackwell (sm_100 / sm_103). Cached after first call."""
    global _TOKENSPEED_MLA_AVAILABLE
    if _TOKENSPEED_MLA_AVAILABLE is not None:
        return _TOKENSPEED_MLA_AVAILABLE
    try:
        import tokenspeed_mla  # noqa: F401
    except Exception:
        _TOKENSPEED_MLA_AVAILABLE = False
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        # SM 10.x — Blackwell data-center (B200 sm_100, B300 sm_103).
        # TokenSpeed CuTe DSL kernels only build for this arch family.
        _TOKENSPEED_MLA_AVAILABLE = major == 10
    except Exception:
        _TOKENSPEED_MLA_AVAILABLE = False
    return _TOKENSPEED_MLA_AVAILABLE


def tokenspeed_batch_decode_with_kv_cache_mla(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Optional[float] = None,
    bmm2_scale: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FlashInfer MLA decode signature; calls into TokenSpeed CuTe DSL kernel.

    Parameters mirror `flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla`:
      - ``query``: ``[B, q_len, H, qk_nope_head_dim + qk_rope_head_dim + kv_lora_rank]``
        as packed by trtllm-gen's MLA generation prep (Q absorbs into the
        latent dim, so the kernel sees ``D_qk = kv_lora_rank + qk_rope_head_dim``
        for the inner BMM2-against-KV step).
      - ``kv_cache``: ``[num_pages, page_size, kv_lora_rank + qk_rope_head_dim]``
        or 4D with a singleton head axis.
      - ``workspace_buffer``: any dtype, will be reinterpreted to uint8.
      - ``bmm1_scale``: softmax pre-scale; absorbs ``1/sqrt(qk_nope+qk_rope)``
        already at the call site.
      - ``bmm2_scale``: output post-scale (typically 1.0).
      - ``sinks``: NOT SUPPORTED. Must be None. The kernel has no sinks path.

    Returns the MLA decode output ``[B, q_len, H, kv_lora_rank]``. If ``out``
    is provided, writes into it and returns it (for q_len_per_req == 1, the
    caller passes its pre-allocated buffer).
    """
    if sinks is not None:
        raise NotImplementedError(
            "TokenSpeed MLA wrapper does not support attention sinks; "
            "fall back to FlashInfer or TrtllmAttention.")

    if not is_tokenspeed_mla_available():
        raise RuntimeError(
            "tokenspeed_mla is not available in this environment; check "
            "is_tokenspeed_mla_available() before calling.")

    # Local import so the host file can be present without the package.
    from tokenspeed_mla import tokenspeed_mla_decode

    # Reinterpret workspace as int8 — the TokenSpeed kernel's binding
    # asserts `workspace_buffer.dtype == torch.int8`. The spike call site
    # passes `params.workspace.view(-1, 4)` (an int8 view).
    ws = workspace_buffer
    if ws.dtype != torch.int8:
        ws = ws.view(torch.int8)

    # FlashInfer's "bmm1_scale" already includes 1/sqrt(D); the TokenSpeed
    # kernel expects `softmax_scale` in the same convention.
    softmax_scale = (
        bmm1_scale if bmm1_scale is not None
        else 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim))
    output_scale = bmm2_scale if bmm2_scale is not None else 1.0

    # Pre-allocate output if caller didn't (matches FlashInfer behavior for
    # the q_len_per_req > 1 spec-decode path).
    if out is None:
        batch_size, q_len, num_heads, _ = query.shape
        out = torch.empty(
            (batch_size, q_len, num_heads, kv_lora_rank),
            dtype=query.dtype, device=query.device)

    # Normalize 4D kv_cache layout (with a singleton head axis) → 3D, which is
    # what tokenspeed_mla_decode expects per its docstring. Matches the spike
    # step 6 / step 7 wrapper code.
    if kv_cache.ndim == 4 and kv_cache.shape[1] == 1:
        kv_cache = kv_cache.squeeze(1)

    return tokenspeed_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        out=out,
        is_var_seq=True,
        causal_mask=True,
        enable_pdl=False,
    )
