# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi K3 MLA context phase in CuTe DSL: latent append + k_pe broadcast, then the causal varlen FMHA.

Replaces what ``thop.attention`` does for an MLA context-only call on a BF16 cache with the identity rope table:
``applyMLARopeAndAssignQKVKernelOptContext`` (cache append, K rope columns; the Q rewrite is a no-op under the
identity table), ``computeSeqAndPaddingOffsets`` (its cu_seqlens / padding offsets / tile counter live in the op's
workspace and are not read after the call) and the trtllm-gen context FMHA. Calls this library cannot serve (no
latent cache, cached-prefix / chunked prefill, softmax stats) run ``FallbackFmha`` unchanged.

The latent row arrives either as ``forward_args.latent_cache`` or, when ``MLA`` skipped the concat, as its two halves
in ``metadata._k3_ctx_latent = (layer_idx, compressed_kv, k_pe)``. A layer with those halves is always claimed, so
the concat is materialized here before any fallback to ``thop.attention`` (which needs the contiguous row).
"""

import math
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.bindings import DataType

from .fallback import FallbackFmha
from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class CuteDslMlaContextFmha(PhasedFmha):
    """Kimi K3 MLA context attention (qk 192 / v 128, BF16 cache) behind TLLM_K3_CUTE_MLA=ctx."""

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        self._fallback = FallbackFmha(attn)

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        return (
            IS_CUTLASS_DSL_AVAILABLE
            and get_sm_version() in (100, 103, 107)
            and attn.is_mla_enable
            and getattr(attn, "sparse_params", None) is None
            and not getattr(attn, "uses_fp4_mla_attention", False)
            and attn.kv_lora_rank == 512
            and attn.qk_rope_head_dim == 64
            and attn.qk_nope_head_dim == 128
            and attn.v_head_dim == 128
        )

    def _latent_parts(self, metadata: "TrtllmAttentionMetadata"):
        parts = getattr(metadata, "_k3_ctx_latent", None)
        if parts is None or parts[0] != self.attn.layer_idx:
            return None
        return parts[1], parts[2]

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        if phase == FmhaPhase.CONTEXT and self._latent_parts(metadata) is not None:
            return True
        return self._serves(q, metadata, forward_args, phase=phase)

    def _serves(
        self,
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        from tensorrt_llm._torch.attention.mla import k3_cute_mla_enabled

        attn = self.attn
        kv_cache_manager = metadata.kv_cache_manager
        return (
            phase == FmhaPhase.CONTEXT
            and getattr(attn, "_k3_cute_ctx", False)
            and k3_cute_mla_enabled("ctx")
            and forward_args.attention_input_type == AttentionInputType.context_only
            and not getattr(attn, "has_fp8_kv_cache", False)
            and kv_cache_manager is not None
            and kv_cache_manager.dtype == DataType.BF16
            and q.dtype == torch.bfloat16
            and forward_args.output is not None
            and forward_args.output.dtype == torch.bfloat16
            and forward_args.output_sf is None
            and forward_args.attention_mask == PredefinedAttentionMask.CAUSAL
            and forward_args.attention_mask_data is None
            and forward_args.attention_sinks is None
            and metadata.beam_width == 1
            and getattr(metadata, "helix_position_offsets", None) is None
            and not metadata.is_cross
        )

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        parts = self._latent_parts(metadata)
        try:
            if (
                k is None
                or v is None
                or (forward_args.latent_cache is None and parts is None)
                or forward_args.softmax_stats_tensor is not None
                or getattr(metadata, "num_ctx_cached_tokens", 0) != 0
                or not self._serves(q, metadata, forward_args, phase=FmhaPhase.CONTEXT)
            ):
                if forward_args.latent_cache is None and parts is not None:
                    forward_args.latent_cache = torch.cat(parts, dim=-1)
                self._fallback.forward(q, k, self._fallback_v(v), metadata, forward_args)
                return
            super().forward(q, k, v, metadata, forward_args)
        finally:
            if parts is not None:
                metadata._k3_ctx_latent = None

    def _fallback_v(self, v: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Restore the kv-shaped V layout the thop MLA context path expects (row stride H * (nope + v))."""
        attn = self.attn
        if v is None:
            return v
        nope_cols = attn.num_heads * attn.qk_nope_head_dim
        v_cols = attn.num_heads * attn.v_head_dim
        if v.stride(0) == nope_cols + v_cols:
            return v
        kv_buf = torch.empty(v.shape[0], nope_cols + v_cols, dtype=v.dtype, device=v.device)
        kv_buf[:, nope_cols:].copy_(v.reshape(v.shape[0], v_cols))
        return kv_buf[:, nope_cols:]

    @staticmethod
    def _ctx_cu_seqlens(
        meta: "TrtllmAttentionMetadata", batch: int, device: torch.device
    ) -> torch.Tensor:
        """Context cu_seqlens as an H2D copy of the host prefix sums (no device kernel), reused while the lengths
        are unchanged. Prefill runs eager, so the host lengths are this step's."""
        lens = meta.seq_lens[:batch].tolist()
        cached = getattr(meta, "_k3_ctx_cu_seqlens", None)
        if cached is None or cached[0] != lens or cached[1].device != device:
            cu = torch.zeros(batch + 1, dtype=torch.int32, pin_memory=prefer_pinned())
            cu[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
            cached = (lens, cu.to(device, non_blocking=True))
            meta._k3_ctx_cu_seqlens = cached
        return cached[1]

    def run_mla_context(self, params: FmhaParams) -> None:
        from tensorrt_llm._torch.attention.kernels.rubin.kimi_k3_mla.mla_context_append import (
            mla_context_append,
        )
        from tensorrt_llm._torch.attention.kernels.rubin.kimi_k3_mla.mla_context_fmha import (
            mla_context_fmha,
        )

        attn = params.attn
        meta = params.meta
        num_tokens = params.num_tokens
        batch = params.batch_size
        num_heads = attn.num_heads
        nope, rope, dv = attn.qk_nope_head_dim, attn.qk_rope_head_dim, attn.v_head_dim
        latent_dim = attn.kv_lora_rank

        q = params.query_input.view(num_tokens, num_heads, nope + rope)
        k = params.key_input.view(num_tokens, num_heads, nope + rope)
        v = params.value_input
        if not v.is_contiguous():
            v = v.contiguous()
        v = v.view(num_tokens, num_heads, dv)
        out = params.output.view(num_tokens, num_heads, dv)
        out_kernel = (
            out if out.is_contiguous() and out.data_ptr() % 32 == 0 else torch.empty_like(out)
        )

        parts = self._latent_parts(meta)
        if parts is not None:
            compressed_kv, k_pe = parts[0][:num_tokens], parts[1][:num_tokens]
        else:
            latent = params.fwd.latent_cache[:num_tokens]
            compressed_kv, k_pe = latent[:, :latent_dim], latent[:, latent_dim:]
        local_layer_idx = attn.get_local_layer_idx(meta)
        pool_idx = int(meta.host_kv_cache_pool_mapping[local_layer_idx, 0])
        page_rows = meta.kv_cache_block_offsets[pool_idx, :batch, 0, :]
        kv_lens = params.sequence_lengths[:batch].to(torch.int32)
        cu_seqlens = self._ctx_cu_seqlens(meta, batch, q.device)
        max_seqlen = params.input_seq_length
        softmax_scale = 1.0 / (math.sqrt(nope + rope) * attn.q_scaling)

        mla_context_append(
            compressed_kv,
            k_pe,
            k,
            meta.kv_cache_manager.get_buffers(attn.layer_idx),
            page_rows,
            kv_lens,
            cu_seqlens,
            max_seqlen,
        )
        mla_context_fmha(q, k, v, out_kernel, cu_seqlens, max_seqlen, softmax_scale)
        if out_kernel is not out:
            out.copy_(out_kernel)
