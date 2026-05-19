# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TokenSpeed MLA attention backend.

A subclass of :class:`TrtllmAttention` that overrides ``_run`` for the
MLA-decode-generation-only branch. The override routes the call into the
TokenSpeed CuTe DSL MLA decode kernel via
:func:`tokenspeed_batch_decode_with_kv_cache_mla`. Everything else
(context, mixed batch, non-MLA, sparse attention, attention sinks, helix,
SAGE) falls back to ``super()._run(...)`` and behaves identically to
``TrtllmAttention``.

Design references:
  - `/.claude_docs/tokenspeed-kimik25/understanding.md` §5, §8c.
  - The shape and scale prep mirrors
    ``FlashInferTrtllmGenAttention.run_mla_generation`` in
    ``trtllm_gen.py`` so behavior is auditable line-by-line against the
    canonical FlashInfer wrapper call site.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .interface import AttentionInputType
from .tokenspeed_mla import (
    is_tokenspeed_mla_available,
    tokenspeed_batch_decode_with_kv_cache_mla,
)
from .trtllm import TrtllmAttention, TrtllmAttentionMetadata
from .trtllm_gen import DEFAULT_KV_LAYOUT  # "HND" — match run_mla_generation

# Per-call dispatch gate. The kernel itself templates on the Q dtype; BF16
# and FP16 are the two paths that have been built and validated against
# FlashInfer in the DSV3-Lite spike (see
# `.claude_docs/tokenspeed-mla-dsv3-lite/`).
_TOKENSPEED_SUPPORTED_Q_DTYPES = {torch.bfloat16, torch.float16}


def _tokenspeed_can_dispatch(
    *,
    is_mla_enable: bool,
    q_dtype: torch.dtype,
    kv_cache_dtype: Optional[torch.dtype],
    attention_input_type: AttentionInputType,
    attention_sinks: Optional[torch.Tensor],
    helix_active: bool,
    use_sage_attn: bool,
    sparse_attn_indices: Optional[torch.Tensor],
    num_generations: int,
) -> Tuple[bool, str]:
    """Return (ok, reason). Cheap check — must be safe to call per layer."""
    if not is_mla_enable:
        return False, "non-MLA call falls through to TrtllmAttention"
    if attention_input_type != AttentionInputType.generation_only:
        return False, (
            f"only generation_only is wired up (got {attention_input_type})"
        )
    if num_generations <= 0:
        return False, "no generation requests in this batch"
    if q_dtype not in _TOKENSPEED_SUPPORTED_Q_DTYPES:
        return False, f"Q dtype {q_dtype} not in {_TOKENSPEED_SUPPORTED_Q_DTYPES}"
    # TokenSpeed kernel asserts kv_cache.dtype == query.dtype at runtime,
    # so do that check up front. FP8 KV (the default for K2.6 NVFP4) is
    # *not* compatible with BF16 Q; the upstream path is the patched
    # BF16-KV snapshot or a fix in TokenSpeed itself.
    if kv_cache_dtype is not None and kv_cache_dtype != q_dtype:
        return False, (
            f"TokenSpeed requires KV dtype == Q dtype, got "
            f"KV={kv_cache_dtype} vs Q={q_dtype}")
    if attention_sinks is not None:
        return False, "attention sinks not supported by TokenSpeed kernel"
    if helix_active:
        return False, "helix attention not supported by TokenSpeed kernel"
    if use_sage_attn:
        return False, "SAGE attention not supported by TokenSpeed kernel"
    if sparse_attn_indices is not None and sparse_attn_indices.numel() > 0:
        return False, "sparse attention not supported by TokenSpeed kernel"
    if not is_tokenspeed_mla_available():
        return False, (
            "tokenspeed_mla package not importable or not running on "
            "data-center Blackwell (sm_10x)"
        )
    return True, ""


def _build_mla_block_tables(
    *,
    kv_cache_block_offsets: torch.Tensor,
    host_kv_cache_pool_mapping: torch.Tensor,
    layer_idx: int,
    kv_cache_manager,
    global_layer_idx: int,
    batch_start: int,
    batch_size: int,
) -> torch.Tensor:
    """Replicates ``FlashInferTrtllmGenAttention._build_block_tables``.

    See ``trtllm_gen.py::_build_block_tables`` for the canonical formula
    and a long-form explanation. We avoid importing the
    FlashInferTrtllmGenAttention class only to call this single method.
    """
    pool_idx = int(host_kv_cache_pool_mapping[layer_idx, 0])
    k_offsets = kv_cache_block_offsets[
        pool_idx, batch_start:batch_start + batch_size, 0, :
    ]
    kv_buf = kv_cache_manager.get_buffers(global_layer_idx,
                                          kv_layout=DEFAULT_KV_LAYOUT)
    single_kv_block_elems = kv_buf.shape[2] * kv_buf.shape[3] * kv_buf.shape[4]
    divisor = kv_buf.stride(0) // single_kv_block_elems
    return k_offsets // divisor


class TokenSpeedMLAAttention(TrtllmAttention):
    """MLA attention backend that uses TokenSpeed for the decode kernel.

    All non-decode paths delegate to ``TrtllmAttention._run`` unchanged.
    Context fills the KV cache exactly as today; only the
    ``attention_input_type == generation_only`` path is rerouted, and
    only when the dispatch gate
    (:func:`_tokenspeed_can_dispatch`) returns OK.
    """

    Metadata = TrtllmAttentionMetadata

    # The Python-side `metadata.workspace` is a 0-sized placeholder; the C++
    # thop path manages its own workspace. TokenSpeed runs from Python and
    # needs a real workspace tensor (sized by the kernel per call —
    # typically B*H*q_len*split_kv*(kv_lora_rank+1)*4 bytes, max ~2 MiB
    # for typical K2.6 shapes). We over-allocate a 32 MiB buffer once per
    # process and reuse — matches the spike's parity-test sizing.
    _TS_WORKSPACE_BYTES = 32 * 1024 * 1024
    _ts_workspace: Optional[torch.Tensor] = None

    @classmethod
    def support_mla(cls) -> bool:
        return True

    @classmethod
    def _get_tokenspeed_workspace(cls, device: torch.device,
                                  dtype: torch.dtype = torch.int8
                                  ) -> torch.Tensor:
        ws = cls._ts_workspace
        if (ws is None or ws.device != device or ws.dtype != dtype
                or ws.numel() < cls._TS_WORKSPACE_BYTES):
            cls._ts_workspace = torch.empty(
                (cls._TS_WORKSPACE_BYTES, ), dtype=dtype, device=device)
        return cls._ts_workspace

    def _run(self, q, k, v, output, output_sf, metadata, forward_args, *args,
             **kwargs) -> None:
        # Cheap gating first — most layers don't hit the MLA decode branch
        # (mixed batches, context-only calls), so this must be O(1).
        helix_active = metadata.helix_position_offsets is not None
        use_sage_attn = (forward_args.sage_attn_num_elts_per_blk_q > 0
                         or forward_args.sage_attn_num_elts_per_blk_k > 0
                         or forward_args.sage_attn_num_elts_per_blk_v > 0)
        sparse_attn_indices = args[3] if len(args) >= 4 else None

        # Look up the torch dtype of the underlying KV cache buffer once.
        # The KVCacheManager's `.dtype` field is a DataType enum (bindings),
        # not a torch.dtype — to compare against `q.dtype` we read the
        # actual tensor dtype.
        kv_cache_dtype = None
        if (self.is_mla_enable and metadata.kv_cache_manager is not None
                and forward_args.attention_input_type
                == AttentionInputType.generation_only):
            try:
                kv_buf_peek = metadata.kv_cache_manager.get_buffers(
                    self.layer_idx, kv_layout=DEFAULT_KV_LAYOUT)
                kv_cache_dtype = (kv_buf_peek.dtype
                                  if kv_buf_peek is not None else None)
            except Exception:
                kv_cache_dtype = None

        ok, _reason = _tokenspeed_can_dispatch(
            is_mla_enable=self.is_mla_enable,
            q_dtype=q.dtype,
            kv_cache_dtype=kv_cache_dtype,
            attention_input_type=forward_args.attention_input_type,
            attention_sinks=forward_args.attention_sinks,
            helix_active=helix_active,
            use_sage_attn=use_sage_attn,
            sparse_attn_indices=sparse_attn_indices,
            num_generations=metadata.num_generations,
        )
        if not ok:
            return super()._run(q, k, v, output, output_sf, metadata,
                                forward_args, *args, **kwargs)

        # --- MLA generation-only path: route to TokenSpeed kernel. -------
        # Layout convention follows trtllm_gen.run_mla_generation
        # (trtllm_gen.py:1368). The MLA module has already written the
        # absorbed Q into `q` with shape:
        #   q: [num_gen_tokens, num_heads * (kv_lora_rank + qk_rope_head_dim)]
        # and pre-allocated `output` with shape:
        #   output: [num_gen_tokens, num_heads * kv_lora_rank].
        num_heads = self.num_heads
        kv_lora_rank = self.kv_lora_rank
        qk_nope_head_dim = self.qk_nope_head_dim
        qk_rope_head_dim = self.qk_rope_head_dim
        d_qk = kv_lora_rank + qk_rope_head_dim

        num_gen_tokens = q.shape[0]
        batch_size = metadata.num_generations
        q_len_per_req = num_gen_tokens // batch_size if batch_size > 0 else 1
        assert batch_size * q_len_per_req == num_gen_tokens, (
            f"num_gen_tokens={num_gen_tokens} not divisible by "
            f"num_generations={batch_size}")

        # Pull the layer-local kv_cache buffer and reshape if needed.
        layer_idx_local = self.get_local_layer_idx(metadata)
        global_layer_idx = self.layer_idx
        kv_cache = metadata.kv_cache_manager.get_buffers(
            global_layer_idx, kv_layout=DEFAULT_KV_LAYOUT)
        if kv_cache.ndim == 5:
            # HND layout: [num_blocks, kv_factor, num_kv_heads, page_size, D].
            # For MLA num_kv_heads == 1; squeezing axis 2 yields the 4D
            # shape FlashInfer's `trtllm_batch_decode_with_kv_cache_mla`
            # expects. The TokenSpeed wrapper further collapses the
            # kv_factor==1 axis internally.
            kv_cache = kv_cache.squeeze(2)

        # Build block_tables from KVBlockArray offsets -> page indices.
        # For generation-only, the gen requests live at the tail of the
        # full batch (after num_contexts), so batch_start = num_contexts.
        block_tables = _build_mla_block_tables(
            kv_cache_block_offsets=metadata.kv_cache_block_offsets,
            host_kv_cache_pool_mapping=metadata.host_kv_cache_pool_mapping,
            layer_idx=layer_idx_local,
            kv_cache_manager=metadata.kv_cache_manager,
            global_layer_idx=global_layer_idx,
            batch_start=metadata.num_contexts,
            batch_size=batch_size,
        )

        # Pad block_tables to a multiple of (128 / tokens_per_block) — this
        # matches trtllm-gen's superblock alignment; the TokenSpeed kernel
        # does not require padding but accepts it. Mirror the canonical
        # call to keep the kv_cache memory walk identical.
        pages_per_superblock = 128 // metadata.tokens_per_block
        if pages_per_superblock > 1:
            num_blocks_in_table = block_tables.size(-1)
            remainder = num_blocks_in_table % pages_per_superblock
            if remainder != 0:
                pad = pages_per_superblock - remainder
                block_tables = torch.nn.functional.pad(
                    block_tables, (0, pad), value=0)

        # Slice per-request sequence lengths to the generation tail.
        seq_lens = metadata.kv_lens_cuda_runtime[
            metadata.num_contexts:metadata.num_contexts + batch_size]

        # max_seq_len: per-batch GPU-side reduction is expensive in this
        # hot path. The canonical run_mla_generation reads it from
        # params.max_past_kv_length, which the metadata computes once on
        # the CPU side. We use kv_lens_runtime (CPU) for the same value.
        gen_kv_lens_cpu = metadata.kv_lens_runtime[
            metadata.num_contexts:metadata.num_contexts + batch_size]
        max_seq_len = int(gen_kv_lens_cpu.max().item()) if batch_size else 0

        # Scales — FlashInfer's bmm1_scale already folds 1/sqrt(D); see
        # the run_mla_generation site.
        bmm1_scale = 1.0 / (
            self.q_scaling * math.sqrt(qk_nope_head_dim + qk_rope_head_dim))
        bmm2_scale = 1.0

        # Reshape query and output to TokenSpeed's expected 4D layout.
        query = q.view(batch_size, q_len_per_req, num_heads, d_qk)
        out_view = output.view(batch_size, q_len_per_req, num_heads,
                               kv_lora_rank)

        # Workspace: `metadata.workspace` is a 0-byte placeholder; the C++
        # thop path resizes it implicitly. We're Python-side; allocate a
        # dedicated 32 MiB int8 buffer and reuse across calls.
        workspace = self._get_tokenspeed_workspace(device=q.device)

        tokenspeed_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            out=out_view,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=None,
        )
