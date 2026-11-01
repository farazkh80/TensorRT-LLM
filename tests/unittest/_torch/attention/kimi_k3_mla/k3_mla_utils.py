# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the Kimi K3 MLA CuTe DSL kernel tests (one DEP rank: 96 heads, BF16 latent cache)."""

import math

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

H, NOPE, ROPE, LAT, DV, QLORA, PAGE = 96, 128, 64, 512, 128, 1536, 64
D = NOPE + ROPE
ROW = LAT + ROPE

requires_cute_mla = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103, 107),
    reason="Kimi K3 MLA CuTe DSL kernels run on SM100/SM103/SM107 only",
)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def cu_list(lens) -> list[int]:
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    return cu


def reference(q, k, v, scale, cu=None, chunk=8):
    """fp32 causal attention, q/k [T, H, D], v [T, H, DV], ``cu`` cumulative lengths (None = one sequence)."""
    tf32 = torch.backends.cuda.matmul.allow_tf32
    # TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 would otherwise turn the fp32 reference into TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        total = q.shape[0]
        cu = cu or [0, total]
        out = torch.empty(total, H, DV, dtype=torch.float32, device=q.device)
        for s0, s1 in zip(cu[:-1], cu[1:]):
            n = s1 - s0
            mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
            for h0 in range(0, H, chunk):
                qh = q[s0:s1, h0 : h0 + chunk].float().transpose(0, 1)
                kh = k[s0:s1, h0 : h0 + chunk].float().transpose(0, 1)
                vh = v[s0:s1, h0 : h0 + chunk].float().transpose(0, 1)
                s = torch.matmul(qh, kh.transpose(1, 2)) * scale
                s.masked_fill_(mask, float("-inf"))
                out[s0:s1, h0 : h0 + chunk] = torch.matmul(torch.softmax(s, -1), vh).transpose(0, 1)
                del s
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def err(o, ref):
    """(rel-L2, max-abs, all finite, any non-zero) of ``o`` against ``ref``."""
    o = o.float().reshape(ref.shape)
    d = o - ref
    return (
        float(d.norm() / ref.norm()),
        float(d.abs().max()),
        bool(torch.isfinite(o).all()),
        bool(o.abs().sum() > 0),
    )


def build_context_layer(lens, layers, layer_idx):
    """A K3-shaped TrtllmAttention MLA layer (identity rope table) on a V1 KVCacheManager with ``layers`` layers
    interleaved in one pool, and the context metadata for ``lens``. Returns (layer, kv_cache_manager, metadata)."""
    from tensorrt_llm._torch.attention.backends.interface import MLAParams
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import (
        _install_identity_rope_table,
        _make_pos_embd_params,
    )
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.sampling_params import SamplingParams

    cls = get_attention_backend("TRTLLM")
    mla_params = MLAParams(
        q_lora_rank=QLORA,
        kv_lora_rank=LAT,
        qk_rope_head_dim=ROPE,
        qk_nope_head_dim=NOPE,
        v_head_dim=DV,
        predicted_tokens_per_seq=1,
    )
    pos = _make_pos_embd_params(qk_rope_head_dim=ROPE, max_position_embeddings=max(lens) + 128)
    layer = cls(
        layer_idx=layer_idx,
        num_heads=H,
        head_dim=D,
        num_kv_heads=H,
        q_scaling=1.0,
        pos_embd_params=pos,
        mla_params=mla_params,
    )
    _install_identity_rope_table(layer)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    blocks = sum(math.ceil(n / PAGE) for n in lens) + len(lens) + 2
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=blocks * PAGE, enable_block_reuse=False),
        CacheType.SELFKONLY,
        num_layers=layers,
        num_kv_heads=1,
        head_dim=ROW,
        tokens_per_block=PAGE,
        max_seq_len=max(lens) + PAGE,
        max_batch_size=len(lens),
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.bfloat16)),
    )
    for i, n in enumerate(lens):
        req = LlmRequest(
            request_id=i,
            max_new_tokens=1,
            input_tokens=[1] * n,
            sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        req.paged_kv_block_ids = []
        kv_cache_manager.impl.add_sequence_batch([(i, n, 1)], [req])
    metadata = cls.Metadata(
        seq_lens=torch.tensor(lens, dtype=torch.int),
        request_ids=list(range(len(lens))),
        max_num_requests=len(lens),
        num_contexts=len(lens),
        prompt_lens=list(lens),
        max_num_tokens=sum(lens),
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0] * len(lens)),
        mapping=mapping,
    )
    metadata.prepare()
    return layer, kv_cache_manager, metadata


class LatentAppendCase:
    """Decode-step inputs for the latent append on a pool imitating KVCacheManager V1 interleaving.

    Storage is [blocks, layers, 1, page, 1, 576]; the layer's ``get_buffers()`` view is ``pool[:, layer]``, block
    offsets are block_id * layers and the host pool pointer is the pool base.
    """

    def __init__(self, batch, kv_lens, layers=2, layer=1, seq_len_q=1, seed=0):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.manual_seed(seed)
        bf = torch.bfloat16
        self.B, self.L, self.layer, self.sq = batch, layers, layer, seq_len_q
        self.T = batch * seq_len_q
        self.kv_lens_list = list(kv_lens)
        self.maxb = math.ceil(max(self.kv_lens_list) / PAGE) + 1
        self.nblocks = batch * self.maxb + 3
        self.pool0 = torch.randn(self.nblocks, layers, 1, PAGE, 1, ROW, dtype=bf, device="cuda")
        perm = torch.randperm(self.nblocks, generator=generator)[: batch * self.maxb]
        offsets = torch.zeros(1, batch, 2, self.maxb, dtype=torch.int32)
        offsets[0, :, 0] = offsets[0, :, 1] = (perm.view(batch, self.maxb) * layers).int()
        self.block_offsets = offsets.cuda()
        self.mapping = torch.tensor([[0, i] for i in range(layers)], dtype=torch.int32)
        self.kv_lens = torch.tensor(self.kv_lens_list, dtype=torch.int32, device="cuda")
        self.kv_lens_cpu = self.kv_lens.cpu()
        self.ctx_lens_cpu = torch.ones(batch, dtype=torch.int32)
        q = torch.randn(self.T, H * D, dtype=bf, device="cuda")
        self.q_pe = q.view(self.T, H, D)[..., NOPE:]
        ag = torch.randn(self.T, QLORA + ROW, dtype=bf, device="cuda")
        self.k_pe = ag[:, QLORA + LAT :]
        self.ckv = torch.randn(self.T, LAT, dtype=bf, device="cuda")
        self.fused_q0 = torch.randn(self.T, H, ROW, dtype=bf, device="cuda")
        # Identity rope table (cos 1, sin 0), as Kimi K3 installs it.
        self.cos_sin = torch.empty(max(self.kv_lens_list) + 1024, 2 * ROPE, device="cuda")
        self.cos_sin.view(-1)[0::2] = 1.0
        self.cos_sin.view(-1)[1::2] = 0.0
        self._keep = (q, ag)

    def outputs(self) -> dict:
        return dict(
            pool=self.pool0.clone(),
            fused_q=self.fused_q0.clone(),
            cu_q=torch.full((self.B + 1,), -7, dtype=torch.int32, device="cuda"),
            cu_kv=torch.full((self.B + 1,), -7, dtype=torch.int32, device="cuda"),
            counter=torch.full((1,), 12345, dtype=torch.int32, device="cuda"),
        )

    def page_rows(self) -> torch.Tensor:
        return self.block_offsets[0, : self.B, 0, :]

    def stock(self, o: dict) -> torch.Tensor:
        """concat + mla_rope_generation + page-table add, as TrtllmAttention runs them. Returns the page table."""
        latent_cache = torch.concat([self.ckv, self.k_pe], dim=-1)
        pool_pointers = torch.tensor([[o["pool"].data_ptr(), 0]], dtype=torch.int64)
        # Required arguments only; the trailing optional ones keep their defaults.
        torch.ops.trtllm.mla_rope_generation(
            o["fused_q"], self.q_pe, latent_cache, self.cos_sin, o["cu_q"], o["cu_kv"], o["counter"],
            None, None, None,
            self.kv_lens, self.kv_lens_cpu, self.ctx_lens_cpu, 0,
            self.block_offsets, pool_pointers, self.mapping,
            None, None, None, None, None, [None, None],
            self.sq, self.layer, H, 1, ROW, 0, PAGE, int(self.cos_sin.shape[0]), 1, 0, 1.0,
            QLORA, LAT, NOPE, ROPE, NOPE, True,
        )  # fmt: skip
        page_table = self.page_rows().transpose(0, 1).to(torch.int32)
        return page_table + self.layer if self.L > 1 else page_table

    def cute(self, o: dict, with_meta: bool = True) -> torch.Tensor:
        from tensorrt_llm._torch.attention.kernels.rubin.kimi_k3_mla.mla_latent_append import (
            mla_latent_append,
        )

        meta = {}
        if with_meta:
            meta = dict(
                cu_q_seqlens=o["cu_q"],
                cu_kv_seqlens=o["cu_kv"],
                fmha_scheduler_counter=o["counter"],
            )
        return mla_latent_append(
            self.q_pe,
            self.ckv,
            self.k_pe,
            o["fused_q"],
            o["pool"][:, self.layer],
            self.page_rows(),
            self.kv_lens,
            **meta,
        )
