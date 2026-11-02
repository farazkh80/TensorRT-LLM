# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA context path (CuTe DSL append + causal varlen FMHA) vs the TrtllmAttention context forward.

Stock: TrtllmAttention.forward(context_only, latent_cache) -> applyMLARopeAndAssignQKVKernelOptContext + the
trtllm-gen context FMHA, V strided as in K3 (row stride H * 256). CuTe: mla_context_append + mla_context_fmha with
a contiguous V. Both run on identical inputs and an identical, random starting cache.

The high-contrast cases force the online-softmax rescale: Q/K nope are U(-2, 2) and every key carries a k_pe ramp
along the sequence that every query sees through constant q rope columns, so the row max grows on almost every KV
block and attention is peaked.
"""

import math

import pytest
import torch
from k3_mla_utils import (
    DV,
    LAT,
    NOPE,
    ROPE,
    D,
    H,
    bits,
    build_context_layer,
    cu_list,
    err,
    reference,
    requires_cute_mla,
)

from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm._torch.attention.kernels.rubin.kimi_k3_mla.mla_context_append import (
    mla_context_append,
)
from tensorrt_llm._torch.attention.kernels.rubin.kimi_k3_mla.mla_context_fmha import (
    mla_context_fmha,
)

pytestmark = requires_cute_mla

SCALE = 1.0 / math.sqrt(D)


def _uniform_inputs(total: int):
    bf = torch.bfloat16
    q = torch.empty(total, H, D, dtype=bf, device="cuda").uniform_(-1, 1)
    k = torch.empty(total, H, D, dtype=bf, device="cuda").uniform_(-1, 1)
    ckv = torch.empty(total, LAT, dtype=bf, device="cuda").uniform_(-1, 1)
    k_pe = torch.empty(total, ROPE, dtype=bf, device="cuda").uniform_(-1, 1)
    return q, k, ckv, k_pe


def _high_contrast_inputs(lens, ramp: float):
    bf = torch.bfloat16
    total = sum(lens)
    q = torch.empty(total, H, D, device="cuda").uniform_(-2, 2)
    q[..., NOPE:] = 0.5
    k = torch.empty(total, H, D, device="cuda").uniform_(-2, 2)
    pos = torch.cat(
        [torch.arange(n, device="cuda", dtype=torch.float32) / max(n - 1, 1) for n in lens]
    )
    k_pe = (ramp * pos).unsqueeze(1).expand(total, ROPE).contiguous()
    k[..., NOPE:] = k_pe.unsqueeze(1)
    ckv = torch.empty(total, LAT, device="cuda").uniform_(-1, 1).to(bf)
    return q.to(bf), k.to(bf), ckv, k_pe.to(bf)


def _strided_v(total: int) -> torch.Tensor:
    kv_buf = torch.empty(total, H * (NOPE + DV), dtype=torch.bfloat16, device="cuda").uniform_(
        -1, 1
    )
    return kv_buf[:, H * NOPE :]


def _run_stock_and_cute(lens, layers, layer_idx, q, k, ckv, k_pe):
    """Returns (stock output, stock K, stock caches, CuTe output, CuTe K, CuTe caches, starting caches, v)."""
    total = sum(lens)
    v_strided = _strided_v(total)
    latent = torch.cat([ckv, k_pe], -1)
    layer, kv_cache_manager, metadata = build_context_layer(lens, layers, layer_idx)
    try:
        pools = [kv_cache_manager.get_buffers(i) for i in range(layers)]
        for pool in pools:
            pool.copy_(torch.randn(pool.shape, device="cuda").to(torch.bfloat16))
        start = [pool.clone() for pool in pools]

        k_stock = k.clone()
        out_stock = layer.forward(
            q.reshape(total, H * D).clone(),
            k_stock.view(total, H * D),
            v_strided,
            metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent,
        )
        torch.cuda.synchronize()
        cache_stock = [pool.clone() for pool in pools]
        for pool, saved in zip(pools, start):
            pool.copy_(saved)

        pool_idx = int(metadata.host_kv_cache_pool_mapping[layer.get_local_layer_idx(metadata), 0])
        rows = metadata.kv_cache_block_offsets[pool_idx, : len(lens), 0, :]
        cu = torch.tensor(cu_list(lens), dtype=torch.int32, device="cuda")
        meta_cu = metadata.mla_prepare_ctx_cu_seqlens()
        assert meta_cu is not None and torch.equal(meta_cu.to(torch.int32), cu)
        kv_lens = metadata.kv_lens_cuda_runtime[: len(lens)].to(torch.int32)
        k_cute = k.clone()
        mla_context_append(
            ckv, k_pe, k_cute, kv_cache_manager.get_buffers(layer_idx), rows, kv_lens, cu, max(lens)
        )
        v = v_strided.contiguous().view(total, H, DV)
        out_cute = torch.empty(total, H, DV, dtype=torch.bfloat16, device="cuda")
        mla_context_fmha(q, k_cute, v, out_cute, cu, max(lens), SCALE)
        torch.cuda.synchronize()
        cache_cute = [pool.clone() for pool in pools]
    finally:
        kv_cache_manager.shutdown()
    return out_stock, k_stock.view(total, H, D), cache_stock, out_cute, k_cute, cache_cute, start, v


@pytest.mark.parametrize(
    "lens,layers,layer_idx",
    [
        ([1], 2, 1),
        ([100, 1000, 3000, 64, 1, 4097], 2, 1),
        ([63, 65, 128], 1, 0),
        ([513, 2, 777], 3, 0),
        ([8192], 2, 1),
    ],
    ids=["one_token", "ragged6_v1pool", "page_edges_1layer", "ragged3_layer0", "isl8192_v1pool"],
)
def test_context_append_and_fmha_match_stock(lens, layers, layer_idx) -> None:
    torch.manual_seed(0)
    total = sum(lens)
    q, k, ckv, k_pe = _uniform_inputs(total)
    out_s, k_s, cache_s, out_c, k_c, cache_c, start, v = _run_stock_and_cute(
        lens, layers, layer_idx, q, k, ckv, k_pe
    )
    for i in range(layers):
        assert torch.equal(bits(cache_s[i]), bits(cache_c[i])), f"cache layer {i}"
    assert int((cache_s[layer_idx] != start[layer_idx]).any(-1).sum()) == total
    assert torch.equal(bits(k_s[..., NOPE:]), bits(k_c[..., NOPE:]))
    assert torch.equal(bits(k_c[..., :NOPE]), bits(k[..., :NOPE]))
    assert torch.equal(k_c[..., NOPE:], k_pe.unsqueeze(1).expand(total, H, ROPE))

    ref = reference(q, k_c, v, SCALE, cu_list(lens))
    e_stock, e_cute = err(out_s, ref), err(out_c, ref)
    assert e_cute[2] and e_cute[3]
    assert e_cute[0] <= 1.05 * e_stock[0], f"CuTe rel-L2 {e_cute[0]:.2e} vs stock {e_stock[0]:.2e}"


@pytest.mark.parametrize("ramp", [4.0, 16.0])
def test_context_fmha_high_contrast(ramp: float) -> None:
    torch.manual_seed(0)
    seq = 8192
    q, k, ckv, k_pe = _high_contrast_inputs([seq], ramp)
    out_s, _, _, out_c, k_c, _, _, v = _run_stock_and_cute([seq], 1, 0, q, k, ckv, k_pe)
    ref = reference(q, k_c, v, SCALE)
    e_stock, e_cute = err(out_s, ref), err(out_c, ref)
    assert e_cute[2] and e_cute[3]
    assert e_cute[0] <= 1.05 * e_stock[0], f"rel-L2 CuTe {e_cute[0]:.2e} stock {e_stock[0]:.2e}"
    assert e_cute[1] <= 1.5 * e_stock[1] + 1e-6, (
        f"max-abs CuTe {e_cute[1]:.2e} stock {e_stock[1]:.2e}"
    )

    lens = [100, 1000, 3000, 64, 1, 4097]
    torch.manual_seed(1)
    q, k, _, _ = _high_contrast_inputs(lens, ramp)
    total = sum(lens)
    v = _strided_v(total).contiguous().view(total, H, DV)
    out = torch.empty(total, H, DV, dtype=torch.bfloat16, device="cuda")
    cu = cu_list(lens)
    mla_context_fmha(
        q, k, v, out, torch.tensor(cu, dtype=torch.int32, device="cuda"), max(lens), SCALE
    )
    e_ragged = err(out, reference(q, k, v, SCALE, cu))
    assert e_ragged[2] and e_ragged[3]
    assert e_ragged[0] <= 1.5 * e_stock[0], (
        f"ragged rel-L2 {e_ragged[0]:.2e} vs stock {e_stock[0]:.2e}"
    )
