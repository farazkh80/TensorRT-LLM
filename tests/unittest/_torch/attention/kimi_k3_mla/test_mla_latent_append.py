# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA decode latent append (CuTe DSL) vs concat + mla_rope_generation + the page-table add.

Both run on identical copies of the same inputs and are compared bitwise: the whole paged pool (every layer of the
interleaved pool, so stray writes show), fused_q, cu_q_seqlens, cu_kv_seqlens, the scheduler counter and the decode
page table (values and strides).
"""

import pytest
import torch
from k3_mla_utils import LAT, PAGE, LatentAppendCase, bits, requires_cute_mla

import tensorrt_llm  # noqa: F401  registers torch.ops.trtllm

pytestmark = requires_cute_mla

CASES = {
    **{
        f"bs{b}_kv8193_v1pool": dict(batch=b, kv_lens=[8193] * b, layers=2, layer=1)
        for b in (1, 4, 16, 64)
    },
    **{
        f"bs{b}_kv8193_1layer": dict(batch=b, kv_lens=[8193] * b, layers=1, layer=0)
        for b in (1, 4, 16, 64)
    },
    "bs16_mixed_kv": dict(
        batch=16, kv_lens=[1 + (977 * i) % 8193 for i in range(16)], layers=3, layer=2
    ),
    "bs33_mixed_kv_layer0": dict(
        batch=33, kv_lens=[64 * (i + 1) + (i % 3) for i in range(33)], layers=2, layer=0
    ),
    "bs8_q2_mtp": dict(
        batch=8, kv_lens=[8193 + i for i in range(8)], layers=2, layer=1, seq_len_q=2
    ),
}


def _check_stock_harness(case: LatentAppendCase, stock: dict) -> None:
    """The stock path changed exactly the T expected rows, each to [compressed_kv | k_pe]."""
    changed = (bits(stock["pool"]) != bits(case.pool0)).any(-1).view(case.nblocks, case.L, PAGE)
    expected = torch.zeros_like(changed)
    rows = torch.cat([case.ckv, case.k_pe], -1)
    for t in range(case.T):
        b, u = divmod(t, case.sq)
        pos = case.kv_lens_list[b] - case.sq + u
        block = int(case.block_offsets[0, b, 0, pos // PAGE]) // case.L
        expected[block, case.layer, pos % PAGE] = True
        assert torch.equal(bits(stock["pool"][block, case.layer, 0, pos % PAGE, 0]), bits(rows[t]))
    assert torch.equal(changed, expected)
    assert torch.equal(bits(stock["fused_q"][..., LAT:]), bits(case.q_pe))
    assert torch.equal(bits(stock["fused_q"][..., :LAT]), bits(case.fused_q0[..., :LAT]))


@pytest.mark.parametrize("name", list(CASES))
def test_latent_append_matches_stock(name: str) -> None:
    case = LatentAppendCase(**CASES[name])
    stock, cute = case.outputs(), case.outputs()
    page_table_stock = case.stock(stock)
    page_table_cute = case.cute(cute)
    torch.cuda.synchronize()
    _check_stock_harness(case, stock)
    assert torch.equal(bits(stock["pool"]), bits(cute["pool"]))
    assert torch.equal(bits(stock["fused_q"]), bits(cute["fused_q"]))
    for key in ("cu_q", "cu_kv", "counter"):
        assert torch.equal(stock[key], cute[key]), key
    assert torch.equal(page_table_stock, page_table_cute)
    if case.L > 1:
        assert page_table_stock.stride() == page_table_cute.stride()


def test_latent_append_without_scheduler_metadata() -> None:
    case = LatentAppendCase(batch=16, kv_lens=[8193] * 16, layers=2, layer=1)
    stock, cute = case.outputs(), case.outputs()
    case.stock(stock)
    case.cute(cute, with_meta=False)
    torch.cuda.synchronize()
    assert torch.equal(bits(stock["pool"]), bits(cute["pool"]))
    assert torch.equal(bits(stock["fused_q"]), bits(cute["fused_q"]))
    assert bool((cute["cu_q"] == -7).all()) and bool((cute["cu_kv"] == -7).all())
    assert int(cute["counter"]) == 12345
