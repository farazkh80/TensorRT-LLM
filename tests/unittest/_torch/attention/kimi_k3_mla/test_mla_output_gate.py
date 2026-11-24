# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA output gate (CuTe DSL) vs ATen ``o * g.sigmoid()`` on [M, 96 * 128] bf16: bitwise (NaN == NaN)."""

import pytest
import torch
from k3_mla_utils import requires_cute_mla

from tensorrt_llm._torch.attention.kernels.rubin.kimi_k3_mla.mla_output_gate import (
    DECODE_CONFIG,
    PREFILL_CONFIG,
    mla_output_gate,
)

pytestmark = requires_cute_mla

N = 96 * 128
CONFIGS = [None, DECODE_CONFIG, PREFILL_CONFIG, (8, 4, True), (16, 4, True)]
EDGE = [0.0, -0.0, 1.0, -1.0, 8.0, -8.0, 16.0, -16.0, 20.0, -20.0, 40.0, -40.0, 80.0, -80.0, 87.0, -87.0, 88.0,
        -88.0, 89.0, -89.0, 100.0, -100.0, 103.0, -103.0, 104.0, -104.0, 200.0, -200.0, 1e-30, -1e-30, 3e38, -3e38,
        float("inf"), float("-inf"), float("nan")]  # fmt: skip


def _assert_matches_aten(o: torch.Tensor, g: torch.Tensor, config) -> None:
    expected = o * g.sigmoid()
    actual = mla_output_gate(o, g, config=config)
    same = (expected.view(torch.int16) == actual.view(torch.int16)) | (
        torch.isnan(expected) & torch.isnan(actual)
    )
    assert bool(same.all()), f"{int((~same).sum())} elements differ from ATen"


@pytest.mark.parametrize("config", CONFIGS, ids=str)
@pytest.mark.parametrize("m", [1, 3, 4, 16, 64, 8192])
def test_output_gate_random(m: int, config) -> None:
    torch.manual_seed(m)
    o = torch.randn(m, N, dtype=torch.bfloat16, device="cuda")
    g = (torch.randn(m, N, device="cuda") * 4).to(torch.bfloat16)
    _assert_matches_aten(o, g, config)


@pytest.mark.parametrize("config", CONFIGS, ids=str)
def test_output_gate_every_bf16_gate_value(config) -> None:
    torch.manual_seed(0)
    every = (
        torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
    )
    pad = torch.zeros(-every.numel() % N, dtype=torch.bfloat16, device="cuda")
    g = torch.cat([every, pad]).view(-1, N)
    o = torch.randn(g.shape, dtype=torch.bfloat16, device="cuda")
    _assert_matches_aten(o, g, config)


@pytest.mark.parametrize("config", CONFIGS, ids=str)
def test_output_gate_edge_values(config) -> None:
    torch.manual_seed(0)
    edge = torch.tensor(EDGE, dtype=torch.bfloat16, device="cuda")
    g = edge.repeat(N // len(EDGE) + 1)[:N].view(1, N).repeat(2, 1)
    o = torch.randn(2, N, dtype=torch.bfloat16, device="cuda")
    o[1] = float("inf")
    _assert_matches_aten(o, g, config)


@pytest.mark.parametrize("config", CONFIGS, ids=str)
def test_output_gate_in_place(config) -> None:
    torch.manual_seed(0)
    o = torch.randn(16, N, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(16, N, dtype=torch.bfloat16, device="cuda")
    expected = o * g.sigmoid()
    mla_output_gate(o, g, out=o, config=config)
    assert torch.equal(expected.view(torch.int16), o.view(torch.int16))


NEG_ZERO = -32768  # bf16 -0.0 as int16, the fine-grained-sync sentinel


def _run_producer(o: torch.Tensor, g: torch.Tensor, config=None) -> torch.Tensor:
    """fgs_producer into a sentinel-filled slot; checks it equals ATen with -0.0 as +0.0 and holds no -0.0."""
    expected = o * g.sigmoid()
    expected.view(torch.int16)[expected.view(torch.int16) == NEG_ZERO] = 0
    slot = torch.full_like(o, -0.0)
    mla_output_gate(o, g, out=slot, config=config, fgs_producer=True)
    bits = slot.view(torch.int16)
    assert not bool((bits == NEG_ZERO).any()), "fgs_producer stored -0.0"
    same = (bits == expected.view(torch.int16)) | (torch.isnan(expected) & torch.isnan(slot))
    assert bool(same.all()), f"{int((~same).sum())} elements differ from ATen (-0.0 as +0.0)"
    return slot


@pytest.mark.parametrize("config", [None, DECODE_CONFIG, (8, 4, True)], ids=str)
@pytest.mark.parametrize("m", [1, 4, 16, 64, 8192])
def test_output_gate_fgs_producer_random(m: int, config) -> None:
    torch.manual_seed(m)
    o = torch.randn(m, N, dtype=torch.bfloat16, device="cuda")
    g = (torch.randn(m, N, device="cuda") * 4).to(torch.bfloat16)
    _run_producer(o, g, config)


def test_output_gate_fgs_producer_every_bf16_gate_value() -> None:
    torch.manual_seed(0)
    every = (
        torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
    )
    pad = torch.zeros(-every.numel() % N, dtype=torch.bfloat16, device="cuda")
    g = torch.cat([every, pad]).view(-1, N)
    o = torch.randn(g.shape, dtype=torch.bfloat16, device="cuda")
    _run_producer(o, g)


def test_output_gate_fgs_producer_negative_zeros() -> None:
    """Inputs whose product is -0.0: o = -0.0, o = +-0.0 with tiny gates, and negative products that underflow."""
    torch.manual_seed(0)
    edge = torch.tensor(EDGE, dtype=torch.bfloat16, device="cuda")
    g_edge = edge.repeat(N // len(EDGE) + 1)[:N]
    tiny = torch.tensor(
        [-1e-38, -1e-30, -1e-20, -3e-39, -9.2e-41], dtype=torch.bfloat16, device="cuda"
    )
    o = torch.empty(6, N, dtype=torch.bfloat16, device="cuda")
    g = torch.empty_like(o)
    o[0], g[0] = -0.0, g_edge
    o[1], g[1] = 0.0, g_edge
    o[2], g[2] = -1.0, -200.0
    o[3], g[3] = tiny.repeat(N // len(tiny) + 1)[:N], -100.0
    o[4], g[4] = tiny.repeat(N // len(tiny) + 1)[:N], g_edge
    o[5], g[5] = torch.randn(N, device="cuda").to(torch.bfloat16).abs().neg(), -90.0
    expected = o * g.sigmoid()
    assert bool((expected.view(torch.int16) == NEG_ZERO).any()), (
        "the inputs must produce -0.0 in ATen"
    )
    _run_producer(o, g)


def test_output_gate_fgs_producer_rejects_unsafe_calls() -> None:
    o = torch.randn(16, N, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(16, N, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        mla_output_gate(o, g, fgs_producer=True)
    with pytest.raises(ValueError):
        mla_output_gate(o, g, out=o, fgs_producer=True)
    with pytest.raises(ValueError):
        mla_output_gate(o, g, out=torch.empty_like(o), config=(16, 2, True), fgs_producer=True)
