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
