# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL split-KV plan vs the Triton plan kernel of CuteDslMlaFmha and its host mirror (exact integers)."""

import random

import pytest
import torch
from k3_mla_utils import requires_cute_mla

from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import (
    launch_mla_split_kv_plan as triton_plan,
)
from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import plan_mla_split_kv
from tensorrt_llm._torch.attention.kernels.rubin.kimi_k3_mla.mla_split_kv_plan import (
    launch_mla_split_kv_plan as cute_plan,
)

pytestmark = requires_cute_mla


def _cases():
    rnd = random.Random(0)
    cases = {}
    for b in (1, 4, 16, 64):
        for waves in (53, 106):
            cases[f"k3_kv8193_bs{b}_w{waves}"] = ([8193] * b, 1, waves)
    cases["zeros_and_ones"] = ([0, 1, 0, 1, 2], 1, 106)
    cases["page_edges"] = ([64 * i + d for i in range(1, 40) for d in (-1, 0, 1)], 1, 106)
    cases["long_tail"] = ([300000] + [128] * 63, 2, 106)
    for b in (33, 255, 256, 257, 1023, 1024):
        lens = [rnd.randint(0, 131072) for _ in range(b)]
        cases[f"random_bs{b}"] = (lens, rnd.choice((1, 2, 4)), rnd.choice((1, 26, 53, 106)))
    return cases


CASES = _cases()


@pytest.mark.parametrize("name", list(CASES))
def test_split_plan_matches_triton_and_host(name: str) -> None:
    lens, rows, waves = CASES[name]
    batch = len(lens)
    seqs = torch.tensor(lens, dtype=torch.int32, device="cuda")
    plan_cute = torch.full((batch,), -1, dtype=torch.int32, device="cuda")
    plan_triton = torch.full((batch,), -1, dtype=torch.int32, device="cuda")
    kwargs = dict(num_requests=batch, rows_per_request=rows, wave_clusters=waves, max_split=32)
    cute_plan(seqs, plan_cute, **kwargs)
    triton_plan(seqs, plan_triton, **kwargs)
    host = plan_mla_split_kv(lens, rows_per_request=rows, wave_clusters=waves, max_split=32)
    assert plan_cute.tolist() == plan_triton.tolist() == host
