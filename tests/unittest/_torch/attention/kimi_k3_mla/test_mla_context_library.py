# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cute_dsl_mla_context FMHA library behind TLLM_K3_CUTE_MLA=ctx.

Two TrtllmAttention context layers on identical inputs and starting caches: one flagged ``_k3_cute_ctx`` (served by
cute_dsl_mla_context), one not (thop.attention). Checks which library ran, the paged cache bitwise, K's rope columns
bitwise and both outputs against an fp32 reference; a flagged call without a latent cache must fall back.
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

from tensorrt_llm._torch.attention import mla as mla_module
from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla_context import CuteDslMlaContextFmha
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType

pytestmark = requires_cute_mla


@pytest.fixture
def cute_ctx_calls(monkeypatch):
    monkeypatch.setenv("TLLM_K3_CUTE_MLA", "ctx")
    mla_module.k3_cute_mla_enabled.cache_clear()
    calls = []
    original = CuteDslMlaContextFmha.run_mla_context

    def counted(self, params):
        calls.append(params.num_tokens)
        return original(self, params)

    monkeypatch.setattr(CuteDslMlaContextFmha, "run_mla_context", counted)
    yield calls
    mla_module.k3_cute_mla_enabled.cache_clear()


@pytest.mark.parametrize(
    "lens", [[8192], [100, 1000, 3000, 64, 1, 4097]], ids=["isl8192", "ragged6"]
)
def test_flagged_layer_runs_cute_context(cute_ctx_calls, lens) -> None:
    torch.manual_seed(0)
    total = sum(lens)
    bf = torch.bfloat16
    q = torch.empty(total, H * D, dtype=bf, device="cuda").uniform_(-1, 1)
    latent = torch.empty(total, LAT + ROPE, dtype=bf, device="cuda").uniform_(-1, 1)
    k0 = torch.empty(total, H * D, dtype=bf, device="cuda").uniform_(-1, 1)
    kv_buf = torch.empty(total, H * (NOPE + DV), dtype=bf, device="cuda").uniform_(-1, 1)
    v = kv_buf[:, H * NOPE :]
    caches, outs, ks, ran = [], [], [], []
    for flagged in (False, True):
        layer, kv_cache_manager, metadata = build_context_layer(lens, 2, 1)
        try:
            layer._k3_cute_ctx = flagged
            for i in range(2):
                pool = kv_cache_manager.get_buffers(i)
                generator = torch.Generator(device="cuda").manual_seed(7 + i)
                pool.copy_(torch.randn(pool.shape, generator=generator, device="cuda").to(bf))
            k = k0.clone()
            before = len(cute_ctx_calls)
            out = layer.forward(
                q.clone(),
                k,
                v,
                metadata,
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent,
            )
            torch.cuda.synchronize()
            ran.append(len(cute_ctx_calls) - before)
            caches.append([kv_cache_manager.get_buffers(i).clone() for i in range(2)])
            outs.append(out.clone())
            ks.append(k.view(total, H, D).clone())
            if flagged:
                before = len(cute_ctx_calls)
                layer.forward(
                    q.clone(),
                    k0.clone(),
                    v,
                    metadata,
                    attention_input_type=AttentionInputType.context_only,
                    latent_cache=None,
                )
                torch.cuda.synchronize()
                assert len(cute_ctx_calls) == before, "a call without latent cache must fall back"
        finally:
            kv_cache_manager.shutdown()
    assert ran == [0, 1]
    for i in range(2):
        assert torch.equal(bits(caches[0][i]), bits(caches[1][i])), f"cache layer {i}"
    assert torch.equal(bits(ks[0][..., NOPE:]), bits(ks[1][..., NOPE:]))
    ref = reference(
        q.view(total, H, D), ks[0], v.reshape(total, H, DV), 1.0 / math.sqrt(D), cu_list(lens)
    )
    e_stock, e_cute = err(outs[0], ref), err(outs[1], ref)
    assert e_cute[2] and e_cute[3]
    assert e_cute[0] <= 1.05 * e_stock[0], f"CuTe rel-L2 {e_cute[0]:.2e} vs stock {e_stock[0]:.2e}"
