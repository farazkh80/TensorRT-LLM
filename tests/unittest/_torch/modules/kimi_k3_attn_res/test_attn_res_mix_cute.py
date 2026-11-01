# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity of the CuTe DSL attention-residual mix with the stock ops it replaces.

Every case runs ``attn_res_mix`` at the dispatch default for its token count and the stock path on identical
inputs, and requires the CuTe error against an fp64 reference to be no worse than stock's (relative L2 within
1.1x, max abs within 2x). Stock is ``attn_res_fwd`` for the mix, the C++ ``attn_res_[add_]rmsnorm_fwd`` for the
KimiK3RMSNorm rounding, and ATen add + ``attn_res_fwd`` + ``RMSNorm`` for the FlashInfer rounding.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_k3_attn_res.attn_res_mix_cute import (
    attn_res_mix,
    default_config,
    is_attn_res_mix_cute_available,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

HIDDEN_SIZE = 7168
EPS = 1e-5
# Dispatch boundaries (16/17, 64/65), pipe grids below one CTA per SM (65..211) and mixed-step token counts.
NUM_TOKENS = [1, 2, 16, 17, 63, 64, 65, 127, 213, 8192, 8256]
NUM_CANDIDATES = list(range(2, 10))

pytestmark = pytest.mark.skipif(
    not is_attn_res_mix_cute_available() or not hasattr(torch.ops.trtllm, "attn_res_fwd"),
    reason="needs an sm_100-family GPU, nvidia-cutlass-dsl and the trtllm attn_res ops",
)


def _inputs(num_tokens: int, num_candidates: int, seed: int = 0) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = num_candidates - 1
    scales = torch.tensor([0.5 * 1.6**i for i in range(k)], device="cuda").view(k, 1, 1)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=g)

    return dict(
        bank=(randn(k, num_tokens, HIDDEN_SIZE) * scales).to(torch.bfloat16),
        prefix=(randn(num_tokens, HIDDEN_SIZE) * 2.0).to(torch.bfloat16),
        addend=(randn(num_tokens, HIDDEN_SIZE) * 0.7).to(torch.bfloat16),
        res_w=(randn(HIDDEN_SIZE) * 0.03).to(torch.bfloat16),
        rms_w=(1.0 + 0.2 * randn(HIDDEN_SIZE)).to(torch.bfloat16),
        out_w=(1.0 + 0.2 * randn(HIDDEN_SIZE)).to(torch.bfloat16),
    )


def _ref_mix(prefix, bank, res_w, rms_w, chunk=1024):
    q = (rms_w.float() * res_w.float()).double()
    out = torch.empty(prefix.shape, dtype=torch.float64, device=prefix.device)
    for s in range(0, prefix.shape[0], chunk):
        v = torch.cat([bank[:, s : s + chunk], prefix[s : s + chunk].unsqueeze(0)]).double()
        logits = torch.rsqrt(v.square().mean(-1) + EPS) * (v * q).sum(-1)
        out[s : s + chunk] = (torch.softmax(logits, 0).unsqueeze(-1) * v).sum(0)
    return out


def _ref_norm(x, w):
    x = x.double()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * w.double()


def _assert_no_worse(out, stock, ref, tag):
    assert torch.isfinite(out).all() and out.abs().max() > 0, (
        f"{tag}: output not finite or all zero"
    )
    err = (out.double() - ref).abs().max().item(), ((out.double() - ref).norm() / ref.norm()).item()
    stock_err = (
        (stock.double() - ref).abs().max().item(),
        ((stock.double() - ref).norm() / ref.norm()).item(),
    )
    assert err[1] <= 1.1 * stock_err[1] + 1e-6 and err[0] <= 2 * stock_err[0] + 1e-6, (
        f"{tag}: cute max/rel {err[0]:.3e}/{err[1]:.3e} vs stock {stock_err[0]:.3e}/{stock_err[1]:.3e}"
    )


def _stock_mix(x, prefix):
    m, k = prefix.shape[0], x["bank"].shape[0]
    out = torch.ops.trtllm.attn_res_fwd(
        prefix.view(m, 1, HIDDEN_SIZE),
        x["bank"].view(k, m, 1, HIDDEN_SIZE),
        x["res_w"],
        x["rms_w"],
        EPS,
    )[0]
    return out.view(m, HIDDEN_SIZE)


def _stock_mix_norm(x, add: bool, mode: str):
    m, k = x["prefix"].shape[0], x["bank"].shape[0]
    if mode == "kimi":
        args = (x["bank"].view(k, m, 1, HIDDEN_SIZE), x["res_w"], x["rms_w"], x["out_w"], EPS, EPS)
        prefix = x["prefix"].view(m, 1, HIDDEN_SIZE)
        if add:
            out = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
                prefix, x["addend"].view(m, 1, HIDDEN_SIZE), *args
            )[1]
        else:
            out = torch.ops.trtllm.attn_res_rmsnorm_fwd(prefix, *args)
        return out.view(m, HIDDEN_SIZE)
    norm = RMSNorm(hidden_size=HIDDEN_SIZE, eps=EPS, dtype=torch.bfloat16).to("cuda")
    norm.weight.data.copy_(x["out_w"])
    prefix = x["prefix"] + x["addend"] if add else x["prefix"]
    return norm(_stock_mix(x, prefix))


@pytest.mark.parametrize("num_candidates", NUM_CANDIDATES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.no_grad()
def test_mix_matches_stock(num_tokens: int, num_candidates: int) -> None:
    x = _inputs(num_tokens, num_candidates)
    out = attn_res_mix(x["prefix"], x["bank"], x["res_w"], x["rms_w"], EPS)
    ref = _ref_mix(x["prefix"], x["bank"], x["res_w"], x["rms_w"])
    _assert_no_worse(out, _stock_mix(x, x["prefix"]), ref, f"mix {default_config(num_tokens)}")


@pytest.mark.parametrize("mode", ["flashinfer", "kimi"])
@pytest.mark.parametrize("add", [False, True], ids=["norm", "add_norm"])
@pytest.mark.parametrize("num_candidates", NUM_CANDIDATES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.no_grad()
def test_mix_norm_matches_stock(num_tokens: int, num_candidates: int, add: bool, mode: str) -> None:
    if mode == "kimi" and not hasattr(torch.ops.trtllm, "attn_res_add_rmsnorm_fwd"):
        pytest.skip("C++ fused attn_res ops not built")
    x = _inputs(num_tokens, num_candidates, seed=1)
    kwargs = dict(out_norm_weight=x["out_w"], out_norm_eps=EPS, norm_mode=mode)
    prefix = x["prefix"]
    if add:
        # The updated prefix goes to the slot after the bank in the same allocation, as on snapshot layers.
        full = torch.empty(
            num_candidates, num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        full[:-1].copy_(x["bank"])
        updated, out = attn_res_mix(
            prefix,
            full[:-1],
            x["res_w"],
            x["rms_w"],
            EPS,
            addend=x["addend"],
            updated=full[-1],
            **kwargs,
        )
        assert updated.data_ptr() == full[-1].data_ptr()
        assert torch.equal(full[-1], prefix + x["addend"]), (
            "updated prefix differs from the ATen add"
        )
        assert torch.equal(full[:-1], x["bank"]), "bank rows were modified"
        prefix = prefix + x["addend"]
    else:
        out = attn_res_mix(prefix, x["bank"], x["res_w"], x["rms_w"], EPS, **kwargs)
    ref = _ref_norm(
        _ref_mix(prefix, x["bank"], x["res_w"], x["rms_w"]).to(torch.bfloat16), x["out_w"]
    )
    tag = f"{'add+' if add else ''}mix+norm[{mode}] {default_config(num_tokens)}"
    _assert_no_worse(out, _stock_mix_norm(x, add, mode), ref, tag)


@pytest.mark.parametrize(
    "num_tokens,num_candidates,config,grid_x",
    [
        (64, 9, ("reg", 7, 1), 3),  # persistent clusters: double-buffered slots and mbarrier phases
        (64, 9, ("reg", 1, 2), 5),
        (300, 9, ("reg", 2, 1), 0),
        (64, 9, ("pipe", 2, 4, 3), 3),  # ring shorter than one token's rows
        (64, 5, ("pipe", 2, 2, 2), 7),
        (300, 7, ("pipe", 4, 3, 2, True), 0),  # split-phase reductions
        (300, 9, ("pipe", 1, 12, 3), 0),
        (1000, 12, None, 0),
    ],
)
@torch.no_grad()
def test_non_default_configs(num_tokens, num_candidates, config, grid_x) -> None:
    x = _inputs(num_tokens, num_candidates, seed=2)
    updated, out = attn_res_mix(
        x["prefix"],
        x["bank"],
        x["res_w"],
        x["rms_w"],
        EPS,
        addend=x["addend"],
        out_norm_weight=x["out_w"],
        out_norm_eps=EPS,
        config=config,
        grid_x=grid_x,
    )
    prefix = x["prefix"] + x["addend"]
    assert torch.equal(updated, prefix)
    ref = _ref_norm(
        _ref_mix(prefix, x["bank"], x["res_w"], x["rms_w"]).to(torch.bfloat16), x["out_w"]
    )
    _assert_no_worse(
        out, _stock_mix_norm(x, True, "flashinfer"), ref, f"add+mix+norm {config} grid {grid_x}"
    )
