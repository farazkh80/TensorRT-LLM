# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decoder wiring of the CuTe attention-residual path (TLLM_K3_CUTE_NORM / TLLM_K3_CUTE_NORM_FOLD).

* The attn-res helpers with the switch on are compared with the same helpers with the switch off (the stock path
  the decoder runs at that shape), both against an fp64 reference.
* The residual-add fold is compared bit for bit with the unfolded CuTe path over a stack of real
  ``KimiLinearDecoderLayer.forward`` calls (stub attention and MLP), including snapshot layers and layers that cannot
  fold, and through ``KimiLinearModel.forward`` (final output, speculative-decoding taps, helix first layer).
* With the switch off, the fold arguments only move the stock add and must not change any bit (and must never
  reach the kernel).
"""

import inspect
import math
import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_kimi_linear as mk
from tensorrt_llm._torch.modules.kimi_k3_attn_res.attn_res_mix_cute import (
    attn_res_mix,
    is_attn_res_mix_cute_available,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

HIDDEN_SIZE = 7168
EPS = 1e-5
HAS_HELIX = "helix_a2a" in inspect.signature(mk.KimiLinearDecoderLayer.forward).parameters

pytestmark = pytest.mark.skipif(
    not is_attn_res_mix_cute_available() or not hasattr(torch.ops.trtllm, "attn_res_fwd"),
    reason="needs an sm_100-family GPU, nvidia-cutlass-dsl and the trtllm attn_res ops",
)


@pytest.fixture
def switches(monkeypatch):
    monkeypatch.setattr(mk, "_cute_attn_res_mix", attn_res_mix, raising=False)

    def set_switches(norm: bool, fold: bool = False) -> None:
        monkeypatch.setattr(mk, "_CUTE_NORM_ENABLED", norm)
        monkeypatch.setattr(mk, "_CUTE_NORM_FOLD_ENABLED", fold)

    def drop_kernel() -> None:
        monkeypatch.delattr(mk, "_cute_attn_res_mix", raising=False)

    set_switches.drop_kernel = drop_kernel
    set_switches(False)
    return set_switches


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(g, *shape, scale=1.0, shift=0.0):
    return (shift + scale * torch.randn(*shape, device="cuda", generator=g)).to(torch.bfloat16)


def _proj(seed: int) -> nn.Linear:
    proj = nn.Linear(HIDDEN_SIZE, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    proj.weight.data.copy_(_randn(_generator(seed), 1, HIDDEN_SIZE, scale=0.03))
    return proj


def _res_norm(seed: int) -> nn.Module:
    norm = mk.KimiK3RMSNorm(HIDDEN_SIZE, eps=EPS, dtype=torch.bfloat16, device="cuda")
    norm.weight.data.copy_(_randn(_generator(seed), HIDDEN_SIZE, scale=0.2, shift=1.0))
    return norm


def _out_norm(seed: int) -> nn.Module:
    norm = RMSNorm(hidden_size=HIDDEN_SIZE, eps=EPS, dtype=torch.bfloat16).to("cuda")
    norm.weight.data.copy_(_randn(_generator(seed), HIDDEN_SIZE, scale=0.2, shift=1.0))
    return norm


def _ref_mix(prefix, bank, proj, norm):
    q = (norm.weight.float() * proj.weight.reshape(-1).float()).double()
    v = torch.cat([bank, prefix.unsqueeze(0)]).double()
    logits = torch.rsqrt(v.square().mean(-1) + EPS) * (v * q).sum(-1)
    return (torch.softmax(logits, 0).unsqueeze(-1) * v).sum(0)


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


@pytest.mark.parametrize("num_candidates", [2, 5, 7, 9])
@pytest.mark.parametrize("num_tokens", [1, 2, 16, 64, 65, 8192])
@torch.no_grad()
def test_helpers_match_stock_path(switches, num_tokens: int, num_candidates: int) -> None:
    g = _generator(3)
    prefix = _randn(g, num_tokens, HIDDEN_SIZE, scale=2.0)
    addend = _randn(g, num_tokens, HIDDEN_SIZE, scale=0.7)
    bank = _randn(g, num_candidates - 1, num_tokens, HIDDEN_SIZE, scale=1.5)
    proj, norm, out_norm = _proj(4), _res_norm(5), _out_norm(6)

    def both(fn, *args, **kwargs):
        switches(norm=False)
        stock = fn(*args, **kwargs)
        switches(norm=True)
        return fn(*args, **kwargs), stock

    mix_ref = _ref_mix(prefix, bank, proj, norm)
    cute, stock = both(mk._apply_attn_res, prefix, bank, proj, norm)
    _assert_no_worse(cute, stock, mix_ref, "_apply_attn_res")

    ref = _ref_norm(mix_ref.to(torch.bfloat16), out_norm.weight)
    cute, stock = both(mk._apply_attn_res_and_rmsnorm, prefix, bank, proj, norm, out_norm)
    _assert_no_worse(cute, stock, ref, "_apply_attn_res_and_rmsnorm")

    calls = {False: 0, True: 0}
    for norm_on in (False, True):
        switches(norm=norm_on)

        def hook(norm_on=norm_on):
            calls[norm_on] += 1

        out = mk._apply_attn_res_and_rmsnorm(prefix, bank, proj, norm, out_norm, before_norm=hook)
        if norm_on:
            _assert_no_worse(out, stock, ref, "_apply_attn_res_and_rmsnorm(before_norm)")
    assert calls[True] == calls[False], "before_norm must run exactly where the stock path runs it"

    updated_ref = prefix + addend
    ref = _ref_norm(_ref_mix(updated_ref, bank, proj, norm).to(torch.bfloat16), out_norm.weight)
    (updated, cute), (updated_stock, stock) = both(
        mk._apply_attn_res_add_and_rmsnorm, prefix, addend, bank, proj, norm, out_norm
    )
    assert torch.equal(updated, updated_ref) and torch.equal(updated_stock, updated_ref)
    _assert_no_worse(cute, stock, ref, "_apply_attn_res_add_and_rmsnorm")


class _StubMixer(nn.Module):
    """Deterministic stand-in for attention / MLP that records what it was fed."""

    def __init__(self, seed: int):
        super().__init__()
        g = _generator(seed)
        self.scale = _randn(g, HIDDEN_SIZE, scale=0.5)
        self.shift = _randn(g, HIDDEN_SIZE, scale=0.1)
        self.inputs = []

    def forward(self, x, attn_metadata=None):
        self.inputs.append(x.clone())
        return (torch.tanh(x.float() * self.scale.float() + self.shift.float()) * 2.0).to(
            torch.bfloat16
        )


class _IdentityHelix:
    def scatter_rows(self, x, num_tokens):
        return x

    def all_gather(self, x, num_tokens):
        return x

    def reduce_scatter(self, x, num_tokens):
        return x

    def all_rank_num_tokens(self, num_tokens):
        return [num_tokens]


class _Taps:
    _capture_layer_set = None

    def __init__(self):
        self.taps = []

    def maybe_capture_hidden_states(self, layer_id, hidden_states, residual):
        self.taps.append((layer_id, hidden_states.clone()))


def _layer(layer_idx: int, block: int, token_sliced: bool = False) -> nn.Module:
    layer = mk.KimiLinearDecoderLayer.__new__(mk.KimiLinearDecoderLayer)
    nn.Module.__init__(layer)
    seed = 100 * (layer_idx + 1)
    layer.layer_idx = layer_idx
    layer.attn_res_block_size = block
    layer.is_kda = layer_idx % 2 == 0
    layer.is_moe = False
    layer.token_sliced = token_sliced
    layer.linear_attn = layer.self_attn = _StubMixer(seed)
    layer.mlp = _StubMixer(seed + 1)
    layer.input_layernorm = _out_norm(seed + 2)
    layer.post_attention_layernorm = _out_norm(seed + 3)
    layer.self_attention_res_proj = _proj(seed + 4)
    layer.self_attention_res_norm = _res_norm(seed + 5)
    layer.mlp_res_proj = _proj(seed + 6)
    layer.mlp_res_norm = _res_norm(seed + 7)
    return layer


def _records(layers):
    return [t for layer in layers for t in layer.linear_attn.inputs + layer.mlp.inputs]


def _clear(layers):
    for layer in layers:
        layer.linear_attn.inputs.clear()
        layer.mlp.inputs.clear()


def _run_stack(layers, hidden_states, num_snapshots_max, fold: bool, capture_at=()):
    """Layer-wise-runner loop: fold threads the deferred add and finishes the last one at the end."""
    metadata = types.SimpleNamespace(num_contexts=0, num_tokens=hidden_states.shape[0])
    bank = hidden_states.new_zeros(num_snapshots_max, *hidden_states.shape)
    num_snapshots, addend, taps = 0, None, _Taps()
    for i, layer in enumerate(layers):
        capture = (taps, i - 1) if i in capture_at else None
        if fold:
            hidden_states, num_snapshots, addend = layer(
                hidden_states,
                bank,
                num_snapshots,
                metadata,
                capture=capture,
                residual_addend=addend,
                defer_residual_add=True,
            )
        else:
            hidden_states, num_snapshots = layer(
                hidden_states, bank, num_snapshots, metadata, capture=capture
            )
    if addend is not None:
        hidden_states = hidden_states + addend
    return hidden_states, bank[:num_snapshots].clone(), taps.taps


def _assert_bitwise(a, b, what):
    assert len(a) == len(b), f"{what}: {len(a)} vs {len(b)} tensors"
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x, y), f"{what}[{i}] differs"


@pytest.mark.parametrize("norm_on", [True, False], ids=["cute", "stock"])
@pytest.mark.parametrize("num_tokens", [1, 16, 40, 300])
@torch.no_grad()
def test_fold_is_bit_exact_over_a_layer_stack(switches, num_tokens: int, norm_on: bool) -> None:
    """Snapshot layers 0/3/6 (block 3); layer 0 has no snapshot to mix and layer 4 taps, so both add first."""
    switches(norm=norm_on)
    if not norm_on:
        # The fold arguments must not reach the CuTe kernel when it is not enabled.
        switches.drop_kernel()
    layers = [_layer(i, block=3) for i in range(8)]
    hidden = _randn(_generator(7), num_tokens, HIDDEN_SIZE, scale=2.0)
    results = {}
    for fold in (False, True):
        _clear(layers)
        out, bank, taps = _run_stack(
            layers, hidden, num_snapshots_max=3, fold=fold, capture_at=(4,)
        )
        results[fold] = (out, bank, [t for _, t in taps], _records(layers))
    for idx, what in enumerate(["output", "snapshot bank", "taps", "attention/MLP inputs"]):
        lhs, rhs = results[True][idx], results[False][idx]
        _assert_bitwise(
            lhs if isinstance(lhs, list) else [lhs], rhs if isinstance(rhs, list) else [rhs], what
        )


def _model(layers, block: int, helix_first_layer=None) -> nn.Module:
    model = mk.KimiLinearModel.__new__(mk.KimiLinearModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList(layers)
    model.num_attn_res_snapshots = math.ceil(len(layers) / block)
    model.embed_tokens = None
    model.output_attn_res_proj = _proj(11)
    model.output_attn_res_norm = _res_norm(12)
    model.norm = _out_norm(13)
    if HAS_HELIX:
        model.helix_a2a = _IdentityHelix() if helix_first_layer is not None else None
        model.helix_a2a_first_layer = (
            len(layers) if helix_first_layer is None else helix_first_layer
        )
    return model


@pytest.mark.parametrize("site", ["final_output", "spec_taps", "helix_first_layer"])
@pytest.mark.parametrize("num_tokens", [1, 16, 300])
@torch.no_grad()
def test_fold_model_sites_are_bit_exact(switches, num_tokens: int, site: str) -> None:
    if site == "helix_first_layer" and not HAS_HELIX:
        pytest.skip("decoder has no helix token slicing")
    helix_first_layer = 4 if site == "helix_first_layer" else None
    layers = [
        _layer(i, block=3, token_sliced=helix_first_layer is not None and i >= helix_first_layer)
        for i in range(7)
    ]
    model = _model(layers, block=3, helix_first_layer=helix_first_layer)
    hidden = _randn(_generator(8), num_tokens, HIDDEN_SIZE, scale=2.0)
    metadata = types.SimpleNamespace(num_contexts=0, num_tokens=num_tokens)
    results = {}
    for fold in (False, True):
        switches(norm=True, fold=fold)
        _clear(layers)
        spec = _Taps() if site == "spec_taps" else None
        out = model(metadata, inputs_embeds=hidden, spec_metadata=spec)
        taps = [t for _, t in spec.taps] if spec is not None else []
        results[fold] = ([out], taps, _records(layers))
    for idx, what in enumerate(["model output", "spec taps", "attention/MLP inputs"]):
        _assert_bitwise(results[True][idx], results[False][idx], what)
