# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 KDA CuTe DSL kernels (TLLM_K3_CUTE_KDA) against the kernels they replace.

Each kernel runs on identical inputs with the stock kernel; errors are measured against an fp64 reference and
must be no worse than the stock kernel's. The mixer-level tests also check that the switched path reached the
CuTe DSL kernel (``CUTE_KDA_PATHS``) instead of falling back silently.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")

from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda import (  # noqa: E402
    fused_decode,
    gated_rmsnorm,
    reset_state_rows,
)
from tensorrt_llm._torch.modules.kimi_kda import kimi_kda_mixer  # noqa: E402
from tensorrt_llm._torch.modules.kimi_kda._kda_decode import (
    run_kda_decode_fusion_cuda,  # noqa: E402
)
from tensorrt_llm._torch.modules.mamba.layernorm_gated import (
    rms_norm_gated_token_major,  # noqa: E402
)
from tensorrt_llm._torch.modules.mamba.recurrent_state_cache import (  # noqa: E402
    reset_recurrent_state_rows,
)

H = 96
D = 128
DIM = H * D
LOWER_BOUND = -5.0
EPS = 1e-5
SCALE = D**-0.5

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="CuTe DSL KDA kernels target the SM10x family",
)


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected.double()).norm() / expected.double().norm()).item()


def _padded(t: torch.Tensor, pad: int) -> torch.Tensor:
    """Copy of ``t`` whose leading (slot) stride is padded by ``pad`` elements."""
    if pad == 0:
        return t.clone()
    rows = torch.zeros(t.shape[0], t[0].numel() + pad, dtype=t.dtype, device=t.device)
    view = rows[:, : t[0].numel()].view(t.shape)
    view.copy_(t)
    return view


class _DecodeInputs:
    """One decode step at Kimi K3 DEP shapes with mixer-like views of the projection outputs."""

    def __init__(self, batch: int, seed: int, extra_slots: int = 6):
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def randn(*shape, scale=1.0, dtype=torch.bfloat16):
            return (torch.randn(*shape, device="cuda", generator=gen) * scale).to(dtype)

        self.batch = batch
        self.slots_total = batch + extra_slots
        self.qkvg = randn(batch, 4 * DIM)
        self.bfa = randn(batch, 224, scale=2.0)
        self.g = randn(batch, DIM, scale=2.0)
        self.w = [randn(4, DIM, scale=0.4) for _ in range(3)]
        self.A_log = torch.log(torch.rand(H, device="cuda", generator=gen) * 15 + 1)
        self.dt_bias = torch.rand(DIM, device="cuda", generator=gen) * 4.6 - 6.9
        self.norm_w = 1.0 + 0.2 * torch.randn(D, device="cuda", generator=gen)
        self.conv = randn(self.slots_total, 3 * DIM, 3)
        self.state = torch.randn(self.slots_total, H, D, D, device="cuda", generator=gen) * 0.05
        slots = torch.randperm(self.slots_total, device="cuda", generator=gen)[:batch]
        if not (slots == 0).any():
            slots[0] = 0
        self.slots = slots.to(torch.int32).contiguous()
        self.packed_w = fused_decode.pack_conv_weight(*self.w)

    @property
    def x(self):
        return self.qkvg[:, : 3 * DIM]

    @property
    def gate(self):
        return self.qkvg[:, 3 * DIM :].view(self.batch, H, D)

    @property
    def beta(self):
        return self.bfa[:, 128:224]

    def run_stock(self, conv, state, out):
        x = self.x
        run_kda_decode_fusion_cuda(
            x_q=x[:, :DIM].unflatten(-1, (H, D)).unsqueeze(0),
            x_k=x[:, DIM : 2 * DIM].unflatten(-1, (H, D)).unsqueeze(0),
            x_v=x[:, 2 * DIM :].unflatten(-1, (H, D)).unsqueeze(0),
            w_q_t=self.w[0],
            w_k_t=self.w[1],
            w_v_t=self.w[2],
            bias_q=None,
            bias_k=None,
            bias_v=None,
            cs_q=conv[:, :DIM],
            cs_k=conv[:, DIM : 2 * DIM],
            cs_v=conv[:, 2 * DIM :],
            A_log=self.A_log,
            g=self.g.unflatten(-1, (H, D)).unsqueeze(0),
            dt_bias=self.dt_bias,
            beta=self.beta.unsqueeze(0),
            state=state,
            onorm_g=self.gate.unsqueeze(0),
            onorm_weight=self.norm_w,
            out=out.view(self.batch, 1, H, D),
            ssm_state_indices=self.slots,
            scale=SCALE,
            onorm_eps=EPS,
            lower_bound=LOWER_BOUND,
            use_beta_sigmoid_in_kernel=True,
            update_conv_cache=True,
        )

    def run_cute(self, conv, state, out, slots=None, **kwargs):
        fused_decode.kda_decode_t1(
            self.x,
            self.packed_w,
            conv,
            self.g,
            self.beta,
            self.A_log,
            self.dt_bias,
            self.slots if slots is None else slots,
            state,
            self.gate,
            self.norm_w,
            out,
            lower_bound=LOWER_BOUND,
            norm_eps=EPS,
            scale=SCALE,
            **kwargs,
        )

    def reference(self):
        """fp64 decode of the stock semantics: (output, new state rows, new conv rows)."""
        sl = self.slots.long()
        window = torch.cat((self.conv[sl].double(), self.x.double().unsqueeze(-1)), -1)
        weight = torch.stack(self.w).double().permute(0, 2, 1).reshape(3 * DIM, 4)
        acc = (window * weight).sum(-1)
        act = (acc * torch.sigmoid(acc)).view(self.batch, 3, H, D)
        q, k, v = act[:, 0], act[:, 1], act[:, 2]
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        a = torch.exp(self.A_log.double())[None, :, None]
        g = self.g.double().view(self.batch, H, D) + self.dt_bias.double().view(H, D)
        decay = torch.exp(LOWER_BOUND * torch.sigmoid(a * g))
        beta = torch.sigmoid(self.beta.double())
        S = self.state[sl].double() * decay[:, :, None, :]
        u = (v - (S * k[:, :, None, :]).sum(-1)) * beta[..., None]
        S = S + k[:, :, None, :] * u[..., None]
        o = (S * q[:, :, None, :]).sum(-1)
        y = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS) * self.norm_w.double()
        return y * torch.sigmoid(self.gate.double()), S, window[..., 1:]


@pytest.mark.parametrize("state_pad,conv_pad", [(0, 0), (64, 4)], ids=["dense", "slot_padded"])
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 8, 16, 33, 64])
@torch.no_grad()
def test_decode_matches_stock_kernel(batch: int, state_pad: int, conv_pad: int) -> None:
    t = _DecodeInputs(batch, seed=batch)
    conv_s, state_s = _padded(t.conv, conv_pad), _padded(t.state, state_pad)
    conv_c, state_c = _padded(t.conv, conv_pad), _padded(t.state, state_pad)
    out_s = torch.zeros(batch, H, D, device="cuda", dtype=torch.bfloat16)
    out_c = torch.zeros_like(out_s)
    t.run_stock(conv_s, state_s, out_s)
    t.run_cute(conv_c, state_c, out_c)
    y, S, window = t.reference()
    sl = t.slots.long()
    untouched = torch.ones(t.slots_total, dtype=torch.bool, device="cuda")
    untouched[sl] = False

    assert torch.isfinite(out_c.float()).all() and out_c.abs().max() > 0
    assert torch.equal(conv_c, conv_s)
    assert torch.equal(conv_c[sl].double(), window)
    assert torch.equal(state_c[untouched], t.state[untouched])
    assert torch.equal(conv_c[untouched], t.conv[untouched])
    assert _rel(out_c, y) <= 1.05 * _rel(out_s, y)
    assert _rel(state_c[sl], S) <= 1.05 * _rel(state_s[sl], S)


@pytest.mark.parametrize("batch", [1, 2, 5])
@torch.no_grad()
def test_decode_negative_slot_is_inert(batch: int) -> None:
    t = _DecodeInputs(batch, seed=50 + batch)
    slots = t.slots.clone()
    slots[-1] = -1
    conv, state = t.conv.clone(), t.state.clone()
    out = torch.full((batch, H, D), 3.0, device="cuda", dtype=torch.bfloat16)
    t.run_cute(conv, state, out, slots=slots)
    keep = torch.ones(t.slots_total, dtype=torch.bool, device="cuda")
    keep[slots[slots >= 0].long()] = False
    assert (out[-1] == 0).all()
    assert torch.equal(state[keep], t.state[keep])
    assert torch.equal(conv[keep], t.conv[keep])


@pytest.mark.parametrize("batch", [1, 4, 33])
@torch.no_grad()
def test_decode_cuda_graph_replays_new_slots(batch: int) -> None:
    """Graph captured once, replayed after new slots are written into the captured index buffer."""
    t = _DecodeInputs(batch, seed=70 + batch, extra_slots=batch + 6)
    conv0, state0 = t.conv.clone(), t.state.clone()
    slot_buf = t.slots.clone()
    out = torch.empty(batch, H, D, device="cuda", dtype=torch.bfloat16)
    t.run_cute(t.conv, t.state, out, slots=slot_buf)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        t.run_cute(t.conv, t.state, out, slots=slot_buf)
    gen = torch.Generator(device="cuda").manual_seed(batch)
    new = torch.randperm(t.slots_total, device="cuda", generator=gen)[:batch].to(torch.int32)
    slot_buf.copy_(new)
    t.conv.copy_(conv0)
    t.state.copy_(state0)
    graph.replay()
    t.slots = new
    conv_e, state_e = conv0.clone(), state0.clone()
    out_e = torch.empty_like(out)
    t.run_cute(conv_e, state_e, out_e)
    assert torch.equal(out, out_e)
    assert torch.equal(t.state, state_e)
    assert torch.equal(t.conv, conv_e)


_BF16_NEG_ZERO = -32768


@pytest.mark.parametrize(
    "producer",
    [
        dict(),
        dict(fgs_head_group=1),
        dict(fgs_head_group=8),
        dict(fgs_persistent=True),
        dict(fgs_persistent=True, fgs_head_group=1),
        dict(fgs_persistent=True, fgs_ctas=212),
    ],
    ids=[
        "grid",
        "grid_rows_first",
        "grid_head_group8",
        "persistent",
        "persistent_rows_first",
        "persistent_212",
    ],
)
@pytest.mark.parametrize("inputs", ["random", "signed_zero_output"])
@pytest.mark.parametrize("batch", [1, 4, 16, 33])
@torch.no_grad()
def test_decode_fgs_producer_matches_default(batch: int, inputs: str, producer: dict) -> None:
    """fgs_producer output equals the default build's with -0.0 mapped to +0.0, covers every
    16-B chunk of a -0.0-initialized buffer and writes nothing past H*128 in a padded row."""
    t = _DecodeInputs(batch, seed=90 + batch)
    if inputs == "signed_zero_output":
        # Zero state and v with q < 0 give o = -0.0; signed norm weights and a very negative
        # output gate make the default build store -0.0 (also from underflowing products).
        t.conv.zero_()
        t.state.zero_()
        t.qkvg[:, :DIM] = -t.qkvg[:, :DIM].abs() - 0.5
        t.qkvg[:, 2 * DIM : 3 * DIM] = 0
        t.qkvg[:, 3 * DIM :: 3] = -100.0
        t.w[0][3] = t.w[0][3].abs() + 0.1
        t.packed_w = fused_decode.pack_conv_weight(*t.w)
        t.norm_w = torch.tensor([1.0, -1.0, -1e-41, -1e-30], device="cuda").repeat(D // 4)
    slots = t.slots.clone()
    if batch > 1:
        slots[-1] = -1

    def run(**kwargs):
        rows = torch.full((batch, DIM + 64), -0.0, device="cuda", dtype=torch.bfloat16)
        conv, state = t.conv.clone(), t.state.clone()
        t.run_cute(conv, state, rows[:, :DIM].view(batch, H, D), slots=slots, **kwargs)
        return rows.view(torch.int16), conv, state

    ref, conv_ref, state_ref = run()
    out, conv, state = run(fgs_producer=True, **producer)
    expected = torch.where(ref[:, :DIM] == _BF16_NEG_ZERO, 0, ref[:, :DIM])
    if inputs == "signed_zero_output":
        assert (ref[:, :DIM] == _BF16_NEG_ZERO).any()
    assert torch.equal(out[:, :DIM], expected)
    assert not (out[:, :DIM] == _BF16_NEG_ZERO).any()
    assert (out[:, DIM:] == _BF16_NEG_ZERO).all()
    assert torch.equal(state, state_ref)
    assert torch.equal(conv, conv_ref)


def _decode_fgs_run(t: _DecodeInputs, **kwargs):
    rows = torch.full((t.batch, DIM + 64), -0.0, device="cuda", dtype=torch.bfloat16)
    conv, state = t.conv.clone(), t.state.clone()
    t.run_cute(conv, state, rows[:, :DIM].view(t.batch, H, D), **kwargs)
    return rows.view(torch.int16), conv, state


def _canonical_inputs(t: _DecodeInputs) -> None:
    for buf in (t.qkvg, t.bfa, t.g):
        bits = buf.view(torch.int16)
        bits.copy_(torch.where(bits == _BF16_NEG_ZERO, 0, bits))


@pytest.mark.parametrize("producer", [False, True], ids=["consumer", "consumer_producer"])
@pytest.mark.parametrize("batch", [1, 4, 16, 33])
@torch.no_grad()
def test_decode_fgs_consumer_ready_input(batch: int, producer: bool) -> None:
    """Inputs written before launch: the same bits as the default (or producer) build and no reload."""
    t = _DecodeInputs(batch, seed=70 + batch)
    _canonical_inputs(t)
    ref, conv_ref, state_ref = _decode_fgs_run(t, fgs_producer=producer)
    stats = torch.zeros(2, device="cuda", dtype=torch.int32)
    out, conv, state = _decode_fgs_run(t, fgs_consumer=True, fgs_producer=producer, fgs_stats=stats)
    assert torch.equal(out, ref)
    assert torch.equal(state, state_ref)
    assert torch.equal(conv, conv_ref)
    assert stats.tolist() == [0, 0]


@torch.no_grad()
def test_decode_fgs_consumer_waits_for_late_inputs() -> None:
    """qkvg, g and bfa start as -0.0 and another stream publishes them after a delay. One row
    (96 CTAs) leaves SMs free for the writer; whether the consumer saw a sentinel depends on
    scheduling, so only the result is checked."""
    t = _DecodeInputs(1, seed=3)
    _canonical_inputs(t)
    ref, conv_ref, state_ref = _decode_fgs_run(t)
    final = (t.qkvg.clone(), t.g.clone(), t.bfa.clone())
    stats = torch.zeros(2, device="cuda", dtype=torch.int32)
    _decode_fgs_run(t, fgs_consumer=True, fgs_stats=stats)
    for buf in (t.qkvg, t.g, t.bfa):
        buf.fill_(-0.0)
    stats.zero_()
    writer = torch.cuda.Stream()
    writer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(writer):
        for buf, value in zip((t.g, t.bfa, t.qkvg), (final[1], final[2], final[0])):
            torch.cuda._sleep(100_000)
            buf.copy_(value)
    out, conv, state = _decode_fgs_run(t, fgs_consumer=True, fgs_stats=stats)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    assert torch.equal(state, state_ref)
    assert torch.equal(conv, conv_ref)
    assert stats[1].item() == 0


@torch.no_grad()
def test_decode_fgs_consumer_falls_back_to_grid_wait() -> None:
    """Chunks that stay -0.0 past the spin budget are taken as final after griddepcontrol.wait."""
    t = _DecodeInputs(4, seed=5)
    _canonical_inputs(t)
    t.qkvg.view(torch.int16)[0, 40] = _BF16_NEG_ZERO
    t.qkvg.view(torch.int16)[3, 3 * DIM + 24] = _BF16_NEG_ZERO
    t.g.view(torch.int16)[1, 136] = _BF16_NEG_ZERO
    t.bfa.view(torch.int16)[2, 128] = _BF16_NEG_ZERO
    ref, conv_ref, state_ref = _decode_fgs_run(t)
    stats = torch.zeros(2, device="cuda", dtype=torch.int32)
    out, conv, state = _decode_fgs_run(t, fgs_consumer=True, fgs_max_spins=4, fgs_stats=stats)
    assert torch.equal(out, ref)
    assert torch.equal(state, state_ref)
    assert torch.equal(conv, conv_ref)
    assert stats[1].item() > 0


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_layout", ["qkvg_slice", "contiguous"])
@pytest.mark.parametrize("tokens", [1, 3, 64, 1000])
@torch.no_grad()
def test_gated_rmsnorm_matches_triton(
    tokens: int, gate_layout: str, weight_dtype: torch.dtype
) -> None:
    torch.manual_seed(tokens)
    x = torch.randn(tokens * H, D, device="cuda", dtype=torch.bfloat16) * 2.0
    if gate_layout == "qkvg_slice":
        qkvg = torch.randn(tokens, 4 * DIM, device="cuda", dtype=torch.bfloat16) * 3.0
        z = qkvg[:, 3 * DIM :].view(tokens, H, D)
    else:
        z = torch.randn(tokens, H, D, device="cuda", dtype=torch.bfloat16) * 3.0
    w = (1.0 + 0.2 * torch.randn(D, device="cuda")).to(weight_dtype)
    stock = rms_norm_gated_token_major(x, z, w, EPS, gate_activation="sigmoid")
    out_buf = torch.full((tokens * H, 2 * D), 7.0, device="cuda", dtype=torch.bfloat16)
    cute = gated_rmsnorm.kda_gated_rmsnorm(x, z, w, EPS, out=out_buf[:, :D])
    xd = x.double()
    ref = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + EPS) * w.double()
    ref = ref * torch.sigmoid(z.reshape(x.shape).double())
    assert cute.data_ptr() == out_buf.data_ptr()
    assert (out_buf[:, D:] == 7.0).all()
    assert _rel(cute, ref) <= 1.05 * _rel(stock, ref)
    torch.testing.assert_close(cute.float(), stock.float(), rtol=2 * 2**-8, atol=1e-3)


def _canonical_bf16(t: torch.Tensor) -> torch.Tensor:
    bits = t.view(torch.int16)
    return torch.where(bits == _BF16_NEG_ZERO, 0, bits).view(torch.bfloat16)


def _norm_inputs(tokens: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    o = torch.randn(tokens * H, D, device="cuda", generator=gen).to(torch.bfloat16)
    qkvg = (torch.randn(tokens, 4 * DIM, device="cuda", generator=gen) * 3.0).to(torch.bfloat16)
    w = 1.0 + 0.2 * torch.randn(D, device="cuda", generator=gen)
    return _canonical_bf16(o), qkvg[:, 3 * DIM :].view(tokens, H, D), w


@pytest.mark.parametrize("ctas", [None, 8], ids=["full_grid", "8_ctas"])
@pytest.mark.parametrize("tokens", [1, 64, 1000])
@torch.no_grad()
def test_gated_rmsnorm_fgs_consumer_ready_input(tokens: int, ctas: int | None) -> None:
    """A fully written slot: same bits as the default build and no reload."""
    o, z, w = _norm_inputs(tokens, seed=tokens)
    ref = gated_rmsnorm.kda_gated_rmsnorm(o, z, w, EPS)
    stats = torch.zeros(2, device="cuda", dtype=torch.int32)
    out = gated_rmsnorm.kda_gated_rmsnorm(
        o, z, w, EPS, fgs_consumer=True, fgs_stats=stats, fgs_ctas=ctas
    )
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    assert stats.tolist() == [0, 0]


@pytest.mark.parametrize("ctas", [None, 8], ids=["full_grid", "8_ctas"])
@pytest.mark.parametrize("order", ["forward", "reverse"])
@torch.no_grad()
def test_gated_rmsnorm_fgs_consumer_waits_for_late_chunks(order: str, ctas: int | None) -> None:
    """The slot starts as -0.0 and another stream publishes it in token blocks after a delay.

    Whether the consumer reaches a chunk before its block lands depends on scheduling, so only
    the result is checked; test_gated_rmsnorm_fgs_consumer_falls_back_to_grid_wait covers reloads.
    """
    tokens, block = 64, 16
    o, z, w = _norm_inputs(tokens, seed=5)
    ref = gated_rmsnorm.kda_gated_rmsnorm(o, z, w, EPS)
    stats = torch.zeros(2, device="cuda", dtype=torch.int32)
    timeline = torch.tensor([2**63 - 1, 0], device="cuda", dtype=torch.int64)
    # Compile before the writer starts so the consumer launches while the writer still sleeps.
    gated_rmsnorm.kda_gated_rmsnorm(
        o, z, w, EPS, fgs_consumer=True, fgs_stats=stats, fgs_ctas=ctas, fgs_timeline=timeline
    )
    stats.zero_()
    timeline.copy_(torch.tensor([2**63 - 1, 0]))
    slot = torch.full_like(o, -0.0)
    writer = torch.cuda.Stream()
    writer.wait_stream(torch.cuda.current_stream())
    starts = list(range(0, tokens, block))
    if order == "reverse":
        starts.reverse()
    with torch.cuda.stream(writer):
        for start in starts:
            torch.cuda._sleep(200_000)
            rows = slice(start * H, (start + block) * H)
            slot[rows].copy_(o[rows])
    out = gated_rmsnorm.kda_gated_rmsnorm(
        slot, z, w, EPS, fgs_consumer=True, fgs_stats=stats, fgs_ctas=ctas, fgs_timeline=timeline
    )
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    assert stats[1].item() == 0
    assert 0 < timeline[0].item() < timeline[1].item()


@torch.no_grad()
def test_gated_rmsnorm_fgs_consumer_falls_back_to_grid_wait() -> None:
    """Chunks that stay -0.0 past the spin budget are taken as final after griddepcontrol.wait."""
    o, z, w = _norm_inputs(64, seed=9)
    o.view(torch.int16)[::5, ::8] = _BF16_NEG_ZERO
    ref = gated_rmsnorm.kda_gated_rmsnorm(o, z, w, EPS)
    stats = torch.zeros(2, device="cuda", dtype=torch.int32)
    out = gated_rmsnorm.kda_gated_rmsnorm(
        o, z, w, EPS, fgs_consumer=True, fgs_max_spins=4, fgs_stats=stats
    )
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    assert stats[1].item() > 0


@pytest.mark.parametrize("pad", [0, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_reset_state_rows_matches_triton(dtype: torch.dtype, pad: int) -> None:
    slots_total = 8
    base = torch.randn(slots_total, H * D * D + pad, device="cuda").to(dtype)
    pool0 = base[:, : H * D * D].view(slots_total, H, D, D)
    slots = torch.tensor([2, 0, 5, -1, 9, 1], device="cuda", dtype=torch.int32)
    has_init = torch.tensor([False, True, False, False, False, True], device="cuda")
    stock = torch.empty_strided(pool0.shape, pool0.stride(), dtype=dtype, device="cuda").copy_(
        pool0
    )
    cute = torch.empty_strided(pool0.shape, pool0.stride(), dtype=dtype, device="cuda").copy_(pool0)
    reset_recurrent_state_rows(stock, slots, has_init)
    assert reset_state_rows.supports(cute, slots, has_init)
    reset_state_rows.reset_state_rows(cute, slots, has_init)
    assert torch.equal(stock, cute)


def _decode_module():
    cfg = SimpleNamespace(
        hidden_size=7168,
        rms_norm_eps=EPS,
        linear_attn_config={
            "num_heads": H,
            "head_dim": D,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": LOWER_BOUND,
        },
    )
    module = kimi_kda_mixer.KimiKDALinearAttention(cfg, layer_idx=0).to("cuda")
    with torch.no_grad():
        for param in module.parameters():
            if param.dim() >= 2:
                param.normal_(0.0, 0.02)
    module.finalize_decode_weights()
    return module


@pytest.mark.parametrize("batch", [4, 33])
@torch.no_grad()
def test_mixer_decode_switch_reaches_cute_kernel(
    monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    """``forward_decode`` with the ``decode`` part takes the CuTe DSL kernel on a slot-padded pool."""
    torch.manual_seed(batch)
    stock_module = _decode_module()
    # The CuTe module's FP32 conv weights are built at finalize time, under the switch.
    monkeypatch.setattr(kimi_kda_mixer, "_CUTE_KDA", frozenset({"decode"}))
    torch.manual_seed(batch)
    cute_module = _decode_module()
    if cute_module._dispatch.decode_kernel_path != "optimized":
        pytest.skip("optimized KDA decode is not available on this GPU")

    slots_total = batch + 5
    hidden = torch.randn(batch, 7168, device="cuda", dtype=torch.bfloat16) * 0.05
    conv0 = torch.randn(slots_total, 3 * DIM, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    state0 = torch.randn(slots_total, H, D, D, device="cuda") * 0.05
    slots = torch.randperm(slots_total, device="cuda")[:batch].to(torch.int32)
    metadata = SimpleNamespace(
        _arange_buffer=torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    )

    def run(module, monkey_parts):
        monkeypatch.setattr(kimi_kda_mixer, "_CUTE_KDA", monkey_parts)
        conv, state = _padded(conv0, 4), _padded(state0, 64)
        core = module.forward_decode(
            hidden, conv, state, slots.long(), metadata, ssm_state_indices=slots
        )
        return core.clone(), conv, state

    core_s, conv_s, state_s = run(stock_module, frozenset())
    kimi_kda_mixer.CUTE_KDA_PATHS.clear()
    core_c, conv_c, state_c = run(cute_module, frozenset({"decode"}))
    assert kimi_kda_mixer.CUTE_KDA_PATHS["decode.cute"] == 1
    assert kimi_kda_mixer.CUTE_KDA_PATHS["decode.stock"] == 0

    sl = slots.long()
    keep = torch.ones(slots_total, dtype=torch.bool, device="cuda")
    keep[sl] = False
    assert torch.equal(conv_c, conv_s)
    assert torch.equal(state_c[keep], state0[keep])
    assert _rel(core_c, core_s) < 1e-2
    assert _rel(state_c[sl], state_s[sl]) < 1e-5
