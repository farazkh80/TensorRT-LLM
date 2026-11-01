# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity of the CuTe DSL attention-residual mix with the stock ops it replaces.

Every case runs ``attn_res_mix`` at the dispatch default for its token count and the stock path on identical
inputs, and requires the CuTe error against an fp64 reference to be no worse than stock's (relative L2 within
1.1x, max abs within 2x). Stock is ``attn_res_fwd`` for the mix, the C++ ``attn_res_[add_]rmsnorm_fwd`` for the
KimiK3RMSNorm rounding, and ATen add + ``attn_res_fwd`` + ``RMSNorm`` for the FlashInfer rounding.
"""

import functools

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_k3_attn_res.attn_res_mix_cute import (
    attn_res_mix,
    default_config,
    is_attn_res_mix_cute_available,
    make_fgs_counters,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op
except ImportError:
    cutlass = None

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


def _fgs_call(x, attn, add, counters, **kwargs):
    out = torch.empty_like(x["prefix"])
    updated = torch.empty_like(x["prefix"]) if add else None
    return attn_res_mix(
        x["prefix"] if add else attn,
        x["bank"],
        x["res_w"],
        x["rms_w"],
        EPS,
        addend=attn if add else None,
        out_norm_weight=x["out_w"],
        out_norm_eps=EPS,
        out=out,
        updated=updated,
        fgs_consumer=True,
        fgs_counters=counters,
        **kwargs,
    )


def _default_call(x, attn, add, **kwargs):
    return attn_res_mix(
        x["prefix"] if add else attn,
        x["bank"],
        x["res_w"],
        x["rms_w"],
        EPS,
        addend=attn if add else None,
        out_norm_weight=x["out_w"],
        out_norm_eps=EPS,
        **kwargs,
    )


def _as_tuple(result):
    return result if isinstance(result, tuple) else (result,)


if cutlass is not None:

    @dsl_user_op
    def _globaltimer(*, loc=None, ip=None):
        return Int64(
            llvm.inline_asm(
                T.i64(),
                [],
                "mov.u64 $0, %globaltimer;",
                "=l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    class _LatePublisher:
        """Stand-in FGS producer: CTA j copies block ``order[j]`` of ``src`` to ``dst`` (j + 1) * ``delay_ns`` after it
        starts, in whole 16-B chunks. It triggers its dependents at entry, so a consumer launched after it with PDL
        starts only once every publishing CTA is resident."""

        def __init__(self, block_rows: int):
            self.block_rows = block_rows

        @cute.jit
        def __call__(
            self, src: cute.Tensor, dst: cute.Tensor, order: cute.Tensor, delay_ns: Int32, stream
        ):
            vec = (1, 8)
            self.kernel(
                cute.zipped_divide(src, vec), cute.zipped_divide(dst, vec), order, delay_ns
            ).launch(grid=[cute.size(order, mode=[0]), 1, 1], block=[256, 1, 1], stream=stream)

        @cute.kernel
        def kernel(self, gS: cute.Tensor, gD: cute.Tensor, order: cute.Tensor, delay_ns: Int32):
            cute.arch.griddepcontrol_launch_dependents()
            tidx, _, _ = cute.arch.thread_idx()
            j, _, _ = cute.arch.block_idx()
            due = _globaltimer() + Int64(delay_ns) * Int64(j + 1)
            now = _globaltimer()
            while now < due:
                now = _globaltimer()
            cols = HIDDEN_SIZE // 8
            first = order[j] * self.block_rows
            for i in cutlass.range(tidx, self.block_rows * cols, 256, unroll=1):
                gD[(None, (first + i // cols, i % cols))].store(
                    gS[(None, (first + i // cols, i % cols))].load()
                )


@functools.lru_cache(maxsize=None)
def _publisher(block_rows: int):
    act = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), HIDDEN_SIZE), stride_order=(1, 0), assumed_align=16
    )
    order = cute.runtime.make_fake_compact_tensor(Int32, (cute.sym_int(),))
    return cute.compile(
        _LatePublisher(block_rows),
        act,
        act,
        order,
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@pytest.mark.parametrize("add", [True, False], ids=["add_norm", "snapshot_norm"])
@pytest.mark.parametrize(
    "num_tokens,num_candidates", [(1, 2), (16, 9), (64, 5), (300, 2), (300, 7), (8192, 9)]
)
@torch.no_grad()
def test_fgs_consumer_ready_input_matches_default(num_tokens, num_candidates, add) -> None:
    x = _inputs(num_tokens, num_candidates, seed=3)
    attn = x["addend"] if add else x["prefix"]
    counters = make_fgs_counters()
    got = _fgs_call(x, attn, add, counters)
    ref = _default_call(x, attn, add)
    for a, b in zip(_as_tuple(got), _as_tuple(ref)):
        assert torch.equal(a, b)
    assert counters.tolist() == [0] * 4, "ready input must never be re-read"


@pytest.mark.parametrize("order", ["forward", "reverse", "random"])
@pytest.mark.parametrize("add", [True, False], ids=["add_norm", "snapshot_norm"])
@pytest.mark.parametrize("num_tokens,block", [(64, 8), (1024, 128)], ids=["reg", "pipe"])
@torch.no_grad()
def test_fgs_consumer_late_rows_match_default(num_tokens: int, block: int, add: bool, order: str):
    """The attention output starts as -0.0 and a stand-in producer publishes it block by block.

    Whether the consumer reaches a row before it lands depends on scheduling, so only the result is checked;
    test_fgs_consumer_falls_back_to_grid_wait covers re-reads deterministically.
    """
    x = _inputs(num_tokens, 3, seed=4)
    attn = x["addend"] if add else x["prefix"]
    ref = _default_call(x, attn, add)
    _fgs_call(x, attn, add, make_fgs_counters())
    blocks = list(range(num_tokens // block))
    if order == "reverse":
        blocks.reverse()
    elif order == "random":
        blocks = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(0)).tolist()
    buf = torch.empty_like(attn)
    buf.view(torch.int16).fill_(-32768)
    counters = make_fgs_counters()
    publish = _publisher(block)
    publish(attn, buf, torch.tensor(blocks, dtype=torch.int32, device="cuda"), 20_000)
    got = _fgs_call(x, buf, add, counters)
    torch.cuda.synchronize()
    for a, b in zip(_as_tuple(got), _as_tuple(ref)):
        assert torch.equal(a, b)
    assert counters[2].item() == 0, counters.tolist()


@pytest.mark.parametrize("add", [True, False], ids=["add_norm", "snapshot_norm"])
@pytest.mark.parametrize("num_tokens", [64, 1024], ids=["reg", "pipe"])
@torch.no_grad()
def test_fgs_consumer_falls_back_to_grid_wait(num_tokens: int, add: bool) -> None:
    """Chunks that keep a -0.0 past the spin budget are re-read, then taken as final after the PDL wait."""
    x = _inputs(num_tokens, 3, seed=9)
    attn = x["addend"] if add else x["prefix"]
    attn.view(torch.int16)[::5, ::8] = -32768
    ref = _default_call(x, attn, add)
    counters = make_fgs_counters()
    got = _fgs_call(x, attn, add, counters, fgs_max_spins=4)
    torch.cuda.synchronize()
    for a, b in zip(_as_tuple(got), _as_tuple(ref)):
        assert torch.equal(a, b)
    waits, rounds, fallbacks, _ = counters.tolist()
    assert waits > 0 and fallbacks == waits and rounds == 5 * waits, counters.tolist()


@pytest.mark.parametrize("mode", ["graph", "eager"])
@pytest.mark.parametrize("num_tokens", [64, 1024], ids=["reg", "pipe"])
@torch.no_grad()
def test_fgs_consumer_back_to_back_calls(num_tokens: int, mode: str) -> None:
    """24 consumers back to back: without the PDL wait each may start while the previous one still runs."""
    calls, replays = 24, 3
    inputs = [_inputs(num_tokens, 3, seed=10 + i) for i in range(2)]
    refs = [_default_call(x, x["addend"], True) for x in inputs]
    outs = [
        (torch.empty_like(inputs[0]["prefix"]), torch.empty_like(inputs[0]["prefix"]))
        for _ in range(calls)
    ]
    counters = make_fgs_counters()

    def run():
        for i, (updated, out) in enumerate(outs):
            x = inputs[i % 2]
            attn_res_mix(
                x["prefix"],
                x["bank"],
                x["res_w"],
                x["rms_w"],
                EPS,
                addend=x["addend"],
                out_norm_weight=x["out_w"],
                out_norm_eps=EPS,
                out=out,
                updated=updated,
                fgs_consumer=True,
                fgs_counters=counters,
            )

    run()
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    for _ in range(replays):
        for pair in outs:
            for t in pair:
                t.fill_(float("nan"))
        if mode == "graph":
            graph.replay()
        else:
            run()
        torch.cuda.synchronize()
        for i, got in enumerate(outs):
            for a, b in zip(got, refs[i % 2]):
                assert torch.equal(a, b), f"call {i}"
    assert counters.tolist() == [0] * 4


@pytest.mark.parametrize("norm", [True, False], ids=["norm", "mix"])
@pytest.mark.parametrize("num_tokens", [1, 16, 64])
@torch.no_grad()
def test_fgs_producer_matches_default_without_negative_zero(num_tokens: int, norm: bool) -> None:
    """Columns where every input is -0.0 come out as -0.0 by default; the producer build writes +0.0."""
    x = _inputs(num_tokens, 4, seed=6)
    for key in ("prefix", "addend"):
        x[key][:, :64] = -0.0
    x["bank"][:, :, :64] = -0.0
    kwargs = dict(out_norm_weight=x["out_w"] if norm else None)
    args = (x["prefix"], x["bank"], x["res_w"], x["rms_w"], EPS)
    ref = attn_res_mix(*args, addend=x["addend"], out_norm_eps=EPS, **kwargs)
    got = attn_res_mix(*args, addend=x["addend"], out_norm_eps=EPS, fgs_producer=True, **kwargs)
    for a, b in zip(got, ref):
        assert torch.equal(a, b)
        assert (b.view(torch.int16) == -32768).any(), "the inputs must produce -0.0 by default"
        assert not (a.view(torch.int16) == -32768).any()


@pytest.mark.parametrize("num_tokens", [16, 300, 8192])
@torch.no_grad()
def test_fgs_build_leaves_default_dispatch_intact(num_tokens: int) -> None:
    """The FGS builds have their own host entries; the default one must keep its argument list and results."""
    x = _inputs(num_tokens, 5, seed=5)
    _fgs_call(x, x["addend"], True, make_fgs_counters())
    if num_tokens <= 64:
        _default_call(x, x["addend"], True, fgs_producer=True)
    updated, out = attn_res_mix(
        x["prefix"],
        x["bank"],
        x["res_w"],
        x["rms_w"],
        EPS,
        addend=x["addend"],
        out_norm_weight=x["out_w"],
        out_norm_eps=EPS,
    )
    prefix = x["prefix"] + x["addend"]
    assert torch.equal(updated, prefix)
    ref = _ref_norm(
        _ref_mix(prefix, x["bank"], x["res_w"], x["rms_w"]).to(torch.bfloat16), x["out_w"]
    )
    _assert_no_worse(
        out, _stock_mix_norm(x, True, "flashinfer"), ref, f"default {default_config(num_tokens)}"
    )
