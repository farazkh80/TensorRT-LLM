# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TLLM_K3_FGS edges: slot lease -> kimi_k3_cute_gemm.linear consumer -> release, eager and under a graph."""

import threading

import cutlass.cute as cute
import pytest
import torch

from tensorrt_llm._torch.modules import fine_grained_sync, kimi_k3_cute_gemm
from tensorrt_llm._torch.modules.kimi_k3_fgs import edges

N, K = 7168, 12288
EDGE = "test.core_o"
LAYERS = 3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or not hasattr(cute.arch, "mbarrier_v1_init")
    or not hasattr(fine_grained_sync, "EdgeWorkspaces"),
    reason="needs SM107, a CuTe DSL with v1 mbarriers and EdgeWorkspaces",
)


@pytest.fixture
def lamport(monkeypatch):
    monkeypatch.setenv(edges.ENV, "lamport")
    monkeypatch.setenv("TLLM_K3_CUTE_GEMM", "1")
    monkeypatch.setattr(edges, "_declared", {})
    monkeypatch.setattr(edges, "_local", threading.local())
    for layer_idx in range(LAYERS):
        edges.declare(EDGE, K, layer_idx)
    yield


def _inputs(m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16, generator=g)
    # -0.0 is the slot sentinel; producers never store it.
    x.masked_fill_(x == 0, 0.5)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.05
    return x, w


def _layer(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> bool:
    lease = edges.acquire(EDGE, x.shape[0])
    src = x if lease is None else lease.tensor.copy_(x)
    kimi_k3_cute_gemm.linear(src, w, out)
    return lease is not None


def _assert_close(out: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> None:
    ref = x.float() @ w.float().t()
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-2


def _assert_armed() -> None:
    torch.cuda.synchronize()
    for slot in edges._state()[0][EDGE].slots:
        assert bool((slot.view(torch.int16) == -0x8000).all()), "slot not re-armed"


@pytest.mark.parametrize("m", [16, 32, 64])
def test_lease_consume_release_eager(lamport, monkeypatch, m: int) -> None:
    monkeypatch.setattr(edges, "MAX_ROWS", 64)
    x, w = _inputs(m, m)
    out = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
    with edges.model_scope():
        assert _layer(x, w, out)
        assert not edges._local.live
    _assert_close(out, x, w)
    if m == 16:
        # Same tile and split as the table entry at M = 16: bit-exact.
        assert torch.equal(out, kimi_k3_cute_gemm.linear(x, w))
    _assert_armed()


@pytest.mark.parametrize("m", [1, 4, 17])
def test_rows_outside_window_keep_own_buffer(lamport, m: int) -> None:
    x, w = _inputs(m, m)
    out = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
    with edges.model_scope():
        assert not _layer(x, w, out)
    _assert_close(out, x, w)


def test_off_is_inert(monkeypatch) -> None:
    monkeypatch.setenv(edges.ENV, "off")
    monkeypatch.setattr(edges, "_declared", {})
    monkeypatch.setattr(edges, "_local", threading.local())
    edges.declare(EDGE, K, 0)
    assert not edges._declared
    with edges.model_scope():
        assert edges.acquire(EDGE, 16) is None
    assert edges.ledger() is None


def test_unreleased_slot_raises(lamport) -> None:
    with pytest.raises(RuntimeError, match="unreleased"):
        with edges.model_scope():
            assert edges.acquire(EDGE, 16) is not None


@pytest.mark.parametrize("policy", ["release", "step"])
def test_graph_replays(lamport, monkeypatch, policy: str) -> None:
    monkeypatch.setenv("TLLM_FGS_RESET_POLICY", policy)
    m = 16
    x_static, w = _inputs(m, 0)
    outs = [torch.empty(m, N, device="cuda", dtype=torch.bfloat16) for _ in range(LAYERS)]

    def forward():
        with edges.model_scope():
            for out in outs:
                assert _layer(x_static, w, out)

    # Eager forward registers the edge; the capture then only leases and releases.
    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for seed in range(1, 4):
        x, _ = _inputs(m, seed)
        x_static.copy_(x)
        for out in outs:
            out.zero_()
        graph.replay()
        ref = kimi_k3_cute_gemm.linear(x, w)
        for out in outs:
            assert torch.equal(out, ref)
    _assert_armed()


def test_slot_views_route_to_fgs_builds(lamport) -> None:
    """bfa (split-K producer build) writes a slot; two f_b readers (in-tree consumer build on a strided view) consume
    it back to back with PDL; the second reader releases and re-arms the slot."""
    from tensorrt_llm._torch.modules.kimi_k3_fgs import dense_gemm_fgs

    edge, m, k_in, width, n_fb = "test.bfa", 16, 7168, 224, 12288
    edges.declare(edge, width, 0)
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(m, k_in, device="cuda", dtype=torch.bfloat16, generator=g)
    x.masked_fill_(x == 0, 0.5)
    w_bfa = torch.randn(width, k_in, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w_fb = torch.randn(n_fb, 128, device="cuda", dtype=torch.bfloat16, generator=g) * 0.05
    outs = [torch.empty(m, n_fb, device="cuda", dtype=torch.bfloat16) for _ in range(2)]

    with edges.model_scope():
        lease = edges.acquire(edge, m, readers=2)
        assert lease is not None
        kimi_k3_cute_gemm.linear(x, w_bfa, lease.tensor)
        kimi_k3_cute_gemm.linear(lease.tensor[:, :128], w_fb, outs[0])
        assert edges._local.live
        kimi_k3_cute_gemm.linear(lease.tensor[:, :128], w_fb, outs[1])
        assert not edges._local.live

    bfa = kimi_k3_cute_gemm.linear(x, w_bfa)
    bfa.view(torch.int16).masked_fill_(bfa.view(torch.int16) == -0x8000, 0)
    ref = torch.empty_like(outs[0])
    dense_gemm_fgs.run_gemm_fgs(
        bfa[:, :128].unsqueeze(0),
        w_fb.unsqueeze(0),
        ref.unsqueeze(0),
        kimi_k3_cute_gemm._FGS_SMALL_TACTIC,
        report_valid=False,
        retry_warp=False,
        wait_input=True,
    )
    torch.cuda.synchronize()
    for out in outs:
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    for slot in edges._state()[0][edge].slots:
        assert bool((slot.view(torch.int16) == -0x8000).all()), "slot not re-armed"
