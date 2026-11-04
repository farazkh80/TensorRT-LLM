# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FGS producer build of the SM107 BF16 persistent GEMM at the prefill o_proj shape."""

import importlib.util
import os
import sys

import cutlass.cute as cute
import pytest
import torch

if os.environ.get("FGS_STANDALONE") == "1":
    # Copy next to this file, loaded under its package name so its relative imports resolve.
    _NAME = "tensorrt_llm._torch.modules.kimi_k3_fgs.dense_gemm_fgs"
    _spec = importlib.util.spec_from_file_location(
        _NAME, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dense_gemm_fgs.py")
    )
    fgs = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = fgs
    _spec.loader.exec_module(fgs)
else:
    from tensorrt_llm._torch.modules.kimi_k3_fgs import dense_gemm_fgs as fgs

from tensorrt_llm._torch.modules import kimi_k3_cute_gemm

N, K = 7168, 12288
NEG0 = -32768
NEG0_ROWS, NEG0_COLS = 8, 16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or not hasattr(cute.arch, "mbarrier_v1_init"),
    reason="the .pzo convert needs SM107 and CuTe DSL 4.9",
)


def _inputs(m: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, K, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    # -0.0 rows times positive weight rows: these outputs are -0.0 without .pzo.
    a[:NEG0_ROWS] = -0.0
    w[:NEG0_COLS] = w[:NEG0_COLS].abs()
    return a, w


def _canonical(t: torch.Tensor) -> torch.Tensor:
    bits = t.view(torch.int16).clone()
    bits[bits == NEG0] = 0
    return bits


def _run(a, w, **flags):
    out = torch.full((a.shape[0], N), float("nan"), device="cuda", dtype=torch.bfloat16)
    fgs.run_dense_fgs(a, w, out, **flags)
    return out


STOCK = dict(pzo=False, early_trigger=False, raster_along="m")


@pytest.mark.parametrize("m", [1000, 8192])
def test_stock_build_matches_table_gemm(m: int) -> None:
    a, w = _inputs(m)
    out = _run(a, w, **STOCK)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), kimi_k3_cute_gemm.linear(a, w).view(torch.int16))


@pytest.mark.parametrize(
    "flags",
    [
        dict(pzo=True, early_trigger=False, raster_along="m"),
        dict(pzo=False, early_trigger=True, raster_along="m"),
        dict(pzo=False, early_trigger=False, raster_along="n"),
        dict(pzo=False, early_trigger=False, raster_along="n", swizzle_size=4),
        dict(pzo=True, early_trigger=True, raster_along="n"),
    ],
    ids=["pzo", "early", "raster_n", "raster_n_s4", "fgs"],
)
@pytest.mark.parametrize("m", [1000, 8192])
def test_flags_change_only_negative_zero(m: int, flags: dict) -> None:
    a, w = _inputs(m)
    ref = _run(a, w, **STOCK)
    out = _run(a, w, **flags)
    torch.cuda.synchronize()
    assert torch.equal(_canonical(out), _canonical(ref))
    if flags["pzo"]:
        assert not bool((out.view(torch.int16) == NEG0).any())
    else:
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("raster_along", ["m", "n"])
def test_timeline(raster_along: str) -> None:
    m = 8192
    a, w = _inputs(m)
    stamps = torch.zeros(fgs.timeline_slots(m, N), dtype=torch.int32, device="cuda")
    out = _run(a, w, pzo=True, early_trigger=True, raster_along=raster_along, stamps=stamps)
    torch.cuda.synchronize()
    assert torch.equal(_canonical(out), _canonical(_run(a, w, **STOCK)))
    cta_m = fgs.O_PROJ_TACTIC[2][0] // 2
    rows, cols = m // cta_m, N // fgs.O_PROJ_TACTIC[2][1]
    tiles = stamps[: rows * cols].to(torch.int64).view(rows, cols)
    starts = stamps[rows * cols :].to(torch.int64)
    starts = starts[starts != 0]
    assert bool((tiles != 0).all()) and starts.numel() > 0
    base = starts.min()
    row_ready = ((tiles - base) % (1 << 32)).max(dim=1).values.double()
    quarter = rows // 4
    first, last = row_ready[:quarter].median(), row_ready[-quarter:].median()
    if raster_along == "n":
        # Row blocks complete in row order: the first quarter well before the last.
        assert first < 0.5 * last


QKVG_N, QKVG_K, QKVG_HEADS = 49152, 7168, 96


@pytest.mark.parametrize("m", [16, 64])
@pytest.mark.parametrize("h1", [1, 70, 96])
def test_qkvg_head_order_is_bit_exact(m: int, h1: int) -> None:
    """Head-ordered tiles (producer and consumer + producer builds) equal the table GEMM up to -0.0."""
    g = torch.Generator(device="cuda").manual_seed(3)
    a = torch.randn(m, QKVG_K, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(QKVG_N, QKVG_K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    tactic = kimi_k3_cute_gemm._select(kimi_k3_cute_gemm._GEMM_TABLE[(QKVG_N, QKVG_K)], m)[1]
    ref = _canonical(kimi_k3_cute_gemm.linear(a, w))
    for consume in (False, True):
        out = torch.full((m, QKVG_N), float("nan"), device="cuda", dtype=torch.bfloat16)
        fgs.run_gemm_fgs(
            a.unsqueeze(0),
            w.unsqueeze(0),
            out.unsqueeze(0),
            tactic,
            report_valid=consume,
            retry_warp=consume,
            wait_input=not consume,
            pzo=True,
            early_trigger=True,
            n_order=(QKVG_HEADS, h1),
        )
        torch.cuda.synchronize()
        assert torch.equal(_canonical(out), ref), f"consume={consume}"


def test_qkvg_head_order_stores_heads_first() -> None:
    """With h1 = 70 the q / k / v tiles of heads 0..69 are stored before the gate tiles of heads 2..95."""
    m = 16
    g = torch.Generator(device="cuda").manual_seed(4)
    a = torch.randn(m, QKVG_K, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(QKVG_N, QKVG_K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    tactic = kimi_k3_cute_gemm._select(kimi_k3_cute_gemm._GEMM_TABLE[(QKVG_N, QKVG_K)], m)[1]
    tiles_n = QKVG_N // tactic[2][1]
    stamps = torch.zeros(fgs.timeline_slots(m, QKVG_N, tactic), dtype=torch.int32, device="cuda")
    out = torch.empty(m, QKVG_N, device="cuda", dtype=torch.bfloat16)
    fgs.run_dense_fgs(
        a, w, out, raster_along="m", tactic=tactic, stamps=stamps, n_order=(QKVG_HEADS, 70)
    )
    torch.cuda.synchronize()
    assert torch.equal(_canonical(out), _canonical(kimi_k3_cute_gemm.linear(a, w)))
    starts = stamps[tiles_n:].to(torch.int64)
    base = starts[starts != 0].min()
    t = ((stamps[:tiles_n].to(torch.int64) - base) % (1 << 32)).view(4, QKVG_HEADS)
    if starts[starts != 0].numel() < 3 * 70:
        pytest.skip("fewer persistent CTAs than the 210 head-ordered tiles of round 1")
    assert t[:3, :70].max() < t[3, 2:].min()
