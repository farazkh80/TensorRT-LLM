# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FGS split-K GEMM consumer (decode o_proj shape) on SM107: ready inputs and one late-published chunk."""

import os

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

if os.environ.get("FGS_STANDALONE") == "1":
    import splitk_gemm_fgs as fgs
else:
    from tensorrt_llm._torch.modules.kimi_k3_fgs import splitk_gemm_fgs as fgs

N, K = 7168, 12288
NEG0 = -32768
TACTICS = {
    1: fgs.SplitKTactic(128, 8, 3, 6),
    4: fgs.SplitKTactic(128, 8, 3, 6),
    16: fgs.SplitKTactic(64, 16, 3, 10),
    32: fgs.SplitKTactic(64, 32, 3, 8),
}
VARIANTS = {
    "plain": dict(report_valid=False, retry_warp=False, pdl_wait_activation=True),
    "skeleton": dict(report_valid=False, retry_warp=True, pdl_wait_activation=True),
    "fgs": dict(report_valid=True, retry_warp=True, pdl_wait_activation=False),
}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or not hasattr(cute.arch, "mbarrier_v1_init"),
    reason="TMA report_valid needs SM107 and a CuTe DSL with v1 mbarriers",
)


def _inputs(m: int, seed: int):
    torch.manual_seed(seed)
    x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
    x[x == 0] = 1.0
    w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    return x, w


def _run(x, w, variant: str, counter=None, pdl=True):
    out = torch.full((x.shape[0], N), float("nan"), device="cuda", dtype=torch.bfloat16)
    fgs.run_splitk_dense_fgs(
        x, w.t(), None, out, pdl, TACTICS[x.shape[0]], retry_counter=counter, **VARIANTS[variant]
    )
    return out


@pytest.mark.parametrize("m", [1, 4, 16, 32])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_ready_inputs_match_plain(m: int, variant: str) -> None:
    x, w = _inputs(m, seed=m)
    counter = torch.zeros(fgs.RETRY_COUNTER_SLOTS, dtype=torch.int32, device="cuda")
    out = _run(x, w, variant, counter)
    plain = _run(x, w, "plain")
    torch.cuda.synchronize()
    ref = x.float() @ w.float().t()
    assert bool(torch.isfinite(out.float()).all()) and out.abs().max().item() > 0
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-2
    assert torch.equal(out.view(torch.int16), plain.view(torch.int16))
    assert counter.tolist() == [0, 0]


@pytest.mark.parametrize("stages", [2, 6, 10])
def test_coresident_consumer_matches_plain(stages: int) -> None:
    """Fewer stages and a tactic-sized smem request change neither the result nor the retry count."""
    x, w = _inputs(16, seed=stages)
    counter = torch.zeros(fgs.RETRY_COUNTER_SLOTS, dtype=torch.int32, device="cuda")
    out = torch.full((16, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    fgs.run_splitk_dense_fgs(
        x,
        w.t(),
        None,
        out,
        True,
        fgs.SplitKTactic(64, 16, 3, stages),
        retry_counter=counter,
        exact_smem=True,
        smem_carveout=100,
        **VARIANTS["fgs"],
    )
    plain = _run(x, w, "plain")
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), plain.view(torch.int16))
    assert counter.tolist() == [0, 0]


def test_timeline_stamps_ready_inputs() -> None:
    """The timeline build gives the same output and per-CTA stamps start <= first valid <= last valid."""
    x, w = _inputs(16, seed=5)
    tactic = TACTICS[16]
    counter = torch.zeros(fgs.timeline_slots(tactic, 16, N), dtype=torch.int32, device="cuda")
    out = torch.full((16, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    fgs.run_splitk_dense_fgs(
        x,
        w.t(),
        None,
        out,
        True,
        tactic,
        retry_counter=counter,
        timeline=True,
        **VARIANTS["fgs"],
    )
    plain = _run(x, w, "plain")
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), plain.view(torch.int16))
    assert counter[:2].tolist() == [0, 0]
    stamps = counter[2:].view(-1, fgs.TIMELINE_FIELDS).long()
    # Differences of 32-bit %globaltimer_lo values, unwrapped.
    first = (stamps[:, 1] - stamps[:, 0]) % (1 << 32)
    last = (stamps[:, 3] - stamps[:, 1]) % (1 << 32)
    assert bool((first < 1_000_000).all()) and bool((last < 1_000_000).all())
    assert bool((stamps[:, 2] == 0).all()) and bool((stamps[:, 4] == 0).all())


@dsl_user_op
def _copy16(dst: Int64, src: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [dst.ir_value(loc=loc, ip=ip), src.ir_value(loc=loc, ip=ip)],
        "{\n.reg .b32 r<4>;\nld.global.v4.b32 {r0, r1, r2, r3}, [$1];\n"
        "st.global.v4.b32 [$0], {r0, r1, r2, r3};\n}",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_relaxed_u32(addr: Int64, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.kernel
def _late_copy16_kernel(dst: cute.Tensor, src: cute.Tensor, flag: cute.Tensor, delay_ns: Int64):
    while _ld_relaxed_u32(flag.iterator.toint()) == 0:
        cute.arch.nanosleep(sleep_time=500)
    start = cute.arch.globaltimer()
    while cute.arch.globaltimer() < start + delay_ns:
        cute.arch.nanosleep(sleep_time=500)
    if cute.arch.thread_idx()[0] == 0:
        _copy16(dst.iterator.toint(), src.iterator.toint())
        cute.arch.fence_acq_rel_gpu()


@cute.jit
def _late_copy16(dst: cute.Tensor, src: cute.Tensor, flag: cute.Tensor, delay_ns: Int64, stream):
    _late_copy16_kernel(dst, src, flag, delay_ns).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream
    )


def _late_publish(dst: torch.Tensor, src: torch.Tensor, delay_ns: int) -> torch.Tensor:
    """Start a 1-CTA writer on its own stream that copies the 16 B chunk ``src`` into ``dst`` ``delay_ns`` after the
    returned flag turns non-zero. Set the flag on the consumer's stream right before the consumer, so the consumer
    reads the sentinel first however slow the host launch path is."""
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    args = [from_dlpack(t, assumed_align=16 if t is not flag else 4) for t in (dst, src, flag)]
    s_prod = torch.cuda.Stream()
    with torch.cuda.stream(s_prod):
        stream = cuda.CUstream(s_prod.cuda_stream)
        writer = cute.compile(_late_copy16, *args, cutlass.Int64(0), stream)
        torch.cuda.synchronize()
        writer(*args, cutlass.Int64(delay_ns), stream)
    return flag


def test_retry_reissues_late_chunk() -> None:
    """A 16 B -0.0 chunk, published from another stream after the consumer started, is retried, not consumed."""
    m = 16
    x, w = _inputs(m, seed=3)
    expected = _run(x, w, "plain")
    counter = torch.zeros(fgs.RETRY_COUNTER_SLOTS, dtype=torch.int32, device="cuda")
    _run(x, w, "fgs", counter)  # JIT before the producer starts its delay
    slot = x.clone()
    row, col = m // 2, K // 2 + 8
    slot.view(torch.int16)[row, col : col + 8] = NEG0
    counter.zero_()
    flag = _late_publish(slot[row, col : col + 8], x[row, col : col + 8], 500_000)
    s_cons = torch.cuda.Stream()
    with torch.cuda.stream(s_cons):
        flag.fill_(1)
        out = _run(slot, w, "fgs", counter)
    torch.cuda.synchronize()
    reissues, unchecked = counter.tolist()
    assert reissues > 0 and unchecked == 0
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


# The other decode split-K uses of kimi_k3_cute_gemm at their table tactics: (N, K) -> rows -> tactic.
DECODE_SPLITK = {
    "bfa": (224, 7168, {8: (64, 8, 4, 12), 16: (64, 8, 4, 11), 32: (64, 8, 4, 11)}),
    "qa_kva": (2112, 7168, {1: (64, 8, 4, 11), 16: (64, 16, 4, 10), 32: (64, 32, 4, 8)}),
    "qb": (18432, 1536, {1: (128, 8, 1, 6), 16: (128, 16, 1, 6), 32: (128, 32, 1, 5)}),
    "gproj": (12288, 7168, {4: (128, 8, 2, 6), 16: (64, 16, 1, 9), 32: (64, 32, 1, 9)}),
}


@pytest.mark.parametrize(
    "shape,m", [(s, m) for s, (_, _, tactics) in DECODE_SPLITK.items() for m in tactics]
)
def test_decode_shapes_ready_and_late_chunk(shape: str, m: int) -> None:
    """Ready input equals the plain build; one 16 B -0.0 chunk published late is retried, not consumed."""
    n, k, tactics = DECODE_SPLITK[shape]
    tactic = fgs.SplitKTactic(*tactics[m])
    torch.manual_seed(m)
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    x[x == 0] = 1.0
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)

    def run(a, variant, counter=None):
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        fgs.run_splitk_dense_fgs(
            a, w.t(), None, out, True, tactic, retry_counter=counter, **VARIANTS[variant]
        )
        return out

    plain = run(x, "plain")
    counter = torch.zeros(fgs.RETRY_COUNTER_SLOTS, dtype=torch.int32, device="cuda")
    ready = run(x, "fgs", counter)
    torch.cuda.synchronize()
    assert torch.equal(ready.view(torch.int16), plain.view(torch.int16))
    assert counter.tolist() == [0, 0]

    slot = x.clone()
    row, col = m // 2, k // 2 + 8
    slot.view(torch.int16)[row, col : col + 8] = NEG0
    counter.zero_()
    flag = _late_publish(slot[row, col : col + 8], x[row, col : col + 8], 500_000)
    s_cons = torch.cuda.Stream()
    with torch.cuda.stream(s_cons):
        flag.fill_(1)
        out = run(slot, "fgs", counter)
    torch.cuda.synchronize()
    reissues, unchecked = counter.tolist()
    diff = int((out.view(torch.int16) != plain.view(torch.int16)).sum())
    assert diff == 0, f"{diff} output elements differ ({reissues} re-issues)"
    assert reissues > 0 and unchecked == 0, f"re-issues {reissues}, unchecked {unchecked}"


PRODUCER = dict(
    report_valid=False, retry_warp=False, pdl_wait_activation=True, pzo=True, early_trigger=True
)


@pytest.mark.parametrize(
    "shape,m",
    [("bfa", 16), ("qa_kva", 16), ("qb", 16), ("gproj", 16), ("qb", 32), ("bfa", 8), ("qa_kva", 1)],
)
def test_producer_build_changes_only_negative_zero(shape: str, m: int) -> None:
    """.pzo + early trigger + 16 B owner stores equal the plain build with -0.0 read as +0.0, rows past M untouched."""
    n, k, tactics = DECODE_SPLITK[shape]
    tactic = fgs.SplitKTactic(*tactics[m])
    torch.manual_seed(m)
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    x[: m // 2] = -0.0
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    w[:16] = w[:16].abs()  # -0.0 rows times positive weight rows: -0.0 without .pzo

    def run(**flags):
        buf = torch.full((m + 1, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        fgs.run_splitk_dense_fgs(x, w.t(), None, buf[:m], True, tactic, **flags)
        return buf

    plain = run(**VARIANTS["plain"])
    prod = run(**PRODUCER, store16=True)
    torch.cuda.synchronize()
    canon = plain.view(torch.int16).clone()
    canon[canon == NEG0] = 0
    assert torch.equal(prod.view(torch.int16), canon)
    assert bool(torch.isnan(prod[m].float()).all())
