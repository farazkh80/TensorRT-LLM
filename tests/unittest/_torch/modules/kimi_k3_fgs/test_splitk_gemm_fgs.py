# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FGS split-K GEMM consumer (decode o_proj shape) on SM107: ready inputs and one late-published chunk."""

import os

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

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


@cute.kernel
def _late_copy16_kernel(dst: cute.Tensor, src: cute.Tensor, delay_ns: Int64):
    start = cute.arch.globaltimer()
    while cute.arch.globaltimer() < start + delay_ns:
        cute.arch.nanosleep(sleep_time=500)
    if cute.arch.thread_idx()[0] == 0:
        _copy16(dst.iterator.toint(), src.iterator.toint())
        cute.arch.fence_acq_rel_gpu()


@cute.jit
def _late_copy16(dst: cute.Tensor, src: cute.Tensor, delay_ns: Int64, stream):
    _late_copy16_kernel(dst, src, delay_ns).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def test_retry_reissues_late_chunk() -> None:
    """A 16 B -0.0 chunk, published from another stream after the consumer started, is retried, not consumed."""
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    m = 16
    x, w = _inputs(m, seed=3)
    expected = _run(x, w, "plain")
    counter = torch.zeros(fgs.RETRY_COUNTER_SLOTS, dtype=torch.int32, device="cuda")
    _run(x, w, "fgs", counter)  # JIT before the producer starts its delay
    slot = x.clone()
    row, col = m // 2, K // 2 + 8
    slot.view(torch.int16)[row, col : col + 8] = NEG0
    chunks = (slot[row, col : col + 8], x[row, col : col + 8])
    args = [from_dlpack(t, assumed_align=16) for t in chunks]
    s_prod, s_cons = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(s_prod):
        stream = cuda.CUstream(s_prod.cuda_stream)
        writer = cute.compile(_late_copy16, *args, cutlass.Int64(0), stream)
        torch.cuda.synchronize()
        counter.zero_()
        writer(*args, cutlass.Int64(2_000_000), stream)
    with torch.cuda.stream(s_cons):
        out = _run(slot, w, "fgs", counter)
    torch.cuda.synchronize()
    reissues, unchecked = counter.tolist()
    assert reissues > 0 and unchecked == 0
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
