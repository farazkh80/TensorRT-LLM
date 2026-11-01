# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FGS consumer build of the SM107 BF16 persistent GEMM / BMM at the Kimi K3 attention-core shapes.

Ready inputs: every build equals the in-tree runner bitwise at the same tactic. Hazards: A starts as the -0.0
sentinel and a writer kernel on another stream publishes it block by block, late and out of order; the consumer must
re-issue (counted) and still match bitwise.
"""

import importlib.util
import os
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

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

NEG0 = -32768

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or not hasattr(cute.arch, "mbarrier_v1_init"),
    reason="TMA report_valid needs SM107 and CuTe DSL 4.9",
)

DECODE_1CTA = ("base", False, (128, 128), (1, 1), 0)
SMALL_1CTA = ("base", False, (64, 128), (1, 1), 0)
PREFILL_2CTA = ("base", True, (256, 256), (2, 1), 0)

# name -> (kind, L, M, N, K, tactic): kimi_k3_cute_gemm shapes; "fb" A is the f_a columns of the [M, 224] bfa output,
# "wuk" / "wuv" A and out are head-major views of token-major buffers.
CASES = {
    "qkvg_m1": ("dense", 1, 1, 49152, 7168, DECODE_1CTA),
    "qkvg_m16": ("dense", 1, 16, 49152, 7168, DECODE_1CTA),
    "qkvg_m64": ("dense", 1, 64, 49152, 7168, DECODE_1CTA),
    "fb_m16": ("fb", 1, 16, 12288, 128, SMALL_1CTA),
    "fb_m16_x4": ("fb", 1, 16, 12288, 128, ("base", False, (64, 128), (1, 4), 0)),
    "fb_m64": ("fb", 1, 64, 12288, 128, ("base", False, (64, 64), (1, 1), 0)),
    "wuk_m16": ("wuk", 96, 16, 512, 128, SMALL_1CTA),
    "wuv_m16": ("wuv", 96, 16, 128, 512, SMALL_1CTA),
    "qkvg_pf": ("dense", 1, 1024, 49152, 7168, PREFILL_2CTA),
    "oproj_pf": ("dense", 1, 1024, 7168, 12288, PREFILL_2CTA),
}
VARIANTS = {
    "v": dict(report_valid=False, retry_warp=False, wait_input=True),
    "b": dict(report_valid=False, retry_warp=True, wait_input=True),
    "c": dict(report_valid=True, retry_warp=True, wait_input=False),
    "c_test": dict(report_valid=True, retry_warp=True, wait_input=False, retry_poll="test"),
    "cw": dict(report_valid=True, retry_warp=True, wait_input=True),
    "c_store_wait": dict(report_valid=True, retry_warp=True, wait_input=False, store_wait=True),
}


class Case:
    """Operands of one case: ``a`` / ``out`` are [L, M, K] / [L, M, N] views as the model passes them."""

    def __init__(self, name: str, seed: int = 0):
        kind, L, M, N, K, tactic = CASES[name]
        self.kind, self.tactic = kind, tactic
        g = torch.Generator(device="cuda").manual_seed(seed)

        def randn(*shape, scale=1.0):
            t = (torch.randn(*shape, device="cuda", generator=g) * scale).to(torch.bfloat16)
            t[t == 0] = 1.0  # published data never carries the -0.0 sentinel
            return t

        if kind == "dense":
            self.base = randn(M, K)
            self.a = self.base.unsqueeze(0)
            self.w = randn(1, N, K, scale=0.02)
            self.out_shape = (M, N)
        elif kind == "fb":
            self.base = randn(M, 224)
            self.a = self.base[:, :K].unsqueeze(0)
            self.w = randn(1, N, K, scale=0.02)
            self.out_shape = (M, N)
        elif kind == "wuk":
            self.base = randn(M, L, 192)
            self.a = self.base[:, :, :K].transpose(0, 1)
            self.w = randn(L, N, K, scale=0.05)
            self.out_shape = (M, L, 576)
        else:
            self.base = randn(M, L, K)
            self.a = self.base.transpose(0, 1)
            self.w = randn(L, N, K, scale=0.05)
            self.out_shape = (M, L, N)
        self.L, self.M, self.N, self.K = L, M, N, K

    def new_out(self):
        buf = torch.full(self.out_shape, float("nan"), device="cuda", dtype=torch.bfloat16)
        if self.kind in ("dense", "fb"):
            return buf, buf.unsqueeze(0)
        if self.kind == "wuk":
            return buf, buf[:, :, : self.N].transpose(0, 1)
        return buf, buf.transpose(0, 1)

    def written(self, buf: torch.Tensor) -> torch.Tensor:
        """The part of an output buffer the GEMM writes (W_UK fills only the first N of 576 columns)."""
        return buf[:, :, : self.N] if self.kind == "wuk" else buf

    def with_a(self, base: torch.Tensor) -> torch.Tensor:
        """``self.a``'s view of another buffer shaped like ``self.base``."""
        return base.as_strided(self.a.shape, self.a.stride(), self.a.storage_offset())


def run_stock(case: Case, a: torch.Tensor, out: torch.Tensor, w=None) -> None:
    """The in-tree runner at the case tactic (``kimi_k3_cute_gemm`` routing: dense GEMM, else the batched runner)."""
    w = case.w if w is None else w
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        CuteDSLBf16RubinBmmRunner,
        CuteDSLBf16RubinGemmRunner,
    )

    if case.kind == "dense":
        CuteDSLBf16RubinGemmRunner(output_dtype=torch.bfloat16)(
            [a[0], w[0], out[0]], tactic=(*case.tactic, 1)
        )
    else:
        CuteDSLBf16RubinBmmRunner()([a, w, out], tactic=case.tactic)


def run_variant(case: Case, a, out, variant: str, counter=None, **extra) -> None:
    fgs.run_gemm_fgs(
        a, case.w, out, case.tactic, retry_counter=counter, **VARIANTS[variant], **extra
    )


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def reference(case: Case) -> torch.Tensor:
    buf, out = case.new_out()
    run_stock(case, case.a, out)
    torch.cuda.synchronize()
    return buf


@dsl_user_op
def _ld_global_v4(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


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


@dsl_user_op
def _st_global_v4(addr: Int64, v, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [addr.ir_value(loc=loc, ip=ip)] + [x.ir_value(loc=loc, ip=ip) for x in v],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class BlockWriter:
    """Test producer: copies ``src`` -> ``dst`` ([L, M, K] views, K innermost) one (block_rows x block_cols) block
    per CTA. Block p (l-major, then rows, then K) is published ``delay + ranks[p] * step`` after ``flag`` turns
    non-zero (set on the consumer's stream right before the consumer), 16 B per store instruction. It never writes
    sentinels."""

    def __init__(self, block_rows: int, block_cols: int, threads: int = 128):
        assert block_cols % 8 == 0
        self.block_rows, self.block_cols, self.threads = block_rows, block_cols, threads

    @cute.jit
    def __call__(
        self, src, dst, ranks, flag, delay_ns: Int64, step_ns: Int64, stream: cuda.CUstream
    ):
        self.kernel(src, dst, ranks, flag, delay_ns, step_ns).launch(
            grid=(cute.size(ranks), 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        ranks: cute.Tensor,
        flag: cute.Tensor,
        delay_ns: Int64,
        step_ns: Int64,
    ):
        p, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        m = src.shape[1]
        k = src.shape[2]
        blocks_m = (m + self.block_rows - 1) // self.block_rows
        blocks_k = (k + self.block_cols - 1) // self.block_cols
        kb = p % blocks_k
        mb = (p // blocks_k) % blocks_m
        lb = p // (blocks_k * blocks_m)

        while _ld_relaxed_u32(flag.iterator.toint()) == 0:
            cute.arch.nanosleep(sleep_time=200)
        target = cute.arch.globaltimer() + delay_ns + Int64(ranks[p]) * step_ns
        while cute.arch.globaltimer() < target:
            cute.arch.nanosleep(sleep_time=200)

        rows = m - mb * self.block_rows
        if rows > self.block_rows:
            rows = Int32(self.block_rows)
        cols = k - kb * self.block_cols
        if cols > self.block_cols:
            cols = Int32(self.block_cols)
        chunks = cols // 8
        src_base = src.iterator.toint()
        dst_base = dst.iterator.toint()
        for c in cutlass.range(tid, rows * chunks, self.threads, unroll=1):
            r = c // chunks
            row = mb * self.block_rows + r
            col = kb * self.block_cols + (c - r * chunks) * 8
            src_off = (
                Int64(lb) * Int64(src.stride[0]) + Int64(row) * Int64(src.stride[1]) + Int64(col)
            )
            dst_off = (
                Int64(lb) * Int64(dst.stride[0]) + Int64(row) * Int64(dst.stride[1]) + Int64(col)
            )
            _st_global_v4(dst_base + dst_off * 2, _ld_global_v4(src_base + src_off * 2))
        cute.arch.fence_acq_rel_gpu()


_WRITERS = {}


def num_blocks(a: torch.Tensor, block_rows: int, block_cols: int) -> int:
    L, M, K = a.shape
    return L * -(-M // block_rows) * -(-K // block_cols)


def late_write(
    src, dst, ranks, delay_us: float, step_us: float, block_rows: int, block_cols: int, flag=None
):
    """Launch the writer on the current stream; without ``flag`` it starts at once."""
    if flag is None:
        flag = torch.ones(1, dtype=torch.int32, device="cuda")
    args = [from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=2) for t in (src, dst)]
    ranks_t = from_dlpack(ranks, assumed_align=4).mark_layout_dynamic()
    flag_t = from_dlpack(flag, assumed_align=4)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (block_rows, block_cols)
    if key not in _WRITERS:
        _WRITERS[key] = cute.compile(
            BlockWriter(block_rows, block_cols), *args, ranks_t, flag_t, Int64(0), Int64(0), stream
        )
    _WRITERS[key](
        *args, ranks_t, flag_t, Int64(int(delay_us * 1e3)), Int64(int(step_us * 1e3)), stream
    )


def block_shape(case: Case):
    """Publication granularity: K slices of all rows for decode (head-by-head producers, at least 8 slices), row
    blocks for prefill."""
    if case.M >= 256:
        return 128, case.K
    return case.M, min(128, max(32, case.K // 8))


def ranks_for(pattern: str, n: int, seed: int = 0) -> torch.Tensor:
    if pattern == "forward":
        r = torch.arange(n)
    elif pattern == "reverse":
        r = torch.arange(n - 1, -1, -1)
    elif pattern == "random":
        r = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    elif pattern == "one_late":
        r = torch.zeros(n, dtype=torch.int64)
        r[n // 2] = 50
    else:
        raise ValueError(pattern)
    return r.to(torch.int32).cuda()


def run_hazard(
    case: Case, pattern: str, variant: str = "c", delay_us: float = 300.0, step_us: float = 2.0
):
    """Writer on its own stream first, consumer second; returns (output buffer, re-issues, published A).

    The writer's clock starts at a flag the consumer's stream sets right before the consumer, so the first block
    lands ``delay_us`` after the consumer started however slow the host launch path is."""
    ref_a = case.a
    slot_base = torch.full_like(case.base, -0.0)
    slot_a = case.with_a(slot_base)
    br, bc = block_shape(case)
    ranks = ranks_for(pattern, num_blocks(ref_a, br, bc))
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    buf, out = case.new_out()
    # Compile both kernels before the timed pair, so the writer cannot finish while the host JITs.
    warm_base = case.base.clone()
    late_write(ref_a, case.with_a(warm_base), ranks, 0.0, 0.0, br, bc)
    run_variant(case, case.with_a(warm_base), case.new_out()[1], variant)
    torch.cuda.synchronize()

    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    sp, sc = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(sp):
        late_write(ref_a, slot_a, ranks, delay_us, step_us, br, bc, flag)
    with torch.cuda.stream(sc):
        flag.fill_(1)
        run_variant(case, slot_a, out, variant, counter=counter)
    torch.cuda.synchronize()
    return buf, int(counter.item()), bool(torch.equal(bits(slot_a), bits(ref_a)))


READY = [(n, v) for n in CASES for v in ("v", "b", "c")] + [
    (n, v) for n in ("qkvg_m16", "fb_m16_x4", "oproj_pf") for v in ("c_test", "cw", "c_store_wait")
]


@pytest.mark.parametrize("name,variant", READY)
def test_ready_input_matches_in_tree_runner(name: str, variant: str) -> None:
    case = Case(name)
    ref = reference(case)
    f = case.written(ref).float()
    assert bool(torch.isfinite(f).all()) and bool(f.abs().max() > 0)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    buf, out = case.new_out()
    run_variant(case, case.a, out, variant, counter=counter)
    torch.cuda.synchronize()
    assert torch.equal(bits(buf), bits(ref)), f"{name}/{variant} differs from the in-tree runner"
    assert counter.item() == 0


@pytest.mark.parametrize("name", ["qkvg_m16", "fb_m16_x4", "wuk_m16", "oproj_pf"])
def test_middle_kernel_pzo_early_trigger(name: str) -> None:
    """Consumer of its input and early-triggering PZO producer of its output: equal up to -0.0,
    which it never stores."""
    case = Case(name)
    # -0.0 rows times positive weight rows give -0.0 without .pzo
    case.w[:, :16] = case.w[:, :16].abs()
    neg = case.base.clone()
    neg[: max(1, case.M // 4)] = -0.0
    a = case.with_a(neg)
    ref_buf, ref_out = case.new_out()
    run_stock(case, a, ref_out)
    buf, out = case.new_out()
    fgs.run_gemm_fgs(
        a,
        case.w,
        out,
        case.tactic,
        report_valid=False,
        retry_warp=True,
        wait_input=False,
        pzo=True,
        early_trigger=True,
    )
    torch.cuda.synchronize()
    canon = bits(ref_buf).clone()
    canon[canon == NEG0] = 0
    assert torch.equal(bits(buf), canon)
    assert not bool((bits(out) == NEG0).any())


@pytest.mark.parametrize("pattern", ["forward", "reverse", "random", "one_late"])
@pytest.mark.parametrize(
    "name", ["qkvg_m16", "fb_m16", "fb_m16_x4", "wuk_m16", "wuv_m16", "qkvg_pf", "oproj_pf"]
)
def test_late_out_of_order_producer(name: str, pattern: str) -> None:
    case = Case(name, seed=1)
    ref = reference(case)
    buf, reissues, published = run_hazard(case, pattern)
    assert published, f"{name}/{pattern}: the writer did not publish A"
    diff = int((bits(buf) != bits(ref)).sum())
    assert diff == 0, f"{name}/{pattern}: {diff} output elements differ ({reissues} re-issues)"
    assert reissues > 0, (
        f"{name}/{pattern}: the writer finished before the consumer read (no re-issue)"
    )


@pytest.mark.parametrize("name", ["qkvg_m16", "oproj_pf"])
def test_one_late_chunk(name: str) -> None:
    """A valid slot except one 16 B chunk; the writer republishes every block 300 us after the consumer starts."""
    case = Case(name, seed=2)
    ref = reference(case)
    slot_base = case.base.clone()
    slot_a = case.with_a(slot_base)
    r, c = case.M // 2, (case.K // 2) // 8 * 8
    bits(slot_a[0, r, c : c + 8]).fill_(NEG0)
    br, bc = block_shape(case)
    ranks = torch.zeros(num_blocks(case.a, br, bc), dtype=torch.int32, device="cuda")
    run_variant(case, case.a, case.new_out()[1], "c")
    late_write(case.a, case.with_a(case.base.clone()), ranks, 0.0, 0.0, br, bc)
    torch.cuda.synchronize()

    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    buf, out = case.new_out()
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    sp, sc = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(sp):
        late_write(case.a, slot_a, ranks, 300.0, 0.0, br, bc, flag)
    with torch.cuda.stream(sc):
        flag.fill_(1)
        run_variant(case, slot_a, out, "c", counter=counter)
    torch.cuda.synchronize()
    diff = int((bits(buf) != bits(ref)).sum())
    assert diff == 0, f"{name}: {diff} output elements differ ({counter.item()} re-issues)"
    assert counter.item() > 0
