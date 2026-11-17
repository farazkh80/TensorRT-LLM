# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the prefill token-M kernels: inline PTX helpers, the cp.async -> UMMA pipeline, SF S2T copy."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import T, dsl_user_op

HIDDEN = 3584
INTER = 3072
TOP_K = 16
SF_VEC = 32
W_ROWS = 2 * INTER
SMEM_ARCH = "sm_107"


def _asm(ret_ty, args, asm, constraints, *, side_effects=False, loc=None, ip=None):
    return llvm.inline_asm(
        ret_ty,
        args,
        asm,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def f32_bits(x, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bits_f32(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ex2(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def rcp(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def tanh_approx(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


# tanh(y) = y * (1 + y^2 * P(y^2)) for |y| < 0.6 (least-squares fit, 1e-7 relative in fp32).
_TANH_POLY = (
    -0.3333333134651184,
    0.13333244621753693,
    -0.053952112793922424,
    0.021732505410909653,
    -0.008262349292635918,
    0.0022267044987529516,
)
LOG2E = 1.4426950408889634


def tanh_accurate(y):
    """About 2^-22 relative: the polynomial below |y| = 0.6, 1 - 2 / (e^(2|y|) + 1) above (decode fc1.py)."""
    ay = cute.arch.fmax(y, -y)
    big = cutlass.Float32(1.0) - cutlass.Float32(2.0) * rcp(
        ex2(ay * (2.0 * LOG2E)) + cutlass.Float32(1.0)
    )
    big = cutlass.select_(y < cutlass.Float32(0.0), -big, big)
    y2 = y * y
    poly = cutlass.Float32(_TANH_POLY[-1])
    for c in reversed(_TANH_POLY[:-1]):
        poly = poly * y2 + cutlass.Float32(c)
    small = y + y * (y2 * poly)
    return cutlass.select_(ay < cutlass.Float32(0.6), small, big)


def sigmoid_accurate(x):
    return rcp(cutlass.Float32(1.0) + ex2(x * (-LOG2E)))


@dsl_user_op
def e4m3x2(hi, lo, *, loc=None, ip=None):
    """Two fp32 -> E4M3 codes (round to nearest even, saturating), packed as (hi << 8) | lo in an Int32."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .b16 h;\ncvt.rn.satfinite.e4m3x2.f32 h, $1, $2;\ncvt.u32.u16 $0, h;\n}",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)]
        + [cutlass.Int32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_release_gpu_add(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(v).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_acquire_gpu_b32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.global;", "", side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def st_global_b32(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_b16(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_cg_v4(addr, *, loc=None, ip=None):
    """16 B L2-only global load (bypasses L1) -> four Int32 words."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def st_shared_b32(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_b8(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.global.b8 [$0], h;\n}",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_load_32x32b_x32(addrs, *, loc=None, ip=None):
    """One tcgen05.ld.32x32b.x32 per TMEM address (thread T reads lane T % 32 of its warp's quarter, 32 consecutive
    columns from the address's column), then tcgen05.wait::ld."""
    n = len(addrs)
    asm = []
    for k in range(n):
        regs = ", ".join(f"${32 * k + r}" for r in range(32))
        asm.append(f"tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{regs}}}, [${32 * n + k}];")
    asm.append("tcgen05.wait::ld.sync.aligned;")
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * (32 * n)),
        [cutlass.Int32(a).ir_value(loc=loc, ip=ip) for a in addrs],
        "\n".join(asm),
        ",".join(["=f"] * (32 * n) + ["r"] * n),
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return [
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [k], loc=loc, ip=ip)) for k in range(32 * n)
    ]


@dsl_user_op
def tma_gather4(tma_atom, smem_dst_ptr, mbar_ptr, col, r0, r1, r2, r3, *, loc=None, ip=None):
    """One ``cp.async.bulk.tensor.2d.tile::gather4`` load of rows r0..r3 (box width from the atom; -1 = zero fill)
    at column ``col`` into ``smem_dst_ptr`` (4 consecutive box rows), completing tx bytes on ``mbar_ptr``.
    Inline PTX because the DSL has no gather4 op (in-tree ``rubin/inline_ptx.py``, 1-CTA variant)."""
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    desc_ptr_ty = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(), AddressSpace.generic, 64
    )
    desc = (
        _cute_nvgpu_ir.get_tma_desc_addr(desc_ptr_ty, exec_atom, loc=loc, ip=ip)
        .toint()
        .ir_value(loc=loc, ip=ip)
    )
    operands = [
        cutlass.Int32(smem_dst_ptr.toint()).ir_value(loc=loc, ip=ip),
        desc,
        cutlass.Int32(mbar_ptr.toint()).ir_value(loc=loc, ip=ip),
    ] + [cutlass.Int32(v).ir_value(loc=loc, ip=ip) for v in (col, r0, r1, r2, r3)]
    operands.append(cutlass.Int64(0).ir_value(loc=loc, ip=ip))
    _asm(
        None,
        operands,
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], $8;",
        "r,l,r,r,r,r,r,r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tcgen05_fence(kind: str, *, loc=None, ip=None):
    _asm(None, [], f"tcgen05.fence::{kind}_thread_sync;", "", side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def bar_sync(barrier_id: int, num_threads: int, *, loc=None, ip=None):
    _asm(None, [], f"bar.sync {barrier_id}, {num_threads};", "", side_effects=True, loc=loc, ip=ip)


@dataclass(frozen=True)
class UnalignedBarrier:
    barrier_id: int
    num_threads: int

    def arrive_and_wait(self):
        bar_sync(self.barrier_id, self.num_threads)

    def wait_unaligned(self):
        bar_sync(self.barrier_id, self.num_threads)


@dataclass(frozen=True)
class CpAsyncUmmaPipeline(pipeline.PipelineAsync):
    """cp.async producer threads (noinc arrivals once their copies landed) -> tcgen05 MMA consumer (commit)."""

    cta_group: tcgen05.CtaGroup

    @staticmethod
    def create(
        num_stages: int,
        num_producer_threads: int,
        barrier_storage: cute.Pointer,
        async_load: bool = True,
    ):
        full = pipeline.MbarrierArray(
            barrier_storage=barrier_storage,
            num_stages=num_stages,
            agent=(
                pipeline.PipelineOp.AsyncLoad if async_load else pipeline.PipelineOp.AsyncThread,
                pipeline.CooperativeGroup(pipeline.Agent.Thread, num_producer_threads),
            ),
        )
        empty = pipeline.MbarrierArray(
            barrier_storage=barrier_storage + num_stages,
            num_stages=num_stages,
            agent=(
                pipeline.PipelineOp.TCGen05Mma,
                pipeline.CooperativeGroup(pipeline.Agent.Thread),
            ),
        )
        return CpAsyncUmmaPipeline(full, empty, num_stages, None, None, tcgen05.CtaGroup.ONE)

    def consumer_release(self, state: pipeline.PipelineState):
        self.sync_object_empty.arrive(state.index, self.consumer_mask, self.cta_group)


def s2t_copy_and_partition(cta_group, sSF: cute.Tensor, tSF: cute.Tensor):
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), cutlass.Float8E8M0FNU)
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
    tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t


def ue8m0_scale(amax):
    """UE8M0 byte and exact inverse scale for a 32-element group: e = ceil(log2(amax / 448)) (448 = 1.75 * 2^8:
    the binary exponent minus 8, plus one when the mantissa exceeds 1.75), byte 0 and scale 1 when amax is 0."""
    bits = f32_bits(amax) & cutlass.Int32(0x7FFFFFFF)
    e = (bits >> 23) - 135
    e = e + cutlass.select_(
        (bits & cutlass.Int32(0x7FFFFF)) > cutlass.Int32(0x600000),
        cutlass.Int32(1),
        cutlass.Int32(0),
    )
    e = cutlass.max(cutlass.min(e, cutlass.Int32(126)), cutlass.Int32(-127))
    sf_byte = cutlass.select_(bits == 0, cutlass.Int32(0), e + 127)
    inv_scale = cutlass.select_(bits == 0, cutlass.Float32(1.0), bits_f32((127 - e) << 23))
    return sf_byte, inv_scale


def use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


@functools.cache
def num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def gmem_ptr(dtype, tensor: torch.Tensor, align: int):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def check_tensor(name, t, dtype, shape, align=16):
    if t.dtype != dtype or tuple(t.shape) != tuple(shape) or not t.is_contiguous():
        raise ValueError(
            f"{name}: expected contiguous {dtype} {tuple(shape)}, got {t.dtype} {tuple(t.shape)}"
        )
    if t.data_ptr() % align:
        raise ValueError(f"{name}: needs {align} B alignment")
