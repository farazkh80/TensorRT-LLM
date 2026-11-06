# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The main loop follows kimi_k3_fgs/moe/latent_down.py (derived from FlashInfer v0.6.18 (39c024e2b5)
# flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py, Apache-2.0).
"""Kimi K3 decode shared expert as two persistent CuTe DSL GEMMs with a CTA cap (SM107).

    h = situ(bf16(x @ Wg^T), bf16(x @ Wu^T))   x [M, 7168], W_gate_up [12288, 7168] = [Wg; Wu], h [M, 6144]
    y = bf16(h @ Wd^T)                          W_down [7168, 6144], y [M, 7168]           (bf16, M <= 128)

``situ(g, u) = (beta * tanh(g / beta) * sigmoid(g)) * (linear_beta * tanh(u / linear_beta))`` in fp32 on the
bf16-rounded GEMM outputs, rounded to bf16: the same instruction sequence as the stock Triton ``situ_and_mul``, so
for equal bf16 inputs the outputs are bitwise equal.

Each GEMM is swap-AB tcgen05 (kernel-M = output columns, kernel-N = the tokens) over ``units`` = output tile x K
slice. The grid is at most ``ctas`` CTAs and each CTA loops over units ``bid, bid + grid, ...``, so a capped grid
still streams every weight. Weight and activation TMA warps run ahead across units; TMEM accumulators are double
buffered, so the next unit's MMAs overlap the epilogue of the previous one.

Gate/up units cover output columns [64 t, 64 t + 64) of both halves: two M=64 MMAs (gate rows, up rows) share the
activation stage, so the epilogue sees g and u of the same column in one CTA.

K split (``split`` > 1): every unit writes its fp32 partial to a global workspace slot and adds 1 (acq_rel) to its
tile's counter after a CTA barrier; the CTA that brings the counter to ``split`` sums the slots in split order
0, 1, ..., runs the epilogue, and re-arms the counter to 0. Deterministic for a given (ctas, M); no atomics on data.

The weights stream from launch; with PDL the activation loads and the first global write wait on
``griddepcontrol.wait``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
from typing import Dict, Optional, Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from .sync import atom_add_acq_rel, bits_to_f32, ld_cg_v4

HIDDEN = 7168
INTER = 6144
MAX_ROWS = 128
GATE_CAP = 4.0
LINEAR_CAP = 25.0

_CTA_K = 128
_THREADS = 256
_EPI_THREADS = 128
_EPI_BARRIER = 1
_TMEM_COLS = 512
_SMEM_CAPACITY = utils.get_smem_capacity_in_bytes("sm_100")
_MAX_SPLIT = 8


@dataclasses.dataclass(frozen=True)
class Tactic:
    mma_m: int
    mma_n: int
    split: int
    stages: int

    def tiles(self, dual: bool) -> int:
        return INTER // 64 if dual else HIDDEN // self.mma_m

    def width(self, dual: bool) -> int:
        """fp32 columns of one unit's partial (gate and up side by side for gate/up)."""
        return 128 if dual else self.mma_m


def _k_tiles(dual: bool) -> int:
    return (HIDDEN if dual else INTER) // _CTA_K


def _stage_bytes(t: Tactic, dual: bool) -> int:
    return ((2 if dual else 1) * t.mma_m + t.mma_n) * _CTA_K * 2


def _smem_bytes(t: Tactic, dual: bool) -> int:
    return _stage_bytes(t, dual) * t.stages + t.width(dual) * t.mma_n * 4 + 2048


def _mma_n(m: int) -> int:
    for n in (8, 16, 32, 64, 128):
        if m <= n:
            return n
    raise ValueError(f"shared_expert: at most {MAX_ROWS} rows, got {m}")


def _split_for(tiles: int, k_tiles: int, ctas: int) -> int:
    """Smallest per-CTA K-tile load that still gives every CTA a unit: rounds x (k_tiles / split + 0.5)."""
    best = None
    for s in range(1, _MAX_SPLIT + 1):
        units = tiles * s
        if units < ctas and s < _MAX_SPLIT:
            continue
        cost = math.ceil(units / ctas) * (math.ceil(k_tiles / s) + 0.5)
        if best is None or cost < best[0]:
            best = (cost, s)
    return best[1]


def default_tactic(m: int, ctas: int, dual: bool, smem_budget: Optional[int] = None) -> Tactic:
    """Tactic for M rows on a grid of at most ``ctas`` CTAs (``TLLM_K3_SHARED_TACTIC=mma_m,split,stages`` overrides
    for both GEMMs)."""
    n = _mma_n(m)
    override = os.environ.get("TLLM_K3_SHARED_TACTIC")
    if override:
        mma_m, split, stages = (int(v) for v in override.split(","))
        return Tactic(64 if dual else mma_m, n, split, stages)
    if smem_budget is None:
        # Stage depth bounds a capped grid's bandwidth; the SMs above the cap host the comm CTAs. 190 leaves room
        # for a dispatch CTA beside every shared CTA.
        smem_budget = int(os.environ.get("TLLM_K3_SHARED_SMEM_KB", "224")) * 1024
    mma_m = 64 if dual else 128
    probe = Tactic(mma_m, n, 1, 0)
    split = _split_for(probe.tiles(dual), _k_tiles(dual), ctas)
    stages = (smem_budget - _smem_bytes(probe, dual)) // _stage_bytes(probe, dual)
    return Tactic(mma_m, n, split, max(2, min(8, stages)))


def validate(t: Tactic, dual: bool, m: int) -> None:
    if (64 if dual else t.mma_m) != t.mma_m or t.mma_m not in (64, 128):
        raise ValueError(f"shared_expert: unsupported tile {t}")
    if not 1 <= m <= t.mma_n or t.mma_n not in (8, 16, 32, 64, 128):
        raise ValueError(f"shared_expert: {m} rows do not fit mma_n {t.mma_n}")
    if not 1 <= t.split <= _MAX_SPLIT or t.split > _k_tiles(dual):
        raise ValueError(f"shared_expert: split {t.split} out of range")
    if t.stages < 2 or _smem_bytes(t, dual) > _SMEM_CAPACITY:
        raise ValueError(f"shared_expert: {t} needs {_smem_bytes(t, dual)} B of shared memory")
    if (4 if dual else 2) * t.mma_n > _TMEM_COLS:
        raise ValueError(f"shared_expert: {t} does not fit TMEM")


def _asm(res, operands, text, constraints, loc, ip, side_effects=True):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_v4(addr: Int32, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc,
        ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _st_v4(addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Int32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def _st_v4f(addr: Int64, v0: Float32, v1: Float32, v2: Float32, v3: Float32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        loc,
        ip,
    )


@dsl_user_op
def _st_b32(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.relaxed.gpu.global.b32 [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def _bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc,
            ip,
            side_effects=False,
        )
    )


@dsl_user_op
def _bf16x2_unpack(w: Int32, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xFFFF0000;\nmov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
        "=f,=f,r",
        loc,
        ip,
        side_effects=False,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2)]


# tanh(y) as libdevice tanhf: odd polynomial for |y| < 0.6, else 1 - 2 / (2^(2|y| log2 e) + 1) with the sign of y,
# 1 for |y| >= 9.01. Both branches are evaluated and selected (no labels in inline asm).
_TANH = (
    "div.full.f32 y, {x}, {cap};\n"
    "abs.ftz.f32 a, y;\n"
    "mul.f32 s, y, y;\n"
    "fma.rn.ftz.f32 p, 0f3C80F082, s, 0fBD563CAE;\n"
    "fma.rn.ftz.f32 p, p, s, 0f3E085941;\n"
    "fma.rn.ftz.f32 p, p, s, 0fBEAAA9ED;\n"
    "fma.rn.ftz.f32 p, p, s, 0f00000000;\n"
    "fma.rn.ftz.f32 p, p, y, y;\n"
    "mul.f32 e, a, 0f4038AA3B;\n"
    "ex2.approx.ftz.f32 e, e;\n"
    "add.f32 e, e, 0f3F800000;\n"
    "rcp.approx.ftz.f32 e, e;\n"
    "fma.rn.ftz.f32 e, e, 0fC0000000, 0f3F800000;\n"
    "setp.ge.f32 big, a, 0f41102CB4;\n"
    "selp.f32 e, 0f3F800000, e, big;\n"
    "mov.b32 sb, y;\n"
    "and.b32 sb, sb, 0x80000000;\n"
    "mov.b32 eb, e;\n"
    "or.b32 eb, eb, sb;\n"
    "mov.b32 e, eb;\n"
    "setp.ltu.f32 small, a, 0f3F19999A;\n"
    "selp.f32 {out}, p, e, small;\n"
)


@dsl_user_op
def _situ(g: Float32, u: Float32, cap: Float32, lin: Float32, *, loc=None, ip=None) -> Float32:
    """(cap * tanh(g / cap) * sigmoid(g)) * (lin * tanh(u / lin)) with the stock kernel's instruction sequence."""
    text = (
        "{\n"
        ".reg .f32 y, a, s, p, e, tg, tu, sg;\n"
        ".reg .b32 sb, eb;\n"
        ".reg .pred big, small;\n"
        + _TANH.format(x="$1", cap="$3", out="tg")
        + _TANH.format(x="$2", cap="$4", out="tu")
        + "mul.f32 tg, $3, tg;\n"
        "mul.f32 sg, $1, 0fBFB8AA3B;\n"
        "ex2.approx.f32 sg, sg;\n"
        "add.f32 sg, sg, 0f3F800000;\n"
        "div.full.f32 sg, 0f3F800000, sg;\n"
        "mul.f32 tg, tg, sg;\n"
        "mul.f32 tu, $4, tu;\n"
        "mul.f32 $0, tu, tg;\n"
        "}"
    )
    return Float32(
        _asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (g, u, cap, lin)],
            text,
            "=f,f,f,f,f",
            loc,
            ip,
            side_effects=False,
        )
    )


class SharedGemmKernel:
    def __init__(
        self, tactic: Tactic, *, dual: bool, use_pdl: bool, gate_cap: float, linear_cap: float
    ):
        self.dual = dual
        self.cta_m, self.cta_n = tactic.mma_m, tactic.mma_n
        self.split = tactic.split
        self.stages = tactic.stages
        self.tiles = tactic.tiles(dual)
        self.units = self.tiles * self.split
        self.k_tiles = _k_tiles(dual)
        self.width = tactic.width(dual)
        self.n_acc = 2 if dual else 1
        self.use_pdl = use_pdl
        self.gate_cap = gate_cap
        self.linear_cap = linear_cap
        self.acc_dtype = Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
        self.launch_smem = _smem_bytes(tactic, dual)

    @cute.experimental.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mOut: cute.Tensor,
        mWs: cute.Tensor,
        mCnt: cute.Tensor,
        ctas: Int32,
        stream: _cuda.CUstream,
    ):
        self.kernel(mA, mB, mOut, mWs, mCnt).launch(
            grid=(ctas, 1, 1),
            block=(_THREADS, 1, 1),
            smem=cute.Int64(self.launch_smem),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    def _unit(self, u: Int32):
        """(output tile, first k tile, k tiles) of unit u; K slices differ by at most one k tile, the lower
        slices take the longer ones."""
        tile = u // self.split
        s = u - tile * self.split
        k_base = self.k_tiles // self.split
        k_rem = self.k_tiles - k_base * self.split
        k_count = k_base + (k_rem - s + self.split - 1) // self.split
        return tile, s * k_base + cutlass.min(s, k_rem), k_count

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (rows, K, 1) weights, K-major
        mB: cute.Tensor,  # (M, K, 1) activations, K-major
        mOut: cute.Tensor,  # bf16 [M, ld]
        mWs: cute.Tensor,  # fp32 [units * mma_n * width] (split > 1)
        mCnt: cute.Tensor,  # int32 [tiles], zero-armed (split > 1)
    ):
        stages = self.stages
        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            (self.cta_m, self.cta_n),
        )
        mnk_tiler = (self.cta_m, self.cta_n, _CTA_K)
        bid = cute.arch.block_idx()[0]
        grid = cute.arch.grid_dim()[0]
        count = (self.units - bid + grid - 1) // grid
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages)
        sA0 = cute_ext.allocate(ab_dtype, cute.AddressSpace.smem, a_layout, alignment=1024)
        sA1 = sA0
        if cutlass.const_expr(self.dual):
            sA1 = cute_ext.allocate(ab_dtype, cute.AddressSpace.smem, a_layout, alignment=1024)
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=1024,
        )
        # This unit's fp32 partial, [token][column]; gate/up keep gate in columns 0..63 and up in 64..127.
        sP = cute_ext.allocate(
            Float32, cute.AddressSpace.smem, cute.make_layout(self.cta_n * self.width), alignment=16
        )
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, (self.cta_m, self.cta_n), acc_stage=2 * self.n_acc
        )

        def _mbars(n):
            return cute_ext.allocate(
                cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(n), alignment=8
            ).iterator

        bar_full = _mbars(stages)
        bar_empty = _mbars(stages)
        acc_full = _mbars(2)
        acc_empty = _mbars(2)
        bar_tmem = _mbars(1)
        tmem_base_ptr = cute_ext.allocate(
            Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        last_flag = cute_ext.allocate(
            Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                for i in range(2):
                    cute.arch.mbarrier_init(acc_full + i, 1)
                    cute.arch.mbarrier_init(acc_empty + i, _EPI_THREADS)
                cute.arch.mbarrier_init(bar_tmem, 32 + _EPI_THREADS)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        a_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
        b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
        if warp_idx == 0:
            self.weight_warp(bar_full, bar_empty, mA, sA0, sA1, a_map, bid, grid, count)
        elif warp_idx == 1:
            self.act_warp(bar_full, bar_empty, mB, sB, b_map, bid, grid, count)
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                acc_full,
                acc_empty,
                bar_tmem,
                tiled_mma,
                sA0,
                sA1,
                sB,
                tmem_base_ptr,
                acc_layout,
                _CTA_K // cute.size(tiled_mma.shape_mnk, mode=[2]),
                bid,
                grid,
                count,
            )
        elif warp_idx >= 4:
            self.epilog_warps(
                acc_full,
                acc_empty,
                bar_tmem,
                tmem_base_ptr,
                acc_layout,
                sP,
                last_flag,
                mB,
                mOut,
                mWs,
                mCnt,
                cute.arch.thread_idx()[0] - 128,
                bid,
                grid,
                count,
            )

    @cute.experimental.jit
    def weight_warp(self, bar_full, bar_empty, mA, sA0, sA1, a_map, bid, grid, count):
        stages = self.stages
        tx = cute.size_in_bytes(sA0.element_type, cute.slice_(sA0.layout, (None, None, None, 0)))
        if cutlass.const_expr(self.dual):
            tx = tx * 2
        it = Int32(0)
        for i in cutlass.range(count, unroll=1):
            tile, k_start, k_count = self._unit(bid + i * grid)
            g0 = cute.local_tile(mA, (self.cta_m, _CTA_K), (tile, None, 0))
            g1 = cute.local_tile(mA, (self.cta_m, _CTA_K), (tile + self.tiles, None, 0))
            for k in cutlass.range(k_count, unroll=1):
                stage = it % stages
                cute.arch.mbarrier_wait(bar_empty + stage, ((it // stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, tx)
                cute_ext.tma_load(
                    g0[None, None, k_start + k],
                    sA0[None, None, None, stage],
                    (bar_full + stage).value,
                    cta_v_map=a_map,
                    tma_operation_type=self.tma_op,
                    update_expect_tx=False,
                )
                if cutlass.const_expr(self.dual):
                    cute_ext.tma_load(
                        g1[None, None, k_start + k],
                        sA1[None, None, None, stage],
                        (bar_full + stage).value,
                        cta_v_map=a_map,
                        tma_operation_type=self.tma_op,
                        update_expect_tx=False,
                    )
                it = it + 1

    @cute.experimental.jit
    def act_warp(self, bar_full, bar_empty, mB, sB, b_map, bid, grid, count):
        stages = self.stages
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        tx = cute.size_in_bytes(sB.element_type, cute.slice_(sB.layout, (None, None, None, 0)))
        gB = cute.local_tile(mB, (self.cta_n, _CTA_K), (0, None, 0))
        it = Int32(0)
        for i in cutlass.range(count, unroll=1):
            tile, k_start, k_count = self._unit(bid + i * grid)
            for k in cutlass.range(k_count, unroll=1):
                stage = it % stages
                cute.arch.mbarrier_wait(bar_empty + stage, ((it // stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, tx)
                cute_ext.tma_load(
                    gB[None, None, k_start + k],
                    sB[None, None, None, stage],
                    (bar_full + stage).value,
                    cta_v_map=b_map,
                    tma_operation_type=self.tma_op,
                    update_expect_tx=False,
                )
                it = it + 1

    @cute.experimental.jit
    def mma_warp(
        self,
        bar_full,
        bar_empty,
        acc_full,
        acc_empty,
        bar_tmem,
        tiled_mma: cute.TiledMma,
        sA0: cute.Tensor,
        sA1: cute.Tensor,
        sB: cute.Tensor,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,
        bid,
        grid,
        count,
    ):
        cute.arch.alloc_tmem(_TMEM_COLS, tmem_base_ptr, is_two_cta=False)
        cute.arch.mbarrier_arrive(bar_tmem)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        acc = cute.make_tensor(tmem_ptr, acc_layout)
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        it = Int32(0)
        for i in cutlass.range(count, unroll=1):
            tile, k_start, k_count = self._unit(bid + i * grid)
            buf = i % 2
            cute.arch.mbarrier_wait(acc_empty + buf, ((i // 2) & 1) ^ 1)
            for k in cutlass.range(k_count, unroll=1):
                stage = it % self.stages
                cute.arch.mbarrier_wait(bar_full + stage, (it // self.stages) & 1)
                for k_block in range(mma_inst_tile_k):
                    if k_block == 0:
                        mma_atom.set(tcgen05.Field.ACCUMULATE, k != 0)
                    else:
                        mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                    cute_ext.dot(
                        mma_atom,
                        cute.append_ones(sA0[None, None, k_block, stage], up_to_rank=3),
                        cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                        acc[None, None, None, buf * self.n_acc],
                    )
                    if cutlass.const_expr(self.dual):
                        cute_ext.dot(
                            mma_atom,
                            cute.append_ones(sA1[None, None, k_block, stage], up_to_rank=3),
                            cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                            acc[None, None, None, buf * self.n_acc + 1],
                        )
                with cute.arch.elect_one():
                    tcgen05.commit(bar_empty + stage, None, self.cta_group)
                it = it + 1
            with cute.arch.elect_one():
                tcgen05.commit(acc_full + buf, None, self.cta_group)
        cute.arch.mbarrier_arrive(bar_tmem)
        cute.arch.mbarrier_wait(bar_tmem, 1)
        cute.arch.dealloc_tmem(tmem_ptr, _TMEM_COLS, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warps(
        self,
        acc_full,
        acc_empty,
        bar_tmem,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        sP: cute.Tensor,
        last_flag,
        mB: cute.Tensor,
        mOut: cute.Tensor,
        mWs: cute.Tensor,
        mCnt: cute.Tensor,
        tid: Int32,
        bid,
        grid,
        count,
    ):
        cute.arch.mbarrier_arrive(bar_tmem)
        cute.arch.mbarrier_wait(bar_tmem, 0)
        acc = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout
        )
        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, _CTA_K),
                utils.LayoutEnum.COL_MAJOR,
                self.acc_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc[((None, None), 0, 0, 0)],
        )
        thr_t2r = tiled_copy_t2r.get_slice(tid)
        sP0 = cute.make_tensor(
            sP.iterator, cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.width))
        )
        sP1 = cute.make_tensor(
            sP.iterator + self.cta_m,
            cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.width)),
        )
        r0 = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, cute.flat_divide(sP0, epi_tile), tid),
            alignment=32,
        )
        r1 = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, cute.flat_divide(sP1, epi_tile), tid),
            alignment=32,
        )
        rows = cute.size(mB, mode=[0])
        if cutlass.const_expr(self.use_pdl):
            # A previous launch's consumers may still read the output; the workspace belongs to the previous
            # launch until it completes.
            cute.arch.griddepcontrol_wait()
        for i in cutlass.range(count, unroll=1):
            u = bid + i * grid
            tile = u // self.split
            buf = i % 2
            cute.arch.mbarrier_wait(acc_full + buf, (i // 2) & 1)
            cute_ext.partition_and_copy(thr_t2r, acc[((None, None), 0, 0, buf * self.n_acc)], r0)
            if cutlass.const_expr(self.dual):
                cute_ext.partition_and_copy(
                    thr_t2r, acc[((None, None), 0, 0, buf * self.n_acc + 1)], r1
                )
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(acc_empty + buf)
            cute_ext.partition_and_copy(thr_t2r, r0, sP0)
            if cutlass.const_expr(self.dual):
                cute_ext.partition_and_copy(thr_t2r, r1, sP1)
            cute.arch.barrier(barrier_id=_EPI_BARRIER, number_of_threads=_EPI_THREADS)
            if cutlass.const_expr(self.split == 1):
                self.finish(sP, mOut, mWs, tile, rows, tid, False)
            else:
                self.publish(sP, mWs, u, rows, tid)
                cute.arch.barrier(barrier_id=_EPI_BARRIER, number_of_threads=_EPI_THREADS)
                if tid == 0:
                    old = atom_add_acq_rel(mCnt.iterator.toint() + Int64(tile) * 4, Int32(1))
                    last_flag[0] = old
                cute.arch.barrier(barrier_id=_EPI_BARRIER, number_of_threads=_EPI_THREADS)
                if last_flag[0] == self.split - 1:
                    if tid == 0:
                        _st_b32(mCnt.iterator.toint() + Int64(tile) * 4, Int32(0))
                    self.finish(sP, mOut, mWs, tile, rows, tid, True)
            cute.arch.barrier(barrier_id=_EPI_BARRIER, number_of_threads=_EPI_THREADS)
        cute.arch.mbarrier_arrive(bar_tmem)

    @cute.experimental.jit
    def publish(self, sP: cute.Tensor, mWs: cute.Tensor, u: Int32, rows: Int32, tid: Int32):
        """Copy the unit's partial (rows x width fp32) from shared memory to its workspace slot."""
        vecs = rows * (self.width // 4)
        s_base = sP.iterator.toint()
        w_base = mWs.iterator.toint() + Int64(u) * (self.cta_n * self.width * 4)
        for v in cutlass.range(tid, vecs, _EPI_THREADS, unroll=1):
            x = _ld_shared_v4(s_base + v * 16)
            _st_v4f(w_base + Int64(v) * 16, x[0], x[1], x[2], x[3])

    @cute.experimental.jit
    def finish(
        self,
        sP: cute.Tensor,
        mOut: cute.Tensor,
        mWs: cute.Tensor,
        tile: Int32,
        rows: Int32,
        tid: Int32,
        from_ws: cutlass.Constexpr,
    ):
        """Sum the partials (split order), round, apply SiTU (gate/up) and store 8 bf16 columns per chunk."""
        out_cols = 64 if self.dual else self.cta_m
        chunks = out_cols // 8
        s_base = sP.iterator.toint()
        w_base = mWs.iterator.toint() + Int64(tile * self.split) * (self.cta_n * self.width * 4)
        slot = Int64(self.cta_n * self.width * 4)
        o_base = mOut.iterator.toint()
        ld = Int64(mOut.stride[0])
        cap = Float32(self.gate_cap)
        lin = Float32(self.linear_cap)
        for c in cutlass.range(tid, rows * chunks, _EPI_THREADS, unroll=1):
            row = c // chunks
            col = (c - row * chunks) * 8
            vals = []
            for half in cutlass.range_constexpr(self.n_acc):
                off = (row * self.width + half * 64 + col) * 4
                v = [Float32(0.0)] * 8
                if cutlass.const_expr(from_ws):
                    for s in cutlass.range_constexpr(self.split):
                        for j in cutlass.range_constexpr(2):
                            w = ld_cg_v4(w_base + slot * s + Int64(off + 16 * j))
                            for e in cutlass.range_constexpr(4):
                                x = bits_to_f32(w[e])
                                v[4 * j + e] = x if s == 0 else v[4 * j + e] + x
                else:
                    for j in cutlass.range_constexpr(2):
                        x = _ld_shared_v4(s_base + off + 16 * j)
                        for e in cutlass.range_constexpr(4):
                            v[4 * j + e] = x[e]
                vals.append([_bf16x2(v[2 * e], v[2 * e + 1]) for e in range(4)])
            words = vals[0]
            if cutlass.const_expr(self.dual):
                words = []
                for e in cutlass.range_constexpr(4):
                    g = _bf16x2_unpack(vals[0][e])
                    u = _bf16x2_unpack(vals[1][e])
                    words.append(_bf16x2(_situ(g[0], u[0], cap, lin), _situ(g[1], u[1], cap, lin)))
            addr = o_base + (Int64(row) * ld + Int64(tile * out_cols + col)) * 2
            _st_v4(addr, words[0], words[1], words[2], words[3])


_SCRATCH: dict = {}


def _scratch(device: torch.device, ws_floats: int, tiles: int, key: str) -> Dict[str, torch.Tensor]:
    cur = _SCRATCH.get((device, key))
    if cur is None or cur["ws"].numel() < ws_floats or cur["cnt"].numel() < tiles:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "shared_expert: size the workspace (workspace_spec) before graph capture"
            )
        cur = {
            "ws": torch.empty(max(ws_floats, 4), dtype=torch.float32, device=device),
            "cnt": torch.zeros(max(tiles, 1), dtype=torch.int32, device=device),
        }
        _SCRATCH[(device, key)] = cur
    return cur


def workspace_spec(ctas: int, max_rows: int = MAX_ROWS) -> Dict[str, Tuple[tuple, torch.dtype]]:
    """Largest split-K workspace and counter slab any M <= max_rows needs at this CTA cap, per GEMM."""
    spec = {}
    for name, dual in (("gate_up", True), ("down", False)):
        floats, tiles = 4, 1
        for m in (8, 16, 32, 64, 128):
            if m > _mma_n(max_rows):
                break
            t = default_tactic(m, ctas, dual)
            floats = max(floats, t.tiles(dual) * t.split * t.mma_n * t.width(dual))
            tiles = max(tiles, t.tiles(dual))
        spec[f"{name}_ws"] = ((floats,), torch.float32)
        spec[f"{name}_cnt"] = ((tiles,), torch.int32)
    return spec


def reserve(device: torch.device, ctas: int, max_rows: int = MAX_ROWS) -> None:
    """Size the module scratch for every M <= max_rows at this cap (call before graph capture)."""
    spec = workspace_spec(ctas, max_rows)
    for name in ("gate_up", "down"):
        _scratch(device, spec[f"{name}_ws"][0][0], spec[f"{name}_cnt"][0][0], name)


def _dyn(t: torch.Tensor, leading: int, align: int = 16):
    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=leading)


@cute.experimental.jit
def _launch(
    op: cutlass.Constexpr,
    w: cute.Tensor,
    x: cute.Tensor,
    out: cute.Tensor,
    ws: cute.Tensor,
    cnt: cute.Tensor,
    ctas: Int32,
    stream: _cuda.CUstream,
):
    op(
        cute.make_tensor(w.iterator, cute.select(w.layout, mode=[1, 2, 0])),
        cute.make_tensor(x.iterator, cute.select(x.layout, mode=[1, 2, 0])),
        out,
        ws,
        cnt,
        ctas,
        stream,
    )


def _args(w, x, out, ws, cnt):
    return (
        _dyn(w.unsqueeze(0), 2),
        _dyn(x.unsqueeze(0), 2),
        _dyn(out, 1),
        from_dlpack(ws, assumed_align=16).mark_layout_dynamic(),
        from_dlpack(cnt, assumed_align=4).mark_layout_dynamic(),
    )


@functools.lru_cache(maxsize=None)
def _compiled(
    device_index: int, tactic: Tactic, dual: bool, use_pdl: bool, caps: Tuple[float, float]
):
    with torch.cuda.device(device_index):
        dev = torch.device("cuda", device_index)
        k = HIDDEN if dual else INTER
        n_out = INTER if dual else HIDDEN
        w = torch.empty((2 * INTER) if dual else HIDDEN, k, dtype=torch.bfloat16, device=dev)
        x = torch.empty(2, k, dtype=torch.bfloat16, device=dev)
        out = torch.empty(2, n_out, dtype=torch.bfloat16, device=dev)
        ws = torch.empty(4, dtype=torch.float32, device=dev)
        cnt = torch.zeros(1, dtype=torch.int32, device=dev)
        kernel = SharedGemmKernel(
            tactic, dual=dual, use_pdl=use_pdl, gate_cap=caps[0], linear_cap=caps[1]
        )
        stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return cute_ext.compile(_launch, kernel, *_args(w, x, out, ws, cnt), Int32(1), stream)


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def _gemm(x, w, out, ctas, dual, tactic, workspace, enable_pdl, caps):
    m = x.shape[0]
    if m == 0:
        return
    t = tactic or default_tactic(m, ctas, dual)
    validate(t, dual, m)
    k = HIDDEN if dual else INTER
    n_out = INTER if dual else HIDDEN
    w_rows = 2 * INTER if dual else HIDDEN
    for name, a, shape in (("x", x, (m, k)), ("w", w, (w_rows, k))):
        if (
            tuple(a.shape) != shape
            or a.dtype != torch.bfloat16
            or not a.is_contiguous()
            or a.data_ptr() % 32
        ):
            raise ValueError(
                f"shared_expert: {name} must be contiguous 32-byte aligned bf16 {shape}"
            )
    if (
        out.dtype != torch.bfloat16
        or tuple(out.shape) != (m, n_out)
        or out.stride(1) != 1
        or out.stride(0) % 8
        or out.data_ptr() % 16
    ):
        raise ValueError(
            f"shared_expert: out must be bf16 [{m}, {n_out}] with 16-byte aligned rows"
        )
    units = t.tiles(dual) * t.split
    grid = min(ctas, units)
    name = "gate_up" if dual else "down"
    if workspace is not None:
        ws, cnt = workspace[f"{name}_ws"], workspace[f"{name}_cnt"]
        if ws.numel() < units * t.mma_n * t.width(dual) or cnt.numel() < t.tiles(dual):
            raise ValueError(f"shared_expert: workspace too small for {t}")
    else:
        s = _scratch(x.device, units * t.mma_n * t.width(dual), t.tiles(dual), name)
        ws, cnt = s["ws"], s["cnt"]
    fn = _compiled(x.get_device(), t, dual, enable_pdl, caps)
    fn(
        *_args(w, x, out, ws, cnt),
        Int32(grid),
        _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )


def gate_up(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    h: torch.Tensor,
    *,
    ctas: int,
    gate_cap: float = GATE_CAP,
    linear_cap: float = LINEAR_CAP,
    tactic: Optional[Tactic] = None,
    workspace: Optional[Dict[str, torch.Tensor]] = None,
    enable_pdl: Optional[bool] = None,
) -> None:
    """h [M, 6144] = situ(x @ Wg^T, x @ Wu^T) for x [M, 7168], w_gate_up [12288, 7168] = [Wg; Wu], on at most
    ``ctas`` CTAs."""
    pdl = _use_pdl() if enable_pdl is None else enable_pdl
    _gemm(x, w_gate_up, h, ctas, True, tactic, workspace, pdl, (float(gate_cap), float(linear_cap)))


def down(
    h: torch.Tensor,
    w_down: torch.Tensor,
    y: torch.Tensor,
    *,
    ctas: int,
    tactic: Optional[Tactic] = None,
    workspace: Optional[Dict[str, torch.Tensor]] = None,
    enable_pdl: Optional[bool] = None,
) -> None:
    """y [M, 7168] = h @ w_down^T for h [M, 6144], w_down [7168, 6144], on at most ``ctas`` CTAs."""
    pdl = _use_pdl() if enable_pdl is None else enable_pdl
    _gemm(h, w_down, y, ctas, False, tactic, workspace, pdl, (GATE_CAP, LINEAR_CAP))


def config(rows: int) -> Optional[int]:
    """CTA cap of the CuTe shared expert for a forward of ``rows`` tokens, or None for the stock path.

    ``TLLM_K3_MOE_SHARED=cute`` enables it (default off); ``TLLM_K3_MOE_SHARED_CTAS`` caps the grid of both GEMMs
    (default 128); ``TLLM_K3_MOE_SHARED_MAX_TOKENS`` (default 32) keeps larger batches on the stock path."""
    if os.environ.get("TLLM_K3_MOE_SHARED", "") != "cute":
        return None
    if not 1 <= rows <= int(os.environ.get("TLLM_K3_MOE_SHARED_MAX_TOKENS", "32")):
        return None
    return int(os.environ.get("TLLM_K3_MOE_SHARED_CTAS", "128"))


def run(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    ctas: int,
    gate_cap: float = GATE_CAP,
    linear_cap: float = LINEAR_CAP,
    before_down=None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Shared expert y [M, 7168] = down(situ(gate_up(x))) on the current stream, both GEMMs on at most ``ctas``
    CTAs. ``before_down()`` runs on the host between the two launches (e.g. an event wait that holds the down
    GEMM for a later window)."""
    m = x.shape[0]
    h = torch.empty(m, INTER, dtype=torch.bfloat16, device=x.device)
    gate_up(x, w_gate_up, h, ctas=ctas, gate_cap=gate_cap, linear_cap=linear_cap)
    if before_down is not None:
        before_down()
    y = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=x.device) if out is None else out
    down(h, w_down, y, ctas=ctas)
    return y
