# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The main loop follows kimi_k3_fgs/splitk_gemm_fgs.py, derived from FlashInfer v0.6.18 (39c024e2b5)
# flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py, Apache-2.0.
"""Kimi K3 decode latent down-projection with the MXFP8 dispatch quantize fused in, as one CuTe DSL kernel (SM107).

    y = bf16(x @ W^T),  q, sf = mxfp8(y)        x [M, 7168] bf16, W [3584, 7168] bf16, M <= 128 tokens

Stock runs a cuBLASLt split-K GEMM, its splitKreduce and the MXFP8 quantize (``quantize_with_block_size`` or the
fused route + quant kernel); this kernel replaces all three.

GEMM (swap-AB, tcgen05, kernel-M = the 3584 output columns, kernel-N = the tokens): the ``split_k`` CTAs of a
cluster share output tile m (``mma_m`` columns x all tokens) and each accumulates one K slice in fp32, weights
first. Each CTA leaves its partial in its own shared memory; after a cluster barrier CTA r reduces the 32-column
blocks r, r + split_k, ... of the tile, reading the ``split_k`` partials through DSMEM and adding them in rank order,
so the result is deterministic and the reduction is spread over the cluster. No workspace, no atomics.

Epilogue, on the bf16-rounded sums, with the stock MXFP8 recipe (``cvt_warp_fp16_to_mxfp8``): one UE8M0 scale per 32
columns = ``cvt.rp.satfinite.ue8m0(amax * rcp.approx.ftz(448))``, elements ``cvt.rn.satfinite.e4m3(y * 2^(127-e))``
(scale 1 for e = 0, 0 for an all-zero block). An e4m3 -0 (0x80) is stored as +0, so the output never holds 0x80;
this is the only difference from the stock bytes. Outputs: q [M, ldq] e4m3 bytes and sf [M, ldsf] linear UE8M0, row
strides free (the stock dispatch takes q [M, 3584] and sf [M, 112]).

Fine-grained sync (compile-time roles; FUSED-OFF compiles them out, the activation warp waits on
``griddepcontrol.wait``):

- ``fgs_in`` (edge ``mix_latent``): h is a bf16 -0-sentinel slot written by a running producer that stores whole
  16 B chunks and never -0. Weights stream from launch; the activation TMA checks element 0 of each 16 B chunk
  (``act_check``, TMA ``report_valid``) and warp 3 re-loads a stage that landed with sentinel data, without a
  ``griddepcontrol.wait`` before the loads (``act_wait`` keeps it: the ``ready`` diagnostic). The epilogue still
  waits for the PDL predecessor before it writes the outputs, which a previous launch's consumers
  may still read.
- ``fgs_out`` (edge ``latent_send``): after its stores every CTA adds 1 (release, after a CTA barrier) to
  ``ready[t]`` of every token t, so token t's q and sf rows are complete at ``ready[t] == m_tiles * split_k``
  (:func:`ready_target`). The caller zero-arms ``ready``.

Dependents launch after the activation warp's grid-dependency wait (``early_trigger``: at entry).
"""

from __future__ import annotations

import dataclasses
import functools
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

from .sync import release_add, spin_ns

HIDDEN = 7168
LATENT = 3584
SF_VEC = 32
MAX_ROWS = 128

_CTA_K = 128
_THREADS = 256
# Columns per reducing thread: one 16 B e4m3 store, half an MXFP8 block.
_VALS = 16
_SMEM_CAPACITY = utils.get_smem_capacity_in_bytes("sm_100")
_RETRY_WARP = 3
RETRY_SLOTS = 2


@dataclasses.dataclass(frozen=True)
class Tactic:
    mma_m: int
    mma_n: int
    split_k: int
    ab_stages: int
    # Request only this tactic's shared memory (two CTAs per SM when it fits) instead of the whole SM.
    exact_smem: bool = False

    @property
    def m_tiles(self) -> int:
        return LATENT // self.mma_m


def _smem_bytes(t: Tactic) -> int:
    ab = (t.mma_m + t.mma_n) * _CTA_K * 2 * t.ab_stages
    return ab + 3072 + (2 * t.ab_stages + 4) * 8


def validate(t: Tactic, m: int) -> None:
    if t.mma_m not in (64, 128) or t.mma_n not in (8, 16, 32, 64, 96, 128):
        raise ValueError(f"latent_down: unsupported tile {t}")
    if not 1 <= m <= t.mma_n:
        raise ValueError(f"latent_down: {m} rows do not fit mma_n {t.mma_n}")
    if not 1 <= t.split_k <= 8:
        raise ValueError(f"latent_down: split_k {t.split_k} is the cluster size, 1 .. 8")
    if t.mma_n * 4 > t.ab_stages * _CTA_K * 2:
        raise ValueError(f"latent_down: the A stages of {t} cannot hold the fp32 partial")
    if not 1 <= t.ab_stages <= HIDDEN // _CTA_K // t.split_k + 1:
        raise ValueError(f"latent_down: ab_stages {t.ab_stages} out of range")
    if _smem_bytes(t) > _SMEM_CAPACITY:
        raise ValueError(f"latent_down: {t} needs {_smem_bytes(t)} B of shared memory")


# (max rows, tactic), measured on SM107: ready-input CUDA-graph replays, weights cold in L2.
_TACTICS = (
    (8, Tactic(64, 8, 3, 8)),
    (16, Tactic(64, 16, 3, 8)),
    (32, Tactic(64, 32, 3, 7)),
    (64, Tactic(64, 64, 3, 6)),
    (96, Tactic(64, 96, 3, 5)),
    (128, Tactic(64, 128, 3, 4)),
)


def default_tactic(m: int) -> Tactic:
    override = os.environ.get("TLLM_K3_LATENT_DOWN_TACTIC")
    if override:
        return Tactic(*(int(v) for v in override.split(",")))  # m,n,split,stages[,exact_smem]
    for max_m, t in _TACTICS:
        if m <= max_m:
            return t
    raise ValueError(f"latent_down: at most {MAX_ROWS} rows, got {m}")


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
def _mapa(addr: Int32, rank: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip), Int32(rank).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            loc,
            ip,
            side_effects=False,
        )
    )


@dsl_user_op
def _ld_dsmem_v4(addr: Int32, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared::cluster.v4.f32 {$0, $1, $2, $3}, [$4];",
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
def _st_u8(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def _bf16x2_round(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
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


@dsl_user_op
def _mx_scale(amax: Float32, *, loc=None, ip=None):
    """UE8M0 byte and the element multiplier of a 32-wide block with the given amax (stock recipe)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [Float32(amax).ir_value(loc=loc, ip=ip)],
        "{\n"
        ".reg .f32 r, s;\n"
        ".reg .b16 e;\n"
        ".reg .b32 eb, sb;\n"
        ".reg .pred pe, pz;\n"
        "mov.f32 r, 0f43E00000;\n"
        "rcp.approx.ftz.f32 r, r;\n"
        "mul.f32 s, $2, r;\n"
        "cvt.rp.satfinite.ue8m0x2.f32 e, 0f00000000, s;\n"
        "cvt.u32.u16 eb, e;\n"
        "and.b32 eb, eb, 255;\n"
        "sub.u32 sb, 254, eb;\n"
        "shl.b32 sb, sb, 23;\n"
        "setp.eq.u32 pe, eb, 0;\n"
        "selp.b32 sb, 0x3F800000, sb, pe;\n"
        "setp.eq.f32 pz, $2, 0f00000000;\n"
        "selp.b32 sb, 0, sb, pz;\n"
        "mov.b32 $0, eb;\n"
        "mov.b32 $1, sb;\n"
        "}",
        "=r,=f,f",
        loc,
        ip,
        side_effects=False,
    )
    return Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)), Float32(
        llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)
    )


@dsl_user_op
def _e4m3x4(
    x0: Float32, x1: Float32, x2: Float32, x3: Float32, scale: Float32, *, loc=None, ip=None
) -> Int32:
    """Four e4m3 bytes of x * scale (x0 in the low byte), with -0 (0x80) stored as +0."""
    return Int32(
        _asm(
            T.i32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (x0, x1, x2, x3, scale)],
            "{\n"
            ".reg .f32 a, b, c, d;\n"
            ".reg .b16 l, h;\n"
            ".reg .b32 w, m;\n"
            "mul.f32 a, $1, $5;\n"
            "mul.f32 b, $2, $5;\n"
            "mul.f32 c, $3, $5;\n"
            "mul.f32 d, $4, $5;\n"
            "cvt.rn.satfinite.e4m3x2.f32 l, b, a;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h, d, c;\n"
            "mov.b32 w, {l, h};\n"
            "and.b32 m, w, 0x7F7F7F7F;\n"
            "add.u32 m, m, 0x7F7F7F7F;\n"
            "and.b32 m, m, 0x80808080;\n"
            "or.b32 m, m, 0x7F7F7F7F;\n"
            "and.b32 $0, w, m;\n"
            "}",
            "=r,f,f,f,f,f",
            loc,
            ip,
            side_effects=False,
        )
    )


@dsl_user_op
def _globaltimer_lo(*, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %globaltimer_lo;", "=r", loc, ip))


# Diagnostic timeline (``timeline``): per CTA, after the two retry counters, %globaltimer_lo at entry, first full
# stage, accumulator ready, partials published (cluster barrier), reduction stored, second cluster barrier.
TIMELINE_FIELDS = 6


class LatentDownKernel:
    def __init__(
        self,
        tactic: Tactic,
        *,
        use_pdl: bool,
        fgs_in: bool = False,
        act_check: bool = False,
        act_wait: bool = True,
        fgs_out: bool = False,
        bf16_out: bool = False,
        hazard_delay_ns: int = 0,
        early_trigger: bool = False,
        timeline: bool = False,
    ):
        if (fgs_in or fgs_out) and not use_pdl:
            raise ValueError("the FGS builds launch with PDL")
        if act_check and not fgs_in:
            raise ValueError("act_check needs the retry warp (fgs_in)")
        self.cta_m, self.cta_n, self.split_k = tactic.mma_m, tactic.mma_n, tactic.split_k
        self.stages = tactic.ab_stages
        self.use_pdl = use_pdl
        self.fgs_in = fgs_in
        self.report_valid = act_check
        self.act_wait = act_wait and use_pdl
        self.fgs_out = fgs_out
        self.bf16_out = bf16_out
        self.hazard_delay_ns = hazard_delay_ns
        self.early_trigger = early_trigger and use_pdl
        self.timeline = timeline
        self.acc_dtype = Float32
        self.mma_tiler_mn = (self.cta_m, self.cta_n)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
        self.launch_smem = _smem_bytes(tactic) if tactic.exact_smem else _SMEM_CAPACITY

    @cute.experimental.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mQ: cute.Tensor,
        mSf: cute.Tensor,
        mReady: cute.Tensor,
        mY: cute.Tensor,
        mRetry: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        self.kernel(mA, mB, mQ, mSf, mReady, mY, mRetry).launch(
            grid=(LATENT // self.cta_m, self.split_k, 1),
            block=(_THREADS, 1, 1),
            cluster=(1, self.split_k, 1),
            smem=cute.Int64(self.launch_smem),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (3584, 7168, 1) weights, K-major
        mB: cute.Tensor,  # (M, 7168, 1) activations, K-major
        mQ: cute.Tensor,  # uint8 [M, ldq]
        mSf: cute.Tensor,  # uint8 [M, ldsf]
        mReady: cute.Tensor,  # int32 [M] (fgs_out)
        mY: cute.Tensor,  # bf16 [M, 3584] (bf16_out)
        mRetry: cute.Tensor,  # int32 [2] activation re-issues (fgs_in)
    ):
        if cutlass.const_expr(self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()
        stages = self.stages
        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        mnk_tiler = (self.cta_m, self.cta_n, _CTA_K)
        m_tile = cute.arch.block_idx()[0]
        split = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        stamps = (
            mRetry.iterator
            + RETRY_SLOTS
            + (split * (LATENT // self.cta_m) + m_tile) * TIMELINE_FIELDS
        )
        if cutlass.const_expr(self.timeline):
            if cute.arch.thread_idx()[0] == 0:
                cute.arch.store(stamps, _globaltimer_lo())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=1024,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=1024,
        )
        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)

        def _mbars(n):
            return cute_ext.allocate(
                cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(n), alignment=8
            ).iterator

        bar_full = _mbars(stages)
        bar_empty = _mbars(stages)
        bar_mma_epilog = _mbars(1)
        bar_tmem_alloc = _mbars(1)
        tmem_base_ptr = cute_ext.allocate(
            Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        # This CTA's fp32 partial, [token][column], in the A stages (idle after the last MMA).
        sP = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=Float32),
            cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    if cutlass.const_expr(self.fgs_in):
                        # Weight and activation warps each arrive once; the empty barrier counts the MMA commit
                        # and the retry warp's release.
                        cute.arch.mbarrier_v1_init(bar_full + i, 2)
                        cute.arch.mbarrier_init(bar_empty + i, 2)
                    else:
                        cute.arch.mbarrier_init(bar_full + i, 2)
                        cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # K slices differ by at most one K tile; the lower ranks take the longer ones.
        k_tiles = cute.size(mA, mode=[1]) // _CTA_K
        k_base = k_tiles // self.split_k
        k_rem = k_tiles - k_base * self.split_k
        k_tile_count = k_base
        if split < k_rem:
            k_tile_count = k_base + 1
        k_tile_start = split * k_base + cutlass.min(split, k_rem)
        gB = cute.local_tile(mB, (self.cta_n, _CTA_K), (0, None, 0))
        b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")

        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                cute.local_tile(mA, (self.cta_m, _CTA_K), (m_tile, None, 0)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(bar_full, bar_empty, gB, sB, b_map, k_tile_start, k_tile_count, False)
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                _CTA_K // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
                stamps,
            )
        elif warp_idx == _RETRY_WARP:
            if cutlass.const_expr(self.fgs_in):
                self.retry_warp(
                    bar_full, bar_empty, gB, sB, b_map, k_tile_start, k_tile_count, mRetry
                )
        elif warp_idx >= 4:
            self.epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                sP,
                cute.arch.thread_idx()[0] - 128,
                stamps,
            )

        # Every partial of the tile is in its CTA's shared memory.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.timeline):
            if cute.arch.thread_idx()[0] == 0:
                cute.arch.store(stamps + 3, _globaltimer_lo())
        rows = cute.size(mB, mode=[0])
        self.reduce_quantize(sP, mQ, mSf, mY, m_tile, split, rows)
        if cutlass.const_expr(self.timeline):
            if cute.arch.thread_idx()[0] == 0:
                cute.arch.store(stamps + 4, _globaltimer_lo())
        if cutlass.const_expr(self.fgs_out):
            cute.arch.barrier()
            tid = cute.arch.thread_idx()[0]
            if tid < rows:
                release_add(mReady.iterator.toint() + Int64(tid) * 4, Int32(1))
        # Peers read this CTA's partial until they arrive. Their DSMEM loads have returned before they arrive
        # (the values feed their stores), so the arrive need not order anything else.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.timeline):
            if cute.arch.thread_idx()[0] == 0:
                cute.arch.store(stamps + 5, _globaltimer_lo())

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: Int32,
        k_tile_count: Int32,
        is_a: cutlass.Constexpr,
    ):
        stages = self.stages
        if cutlass.const_expr(not is_a and self.act_wait):
            cute.arch.griddepcontrol_wait()
        empty_phase = Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
                    ),
                )
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
                report_valid=(not is_a) and self.report_valid,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
        if cutlass.const_expr(not is_a and self.use_pdl):
            # Dependents launch only after this grid's own dependency wait.
            if cutlass.const_expr(not self.act_wait):
                cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(not self.early_trigger):
                cute.arch.griddepcontrol_launch_dependents()
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def retry_warp(
        self,
        bar_full,
        bar_empty,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: Int32,
        k_tile_count: Int32,
        mRetry: cute.Tensor,
    ):
        """Release each activation stage once its primary phase completes with valid data; re-arm and re-load a
        stage that landed with sentinel data. The k tiles in flight always lie in [base, base + stages) and tile t
        lives in stage t % stages; bit s of ``done`` marks stage s released in that window, bit s of ``parity`` is
        the primary-phase parity stage s completes next."""
        stages = self.stages
        tx_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        base = Int32(0)
        base_stage = Int32(0)
        done = Int32(0)
        parity = Int32(0)
        retries = Int32(0)
        while base < k_tile_count:
            stage = base_stage
            for i in cutlass.range(stages, unroll=1):
                bit = Int32(1) << stage
                if base + i < k_tile_count and (done & bit) == 0:
                    complete, invalid = cute.arch.mbarrier_try_wait_primary(
                        bar_full + stage, (parity >> stage) & 1
                    )
                    if complete:
                        parity = parity ^ bit
                        if invalid:
                            retries = retries + 1
                            # Re-arm with both producer arrivals; only the activation bytes are re-loaded.
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, tx_bytes)
                                cute.arch.mbarrier_arrive(bar_full + stage)
                            cute_ext.tma_load(
                                g_tile[None, None, k_tile_start + base + i],
                                s_tile[None, None, None, stage],
                                (bar_full + stage).value,
                                cta_v_map=cta_v_map,
                                tma_operation_type=self.tma_op,
                                update_expect_tx=False,
                                report_valid=self.report_valid,
                            )
                        else:
                            done = done | bit
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(bar_empty + stage)
                stage = stage + 1
                if stage == stages:
                    stage = Int32(0)
            while (done >> base_stage) & 1 == 1:
                done = done ^ (Int32(1) << base_stage)
                base = base + 1
                base_stage = base_stage + 1
                if base_stage == stages:
                    base_stage = Int32(0)
        with cute.arch.elect_one():
            if retries > 0:
                cute.arch.atomic_add(mRetry.iterator, retries)

    @cute.experimental.jit
    def mma_warp(
        self,
        bar_full,
        bar_empty,
        bar_mma_epilog,
        bar_tmem_alloc,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: Int32,
        stamps,
    ):
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase = Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.stages
            if cutlass.const_expr(self.fgs_in):
                cute.arch.mbarrier_wait_conditional(bar_full + stage, full_phase)
            else:
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            if cutlass.const_expr(self.timeline):
                if k_tile == 0:
                    with cute.arch.elect_one():
                        cute.arch.store(stamps + 1, _globaltimer_lo())
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                    accumulator,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
            if stage == self.stages - 1:
                full_phase = full_phase ^ 1
        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        sP: cute.Tensor,
        epi_tid: Int32,
        stamps,
    ):
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
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
            acc_view,
        )
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, cute.flat_divide(sP, epi_tile), epi_tid),
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        if cutlass.const_expr(self.timeline):
            if epi_tid == 0:
                cute.arch.store(stamps + 2, _globaltimer_lo())
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute_ext.partition_and_copy(thr_t2r, rAcc, sP)
        if cutlass.const_expr(self.fgs_in):
            # A previous launch's consumers may still read the outputs.
            cute.arch.griddepcontrol_wait()

    @cute.experimental.jit
    def reduce_quantize(
        self,
        sP: cute.Tensor,
        mQ: cute.Tensor,
        mSf: cute.Tensor,
        mY: cute.Tensor,
        m_tile: Int32,
        rank: Int32,
        rows: Int32,
    ):
        """CTA ``rank`` of the cluster sums, quantizes and stores the 32-column blocks p = rank, rank + split_k, ...
        of the tile ([token][block]), two threads per block, reading the split_k partials through DSMEM in rank
        order."""
        tid = cute.arch.thread_idx()[0]
        if cutlass.const_expr(self.hazard_delay_ns > 0):
            spin_ns(Int64(self.hazard_delay_ns))
        blocks_per_row = self.cta_m // SF_VEC
        blocks = rows * blocks_per_row
        mine = cute.ceil_div(blocks - rank, self.split_k)
        s_base = sP.iterator.toint()
        q_base = mQ.iterator.toint()
        sf_base = mSf.iterator.toint()
        ldq = Int64(mQ.stride[0])
        ldsf = Int64(mSf.stride[0])
        for it in cutlass.range(cute.ceil_div(mine, _THREADS // 2), unroll=1):
            k = it * (_THREADS // 2) + tid // 2
            valid = k < mine
            p = cutlass.min(rank + k * self.split_k, blocks - 1)
            row = p // blocks_per_row
            grp = (p - row * blocks_per_row) * 2 + tid % 2
            off = s_base + (row * self.cta_m + grp * _VALS) * 4
            acc = [Float32(0.0)] * _VALS
            for s in cutlass.range_constexpr(self.split_k):
                for j in cutlass.range_constexpr(_VALS // 4):
                    v = _ld_dsmem_v4(_mapa(off + 16 * j, Int32(s)))
                    for i in cutlass.range_constexpr(4):
                        acc[4 * j + i] = v[i] if s == 0 else acc[4 * j + i] + v[i]

            words = [_bf16x2_round(acc[2 * i], acc[2 * i + 1]) for i in range(_VALS // 2)]
            y = []
            for w in words:
                y.extend(_bf16x2_unpack(w))
            amax = Float32(0.0)
            for v in y:
                amax = cute.arch.fmax(amax, cute.arch.fmax(v, -v))
            amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
            e8m0, scale = _mx_scale(amax)
            q = [
                _e4m3x4(y[4 * i], y[4 * i + 1], y[4 * i + 2], y[4 * i + 3], scale)
                for i in range(_VALS // 4)
            ]

            if valid:
                col = m_tile * self.cta_m + grp * _VALS
                _st_v4(q_base + Int64(row) * ldq + col, q[0], q[1], q[2], q[3])
                if tid % 2 == 0:
                    _st_u8(sf_base + Int64(row) * ldsf + col // SF_VEC, e8m0)
                if cutlass.const_expr(self.bf16_out):
                    y_addr = mY.iterator.toint() + (Int64(row) * LATENT + col) * 2
                    _st_v4(y_addr, words[0], words[1], words[2], words[3])
                    _st_v4(y_addr + 16, words[4], words[5], words[6], words[7])


FGS_WAITS = ("mix_latent",)
FGS_PUBLISHES = ("latent_send",)


def workspace_spec(max_rows: int = MAX_ROWS) -> Dict[str, Tuple[tuple, torch.dtype]]:
    """Scratch of :func:`run`: the activation re-issue counters of the ``mix_latent`` consumer (diagnostic)."""
    return {"retry": ((RETRY_SLOTS,), torch.int32)}


_SCRATCH: dict = {}


def _scratch(device: torch.device) -> Dict[str, torch.Tensor]:
    if device not in _SCRATCH:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("latent_down: allocate the workspace before graph capture")
        _SCRATCH[device] = {
            "retry": torch.zeros(RETRY_SLOTS, dtype=torch.int32, device=device),
            "ready": torch.zeros(MAX_ROWS, dtype=torch.int32, device=device),
            "y": torch.empty(MAX_ROWS, LATENT, dtype=torch.bfloat16, device=device),
        }
    return _SCRATCH[device]


def _dyn(t: torch.Tensor, leading: int, align: int = 16):
    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=leading)


@cute.experimental.jit
def _launch(
    op: cutlass.Constexpr,
    w: cute.Tensor,
    x: cute.Tensor,
    q: cute.Tensor,
    sf: cute.Tensor,
    ready: cute.Tensor,
    y: cute.Tensor,
    retry: cute.Tensor,
    stream: _cuda.CUstream,
):
    op(
        cute.make_tensor(w.iterator, cute.select(w.layout, mode=[1, 2, 0])),
        cute.make_tensor(x.iterator, cute.select(x.layout, mode=[1, 2, 0])),
        q,
        sf,
        ready,
        y,
        retry,
        stream,
    )


def _args(w, x, q, sf, ready, y, retry):
    return (
        _dyn(w.unsqueeze(0), 2),
        _dyn(x.unsqueeze(0), 2),
        _dyn(q, 1),
        _dyn(sf, 1, align=1),
        from_dlpack(ready, assumed_align=4).mark_layout_dynamic(),
        _dyn(y, 1),
        from_dlpack(retry, assumed_align=4).mark_layout_dynamic(),
    )


@functools.lru_cache(maxsize=None)
def _compiled(device_index: int, tactic: Tactic, flags: Tuple[Tuple[str, object], ...]):
    with torch.cuda.device(device_index):
        dev = torch.device("cuda", device_index)
        scratch = _scratch(dev)
        w = torch.empty(256, HIDDEN, dtype=torch.bfloat16, device=dev)
        x = torch.empty(2, HIDDEN, dtype=torch.bfloat16, device=dev)
        q = torch.empty(2, LATENT, dtype=torch.uint8, device=dev)
        sf = torch.empty(2, LATENT // SF_VEC, dtype=torch.uint8, device=dev)
        kernel = LatentDownKernel(tactic, **dict(flags))
        stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = _args(w, x, q, sf, scratch["ready"], scratch["y"][:2], scratch["retry"])
        return cute_ext.compile(_launch, kernel, *args, stream)


def ready_target(rows: int, tactic: Optional[Tactic] = None) -> int:
    """Value ``latent_send``'s counter reaches for every token once its q and sf rows are stored."""
    t = tactic or default_tactic(rows)
    return t.m_tiles * t.split_k


def run(
    h: torch.Tensor,
    w_down: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    *,
    fgs=None,
    enable_pdl: bool = True,
    send=None,
    tactic: Optional[Tactic] = None,
    workspace: Optional[Dict[str, torch.Tensor]] = None,
    bf16_out: Optional[torch.Tensor] = None,
    early_trigger: bool = False,
    timeline: Optional[torch.Tensor] = None,
) -> None:
    """x, x_sf = mxfp8(bf16(h @ w_down^T)) for h [M, 7168] bf16 (M <= 128), w_down [3584, 7168] bf16.

    x: [M, >=3584] e4m3 (or uint8) and x_sf: [M, >=112] uint8, row-major with 16-byte aligned rows (any row stride).
    ``fgs`` (moe_chain ``StageSync``): waits ``mix_latent`` (h is the -0-armed slot), publishes ``latent_send``
    (int32 [>=M] counters, zero-armed, reach :func:`ready_target` per token). ``workspace``: buffers of
    :func:`workspace_spec`. ``bf16_out`` [M, 3584] also receives the bf16 GEMM output (tests).
    """
    if send is not None:
        raise NotImplementedError("latent_down: the send epilogue is not built yet")
    m = h.shape[0]
    if m == 0:
        return
    t = tactic or default_tactic(m)
    validate(t, m)
    if h.shape[1] != HIDDEN or w_down.shape != (LATENT, HIDDEN):
        raise ValueError(f"latent_down: h {tuple(h.shape)}, w_down {tuple(w_down.shape)}")
    for a in (h, w_down):
        if a.dtype != torch.bfloat16 or not a.is_contiguous() or a.data_ptr() % 32:
            raise ValueError("latent_down: h and w_down must be contiguous 32-byte aligned bf16")
    q8 = x.view(torch.uint8)
    if (
        q8.shape[0] < m
        or q8.shape[1] < LATENT
        or q8.stride(1) != 1
        or q8.data_ptr() % 16
        or q8.stride(0) % 16
    ):
        raise ValueError("latent_down: x must be [M, >=3584] bytes with 16-byte aligned rows")
    if (
        x_sf.dtype != torch.uint8
        or x_sf.shape[0] < m
        or x_sf.shape[1] < LATENT // SF_VEC
        or x_sf.stride(1) != 1
    ):
        raise ValueError("latent_down: x_sf must be uint8 [M, >=112]")
    if bf16_out is not None and (bf16_out.shape != (m, LATENT) or not bf16_out.is_contiguous()):
        raise ValueError("latent_down: bf16_out must be contiguous [M, 3584]")

    wait = fgs.waits.get("mix_latent") if fgs is not None else None
    pub = fgs.publishes.get("latent_send") if fgs is not None else None
    flags = {"use_pdl": bool(enable_pdl or wait or pub), "early_trigger": early_trigger}
    if wait is not None:
        if wait.tensor.data_ptr() != h.data_ptr():
            raise ValueError("latent_down: mix_latent waits on h itself (the armed slot)")
        skip = wait.diag == "oracle" or wait.hazard_nowait
        flags.update(fgs_in=True, act_check=not skip, act_wait=wait.diag == "ready")
    ready = None
    if pub is not None:
        ready = pub.tensor
        if ready.dtype != torch.int32 or ready.numel() < m or not ready.is_contiguous():
            raise ValueError("latent_down: latent_send needs a contiguous int32 counter per token")
        if pub.expected not in (0, ready_target(m, t)):
            raise ValueError(
                f"latent_down: latent_send reaches {ready_target(m, t)} per token, not {pub.expected}"
            )
        flags.update(fgs_out=True, hazard_delay_ns=pub.hazard_delay_ns)
    if bf16_out is not None:
        flags["bf16_out"] = True
    if timeline is not None:
        flags["timeline"] = True

    scratch = _scratch(h.device)
    ws = workspace if workspace is not None else scratch
    fn = _compiled(h.get_device(), t, tuple(sorted(flags.items())))
    fn(
        *_args(
            w_down,
            h,
            q8[:m],
            x_sf[:m],
            ready if ready is not None else scratch["ready"],
            bf16_out if bf16_out is not None else scratch["y"][:m],
            timeline if timeline is not None else ws["retry"],
        ),
        _cuda.CUstream(torch.cuda.current_stream(h.device).cuda_stream),
    )
