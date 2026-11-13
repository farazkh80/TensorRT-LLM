# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from FlashInfer (05ebb2d79)
# flashinfer/fused_moe/cute_dsl/blackwell/blockscaled_swapab_grouped_gemm.py, Apache-2.0.
"""Kimi K3 decode MoE FC1 (SM107): gathered MXFP8 rows x MXFP4 gate_up weights, SiTU-GLU, MXFP8 output.

Swap-AB block-scaled grouped GEMM: the MMA M operand is a 128-row tile of the expert's gate_up weights (TMA), the N
operand is up to two consecutive ``tile_n``-row token tiles (tile_n 8 / 16 / 32) of the same expert, gathered from the
received activations by cp.async warps (row ``permuted_idx_to_expanded_idx[p] // 16``). SM107 ``kind::mxf8f6f4`` with
K = 64 per instruction, reading the E2M1 weights from their packed shared-memory image.

The weights are read in place in the TRTLLM-Gen W4A8_MXFP4_MXFP8 layout: rows [up; gate] interleaved per row pair,
then permuted in 32-row blocks (physical row p of a block holds interleaved row 4 * (p % 8) + p // 8); the scales
follow the same row order in the 128x4 scale-factor atom the MMA consumes. In weight tile t, TMEM lane 32 * w + L
therefore holds intermediate channel 64 * t + 16 * w + 2 * (L % 8) + L // 16, as the up row when bit 3 of L is clear
and as the gate row otherwise (its partner is lane L ^ 8). An MXFP8 output group of 32 channels spans warps 2h and
2h + 1.

Work items are (token tiles, weight tile), weight tile fastest, dealt evenly over a persistent grid in token-tile
order, so a token tile's 48 weight tiles finish close together. Each output row depends only on its own activation row
(no cross-row reduction), so results are independent of the permutation and the tile composition.

Epilogue numerics: SiTU-GLU in fp32, act = (gate_cap * tanh(g / gate_cap) * sigmoid(g)) * (linear_cap *
tanh(u / linear_cap)), with tanh and sigmoid to about 2^-22 relative; per 32 channels the UE8M0 scale is
2^ceil(log2(amax / 448)) (byte 0 when amax is 0), and the elements are scaled by its exact inverse and rounded to E4M3
(RN, saturating). Rows past the tile's ``tile_idx_to_mn_limit`` are not written.

Routed build (``run_routed`` / ``run_v2``): the routing kernel fused in. Every CTA scans the received top-k ids into
its own expert-sorted pair list in shared memory and schedules exactly the items above from it; it also writes a share
of the RoutingPlan that FC2 and the finalize read. With ``prefetch`` the weight TMA warp issues the stages of a
speculative first unit at launch; with ``source`` the scan waits per row step on the all-to-all's per-source byte
counters.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from .. import edges as k3_edges
from .sync import (
    EARLY_TILES,
    atom_add_acq_rel,
    fence_fabric_acquire,
    fence_proxy_async_global,
    fgs_diag,
    ld_cg_b32,
    ld_cg_v4,
    ld_volatile_b32,
    ld_volatile_b64,
    match_any,
    spin_ns,
    st_b64,
    wait_at_least,
    wait_at_least_short,
    wait_ge64,
    wait_ne_sys,
)

HIDDEN = 3584
INTER = 3072
TOP_K = 16
SF_VEC = 32
W_ROWS = 2 * INTER
M_TILE = 128
NUM_M_TILES = W_ROWS // M_TILE
# fc1_done words per token tile: one per 256-channel K stage of FC2, complete at 4 (64-channel weight tiles).
DONE_PER_TILE = INTER // 256
SMEM_ARCH = "sm_107"
# Above this the L1 carveout drops to a few KB and the cp.async row gather slows down sharply.
SMEM_BUDGET = 227 * 1024
# A routed pair-list entry packs (local expert << 20) | expanded index (row * 16 + k).
IDX_BITS = 20
IDX_MASK = (1 << IDX_BITS) - 1
# Byte stride of the per-source dispatch byte counters in the all-to-all workspace.
COUNTER_STRIDE = 256
# Shared memory a routed FC1 CTA leaves free on its SM: it may spin on data that another kernel on this GPU produces.
CORESIDENT_SMEM = 8 * 1024


@dsl_user_op
def _f32_bits(x, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bits_f32(x, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Int32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ex2(x, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rcp(x, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
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
_LOG2E = 1.4426950408889634


def _tanh(y):
    """Branch-free tanh with about 2^-22 relative error: the polynomial below |y| = 0.6, 1 - 2 / (e^(2|y|) + 1)
    above, so neither side cancels."""
    ay = cute.arch.fmax(y, -y)
    big = cutlass.Float32(1.0) - cutlass.Float32(2.0) * _rcp(
        _ex2(ay * (2.0 * _LOG2E)) + cutlass.Float32(1.0)
    )
    big = cutlass.select_(y < cutlass.Float32(0.0), -big, big)
    y2 = y * y
    poly = cutlass.Float32(_TANH_POLY[-1])
    for c in reversed(_TANH_POLY[:-1]):
        poly = poly * y2 + cutlass.Float32(c)
    small = y + y * (y2 * poly)
    return cutlass.select_(ay < cutlass.Float32(0.6), small, big)


def _sigmoid(x):
    return _rcp(cutlass.Float32(1.0) + _ex2(x * (-_LOG2E)))


@dsl_user_op
def _e4m3x2(hi, lo, *, loc=None, ip=None):
    """Two fp32 -> E4M3 codes (round to nearest even, saturating), packed as (hi << 8) | lo in an Int32."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .b16 h;\ncvt.rn.satfinite.e4m3x2.f32 h, $1, $2;\ncvt.u32.u16 $0, h;\n}",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _tmem_load_16x256b(addrs, *, loc=None, ip=None):
    """One tcgen05.ld.16x256b.x1 per TMEM address, then tcgen05.wait::ld. For each address, thread T
    receives lane T // 4 (values 0, 1) and lane T // 4 + 8 (values 2, 3), both at columns 2 * (T % 4) and
    2 * (T % 4) + 1."""
    n = len(addrs)
    asm = []
    for k in range(n):
        regs = ", ".join(f"${4 * k + r}" for r in range(4))
        asm.append(f"tcgen05.ld.sync.aligned.16x256b.x1.b32 {{{regs}}}, [${4 * n + k}];")
    asm.append("tcgen05.wait::ld.sync.aligned;")
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * (4 * n)),
        [cutlass.Int32(a).ir_value(loc=loc, ip=ip) for a in addrs],
        "\n".join(asm),
        ",".join(["=f"] * (4 * n) + ["r"] * n),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [k], loc=loc, ip=ip)) for k in range(4 * n)
    ]


@dsl_user_op
def _red_add_gpu(addr, val, *, loc=None, ip=None):
    """Relaxed GPU-scope ``red.add`` on a global int32 word; ordering comes from a preceding fence."""
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tcgen05_fence(kind: str, *, loc=None, ip=None):
    """``tcgen05.fence::before_thread_sync`` / ``after_thread_sync`` around the TMEM accumulator handoff."""
    llvm.inline_asm(
        None,
        [],
        f"tcgen05.fence::{kind}_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bar_sync(barrier_id: int, num_threads: int, *, loc=None, ip=None):
    """Named barrier for warps that meet at different instruction sites (non-aligned form)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(barrier_id).ir_value(loc=loc, ip=ip),
            cutlass.Int32(num_threads).ir_value(loc=loc, ip=ip),
        ],
        "barrier.sync $0, $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dataclass(frozen=True)
class _UnalignedBarrier:
    barrier_id: int
    num_threads: int

    def arrive_and_wait(self):
        _bar_sync(self.barrier_id, self.num_threads)

    def wait_unaligned(self):
        _bar_sync(self.barrier_id, self.num_threads)


@dataclass(frozen=True)
class _CpAsyncUmmaPipeline(pipeline.PipelineAsync):
    """cp.async producer warp(s) -> tcgen05 MMA consumer: each producer thread arrives (noinc) on the full barrier
    when its copies have landed; the MMA releases a stage with tcgen05.commit."""

    cta_group: tcgen05.CtaGroup

    @staticmethod
    def create(num_stages: int, num_producer_threads: int, barrier_storage: cute.Pointer):
        full = pipeline.MbarrierArray(
            barrier_storage=barrier_storage,
            num_stages=num_stages,
            agent=(
                pipeline.PipelineOp.AsyncLoad,
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
        return _CpAsyncUmmaPipeline(full, empty, num_stages, None, None, tcgen05.CtaGroup.ONE)

    def consumer_release(self, state: pipeline.PipelineState):
        self.sync_object_empty.arrive(state.index, self.consumer_mask, self.cta_group)


class Fc1Kernel:
    """Persistent warp-specialized FC1. Warps: epilogue 0-3, MMA 4, weight TMA 5, scheduler 6, row gather 7 ..
    (1, 2 or 4 warps). An item multiplies one 128-row weight tile with up to two consecutive token tiles of the
    same expert (MMA N = 2 * n_tile), so a weight tile streams once per pair of token tiles."""

    def __init__(
        self,
        n_tile: int,
        gate_cap: float,
        linear_cap: float,
        use_pdl: bool,
        wait_plan: bool = False,
        signal_done: bool = False,
        max_ab_stages: int = 16,
        num_acc_stages: int = 2,
        num_tile_stages: int = 6,
        num_stored_stages: int = 4,
        delay_ns: int = 0,
        full_wait: bool = False,
        routed: bool = False,
        num_local: int = 0,
        max_rows: int = 0,
        prefetch: bool = False,
        source: Optional[Tuple] = None,
        plan_pub: bool = False,
        nowait: bool = False,
        lamport: bool = False,
    ):
        if n_tile not in (8, 16, 32):
            raise ValueError(f"n_tile must be 8, 16 or 32, got {n_tile}")
        if routed and (wait_plan or not 1 <= num_local <= 256 or max_rows < 1):
            raise ValueError(
                "routed FC1 builds its own plan: no plan_ready, 1..256 local experts, max_rows >= 1"
            )
        self.n_tile = n_tile
        self.n_mma = 2 * n_tile
        # Scale-factor blocks (of 32 K) per mainloop stage.
        self.k_blocks = 16 if self.n_mma == 16 else 8
        self.gate_cap = float(gate_cap)
        self.linear_cap = float(linear_cap)
        self.use_pdl = use_pdl
        self.wait_plan = wait_plan
        self.signal_done = signal_done
        self.routed = routed
        self.num_local = num_local
        self.em = max(64, (num_local + 31) // 32 * 32)
        self.max_rows = max_rows
        self.prefetch = prefetch and routed
        # source: (send_only, ep, payload bytes per token, then the workspace offsets of the round flag, the count
        # words, the per-source byte counters, their baselines and the self-row counter) of a dispatch that left the
        # data wait to this kernel. send_only: count words may still be -1, rows past a source's
        # count are stale, self rows land behind their own counter, and the combine advances the baselines.
        self.source = source if routed else None
        self.plan_pub = plan_pub and routed
        self.nowait = nowait
        if lamport and signal_done:
            raise ValueError("the Lamport FC1 -> FC2 build publishes no fc1_done counters")
        # lamport: act / act_sf are FC1 -> FC2 slots armed with 0x80 / 0xFF, which FC2 validates byte by byte.
        self.lamport = lamport
        self.delay_ns = delay_ns if (signal_done or self.plan_pub or lamport) else 0
        self.full_wait = full_wait
        self.info_fields = 6 if routed else 5
        self.max_ab_stages = max_ab_stages
        self.num_acc_stage = num_acc_stages
        self.num_tile_stage = num_tile_stages
        self.num_stored_stage = num_stored_stages
        self.sf_vec_size = SF_VEC
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)

        self.num_gather_warps = {16: 1, 32: 2, 64: 4}[self.n_mma]
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        # With signal_done, warp 7 publishes fc1_done so the epilogue never waits on a GPU-scope release.
        self.signal_warp_id = 7
        self.gather_warp_id = 8 if signal_done else 7
        self.threads_per_warp = 32
        self.threads_per_cta = 32 * (self.gather_warp_id + self.num_gather_warps)
        self.num_epilog_threads = 32 * len(self.epilog_warp_id)
        # Tile-info consumers: every warp but the scheduler.
        self.num_info_consumers = self.threads_per_cta - 32

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.num_epilog_threads
        )
        self.tmem_alloc_barrier = _UnalignedBarrier(
            barrier_id=3, num_threads=32 * (1 + len(self.epilog_warp_id))
        )
        # Routed FC1: every warp but the weight TMA warp scans the received ids.
        self.num_scan_warps = self.threads_per_cta // 32 - 1
        self.scan_barrier = pipeline.NamedBarrier(
            barrier_id=4, num_threads=32 * self.num_scan_warps
        )
        self.smem_capacity = min(utils.get_smem_capacity_in_bytes(SMEM_ARCH), SMEM_BUDGET)

    def _setup_attributes(self):
        self.mma_inst_shape_mn = (M_TILE, self.n_mma)
        # The row operand's scale factors keep the 128-row SF atom the S2T copy expects; rows [0, n_mma) are filled.
        self.mma_inst_shape_mn_sfb = (M_TILE, 128)
        tiled_mma = self._make_tiled_mma(self.mma_inst_shape_mn)
        tiled_mma_sfb = self._make_tiled_mma(self.mma_inst_shape_mn_sfb)
        self.mma_tiler = (M_TILE, self.n_mma, SF_VEC * self.k_blocks)
        self.mma_tiler_sfb = (M_TILE, 128, self.mma_tiler[2])
        self.k_stage = self.mma_tiler[2]
        self.num_k_tiles = HIDDEN // self.k_stage
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )

        a_one = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, cutlass.Float4E2M1FN, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, 1)
        sfa_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        sfb_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_sfb, self.mma_tiler_sfb, self.sf_vec_size, 1
        )
        buffer_bytes = (
            cute.size_in_bytes(cutlass.Float4E2M1FN, a_one),
            cute.size_in_bytes(cutlass.Float8E4M3FN, b_one),
            cute.size_in_bytes(cutlass.Float8E8M0FNU, sfa_one),
            cute.size_in_bytes(cutlass.Float8E8M0FNU, sfb_one),
        )
        stage_bytes = sum(buffer_bytes)
        # Barriers, the tile-info and row rings, the amax exchange, 1 KB alignment of the four stage buffers, slack.
        reserved = (
            2048
            + (self.info_fields + self.n_mma) * 4 * self.num_tile_stage
            + 2 * 4 * self.n_mma * 4
            + 4 * 1024
            + 2048
        )
        self.num_ab_stage = min((self.smem_capacity - reserved) // stage_bytes, self.max_ab_stages)
        if self.num_ab_stage < 2:
            raise ValueError("not enough shared memory for two mainloop stages")
        self.seg = 0
        self.sorted_cap = 0
        self.mask_base = 0
        if self.routed:
            self._size_routed(buffer_bytes)
        # Weight stages a routed CTA issues for its speculative first unit before the plan exists.
        self.num_spec_stages = min(self.num_ab_stage, HIDDEN // (SF_VEC * self.k_blocks))

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.Float4E2M1FN, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_sfb, self.mma_tiler_sfb, self.sf_vec_size, self.num_ab_stage
        )

        sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma_sfb,
            self.mma_tiler_sfb,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )
        # TMEM addresses carry the column in the low 16 bits; each SF region starts on a 16-column boundary.
        self.num_sfa_tmem_cols = cute.round_up(
            cute.cosize(cute.recast_layout(32, 8, sfa_tmem_layout)) & 0xFFFF, 16
        )
        self.num_sfb_tmem_cols = cute.round_up(
            cute.cosize(cute.recast_layout(32, 8, sfb_tmem_layout)) & 0xFFFF, 16
        )
        self.num_acc_tmem_cols = self.n_mma * self.num_acc_stage
        total = self.num_acc_tmem_cols + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        alloc = 32
        while alloc < total:
            alloc *= 2
        self.num_tmem_alloc_cols = alloc

    def _size_routed(self, buffer_bytes: Tuple[int, ...]):
        """Routed FC1 shared memory, sized exactly (the estimate above is loose): per-source targets,
        per-(expert, scan warp) counts, segment lengths, counts, both offset arrays, the scheduler's row scratch and
        flags; one int32 pair-list segment per scan warp and the uint16 expert-sorted list, each at least the largest
        R of the bucket (one local pair per row, Balanced routing). Mainloop stages are the most that still fit with
        ``CORESIDENT_SMEM`` left for a small kernel on the same SM (the dispatch or combine CTAs that FC1 waits on)."""

        def up(x, a):
            return (x + a - 1) // a * a

        nws = self.num_scan_warps
        nts = self.num_tile_stage

        def main_bytes(stages):
            head = up((self.info_fields + self.n_mma) * 4 * nts, 16) + 2 * 4 * self.n_mma * 4
            head = up(head, 8) + 8 * (
                4 * stages + 2 * self.num_acc_stage + 2 * nts + 2 * self.num_stored_stage
            )
            return up(head + 4, 1024) + sum(up(stages * b, 1024) for b in buffer_bytes)

        fixed = (
            128 + 12 * 64 + 4 * (self.em * nws + nws + self.em + 2 * (self.em + 16) + 64 + 16) + 64
        )
        seg_min = up(-(-self.max_rows // nws), 32)
        capacity = self.smem_capacity - CORESIDENT_SMEM
        stages = self.max_ab_stages
        while (
            stages > 2
            and main_bytes(stages) + fixed + 4 * nws * seg_min + 2 * self.max_rows > capacity
        ):
            stages -= 1
        self.num_ab_stage = stages
        left = capacity - main_bytes(stages) - fixed
        self.seg = max(seg_min, left // 2 // (4 * nws) // 32 * 32)
        self.sorted_cap = (left - 4 * nws * self.seg) // 2 // 8 * 8
        # When the segments overflow, the sorted list reuses the whole list region; per-(expert, scan warp) lane
        # masks take its last em * nws words.
        self.mask_base = nws * self.seg + self.sorted_cap // 2 - self.em * nws
        assert self.mask_base > 0, self.mask_base

    def _make_tiled_mma(self, inst_shape_mn: Tuple[int, int]):
        return sm107_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            self.sf_vec_size,
            self.cta_group,
            (*inst_shape_mn, 64),
        )

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        w_sf_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        x_sf_ptr: cute.Pointer,
        act_ptr: cute.Pointer,
        act_sf_ptr: cute.Pointer,
        p2e_ptr: cute.Pointer,
        t2e_ptr: cute.Pointer,
        t2l_ptr: cute.Pointer,
        ntiles_ptr: cute.Pointer,
        plan_ready_ptr: cute.Pointer,
        fc1_done_ptr: cute.Pointer,
        ids_ptr: cute.Pointer,
        counts_ptr: cute.Pointer,
        offsets_ptr: cute.Pointer,
        e2p_ptr: cute.Pointer,
        total_ptr: cute.Pointer,
        plan_pub_ptr: cute.Pointer,
        num_experts: cutlass.Int32,
        num_rows: cutlass.Int32,
        num_permuted: cutlass.Int32,
        num_tiles_cap: cutlass.Int32,
        num_ctas: cutlass.Int32,
        local_expert_offset: cutlass.Int32,
        my_ws: cutlass.Int64,
        max_tokens: cutlass.Int32,
        rank: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        # Weights [E, 2I, H/2] bytes as E2M1 (2I, H, E), K-major.
        a = cute.make_tensor(
            w_ptr,
            cute.make_layout((W_ROWS, HIDDEN, num_experts), stride=(HIDDEN, 1, W_ROWS * HIDDEN)),
        )
        sfa = cute.make_tensor(
            w_sf_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                (W_ROWS, HIDDEN, num_experts), self.sf_vec_size
            ),
        )
        x = cute.make_tensor(x_ptr, cute.make_layout((num_rows, HIDDEN), stride=(HIDDEN, 1)))
        x_sf = cute.make_tensor(
            x_sf_ptr, cute.make_layout((num_rows, HIDDEN // SF_VEC), stride=(HIDDEN // SF_VEC, 1))
        )
        act = cute.make_tensor(act_ptr, cute.make_layout((num_permuted, INTER), stride=(INTER, 1)))
        act_sf = cute.make_tensor(
            act_sf_ptr,
            cute.make_layout((num_permuted, INTER // SF_VEC), stride=(INTER // SF_VEC, 1)),
        )
        p2e = cute.make_tensor(p2e_ptr, cute.make_layout((num_permuted,)))
        t2e = cute.make_tensor(t2e_ptr, cute.make_layout((num_tiles_cap,)))
        t2l = cute.make_tensor(t2l_ptr, cute.make_layout((num_tiles_cap,)))
        ntiles = cute.make_tensor(ntiles_ptr, cute.make_layout((1,)))
        plan_ready = cute.make_tensor(plan_ready_ptr, cute.make_layout((1,)))
        fc1_done = cute.make_tensor(
            fc1_done_ptr, cute.make_layout((num_tiles_cap * DONE_PER_TILE,))
        )
        ids = cute.make_tensor(ids_ptr, cute.make_layout((num_rows * TOP_K,)))
        counts = cute.make_tensor(counts_ptr, cute.make_layout((self.em,)))
        offsets = cute.make_tensor(offsets_ptr, cute.make_layout((self.em + 1,)))
        e2p = cute.make_tensor(e2p_ptr, cute.make_layout((num_rows * TOP_K,)))
        total = cute.make_tensor(total_ptr, cute.make_layout((1,)))
        plan_pub = cute.make_tensor(plan_pub_ptr, cute.make_layout((1,)))

        tiled_mma = self._make_tiled_mma(self.mma_inst_shape_mn)
        tiled_mma_sfb = self._make_tiled_mma(self.mma_inst_shape_mn_sfb)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        self.num_tma_load_bytes = cute.size_in_bytes(
            cutlass.Float4E2M1FN, a_smem_layout
        ) + cute.size_in_bytes(cutlass.Float8E8M0FNU, sfa_smem_layout)

        @cute.struct
        class SharedStorage:
            info: cute.struct.MemRange[cutlass.Int32, self.info_fields * self.num_tile_stage]
            rows: cute.struct.MemRange[cutlass.Int32, self.n_mma * self.num_tile_stage]
            amax: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 2 * 4 * self.n_mma], 16]
            ab_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            info_mbar: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            stored_mbar: cute.struct.MemRange[cutlass.Int64, self.num_stored_stage * 2]
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float4E2M1FN, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E4M3FN, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                1024,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU, cute.cosize(self.sfa_smem_layout_staged)
                ],
                1024,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E8M0FNU, cute.cosize(self.sfb_smem_layout_staged)
                ],
                1024,
            ]

        self.shared_storage = SharedStorage
        smem_bytes = self.shared_storage.size_in_bytes()
        nws = self.num_scan_warps
        em = self.em

        @cute.struct
        class RoutingStorage:
            target: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 64], 16]
            src_cnt: cute.struct.MemRange[cutlass.Int32, 64]
            wcnt: cute.struct.MemRange[cutlass.Int32, em * nws]
            seg_len: cute.struct.MemRange[cutlass.Int32, nws]
            cnt: cute.struct.MemRange[cutlass.Int32, em]
            off: cute.struct.MemRange[cutlass.Int32, em + 16]
            uoff: cute.struct.MemRange[cutlass.Int32, em + 16]
            pair_rows: cute.struct.MemRange[cutlass.Int32, 64]
            misc: cute.struct.MemRange[cutlass.Int32, 16]
            lists: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, max(nws * self.seg + self.sorted_cap // 2, 32)],
                16,
            ]

        self.routing_storage = RoutingStorage
        if cutlass.const_expr(self.routed):
            smem_bytes = smem_bytes + 128 + self.routing_storage.size_in_bytes()

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_sfa,
            tma_tensor_sfa,
            x,
            x_sf,
            act,
            act_sf,
            p2e,
            t2e,
            t2l,
            ntiles,
            plan_ready,
            fc1_done,
            ids,
            counts,
            offsets,
            e2p,
            total,
            plan_pub,
            num_ctas,
            local_expert_offset,
            my_ws,
            max_tokens,
            rank,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            smem=smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    def _s2t_copy_and_partition(self, sSF: cute.Tensor, tSF: cute.Tensor):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), cutlass.Float8E8M0FNU
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def _plan_in_cta(
        self,
        scan_warp: cutlass.Int32,
        lane: cutlass.Int32,
        ids_addr: cutlass.Int64,
        rows: cutlass.Int32,
        lo: cutlass.Int32,
        my_ws: cutlass.Int64,
        max_tokens: cutlass.Int32,
        rank: cutlass.Int32,
        sTarget: cute.Tensor,
        sSrcCnt: cute.Tensor,
        sWCnt: cute.Tensor,
        sSegLen: cute.Tensor,
        sCnt: cute.Tensor,
        sOff: cute.Tensor,
        sUoff: cute.Tensor,
        sMisc: cute.Tensor,
        sPairs: cute.Tensor,
        sSorted: cute.Tensor,
    ):
        """Every scan warp: this CTA's copy of the routing. Scan warp w owns a contiguous block of rows, read 32 rows
        at a time (a row per lane), and appends their local (row, k) pairs to its segment in ascending expanded index.
        Then a counting sort without atomics: each warp counts its segment per local expert (``match.any``, one leader
        lane per expert), the per-(expert, warp) counts become exclusive bases, and each warp places its entries into
        ``sSorted`` (expert-major, ascending expanded index within an expert). Also writes the per-expert counts, the
        tile-padded and the unpadded expert offsets, ``sMisc[0]`` (the list mode) and ``sMisc[1]`` (where the sorted
        list starts in ``sSorted``). Mode 0: as above. Mode 1, a segment or the sorted list ran out of room: each warp
        reads its rows' ids again and places them straight into a sorted list that takes the whole region. Mode 2,
        not even that list holds every pair: the plan is taken from the received ids directly."""
        E = self.num_local
        TN = self.n_tile
        nws = self.num_scan_warps
        seg = self.seg
        lanes_below = cutlass.Int32(cute.arch.lanemask_lt())
        st = scan_warp * 32 + lane
        if cutlass.const_expr(self.source is not None):
            send_only, ep, bytes_per_token, off_flag, off_recv, _, off_base, _ = self.source
            if st < ep:
                flag = ld_volatile_b32(my_ws + off_flag)
                parity = ((flag - 1) >> 1) & 1
                at = my_ws + off_recv + cutlass.Int64(parity * ep + st) * 4
                if cutlass.const_expr(send_only):
                    cnt = wait_ne_sys(at, cutlass.Int32(-1))
                else:
                    cnt = ld_volatile_b32(at)
                sSrcCnt[st] = cnt
                base = ld_volatile_b64(my_ws + off_base + cutlass.Int64(st) * 8)
                sTarget[st] = base + cutlass.Int64(cnt) * cutlass.Int64(bytes_per_token)
            self.scan_barrier.arrive_and_wait()

        rows_per_warp = (rows + nws * 32 - 1) // (nws * 32) * 32
        r0 = scan_warp * rows_per_warp
        r_end = cutlass.min(rows, r0 + rows_per_warp)
        raw = cute.make_rmem_tensor((2 * TOP_K,), cutlass.Int32)
        n_seg = cutlass.Int32(0)
        # Two 32-row steps per iteration, both rows' loads in flight before either is processed.
        for s in cutlass.range(r0, r_end, 64, unroll=1):
            if cutlass.const_expr(self.source is not None and not self.nowait):
                send_only, off_cnt, off_self = self.source[0], self.source[5], self.source[7]
                src = s // max_tokens + lane
                if src <= (cutlass.min(s + 64, r_end) - 1) // max_tokens:
                    if src != rank:
                        wait_ge64(
                            my_ws + off_cnt + cutlass.Int64(src) * COUNTER_STRIDE, sTarget[src]
                        )
                    elif cutlass.const_expr(send_only):
                        wait_at_least_short(my_ws + off_self, sSrcCnt[src])
                # Lanes read rows other lanes waited for: every lane fences after the whole warp's waits.
                cute.arch.sync_warp()
                fence_fabric_acquire()
            self._load_ids(raw, s, r_end, lane, ids_addr, max_tokens, sSrcCnt)
            for h in cutlass.range_constexpr(2):
                row = s + 32 * h + lane
                n = cutlass.Int32(0)
                for k in cutlass.range_constexpr(TOP_K):
                    v = raw[TOP_K * h + k] - lo
                    n = n + cutlass.select_((v >= 0) & (v < E), cutlass.Int32(1), cutlass.Int32(0))
                x = n
                for i in cutlass.range_constexpr(5):
                    t = cute.arch.shuffle_sync_up(x, offset=1 << i)
                    if lane >= (1 << i):
                        x = x + t
                pos = n_seg + x - n
                for k in cutlass.range_constexpr(TOP_K):
                    v = raw[TOP_K * h + k] - lo
                    if (v >= 0) & (v < E):
                        if pos < seg:
                            sPairs[scan_warp * seg + pos] = (v << IDX_BITS) | (row * TOP_K + k)
                        pos = pos + 1
                n_seg = n_seg + cute.arch.shuffle_sync(x, 31)

        # Per-(expert, warp) counts: from the warp's segment, or from its rows when the segment overflowed.
        for i in cutlass.range_constexpr(self.em // 32):
            sWCnt[(lane + 32 * i, scan_warp)] = cutlass.Int32(0)
        cute.arch.sync_warp()
        if n_seg <= seg:
            for b in cutlass.range(0, n_seg, 32, unroll=1):
                e = cutlass.Int32(E)
                if b + lane < n_seg:
                    e = sPairs[scan_warp * seg + b + lane] >> IDX_BITS
                peers = match_any(e)
                if (e < E) & ((peers & lanes_below) == 0):
                    sWCnt[(e, scan_warp)] = sWCnt[(e, scan_warp)] + cute.arch.popc(peers)
                cute.arch.sync_warp()
        else:
            for s in cutlass.range(r0, r_end, 64, unroll=1):
                self._load_ids(raw, s, r_end, lane, ids_addr, max_tokens, sSrcCnt)
                for k in cutlass.range_constexpr(2 * TOP_K):
                    v = raw[k] - lo
                    local = (v >= 0) & (v < E)
                    peers = match_any(cutlass.select_(local, v, cutlass.Int32(E)))
                    if local & ((peers & lanes_below) == 0):
                        sWCnt[(v, scan_warp)] = sWCnt[(v, scan_warp)] + cute.arch.popc(peers)
                    cute.arch.sync_warp()
        if lane == 0:
            sSegLen[scan_warp] = n_seg
        self.scan_barrier.arrive_and_wait()

        # Warp 0: counts (the per-(expert, warp) counts become exclusive bases within the expert), the overflow flag,
        # then the tile-padded and the unpadded expert offsets.
        if scan_warp == 0:
            per = self.em // 32
            own = cutlass.Int32(0)
            own_u = cutlass.Int32(0)
            for i in cutlass.range_constexpr(per):
                c = cutlass.Int32(0)
                for w in cutlass.range_constexpr(nws):
                    wc = sWCnt[(lane * per + i, w)]
                    sWCnt[(lane * per + i, w)] = c
                    c = c + wc
                sCnt[lane * per + i] = c
                own = own + (c + (TN - 1)) // TN * TN
                own_u = own_u + c
            x = own
            xu = own_u
            for i in cutlass.range_constexpr(5):
                t = cute.arch.shuffle_sync_up(x, offset=1 << i)
                tu = cute.arch.shuffle_sync_up(xu, offset=1 << i)
                if lane >= (1 << i):
                    x = x + t
                    xu = xu + tu
            run = x - own
            run_u = xu - own_u
            for i in cutlass.range_constexpr(per):
                c = sCnt[lane * per + i]
                sOff[lane * per + i] = run
                sUoff[lane * per + i] = run_u
                run = run + (c + (TN - 1)) // TN * TN
                run_u = run_u + c
            if lane == 31:
                sOff[self.em] = x
                sUoff[self.em] = xu
                over = cutlass.select_(xu > self.sorted_cap, cutlass.Int32(1), cutlass.Int32(0))
                for w in cutlass.range_constexpr(nws):
                    over = over | cutlass.select_(
                        sSegLen[w] > seg, cutlass.Int32(1), cutlass.Int32(0)
                    )
                mode = cutlass.select_(
                    over == 0,
                    cutlass.Int32(0),
                    cutlass.select_(xu > 2 * self.mask_base, cutlass.Int32(2), cutlass.Int32(1)),
                )
                sMisc[0] = mode
                sMisc[1] = cutlass.select_(
                    mode == 0, cutlass.Int32(2 * nws * seg), cutlass.Int32(0)
                )
        self.scan_barrier.arrive_and_wait()

        mode = sMisc[0]
        if mode == 0:
            for b in cutlass.range(0, n_seg, 32, unroll=1):
                ent = cutlass.Int32(-1)
                e = cutlass.Int32(E)
                if b + lane < n_seg:
                    ent = sPairs[scan_warp * seg + b + lane]
                    e = ent >> IDX_BITS
                peers = match_any(e)
                if e < E:
                    at = sUoff[e] + sWCnt[(e, scan_warp)] + cute.arch.popc(peers & lanes_below)
                    sSorted[2 * nws * seg + at] = cutlass.Uint16(ent & IDX_MASK)
                cute.arch.sync_warp()
                if (e < E) & ((peers & lanes_below) == 0):
                    sWCnt[(e, scan_warp)] = sWCnt[(e, scan_warp)] + cute.arch.popc(peers)
                cute.arch.sync_warp()
        elif mode == 1:
            # A row holds an expert at most once, so an expert's rank within a 32-row step is the number of lower
            # lanes holding it at any k: OR the per-k match masks into sMask first, then place, then advance the base
            # from the lowest such lane (one writer per expert).
            sMask = cute.make_tensor(
                sPairs.iterator + self.mask_base,
                cute.make_layout((self.em, nws), stride=(1, self.em)),
            )
            for i in cutlass.range_constexpr(self.em // 32):
                sMask[(lane + 32 * i, scan_warp)] = cutlass.Int32(0)
            cute.arch.sync_warp()
            for s in cutlass.range(r0, r_end, 64, unroll=1):
                self._load_ids(raw, s, r_end, lane, ids_addr, max_tokens, sSrcCnt)
                for h in cutlass.range_constexpr(2):
                    for k in cutlass.range_constexpr(TOP_K):
                        v = raw[TOP_K * h + k] - lo
                        local = (v >= 0) & (v < E)
                        peers = match_any(cutlass.select_(local, v, cutlass.Int32(E)))
                        if local & ((peers & lanes_below) == 0):
                            sMask[(v, scan_warp)] = sMask[(v, scan_warp)] | peers
                        cute.arch.sync_warp()
                    lowest = cutlass.Int32(0)
                    for k in cutlass.range_constexpr(TOP_K):
                        v = raw[TOP_K * h + k] - lo
                        if (v >= 0) & (v < E):
                            m = sMask[(v, scan_warp)]
                            at = sUoff[v] + sWCnt[(v, scan_warp)] + cute.arch.popc(m & lanes_below)
                            sSorted[at] = cutlass.Uint16((s + 32 * h + lane) * TOP_K + k)
                            if (m & lanes_below) == 0:
                                lowest = lowest | (1 << k)
                    cute.arch.sync_warp()
                    for k in cutlass.range_constexpr(TOP_K):
                        if ((lowest >> k) & 1) == 1:
                            v = raw[TOP_K * h + k] - lo
                            sWCnt[(v, scan_warp)] = sWCnt[(v, scan_warp)] + cute.arch.popc(
                                sMask[(v, scan_warp)]
                            )
                            sMask[(v, scan_warp)] = cutlass.Int32(0)
                    cute.arch.sync_warp()
        self.scan_barrier.arrive_and_wait()
        if cutlass.const_expr(self.source is not None):
            fence_fabric_acquire()
            fence_proxy_async_global()

    @cute.jit
    def _load_ids(
        self,
        raw: cute.Tensor,
        s: cutlass.Int32,
        r_end: cutlass.Int32,
        lane: cutlass.Int32,
        ids_addr: cutlass.Int64,
        max_tokens: cutlass.Int32,
        sSrcCnt: cute.Tensor,
    ):
        """The top-k ids of rows s + lane and s + 32 + lane into raw[0:16] and raw[16:32], all loads in flight
        together (-1 for a row that is not a received token)."""
        for h in cutlass.range_constexpr(2):
            row = s + 32 * h + lane
            for k in cutlass.range_constexpr(TOP_K):
                raw[TOP_K * h + k] = cutlass.Int32(-1)
            if self._row_valid(row, r_end, max_tokens, sSrcCnt):
                for q in cutlass.range_constexpr(TOP_K // 4):
                    w = ld_cg_v4(ids_addr + (cutlass.Int64(row) * TOP_K + 4 * q) * 4)
                    for i in cutlass.range_constexpr(4):
                        raw[TOP_K * h + 4 * q + i] = w[i]

    @cute.jit
    def _row_valid(
        self,
        row: cutlass.Int32,
        rows: cutlass.Int32,
        max_tokens: cutlass.Int32,
        sSrcCnt: cute.Tensor,
    ) -> cutlass.Boolean:
        """Row ``row`` < ``rows`` is a received token: with a source, its slot within the source block is below the
        block's count (after a send-only dispatch later slots hold stale rows, not -1 ids)."""
        ok = row < rows
        if cutlass.const_expr(self.source is not None):
            src = cutlass.min(row // max_tokens, cutlass.Int32(63))
            ok = ok & (row - (row // max_tokens) * max_tokens < sSrcCnt[src])
        return ok

    @cute.jit
    def _walk_ids(
        self,
        e: cutlass.Int32,
        first_rank: cutlass.Int32,
        collect: cutlass.Constexpr,
        lane: cutlass.Int32,
        ids_addr: cutlass.Int64,
        rows: cutlass.Int32,
        lo: cutlass.Int32,
        max_tokens: cutlass.Int32,
        sSrcCnt: cute.Tensor,
        sPairRows: cute.Tensor,
        mP2e: cute.Tensor,
        mE2p: cute.Tensor,
        off: cutlass.Int32,
    ):
        """Overflow path (more local pairs than the shared-memory lists hold): a warp-wide walk over the received ids
        for the pairs of expert ``e`` in ascending expanded index (a row holds an expert at most once). ``collect``:
        the source row of each pair of rank first_rank .. first_rank + n_mma - 1 into ``sPairRows``; otherwise the
        expert's plan slice, permuted_idx_to_expanded_idx[off + rank] and its inverse."""
        lanes_below = cutlass.Int32(cute.arch.lanemask_lt())
        running = cutlass.Int32(0)
        want = lo + e
        for b in cutlass.range(0, rows, 32, unroll=1):
            row = b + lane
            expanded = cutlass.Int32(-1)
            if self._row_valid(row, rows, max_tokens, sSrcCnt):
                for q in cutlass.range_constexpr(TOP_K // 4):
                    w = ld_cg_v4(ids_addr + (cutlass.Int64(row) * TOP_K + 4 * q) * 4)
                    for i in cutlass.range_constexpr(4):
                        if w[i] == want:
                            expanded = row * TOP_K + 4 * q + i
            hit = expanded >= 0
            ballot = cute.arch.vote_ballot_sync(hit)
            rank = running + cute.arch.popc(ballot & lanes_below)
            if cutlass.const_expr(collect):
                if hit & (rank >= first_rank) & (rank < first_rank + self.n_mma):
                    sPairRows[rank - first_rank] = row
            else:
                if hit:
                    mP2e[off + rank] = expanded
                    mE2p[expanded] = off + rank
            running = running + cute.arch.popc(ballot)

    @cute.jit
    def _publish_plan(
        self,
        tidx: cutlass.Int32,
        bidx: cutlass.Int32,
        num_ctas: cutlass.Int32,
        ids_addr: cutlass.Int64,
        rows: cutlass.Int32,
        lo: cutlass.Int32,
        my_ws: cutlass.Int64,
        rank: cutlass.Int32,
        max_tokens: cutlass.Int32,
        sTarget: cute.Tensor,
        sSrcCnt: cute.Tensor,
        sCnt: cute.Tensor,
        sOff: cute.Tensor,
        sUoff: cute.Tensor,
        sMisc: cute.Tensor,
        sSorted: cute.Tensor,
        mCounts: cute.Tensor,
        mOffsets: cute.Tensor,
        mP2e: cute.Tensor,
        mE2p: cute.Tensor,
        mT2e: cute.Tensor,
        mT2l: cute.Tensor,
        mNumTiles: cute.Tensor,
        mTotal: cute.Tensor,
        mPlanPub: cute.Tensor,
    ):
        """Epilogue warps: this CTA's share of the RoutingPlan (the contents ``routing.run`` writes). CTA 0 writes
        the counts, offsets and the two totals; epilogue warp w of CTA c publishes experts c + w * grid, c + (w + 4) *
        grid, ...: their permuted_idx_to_expanded_idx slice (padding -1), expanded_idx_to_permuted_idx of their pairs
        and their tile maps; CTA c writes -1 into expanded_idx_to_permuted_idx for the non-local pairs of its block of
        rows. Then one thread adds 1 to ``plan_pub``; the add that completes the grid also advances the per-source
        byte-counter baselines (every CTA read them before its add)."""
        E = self.num_local
        TN = self.n_tile
        warp = tidx // 32
        lane = tidx & 31
        total = sOff[self.em]
        if cutlass.const_expr(self.plan_pub and self.delay_ns > 0):
            if bidx < E:
                spin_ns(cutlass.Int64(self.delay_ns))
        if bidx == 0:
            for e in cutlass.range(tidx, E, self.num_epilog_threads, unroll=1):
                mCounts[e] = sCnt[e]
                mOffsets[e] = sOff[e]
            if tidx == 0:
                mOffsets[E] = total
                mTotal[0] = total
                mNumTiles[0] = total // TN
        from_list = sMisc[0] < 2
        list_base = sMisc[1]
        for e in cutlass.range(bidx + warp * num_ctas, E, 4 * num_ctas, unroll=1):
            cnt = sCnt[e]
            off = sOff[e]
            end = sOff[e + 1]
            if from_list:
                for j in cutlass.range(lane, cnt, 32, unroll=1):
                    expanded = cutlass.Int32(sSorted[list_base + sUoff[e] + j])
                    mP2e[off + j] = expanded
                    mE2p[expanded] = off + j
            else:
                self._walk_ids(
                    e,
                    cutlass.Int32(0),
                    False,
                    lane,
                    ids_addr,
                    rows,
                    lo,
                    max_tokens,
                    sSrcCnt,
                    sOff,
                    mP2e,
                    mE2p,
                    off,
                )
            for j in cutlass.range(off + cnt + lane, end, 32, unroll=1):
                mP2e[j] = cutlass.Int32(-1)
            for t in cutlass.range(off // TN + lane, end // TN, 32, unroll=1):
                mT2e[t] = e
                mT2l[t] = cutlass.min((t + 1) * TN, off + cnt)
        rows_per_cta = (rows + num_ctas - 1) // num_ctas
        r_lo = cutlass.min(rows, bidx * rows_per_cta)
        r_hi = cutlass.min(rows, r_lo + rows_per_cta)
        for i in cutlass.range(
            r_lo * TOP_K + tidx, r_hi * TOP_K, self.num_epilog_threads, unroll=1
        ):
            v = ld_cg_b32(ids_addr + cutlass.Int64(i) * 4) - lo
            valid = cutlass.select_(
                self._row_valid(i // TOP_K, rows, max_tokens, sSrcCnt),
                cutlass.Int32(1),
                cutlass.Int32(0),
            )
            if (v < 0) | (v >= E) | (valid == 0):
                mE2p[i] = cutlass.Int32(-1)
        self.epilog_sync_barrier.arrive_and_wait()
        if cutlass.const_expr(self.plan_pub):
            if tidx == 0:
                cute.arch.fence_acq_rel_gpu()
                old = atom_add_acq_rel(mPlanPub.iterator.toint(), cutlass.Int32(1))
                if cutlass.const_expr(self.source is not None and not self.source[0]):
                    ep, off_base = self.source[1], self.source[6]
                    if old == num_ctas - 1:
                        for s in cutlass.range_constexpr(ep):
                            if s != rank:
                                st_b64(my_ws + off_base + s * 8, sTarget[s])

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        mX: cute.Tensor,
        mXsf: cute.Tensor,
        mAct: cute.Tensor,
        mActSf: cute.Tensor,
        mP2e: cute.Tensor,
        mT2e: cute.Tensor,
        mT2l: cute.Tensor,
        mNumTiles: cute.Tensor,
        mPlanReady: cute.Tensor,
        mFc1Done: cute.Tensor,
        mIds: cute.Tensor,
        mCounts: cute.Tensor,
        mOffsets: cute.Tensor,
        mE2p: cute.Tensor,
        mTotal: cute.Tensor,
        mPlanPub: cute.Tensor,
        num_ctas: cutlass.Int32,
        lo: cutlass.Int32,
        my_ws: cutlass.Int64,
        max_tokens: cutlass.Int32,
        rank: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        n_tile = self.n_tile
        n_mma = self.n_mma

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        if cutlass.const_expr(self.routed):
            rstore = smem.allocate(self.routing_storage, byte_alignment=128)

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        b_pipeline = _CpAsyncUmmaPipeline.create(
            self.num_ab_stage,
            self.threads_per_warp * self.num_gather_warps,
            storage.b_mbar.data_ptr(),
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_epilog_threads
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.info_mbar.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_info_consumers
            ),
        )
        # Epilogue -> signal warp: an item's act / act_sf stores are done (mbarrier arrive / wait order them at CTA
        # scope; the signal warp's release then publishes them at GPU scope).
        if cutlass.const_expr(self.signal_done):
            stored_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.stored_mbar.data_ptr(),
                num_stages=self.num_stored_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_epilog_threads
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.threads_per_warp
                ),
            )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
        )
        tmem.allocate(self.num_tmem_alloc_cols)

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        nf = self.info_fields
        sInfo = storage.info.get_tensor(cute.make_layout((nf, self.num_tile_stage), stride=(1, nf)))
        sRows = storage.rows.get_tensor(
            cute.make_layout((n_mma, self.num_tile_stage), stride=(1, n_mma))
        )
        sAmax = storage.amax.get_tensor(
            cute.make_layout((n_mma, 4, 2), stride=(1, n_mma, 4 * n_mma))
        )

        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, a_cta_layout, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        if cutlass.const_expr(self.routed):
            sTarget = rstore.target.get_tensor(cute.make_layout((64,)))
            sSrcCnt = rstore.src_cnt.get_tensor(cute.make_layout((64,)))
            sWCnt = rstore.wcnt.get_tensor(
                cute.make_layout((self.em, self.num_scan_warps), stride=(1, self.em))
            )
            sSegLen = rstore.seg_len.get_tensor(cute.make_layout((self.num_scan_warps,)))
            sCnt = rstore.cnt.get_tensor(cute.make_layout((self.em,)))
            sOff = rstore.off.get_tensor(cute.make_layout((self.em + 16,)))
            sUoff = rstore.uoff.get_tensor(cute.make_layout((self.em + 16,)))
            sPairRows = rstore.pair_rows.get_tensor(cute.make_layout((64,)))
            sMisc = rstore.misc.get_tensor(cute.make_layout((16,)))
            sPairs = rstore.lists.get_tensor(cute.make_layout((self.num_scan_warps * self.seg,)))
            sSorted = cute.make_tensor(
                cute.recast_ptr(sPairs.iterator, dtype=cutlass.Uint16),
                cute.make_layout((2 * self.num_scan_warps * self.seg + self.sorted_cap,)),
            )
            ids_addr = mIds.iterator.toint()
            # Without prefetch every warp starts after the predecessor; with it the weight TMA warp issues its
            # speculative stages at launch and only the other warps wait.
            if cutlass.const_expr(self.use_pdl and not self.prefetch):
                cute.arch.griddepcontrol_wait()
            self.cta_sync_barrier.arrive_and_wait()
            if warp_idx != self.tma_warp_id:
                if cutlass.const_expr(self.use_pdl and self.prefetch):
                    cute.arch.griddepcontrol_wait()
                scan_warp = cutlass.select_(
                    warp_idx < self.tma_warp_id, warp_idx, warp_idx - cutlass.Int32(1)
                )
                self._plan_in_cta(
                    scan_warp,
                    tidx & 31,
                    ids_addr,
                    mX.shape[0],
                    lo,
                    my_ws,
                    max_tokens,
                    rank,
                    sTarget,
                    sSrcCnt,
                    sWCnt,
                    sSegLen,
                    sCnt,
                    sOff,
                    sUoff,
                    sMisc,
                    sPairs,
                    sSorted,
                )
        else:
            # Every read of a predecessor's output and every global write comes after this wait; with wait_plan the
            # predecessor (routing) publishes the plan through plan_ready instead, and only the scheduler reads it.
            if cutlass.const_expr(self.use_pdl and (not self.wait_plan or self.full_wait)):
                cute.arch.griddepcontrol_wait()
            self.cta_sync_barrier.arrive_and_wait()

        # Scheduler warp. The tiles of one expert pair up from its first tile: tile t leads an item when it is an even
        # number of tiles past the expert's first tile, and the item also covers tile t + 1 when that tile belongs to
        # the same expert. Leading tiles are numbered in tile order; the r-th one owns work items r * 48 .. r * 48 + 47
        # (weight tile fastest) and CTA c takes the items equal to c modulo the grid, so every CTA gets the same
        # number of items whatever the routing. The ring carries (weight tile, token tile, expert, row limit, valid)
        # and the source row of each of the n_mma token rows (-1 at or past the row limit). The row limit is the
        # tile_idx_to_mn_limit of the item's last tile, clipped to that tile's end.
        # Routed: the same items, built from this CTA's own counts. Lead pairs are numbered in expert order; the
        # rows of pair q of expert e are its pairs of rank q * n_mma .. q * n_mma + n_mma - 1 (ascending expanded
        # index). Field 5 of the first record says whether the first item is the speculative unit the weight TMA
        # warp prefetched (weight tile bidx % 48 of expert bidx // 48).
        if cutlass.const_expr(self.routed):
            if warp_idx == self.sched_warp_id:
                if cutlass.const_expr(self.use_pdl and (self.signal_done or self.lamport)):
                    cute.arch.griddepcontrol_launch_dependents()
                info_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_tile_stage
                )
                lane = tidx & 31
                n_lane_rows = (n_mma + 31) // 32
                rows_reg = cute.make_rmem_tensor((n_lane_rows,), cutlass.Int32)
                spec_e = bidx // NUM_M_TILES
                spec_m = bidx - spec_e * NUM_M_TILES
                first = cutlass.Int32(1)
                lead = cutlass.Int32(0)
                from_list = sMisc[0] < 2
                list_base = sMisc[1]
                for e in cutlass.range(self.num_local, unroll=1):
                    cnt = sCnt[e]
                    n_leads = ((cnt + n_tile - 1) // n_tile + 1) // 2
                    for q in cutlass.range(n_leads, unroll=1):
                        first_item = lead * NUM_M_TILES
                        m_tile = bidx - (first_item - (first_item // num_ctas) * num_ctas)
                        m_tile = m_tile + cutlass.select_(m_tile < 0, num_ctas, cutlass.Int32(0))
                        if m_tile < NUM_M_TILES:
                            if from_list:
                                for i in cutlass.range_constexpr(n_lane_rows):
                                    r = q * n_mma + i * 32 + lane
                                    rows_reg[i] = cutlass.Int32(-1)
                                    if r < cnt:
                                        rows_reg[i] = (
                                            cutlass.Int32(sSorted[list_base + sUoff[e] + r])
                                            // TOP_K
                                        )
                            else:
                                for i in cutlass.range_constexpr(n_lane_rows):
                                    if i * 32 + lane < n_mma:
                                        sPairRows[i * 32 + lane] = cutlass.Int32(-1)
                                cute.arch.sync_warp()
                                self._walk_ids(
                                    e,
                                    q * n_mma,
                                    True,
                                    lane,
                                    ids_addr,
                                    mX.shape[0],
                                    lo,
                                    max_tokens,
                                    sSrcCnt,
                                    sPairRows,
                                    mP2e,
                                    mE2p,
                                    cutlass.Int32(0),
                                )
                                cute.arch.sync_warp()
                                for i in cutlass.range_constexpr(n_lane_rows):
                                    rows_reg[i] = sPairRows[cutlass.min(i * 32 + lane, n_mma - 1)]
                        item_tile = sOff[e] // n_tile + 2 * q
                        row_limit = cutlass.min(sOff[e] + cnt, item_tile * n_tile + n_mma)
                        while m_tile < NUM_M_TILES:
                            info_pipeline.producer_acquire(info_state)
                            spec_ok = first & cutlass.select_(
                                (m_tile == spec_m) & (e == spec_e),
                                cutlass.Int32(1),
                                cutlass.Int32(0),
                            )
                            sInfo[(0, info_state.index)] = m_tile
                            sInfo[(1, info_state.index)] = item_tile
                            sInfo[(2, info_state.index)] = e
                            sInfo[(3, info_state.index)] = row_limit
                            sInfo[(4, info_state.index)] = cutlass.Int32(1)
                            sInfo[(5, info_state.index)] = spec_ok
                            for i in cutlass.range_constexpr(n_lane_rows):
                                if i * 32 + lane < n_mma:
                                    sRows[(i * 32 + lane, info_state.index)] = rows_reg[i]
                            info_pipeline.producer_commit(info_state)
                            info_state.advance()
                            first = cutlass.Int32(0)
                            m_tile = m_tile + num_ctas
                        lead = lead + 1
                info_pipeline.producer_acquire(info_state)
                sInfo[(4, info_state.index)] = cutlass.Int32(0)
                sInfo[(5, info_state.index)] = cutlass.Int32(0)
                info_pipeline.producer_commit(info_state)
                info_state.advance()
                info_pipeline.producer_tail(info_state)
        else:
            if warp_idx == self.sched_warp_id:
                if cutlass.const_expr(self.wait_plan):
                    wait_at_least(mPlanReady.iterator.toint(), cutlass.Int32(1))
                # The dependent (FC2) may start once the plan it also reads is complete.
                if cutlass.const_expr(
                    self.use_pdl and (self.wait_plan or self.signal_done or self.lamport)
                ):
                    cute.arch.griddepcontrol_launch_dependents()
                info_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_tile_stage
                )
                lane = tidx & 31
                lanes_below = cutlass.Int32(cute.arch.lanemask_lt())
                lanes_upto = cutlass.Int32(cute.arch.lanemask_le())
                num_rows = mX.shape[0]
                last_prow = mP2e.shape[0] - 1
                n_lane_rows = (n_mma + 31) // 32
                expanded = cute.make_rmem_tensor((n_lane_rows,), cutlass.Int32)
                num_tiles = mNumTiles[0]
                num_leads = cutlass.Int32(0)
                prev_expert = cutlass.Int32(-1)
                prev_run_start = cutlass.Int32(0)
                for step in cutlass.range((num_tiles + 31) // 32, unroll=1):
                    tile = step * 32 + lane
                    in_range = tile < num_tiles
                    expert = mT2e[cutlass.min(tile, num_tiles - 1)]
                    next_expert = mT2e[cutlass.min(tile + 1, num_tiles - 1)]
                    limit = mT2l[cutlass.min(tile, num_tiles - 1)]
                    next_limit = mT2l[cutlass.min(tile + 1, num_tiles - 1)]
                    expert_before = cutlass.select_(
                        lane == 0, prev_expert, cute.arch.shuffle_sync_up(expert, 1)
                    )
                    starts = (
                        cute.arch.vote_ballot_sync(in_range & (expert_before != expert))
                        & lanes_upto
                    )
                    run_start = cutlass.select_(
                        starts != 0,
                        step * 32 + 31 - cutlass.Int32(cute.arch.clz(starts)),
                        prev_run_start,
                    )
                    leads = in_range & (((tile - run_start) & 1) == 0)
                    pairs = (tile + 1 < num_tiles) & (next_expert == expert)
                    item_limit = cutlass.select_(
                        pairs,
                        cutlass.min(next_limit, (tile + 2) * n_tile),
                        cutlass.min(limit, (tile + 1) * n_tile),
                    )
                    lead_mask = cute.arch.vote_ballot_sync(leads)
                    first_item = (num_leads + cute.arch.popc(lead_mask & lanes_below)) * NUM_M_TILES
                    m_first = bidx - (first_item - (first_item // num_ctas) * num_ctas)
                    m_first = m_first + cutlass.select_(m_first < 0, num_ctas, cutlass.Int32(0))
                    pending = cute.arch.vote_ballot_sync(leads & (m_first < NUM_M_TILES))
                    while pending != 0:
                        src = cute.arch.popc((pending & (0 - pending)) - 1)
                        pending = pending & (pending - 1)
                        item_tile = cute.arch.shuffle_sync(tile, src)
                        m_tile = cute.arch.shuffle_sync(m_first, src)
                        item_expert = cute.arch.shuffle_sync(expert, src)
                        row_base = item_tile * n_tile
                        row_limit = cute.arch.shuffle_sync(item_limit, src)
                        for r in cutlass.range_constexpr(n_lane_rows):
                            expanded[r] = mP2e[cutlass.min(row_base + r * 32 + lane, last_prow)]
                        while m_tile < NUM_M_TILES:
                            info_pipeline.producer_acquire(info_state)
                            sInfo[(0, info_state.index)] = m_tile
                            sInfo[(1, info_state.index)] = item_tile
                            sInfo[(2, info_state.index)] = item_expert
                            sInfo[(3, info_state.index)] = row_limit
                            sInfo[(4, info_state.index)] = cutlass.Int32(1)
                            for r in cutlass.range_constexpr(n_lane_rows):
                                row = r * 32 + lane
                                if row < n_mma:
                                    src_row = expanded[r] // TOP_K
                                    ok = (
                                        (row_base + row < row_limit)
                                        & (expanded[r] >= 0)
                                        & (src_row < num_rows)
                                    )
                                    sRows[(row, info_state.index)] = cutlass.select_(
                                        ok, src_row, cutlass.Int32(-1)
                                    )
                            info_pipeline.producer_commit(info_state)
                            info_state.advance()
                            m_tile = m_tile + num_ctas
                    num_leads = num_leads + cute.arch.popc(lead_mask)
                    prev_expert = cute.arch.shuffle_sync(expert, 31)
                    prev_run_start = cute.arch.shuffle_sync(run_start, 31)
                info_pipeline.producer_acquire(info_state)
                sInfo[(4, info_state.index)] = cutlass.Int32(0)
                info_pipeline.producer_commit(info_state)
                info_state.advance()
                info_pipeline.producer_tail(info_state)

        if warp_idx == self.tma_warp_id:
            ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            info_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            info = cute.make_rmem_tensor((nf,), cutlass.Int32)
            # Speculative first unit, before any wait: the MMA warp drains these stages when the first item differs.
            if cutlass.const_expr(self.prefetch):
                spec_e = bidx // NUM_M_TILES
                spec_m = bidx - spec_e * NUM_M_TILES
                spec_e = cutlass.min(spec_e, cutlass.Int32(self.num_local - 1))
                tAgA_s = tAgA[(None, spec_m, None, spec_e)]
                tAgSFA_s = tAgSFA[(None, spec_m, None, spec_e)]
                for spec_k in cutlass.range_constexpr(self.num_spec_stages):
                    ab_pipeline.producer_acquire(ab_state)
                    spec_bar = ab_pipeline.producer_get_barrier(ab_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_s[(None, spec_k)],
                        tAsA[(None, ab_state.index)],
                        tma_bar_ptr=spec_bar,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_s[(None, spec_k)],
                        tAsSFA[(None, ab_state.index)],
                        tma_bar_ptr=spec_bar,
                    )
                    ab_state.advance()
            info_pipeline.consumer_wait(info_state)
            for i in cutlass.range_constexpr(nf):
                info[i] = sInfo[(i, info_state.index)]
            info_pipeline.consumer_release(info_state)
            info_state.advance()
            k_first = cutlass.Int32(0)
            if cutlass.const_expr(self.prefetch):
                k_first = cutlass.select_(
                    info[5] == 1, cutlass.Int32(self.num_spec_stages), cutlass.Int32(0)
                )
            while info[4] == 1:
                tAgA_t = tAgA[(None, info[0], None, info[2])]
                tAgSFA_t = tAgSFA[(None, info[0], None, info[2])]
                for k_tile in cutlass.range(k_first, self.num_k_tiles, unroll=1):
                    ab_pipeline.producer_acquire(ab_state)
                    tma_bar = ab_pipeline.producer_get_barrier(ab_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_t[(None, k_tile)],
                        tAsA[(None, ab_state.index)],
                        tma_bar_ptr=tma_bar,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_t[(None, k_tile)],
                        tAsSFA[(None, ab_state.index)],
                        tma_bar_ptr=tma_bar,
                    )
                    ab_state.advance()
                if cutlass.const_expr(self.prefetch):
                    k_first = cutlass.Int32(0)
                info_pipeline.consumer_wait(info_state)
                for i in cutlass.range_constexpr(nf):
                    info[i] = sInfo[(i, info_state.index)]
                info_pipeline.consumer_release(info_state)
                info_state.advance()
            ab_pipeline.producer_tail(ab_state)

        if warp_idx >= self.gather_warp_id:
            gather_sub = warp_idx - self.gather_warp_id
            b_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            info_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            info = cute.make_rmem_tensor((nf,), cutlass.Int32)
            lane = tidx & 31
            # Logical smem byte of (row, k) in a stage of the SW128 K-major operand: kt * n_mma * 128 + row * 128 +
            # (k % 128), kt = k // 128; the swizzled sB iterator applies the 128 B XOR. SF atom byte of (row, k block):
            # kt * 512 + (row % 32) * 16 + (row // 32) * 4 + k block % 4.
            chunk = lane & 7
            row_in_pass = lane // 8
            # Passes of 4 rows are dealt round-robin over the gather warps; warp 0 also moves the scale bytes.
            n_pass = n_mma // 4 // self.num_gather_warps
            n_sf = (n_mma + 31) // 32
            n_kt = self.k_stage // 128
            b_bytes_per_stage = n_mma * self.k_stage
            sf_bytes_per_stage = 512 * n_kt
            b_atom_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                cutlass.Float8E4M3FN,
                num_bits_per_copy=128,
            )
            sf_atom_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(), cutlass.Uint8, num_bits_per_copy=32
            )
            row_src = cute.make_rmem_tensor((n_pass,), cutlass.Int32)
            row_ok = cute.make_rmem_tensor((n_pass,), cutlass.Boolean)
            sf_src = cute.make_rmem_tensor((n_sf,), cutlass.Int32)
            sf_ok = cute.make_rmem_tensor((n_sf,), cutlass.Boolean)
            pred1 = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)

            info_pipeline.consumer_wait(info_state)
            for i in cutlass.range_constexpr(nf):
                info[i] = sInfo[(i, info_state.index)]
            for i in cutlass.range_constexpr(n_pass):
                src = sRows[
                    ((gather_sub + self.num_gather_warps * i) * 4 + row_in_pass, info_state.index)
                ]
                row_ok[i] = src >= 0
                row_src[i] = cutlass.max(src, cutlass.Int32(0))
            for q in cutlass.range_constexpr(n_sf):
                src = sRows[(cutlass.min(q * 32 + lane, n_mma - 1), info_state.index)]
                sf_ok[q] = (q * 32 + lane < n_mma) & (src >= 0)
                sf_src[q] = cutlass.max(src, cutlass.Int32(0))
            info_pipeline.consumer_release(info_state)
            info_state.advance()
            # cp.async with a false predicate still zero-fills its destination: padding rows become zeros.
            while info[4] == 1:
                for k_tile in cutlass.range(self.num_k_tiles, unroll=1):
                    b_pipeline.producer_acquire(b_state)
                    sB_stage = sB.iterator + b_state.index * b_bytes_per_stage
                    sSFB_stage = sSFB.iterator + b_state.index * sf_bytes_per_stage
                    k0 = k_tile * self.k_stage
                    for kt in cutlass.range_constexpr(n_kt):
                        for i in cutlass.range_constexpr(n_pass):
                            row = (gather_sub + self.num_gather_warps * i) * 4 + row_in_pass
                            dst_off = kt * n_mma * 128 + row * 128 + chunk * 16
                            src_off = cute.assume(
                                row_src[i] * HIDDEN + k0 + kt * 128 + chunk * 16, divby=16
                            )
                            g_b = cute.make_tensor(mX.iterator + src_off, cute.make_layout((16,)))
                            s_b = cute.make_tensor(sB_stage + dst_off, cute.make_layout((16,)))
                            pred1[0] = row_ok[i]
                            cute.copy_atom_call(b_atom_copy, g_b, s_b, pred=pred1)
                        if gather_sub == 0:
                            for q in cutlass.range_constexpr(n_sf):
                                sf_src_off = cute.assume(
                                    sf_src[q] * (HIDDEN // SF_VEC)
                                    + k_tile * self.k_blocks
                                    + kt * 4,
                                    divby=4,
                                )
                                g_sf = cute.make_tensor(
                                    mXsf.iterator + sf_src_off, cute.make_layout((4,))
                                )
                                s_sf = cute.make_tensor(
                                    cute.recast_ptr(
                                        sSFB_stage + kt * 512 + lane * 16 + q * 4,
                                        dtype=cutlass.Uint8,
                                    ),
                                    cute.make_layout((4,)),
                                )
                                pred1[0] = sf_ok[q]
                                cute.copy_atom_call(sf_atom_copy, g_sf, s_sf, pred=pred1)
                    b_pipeline.producer_commit(b_state)
                    b_state.advance()

                info_pipeline.consumer_wait(info_state)
                for i in cutlass.range_constexpr(nf):
                    info[i] = sInfo[(i, info_state.index)]
                for i in cutlass.range_constexpr(n_pass):
                    src = sRows[
                        (
                            (gather_sub + self.num_gather_warps * i) * 4 + row_in_pass,
                            info_state.index,
                        )
                    ]
                    row_ok[i] = src >= 0
                    row_src[i] = cutlass.max(src, cutlass.Int32(0))
                for q in cutlass.range_constexpr(n_sf):
                    src = sRows[(cutlass.min(q * 32 + lane, n_mma - 1), info_state.index)]
                    sf_ok[q] = (q * 32 + lane < n_mma) & (src >= 0)
                    sf_src[q] = cutlass.max(src, cutlass.Int32(0))
                info_pipeline.consumer_release(info_state)
                info_state.advance()
            b_pipeline.producer_tail(b_state)

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_acc_tmem_cols, dtype=cutlass.Float8E8M0FNU
            )
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_acc_tmem_cols + self.num_sfa_tmem_cols,
                dtype=cutlass.Float8E8M0FNU,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            # The MMA reads n_mma SF rows; the S2T copy fills the whole 128-row atom at the same base.
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB_full_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_sfb,
                self.mma_tiler_sfb,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            tCtSFB_full = cute.make_tensor(sfb_tmem_ptr, tCtSFB_full_layout)
            copy_s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = self._s2t_copy_and_partition(sSFA, tCtSFA)
            copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = self._s2t_copy_and_partition(sSFB, tCtSFB_full)

            ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            b_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            info_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            info = cute.make_rmem_tensor((nf,), cutlass.Int32)
            info_pipeline.consumer_wait(info_state)
            for i in cutlass.range_constexpr(nf):
                info[i] = sInfo[(i, info_state.index)]
            info_pipeline.consumer_release(info_state)
            info_state.advance()
            if cutlass.const_expr(self.prefetch):
                n_drain = cutlass.select_(
                    info[5] == 1, cutlass.Int32(0), cutlass.Int32(self.num_spec_stages)
                )
                for _ in cutlass.range(n_drain, unroll=1):
                    ab_pipeline.consumer_wait(ab_state)
                    ab_pipeline.consumer_release(ab_state)
                    ab_state.advance()
            num_kblocks = cute.size(tCrA, mode=[2])
            while info[4] == 1:
                acc_pipeline.producer_acquire(acc_state)
                _tcgen05_fence("after")
                tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile in cutlass.range(self.num_k_tiles, unroll=1):
                    ab_pipeline.consumer_wait(ab_state)
                    b_pipeline.consumer_wait(b_state)
                    # cp.async (generic proxy) writes -> tcgen05 (async proxy) reads.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.copy(
                        copy_s2t_sfa,
                        tCsSFA_s2t[(None, None, None, None, ab_state.index)],
                        tCtSFA_s2t,
                    )
                    cute.copy(
                        copy_s2t_sfb,
                        tCsSFB_s2t[(None, None, None, None, b_state.index)],
                        tCtSFB_s2t,
                    )
                    for kblock in cutlass.range_constexpr(num_kblocks):
                        sf_coord = (None, None, kblock)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_coord].iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[(None, None, kblock, ab_state.index)],
                            tCrB[(None, None, kblock, b_state.index)],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_pipeline.consumer_release(ab_state)
                    ab_state.advance()
                    b_pipeline.consumer_release(b_state)
                    b_state.advance()
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
                info_pipeline.consumer_wait(info_state)
                for i in cutlass.range_constexpr(nf):
                    info[i] = sInfo[(i, info_state.index)]
                info_pipeline.consumer_release(info_state)
                info_state.advance()
            acc_pipeline.producer_tail(acc_state)

        # Epilogue warp w reads TMEM lanes 32 * w .. + 31 in 8-column chunks with two 16x256b loads (lane offsets 0
        # and 16), so each thread holds the up and gate rows of two adjacent channels at two adjacent columns.
        if warp_idx <= self.epilog_warp_id[-1]:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            if cutlass.const_expr(self.routed):
                self._publish_plan(
                    tidx,
                    bidx,
                    num_ctas,
                    ids_addr,
                    mX.shape[0],
                    lo,
                    my_ws,
                    rank,
                    max_tokens,
                    sTarget,
                    sSrcCnt,
                    sCnt,
                    sOff,
                    sUoff,
                    sMisc,
                    sSorted,
                    mCounts,
                    mOffsets,
                    mP2e,
                    mE2p,
                    mT2e,
                    mT2l,
                    mNumTiles,
                    mTotal,
                    mPlanPub,
                )
            n_chunk = n_mma // 8
            mAct16 = cute.make_tensor(
                cute.recast_ptr(mAct.iterator, dtype=cutlass.Uint16),
                cute.make_layout((mAct.shape[0], INTER // 2), stride=(INTER // 2, 1)),
            )

            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            info_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            stored_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stored_stage
            )
            info = cute.make_rmem_tensor((nf,), cutlass.Int32)
            acc = cute.make_rmem_tensor((8 * n_chunk,), cutlass.Float32)
            act = cute.make_rmem_tensor((4 * n_chunk,), cutlass.Float32)
            amax = cute.make_rmem_tensor((2 * n_chunk,), cutlass.Float32)
            info_pipeline.consumer_wait(info_state)
            for i in cutlass.range_constexpr(nf):
                info[i] = sInfo[(i, info_state.index)]
            info_pipeline.consumer_release(info_state)
            info_state.advance()

            quad = tidx // self.threads_per_warp
            lane = tidx & 31
            col_in_chunk = 2 * (lane & 3)
            # Up lane 16 * b + lane // 4 (b = 0, 1) holds channel 16 * quad + 2 * (lane // 4) + b; the gate row of
            # the same channel is 8 lanes up.
            act16_col = 8 * quad + (lane >> 2)
            sf_group = quad >> 1
            partner = quad ^ 1
            tmem_quad = tmem_ptr.toint() + ((32 * quad) << 16)
            gate_cap = cutlass.Float32(self.gate_cap)
            inv_gate_cap = cutlass.Float32(1.0 / self.gate_cap)
            linear_cap = cutlass.Float32(self.linear_cap)
            inv_linear_cap = cutlass.Float32(1.0 / self.linear_cap)
            parity = cutlass.Int32(0)

            if cutlass.const_expr(self.delay_ns > 0):
                # CTA c starts with item c: the first NUM_M_TILES CTAs hold token tile 0.
                if bidx < NUM_M_TILES:
                    spin_ns(cutlass.Int64(self.delay_ns))
            while info[4] == 1:
                m_tile = info[0]
                row_base = info[1] * n_tile
                limit = info[3]
                num_valid = limit - row_base

                acc_pipeline.consumer_wait(acc_state)
                _tcgen05_fence("after")
                tmem_stage = tmem_quad + acc_state.index * n_mma
                addrs = []
                for i in cutlass.range_constexpr(n_chunk):
                    for b in cutlass.range_constexpr(2):
                        addrs.append(tmem_stage + ((16 * b) << 16) + 8 * i)
                loaded = _tmem_load_16x256b(addrs)
                for k in cutlass.range_constexpr(8 * n_chunk):
                    acc[k] = loaded[k]
                _tcgen05_fence("before")
                acc_pipeline.consumer_release(acc_state)
                acc_state.advance()

                # act = (gate_cap * tanh(g / gate_cap) * sigmoid(g)) * (linear_cap * tanh(u / linear_cap)); chunks at
                # or past the last valid row are skipped (warp-uniform). The 32-channel amax of a column spans the 8
                # threads with the same lane % 4 and the partner warp (quad ^ 1).
                for i in cutlass.range_constexpr(n_chunk):
                    chunk_col = 8 * i
                    if chunk_col < num_valid:
                        for j in cutlass.range_constexpr(2):
                            for b in cutlass.range_constexpr(2):
                                g = acc[8 * i + 4 * b + 2 + j]
                                u = acc[8 * i + 4 * b + j]
                                gate = (gate_cap * _tanh(g * inv_gate_cap)) * _sigmoid(g)
                                act[4 * i + 2 * j + b] = gate * (
                                    linear_cap * _tanh(u * inv_linear_cap)
                                )
                        for j in cutlass.range_constexpr(2):
                            v0 = act[4 * i + 2 * j]
                            v1 = act[4 * i + 2 * j + 1]
                            amax[2 * i + j] = cute.arch.fmax(
                                cute.arch.fmax(v0, -v0), cute.arch.fmax(v1, -v1)
                            )
                        for sh in cutlass.range_constexpr(2, 5):
                            for j in cutlass.range_constexpr(2):
                                amax[2 * i + j] = cute.arch.fmax(
                                    amax[2 * i + j],
                                    cute.arch.shuffle_sync_bfly(amax[2 * i + j], 1 << sh),
                                )
                        for j in cutlass.range_constexpr(2):
                            sAmax[(chunk_col + col_in_chunk + j, quad, parity)] = amax[2 * i + j]
                self.epilog_sync_barrier.arrive_and_wait()

                for i in cutlass.range_constexpr(n_chunk):
                    chunk_col = 8 * i
                    if chunk_col < num_valid:
                        for j in cutlass.range_constexpr(2):
                            col = chunk_col + col_in_chunk + j
                            prow = row_base + col
                            g_amax = cute.arch.fmax(amax[2 * i + j], sAmax[(col, partner, parity)])
                            # e = ceil(log2(g_amax / 448)), 448 = 1.75 * 2^8: one more than the binary exponent
                            # minus 8 when the mantissa exceeds 1.75.
                            bits = _f32_bits(g_amax) & cutlass.Int32(0x7FFFFFFF)
                            e = (bits >> 23) - 135
                            e = e + cutlass.select_(
                                (bits & cutlass.Int32(0x7FFFFF)) > cutlass.Int32(0x600000),
                                cutlass.Int32(1),
                                cutlass.Int32(0),
                            )
                            e = cutlass.max(cutlass.min(e, cutlass.Int32(126)), cutlass.Int32(-127))
                            sf_byte = cutlass.select_(bits == 0, cutlass.Int32(0), e + 127)
                            inv_scale = cutlass.select_(
                                bits == 0, cutlass.Float32(1.0), _bits_f32((127 - e) << 23)
                            )
                            codes = _e4m3x2(
                                act[4 * i + 2 * j + 1] * inv_scale, act[4 * i + 2 * j] * inv_scale
                            )
                            if cutlass.const_expr(self.lamport):
                                # E4M3 -0 is the slot sentinel: store +0 (the same value).
                                zero_signs = cutlass.select_(
                                    (codes & 0x7F) == 0, cutlass.Int32(0x80), cutlass.Int32(0)
                                ) | cutlass.select_(
                                    (codes & 0x7F00) == 0, cutlass.Int32(0x8000), cutlass.Int32(0)
                                )
                                codes = codes & (zero_signs ^ cutlass.Int32(-1))
                            if prow < limit:
                                mAct16[(prow, m_tile * 32 + act16_col)] = cutlass.Uint16(codes)
                                if (quad & 1) == 0:
                                    mActSf[(prow, m_tile * 2 + sf_group)] = cutlass.Uint8(sf_byte)
                parity = parity ^ 1
                if cutlass.const_expr(self.signal_done):
                    stored_pipeline.producer_acquire(stored_state)
                    stored_pipeline.producer_commit(stored_state)
                    stored_state.advance()

                info_pipeline.consumer_wait(info_state)
                for i in cutlass.range_constexpr(nf):
                    info[i] = sInfo[(i, info_state.index)]
                info_pipeline.consumer_release(info_state)
                info_state.advance()

            if cutlass.const_expr(self.signal_done):
                stored_pipeline.producer_tail(stored_state)
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # Signal warp: after all stores of (token tile t, weight tile j), fc1_done[t * 12 + j // 4] += 1 with release
        # semantics, for both tiles of a pair and for t < EARLY_TILES only. The GPU-scope fence slows the whole SM,
        # so a pair shares one.
        if cutlass.const_expr(self.signal_done):
            if warp_idx == self.signal_warp_id:
                stored_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_stored_stage
                )
                info_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )
                info = cute.make_rmem_tensor((nf,), cutlass.Int32)
                info_pipeline.consumer_wait(info_state)
                for i in cutlass.range_constexpr(nf):
                    info[i] = sInfo[(i, info_state.index)]
                info_pipeline.consumer_release(info_state)
                info_state.advance()
                while info[4] == 1:
                    stored_pipeline.consumer_wait(stored_state)
                    if (tidx % self.threads_per_warp == 0) & (info[1] < EARLY_TILES):
                        done = mFc1Done.iterator + info[1] * DONE_PER_TILE + info[0] // 4
                        cute.arch.fence_acq_rel_gpu()
                        _red_add_gpu(done.toint(), cutlass.Int32(1))
                        if info[3] > (info[1] + 1) * n_tile:
                            _red_add_gpu((done + DONE_PER_TILE).toint(), cutlass.Int32(1))
                    stored_pipeline.consumer_release(stored_state)
                    stored_state.advance()
                    info_pipeline.consumer_wait(info_state)
                    for i in cutlass.range_constexpr(nf):
                        info[i] = sInfo[(i, info_state.index)]
                    info_pipeline.consumer_release(info_state)
                    info_state.advance()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _gmem_ptr(dtype, tensor: torch.Tensor, align: int):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def num_ctas_for(x: torch.Tensor, plan) -> int:
    """The persistent FC1 grid for this plan (a ``plan_pub`` consumer waits for this many adds)."""
    return min(_num_sms(x.device.index), plan.tile_idx_to_expert_idx.shape[0] * NUM_M_TILES)


def _launch_args(
    x,
    x_sf,
    weights,
    plan,
    act,
    act_sf,
    plan_ready,
    fc1_done,
    topk_ids=None,
    plan_pub=None,
    recv=(0, 1, 0),
):
    num_experts = weights.w3_w1_weight.shape[0]
    num_tiles_cap = plan.tile_idx_to_expert_idx.shape[0]
    num_ctas = num_ctas_for(x, plan)
    dummy = plan.num_non_exiting_tiles
    ids = dummy if topk_ids is None else topk_ids
    return (
        _gmem_ptr(cutlass.Float4E2M1FN, weights.w3_w1_weight, 128),
        _gmem_ptr(cutlass.Float8E8M0FNU, weights.w3_w1_weight_scale, 128),
        _gmem_ptr(cutlass.Float8E4M3FN, x, 16),
        _gmem_ptr(cutlass.Uint8, x_sf, 4),
        _gmem_ptr(cutlass.Uint8, act, 16),
        _gmem_ptr(cutlass.Uint8, act_sf, 16),
        _gmem_ptr(cutlass.Int32, plan.permuted_idx_to_expanded_idx, 4),
        _gmem_ptr(cutlass.Int32, plan.tile_idx_to_expert_idx, 4),
        _gmem_ptr(cutlass.Int32, plan.tile_idx_to_mn_limit, 4),
        _gmem_ptr(cutlass.Int32, plan.num_non_exiting_tiles, 4),
        _gmem_ptr(
            cutlass.Int32, plan.num_non_exiting_tiles if plan_ready is None else plan_ready, 4
        ),
        _gmem_ptr(cutlass.Int32, plan.num_non_exiting_tiles if fc1_done is None else fc1_done, 4),
        _gmem_ptr(cutlass.Int32, ids, 16),
        _gmem_ptr(cutlass.Int32, plan.expert_counts, 4),
        _gmem_ptr(cutlass.Int32, plan.expert_offsets, 4),
        _gmem_ptr(cutlass.Int32, plan.expanded_idx_to_permuted_idx, 4),
        _gmem_ptr(cutlass.Int32, plan.total_num_padded_tokens, 4),
        _gmem_ptr(cutlass.Int32, dummy if plan_pub is None else plan_pub, 4),
        cutlass.Int32(num_experts),
        cutlass.Int32(x.shape[0]),
        cutlass.Int32(act.shape[0]),
        cutlass.Int32(num_tiles_cap),
        cutlass.Int32(num_ctas),
        cutlass.Int32(weights.local_expert_offset),
        cutlass.Int64(recv[0]),
        cutlass.Int32(recv[1]),
        cutlass.Int32(recv[2]),
    )


_compiled = {}


def _check(x, x_sf, weights, plan, act, act_sf):
    rows = x.shape[0]
    p = plan.permuted_idx_to_expanded_idx.shape[0]
    if x.dtype != torch.float8_e4m3fn or tuple(x.shape) != (rows, HIDDEN) or not x.is_contiguous():
        raise ValueError(
            f"x must be contiguous float8_e4m3fn [R, {HIDDEN}], got {x.dtype} {tuple(x.shape)}"
        )
    if (
        x_sf.dtype != torch.uint8
        or tuple(x_sf.shape) != (rows, HIDDEN // SF_VEC)
        or not x_sf.is_contiguous()
    ):
        raise ValueError(
            f"x_sf must be contiguous uint8 [R, {HIDDEN // SF_VEC}], got {tuple(x_sf.shape)}"
        )
    if x.data_ptr() % 16 or x_sf.data_ptr() % 4:
        raise ValueError("x must be 16-byte and x_sf 4-byte aligned")
    w, w_sf = weights.w3_w1_weight, weights.w3_w1_weight_scale
    if (
        w.dtype != torch.uint8
        or tuple(w.shape[1:]) != (W_ROWS, HIDDEN // 2)
        or not w.is_contiguous()
    ):
        raise ValueError(f"w3_w1_weight must be contiguous uint8 [E, {W_ROWS}, {HIDDEN // 2}]")
    if tuple(w_sf.shape) != (w.shape[0], W_ROWS, HIDDEN // SF_VEC) or not w_sf.is_contiguous():
        raise ValueError(f"w3_w1_weight_scale must be contiguous [E, {W_ROWS}, {HIDDEN // SF_VEC}]")
    if (
        act.dtype != torch.float8_e4m3fn
        or tuple(act.shape) != (p, INTER)
        or not act.is_contiguous()
    ):
        raise ValueError(f"act must be contiguous float8_e4m3fn [{p}, {INTER}]")
    if (
        act_sf.dtype != torch.uint8
        or tuple(act_sf.shape) != (p, INTER // SF_VEC)
        or not act_sf.is_contiguous()
    ):
        raise ValueError(f"act_sf must be contiguous uint8 [{p}, {INTER // SF_VEC}]")
    if plan.tile_idx_to_expert_idx.shape[0] * plan.tile_n > p:
        raise ValueError("plan tiles exceed the permuted rows")


def _check_word(name, t, words):
    if t is not None and (t.dtype != torch.int32 or not t.is_contiguous() or t.numel() < words):
        raise ValueError(f"{name} must be a contiguous int32 tensor of at least {words} words")


def run(
    x,
    x_sf,
    weights,
    plan,
    act,
    act_sf,
    *,
    plan_ready=None,
    fc1_done=None,
    lamport=False,
    hazard_delay_ns=0,
):
    """FC1 of the moe_chain contract: act[p], act_sf[p] = MXFP8(SiTU-GLU(x[p2e[p] // 16] @ W31[e]^T)) for every
    valid permuted row p of expert e; rows past each tile's mn_limit are left untouched. Launches one kernel on the
    current stream; no allocation, no host synchronization.

    ``plan_ready`` (int32 word, zero-armed, raised to >= 1 by routing after its plan stores) makes FC1 the routing
    consumer: the plan is read only after the word is acquired. ``fc1_done`` (int32 [T * 12], zero-armed) makes it
    the FC1 -> FC2 producer: after the stores of weight tile j of token tile t < ``EARLY_TILES``,
    fc1_done[t * 12 + j // 4] is incremented with release semantics, so each 256-channel K stage of such a tile
    reaches 4 (later tiles are covered by the grid dependency). Both are owned by the caller. ``lamport`` makes it
    the Lamport FC1 -> FC2 producer instead: ``act`` / ``act_sf`` are slots armed with 0x80 / 0xFF, FC1 never writes
    those bytes (E4M3 -0 is stored as +0, a scale byte is at most 253) and releases FC2 as the ``fc1_done`` build
    does, without counters. ``hazard_delay_ns`` delays the stores of token tile 0 in the ``fc1_done`` and ``lamport``
    builds (hazard tests only)."""
    _check(x, x_sf, weights, plan, act, act_sf)
    _check_word("plan_ready", plan_ready, 1)
    _check_word("fc1_done", fc1_done, plan.tile_idx_to_expert_idx.shape[0] * DONE_PER_TILE)
    key = (
        plan.tile_n,
        weights.gate_cap,
        weights.linear_cap,
        _use_pdl(),
        plan_ready is not None,
        fc1_done is not None,
        hazard_delay_ns if (fc1_done is not None or lamport) else 0,
        plan_ready is not None and fgs_diag("route_fc1") == "ready",
        lamport,
    )
    args = _launch_args(x, x_sf, weights, plan, act, act_sf, plan_ready, fc1_done)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled = _compiled.get(key)
    if compiled is None:
        kernel = Fc1Kernel(*key[:6], delay_ns=key[6], full_wait=key[7], lamport=key[8])
        compiled = _compiled[key] = cute.compile(kernel, *args, stream)
    compiled(*args, stream)


def run_routed(
    x,
    x_sf,
    topk_ids,
    weights,
    plan,
    act,
    act_sf,
    *,
    source=None,
    prefetch=False,
    plan_pub=None,
    fc1_done=None,
    lamport=False,
    hazard_delay_ns=0,
    hazard_nowait=False,
):
    """Routing fused into FC1: the ``routing.run`` + ``run`` pair as one kernel, with the same ``plan`` contents and
    bit-identical ``act`` / ``act_sf``. Every CTA scans ``topk_ids`` [R, 16] (global ids; -1 rows unused) for its
    own copy of the plan and publishes a share of ``plan`` for FC2 and the finalize. Launches one kernel on the
    current stream; no allocation, no host synchronization.

    ``prefetch``: the weight TMA warp issues the stages of its speculative first unit (weight tile bidx % 48 of local
    expert bidx // 48) at launch, before ``griddepcontrol.wait``; they are used when the CTA's first item is that unit
    and dropped otherwise. ``source``: ``CuteA2A.recv_source()`` of a ``send_only`` dispatch (the combine advances
    the byte-counter baselines) or ``CuteA2A.routing_source()`` of a ``defer_data_wait`` dispatch whose receive
    region ``x`` / ``x_sf`` / ``topk_ids`` are: each 32-row step of the scan first waits for the byte counters of its
    source blocks. ``plan_pub`` (int32 word, zero-armed): every CTA adds 1 after its plan share, so a consumer that
    starts before FC1 ends reads the plan once the word reaches ``num_ctas_for(x, plan)``; required with a
    ``defer_data_wait`` source (the add that completes the grid advances the byte-counter baselines). ``fc1_done``,
    ``lamport``: as in ``run``.
    ``hazard_delay_ns`` delays the plan share of the CTAs that publish an expert (and, with ``fc1_done``, the
    token-tile-0 stores); ``hazard_nowait`` skips the source waits (hazard tests only)."""
    _check(x, x_sf, weights, plan, act, act_sf)
    rows = x.shape[0]
    if (
        topk_ids.dtype != torch.int32
        or tuple(topk_ids.shape) != (rows, TOP_K)
        or not topk_ids.is_contiguous()
        or topk_ids.data_ptr() % 16
    ):
        raise ValueError(f"topk_ids must be contiguous 16-byte aligned int32 [{rows}, {TOP_K}]")
    num_local = weights.num_local
    if (
        plan.expert_counts.numel() != num_local
        or rows > plan.max_rows
        or plan.max_rows * TOP_K > 1 << 16
    ):
        raise ValueError(
            f"plan of {plan.expert_counts.numel()} experts / {plan.max_rows} rows cannot take {rows} rows"
        )
    _check_word("fc1_done", fc1_done, plan.tile_idx_to_expert_idx.shape[0] * DONE_PER_TILE)
    _check_word("plan_pub", plan_pub, 1)
    src, recv = None, (0, 1, 0)
    if source is not None:
        if rows != source.ep * source.max_tokens or source.ep > 32:
            raise ValueError(
                f"{rows} rows are not the receive region of {source.ep} x {source.max_tokens}"
            )
        if hasattr(source, "off_self"):
            src = (
                True,
                source.ep,
                source.bytes_per_token,
                source.off_flag,
                source.off_recv,
                source.off_cnt,
                source.off_base,
                source.off_self,
            )
        else:
            if plan_pub is None:
                raise ValueError(
                    "a defer_data_wait source needs plan_pub (its last add advances the baselines)"
                )
            lay = source.lay
            src = (
                False,
                source.ep,
                source.bytes_per_token,
                lay.flag,
                lay.recv_counters,
                lay.dispatch_counters,
                lay.dispatch_baseline,
                0,
            )
        recv = (source.my_ws, source.max_tokens, source.rank)
    key = (
        "routed",
        plan.tile_n,
        weights.gate_cap,
        weights.linear_cap,
        _use_pdl(),
        fc1_done is not None,
        hazard_delay_ns if (fc1_done is not None or plan_pub is not None or lamport) else 0,
        num_local,
        plan.max_rows,
        prefetch,
        src,
        plan_pub is not None,
        hazard_nowait,
        lamport,
    )
    args = _launch_args(
        x, x_sf, weights, plan, act, act_sf, None, fc1_done, topk_ids, plan_pub, recv
    )
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled = _compiled.get(key)
    if compiled is None:
        kernel = Fc1Kernel(
            plan.tile_n,
            weights.gate_cap,
            weights.linear_cap,
            key[4],
            signal_done=fc1_done is not None,
            delay_ns=key[6],
            routed=True,
            num_local=num_local,
            max_rows=plan.max_rows,
            prefetch=prefetch,
            source=src,
            plan_pub=plan_pub is not None,
            nowait=hazard_nowait,
            lamport=lamport,
        )
        compiled = _compiled[key] = cute.compile(kernel, *args, stream)
        k3_edges.debug(
            "moe.fc1_v2",
            rows,
            f"routed build tile {plan.tile_n}: {kernel.num_ab_stage} stages, {kernel.num_scan_warps} scan "
            f"warps x {kernel.seg} pairs, sorted {kernel.sorted_cap} / {2 * kernel.mask_base}, smem "
            f"{kernel.shared_storage.size_in_bytes()} + {kernel.routing_storage.size_in_bytes()}",
        )
    compiled(*args, stream)


FGS_WAITS = {"run_v2": ("recv_fc1",)}
FGS_PUBLISHES = {"run_v2": ("fc1_fc2", "fc1_plan")}


def run_v2(x, x_sf, ids, weights_e, plan, act, act_sf, *, fgs=None, enable_pdl=True):
    """``TLLM_K3_MOE_FC1=v2`` entry (moe_chain contract): ``run_routed`` with its sync objects from ``fgs``.

    ``fgs.waits["recv_fc1"]``: the receive of a send-only dispatch; its ``source`` is the dispatch's receive
    descriptor (``CuteA2A.routing_source()``). With it the weight stages of the speculative first unit are issued at
    launch (FUSED-ON); ``diag`` ``oracle`` skips the counter waits, ``ready`` drops the early weight stages.
    ``fgs.publishes["fc1_fc2"]``: the ``fc1_done`` slab, or for ``kind == "lamport"`` the Lamport build (``act`` /
    ``act_sf`` are the slots). ``fgs.publishes["fc1_plan"]``: the ``plan_pub`` word (needed
    with either edge; an FC2 that overlaps this kernel waits for it to reach ``num_ctas_for(x, plan)``).
    ``enable_pdl`` must match ``TRTLLM_ENABLE_PDL``, which the kernel reads."""
    waits = fgs.waits if fgs is not None else {}
    pubs = fgs.publishes if fgs is not None else {}
    if enable_pdl != _use_pdl():
        raise ValueError("fc1.run_v2: enable_pdl must match TRTLLM_ENABLE_PDL")
    recv = waits.get("recv_fc1")
    done = pubs.get("fc1_fc2")
    pub = pubs.get("fc1_plan")
    if done is not None and pub is None:
        raise ValueError("fc1.run_v2: fc1_fc2 needs the fc1_plan word")
    if recv is not None and getattr(recv, "source", None) is None:
        raise ValueError(
            "fc1.run_v2: recv_fc1 needs the dispatch receive descriptor (EdgeSync.source)"
        )
    delay = max((e.hazard_delay_ns for e in (done, pub) if e is not None), default=0)
    run_routed(
        x,
        x_sf,
        ids,
        weights_e,
        plan,
        act,
        act_sf,
        source=None if recv is None else recv.source,
        prefetch=recv is not None and recv.diag != "ready",
        plan_pub=None if pub is None else pub.tensor,
        fc1_done=None if done is None or done.kind == "lamport" else done.tensor,
        lamport=done is not None and done.kind == "lamport",
        hazard_delay_ns=delay,
        hazard_nowait=recv is not None and (recv.hazard_nowait or recv.diag == "oracle"),
    )
