# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from FlashInfer (05ebb2d79)
# flashinfer/fused_moe/cute_dsl/blackwell/blockscaled_swapab_grouped_gemm.py, Apache-2.0.
"""Kimi K3 decode MoE FC2 (down projection) for SM107 in CuTe DSL.

``run(act, act_sf, weights, plan, y)``: for every valid permuted row p of local expert e,
``y[p] = bf16(dequant(act[p]) @ dequant(W2[e]).T)`` with fp32 accumulation over K = 3072 and one round-to-nearest
conversion. Padding rows of ``y`` are not written.

Swap-AB block-scaled grouped GEMM:

    acc[128 weight rows, n tokens] = W2[e][128-row tile, :]     (A: MXFP4, TMA)
                                   x act[n permuted rows, :]^T  (B: MXFP8, cp.async)

``tcgen05.mma kind::mxf8f6f4`` with the SM107 MMA-K of 64; the FP4 weights stay packed in shared memory. The
TRTLLM-Gen weight buffers are read in place: ``w2_weight`` rows are shuffled in 32-row blocks (stored row 32b + j
holds output channel 32b + 4 (j % 8) + j // 8) and ``w2_weight_scale`` is already in the 128x4 scale-factor atom
layout the MMA consumes. ``act_sf`` is linear UE8M0 [P, 96]; the loader warp drops each row's four bytes per
128-wide K block into the scale-factor atom.

Work unit = (group g, 128-row weight tile m), u = 28 g + m. A group is up to ``group`` consecutive token tiles of
one expert (n = group x tile_n MMA columns), so their weights are read once; the loader warp builds the group table
from the routing plan at kernel start. Units are dealt round-robin to a persistent grid in equal rounds. Each output
element comes from one CTA's fixed-order K loop, so the result is deterministic and independent of the grid size.

Warp roles: 0-3 epilogue (thread = TMEM lane = stored weight row), 4 MMA, 5 weight TMA, 6 activation cp.async.

Fine-grained sync with the neighbouring stages (int32 counter slabs, zero-armed by the caller, T = tile slots):
- ``fc1_done[T * 12]``: before K stage k of token tile t the activation loader waits until ``fc1_done[t * 12 + k] >= 4``
  (FC1's four 64-column units of that stage). A gate warp (7) polls the counters ahead of the loader and hands it one
  smem token per stage; the weight TMA does not wait.
- ``fc2_done[T]``: after a unit's ``y`` stores, a release add of 1 per token tile of the unit, so tile t is complete at
  28. The epilogue hands each unit to a signal warp (7 or 8) through an mbarrier; that warp issues one gpu-scope fence
  per unit and relaxed adds, keeping the fence off the epilogue.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
import torch
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import dsl_user_op

try:
    from cutlass.memory import TmemAllocator
except ImportError:
    from cutlass.utils import TmemAllocator

HIDDEN = 3584
INTER = 3072
SF_VEC = 32
TILE_M = 128
K_STAGE = 256
MMA_K = 64
M_TILES = HIDDEN // TILE_M
K_TILES = INTER // K_STAGE
# Above this much dynamic smem the remaining L1 starves the cp.async activation loads (measured on SM107).
SMEM_BUDGET = 232448
TILE_NS = (8, 16, 32)
# Diagnostic timeline (TLLM_K3_MOE_FC2_TIMELINE=1), %globaltimer_lo per CTA: [0] entry, [1] first weight load
# issued, [2 + i] unit i stored (first 29 units), [31] number of units.
TIMELINE_FIELDS = 32
SCAN_BATCHES = 4
FC1_UNITS_PER_K_STAGE = K_STAGE // 64
SIGNAL_STAGES = 4


def _pow2_at_least(n: int, lo: int) -> int:
    p = lo
    while p < n:
        p *= 2
    return p


@dsl_user_op
def _red_add_relaxed_gpu(ptr: cute.Pointer, val: cutlass.Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _tcgen05_fence_after_thread_sync():
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)


def _tcgen05_fence_before_thread_sync():
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)


class Fc2Kernel:
    def __init__(
        self,
        tile_n: int,
        num_local: int,
        num_stages: int,
        use_pdl: bool,
        paired_store: bool = True,
        group: int = 2,
        group_cap: int = 1024,
        timeline: bool = False,
        wait_fc1: bool = False,
        signal_fc2: bool = False,
    ):
        if tile_n not in TILE_NS:
            raise ValueError(f"tile_n must be one of {TILE_NS}, got {tile_n}")
        if group < 1 or group * tile_n > 128:
            raise ValueError(f"group * tile_n must be in [tile_n, 128], got {group} x {tile_n}")
        if num_local > 128:
            raise ValueError(
                f"at most 128 local experts (7-bit group entry field), got {num_local}"
            )
        self.tile_n = tile_n
        self.num_local = num_local
        self.num_ab_stage = num_stages
        self.num_acc_stage = 2
        self.use_pdl = use_pdl
        self.paired_store = paired_store
        self.group = group
        self.n_max = group * tile_n
        self.group_cap = group_cap
        self.timeline = timeline
        self.wait_fc1 = wait_fc1
        self.signal_fc2 = signal_fc2
        self.mma_tiler = (TILE_M, self.n_max, K_STAGE)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.a_dtype = cutlass.Float4E2M1FN
        self.b_dtype = cutlass.Float8E4M3FN
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.acc_dtype = cutlass.Float32
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.load_warp_id = 6
        self.gate_warp_id = 7
        self.signal_warp_id = 7 + int(wait_fc1)
        self.threads_per_cta = 32 * (7 + int(wait_fc1) + int(signal_fc2))
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * 4)
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * 5)
        self.b_bytes_per_stage = self.n_max * K_STAGE
        self.sf_atom_bytes = 512
        self.num_acc_tmem_cols = self.n_max * self.num_acc_stage

    @staticmethod
    def stage_bytes(n_max: int) -> int:
        # Packed FP4 A + FP8 B + one 512 B scale-factor atom per 128 K for each of SFA and SFB (SFB rows round up
        # to the 128-row atom).
        return TILE_M * K_STAGE // 2 + n_max * K_STAGE + 2 * (K_STAGE // 128) * 512

    @staticmethod
    def max_stages(n_max: int, group_cap: int, smem: int = SMEM_BUDGET) -> int:
        reserve = 6 * 1024 + 4 * (group_cap + 1)
        return min((smem - reserve) // Fc2Kernel.stage_bytes(n_max), 16)

    def _collect_groups(
        self, t, expert, prev, nxt, limit, num_t, seg_carry, n_groups, sGroup, lane
    ):
        """One warp-wide step of the group scan over 32 tiles: returns (group count, last expert start so far).

        Groups pair an expert's tiles from its first tile, so they never span experts. The last tile of each group
        writes the entry: first tile | valid rows << 16 | expert << 24 (other lanes write the spare last slot). A
        tile's valid rows are its mn_limit clipped to the tile, so the plan may give either the per-tile limit or
        the expert's end.
        """
        tile_n, group = self.tile_n, self.group
        valid = t < num_t
        starts = cute.arch.vote_ballot_sync(valid & ((t == 0) | (prev != expert)))
        base = t - lane
        # The expert's first tile: the last start at or below this lane, else the one carried from earlier batches.
        at_or_below = starts & ((cutlass.Int32(2) << lane) - 1)
        seg_start = cutlass.select_(
            at_or_below != 0,
            base + cutlass.Int32(cute.arch.bfind(cutlass.Uint32(at_or_below))),
            seg_carry,
        )
        pos = (t - seg_start) % group
        closes = valid & ((pos == group - 1) | (t + 1 >= num_t) | (nxt != expert))
        rows = pos * tile_n + cutlass.min(limit - t * tile_n, tile_n)
        ballot = cute.arch.vote_ballot_sync(closes)
        g = n_groups + cute.arch.popc(ballot & ((cutlass.Int32(1) << lane) - 1))
        sGroup[cutlass.select_(closes, g, cutlass.Int32(self.group_cap))] = (
            (t - pos) | (rows << 16) | (expert << 24)
        )
        seg_carry = cutlass.select_(
            starts != 0, base + cutlass.Int32(cute.arch.bfind(cutlass.Uint32(starts))), seg_carry
        )
        return n_groups + cute.arch.popc(ballot), seg_carry

    @cute.jit
    def _fc1_missing(self, fc1_done, first_tile, n_tiles, lane):
        """Warp-wide acquire of all fc1_done words of a unit's token tiles: the number not yet complete."""
        missing = cutlass.Int32(0)
        for c in cutlass.range_constexpr((self.group * K_TILES + 31) // 32):
            w = c * 32 + lane
            seen = cutlass.Int32(FC1_UNITS_PER_K_STAGE)
            if w < n_tiles * K_TILES:
                seen = cute.arch.load(
                    fc1_done.iterator + first_tile * K_TILES + w,
                    cutlass.Int32,
                    sem="acquire",
                    scope="gpu",
                )
            ready = cute.arch.vote_ballot_sync(seen >= FC1_UNITS_PER_K_STAGE)
            missing += 32 - cute.arch.popc(ready)
        return missing

    @cute.jit
    def _wait_fc1_stage(self, fc1_done, first_tile, n_tiles, k_tile):
        for j in cutlass.range_constexpr(self.group):
            if j < n_tiles:
                word = fc1_done.iterator + (first_tile + j) * K_TILES + k_tile
                seen = cute.arch.load(word, cutlass.Int32, sem="acquire", scope="gpu")
                while seen < FC1_UNITS_PER_K_STAGE:
                    seen = cute.arch.load(word, cutlass.Int32, sem="acquire", scope="gpu")

    @cute.jit
    def _signal_fc2(self, fc2_done, first_tile, n_tiles):
        # fence + relaxed adds is a release; one fence for all tiles of the unit.
        cute.arch.fence_acq_rel_gpu()
        for j in cutlass.range_constexpr(self.group):
            if j < n_tiles:
                _red_add_relaxed_gpu(fc2_done.iterator + first_tile + j, cutlass.Int32(1))

    def _tmem_cols(self, sf_tmem_layout) -> int:
        # TMEM addresses are (lane << 16) | column: the low 16 bits of the 32-bit cosize are the columns.
        return cute.cosize(cute.recast_layout(32, self.sf_dtype.width, sf_tmem_layout)) & 0xFFFF

    def _s2t_copy_and_partition(self, sSF: cute.Tensor, tSF: cute.Tensor):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        # The 32x128b copy is multicast to the four TMEM sub-partitions: give the source an MN broadcast mode.
        smem_layout = tCsSF_compact.layout
        mn_mode = cute.append(
            cute.get(smem_layout, mode=[0, 0]), cute.make_layout((4,), stride=(0,))
        )
        bcast = cute.append(cute.group_modes(mn_mode, 0), cute.get(smem_layout, mode=[0, 1]))
        bcast = cute.append(cute.group_modes(bcast, 0), cute.get(smem_layout, mode=[1]))
        bcast = cute.append(bcast, cute.get(smem_layout, mode=[2]))
        bcast = cute.append(bcast, cute.get(smem_layout, mode=[3]))
        tCsSF_bcast = cute.make_tensor(tCsSF_compact.iterator, bcast)
        tCsSF_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, thr_copy_s2t.partition_S(tCsSF_bcast)
        )
        tCtSF_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_s2t, tCtSF_s2t

    @cute.jit
    def __call__(
        self,
        w2_ptr: cute.Pointer,
        w2_sf_ptr: cute.Pointer,
        act_ptr: cute.Pointer,
        act_sf_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        tile_expert_ptr: cute.Pointer,
        tile_limit_ptr: cute.Pointer,
        num_tiles_ptr: cute.Pointer,
        stamps_ptr: Optional[cute.Pointer],
        fc1_done_ptr: Optional[cute.Pointer],
        fc2_done_ptr: Optional[cute.Pointer],
        num_rows: cutlass.Int32,
        num_tile_slots: cutlass.Int32,
        grid_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        mA = cute.make_tensor(
            w2_ptr, cute.make_ordered_layout((HIDDEN, INTER, self.num_local), order=(1, 0, 2))
        )
        mSFA = cute.make_tensor(
            w2_sf_ptr,
            blockscaled_utils.tile_atom_to_shape_SF((HIDDEN, INTER, self.num_local), SF_VEC),
        )
        act = cute.make_tensor(act_ptr, cute.make_ordered_layout((num_rows, INTER), order=(1, 0)))
        act_sf = cute.make_tensor(
            act_sf_ptr, cute.make_ordered_layout((num_rows, INTER // SF_VEC), order=(1, 0))
        )
        y = cute.make_tensor(y_ptr, cute.make_ordered_layout((num_rows, HIDDEN), order=(1, 0)))
        tile_expert = cute.make_tensor(tile_expert_ptr, cute.make_layout((num_tile_slots,)))
        tile_limit = cute.make_tensor(tile_limit_ptr, cute.make_layout((num_tile_slots,)))
        num_tiles = cute.make_tensor(num_tiles_ptr, cute.make_layout((1,)))
        stamps = None
        if cutlass.const_expr(self.timeline):
            stamps = cute.make_tensor(stamps_ptr, cute.make_layout((grid_size * TIMELINE_FIELDS,)))
        fc1_done = None
        if cutlass.const_expr(self.wait_fc1):
            fc1_done = cute.make_tensor(fc1_done_ptr, cute.make_layout((num_tile_slots * K_TILES,)))
        fc2_done = None
        if cutlass.const_expr(self.signal_fc2):
            fc2_done = cute.make_tensor(fc2_done_ptr, cute.make_layout((num_tile_slots,)))

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            SF_VEC,
            self.cta_group,
            (TILE_M, self.n_max, MMA_K),
        )
        a_smem_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        b_smem_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        sfa_smem_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC, self.num_ab_stage
        )
        sfb_smem_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, self.num_ab_stage
        )
        sfa_smem = cute.slice_(sfa_smem_staged, (None, None, None, 0))
        sfb_smem = cute.slice_(sfb_smem_staged, (None, None, None, 0))
        self.num_sfa_tmem_cols = self._tmem_cols(
            blockscaled_utils.make_tmem_layout_sfa(tiled_mma, self.mma_tiler, SF_VEC, sfa_smem)
        )
        self.num_sfb_tmem_cols = self._tmem_cols(
            blockscaled_utils.make_tmem_layout_sfb(tiled_mma, self.mma_tiler, SF_VEC, sfb_smem)
        )
        self.num_tmem_alloc_cols = _pow2_at_least(
            self.num_acc_tmem_cols + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols, 32
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        a_smem = cute.slice_(a_smem_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mA, a_smem, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mSFA,
            sfa_smem,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        self.num_tma_load_bytes = cute.size_in_bytes(self.a_dtype, a_smem) + cute.size_in_bytes(
            self.sf_dtype, sfa_smem
        )

        a_bytes = cute.size_in_bytes(self.a_dtype, a_smem) * self.num_ab_stage

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            load_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            gate_mbar: cute.struct.MemRange[cutlass.Int64, K_TILES * 2 if self.wait_fc1 else 1]
            signal_mbar: cute.struct.MemRange[
                cutlass.Int64, SIGNAL_STAGES * 2 if self.signal_fc2 else 1
            ]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            num_groups: cute.struct.MemRange[cutlass.Int32, 1]
            groups: cute.struct.MemRange[cutlass.Int32, self.group_cap + 1]
            sA: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, a_bytes], 1024]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged.outer)], 1024
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_staged)], 1024
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_staged)], 1024
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_sfa,
            tma_tensor_sfa,
            act,
            act_sf,
            y,
            tile_expert,
            tile_limit,
            num_tiles,
            stamps,
            fc1_done,
            fc2_done,
            cluster_layout_vmnk,
            a_smem_staged,
            b_smem_staged,
            sfa_smem_staged,
            sfb_smem_staged,
        ).launch(
            grid=(grid_size, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        act: cute.Tensor,
        act_sf: cute.Tensor,
        y: cute.Tensor,
        tile_expert: cute.Tensor,
        tile_limit: cute.Tensor,
        num_tiles: cute.Tensor,
        stamps: Optional[cute.Tensor],
        fc1_done: Optional[cute.Tensor],
        fc2_done: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.Layout,
        sfb_smem_staged: cute.Layout,
    ):
        tile_n = self.tile_n
        n_max = self.n_max
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        if cutlass.const_expr(self.timeline):
            if tidx == 0:
                stamps[bidx * TIMELINE_FIELDS] = cute.arch.globaltimer_lo()

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # Activation rows + scale factors: every lane of the loader warp arrives through
        # cp.async.mbarrier.arrive.noinc once its own copies of the stage have landed.
        load_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.load_mbar.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        load_pipeline = dataclasses.replace(
            load_pipeline,
            sync_object_full=load_pipeline.sync_object_full.recast_to_new_op_type(
                pipeline.PipelineOp.AsyncLoad
            ),
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * 4),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        if cutlass.const_expr(self.wait_fc1):
            gate_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.gate_mbar.data_ptr(),
                num_stages=K_TILES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                defer_sync=True,
            )
        if cutlass.const_expr(self.signal_fc2):
            signal_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.signal_mbar.data_ptr(),
                num_stages=SIGNAL_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * 4),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                defer_sync=True,
            )
        tmem = TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline.pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sA = storage.sA.get_tensor(
            a_smem_staged.outer, swizzle=a_smem_staged.inner, dtype=self.a_dtype
        )
        sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_staged)
        sNumGroups = storage.num_groups.get_tensor(cute.make_layout((1,)))
        sGroup = storage.groups.get_tensor(cute.make_layout((self.group_cap + 1,)))

        # (bM, bK, nM, nK, L)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (TMA, nM, nK, L)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # With fc1_done, only FC1's act can be outstanding at launch (FC1 triggers dependents after its own wait, so the
        # plan is complete); the act is guarded per K stage.
        if cutlass.const_expr(self.use_pdl and not self.wait_fc1):
            cute.arch.griddepcontrol_wait()
        if warp_idx == self.load_warp_id:
            lane = tidx % 32
            last_slot = tile_expert.shape[0] - 1
            # The first SCAN_BATCHES x 32 tiles are read together with num_tiles: one round trip in the common case.
            batch = []
            for b in cutlass.range_constexpr(SCAN_BATCHES):
                t_safe = cutlass.min(b * 32 + lane, last_slot)
                batch.append(
                    (
                        tile_expert[t_safe],
                        tile_expert[cutlass.max(t_safe - 1, 0)],
                        tile_expert[cutlass.min(t_safe + 1, last_slot)],
                        tile_limit[t_safe],
                    )
                )
            num_t = num_tiles[0]
            n_groups = cutlass.Int32(0)
            seg_carry = cutlass.Int32(0)
            for b in cutlass.range_constexpr(SCAN_BATCHES):
                expert, prev, nxt, limit = batch[b]
                n_groups, seg_carry = self._collect_groups(
                    b * 32 + lane,
                    expert,
                    prev,
                    nxt,
                    limit,
                    num_t,
                    seg_carry,
                    n_groups,
                    sGroup,
                    lane,
                )
            for base in cutlass.range(SCAN_BATCHES * 32, num_t, 32, unroll=1):
                t_safe = cutlass.min(base + lane, last_slot)
                n_groups, seg_carry = self._collect_groups(
                    base + lane,
                    tile_expert[t_safe],
                    tile_expert[t_safe - 1],
                    tile_expert[cutlass.min(t_safe + 1, last_slot)],
                    tile_limit[t_safe],
                    num_t,
                    seg_carry,
                    n_groups,
                    sGroup,
                    lane,
                )
            if lane == 0:
                sNumGroups[0] = n_groups
        pipeline.pipeline_init_wait(cluster_shape_mn=(1, 1))
        cute.arch.griddepcontrol_launch_dependents()
        # Equal rounds: the fewest CTAs that keep the round count, so no CTA finishes a unit behind the rest.
        num_units = sNumGroups[0] * M_TILES
        rounds = (num_units + grid_x - 1) // grid_x
        used_ctas = (num_units + cutlass.max(rounds, 1) - 1) // cutlass.max(rounds, 1)
        num_mine = cutlass.max(num_units - bidx + used_ctas - 1, cutlass.Int32(0)) // cutlass.max(
            used_ctas, 1
        )
        num_mine = cutlass.select_(bidx < used_ctas, num_mine, cutlass.Int32(0))

        if warp_idx == self.tma_warp_id:
            ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            for i in cutlass.range(num_mine, unroll=1):
                unit = bidx + i * used_ctas
                m_tile = unit % M_TILES
                expert = sGroup[unit // M_TILES] >> 24
                tAgA_unit = tAgA[(None, m_tile, None, expert)]
                tAgSFA_unit = tAgSFA[(None, m_tile, None, expert)]
                if cutlass.const_expr(self.timeline):
                    if i == 0:
                        if tidx % 32 == 0:
                            stamps[bidx * TIMELINE_FIELDS + 1] = cute.arch.globaltimer_lo()
                for k_tile in cutlass.range(K_TILES, unroll=1):
                    ab_pipeline.producer_acquire(ab_state)
                    tma_bar = ab_pipeline.producer_get_barrier(ab_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_unit[(None, k_tile)],
                        tAsA[(None, ab_state.index)],
                        tma_bar_ptr=tma_bar,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_unit[(None, k_tile)],
                        tAsSFA[(None, ab_state.index)],
                        tma_bar_ptr=tma_bar,
                    )
                    ab_state.advance()
            ab_pipeline.producer_tail(ab_state)

        # Predicated-off cp.async copies zero-fill their destination: padding rows of the group reach the MMA as 0.
        # Rows past the group are not loaded; their accumulator columns are never stored.
        if warp_idx == self.load_warp_id:
            lane = tidx % 32
            chunk = lane % 8
            row_in_pass = lane // 8
            n_pass = n_max // 4
            n_sf = max(1, n_max // 32)
            n_katom = K_STAGE // 128
            b_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                self.b_dtype,
                num_bits_per_copy=128,
            )
            sf_copy = cute.make_copy_atom(cpasync.CopyG2SOp(), self.sf_dtype, num_bits_per_copy=32)
            pred = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)
            row_src = cute.make_rmem_tensor((n_pass,), cutlass.Int32)
            row_ok = cute.make_rmem_tensor((n_pass,), cutlass.Boolean)
            sf_src = cute.make_rmem_tensor((n_sf,), cutlass.Int32)
            sf_ok = cute.make_rmem_tensor((n_sf,), cutlass.Boolean)
            load_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            if cutlass.const_expr(self.wait_fc1):
                gate_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, K_TILES
                )
            for i in cutlass.range(num_mine, unroll=1):
                entry = sGroup[(bidx + i * used_ctas) // M_TILES]
                row_base = (entry & 0xFFFF) * tile_n
                valid_rows = (entry >> 16) & 0xFF
                group_rows = (valid_rows + tile_n - 1) // tile_n * tile_n
                row_end = row_base + valid_rows
                for p in cutlass.range_constexpr(n_pass):
                    row = row_base + p * 4 + row_in_pass
                    ok = row < row_end
                    row_ok[p] = ok
                    row_src[p] = row * cutlass.Int32(ok)
                for q in cutlass.range_constexpr(n_sf):
                    sf_row = row_base + q * 32 + lane
                    ok = (q * 32 + lane < n_max) & (sf_row < row_end)
                    sf_ok[q] = ok
                    sf_src[q] = sf_row * cutlass.Int32(ok)
                for k_tile in cutlass.range(K_TILES, unroll=1):
                    load_pipeline.producer_acquire(load_state)
                    if cutlass.const_expr(self.wait_fc1):
                        # Every lane acquires the token. cp.async (non-bulk) reads in the generic proxy, so no proxy
                        # fence is needed.
                        gate_pipeline.consumer_wait(gate_state)
                        if lane == 0:
                            gate_pipeline.consumer_release(gate_state)
                        gate_state.advance()
                    sB_stage = sB.iterator + load_state.index * self.b_bytes_per_stage
                    sSFB_stage = sSFB.iterator + load_state.index * (n_katom * self.sf_atom_bytes)
                    for kt in cutlass.range_constexpr(n_katom):
                        k0 = k_tile * K_STAGE + kt * 128
                        for p in cutlass.range_constexpr(n_pass):
                            if p * 4 < group_rows:
                                row = p * 4 + row_in_pass
                                src = cute.make_tensor(
                                    act.iterator
                                    + cute.assume(row_src[p] * INTER + k0 + chunk * 16, divby=16),
                                    cute.make_layout((16,)),
                                )
                                # sB.iterator applies the 128B swizzle to this logical (K atom, row, chunk) offset.
                                dst = cute.make_tensor(
                                    sB_stage + (kt * n_max * 128 + row * 128 + chunk * 16),
                                    cute.make_layout((16,)),
                                )
                                pred[0] = row_ok[p]
                                cute.copy_atom_call(b_copy, src, dst, pred=pred)
                        # Scale-factor atom (128 rows x 4 K blocks, 512 B): row r at byte 16 (r % 32) + 4 (r / 32).
                        for q in cutlass.range_constexpr(n_sf):
                            sf_g = cute.make_tensor(
                                act_sf.iterator
                                + cute.assume(
                                    sf_src[q] * (INTER // SF_VEC)
                                    + k_tile * (K_STAGE // SF_VEC)
                                    + kt * 4,
                                    divby=4,
                                ),
                                cute.make_layout((4,)),
                            )
                            sf_s = cute.make_tensor(
                                sSFB_stage + (kt * self.sf_atom_bytes + lane * 16 + q * 4),
                                cute.make_layout((4,)),
                            )
                            pred[0] = sf_ok[q]
                            cute.copy_atom_call(sf_copy, sf_g, sf_s, pred=pred)
                    load_pipeline.producer_commit(load_state)
                    load_state.advance()
            load_pipeline.producer_tail(load_state)

        if cutlass.const_expr(self.wait_fc1):
            if warp_idx == self.gate_warp_id:
                lane = tidx % 32
                gate_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, K_TILES
                )
                for i in cutlass.range(num_mine, unroll=1):
                    entry = sGroup[(bidx + i * used_ctas) // M_TILES]
                    first_tile = entry & 0xFFFF
                    n_tiles = (((entry >> 16) & 0xFF) + tile_n - 1) // tile_n
                    missing = self._fc1_missing(fc1_done, first_tile, n_tiles, lane)
                    for k_tile in cutlass.range(K_TILES, unroll=1):
                        if missing > 0:
                            self._wait_fc1_stage(fc1_done, first_tile, n_tiles, k_tile)
                        # Lane 0's arrive is ordered after every lane's acquire.
                        cute.arch.sync_warp()
                        if lane == 0:
                            gate_pipeline.producer_acquire(gate_state)
                            gate_pipeline.producer_commit(gate_state)
                        gate_state.advance()

        if cutlass.const_expr(self.signal_fc2):
            if warp_idx == self.signal_warp_id:
                if tidx % 32 == 0:
                    signal_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, SIGNAL_STAGES
                    )
                    for i in cutlass.range(num_mine, unroll=1):
                        entry = sGroup[(bidx + i * used_ctas) // M_TILES]
                        signal_pipeline.consumer_wait(signal_state)
                        self._signal_fc2(
                            fc2_done,
                            entry & 0xFFFF,
                            (((entry >> 16) & 0xFF) + tile_n - 1) // tile_n,
                        )
                        signal_pipeline.consumer_release(signal_state)
                        signal_state.advance()

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_acc_tmem_cols, dtype=self.sf_dtype
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                SF_VEC,
                cute.slice_(sfa_smem_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_acc_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                SF_VEC,
                cute.slice_(sfb_smem_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = self._s2t_copy_and_partition(sSFA, tCtSFA)
            s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = self._s2t_copy_and_partition(sSFB, tCtSFB)

            ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            for _ in cutlass.range(num_mine, unroll=1):
                acc_pipeline.producer_acquire(acc_state)
                _tcgen05_fence_after_thread_sync()
                tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                for k_tile in cutlass.range(K_TILES, unroll=1):
                    ab_pipeline.consumer_wait(ab_state)
                    load_pipeline.consumer_wait(ab_state)
                    # cp.async (generic proxy) writes -> tcgen05 (async proxy) reads
                    cute.arch.fence_proxy("async.shared", space="cta")
                    s2t_stage = (None, None, None, None, ab_state.index)
                    cute.copy(s2t_sfa, tCsSFA_s2t[s2t_stage], tCtSFA_s2t)
                    cute.copy(s2t_sfb, tCsSFB_s2t[s2t_stage], tCtSFB_s2t)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    mma_stage = (None, None, None, ab_state.index)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        [tCrA[mma_stage], tCtSFA],
                        [tCrB[mma_stage], tCtSFB],
                        tCtAcc,
                    )
                    ab_pipeline.consumer_release(ab_state)
                    load_pipeline.consumer_release(ab_state)
                    ab_state.advance()
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
            acc_pipeline.producer_tail(acc_state)

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            epi_tile = (TILE_M, n_max)
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.mma_tiler,
                utils.LayoutEnum.ROW_MAJOR,
                cutlass.BFloat16,
                self.acc_dtype,
                epi_tile,
                False,
            )
            # (EPI_M, EPI_N, nM, nN, STAGE)
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            # (T2R, T2R_M, T2R_N, nM, nN, STAGE)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            gC_fake = cute.make_tensor(
                y.iterator, cute.make_layout((TILE_M, n_max), stride=(n_max, 1))
            )
            tTR_gC = thr_copy_t2r.partition_D(cute.flat_divide(gC_fake, epi_tile))
            tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0)].shape, self.acc_dtype)

            lane = tidx % 32
            channel_in_tile = (tidx // 32) * 32 + (lane % 8) * 4 + lane // 8
            # Lanes l and l ^ 8 hold output channels 2j and 2j + 1: each stores one bf16x2 per column pair.
            odd_channel = (lane // 8) % 2
            store_pair = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=32
            )
            pair = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.BFloat16)
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            if cutlass.const_expr(self.signal_fc2):
                signal_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, SIGNAL_STAGES
                )
            for i in cutlass.range(num_mine, unroll=1):
                unit = bidx + i * used_ctas
                entry = sGroup[unit // M_TILES]
                row_base = (entry & 0xFFFF) * tile_n
                row_end = row_base + ((entry >> 16) & 0xFF)
                channel = unit % M_TILES * TILE_M + channel_in_tile
                acc_pipeline.consumer_wait(acc_state)
                _tcgen05_fence_after_thread_sync()
                cute.copy(
                    tiled_copy_t2r,
                    tTR_tAcc[(None, None, None, 0, 0, acc_state.index)],
                    tTR_rAcc,
                )
                cute.arch.fence_view_async_tmem_load()
                acc = tTR_rAcc.load()
                _tcgen05_fence_before_thread_sync()
                acc_pipeline.consumer_release(acc_state)
                acc_state.advance()
                if cutlass.const_expr(self.paired_store):
                    is_odd = odd_channel == 1
                    pair_channel = channel - odd_channel
                    for c in cutlass.range_constexpr(0, n_max, 2):
                        recv = cute.arch.shuffle_sync_bfly(
                            cutlass.select_(is_odd, acc[c], acc[c + 1]), offset=8
                        )
                        row = row_base + c + odd_channel
                        if row < row_end:
                            pair[0] = cutlass.select_(is_odd, recv, acc[c]).to(cutlass.BFloat16)
                            pair[1] = cutlass.select_(is_odd, acc[c + 1], recv).to(cutlass.BFloat16)
                            dst = cute.make_tensor(
                                y.iterator + cute.assume(row * HIDDEN + pair_channel, divby=2),
                                cute.make_layout((2,)),
                            )
                            cute.copy(store_pair, pair, dst)
                else:
                    for c in cutlass.range_constexpr(n_max):
                        if row_base + c < row_end:
                            y[(row_base + c, channel)] = acc[c].to(cutlass.BFloat16)
                if cutlass.const_expr(self.signal_fc2):
                    # Each thread's mbarrier arrive releases its own y stores to the signal warp (CTA scope).
                    signal_pipeline.producer_acquire(signal_state)
                    signal_pipeline.producer_commit(signal_state)
                    signal_state.advance()
                if cutlass.const_expr(self.timeline):
                    if tidx == 0:
                        if i < TIMELINE_FIELDS - 3:
                            stamps[bidx * TIMELINE_FIELDS + 2 + i] = cute.arch.globaltimer_lo()

            if cutlass.const_expr(self.timeline):
                if tidx == 0:
                    stamps[bidx * TIMELINE_FIELDS + TIMELINE_FIELDS - 1] = num_mine
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

        # Completion of this grid still implies FC1's.
        if cutlass.const_expr(self.use_pdl and self.wait_fc1):
            cute.arch.griddepcontrol_wait()


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(f"TLLM_K3_MOE_FC2_{name}", default))


def _ptr(dtype, tensor: torch.Tensor, align: int):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


@functools.cache
def _compiled(
    tile_n: int,
    num_local: int,
    num_stages: int,
    use_pdl: bool,
    paired_store: bool,
    group: int,
    group_cap: int,
    timeline: bool,
    wait_fc1: bool = False,
    signal_fc2: bool = False,
):
    kernel = Fc2Kernel(
        tile_n,
        num_local,
        num_stages,
        use_pdl,
        paired_store,
        group,
        group_cap,
        timeline,
        wait_fc1,
        signal_fc2,
    )
    fake = cute.AddressSpace.gmem
    args = (
        make_ptr(cutlass.Float4E2M1FN, 1 << 20, fake, assumed_align=32),
        make_ptr(cutlass.Float8E8M0FNU, 1 << 20, fake, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 1 << 20, fake, assumed_align=16),
        make_ptr(cutlass.Float8E8M0FNU, 1 << 20, fake, assumed_align=16),
        make_ptr(cutlass.BFloat16, 1 << 20, fake, assumed_align=16),
        make_ptr(cutlass.Int32, 1 << 20, fake, assumed_align=4),
        make_ptr(cutlass.Int32, 1 << 20, fake, assumed_align=4),
        make_ptr(cutlass.Int32, 1 << 20, fake, assumed_align=4),
        make_ptr(cutlass.Int32, 1 << 20, fake, assumed_align=4) if timeline else None,
        make_ptr(cutlass.Int32, 1 << 20, fake, assumed_align=4) if wait_fc1 else None,
        make_ptr(cutlass.Int32, 1 << 20, fake, assumed_align=4) if signal_fc2 else None,
        cutlass.Int32(tile_n),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cuda.CUstream(0),
    )
    return cute.compile(kernel, *args)


_timeline_buffers: dict = {}


def timeline_buffer(device: torch.device) -> torch.Tensor:
    """Per-CTA stamps, int32 [CTA, TIMELINE_FIELDS], of the last TLLM_K3_MOE_FC2_TIMELINE=1 launch on ``device``."""
    device = torch.device("cuda", torch.device(device).index or torch.cuda.current_device())
    if device not in _timeline_buffers:
        grid = 4 * _sm_count(device.index)
        _timeline_buffers[device] = torch.zeros(
            grid, TIMELINE_FIELDS, dtype=torch.int32, device=device
        )
    return _timeline_buffers[device]


def run(
    act: torch.Tensor,
    act_sf: torch.Tensor,
    weights,
    plan,
    y: torch.Tensor,
    fc1_done: Optional[torch.Tensor] = None,
    fc2_done: Optional[torch.Tensor] = None,
) -> None:
    """``y[p] = bf16(dequant(act[p]) @ dequant(W2[e]).T)`` for every valid permuted row (moe_chain FC2 stage).

    ``fc1_done`` (int32 [T * 12]) and ``fc2_done`` (int32 [T]), T = ``plan.tile_idx_to_expert_idx.numel()``, are the
    counter slabs of the FC1 -> FC2 and FC2 -> finalize edges (module docstring). With ``fc1_done`` the kernel is
    launched with PDL and does not wait for FC1's grid before streaming weights.
    """
    tile_n = plan.tile_n
    rows = act.shape[0]
    assert act.dtype == torch.float8_e4m3fn and act.shape == (rows, INTER) and act.is_contiguous()
    assert (
        act_sf.dtype == torch.uint8
        and act_sf.shape == (rows, INTER // SF_VEC)
        and act_sf.is_contiguous()
    )
    assert y.dtype == torch.bfloat16 and y.shape == (rows, HIDDEN) and y.is_contiguous()
    w2, w2_sf = weights.w2_weight, weights.w2_weight_scale
    num_local = weights.num_local
    assert (
        w2.dtype == torch.uint8
        and w2.shape == (num_local, HIDDEN, INTER // 2)
        and w2.is_contiguous()
    )
    assert w2_sf.shape == (num_local, HIDDEN, INTER // SF_VEC) and w2_sf.is_contiguous()
    tile_slots = plan.tile_idx_to_expert_idx.numel()
    assert tile_slots < 1 << 16
    for done, words in ((fc1_done, tile_slots * K_TILES), (fc2_done, tile_slots)):
        assert done is None or (
            done.dtype == torch.int32 and done.numel() >= words and done.is_contiguous()
        )
    device = act.device
    ctas_per_sm = _env_int("CTAS_PER_SM", 1)
    group = _env_int("GROUP", 2)
    timeline = _env_int("TIMELINE", 0) == 1
    grid = max(1, min(ctas_per_sm * _sm_count(device.index), tile_slots * M_TILES))
    # Groups never outnumber the tile capacity: one shared-memory entry each.
    group_cap = _pow2_at_least(tile_slots, 64)
    stages = _env_int(
        "STAGES", Fc2Kernel.max_stages(group * tile_n, group_cap, SMEM_BUDGET // ctas_per_sm)
    )
    compiled = _compiled(
        tile_n,
        num_local,
        stages,
        _env_int("PDL", 0) == 1 or fc1_done is not None,
        _env_int("PAIRED_STORE", 1) == 1,
        group,
        group_cap,
        timeline,
        fc1_done is not None,
        fc2_done is not None,
    )
    compiled(
        _ptr(cutlass.Float4E2M1FN, w2, 32),
        _ptr(cutlass.Float8E8M0FNU, w2_sf, 16),
        _ptr(cutlass.Float8E4M3FN, act, 16),
        _ptr(cutlass.Float8E8M0FNU, act_sf, 16),
        _ptr(cutlass.BFloat16, y, 16),
        _ptr(cutlass.Int32, plan.tile_idx_to_expert_idx, 4),
        _ptr(cutlass.Int32, plan.tile_idx_to_mn_limit, 4),
        _ptr(cutlass.Int32, plan.num_non_exiting_tiles, 4),
        _ptr(cutlass.Int32, timeline_buffer(device), 4) if timeline else None,
        _ptr(cutlass.Int32, fc1_done, 4) if fc1_done is not None else None,
        _ptr(cutlass.Int32, fc2_done, 4) if fc2_done is not None else None,
        cutlass.Int32(rows),
        cutlass.Int32(tile_slots),
        cutlass.Int32(grid),
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
