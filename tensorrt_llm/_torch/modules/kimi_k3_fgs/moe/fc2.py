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

    acc[128 weight rows, tile_n tokens] = W2[e][128-row tile, :]          (A: MXFP4, TMA)
                                        x act[tile_n permuted rows, :]^T  (B: MXFP8, cp.async)

``tcgen05.mma kind::mxf8f6f4`` with the SM107 MMA-K of 64; the FP4 weights stay packed in shared memory. The
TRTLLM-Gen weight buffers are read in place: ``w2_weight`` rows are shuffled in 32-row blocks (stored row 32b + j
holds output channel 32b + 4 (j % 8) + j // 8) and ``w2_weight_scale`` is already in the 128x4 scale-factor atom
layout the MMA consumes. ``act_sf`` is linear UE8M0 [P, 96]; the loader warp drops each row's four bytes per
128-wide K block into the scale-factor atom.

Work unit = (token tile t, 128-row weight tile m), t-major over ``num_non_exiting_tiles`` x 28 units, dealt
round-robin to a persistent grid. Each output element comes from one CTA's fixed-order K loop, so the result is
deterministic and independent of the grid size and of the order of rows within an expert.

Warp roles: 0-3 epilogue (thread = TMEM lane = stored weight row), 4 MMA, 5 weight TMA, 6 activation cp.async.
"""

from __future__ import annotations

import dataclasses
import functools
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
import torch
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr

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
SMEM_CAPACITY = 334848
TILE_NS = (8, 16, 32)


def _pow2_at_least(n: int, lo: int) -> int:
    p = lo
    while p < n:
        p *= 2
    return p


def _tcgen05_fence_after_thread_sync():
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)


def _tcgen05_fence_before_thread_sync():
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)


class Fc2Kernel:
    def __init__(
        self, tile_n: int, num_local: int, num_stages: int, use_pdl: bool, paired_store: bool = True
    ):
        if tile_n not in TILE_NS:
            raise ValueError(f"tile_n must be one of {TILE_NS}, got {tile_n}")
        self.tile_n = tile_n
        self.num_local = num_local
        self.num_ab_stage = num_stages
        self.num_acc_stage = 2
        self.use_pdl = use_pdl
        self.paired_store = paired_store
        self.mma_tiler = (TILE_M, tile_n, K_STAGE)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.a_dtype = cutlass.Float4E2M1FN
        self.b_dtype = cutlass.Float8E4M3FN
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.acc_dtype = cutlass.Float32
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.load_warp_id = 6
        self.threads_per_cta = 32 * 7
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * 4)
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * 5)
        self.b_bytes_per_stage = tile_n * K_STAGE
        self.sf_atom_bytes = 512
        self.num_acc_tmem_cols = tile_n * self.num_acc_stage

    @staticmethod
    def stage_bytes(tile_n: int) -> int:
        # Packed FP4 A + FP8 B + one 512 B scale-factor atom per 128 K for each of SFA and SFB (SFB rows round up
        # to the 128-row atom).
        return TILE_M * K_STAGE // 2 + tile_n * K_STAGE + 2 * (K_STAGE // 128) * 512

    @staticmethod
    def max_stages(tile_n: int, ctas_per_sm: int = 1) -> int:
        reserve = 8 * 1024
        return min((SMEM_CAPACITY // ctas_per_sm - reserve) // Fc2Kernel.stage_bytes(tile_n), 16)

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

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            SF_VEC,
            self.cta_group,
            (TILE_M, self.tile_n, MMA_K),
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
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
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
        cluster_layout_vmnk: cute.Layout,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.Layout,
        sfb_smem_staged: cute.Layout,
    ):
        tile_n = self.tile_n
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()

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

        pipeline.pipeline_init_wait(cluster_shape_mn=(1, 1))

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        num_units = num_tiles[0] * M_TILES

        if warp_idx == self.tma_warp_id:
            ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            for unit in cutlass.range(bidx, num_units, grid_x, unroll=1):
                tile = unit // M_TILES
                m_tile = unit - tile * M_TILES
                expert = tile_expert[tile]
                tAgA_unit = tAgA[(None, m_tile, None, expert)]
                tAgSFA_unit = tAgSFA[(None, m_tile, None, expert)]
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

        # Predicated-off cp.async copies zero-fill their destination: rows past the expert's end reach the MMA as 0.
        if warp_idx == self.load_warp_id:
            lane = tidx % 32
            chunk = lane % 8
            row_in_pass = lane // 8
            n_pass = tile_n // 4
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
            load_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            for unit in cutlass.range(bidx, num_units, grid_x, unroll=1):
                tile = unit // M_TILES
                row_base = tile * tile_n
                mn_limit = tile_limit[tile]
                for i in cutlass.range_constexpr(n_pass):
                    row = row_base + i * 4 + row_in_pass
                    ok = row < mn_limit
                    row_ok[i] = ok
                    row_src[i] = row * cutlass.Int32(ok)
                sf_row = row_base + lane
                sf_ok = (lane < tile_n) & (sf_row < mn_limit)
                sf_src = sf_row * cutlass.Int32(sf_ok)
                for k_tile in cutlass.range(K_TILES, unroll=1):
                    load_pipeline.producer_acquire(load_state)
                    sB_stage = sB.iterator + load_state.index * self.b_bytes_per_stage
                    sSFB_stage = sSFB.iterator + load_state.index * (n_katom * self.sf_atom_bytes)
                    for kt in cutlass.range_constexpr(n_katom):
                        k0 = k_tile * K_STAGE + kt * 128
                        for i in cutlass.range_constexpr(n_pass):
                            row = i * 4 + row_in_pass
                            src = cute.make_tensor(
                                act.iterator
                                + cute.assume(row_src[i] * INTER + k0 + chunk * 16, divby=16),
                                cute.make_layout((16,)),
                            )
                            # sB.iterator applies the 128B swizzle to this logical (K atom, row, chunk) offset.
                            dst = cute.make_tensor(
                                sB_stage + (kt * tile_n * 128 + row * 128 + chunk * 16),
                                cute.make_layout((16,)),
                            )
                            pred[0] = row_ok[i]
                            cute.copy_atom_call(b_copy, src, dst, pred=pred)
                        # Scale-factor atom (128 rows x 4 K blocks, 512 B): row r < 32 at byte 16 r.
                        sf_g = cute.make_tensor(
                            act_sf.iterator
                            + cute.assume(
                                sf_src * (INTER // SF_VEC) + k_tile * (K_STAGE // SF_VEC) + kt * 4,
                                divby=4,
                            ),
                            cute.make_layout((4,)),
                        )
                        sf_s = cute.make_tensor(
                            sSFB_stage + (kt * self.sf_atom_bytes + lane * 16),
                            cute.make_layout((4,)),
                        )
                        pred[0] = sf_ok
                        cute.copy_atom_call(sf_copy, sf_g, sf_s, pred=pred)
                    load_pipeline.producer_commit(load_state)
                    load_state.advance()
            load_pipeline.producer_tail(load_state)

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
            for unit in cutlass.range(bidx, num_units, grid_x, unroll=1):
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
            epi_tile = (TILE_M, tile_n)
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
                y.iterator, cute.make_layout((TILE_M, tile_n), stride=(tile_n, 1))
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
            for unit in cutlass.range(bidx, num_units, grid_x, unroll=1):
                tile = unit // M_TILES
                m_tile = unit - tile * M_TILES
                row_base = tile * tile_n
                mn_limit = tile_limit[tile]
                channel = m_tile * TILE_M + channel_in_tile
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
                    for c in cutlass.range_constexpr(0, tile_n, 2):
                        recv = cute.arch.shuffle_sync_bfly(
                            cutlass.select_(is_odd, acc[c], acc[c + 1]), offset=8
                        )
                        row = row_base + c + odd_channel
                        if row < mn_limit:
                            pair[0] = cutlass.select_(is_odd, recv, acc[c]).to(cutlass.BFloat16)
                            pair[1] = cutlass.select_(is_odd, acc[c + 1], recv).to(cutlass.BFloat16)
                            dst = cute.make_tensor(
                                y.iterator + cute.assume(row * HIDDEN + pair_channel, divby=2),
                                cute.make_layout((2,)),
                            )
                            cute.copy(store_pair, pair, dst)
                else:
                    for c in cutlass.range_constexpr(tile_n):
                        if row_base + c < mn_limit:
                            y[(row_base + c, channel)] = acc[c].to(cutlass.BFloat16)

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

        cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _ctas_per_sm() -> int:
    return int(os.environ.get("TLLM_K3_MOE_FC2_CTAS_PER_SM", "1"))


def _num_stages(tile_n: int) -> int:
    override = os.environ.get("TLLM_K3_MOE_FC2_STAGES")
    return int(override) if override else Fc2Kernel.max_stages(tile_n, _ctas_per_sm())


def _ptr(dtype, tensor: torch.Tensor, align: int):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


@functools.cache
def _compiled(tile_n: int, num_local: int, num_stages: int, use_pdl: bool, paired_store: bool):
    kernel = Fc2Kernel(tile_n, num_local, num_stages, use_pdl, paired_store)
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
        cutlass.Int32(tile_n),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cuda.CUstream(0),
    )
    return cute.compile(kernel, *args)


def run(act: torch.Tensor, act_sf: torch.Tensor, weights, plan, y: torch.Tensor) -> None:
    """``y[p] = bf16(dequant(act[p]) @ dequant(W2[e]).T)`` for every valid permuted row (moe_chain FC2 stage)."""
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
    device = act.device
    use_pdl = os.environ.get("TLLM_K3_MOE_FC2_PDL", "0") == "1"
    paired_store = os.environ.get("TLLM_K3_MOE_FC2_PAIRED_STORE", "1") == "1"
    compiled = _compiled(tile_n, num_local, _num_stages(tile_n), use_pdl, paired_store)
    grid = max(1, min(_ctas_per_sm() * _sm_count(device.index), tile_slots * M_TILES))
    compiled(
        _ptr(cutlass.Float4E2M1FN, w2, 32),
        _ptr(cutlass.Float8E8M0FNU, w2_sf, 16),
        _ptr(cutlass.Float8E4M3FN, act, 16),
        _ptr(cutlass.Float8E8M0FNU, act_sf, 16),
        _ptr(cutlass.BFloat16, y, 16),
        _ptr(cutlass.Int32, plan.tile_idx_to_expert_idx, 4),
        _ptr(cutlass.Int32, plan.tile_idx_to_mn_limit, 4),
        _ptr(cutlass.Int32, plan.num_non_exiting_tiles, 4),
        cutlass.Int32(rows),
        cutlass.Int32(tile_slots),
        cutlass.Int32(grid),
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
