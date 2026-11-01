# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fine-grained-sync producer build of the SM107 BF16 persistent GEMM (``kimi_k3_cute_gemm`` ``rubin`` backend).

Same tiles, pipelines and per-tile arithmetic as ``kernels.rubin.dense_bf16_gemm_persistent`` at split-K 1 (the
kernel body is the Blackwell one it inherits, without split-K and FP8 quantization). Compile-time producer roles:

- ``pzo``: the accumulator is narrowed with ``cvt.rn.pzo`` (-0.0 stored as +0.0, every other value bit-identical),
  so a report_valid consumer can use bf16 -0.0 as the not-yet-written sentinel. Tiles still leave through TMA
  stores, whose 16 B chunks land whole.
- ``early_trigger``: ``griddepcontrol.launch_dependents`` at kernel entry instead of after the epilogue.
- ``raster_along="n"`` (scheduler, with ``swizzle_size``): consecutive tiles walk N first, so output row blocks
  complete in row order instead of all in the last tile round.
- ``timeline`` (diagnostic): ``%globaltimer_lo`` when each CTA starts and when each CTA tile's last TMA store is
  issued (see :func:`timeline_slots`).
"""

from __future__ import annotations

import functools
from typing import Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ...kernels.blackwell.custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from ...kernels.rubin.dense_bf16_gemm_persistent import PersistentDenseGemmKernel

# kimi_k3_cute_gemm table entry of the prefill o_proj (7168, 12288).
O_PROJ_TACTIC = ("base", True, (256, 256), (2, 1), 0, 1)
# Upper bound on persistent CTAs recorded by the timeline build.
TIMELINE_MAX_CTAS = 1024


class FgsProducerDenseGemmKernel(PersistentDenseGemmKernel):
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
        max_num_ab_stage: int = 0,
        pzo: bool = False,
        early_trigger: bool = False,
        use_pdl: bool = True,
        timeline: bool = False,
    ):
        super().__init__(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_tma_store=True,
            swizzle_size=swizzle_size,
            raster_along=raster_along,
            max_num_ab_stage=max_num_ab_stage,
            split_k_slices=1,
        )
        self.pzo = pzo
        self.early_trigger = early_trigger
        self.use_pdl = use_pdl
        self.timeline = timeline

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stamps: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )

        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
            max_active_clusters,
        )

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            stamps,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        stamps: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        if cutlass.const_expr(self.early_trigger and self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        num_tile_stamps = cute.size(stamps) - TIMELINE_MAX_CTAS
        cta_n = self.cta_tile_shape_mnk[1]
        tiles_n = (mC_mnl.shape[1] + cta_n - 1) // cta_n
        if cutlass.const_expr(self.timeline):
            if tidx == 0:
                gdx, gdy, _ = cute.arch.grid_dim()
                cta_linear = bidx + bidy * gdx + bidz * gdx * gdy
                if cta_linear < TIMELINE_MAX_CTAS:
                    stamps[num_tile_stamps + cta_linear] = cute.arch.globaltimer_lo()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_producer.tail()

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_crd = (None, None, kblock_idx, handle.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_crd], tCrB[kblock_crd], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        handle.release()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            copy_atom_t2r = utils.sm100.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                epi_tile,
                use_2cta_instrs,
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)

            gC_mnl_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
            )

            copy_atom_r2s = utils.sm100.get_smem_store_op(
                self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)

            sC_for_tma = cute.group_modes(sC, 0, 2)
            gC_for_tma = cute.group_modes(gC_mnl_epi, 0, 2)
            bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                tma_atom_c, 0, cute.make_layout(1), sC_for_tma, gC_for_tma
            )

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tile_stamp_idx = cur_tile_coord[0] * tiles_n + cur_tile_coord[1]
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                num_tiles_executed = tile_sched.num_tiles_executed

                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                acc_stage_index = acc_consumer_state.index
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = (num_tiles_executed - 1) * subtile_cnt

                epilog_threads = 32 * len(self.epilogue_warp_id)
                for subtile_idx in cutlass.range(subtile_cnt):
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    if cutlass.const_expr(self.pzo):
                        acc_vec = acc_vec.to(self.c_dtype, pzo=True)
                    else:
                        acc_vec = acc_vec.to(self.c_dtype)
                    tRS_rC.store(acc_vec)

                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])

                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                    if warp_idx == self.epilogue_warp_id[0]:
                        cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)])
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                if cutlass.const_expr(self.timeline):
                    if tidx == 0:
                        if tile_stamp_idx < num_tile_stamps:
                            stamps[tile_stamp_idx] = cute.arch.globaltimer_lo()

                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            c_pipeline.producer_tail()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        if cutlass.const_expr(not self.early_trigger and self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def wrapper_fgs(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        batch_size: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        stamps: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((m, k, batch_size), order=(1, 0, 2))
        )
        b_tensor = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, batch_size), order=(1, 0, 2))
        )
        self(a_tensor, b_tensor, c_tensor, stamps, max_active_clusters, stream)


def timeline_slots(m: int, n: int, tactic: tuple = O_PROJ_TACTIC) -> int:
    """int32 elements of a timeline buffer: one stamp per CTA tile (row-major over [M / cta_m, N / cta_n]),
    then one start stamp per persistent CTA (``TIMELINE_MAX_CTAS`` slots, unused ones stay as written)."""
    _, use_2cta, mma_mn, _, _, _ = tactic
    cta_m = mma_mn[0] // (2 if use_2cta else 1)
    return -(-m // cta_m) * -(-n // mma_mn[1]) + TIMELINE_MAX_CTAS


@functools.lru_cache(maxsize=None)
def _max_active_clusters(device_index: int, cluster_size: int) -> int:
    try:
        from ...custom_ops.cute_dsl_custom_ops import get_max_activate_clusters
    except ImportError:
        return utils.HardwareInfo().get_max_active_clusters(cluster_size)
    with torch.cuda.device(device_index):
        return get_max_activate_clusters(cluster_size)


_COMPILED = {}
_SCRATCH_STAMPS = {}


def _scratch_stamps(device: torch.device) -> torch.Tensor:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SCRATCH_STAMPS:
        _SCRATCH_STAMPS[index] = torch.zeros(TIMELINE_MAX_CTAS, dtype=torch.int32, device=device)
    return _SCRATCH_STAMPS[index]


def run_dense_fgs(
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    *,
    pzo: bool = True,
    early_trigger: bool = True,
    raster_along: Literal["m", "n"] = "n",
    swizzle_size: int = 1,
    use_pdl: bool = True,
    tactic: tuple = O_PROJ_TACTIC,
    stamps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``out[M, N] = a[M, K] @ w[N, K]^T`` in BF16; all three row-major and dense.

    ``stamps`` (int32, :func:`timeline_slots` elements) selects the timeline build.
    """
    m, k = a.shape
    n = w.shape[0]
    if a.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("run_dense_fgs takes bf16 operands")
    if not (a.is_contiguous() and w.is_contiguous() and out.is_contiguous()):
        raise ValueError("run_dense_fgs takes dense row-major operands")
    if tuple(out.shape) != (m, n) or w.shape[1] != k:
        raise ValueError(f"shapes a {tuple(a.shape)} w {tuple(w.shape)} out {tuple(out.shape)}")
    variant, use_2cta, mma_mn, cluster_mn, max_stage, split_k = tactic
    if variant != "base" or split_k != 1:
        raise ValueError(f"run_dense_fgs takes a base split-K 1 tactic, got {tactic}")
    timeline = stamps is not None
    if timeline and (stamps.dtype != torch.int32 or stamps.numel() < timeline_slots(m, n, tactic)):
        raise ValueError(f"stamps must be int32 with >= {timeline_slots(m, n, tactic)} elements")
    if stamps is None:
        stamps = _scratch_stamps(a.device)

    device_index = a.get_device()
    clusters = _max_active_clusters(device_index, cluster_mn[0] * cluster_mn[1])
    a_ptr = make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(cutlass.BFloat16, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_tensor = from_dlpack(out.unsqueeze(-1)).mark_layout_dynamic(leading_dim=1)
    stamps_tensor = from_dlpack(stamps).mark_layout_dynamic()
    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    key = (
        device_index,
        tactic,
        pzo,
        early_trigger,
        raster_along,
        swizzle_size,
        use_pdl,
        timeline,
        clusters,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        gemm = FgsProducerDenseGemmKernel(
            cutlass.Float32,
            use_2cta,
            mma_mn,
            cluster_mn,
            swizzle_size=swizzle_size,
            raster_along=raster_along,
            max_num_ab_stage=max_stage,
            pzo=pzo,
            early_trigger=early_trigger,
            use_pdl=use_pdl,
            timeline=timeline,
        )
        compiled = cute.compile(
            gemm.wrapper_fgs,
            m,
            n,
            k,
            1,
            a_ptr,
            b_ptr,
            c_tensor,
            stamps_tensor,
            clusters,
            stream,
            options="--opt-level 2",
        )
        _COMPILED[key] = compiled
    compiled(m, n, k, 1, a_ptr, b_ptr, c_tensor, stamps_tensor, stream=stream)
    return out
