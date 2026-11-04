# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fine-grained-sync build of the SM107 BF16 persistent GEMM / BMM (``kimi_k3_cute_gemm`` ``rubin`` backend).

Same tiles, pipelines and per-tile arithmetic as ``kernels.rubin.dense_bf16_gemm_persistent`` at split-K 1 (the
kernel body is the Blackwell one it inherits, without split-K and FP8 quantization), on the original persistent grid.
Compile-time roles, all off in the stock build:

Producer (the output feeds a report_valid consumer):

- ``pzo``: the accumulator is narrowed with ``cvt.rn.pzo`` (-0.0 stored as +0.0, every other value bit-identical),
  so a consumer can use bf16 -0.0 as the not-yet-written sentinel. Tiles leave through TMA stores, whose 16 B chunks
  land whole.
- ``early_trigger``: ``griddepcontrol.launch_dependents`` at kernel entry instead of after the epilogue.
- ``raster_along="n"`` (scheduler, with ``swizzle_size``): consecutive tiles walk N first, so output row blocks
  complete in row order instead of all in the last tile round.
- ``n_order=(heads, h1)``: the N tiles are four sections (q | k | v | gate) of ``heads`` one-tile heads, and the
  scheduler's N index t is mapped to the tile it computes and stores as: the q, k, v tiles of heads 0 .. h1 - 1,
  head-major, then the gate tiles of those heads, then all four tiles of each remaining head. Each tile's math is
  unchanged; the first ``3 * h1`` tiles (the first persistent round when the grid holds them) complete whole heads.

Consumer (A is a sentinel-armed edge slot; B, the weights, is never checked):

- ``retry_warp``: full barriers are v1 (``PipelineTmaUmmaRetry``); the MMA warp waits on the conditional phase and an
  extra warp releases each stage once its primary phase completes with valid data. Without ``report_valid`` this is
  the like-resource control.
- ``report_valid``: the A TMA checks the bf16 -0.0 sentinel per 16 B. A stage whose load completes with sentinel data
  is re-armed for the A bytes and A alone is re-loaded into it (B stays). 2-CTA tiles: the leader forwards the
  re-issue to its peer through DSMEM. A multicast across N: the CTAs of the multicast agree on each re-issue.
- ``wait_input=False``: no ``griddepcontrol.wait``, so A and B loads start at launch and B lands before A is valid.
- ``store_wait``: with ``wait_input=False``, the epilogue warps wait for the PDL predecessor before their first
  store (an output buffer the predecessor may still read).
- ``retry_poll``: ``"try"`` suspends on each stage's primary phase (bounded by the hardware time limit), ``"test"``
  polls it.

Diagnostic: ``timeline`` stores ``%globaltimer_lo`` when each CTA starts and when each CTA tile's last TMA store is
issued (see :func:`timeline_slots`); ``retry_count[0]`` accumulates the A re-issues of report_valid builds.

The kernel never writes sentinels; the framework arms the A slot before the producer runs.
"""

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

try:
    from cutlass._mlir.dialects.nvvm import MBarrierScopeKind
    from cutlass.experimental.primitives.nvvm_wrapper import MemOrder, fence_sync_restrict
except ImportError:  # CuTe DSL < 4.9: the consumer roles cannot be built.
    MBarrierScopeKind = MemOrder = fence_sync_restrict = None

# kimi_k3_cute_gemm table entries.
O_PROJ_TACTIC = ("base", True, (256, 256), (2, 1), 0, 1)
QKVG_DECODE_TACTIC = ("base", False, (128, 128), (1, 1), 0, 1)
# Upper bound on persistent CTAs recorded by the timeline build.
TIMELINE_MAX_CTAS = 1024
# int32 elements of the retry counter: A re-issues.
RETRY_COUNTER_SLOTS = 1


class _Stage:
    """The ``state.index`` view pipeline calls take, for a stage tracked outside a pipeline state."""

    def __init__(self, index):
        self.index = index


class FgsDenseGemmKernel(PersistentDenseGemmKernel):
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
        report_valid: bool = False,
        retry_warp: bool = False,
        wait_input: bool = True,
        store_wait: bool = False,
        retry_poll: Literal["try", "test"] = "try",
        n_order: Optional[Tuple[int, int]] = None,
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
        if report_valid and not retry_warp:
            raise ValueError("report_valid needs retry_warp to re-issue invalid stages")
        if retry_warp and MBarrierScopeKind is None:
            raise ValueError(
                "the consumer roles need CuTe DSL 4.9 (v1 mbarriers, TMA report_valid)"
            )
        if retry_poll not in ("try", "test"):
            raise ValueError(f"retry_poll must be 'try' or 'test', got {retry_poll!r}")
        self.pzo = pzo
        self.early_trigger = early_trigger
        self.use_pdl = use_pdl
        self.timeline = timeline
        self.report_valid = report_valid
        self.retry_warp = retry_warp
        self.wait_input = wait_input or not use_pdl
        self.store_wait = store_wait and not self.wait_input
        self.retry_poll = retry_poll
        self.n_order = n_order
        self.a_copy_kwargs = {"report_valid": True} if report_valid else {}
        self.retry_warp_id = 6
        if retry_warp:
            self.threads_per_cta += 32

    def _shared_storage(self):
        stages = self.num_ab_stage
        acc_stages = self.num_acc_stage
        if not self.retry_warp:

            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, stages * 2]
                acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32

            return SharedStorage

        # Peer re-issue message: stage, k tile, m tile, n tile, l.
        coord_buf = cute.struct.MemRange[cutlass.Int32, 5]

        @cute.struct
        class RetrySharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, stages * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
            retry_cluster_mbar_ptr: cute.struct.MemRange[cutlass.Int64, stages]
            peer_retry_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_dealloc_mbar: cutlass.Int64
            peer_retry_coord_buf: cute.struct.Align[coord_buf, 16]
            tmem_holding_buf: cutlass.Int32

        return RetrySharedStorage, coord_buf

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stamps: cute.Tensor,
        retry_count: cute.Tensor,
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
        if cutlass.const_expr(self.report_valid and self.use_2cta_instrs and self.is_a_mcast):
            raise NotImplementedError("report_valid with 2-CTA tiles and A multicast")
        if cutlass.const_expr(self.retry_warp and self.num_ab_stage > 30):
            raise NotImplementedError(
                f"the retry warp tracks <= 30 stages in bitmasks, got {self.num_ab_stage}"
            )

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
        self.num_tma_load_bytes_a = a_copy_size * atom_thr_size

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
            retry_count,
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
        retry_count: cute.Tensor,
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
        need_peer_retry = cutlass.const_expr(self.report_valid and self.use_2cta_instrs)
        need_retry_sync = cutlass.const_expr(self.report_valid and self.is_a_mcast)

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

        coord_buf_type = None
        if cutlass.const_expr(self.retry_warp):
            SharedStorage, coord_buf_type = self._shared_storage()
        else:
            SharedStorage = self._shared_storage()
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline = None
        full_mbar_ptr = None
        empty_mbar_ptr = None
        local_coord_ptr = None
        if cutlass.const_expr(self.retry_warp):
            # The MMA warp waits on the conditional phase; the retry warp tests the primary phase and releases.
            ab_pipeline = pipeline.PipelineTmaUmmaRetry.create(
                barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Warp),
                consumer_group_retry=pipeline.CooperativeGroup(pipeline.Agent.Warp),
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                enable_multicast_signaling=True,
                defer_sync=True,
            )
            ab_producer, ab_consumer, _ = ab_pipeline.make_participants()
            if cutlass.const_expr(need_peer_retry):
                # Ping-pong pair guarding the re-issue message the leader writes into the peer's smem.
                full_mbar_ptr = storage.peer_retry_mbar_ptr.data_ptr()
                empty_mbar_ptr = full_mbar_ptr + 1
                local_coord_ptr = storage.peer_retry_coord_buf.data_ptr()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_init(full_mbar_ptr, 1)
                        cute.arch.mbarrier_init(empty_mbar_ptr, 32)
        else:
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

        # A multicast: every CTA of the multicast re-arms a stage before any of them re-issues its A slice into it.
        retry_cluster_mbar = None
        retry_dst_rank = None
        retry_is_signalling = None
        if cutlass.const_expr(need_retry_sync):
            retry_cluster_mbar = pipeline.MbarrierArray(
                barrier_storage=storage.retry_cluster_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                agent=(
                    pipeline.PipelineOp.AsyncThread,
                    pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mcast_ctas_a - 1),
                ),
            )
            # A is multicast across N: the CTAs sharing this CTA's M coordinate.
            retry_dst_rank, retry_is_signalling = (
                pipeline.PipelineTmaAsync.init_empty_barrier_arrive_signal(
                    cute.make_layout(cluster_layout_vmnk.shape), tidx, (1, 0)
                )
            )
            retry_is_signalling = retry_is_signalling & (retry_dst_rank != cta_rank_in_cluster)

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

        if cutlass.const_expr(self.use_pdl and self.wait_input):
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
                tBgB_slice = tBgB[
                    (None, self._n_tile(mma_tile_coord_mnl[1]), None, mma_tile_coord_mnl[2])
                ]

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
                        **self.a_copy_kwargs,
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

        if cutlass.const_expr(self.retry_warp):
            # 2-CTA loads complete only on the leader's barrier, so the leader alone validates and releases.
            if warp_idx == self.retry_warp_id and is_leader_cta:
                self._retry_leader(
                    ab_pipeline,
                    tile_sched_params,
                    tiled_mma,
                    tma_atom_a,
                    tAgA,
                    tAsA,
                    a_full_mcast_mask,
                    k_tile_cnt,
                    retry_count,
                    retry_cluster_mbar,
                    retry_dst_rank,
                    retry_is_signalling,
                    full_mbar_ptr,
                    empty_mbar_ptr,
                    local_coord_ptr,
                    coord_buf_type,
                    cta_rank_in_cluster,
                )
            if cutlass.const_expr(need_peer_retry):
                if warp_idx == self.retry_warp_id and not is_leader_cta:
                    self._retry_peer(
                        ab_pipeline,
                        tma_atom_a,
                        tAgA,
                        tAsA,
                        a_full_mcast_mask,
                        full_mbar_ptr,
                        empty_mbar_ptr,
                        local_coord_ptr,
                        cta_rank_in_cluster,
                    )

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

            if cutlass.const_expr(self.store_wait):
                cute.arch.griddepcontrol_wait()

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
                n_tile = self._n_tile(cur_tile_coord[1])
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    n_tile,
                    cur_tile_coord[2],
                )
                tile_stamp_idx = cur_tile_coord[0] * tiles_n + n_tile
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
    def _n_tile(self, t):
        """The N tile the scheduler's N index ``t`` computes (``n_order``)."""
        n = t
        if cutlass.const_expr(self.n_order is not None):
            heads, h1 = self.n_order
            if t < 3 * h1:
                head = t // 3
                n = (t - 3 * head) * heads + head
            else:
                u = t - 3 * h1
                if u < h1:
                    n = 3 * heads + u
                else:
                    w = u - h1
                    q = w // 4
                    n = (w - 4 * q) * heads + h1 + q
        return n

    @cute.jit
    def _retry_leader(
        self,
        ab_pipeline,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        tAgA: cute.Tensor,
        tAsA: cute.Tensor,
        a_full_mcast_mask,
        k_tile_cnt: cutlass.Int32,
        retry_count: cute.Tensor,
        retry_cluster_mbar,
        retry_dst_rank,
        retry_is_signalling,
        full_mbar_ptr,
        empty_mbar_ptr,
        local_coord_ptr,
        coord_buf_type: cutlass.Constexpr,
        cta_rank_in_cluster,
    ):
        """Release each A/B stage of every work tile once its primary phase completes with valid data.

        The k tiles of the current work tile in flight lie in [base, base + stages): the producer refills a stage only
        after it is released, and the tile's k tile t lives in stage (first stage + t) % stages. Bit s of ``done`` marks
        stage s's tile in that window as released, bit s of ``parity`` is the primary-phase parity stage s completes
        next. Invalid stages are re-armed, re-loaded and revisited on the next sweep, so several can retry at once.
        """
        need_peer_retry = cutlass.const_expr(self.report_valid and self.use_2cta_instrs)
        need_retry_sync = cutlass.const_expr(self.report_valid and self.is_a_mcast)
        stages = self.num_ab_stage

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        peer_cta_rank = cta_rank_in_cluster ^ 1
        reissues = cutlass.Int32(0)
        base_stage = cutlass.Int32(0)
        parity = cutlass.Int32(0)
        # A multicast: per stage, arrived on the peers' retry barriers this round / their phase parity.
        mc_arrived = cutlass.Int32(0)
        mc_parity = cutlass.Int32(0)
        empty_phase = cutlass.Int32(1)
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )
            tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]

            base = cutlass.Int32(0)
            done = cutlass.Int32(0)
            while base < k_tile_cnt:
                window = k_tile_cnt - base
                if window > stages:
                    window = cutlass.Int32(stages)
                stage = base_stage
                for i in cutlass.range(window, unroll=1):
                    bit = cutlass.Int32(1) << stage
                    if (done & bit) == 0:
                        barrier = ab_pipeline.sync_object_full.get_barrier(stage)
                        if cutlass.const_expr(self.retry_poll == "test"):
                            complete, invalid = cute.arch.mbarrier_test_wait_primary(
                                barrier, (parity >> stage) & 1
                            )
                        else:
                            complete, invalid = cute.arch.mbarrier_try_wait_primary(
                                barrier, (parity >> stage) & 1
                            )
                        if complete:
                            if invalid:
                                do_reissue = cutlass.Boolean(True)
                                if cutlass.const_expr(need_retry_sync):
                                    if (mc_arrived & bit) == 0:
                                        if retry_is_signalling:
                                            cute.arch.mbarrier_arrive(
                                                retry_cluster_mbar.get_barrier(stage),
                                                retry_dst_rank,
                                            )
                                        mc_arrived = mc_arrived | bit
                                    do_reissue = retry_cluster_mbar.test_wait(
                                        stage, (mc_parity >> stage) & 1
                                    )
                                if do_reissue:
                                    parity = parity ^ bit
                                    if cutlass.const_expr(need_retry_sync):
                                        mc_parity = mc_parity ^ bit
                                        mc_arrived = mc_arrived ^ bit
                                    if cutlass.const_expr(need_peer_retry):
                                        cute.arch.mbarrier_wait(empty_mbar_ptr, empty_phase)
                                        empty_phase ^= 1
                                        with cute.arch.elect_one():
                                            cute.arch.mbarrier_arrive_and_expect_tx(
                                                full_mbar_ptr,
                                                coord_buf_type.size_in_bytes,
                                                peer_cta_rank,
                                                relaxed=True,
                                                scope=MBarrierScopeKind.CLUSTER,
                                            )
                                            cute.arch.store_async_dsmem(
                                                local_coord_ptr,
                                                (
                                                    stage,
                                                    base + i,
                                                    mma_tile_coord_mnl[0],
                                                    mma_tile_coord_mnl[1],
                                                ),
                                                full_mbar_ptr,
                                                peer_cta_rank,
                                            )
                                            cute.arch.store_async_dsmem(
                                                local_coord_ptr + 4,
                                                mma_tile_coord_mnl[2],
                                                full_mbar_ptr,
                                                peer_cta_rank,
                                            )
                                    ab_pipeline.retry_expect_tx(
                                        _Stage(stage), self.num_tma_load_bytes_a
                                    )
                                    cute.copy(
                                        tma_atom_a,
                                        tAgA_slice[(None, base + i)],
                                        tAsA[(None, stage)],
                                        tma_bar_ptr=barrier,
                                        mcast_mask=a_full_mcast_mask,
                                        report_valid=True,
                                    )
                                    reissues += 1
                            else:
                                parity = parity ^ bit
                                done = done | bit
                                ab_pipeline.retry_release(_Stage(stage))
                    stage = stage + 1
                    if stage == stages:
                        stage = cutlass.Int32(0)
                while ((done >> base_stage) & 1) == 1:
                    done = done ^ (cutlass.Int32(1) << base_stage)
                    base = base + 1
                    base_stage = base_stage + 1
                    if base_stage == stages:
                        base_stage = cutlass.Int32(0)

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        if cutlass.const_expr(need_peer_retry):
            # Stage -1 releases the peer from its loop; then drain its release of that last message.
            cute.arch.mbarrier_wait(empty_mbar_ptr, empty_phase)
            empty_phase ^= 1
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_mbar_ptr,
                    cutlass.Int32.width // 8,
                    peer_cta_rank,
                    relaxed=True,
                    scope=MBarrierScopeKind.CLUSTER,
                )
                cute.arch.store_async_dsmem(local_coord_ptr, -1, full_mbar_ptr, peer_cta_rank)
            cute.arch.mbarrier_wait(empty_mbar_ptr, empty_phase)

        if cutlass.const_expr(self.report_valid):
            with cute.arch.elect_one():
                if reissues > 0:
                    cute.arch.atomic_add(retry_count.iterator, reissues)

    @cute.jit
    def _retry_peer(
        self,
        ab_pipeline,
        tma_atom_a: cute.CopyAtom,
        tAgA: cute.Tensor,
        tAsA: cute.Tensor,
        a_full_mcast_mask,
        full_mbar_ptr,
        empty_mbar_ptr,
        local_coord_ptr,
        cta_rank_in_cluster,
    ):
        """Re-issue this CTA's half of a 2-CTA A load whenever the leader forwards one (stage -1 ends the loop)."""
        peer_cta_rank = cta_rank_in_cluster ^ 1
        full_phase = cutlass.Int32(0)
        retry_stage_idx = cutlass.Int32(0)
        while retry_stage_idx != -1:
            cute.arch.mbarrier_wait(full_mbar_ptr, full_phase)
            full_phase ^= 1
            retry_stage_idx = local_coord_ptr[0]
            retry_k_tile = local_coord_ptr[1]
            retry_coord_m = local_coord_ptr[2]
            retry_coord_l = local_coord_ptr[4]

            # Order the message reads before handing the buffer back to the leader.
            fence_sync_restrict(MemOrder.RELEASE)
            cute.arch.mbarrier_arrive(
                empty_mbar_ptr,
                peer_cta_rank,
                scope=MBarrierScopeKind.CLUSTER,
                relaxed=True,
            )

            if retry_stage_idx != -1:
                retry_gA_slice = tAgA[(None, retry_coord_m, None, retry_coord_l)]
                cute.copy(
                    tma_atom_a,
                    retry_gA_slice[(None, retry_k_tile)],
                    tAsA[(None, retry_stage_idx)],
                    tma_bar_ptr=ab_pipeline.sync_object_full.get_barrier(retry_stage_idx),
                    mcast_mask=a_full_mcast_mask,
                    report_valid=True,
                )

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
        retry_count: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((m, k, batch_size), order=(1, 0, 2))
        )
        b_tensor = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, batch_size), order=(1, 0, 2))
        )
        self(a_tensor, b_tensor, c_tensor, stamps, retry_count, max_active_clusters, stream)

    @cute.jit
    def wrapper_strided_fgs(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        batch_size: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        a_stride_m: cutlass.Int32,
        a_stride_batch: cutlass.Int32,
        b_stride_n: cutlass.Int32,
        b_stride_batch: cutlass.Int32,
        stamps: cute.Tensor,
        retry_count: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """``wrapper_fgs`` with A and B (M / N, batch) strides; K is innermost."""
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_layout((m, k, batch_size), stride=(a_stride_m, 1, a_stride_batch)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_layout((n, k, batch_size), stride=(b_stride_n, 1, b_stride_batch)),
        )
        self(a_tensor, b_tensor, c_tensor, stamps, retry_count, max_active_clusters, stream)


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
_SCRATCH = {}


def _scratch(device: torch.device, name: str, numel: int) -> torch.Tensor:
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, name)
    if key not in _SCRATCH:
        _SCRATCH[key] = torch.zeros(numel, dtype=torch.int32, device=device)
    return _SCRATCH[key]


def _check_n_order(n_order: Optional[Tuple[int, int]], n: int, mma_mn: Tuple[int, int]) -> None:
    if n_order is None:
        return
    heads, h1 = n_order
    if n != 4 * heads * mma_mn[1] or not 0 < h1 <= heads:
        raise ValueError(
            f"n_order {n_order} needs N = 4 x heads x tile N ({mma_mn[1]}) and 0 < h1 <= heads, got N={n}"
        )


def _parse_tactic(tactic: tuple):
    variant, use_2cta, mma_mn, cluster_mn, max_stage = tactic[:5]
    split_k = tactic[5] if len(tactic) > 5 else 1
    if variant != "base" or split_k != 1:
        raise ValueError(f"the FGS GEMM takes a base split-K 1 tactic, got {tactic}")
    return use_2cta, tuple(mma_mn), tuple(cluster_mn), max_stage


def run_gemm_fgs(
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    tactic: tuple,
    *,
    report_valid: bool = True,
    retry_warp: bool = True,
    wait_input: bool = False,
    store_wait: bool = False,
    retry_poll: Literal["try", "test"] = "try",
    pzo: bool = False,
    early_trigger: bool = False,
    raster_along: Literal["m", "n"] = "m",
    swizzle_size: int = 1,
    use_pdl: bool = True,
    retry_counter: Optional[torch.Tensor] = None,
    max_ctas: Optional[int] = None,
    n_order: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """``out[L, M, N] = a[L, M, K] @ w[L, N, K]^T`` in BF16 at an in-tree ``base`` tactic.

    Any operand may be a strided view with K (``a``, ``w``) / N (``out``) innermost and 16 B aligned rows, as for the
    in-tree batched runner. Defaults are the report_valid consumer; ``retry_counter`` (int32[``RETRY_COUNTER_SLOTS``])
    accumulates the A re-issues. ``max_ctas`` caps the persistent grid (default: the device's max active clusters).
    ``n_order`` (heads, h1) is the kernel's head-ordered tile mapping.
    """
    if a.dim() != 3 or w.dim() != 3 or out.dim() != 3:
        raise ValueError("run_gemm_fgs takes [L, M, K], [L, N, K], [L, M, N] operands")
    batch, m, k = a.shape
    n = w.shape[1]
    if tuple(w.shape) != (batch, n, k) or tuple(out.shape) != (batch, m, n):
        raise ValueError(f"shapes a {tuple(a.shape)} w {tuple(w.shape)} out {tuple(out.shape)}")
    if any(t.dtype != torch.bfloat16 for t in (a, w, out)):
        raise ValueError("run_gemm_fgs takes bf16 operands")
    if (k > 1 and (a.stride(2) != 1 or w.stride(2) != 1)) or (n > 1 and out.stride(2) != 1):
        raise ValueError("K must be innermost in a and w, N innermost in out")
    use_2cta, mma_mn, cluster_mn, max_stage = _parse_tactic(tactic)
    _check_n_order(n_order, n, mma_mn)
    if retry_counter is None:
        retry_counter = _scratch(a.device, "retry", RETRY_COUNTER_SLOTS)
    elif retry_counter.dtype != torch.int32 or retry_counter.numel() != RETRY_COUNTER_SLOTS:
        raise ValueError(f"retry_counter must be int32[{RETRY_COUNTER_SLOTS}]")

    device_index = a.get_device()
    cluster_size = cluster_mn[0] * cluster_mn[1]
    clusters = _max_active_clusters(device_index, cluster_size)
    if max_ctas is not None:
        clusters = max(1, min(clusters, max_ctas // cluster_size))
    a_ptr = make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(cutlass.BFloat16, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_mnl = out.permute(1, 2, 0)
    c_tensor = from_dlpack(c_mnl).mark_layout_dynamic(leading_dim=1)
    stamps_tensor = from_dlpack(
        _scratch(a.device, "stamps", TIMELINE_MAX_CTAS)
    ).mark_layout_dynamic()
    retry_tensor = from_dlpack(retry_counter, assumed_align=4)
    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    strides = (a.stride(1), a.stride(0), w.stride(1), w.stride(0))
    flags = dict(
        pzo=pzo,
        early_trigger=early_trigger,
        use_pdl=use_pdl,
        report_valid=report_valid,
        retry_warp=retry_warp,
        wait_input=wait_input,
        store_wait=store_wait,
        retry_poll=retry_poll,
        n_order=n_order,
    )
    key = (
        "strided",
        device_index,
        use_2cta,
        mma_mn,
        cluster_mn,
        max_stage,
        raster_along,
        swizzle_size,
        tuple(sorted(flags.items())),
        tuple(c_mnl.stride()),
        clusters,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        gemm = FgsDenseGemmKernel(
            cutlass.Float32,
            use_2cta,
            mma_mn,
            cluster_mn,
            swizzle_size=swizzle_size,
            raster_along=raster_along,
            max_num_ab_stage=max_stage,
            **flags,
        )
        compiled = cute.compile(
            gemm.wrapper_strided_fgs,
            m,
            n,
            k,
            batch,
            a_ptr,
            b_ptr,
            c_tensor,
            *strides,
            stamps_tensor,
            retry_tensor,
            clusters,
            stream,
            options="--opt-level 2",
        )
        _COMPILED[key] = compiled
    compiled(
        m, n, k, batch, a_ptr, b_ptr, c_tensor, *strides, stamps_tensor, retry_tensor, stream=stream
    )
    return out


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
    n_order: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """``out[M, N] = a[M, K] @ w[N, K]^T`` in BF16 with the producer roles; all three row-major and dense.

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
    use_2cta, mma_mn, cluster_mn, max_stage = _parse_tactic(tactic)
    _check_n_order(n_order, n, mma_mn)
    timeline = stamps is not None
    if timeline and (stamps.dtype != torch.int32 or stamps.numel() < timeline_slots(m, n, tactic)):
        raise ValueError(f"stamps must be int32 with >= {timeline_slots(m, n, tactic)} elements")
    if stamps is None:
        stamps = _scratch(a.device, "stamps", TIMELINE_MAX_CTAS)

    device_index = a.get_device()
    clusters = _max_active_clusters(device_index, cluster_mn[0] * cluster_mn[1])
    a_ptr = make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(cutlass.BFloat16, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_tensor = from_dlpack(out.unsqueeze(-1)).mark_layout_dynamic(leading_dim=1)
    stamps_tensor = from_dlpack(stamps).mark_layout_dynamic()
    retry_tensor = from_dlpack(_scratch(a.device, "retry", RETRY_COUNTER_SLOTS), assumed_align=4)
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
        n_order,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        gemm = FgsDenseGemmKernel(
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
            n_order=n_order,
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
            retry_tensor,
            clusters,
            stream,
            options="--opt-level 2",
        )
        _COMPILED[key] = compiled
    compiled(m, n, k, 1, a_ptr, b_ptr, c_tensor, stamps_tensor, retry_tensor, stream=stream)
    return out
