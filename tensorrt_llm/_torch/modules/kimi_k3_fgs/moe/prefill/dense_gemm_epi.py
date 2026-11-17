# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The kernel body follows tensorrt_llm/_torch/kernels/blackwell/dense_gemm_persistent.py (split-K 1), derived from the
# CUTLASS example examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py, BSD-3-Clause.
"""SM107 BF16 persistent dense GEMM with fused epilogues for the Kimi K3 prefill MoE (CuTe DSL).

``C[M, N] = A[M, K] @ B[N, K]^T`` with the in-tree Rubin persistent kernel's tiles, pipelines and per-tile arithmetic
(``kernels.rubin.dense_bf16_gemm_persistent`` at split-K 1), and an epilogue chosen at compile time:

- ``mxfp8``: the stock dispatch quantize (``torch.ops.trtllm.mxfp8_quantize(y, is_sf_swizzled_layout=False)``) on
  the bf16-rounded tile: per row and 32 columns ``sf = cvt.rp.satfinite.ue8m0(amax * rcp.approx.ftz(448))``, elements
  ``cvt.rn.satfinite.e4m3(y * 2^(127 - sf))`` (scale 0 for an all-zero block). C is the e4m3 output (TMA stored),
  ``sf`` the linear UE8M0 map ``[M, N / 32]``. Latent down 7168 -> 3584 fused with the quantize the dispatch sends.

Each epilogue thread owns one output row and ``epi_n`` consecutive columns of the subtile (tcgen05 32dp32b TMEM
loads, warp layout (4, 1)), so a 32-column block's amax is thread-local: one pass, no shared-memory atomics.
"""

from __future__ import annotations

import functools
from typing import Dict, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

try:
    from tensorrt_llm._torch.kernels.blackwell.custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
    from tensorrt_llm._torch.kernels.blackwell.utils import griddepcontrol_launch_dependents, griddepcontrol_wait
    from tensorrt_llm._torch.kernels.rubin.dense_bf16_gemm_persistent import PersistentDenseGemmKernel
except ImportError:  # the release image installs the kernels package as cute_dsl_kernels
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.dense_bf16_gemm_persistent import PersistentDenseGemmKernel

SF_VEC = 32
EPILOGUES = ("mxfp8",)
# kimi_k3_cute_gemm's prefill tactic for the 7168-wide projections.
DEFAULT_TACTIC = ("base", True, (256, 256), (2, 1), 0, 1)


@dsl_user_op
def _u32_as_f32(value, *, loc=None, ip=None):
    return cutlass.Float32(llvm.bitcast(T.f32(), cutlass.Uint32(value).ir_value(loc=loc, ip=ip)))


@dsl_user_op
def _rcp_approx_ftz(value, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _round_up_ue8m0(value, *, loc=None, ip=None):
    """Positive fp32 -> UE8M0 rounded toward +inf, saturating (the stock ``__nv_cvt_float_to_e8m0`` call)."""
    packed = llvm.inline_asm(
        T.i16(),
        [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        "cvt.rp.satfinite.ue8m0x2.f32 $0, 0f00000000, $1;",
        "=h,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [packed],
            "mov.b32 $0, { $1, 0 };",
            "=r,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class EpiDenseGemmKernel(PersistentDenseGemmKernel):
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        max_num_ab_stage: int = 0,
        epilogue: str = "mxfp8",
        early_trigger: bool = False,
        use_pdl: bool = True,
    ):
        super().__init__(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_tma_store=True,
            max_num_ab_stage=max_num_ab_stage,
            split_k_slices=1,
        )
        assert epilogue in EPILOGUES, epilogue
        self.epilogue = epilogue
        self.early_trigger = early_trigger
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sf: cute.Tensor,
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
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
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
            sf,
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
        mSf: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

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
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
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

        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        cC_mnl = cute.make_identity_tensor(mC_mnl.shape)
        cC_mnl = cute.local_tile(cC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCcC = thr_mma.partition_C(cC_mnl)

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
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        griddepcontrol_wait()
        if cutlass.const_expr(self.early_trigger):
            griddepcontrol_launch_dependents()

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
                self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, use_2cta_instrs
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)

            gC_mnl_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
            cC_mnl_epi = cute.flat_divide(tCcC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_cC = thr_copy_t2r.partition_D(cC_mnl_epi)
            tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)

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

            c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilogue_warp_id))
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )
            epilog_threads = 32 * len(self.epilogue_warp_id)
            # A thread's values of a subtile: one row, consecutive columns (32dp32b TMEM loads, warp layout (4, 1)).
            vals_per_thread = cute.size(tTR_rAcc.shape)
            assert vals_per_thread % SF_VEC == 0, (vals_per_thread, epi_tile)
            rcp448 = _rcp_approx_ftz(cutlass.Float32(448.0))

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                num_tiles_executed = tile_sched.num_tiles_executed

                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                acc_stage_index = acc_consumer_state.index
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                tTR_cC_tile = tTR_cC[(None, None, None, None, None, *mma_tile_coord_mnl)]
                tTR_cC_tile = cute.group_modes(tTR_cC_tile, 3, cute.rank(tTR_cC_tile))

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = (num_tiles_executed - 1) * subtile_cnt

                for subtile_idx in cutlass.range(subtile_cnt):
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    coords = tTR_cC_tile[(None, None, None, subtile_idx)]
                    acc_vec = tTR_rAcc.load()
                    quantized = cute.make_rmem_tensor(acc_vec.shape, self.c_dtype)
                    for g in cutlass.range_constexpr(vals_per_thread // SF_VEC):
                        rounded = cute.make_rmem_tensor(cute.make_layout((SF_VEC,)), cutlass.Float32)
                        amax = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(SF_VEC):
                            v = cutlass.Float32(acc_vec[g * SF_VEC + i].to(cutlass.BFloat16))
                            rounded[i] = v
                            amax = cute.arch.fmax(amax, cute.arch.fmax(v, -v))
                        sf_byte = _round_up_ue8m0(amax * rcp448)
                        # exp2f_rcp of the stock quantizer: 2^(127 - sf), 1 for sf 0; 0 for an all-zero block.
                        quant_exp = cutlass.Uint32(254) - sf_byte
                        if sf_byte == 0:
                            quant_exp = cutlass.Uint32(127)
                        scale = _u32_as_f32(quant_exp << 23)
                        if amax == 0.0:
                            scale = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(SF_VEC):
                            quantized[g * SF_VEC + i] = (rounded[i] * scale).to(self.c_dtype)
                        coord = coords[g * SF_VEC]
                        if cute.elem_less(coord, mC_mnl.shape):
                            mSf[(coord[0], coord[1] // SF_VEC)] = cutlass.Uint8(sf_byte)
                    tTR_rC.store(quantized.load())

                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads)
                    if warp_idx == self.epilogue_warp_id[0]:
                        cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)])
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads)

                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            c_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        if cutlass.const_expr(not self.early_trigger):
            griddepcontrol_launch_dependents()

    @cute.jit
    def wrapper(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        sf_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)))
        b_tensor = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, 1), order=(1, 0, 2)))
        self(a_tensor, b_tensor, c_tensor, sf_tensor, max_active_clusters, stream)


_compiled: Dict[tuple, object] = {}


@functools.lru_cache(maxsize=None)
def _max_active_clusters(device_index: int, cluster_size: int) -> int:
    return utils.HardwareInfo(device_id=device_index).get_max_active_clusters(cluster_size)


def _parse_tactic(tactic) -> tuple:
    kind, use_2cta, mma_tiler_mn, cluster_shape_mn, max_num_ab_stage, split_k = tactic
    if kind != "base" or split_k != 1:
        raise ValueError(f"dense_gemm_epi: tactic {tactic} must be a base split-K-1 tactic")
    return bool(use_2cta), tuple(mma_tiler_mn), tuple(cluster_shape_mn), int(max_num_ab_stage)


def gemm_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    q: torch.Tensor,
    sf: torch.Tensor,
    *,
    tactic: tuple = DEFAULT_TACTIC,
    early_trigger: bool = False,
    enable_pdl: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``q, sf = mxfp8_quantize(bf16(x @ weight^T))`` in one launch.

    ``x`` [M, K] bf16 and ``weight`` [N, K] bf16, both K-contiguous; ``q`` [M, N] e4m3 and ``sf`` [M, N / 32] uint8
    (the stock linear UE8M0 layout) are written in place; N % 32 == 0, K % 64 == 0, 16 B aligned rows. The bf16
    rounding of the GEMM result is the stock cuBLAS -> quantize numerics; the fp32 accumulation order is the CuTe
    kernel's (``kimi_k3_cute_gemm`` ``rubin`` backend at ``tactic``), so bytes match ``mxfp8_quantize`` applied to
    that kernel's bf16 output bit for bit, and cuBLAS's output up to accumulation-order rounding.
    """
    m, k = x.shape
    n = weight.shape[0]
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or weight.shape[1] != k:
        raise ValueError("gemm_mxfp8: x [M, K] and weight [N, K] must be bf16 with the same K")
    if q.dtype != torch.float8_e4m3fn or tuple(q.shape) != (m, n) or not q.is_contiguous():
        raise ValueError(f"gemm_mxfp8: q must be contiguous e4m3 [{m}, {n}]")
    if sf.dtype != torch.uint8 or tuple(sf.shape) != (m, n // SF_VEC) or not sf.is_contiguous():
        raise ValueError(f"gemm_mxfp8: sf must be contiguous uint8 [{m}, {n // SF_VEC}]")
    if n % SF_VEC or k % 64 or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("gemm_mxfp8: N % 32 == 0, K % 64 == 0, contiguous operands")
    if m == 0:
        return q, sf
    use_2cta, mma_tiler_mn, cluster_shape_mn, max_num_ab_stage = _parse_tactic(tactic)
    max_active_clusters = _max_active_clusters(x.device.index, cluster_shape_mn[0] * cluster_shape_mn[1])
    c_tmp = q.unsqueeze(-1)
    key = (use_2cta, mma_tiler_mn, cluster_shape_mn, max_num_ab_stage, early_trigger, enable_pdl,
           max_active_clusters, tuple(c_tmp.stride()), sf.stride())
    fn = _compiled.get(key)
    if fn is None:
        kernel = EpiDenseGemmKernel(
            cutlass.Float32,
            use_2cta,
            mma_tiler_mn,
            cluster_shape_mn,
            max_num_ab_stage=max_num_ab_stage,
            epilogue="mxfp8",
            early_trigger=early_trigger,
            use_pdl=enable_pdl,
        )
        a_ptr = make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        b_ptr = make_ptr(cutlass.BFloat16, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        c_cute = from_dlpack(c_tmp).mark_layout_dynamic(leading_dim=1)
        sf_cute = from_dlpack(sf).mark_layout_dynamic(leading_dim=1)
        fn = _compiled[key] = cute.compile(
            kernel.wrapper,
            m,
            n,
            k,
            a_ptr,
            b_ptr,
            c_cute,
            sf_cute,
            max_active_clusters,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    fn(m, n, k, x.data_ptr(), weight.data_ptr(), c_tmp, sf)
    return q, sf
