# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 prefill MoE FC2 (SM107): token-M block-scaled grouped GEMM, MXFP8 intermediate x MXFP4 down weights,
bf16 output in permuted order or finalize fused.

The MMA M operand is a 128-row tile of one expert's permuted intermediate rows (``act`` e4m3 [P, 3072] + ``act_sf``
in the 128x4 scale-factor atom layout, both by TMA), the N operand a ``tile_n``-row tile of the expert's W2 (TMA, in
place in the TRTLLM-Gen layout: rows shuffled in 32-row blocks, stored row 32 b + j = output channel 32 b + 4 (j % 8)
+ j // 8; scales in the same row order, 128x4 atom). ``kind::mxf8f6f4``, K = 64 per instruction, 256 per stage. Work
items are (m-tile, n-tile), n-tile fastest, dealt round-robin over a persistent grid.

A TMEM lane is a token row: each epilogue thread holds a whole row and undoes the 32-row shuffle in registers, so a
32-column block is 32 consecutive output channels (64 B, four 16 B stores). Without ``fin`` every valid row p is
stored as ``y[p] = bf16(acc)``. With ``fin = (out, topk_w)`` a row whose token has exactly one local (row, k) pair is
stored as ``out[token] = bf16(f32(bf16(acc)) * topk_w[token, k])`` (the stock finalize's rounding) and other rows
go to ``y`` for a finalize pass over multi-pair rows (decode ``finalize.run(..., skip_single=True)``). Padding rows
(at or past the tile's ``tile_idx_to_mn_limit``) are not written.

The per-K-stage activation load is ``_load_act_stage``. ``sync="lamport"`` is the Lamport FC1 -> FC2 edge: no
grid wait at entry; act and act_sf are loaded with TMA ``report_valid`` onto a v1 mbarrier per stage (sentinel 0x80
checked per 16 B chunk for ``sentinel="80"``, 0xFF per element for ``"ff"``; act_sf always 0xFF per element since
four threads write each 16 B chunk of its atom layout); a stage that landed with a sentinel waits for the FC1 grid
once and is re-loaded plainly on a redo barrier. ``force_slow`` takes the slow path on every stage (test only).
"""

from __future__ import annotations

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
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from .common import (
    HIDDEN,
    INTER,
    SF_VEC,
    SMEM_ARCH,
    TOP_K,
    UnalignedBarrier,
    bits_f32,
    check_tensor,
    fence_proxy_async_global,
    gmem_ptr,
    ld_acquire_gpu_b32,
    num_sms,
    red_release_gpu_add,
    s2t_copy_and_partition,
    st_global_v4,
    tcgen05_fence,
    tmem_load_32x32b_x32,
    use_pdl,
)

M_TILE = 128


@dsl_user_op
def bf16x2(hi, lo, *, loc=None, ip=None):
    """Two fp32 -> bf16 (RN), packed as (hi << 16) | lo."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class Fc2TokenM:
    """Persistent warp-specialized token-M FC2. Warps: epilogue 0-3 (thread = TMEM lane = token row), MMA 4, TMA 5."""

    def __init__(
        self,
        tile_n: int,
        use_pdl: bool,
        fin: bool,
        pdl_trigger: str = "end",
        k_blocks: int = 8,
        num_acc_stages: int = 2,
        max_ab_stages: int = 3,
        smem_budget: Optional[int] = None,
        sync: Optional[str] = None,
        sentinel: str = "80",
        force_slow: bool = False,
        publish: bool = False,
    ):
        if sync not in (None, "lamport", "counter") or sentinel not in ("80", "ff"):
            raise ValueError(
                f"sync must be None, lamport or counter, sentinel 80 or ff, got {sync} / {sentinel}"
            )
        if tile_n not in (128, 256):
            raise ValueError(f"tile_n must be 128 or 256, got {tile_n}")
        if pdl_trigger not in ("end", "start", "none"):
            raise ValueError(f"pdl_trigger must be end, start or none, got {pdl_trigger}")
        if k_blocks not in (4, 8, 16):
            raise ValueError("k_blocks must be 4, 8 or 16")
        self.tile_n = tile_n
        self.num_n_tiles = HIDDEN // tile_n
        self.use_pdl = use_pdl
        self.fin = fin
        self.lamport = sync == "lamport"
        self.counter = sync == "counter"
        self.publish = publish
        self.sentinel = sentinel
        self.force_slow = force_slow
        self.pdl_trigger = pdl_trigger
        self.k_blocks = k_blocks
        self.k_stage = SF_VEC * k_blocks
        self.num_k_tiles = INTER // self.k_stage
        self.num_acc_stage = num_acc_stages
        self.max_ab_stages = max_ab_stages
        self.sf_vec_size = SF_VEC
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_warp = 32
        self.threads_per_cta = 32 * 6
        self.num_epilog_threads = 32 * len(self.epilog_warp_id)
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.num_epilog_threads
        )
        self.tmem_alloc_barrier = UnalignedBarrier(
            barrier_id=3, num_threads=32 * (1 + len(self.epilog_warp_id))
        )
        capacity = utils.get_smem_capacity_in_bytes(SMEM_ARCH)
        self.smem_capacity = capacity if smem_budget is None else min(capacity, smem_budget)

    def _make_tiled_mma(self):
        return sm107_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            self.sf_vec_size,
            self.cta_group,
            (M_TILE, self.tile_n, 64),
        )

    def _setup_attributes(self):
        tiled_mma = self._make_tiled_mma()
        self.mma_tiler = (M_TILE, self.tile_n, self.k_stage)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        a_one = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, cutlass.Float4E2M1FN, 1)
        sfa_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        sfb_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        stage_bytes = (
            cute.size_in_bytes(cutlass.Float8E4M3FN, a_one)
            + cute.size_in_bytes(cutlass.Float4E2M1FN, b_one)
            + cute.size_in_bytes(cutlass.Float8E8M0FNU, sfa_one)
            + cute.size_in_bytes(cutlass.Float8E8M0FNU, sfb_one)
        )
        reserved = 1024 + 4 * 1024 + 1024
        self.num_ab_stage = min((self.smem_capacity - reserved) // stage_bytes, self.max_ab_stages)
        if self.num_ab_stage < 2:
            raise ValueError("not enough shared memory for two mainloop stages")
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.Float4E2M1FN, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
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
        self.num_acc_tmem_cols = self.tile_n * self.num_acc_stage
        total = self.num_acc_tmem_cols + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        max_cols = cute.arch.get_max_tmem_alloc_cols(SMEM_ARCH)
        if total > max_cols:
            raise ValueError(f"TMEM: {total} columns needed, {max_cols} available")
        alloc = 32
        while alloc < total:
            alloc *= 2
        self.num_tmem_alloc_cols = min(alloc, max_cols)

    @cute.jit
    def __call__(
        self,
        act_ptr: cute.Pointer,
        act_sf_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        w_sf_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        topk_w_ptr: cute.Pointer,
        p2e_ptr: cute.Pointer,
        e2p_ptr: cute.Pointer,
        t2e_ptr: cute.Pointer,
        t2l_ptr: cute.Pointer,
        ntiles_ptr: cute.Pointer,
        ctr_ptr: cute.Pointer,
        pub_ptr: cute.Pointer,
        num_experts: cutlass.Int32,
        num_rows: cutlass.Int32,
        num_permuted: cutlass.Int32,
        num_tiles_cap: cutlass.Int32,
        num_ctas: cutlass.Int32,
        counter_target: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        ctr = cute.make_tensor(ctr_ptr, cute.make_layout((num_tiles_cap,)))
        pub = cute.make_tensor(pub_ptr, cute.make_layout((num_tiles_cap,)))
        a = cute.make_tensor(
            act_ptr, cute.make_layout((num_permuted, INTER, 1), stride=(INTER, 1, 0))
        )
        sfa = cute.make_tensor(
            act_sf_ptr,
            blockscaled_utils.tile_atom_to_shape_SF((num_permuted, INTER, 1), self.sf_vec_size),
        )
        # Weights [E, H, I/2] bytes as E2M1 (H, I, E), K-major.
        b = cute.make_tensor(
            w_ptr, cute.make_layout((HIDDEN, INTER, num_experts), stride=(INTER, 1, HIDDEN * INTER))
        )
        sfb = cute.make_tensor(
            w_sf_ptr,
            blockscaled_utils.tile_atom_to_shape_SF((HIDDEN, INTER, num_experts), self.sf_vec_size),
        )
        y = cute.make_tensor(y_ptr, cute.make_layout((num_permuted, HIDDEN), stride=(HIDDEN, 1)))
        out = cute.make_tensor(out_ptr, cute.make_layout((num_rows, HIDDEN), stride=(HIDDEN, 1)))
        topk_w = cute.make_tensor(topk_w_ptr, cute.make_layout((num_rows * TOP_K,)))
        p2e = cute.make_tensor(p2e_ptr, cute.make_layout((num_permuted,)))
        e2p = cute.make_tensor(e2p_ptr, cute.make_layout((num_rows * TOP_K,)))
        t2e = cute.make_tensor(t2e_ptr, cute.make_layout((num_tiles_cap,)))
        t2l = cute.make_tensor(t2l_ptr, cute.make_layout((num_tiles_cap,)))
        ntiles = cute.make_tensor(ntiles_ptr, cute.make_layout((1,)))

        tiled_mma = self._make_tiled_mma()
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        self.num_act_load_bytes = cute.size_in_bytes(
            cutlass.Float8E4M3FN, a_smem_layout
        ) + cute.size_in_bytes(cutlass.Float8E8M0FNU, sfa_smem_layout)
        self.num_b_load_bytes = cute.size_in_bytes(
            cutlass.Float4E2M1FN, b_smem_layout
        ) + cute.size_in_bytes(cutlass.Float8E8M0FNU, sfb_smem_layout)
        self.num_tma_load_bytes = self.num_b_load_bytes + (
            0 if self.lamport else self.num_act_load_bytes
        )

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            act_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage + 1]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_holding_buf: cutlass.Int32
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
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E4M3FN, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float4E2M1FN, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                1024,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfb,
            tma_tensor_sfb,
            y,
            out,
            topk_w,
            p2e,
            e2p,
            t2e,
            t2l,
            ntiles,
            ctr,
            pub,
            counter_target,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    def _load_act_stage(
        self, tma_atom_a, tAgA_k, tAsA_s, tma_atom_sfa, tAgSFA_k, tAsSFA_s, tma_bar, check=False
    ):
        if cutlass.const_expr(check):
            act_pattern = (
                cpasync.SentinelPattern.PER_ELEMENT_FF
                if self.sentinel == "ff"
                else cpasync.SentinelPattern.PER_16BYTES_80
            )
            cute.copy(tma_atom_a, tAgA_k, tAsA_s, tma_bar_ptr=tma_bar, report_valid=act_pattern)
            cute.copy(
                tma_atom_sfa,
                tAgSFA_k,
                tAsSFA_s,
                tma_bar_ptr=tma_bar,
                report_valid=cpasync.SentinelPattern.PER_ELEMENT_FF,
            )
        else:
            cute.copy(tma_atom_a, tAgA_k, tAsA_s, tma_bar_ptr=tma_bar)
            cute.copy(tma_atom_sfa, tAgSFA_k, tAsSFA_s, tma_bar_ptr=tma_bar)

    @cute.jit
    def _stage_invalid(self, bar, phase):
        """Wait for a v1 barrier's primary phase; 1 when a report_valid load on it saw a sentinel (DSL 4.9's blocking
        ``mbarrier_wait_primary`` never returns on SM107, hence the try-wait loop)."""
        done = cutlass.Int32(0)
        invalid = cutlass.Int32(0)
        while done == 0:
            res = cute.arch.mbarrier_try_wait_primary(bar, phase)
            done = cutlass.select_(res[0], cutlass.Int32(1), cutlass.Int32(0))
            invalid = cutlass.select_(res[1], cutlass.Int32(1), cutlass.Int32(0))
        return invalid

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mY: cute.Tensor,
        mOut: cute.Tensor,
        mTopkW: cute.Tensor,
        mP2e: cute.Tensor,
        mE2p: cute.Tensor,
        mT2e: cute.Tensor,
        mT2l: cute.Tensor,
        mNumTiles: cute.Tensor,
        mCtr: cute.Tensor,
        mPub: cute.Tensor,
        counter_target: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tile_n = self.tile_n
        n_tiles = self.num_n_tiles
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
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
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            arch=SMEM_ARCH,
        )
        tmem.allocate(self.num_tmem_alloc_cols)
        act_mbar = storage.act_mbar.data_ptr()
        redo_bar = act_mbar + self.num_ab_stage
        if cutlass.const_expr(self.lamport):
            if warp_idx == self.tma_warp_id:
                with cute.arch.elect_one():
                    for st in cutlass.range_constexpr(self.num_ab_stage):
                        cute.arch.mbarrier_v1_init(act_mbar + st, 1)
                    cute.arch.mbarrier_init(redo_bar, 1)
                cute.arch.mbarrier_init_fence()
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        thr_mma = tiled_mma.get_slice(0)
        # (bM, bK, loopM, loopK, loopL) / (bN, bK, loopN, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFB = thr_mma.partition_B(gSFB_nkl)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
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
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3)
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            0,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Every read of FC1's output comes after this wait; with the Lamport edge the data loads themselves are the
        # wait (the plan was complete before FC1 started), and only an invalid stage waits for the FC1 grid.
        if cutlass.const_expr(self.use_pdl and not (self.lamport or self.counter)):
            cute.arch.griddepcontrol_wait()
        self.cta_sync_barrier.arrive_and_wait()
        if cutlass.const_expr(self.use_pdl and self.pdl_trigger == "start"):
            cute.arch.griddepcontrol_launch_dependents()
        total_items = mNumTiles[0] * n_tiles

        if warp_idx == self.tma_warp_id:
            ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            for item in cutlass.range(bidx, total_items, gdim, unroll=1):
                m_tile = item // n_tiles
                n_tile = item - m_tile * n_tiles
                expert = mT2e[m_tile]
                tAgA_t = tAgA[(None, m_tile, None, 0)]
                tAgSFA_t = tAgSFA[(None, m_tile, None, 0)]
                tBgB_t = tBgB[(None, n_tile, None, expert)]
                tBgSFB_t = tBgSFB[(None, n_tile, None, expert)]
                if cutlass.const_expr(self.counter):
                    # Counter link: FC1's release-adds for this m-tile, then the proxy fence before the TMA reads.
                    ctr_addr = mCtr.iterator.toint() + cutlass.Int64(m_tile) * 4
                    while ld_acquire_gpu_b32(ctr_addr) < counter_target:
                        pass
                    fence_proxy_async_global()
                for k_tile in cutlass.range(self.num_k_tiles, unroll=1):
                    ab_pipeline.producer_acquire(ab_state)
                    tma_bar = ab_pipeline.producer_get_barrier(ab_state)
                    s = ab_state.index
                    cute.copy(
                        tma_atom_b, tBgB_t[(None, k_tile)], tBsB[(None, s)], tma_bar_ptr=tma_bar
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_t[(None, k_tile)],
                        tBsSFB[(None, s)],
                        tma_bar_ptr=tma_bar,
                    )
                    act_bar = tma_bar
                    if cutlass.const_expr(self.lamport):
                        act_bar = act_mbar + s
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                act_bar, self.num_act_load_bytes
                            )
                    self._load_act_stage(
                        tma_atom_a,
                        tAgA_t[(None, k_tile)],
                        tAsA[(None, s)],
                        tma_atom_sfa,
                        tAgSFA_t[(None, k_tile)],
                        tAsSFA[(None, s)],
                        act_bar,
                        check=self.lamport,
                    )
                    ab_state.advance()
            ab_pipeline.producer_tail(ab_state)

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
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            copy_s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = s2t_copy_and_partition(
                self.cta_group, sSFA, tCtSFA
            )
            copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = s2t_copy_and_partition(
                self.cta_group, sSFB, tCtSFB
            )
            ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            num_kblocks = cute.size(tCrA, mode=[2])
            waited = cute.make_rmem_tensor((2,), cutlass.Int32)
            waited[0] = cutlass.Int32(0)
            waited[1] = cutlass.Int32(0)
            for item in cutlass.range(bidx, total_items, gdim, unroll=1):
                acc_pipeline.producer_acquire(acc_state)
                tcgen05_fence("after")
                tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                m_tile = item // n_tiles
                tAgA_t = tAgA[(None, m_tile, None, 0)]
                tAgSFA_t = tAgSFA[(None, m_tile, None, 0)]
                for k_tile in cutlass.range(self.num_k_tiles, unroll=1):
                    ab_pipeline.consumer_wait(ab_state)
                    s = ab_state.index
                    if cutlass.const_expr(self.lamport):
                        invalid = self._stage_invalid(act_mbar + s, ab_state.phase)
                        if cutlass.const_expr(self.force_slow):
                            invalid = cutlass.Int32(1)
                        # FC2 ran ahead of FC1 on this stage (rare): after the FC1 grid every store is visible, so
                        # the stage is loaded once more, plainly, on the redo barrier.
                        if invalid != 0:
                            if waited[0] == 0:
                                cute.arch.griddepcontrol_wait()
                                waited[0] = cutlass.Int32(1)
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    redo_bar, self.num_act_load_bytes
                                )
                            self._load_act_stage(
                                tma_atom_a,
                                tAgA_t[(None, k_tile)],
                                tAsA[(None, s)],
                                tma_atom_sfa,
                                tAgSFA_t[(None, k_tile)],
                                tAsSFA[(None, s)],
                                redo_bar,
                            )
                            cute.arch.mbarrier_wait(redo_bar, waited[1])
                            waited[1] = waited[1] ^ 1
                    cute.copy(copy_s2t_sfa, tCsSFA_s2t[(None, None, None, None, s)], tCtSFA_s2t)
                    cute.copy(copy_s2t_sfb, tCsSFB_s2t[(None, None, None, None, s)], tCtSFB_s2t)
                    for kblock in cutlass.range_constexpr(num_kblocks):
                        sf_coord = (None, None, kblock)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_coord].iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[(None, None, kblock, s)],
                            tCrB[(None, None, kblock, s)],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_pipeline.consumer_release(ab_state)
                    ab_state.advance()
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
            acc_pipeline.producer_tail(acc_state)

        if warp_idx <= self.epilog_warp_id[-1]:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            quad = warp_idx
            lane = tidx & 31
            tmem_quad = tmem_ptr.toint() + ((32 * quad) << 16)
            y_base = mY.iterator.toint()
            out_base = mOut.iterator.toint()
            n_blocks = tile_n // 32
            for item in cutlass.range(bidx, total_items, gdim, unroll=1):
                m_tile = item // n_tiles
                n_tile = item - m_tile * n_tiles
                limit = mT2l[m_tile]
                prow = m_tile * M_TILE + 32 * quad + lane
                valid = prow < limit
                dst = y_base + cutlass.Int64(prow) * (HIDDEN * 2)
                scale = cutlass.Float32(1.0)
                if cutlass.const_expr(self.fin):
                    expanded = cutlass.max(
                        mP2e[cutlass.min(prow, mP2e.shape[0] - 1)], cutlass.Int32(0)
                    )
                    token = expanded // TOP_K
                    npairs = cutlass.Int32(0)
                    for k in cutlass.range_constexpr(TOP_K):
                        npairs = npairs + cutlass.select_(
                            mE2p[token * TOP_K + k] >= 0, cutlass.Int32(1), cutlass.Int32(0)
                        )
                    single = npairs == 1
                    w = bits_f32(cutlass.Int32(mTopkW[expanded]) << 16)
                    dst = cutlass.select_(
                        single, out_base + cutlass.Int64(token) * (HIDDEN * 2), dst
                    )
                    scale = cutlass.select_(single, w, cutlass.Float32(1.0))
                acc_pipeline.consumer_wait(acc_state)
                tcgen05_fence("after")
                tmem_stage = tmem_quad + acc_state.index * tile_n
                for j in cutlass.range_constexpr(n_blocks):
                    cols = tmem_load_32x32b_x32([tmem_stage + 32 * j])
                    if cutlass.const_expr(j == n_blocks - 1):
                        tcgen05_fence("before")
                        acc_pipeline.consumer_release(acc_state)
                        acc_state.advance()
                    # Stored W2 row j of a 32-row block is channel 4 (j % 8) + j // 8: channel q sits at column
                    # 8 (q % 4) + q // 4, so channels q, q + 1 (q even) are columns c, c + 8.
                    words = []
                    for q in cutlass.range_constexpr(0, 32, 2):
                        c = 8 * (q % 4) + q // 4
                        if cutlass.const_expr(self.fin):
                            packed = bf16x2(cols[c + 8], cols[c])
                            lo = bits_f32(packed << 16) * scale
                            hi = bits_f32(packed & cutlass.Int32(-65536)) * scale
                            words.append(bf16x2(hi, lo))
                        else:
                            words.append(bf16x2(cols[c + 8], cols[c]))
                    if valid:
                        addr = dst + cutlass.Int64(n_tile * (tile_n * 2) + j * 64)
                        for v in cutlass.range_constexpr(4):
                            st_global_v4(
                                addr + 16 * v,
                                words[4 * v],
                                words[4 * v + 1],
                                words[4 * v + 2],
                                words[4 * v + 3],
                            )
                if cutlass.const_expr(self.publish):
                    # Once all four epilogue warps stored this (m, n) item, one release-add on the m-tile's word;
                    # the m-tile's out / y rows are complete when it reaches the number of n-tiles.
                    self.epilog_sync_barrier.arrive_and_wait()
                    if tidx == 0:
                        red_release_gpu_add(
                            mPub.iterator.toint() + cutlass.Int64(m_tile) * 4, cutlass.Int32(1)
                        )
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
        if cutlass.const_expr(self.use_pdl and self.pdl_trigger == "end"):
            cute.arch.griddepcontrol_launch_dependents()


_compiled = {}


def num_ctas_for(act: torch.Tensor, plan, tile_n: int) -> int:
    return min(num_sms(act.device.index), plan.tile_idx_to_expert_idx.shape[0] * (HIDDEN // tile_n))


def check(act, act_sf, weights, plan, y, fin):
    num_experts = weights.w2_weight.shape[0]
    num_permuted = plan.permuted_idx_to_expanded_idx.shape[0]
    num_tiles_cap = plan.tile_idx_to_expert_idx.shape[0]
    if plan.tile_n != M_TILE:
        raise ValueError(
            f"the plan must pad experts to {M_TILE} rows (plan.tile_n = {plan.tile_n})"
        )
    if num_tiles_cap * M_TILE > num_permuted or num_permuted % M_TILE:
        raise ValueError("plan: permuted rows must be a multiple of 128 covering every tile")
    check_tensor("act", act, torch.float8_e4m3fn, (num_permuted, INTER), 16)
    check_tensor("act_sf", act_sf, torch.uint8, (num_permuted, INTER // SF_VEC), 128)
    check_tensor(
        "w2_weight", weights.w2_weight, torch.uint8, (num_experts, HIDDEN, INTER // 2), 128
    )
    check_tensor(
        "w2_weight_scale",
        weights.w2_weight_scale,
        torch.uint8,
        (num_experts, HIDDEN, INTER // SF_VEC),
        128,
    )
    check_tensor("y", y, torch.bfloat16, (num_permuted, HIDDEN), 16)
    for name in (
        "permuted_idx_to_expanded_idx",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
    ):
        t = getattr(plan, name)
        if t.dtype != torch.int32 or not t.is_contiguous():
            raise ValueError(f"plan.{name} must be a contiguous int32 tensor")
    if fin is not None:
        out, topk_w = fin
        rows = topk_w.shape[0]
        check_tensor("topk_w", topk_w, torch.bfloat16, (rows, TOP_K), 2)
        check_tensor("out", out, torch.bfloat16, (rows, HIDDEN), 16)
        e2p = plan.expanded_idx_to_permuted_idx
        if e2p.dtype != torch.int32 or not e2p.is_contiguous() or e2p.shape[0] < rows * TOP_K:
            raise ValueError(
                "plan.expanded_idx_to_permuted_idx must be int32 with rows * 16 entries"
            )


def launch_args(act, act_sf, weights, plan, y, fin, num_ctas, counters=None, publish=None):
    out, topk_w = fin if fin is not None else (y, y)
    rows = topk_w.shape[0] if fin is not None else 1
    return (
        gmem_ptr(cutlass.Float8E4M3FN, act, 16),
        gmem_ptr(cutlass.Float8E8M0FNU, act_sf, 128),
        gmem_ptr(cutlass.Float4E2M1FN, weights.w2_weight, 128),
        gmem_ptr(cutlass.Float8E8M0FNU, weights.w2_weight_scale, 128),
        gmem_ptr(cutlass.BFloat16, y, 16),
        gmem_ptr(cutlass.BFloat16, out, 16),
        gmem_ptr(cutlass.Uint16, topk_w, 2),
        gmem_ptr(cutlass.Int32, plan.permuted_idx_to_expanded_idx, 4),
        gmem_ptr(cutlass.Int32, plan.expanded_idx_to_permuted_idx, 4),
        gmem_ptr(cutlass.Int32, plan.tile_idx_to_expert_idx, 4),
        gmem_ptr(cutlass.Int32, plan.tile_idx_to_mn_limit, 4),
        gmem_ptr(cutlass.Int32, plan.num_non_exiting_tiles, 4),
        gmem_ptr(
            cutlass.Int32, plan.tile_idx_to_expert_idx if counters is None else counters[0], 4
        ),
        gmem_ptr(cutlass.Int32, plan.tile_idx_to_expert_idx if publish is None else publish, 4),
        cutlass.Int32(weights.w2_weight.shape[0]),
        cutlass.Int32(rows),
        cutlass.Int32(act.shape[0]),
        cutlass.Int32(plan.tile_idx_to_expert_idx.shape[0]),
        cutlass.Int32(num_ctas),
        cutlass.Int32(0 if counters is None else counters[1]),
    )


def run(
    act,
    act_sf,
    weights,
    plan,
    y,
    *,
    fin=None,
    tile_n: int = 256,
    pdl_trigger: str = "end",
    k_blocks: int = 8,
    max_ab_stages: int = 3,
    smem_budget: Optional[int] = None,
    num_acc_stages: int = 2,
    enable_pdl: Optional[bool] = None,
    num_ctas: Optional[int] = None,
    sync: Optional[str] = None,
    sentinel: str = "80",
    force_slow: bool = False,
    counters=None,
    publish=None,
):
    """y[p] = bf16(dequant(act[p]) @ W2[e]^T) for every valid permuted row p of expert e (``act`` e4m3 [P, 3072],
    ``act_sf`` uint8 [P, 96] in the 128x4 atom layout, ``weights.w2_weight`` uint8 [E, 3584, 1536] +
    ``w2_weight_scale`` uint8 [E, 3584, 96] in the TRTLLM-Gen layout, ``plan`` with tile_n = 128, ``y`` bf16
    [P, 3584]). ``fin = (out bf16 [R, 3584], topk_w bf16 [R, 16])``: rows of tokens with exactly one local pair are
    written scaled into ``out[token]`` instead of ``y`` (see the module docstring). One kernel on the current stream.
    ``publish``: int32 [T] device words (zeroed by the caller); publish[m] is release-added once per (m, n) item after
    the item's out / y stores, so m-tile m's rows are complete when publish[m] == 3584 // tile_n (14 for tile_n 256).
    ``counters=(ctr int32 [T], target)``: with ``sync="counter"`` the loads of m-tile m wait for ctr[m] >= target
    (acquire) and a proxy fence instead of the grid wait."""
    check(act, act_sf, weights, plan, y, fin)
    pdl = use_pdl() if enable_pdl is None else enable_pdl
    key = (
        tile_n,
        pdl,
        fin is not None,
        pdl_trigger,
        k_blocks,
        num_acc_stages,
        max_ab_stages,
        smem_budget,
        sync,
        sentinel,
        force_slow,
        publish is not None,
    )
    if publish is not None and (
        publish.dtype != torch.int32 or publish.numel() < plan.tile_idx_to_expert_idx.shape[0]
    ):
        raise ValueError("publish must be int32 with one word per tile")
    if sync in ("lamport", "counter") and not pdl:
        raise ValueError("sync=lamport/counter needs PDL")
    if (sync == "counter") != (counters is not None):
        raise ValueError("sync=counter needs counters=(ctr int32 [T], target per m-tile)")
    ctas = num_ctas_for(act, plan, tile_n) if num_ctas is None else num_ctas
    args = launch_args(act, act_sf, weights, plan, y, fin, ctas, counters, publish)
    stream = cuda.CUstream(torch.cuda.current_stream(act.device).cuda_stream)
    compiled = _compiled.get(key)
    if compiled is None:
        kernel = Fc2TokenM(
            tile_n,
            pdl,
            fin is not None,
            pdl_trigger=pdl_trigger,
            k_blocks=k_blocks,
            num_acc_stages=num_acc_stages,
            max_ab_stages=max_ab_stages,
            smem_budget=smem_budget,
            sync=sync,
            sentinel=sentinel,
            force_slow=force_slow,
            publish=publish is not None,
        )
        compiled = _compiled[key] = cute.compile(kernel, *args, stream)
    compiled(*args, stream)
