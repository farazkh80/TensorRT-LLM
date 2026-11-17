# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 prefill MoE FC1 (SM107): token-M block-scaled grouped GEMM, gathered MXFP8 rows x MXFP4 gate_up weights,
SiTU-GLU, MXFP8 output.

The MMA M operand is a 128-row tile of one expert's permuted rows (gathered from the received activations by cp.async,
row ``permuted_idx_to_expanded_idx[p] // 16``), the N operand a ``tile_n``-row tile of the expert's gate_up weights
(TMA, read in place in the TRTLLM-Gen W4A8_MXFP4_MXFP8 layout: see decode ``fc1.py``). ``kind::mxf8f6f4`` with K = 64
per instruction, K = 256 per stage. Work items are (m-tile, n-tile), n-tile fastest, dealt round-robin over a
persistent grid; every warp walks the same static sequence.

Weight tile ``n_tile`` column ``n`` holds intermediate channel ``tile_n / 2 * n_tile + 16 (n // 32) + 2 (n % 8) +
(n % 32) // 16``, the up row when bit 3 of ``n`` is clear and the gate row otherwise. A TMEM lane is a token row, so
each epilogue thread holds gate and up of every channel of its row: a 64-column block is one 32-channel MXFP8 group
whose scale needs no cross-lane reduction, and each 32-column half-block is one 16 B store.

Epilogue numerics: SiTU-GLU in fp32, act = (gate_cap * tanh(g / gate_cap) * sigmoid(g)) * (linear_cap *
tanh(u / linear_cap)); ``math="accurate"`` uses the decode kernel's 2^-22 tanh / sigmoid, ``math="fast"`` uses
``tanh.approx.f32`` (sigmoid(g) = 0.5 + 0.5 tanh(g / 2)). Per 32 channels the UE8M0 scale is 2^ceil(log2(amax / 448))
(byte 0 when amax is 0), elements are scaled by its exact inverse and rounded to E4M3 (RN, saturating). Rows at or
past the tile's ``tile_idx_to_mn_limit`` are stored as zeros (act and scale).

``act_sf`` is written either linear ([P, 96], byte (p, kb) at p * 96 + kb, the decode contract) or in the 128x4
scale-factor atom layout the block-scaled MMA reads by TMA (``sf_layout="swizzled"``, the prefill FC2 input):
byte (p, kb) at (p // 128) * 12288 + (kb // 4) * 512 + (p % 32) * 16 + (p % 128 // 32) * 4 + kb % 4.
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
from cutlass.cute.nvgpu import cpasync, tcgen05

from ..sync import spin_ns
from .common import (
    HIDDEN,
    INTER,
    SF_VEC,
    SMEM_ARCH,
    TOP_K,
    W_ROWS,
    CpAsyncUmmaPipeline,
    UnalignedBarrier,
    check_tensor,
    e4m3x2,
    gmem_ptr,
    ld_global_cg_v4,
    num_sms,
    red_release_gpu_add,
    s2t_copy_and_partition,
    sigmoid_accurate,
    st_global_b16,
    st_global_b32,
    st_global_v4,
    st_shared_b32,
    tanh_accurate,
    tanh_approx,
    tcgen05_fence,
    tma_gather4,
    tmem_load_32x32b_x32,
    ue8m0_scale,
    use_pdl,
)

M_TILE = 128
SF_ATOM_BYTES = 512


class Fc1TokenM:
    """Persistent warp-specialized token-M FC1. Warps: epilogue 0-3 (thread = TMEM lane = token row), MMA 4, weight
    TMA 5, row gather 6 .. 6 + num_gather_warps - 1."""

    def __init__(
        self,
        tile_n: int,
        gate_cap: float,
        linear_cap: float,
        use_pdl: bool,
        pdl_trigger: str = "end",
        math: str = "fast",
        store: str = "v4",
        sf_layout: str = "swizzled",
        k_blocks: int = 8,
        num_acc_stages: int = 2,
        max_ab_stages: int = 3,
        smem_budget: Optional[int] = None,
        num_gather_warps: int = 8,
        diag: Optional[str] = None,
        a_path: str = "tma",
        sf_path: str = "ld",
        gather_issue: str = "elect",
        sentinel: Optional[str] = None,
        rearm: bool = False,
        hazard_delay_ns: int = 0,
        counters: bool = False,
    ):
        if sentinel not in (None, "80", "ff"):
            raise ValueError(f"sentinel must be None, 80 or ff, got {sentinel}")
        if gather_issue not in ("lanes", "elect"):
            raise ValueError(f"gather_issue must be lanes or elect, got {gather_issue}")
        if a_path not in ("tma", "cpasync") or sf_path not in ("ld", "cpasync"):
            raise ValueError(
                f"a_path must be tma or cpasync, sf_path ld or cpasync, got {a_path} / {sf_path}"
            )
        if sf_path == "ld" and a_path != "tma":
            raise ValueError("sf_path=ld needs a_path=tma (no cp.async arrivals on the A barrier)")
        if diag not in (None, "nogather", "noepi", "nomma", "nosf", "sfzero"):
            raise ValueError(
                f"diag must be None, nogather, noepi, nomma, nosf or sfzero, got {diag}"
            )
        if tile_n not in (128, 256):
            raise ValueError(f"tile_n must be 128 or 256, got {tile_n}")
        if pdl_trigger not in ("end", "start", "none"):
            raise ValueError(f"pdl_trigger must be end, start or none, got {pdl_trigger}")
        if math not in ("fast", "accurate"):
            raise ValueError(f"math must be fast or accurate, got {math}")
        if store not in ("v4", "u16"):
            raise ValueError(f"store must be v4 or u16, got {store}")
        if sf_layout not in ("linear", "swizzled"):
            raise ValueError(f"sf_layout must be linear or swizzled, got {sf_layout}")
        if k_blocks not in (4, 8, 16) or num_gather_warps not in (1, 2, 4, 8):
            raise ValueError("k_blocks must be 4, 8 or 16; num_gather_warps 1, 2, 4 or 8")
        self.tile_n = tile_n
        self.num_n_tiles = W_ROWS // tile_n
        self.gate_cap = float(gate_cap)
        self.linear_cap = float(linear_cap)
        self.use_pdl = use_pdl
        self.pdl_trigger = pdl_trigger
        self.math = math
        self.store = store
        self.sf_layout = sf_layout
        self.diag = diag
        self.a_path = a_path
        self.sf_path = sf_path
        self.gather_issue = gather_issue
        self.sentinel = sentinel
        self.rearm = rearm
        self.hazard_delay_ns = hazard_delay_ns
        self.counters = counters
        self.k_blocks = k_blocks
        self.k_stage = SF_VEC * k_blocks
        self.num_k_tiles = HIDDEN // self.k_stage
        self.num_acc_stage = num_acc_stages
        self.max_ab_stages = max_ab_stages
        self.num_gather_warps = num_gather_warps
        self.sf_vec_size = SF_VEC
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.gather_warp_id = 6
        self.rearm_warp_id = 6 + num_gather_warps
        self.threads_per_warp = 32
        self.threads_per_cta = 32 * (self.rearm_warp_id + (1 if rearm else 0))
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
        self.a_bytes_per_stage = cute.size_in_bytes(cutlass.Float8E4M3FN, a_one)
        self.sfa_bytes_per_stage = cute.size_in_bytes(cutlass.Float8E8M0FNU, sfa_one)
        stage_bytes = (
            self.a_bytes_per_stage
            + cute.size_in_bytes(cutlass.Float4E2M1FN, b_one)
            + self.sfa_bytes_per_stage
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
        rearm_act_ptr: cute.Pointer,
        rearm_sf_ptr: cute.Pointer,
        rearm_rows_ptr: cute.Pointer,
        ctr_ptr: cute.Pointer,
        rearm_ctr_ptr: cute.Pointer,
        num_experts: cutlass.Int32,
        num_rows: cutlass.Int32,
        num_permuted: cutlass.Int32,
        num_tiles_cap: cutlass.Int32,
        num_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        ctr = cute.make_tensor(ctr_ptr, cute.make_layout((num_tiles_cap,)))
        rearm_ctr = cute.make_tensor(rearm_ctr_ptr, cute.make_layout((num_tiles_cap,)))
        rearm_act = cute.make_tensor(rearm_act_ptr, cute.make_layout((1,)))
        rearm_sf = cute.make_tensor(rearm_sf_ptr, cute.make_layout((1,)))
        rearm_rows = cute.make_tensor(rearm_rows_ptr, cute.make_layout((1,)))
        # Weights [E, 2I, H/2] bytes as E2M1 (2I, H, E), K-major: the MMA N operand.
        b = cute.make_tensor(
            w_ptr,
            cute.make_layout((W_ROWS, HIDDEN, num_experts), stride=(HIDDEN, 1, W_ROWS * HIDDEN)),
        )
        sfb = cute.make_tensor(
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

        tiled_mma = self._make_tiled_mma()
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        self.num_tma_load_bytes = cute.size_in_bytes(
            cutlass.Float4E2M1FN, b_smem_layout
        ) + cute.size_in_bytes(cutlass.Float8E8M0FNU, sfb_smem_layout)
        # Row gather descriptor: one 128 B K_SW128 line of x per box row; gather4 fetches 4 rows per issue.
        x_gather_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, cute.make_layout((1, 128), stride=(128, 1))
        )
        tma_atom_x, _ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), x, x_gather_layout, (1, 128)
        )

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
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
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_x,
            x,
            x_sf,
            act,
            act_sf,
            p2e,
            t2e,
            t2l,
            ntiles,
            rearm_act,
            rearm_sf,
            rearm_rows,
            ctr,
            rearm_ctr,
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

    def _situ(self, g, u, gate_cap, inv_gate_cap, linear_cap, inv_linear_cap):
        if self.math == "fast":
            gate = (gate_cap * tanh_approx(g * inv_gate_cap)) * (
                cutlass.Float32(0.5) * tanh_approx(g * cutlass.Float32(0.5)) + cutlass.Float32(0.5)
            )
            return gate * (linear_cap * tanh_approx(u * inv_linear_cap))
        gate = (gate_cap * tanh_accurate(g * inv_gate_cap)) * sigmoid_accurate(g)
        return gate * (linear_cap * tanh_accurate(u * inv_linear_cap))

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mX: cute.Tensor,
        mXsf: cute.Tensor,
        mAct: cute.Tensor,
        mActSf: cute.Tensor,
        mP2e: cute.Tensor,
        mT2e: cute.Tensor,
        mT2l: cute.Tensor,
        mNumTiles: cute.Tensor,
        mRearmAct: cute.Tensor,
        mRearmSf: cute.Tensor,
        mRearmRows: cute.Tensor,
        mCtr: cute.Tensor,
        mRearmCtr: cute.Tensor,
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
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)
        if warp_idx == self.gather_warp_id:
            cpasync.prefetch_descriptor(tma_atom_x)
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
        a_pipeline = CpAsyncUmmaPipeline.create(
            self.num_ab_stage,
            self.threads_per_warp * self.num_gather_warps,
            storage.a_mbar.data_ptr(),
            async_load=self.sf_path == "cpasync",
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
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFB = thr_mma.partition_B(gSFB_nkl)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
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
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Every read of a predecessor's output (the plan, the activations) comes after this wait.
        if cutlass.const_expr(self.use_pdl):
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
                tBgB_t = tBgB[(None, n_tile, None, expert)]
                tBgSFB_t = tBgSFB[(None, n_tile, None, expert)]
                for k_tile in cutlass.range(self.num_k_tiles, unroll=1):
                    ab_pipeline.producer_acquire(ab_state)
                    tma_bar = ab_pipeline.producer_get_barrier(ab_state)
                    cute.copy(
                        tma_atom_b,
                        tBgB_t[(None, k_tile)],
                        tBsB[(None, ab_state.index)],
                        tma_bar_ptr=tma_bar,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_t[(None, k_tile)],
                        tBsSFB[(None, ab_state.index)],
                        tma_bar_ptr=tma_bar,
                    )
                    ab_state.advance()
            ab_pipeline.producer_tail(ab_state)

        if cutlass.const_expr(self.rearm):
            # Producer re-arm of the previous use's slot pair (its reader, the previous FC2, finished before this
            # kernel's grid wait returned): sentinel words over the rows that use wrote, a slice per CTA.
            if warp_idx == self.rearm_warp_id:
                lane = tidx & 31
                rows_prev = mRearmRows[0]
                per_cta = (rows_prev + gdim - 1) // gdim
                row0 = cutlass.min(bidx * per_cta, rows_prev)
                row1 = cutlass.min(row0 + per_cta, rows_prev)
                act_word = cutlass.Int32(-1) if self.sentinel == "ff" else cutlass.Int32(0x80808080)
                a_base = mRearmAct.iterator.toint() + cutlass.Int64(row0) * INTER
                n_chunks = (row1 - row0) * (INTER // 16)
                for c in cutlass.range(lane, n_chunks, 32, unroll=4):
                    st_global_v4(
                        a_base + cutlass.Int64(c) * 16, act_word, act_word, act_word, act_word
                    )
                s_base = mRearmSf.iterator.toint() + cutlass.Int64(row0) * (INTER // SF_VEC)
                n_chunks = (row1 - row0) * (INTER // SF_VEC // 16)
                for c in cutlass.range(lane, n_chunks, 32, unroll=4):
                    st_global_v4(s_base + cutlass.Int64(c) * 16, -1, -1, -1, -1)
                if cutlass.const_expr(self.counters):
                    for t in cutlass.range(
                        row0 // M_TILE + lane, (row1 + M_TILE - 1) // M_TILE, 32, unroll=1
                    ):
                        mRearmCtr[t] = cutlass.Int32(0)

        if (warp_idx >= self.gather_warp_id) & (warp_idx < self.rearm_warp_id):
            gather_sub = warp_idx - self.gather_warp_id
            a_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            lane = tidx & 31
            # Logical smem byte of (row, k) in a stage of the SW128 K-major A tile: kt * 128 * 128 + row * 128 +
            # k % 128 (kt = k // 128); the swizzled sA iterator applies the 128 B XOR. SF atom byte of (row,
            # k block): kt * 512 + (row % 32) * 16 + (row // 32) * 4 + k block % 4.
            chunk = lane & 7
            row_in_pass = lane >> 3
            rows_per_warp = M_TILE // self.num_gather_warps
            n_pass = rows_per_warp // 4
            n_kt = self.k_stage // 128
            sf_rows_per_warp = rows_per_warp
            n_sf = max(1, sf_rows_per_warp // 32)
            # Lane l < n_issue issues the gather4 of (K chunk l // n_pass, 4-row group l % n_pass) and holds those
            # 4 row ids; with fewer than 32 rows per warp only lanes < rows_per_warp handle a scale row.
            n_issue = n_kt * n_pass
            sf_lane_ok = lane < sf_rows_per_warp if sf_rows_per_warp < 32 else None
            issue_kt = lane // n_pass
            issue_i = lane - issue_kt * n_pass
            a_atom_copy = cute.make_copy_atom(
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
            use_tma = self.a_path == "tma" and self.diag != "nogather"
            sf_ld = self.sf_path == "ld" and self.diag not in ("nogather", "nosf")
            lanes_issue = use_tma and self.gather_issue == "lanes"
            rid = cute.make_rmem_tensor((rows_per_warp,), cutlass.Int32)
            rid4 = cute.make_rmem_tensor((4,), cutlass.Int32)
            sA_plain = storage.sA.data_ptr()
            sSFA_base = storage.sSFA.data_ptr().toint()
            a_tx_bytes = rows_per_warp * self.k_stage
            n_sf_words = HIDDEN // SF_VEC // 4
            sf_words = cute.make_rmem_tensor((n_sf * n_sf_words,), cutlass.Int32)
            xsf_base = mXsf.iterator.toint()
            for item in cutlass.range(bidx, total_items, gdim, unroll=1):
                m_tile = item // n_tiles
                limit = mT2l[m_tile]
                tile_start = m_tile * M_TILE
                if cutlass.const_expr(lanes_issue):
                    for j in cutlass.range_constexpr(4):
                        row = (
                            gather_sub * rows_per_warp
                            + 4 * cutlass.min(issue_i, cutlass.Int32(n_pass - 1))
                            + j
                        )
                        src = mP2e[tile_start + row]
                        ok = (src >= 0) & (tile_start + row < limit)
                        rid4[j] = cutlass.select_(ok, src // TOP_K, cutlass.Int32(-1))
                elif cutlass.const_expr(use_tma):
                    for r in cutlass.range_constexpr(rows_per_warp):
                        row = gather_sub * rows_per_warp + r
                        src = mP2e[tile_start + row]
                        ok = (src >= 0) & (tile_start + row < limit)
                        rid[r] = cutlass.select_(ok, src // TOP_K, cutlass.Int32(-1))
                else:
                    for i in cutlass.range_constexpr(n_pass):
                        row = gather_sub * rows_per_warp + 4 * i + row_in_pass
                        src = mP2e[tile_start + row]
                        row_ok[i] = (src >= 0) & (tile_start + row < limit)
                        row_src[i] = cutlass.max(src, cutlass.Int32(0)) // TOP_K
                for q in cutlass.range_constexpr(n_sf):
                    row = gather_sub * sf_rows_per_warp + 32 * q + lane
                    if cutlass.const_expr(sf_lane_ok is not None):
                        row = cutlass.min(row, cutlass.Int32(M_TILE - 1))
                    src = mP2e[tile_start + row]
                    sf_ok[q] = (src >= 0) & (tile_start + row < limit)
                    if cutlass.const_expr(sf_lane_ok is not None):
                        sf_ok[q] = sf_ok[q] & sf_lane_ok
                    sf_src[q] = cutlass.max(src, cutlass.Int32(0)) // TOP_K
                if cutlass.const_expr(sf_ld):
                    # The row's 112 scale bytes for the whole item, fetched once as 7 x 16 B L2-only loads: 4 B
                    # L1-cached loads thrash the small L1 left beside a large smem carveout (2.5x slower kernel).
                    for q in cutlass.range_constexpr(n_sf):
                        row_addr = xsf_base + cutlass.Int64(sf_src[q]) * (HIDDEN // SF_VEC)
                        for v4 in cutlass.range_constexpr(n_sf_words // 4):
                            if cutlass.const_expr(self.diag == "sfzero"):
                                words = [cutlass.Int32(0)] * 4
                            else:
                                words = ld_global_cg_v4(row_addr + 16 * v4)
                            for j in cutlass.range_constexpr(4):
                                sf_words[q * n_sf_words + 4 * v4 + j] = cutlass.select_(
                                    sf_ok[q], words[j], cutlass.Int32(0)
                                )
                # cp.async with a false predicate still zero-fills its destination: padding rows become zeros.
                # Unrolled: the register-held scale words are indexed by k_tile.
                for k_tile in cutlass.range_constexpr(self.num_k_tiles):
                    a_pipeline.producer_acquire(a_state)
                    sA_stage = sA.iterator + a_state.index * self.a_bytes_per_stage
                    sSFA_stage = sSFA.iterator + a_state.index * self.sfa_bytes_per_stage
                    k0 = k_tile * self.k_stage
                    if cutlass.const_expr(lanes_issue):
                        tma_bar = a_pipeline.producer_get_barrier(a_state)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_expect_tx(tma_bar, a_tx_bytes)
                        if lane < n_issue:
                            dst = sA_plain + (
                                a_state.index * self.a_bytes_per_stage
                                + issue_kt * (M_TILE * 128)
                                + (gather_sub * rows_per_warp + 4 * issue_i) * 128
                            )
                            tma_gather4(
                                tma_atom_x,
                                dst,
                                tma_bar,
                                k0 + issue_kt * 128,
                                rid4[0],
                                rid4[1],
                                rid4[2],
                                rid4[3],
                            )
                    elif cutlass.const_expr(use_tma):
                        with cute.arch.elect_one():
                            tma_bar = a_pipeline.producer_get_barrier(a_state)
                            cute.arch.mbarrier_expect_tx(tma_bar, a_tx_bytes)
                            for kt in cutlass.range_constexpr(n_kt):
                                for i in cutlass.range_constexpr(n_pass):
                                    dst = sA_plain + (
                                        a_state.index * self.a_bytes_per_stage
                                        + kt * M_TILE * 128
                                        + (gather_sub * rows_per_warp + 4 * i) * 128
                                    )
                                    tma_gather4(
                                        tma_atom_x,
                                        dst,
                                        tma_bar,
                                        k0 + kt * 128,
                                        rid[4 * i],
                                        rid[4 * i + 1],
                                        rid[4 * i + 2],
                                        rid[4 * i + 3],
                                    )
                    for kt in cutlass.range_constexpr(
                        0 if (self.diag == "nogather" or use_tma) else n_kt
                    ):
                        for i in cutlass.range_constexpr(n_pass):
                            row = gather_sub * rows_per_warp + 4 * i + row_in_pass
                            dst_off = kt * M_TILE * 128 + row * 128 + chunk * 16
                            src_off = cute.assume(
                                row_src[i] * HIDDEN + k0 + kt * 128 + chunk * 16, divby=16
                            )
                            g_a = cute.make_tensor(mX.iterator + src_off, cute.make_layout((16,)))
                            s_a = cute.make_tensor(sA_stage + dst_off, cute.make_layout((16,)))
                            pred1[0] = row_ok[i]
                            cute.copy_atom_call(a_atom_copy, g_a, s_a, pred=pred1)
                    if cutlass.const_expr(sf_ld):
                        for kt in cutlass.range_constexpr(n_kt):
                            for q in cutlass.range_constexpr(n_sf):
                                row = gather_sub * sf_rows_per_warp + 32 * q + lane
                                if cutlass.const_expr(sf_lane_ok is not None):
                                    row = cutlass.min(row, cutlass.Int32(M_TILE - 1))
                                sf_dst = (
                                    sSFA_base
                                    + a_state.index * self.sfa_bytes_per_stage
                                    + kt * SF_ATOM_BYTES
                                    + (row % 32) * 16
                                    + (row // 32) * 4
                                )
                                sf_val = sf_words[
                                    q * n_sf_words + k_tile * (self.k_blocks // 4) + kt
                                ]
                                if cutlass.const_expr(sf_lane_ok is None):
                                    st_shared_b32(sf_dst, sf_val)
                                else:
                                    if sf_lane_ok:
                                        st_shared_b32(sf_dst, sf_val)
                        cute.arch.fence_proxy("async.shared", space="cta")
                    for kt in cutlass.range_constexpr(
                        0 if (self.diag in ("nogather", "nosf") or sf_ld) else n_kt
                    ):
                        for q in cutlass.range_constexpr(n_sf):
                            row = gather_sub * sf_rows_per_warp + 32 * q + lane
                            sf_src_off = cute.assume(
                                sf_src[q] * (HIDDEN // SF_VEC) + k_tile * self.k_blocks + kt * 4,
                                divby=4,
                            )
                            g_sf = cute.make_tensor(
                                mXsf.iterator + sf_src_off, cute.make_layout((4,))
                            )
                            s_sf = cute.make_tensor(
                                cute.recast_ptr(
                                    sSFA_stage
                                    + kt * SF_ATOM_BYTES
                                    + (row % 32) * 16
                                    + (row // 32) * 4,
                                    dtype=cutlass.Uint8,
                                ),
                                cute.make_layout((4,)),
                            )
                            pred1[0] = sf_ok[q]
                            cute.copy_atom_call(sf_atom_copy, g_sf, s_sf, pred=pred1)
                    a_pipeline.producer_commit(a_state)
                    a_state.advance()
            a_pipeline.producer_tail(a_state)

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
            a_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            num_kblocks = cute.size(tCrA, mode=[2])
            for item in cutlass.range(bidx, total_items, gdim, unroll=1):
                acc_pipeline.producer_acquire(acc_state)
                tcgen05_fence("after")
                tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile in cutlass.range(self.num_k_tiles, unroll=1):
                    ab_pipeline.consumer_wait(ab_state)
                    a_pipeline.consumer_wait(a_state)
                    # cp.async (generic proxy) writes -> tcgen05 (async proxy) reads.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.copy(
                        copy_s2t_sfa,
                        tCsSFA_s2t[(None, None, None, None, a_state.index)],
                        tCtSFA_s2t,
                    )
                    cute.copy(
                        copy_s2t_sfb,
                        tCsSFB_s2t[(None, None, None, None, ab_state.index)],
                        tCtSFB_s2t,
                    )
                    if cutlass.const_expr(self.diag != "nomma"):
                        for kblock in cutlass.range_constexpr(num_kblocks):
                            sf_coord = (None, None, kblock)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_coord].iterator)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[(None, None, kblock, a_state.index)],
                                tCrB[(None, None, kblock, ab_state.index)],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_pipeline.consumer_release(ab_state)
                    ab_state.advance()
                    a_pipeline.consumer_release(a_state)
                    a_state.advance()
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
            act_base = mAct.iterator.toint()
            sf_base = mActSf.iterator.toint()
            gate_cap = cutlass.Float32(self.gate_cap)
            inv_gate_cap = cutlass.Float32(1.0 / self.gate_cap)
            linear_cap = cutlass.Float32(self.linear_cap)
            inv_linear_cap = cutlass.Float32(1.0 / self.linear_cap)
            n_groups = tile_n // 64
            ch_per_tile = tile_n // 2
            if cutlass.const_expr(self.hazard_delay_ns > 0):
                spin_ns(cutlass.Int64(self.hazard_delay_ns))
            for item in cutlass.range(bidx, total_items, gdim, unroll=1):
                m_tile = item // n_tiles
                n_tile = item - m_tile * n_tiles
                limit = mT2l[m_tile]
                prow = m_tile * M_TILE + 32 * quad + lane
                valid = prow < limit
                sf_word = cutlass.Int32(0)
                row_act = (
                    act_base + cutlass.Int64(prow) * INTER + cutlass.Int64(n_tile * ch_per_tile)
                )
                kb0 = n_tile * (ch_per_tile // SF_VEC)
                if cutlass.const_expr(self.sf_layout == "linear"):
                    row_sf = sf_base + cutlass.Int64(prow) * (INTER // SF_VEC) + cutlass.Int64(kb0)
                else:
                    row_sf = (
                        sf_base
                        + cutlass.Int64(m_tile) * (M_TILE * (INTER // SF_VEC))
                        + cutlass.Int64(
                            (kb0 // 4) * SF_ATOM_BYTES + (kb0 % 4) + lane * 16 + quad * 4
                        )
                    )
                acc_pipeline.consumer_wait(acc_state)
                tcgen05_fence("after")
                tmem_stage = tmem_quad + acc_state.index * tile_n
                for h in cutlass.range_constexpr(n_groups):
                    cols = tmem_load_32x32b_x32([tmem_stage + 64 * h, tmem_stage + 64 * h + 32])
                    if cutlass.const_expr(h == n_groups - 1):
                        tcgen05_fence("before")
                        acc_pipeline.consumer_release(acc_state)
                        acc_state.advance()
                    act_v = [None] * 32
                    for jb in cutlass.range_constexpr(2):
                        for q in cutlass.range_constexpr(8):
                            for odd in cutlass.range_constexpr(2):
                                u = cols[32 * jb + 16 * odd + q]
                                g = cols[32 * jb + 16 * odd + 8 + q]
                                if cutlass.const_expr(self.diag == "noepi"):
                                    act_v[16 * jb + 2 * q + odd] = g + u
                                else:
                                    act_v[16 * jb + 2 * q + odd] = self._situ(
                                        g, u, gate_cap, inv_gate_cap, linear_cap, inv_linear_cap
                                    )
                    amax = cute.arch.fmax(act_v[0], -act_v[0])
                    for c in cutlass.range_constexpr(1, 32):
                        amax = cute.arch.fmax(amax, cute.arch.fmax(act_v[c], -act_v[c]))
                    sf_byte, inv_scale = ue8m0_scale(amax)
                    sf_byte = cutlass.select_(valid, sf_byte, cutlass.Int32(0))
                    sf_word = sf_word | (sf_byte << (8 * h))
                    for jb in cutlass.range_constexpr(2):
                        words = []
                        for wq in cutlass.range_constexpr(4):
                            c0 = e4m3x2(
                                act_v[16 * jb + 4 * wq + 1] * inv_scale,
                                act_v[16 * jb + 4 * wq] * inv_scale,
                            )
                            c1 = e4m3x2(
                                act_v[16 * jb + 4 * wq + 3] * inv_scale,
                                act_v[16 * jb + 4 * wq + 2] * inv_scale,
                            )
                            if cutlass.const_expr(self.sentinel == "80"):
                                # E4M3 -0 (0x80) is the slot sentinel: store +0, the same value.
                                c0 = c0 & ~(
                                    cutlass.select_(
                                        (c0 & 0x7F) == 0, cutlass.Int32(0x80), cutlass.Int32(0)
                                    )
                                    | cutlass.select_(
                                        (c0 & 0x7F00) == 0, cutlass.Int32(0x8000), cutlass.Int32(0)
                                    )
                                )
                                c1 = c1 & ~(
                                    cutlass.select_(
                                        (c1 & 0x7F) == 0, cutlass.Int32(0x80), cutlass.Int32(0)
                                    )
                                    | cutlass.select_(
                                        (c1 & 0x7F00) == 0, cutlass.Int32(0x8000), cutlass.Int32(0)
                                    )
                                )
                            words.append(cutlass.select_(valid, c0 | (c1 << 16), cutlass.Int32(0)))
                        addr = row_act + 32 * h + 16 * jb
                        if cutlass.const_expr(self.store == "v4"):
                            st_global_v4(addr, words[0], words[1], words[2], words[3])
                        else:
                            for wq in cutlass.range_constexpr(4):
                                st_global_b16(addr + 4 * wq, words[wq])
                                st_global_b16(addr + 4 * wq + 2, words[wq] >> 16)
                if cutlass.const_expr(n_groups == 4):
                    st_global_b32(row_sf, sf_word)
                else:
                    st_global_b16(row_sf, sf_word)
                if cutlass.const_expr(self.counters):
                    # Counter link: once all four epilogue warps stored this tile, one release-add per (m, n) item;
                    # FC2 acquires n_tiles of them per m-tile.
                    self.epilog_sync_barrier.arrive_and_wait()
                    if tidx == 0:
                        red_release_gpu_add(
                            mCtr.iterator.toint() + cutlass.Int64(m_tile) * 4, cutlass.Int32(1)
                        )
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
        if cutlass.const_expr(self.use_pdl and self.pdl_trigger == "end"):
            cute.arch.griddepcontrol_launch_dependents()


_compiled = {}


def num_ctas_for(x: torch.Tensor, plan, tile_n: int) -> int:
    return min(num_sms(x.device.index), plan.tile_idx_to_expert_idx.shape[0] * (W_ROWS // tile_n))


def check(x, x_sf, weights, plan, act, act_sf):
    rows = x.shape[0]
    num_experts = weights.w3_w1_weight.shape[0]
    check_tensor("x", x, torch.float8_e4m3fn, (rows, HIDDEN), 16)
    check_tensor("x_sf", x_sf, torch.uint8, (rows, HIDDEN // SF_VEC), 4)
    check_tensor(
        "w3_w1_weight", weights.w3_w1_weight, torch.uint8, (num_experts, W_ROWS, HIDDEN // 2), 128
    )
    check_tensor(
        "w3_w1_weight_scale",
        weights.w3_w1_weight_scale,
        torch.uint8,
        (num_experts, W_ROWS, HIDDEN // SF_VEC),
        128,
    )
    if plan.tile_n != M_TILE:
        raise ValueError(
            f"the plan must pad experts to {M_TILE} rows (plan.tile_n = {plan.tile_n})"
        )
    num_permuted = plan.permuted_idx_to_expanded_idx.shape[0]
    num_tiles_cap = plan.tile_idx_to_expert_idx.shape[0]
    if num_tiles_cap * M_TILE > num_permuted:
        raise ValueError(
            "plan: tile_idx_to_expert_idx covers more rows than permuted_idx_to_expanded_idx"
        )
    check_tensor("act", act, torch.float8_e4m3fn, (num_permuted, INTER), 16)
    check_tensor("act_sf", act_sf, torch.uint8, (num_permuted, INTER // SF_VEC), 16)
    for name in (
        "permuted_idx_to_expanded_idx",
        "tile_idx_to_expert_idx",
        "tile_idx_to_mn_limit",
        "num_non_exiting_tiles",
    ):
        t = getattr(plan, name)
        if t.dtype != torch.int32 or not t.is_contiguous():
            raise ValueError(f"plan.{name} must be a contiguous int32 tensor")
    if plan.tile_idx_to_mn_limit.shape[0] != num_tiles_cap:
        raise ValueError("plan: tile_idx_to_mn_limit and tile_idx_to_expert_idx differ in length")


def launch_args(x, x_sf, weights, plan, act, act_sf, num_ctas, rearm=None, counters=None):
    return (
        gmem_ptr(cutlass.Float4E2M1FN, weights.w3_w1_weight, 128),
        gmem_ptr(cutlass.Float8E8M0FNU, weights.w3_w1_weight_scale, 128),
        gmem_ptr(cutlass.Float8E4M3FN, x, 16),
        gmem_ptr(cutlass.Uint8, x_sf, 4),
        gmem_ptr(cutlass.Uint8, act, 16),
        gmem_ptr(cutlass.Uint8, act_sf, 16),
        gmem_ptr(cutlass.Int32, plan.permuted_idx_to_expanded_idx, 4),
        gmem_ptr(cutlass.Int32, plan.tile_idx_to_expert_idx, 4),
        gmem_ptr(cutlass.Int32, plan.tile_idx_to_mn_limit, 4),
        gmem_ptr(cutlass.Int32, plan.num_non_exiting_tiles, 4),
        gmem_ptr(cutlass.Int32, act if rearm is None else rearm[0], 16),
        gmem_ptr(cutlass.Int32, act_sf if rearm is None else rearm[1], 16),
        gmem_ptr(cutlass.Int32, plan.num_non_exiting_tiles if rearm is None else rearm[2], 4),
        gmem_ptr(
            cutlass.Int32, plan.tile_idx_to_expert_idx if counters is None else counters[0], 4
        ),
        gmem_ptr(
            cutlass.Int32, plan.tile_idx_to_expert_idx if counters is None else counters[1], 4
        ),
        cutlass.Int32(weights.w3_w1_weight.shape[0]),
        cutlass.Int32(x.shape[0]),
        cutlass.Int32(act.shape[0]),
        cutlass.Int32(plan.tile_idx_to_expert_idx.shape[0]),
        cutlass.Int32(num_ctas),
    )


def run(
    x,
    x_sf,
    weights,
    plan,
    act,
    act_sf,
    *,
    tile_n: int = 256,
    math: str = "fast",
    store: str = "v4",
    sf_layout: str = "swizzled",
    pdl_trigger: str = "end",
    k_blocks: int = 8,
    max_ab_stages: int = 3,
    smem_budget: Optional[int] = None,
    num_acc_stages: int = 2,
    num_gather_warps: int = 8,
    enable_pdl: Optional[bool] = None,
    num_ctas: Optional[int] = None,
    diag: Optional[str] = None,
    a_path: str = "tma",
    sf_path: str = "ld",
    gather_issue: str = "elect",
    sentinel: Optional[str] = None,
    rearm=None,
    hazard_delay_ns: int = 0,
    counters=None,
):
    """act[p], act_sf[p] = MXFP8(SiTU-GLU(x[p2e[p] // 16] @ W31[e]^T)) for every row p of every valid 128-row tile of
    the plan (rows at or past the tile's mn_limit get zeros). ``x`` e4m3 [R, 3584], ``x_sf`` uint8 [R, 112] (linear
    UE8M0), ``weights.w3_w1_weight`` uint8 [E, 6144, 1792] + ``w3_w1_weight_scale`` uint8 [E, 6144, 112] in the
    TRTLLM-Gen layout, ``weights.gate_cap`` / ``linear_cap``; ``plan`` with ``tile_n`` = 128 and the moe_chain
    RoutingPlan tensors (``permuted_idx_to_expanded_idx`` int32 [P], ``tile_idx_to_expert_idx`` /
    ``tile_idx_to_mn_limit`` int32 [T], ``num_non_exiting_tiles`` int32 [1]); ``act`` e4m3 [P, 3072], ``act_sf``
    uint8 [P, 96] (``sf_layout`` linear or the 128x4 atom layout, see the module docstring) with P >= 128 T. One
    kernel on the current stream; no allocation, no host synchronization.
    Lamport FC1 -> FC2 edge: ``sentinel="80"`` stores E4M3 -0 as +0 (the act slot sentinel; scale bytes never reach
    the 0xFF scale sentinel), ``"ff"`` needs no fix-up; ``rearm=(act_prev, act_sf_prev, rows_prev)`` re-arms the
    previous use's slot pair over its first ``rows_prev[0]`` rows (int32 device word) after the grid wait;
    ``hazard_delay_ns`` delays every store (tests). Counter link: ``counters=(ctr, ctr_prev)``, int32 [T] words
    zeroed before use; ctr[m] is release-added once per (m, n) item after the tile's stores (reaches 6144 / tile_n
    per m-tile), and ctr_prev (the previous use's, finished) is zeroed with the re-arm."""
    check(x, x_sf, weights, plan, act, act_sf)
    if counters is not None and (
        counters[0].dtype != torch.int32
        or counters[0].numel() < plan.tile_idx_to_expert_idx.shape[0]
    ):
        raise ValueError("counters = (ctr int32 [T], ctr_prev int32 [T])")
    if rearm is not None:
        if (
            rearm[2].dtype != torch.int32
            or rearm[0].shape[1] != INTER
            or rearm[1].shape[1] != INTER // SF_VEC
        ):
            raise ValueError(
                "rearm = (act_prev [P, 3072], act_sf_prev [P, 96], rows_prev int32 [1])"
            )
    pdl = use_pdl() if enable_pdl is None else enable_pdl
    key = (
        tile_n,
        weights.gate_cap,
        weights.linear_cap,
        pdl,
        pdl_trigger,
        math,
        store,
        sf_layout,
        k_blocks,
        num_acc_stages,
        max_ab_stages,
        smem_budget,
        num_gather_warps,
        diag,
        a_path,
        sf_path,
        gather_issue,
        sentinel,
        rearm is not None,
        hazard_delay_ns,
        counters is not None,
    )
    ctas = num_ctas_for(x, plan, tile_n) if num_ctas is None else num_ctas
    args = launch_args(x, x_sf, weights, plan, act, act_sf, ctas, rearm, counters)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled = _compiled.get(key)
    if compiled is None:
        kernel = Fc1TokenM(
            tile_n,
            weights.gate_cap,
            weights.linear_cap,
            pdl,
            pdl_trigger=pdl_trigger,
            math=math,
            store=store,
            sf_layout=sf_layout,
            k_blocks=k_blocks,
            num_acc_stages=num_acc_stages,
            max_ab_stages=max_ab_stages,
            smem_budget=smem_budget,
            num_gather_warps=num_gather_warps,
            diag=diag,
            a_path=a_path,
            sf_path=sf_path,
            gather_issue=gather_issue,
            sentinel=sentinel,
            rearm=rearm is not None,
            hazard_delay_ns=hazard_delay_ns,
            counters=counters is not None,
        )
        compiled = _compiled[key] = cute.compile(kernel, *args, stream)
    compiled(*args, stream)
