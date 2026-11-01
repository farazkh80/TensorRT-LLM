# Copyright (c) 2025 by FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RMSNorm for the Kimi K3 MLA q_a / kv_a norms, vendored from FlashInfer 0.6.18 with fine-grained-sync modes.

The default build is FlashInfer's ``RMSNormKernel`` (``flashinfer/norm/kernels/rmsnorm.py``) for the single-CTA
case (no cluster), unchanged: same tiling, same cp.async staging, same reduction order, so the output is
bit-identical to ``flashinfer.norm.rmsnorm``. Only rows whose H fits one CTA (cluster_n == 1) are supported, which
covers the MLA norms (H = 1536 and 512).

Fine-grained sync (Lamport) modes, opt-in and combinable:
  fgs_consumer  x is an FGS slot (the fused q_a / kv_a GEMM output): launched with PDL and no
                ``griddepcontrol.wait``; the weight is loaded first, then x straight into registers with 16-B
                ``ld.global.cg`` loads (no cp.async staging); while any element of a thread's fragment is the -0.0
                sentinel the warp waits (warp-uniform, nanosleep backoff) and reloads that fragment with
                ``ld.relaxed.gpu``. The reduction and the output are those of the default build.
  fgs_producer  y is published to FGS consumers (the q_b GEMM, the latent / context append): dependents are
                triggered on entry and -0.0 is stored as +0.0 (each 16-B chunk is one vector store).
"""

import functools
import math
import operator
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

try:
    from .mla_fgs import SPIN_NS_MAX, SPIN_NS_MIN, STATS_LEN, bf16x2_has_sentinel, canon_bf16x2, nanosleep, trap
except ImportError:
    from mla_fgs import SPIN_NS_MAX, SPIN_NS_MIN, STATS_LEN, bf16x2_has_sentinel, canon_bf16x2, nanosleep, trap

COPY_BITS = 128


def _warp_reduce(val, op, width: cutlass.Constexpr[int] = 32):
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def _block_reduce_multirow(
    val: Float32, op: cutlass.Constexpr, reduction_buffer: cute.Tensor, init_val: Float32
) -> Float32:
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return _warp_reduce(block_reduce_val, op)


@cute.jit
def _row_reduce_sum(
    x: cute.TensorSSA, threads_per_row: cutlass.Constexpr[int], reduction_buffer: cute.Tensor
) -> Float32:
    local_val = x.reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
    warp_val = _warp_reduce(local_val, operator.add, width=min(threads_per_row, 32))
    if cutlass.const_expr(threads_per_row > 32):
        return _block_reduce_multirow(warp_val, operator.add, reduction_buffer, Float32(0.0))
    else:
        return warp_val


@cute.jit
def _predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(tXcX[(0, rest_v), 0, rest_k][1], limit)
    return tXpX


@cute.jit
def _fragment_pending(frag: cute.Tensor) -> Int32:
    words = cute.recast_tensor(frag, Int32)
    acc = Int32(0)
    for i in cutlass.range_constexpr(cute.size(words)):
        acc = acc | bf16x2_has_sentinel(words[i])
    return acc


class MlaRMSNormKernel:
    """y = x / sqrt(mean(x^2) + eps) * w for [M, H] bf16 rows (FlashInfer ``RMSNormKernel``, cluster_n == 1)."""

    def __init__(
        self,
        H: int,
        fgs_consumer: bool = False,
        fgs_producer: bool = False,
        spin_limit: int = 0,
        with_stats: bool = False,
    ):
        self.dtype = cutlass.BFloat16
        self.H = H
        elem_bytes = self.dtype.width // 8
        h_align = H & (-H)
        self.vec_size = min(h_align, COPY_BITS // 8 // elem_bytes)
        self.copy_bits = self.vec_size * self.dtype.width
        self.threads_per_row = self._threads_per_row(H)
        self.num_threads = 128
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)
        self.num_vec_blocks = max(
            1, (H // self.vec_size + self.threads_per_row - 1) // self.threads_per_row
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row
        self.fgs_consumer = fgs_consumer
        self.fgs_producer = fgs_producer
        self.spin_limit = spin_limit
        self.with_stats = with_stats
        # FlashInfer stages x through shared memory with cp.async whenever the tile fits (always for H <= 16384);
        # the FGS consumer loads x into registers so that it can check and reload it.
        self.use_async_copy = self.copy_bits >= 32 and not fgs_consumer
        if self.vec_size * self.dtype.width != COPY_BITS and (fgs_consumer or fgs_producer):
            raise ValueError("FGS modes need 16-byte vectors (H a multiple of 8)")

    @staticmethod
    def _threads_per_row(H: int) -> int:
        if H <= 64:
            return 8
        if H <= 128:
            return 16
        if H <= 3072:
            return 32
        if H <= 6144:
            return 64
        if H <= 16384:
            return 128
        raise ValueError(f"H={H} needs a CTA cluster; not supported here")

    def _tv_layout(self):
        shape = ((self.threads_per_row, self.rows_per_block), (self.vec_size, self.num_vec_blocks))
        stride = (
            (self.vec_size * self.rows_per_block, 1),
            (self.rows_per_block, self.rows_per_block * self.vec_size * self.threads_per_row),
        )
        return cute.make_layout(shape, stride=stride)

    def _smem_size_in_bytes(self) -> int:
        tile = self.rows_per_block * self.cols_per_tile * 2 if self.use_async_copy else 0
        return tile + self.rows_per_block * self.warps_per_row * 4

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mStats: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        self.kernel(mX, mW, mY, mStats, M, eps, enable_pdl, self._tv_layout(), tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl or self.fgs_consumer or self.fgs_producer,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mStats: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if cutlass.const_expr(self.fgs_producer):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(not self.fgs_consumer):
            if cutlass.const_expr(enable_pdl or self.fgs_producer):
                cute.arch.griddepcontrol_wait()

        H = self.H
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=self.copy_bits
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mY.element_type, num_bits_per_copy=self.copy_bits
        )
        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=self.copy_bits
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)
        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)

        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        tXgO = thr_copy_O.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)

        tXpX = _predicate_k(tXcX, limit=H)
        tWpW = _predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)
            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX, tXrX)
        elif cutlass.const_expr(self.fgs_consumer):
            self._load_x_fgs(mStats, tXgX, tXrX, tXpX, tWgW, tWrW, tWpW, copy_atom_sync, row_in_bounds)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = _row_reduce_sum(x_sq, threads_per_row, reduction_buffer)

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        cute.arch.barrier()

        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(0.0))

        tXrO.store(y.to(mY.element_type))
        if cutlass.const_expr(self.fgs_producer):
            words = cute.recast_tensor(tXrO, Int32)
            for i in cutlass.range_constexpr(cute.size(words)):
                words[i] = canon_bf16x2(words[i])

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        if cutlass.const_expr(enable_pdl and not self.fgs_producer):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _load_x_fgs(
        self,
        mStats: cute.Tensor,
        tXgX: cute.Tensor,
        tXrX: cute.Tensor,
        tXpX: cute.Tensor,
        tWgW: cute.Tensor,
        tWrW: cute.Tensor,
        tWpW: cute.Tensor,
        copy_atom_sync: cute.CopyAtom,
        row_in_bounds: cutlass.Boolean,
    ):
        """x for the FGS consumer: weight first, then x (``.cg``); a warp with a sentinel left in any lane's
        fragment sleeps and reloads those fragments (``ld.relaxed.gpu``) until none is left."""
        first_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            tXgX.element_type,
            num_bits_per_copy=self.copy_bits,
            load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
        )
        reload_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            tXgX.element_type,
            num_bits_per_copy=self.copy_bits,
            memory_order=cute.nvgpu.MemoryOrder.RELAXED,
            memory_scope=cute.nvgpu.MemoryScope.GPU,
        )
        cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)
        tXrX.store(cute.zeros_like(tXrX, dtype=tXrX.element_type))
        if row_in_bounds:
            cute.copy(first_atom, tXgX, tXrX, pred=tXpX)
        pending = _fragment_pending(tXrX)
        if cute.arch.vote_any_sync(pending != Int32(0)):
            sweeps = Int32(0)
            sleep_ns = Int32(SPIN_NS_MIN)
            again = cutlass.Boolean(True)
            while again:
                nanosleep(sleep_ns)
                sleep_ns = cutlass.min(sleep_ns * 2, Int32(SPIN_NS_MAX))
                if pending != Int32(0):
                    cute.copy(reload_atom, tXgX, tXrX, pred=tXpX)
                    pending = _fragment_pending(tXrX)
                sweeps = sweeps + 1
                if cutlass.const_expr(self.spin_limit > 0):
                    if sweeps > Int32(self.spin_limit):
                        trap()
                again = cute.arch.vote_any_sync(pending != Int32(0))
            if cutlass.const_expr(self.with_stats):
                if cute.arch.lane_idx() == 0:
                    cute.arch.atomic_add(mStats.iterator, sweeps)
                    cute.arch.atomic_add(mStats.iterator + 1, Int32(1))


@functools.cache
def _compiled(H: int, enable_pdl: bool, consumer: bool, producer: bool, spin_limit: int, with_stats: bool):
    kernel = MlaRMSNormKernel(H, consumer, producer, spin_limit, with_stats)
    sym_m = cute.sym_int(64)
    bf16 = cutlass.BFloat16
    x = cute.runtime.make_fake_tensor(
        bf16, (sym_m, H), (cute.sym_int64(divisibility=kernel.vec_size), 1), assumed_align=16
    )
    y = cute.runtime.make_fake_tensor(
        bf16, (sym_m, H), (cute.sym_int64(divisibility=kernel.vec_size), 1), assumed_align=16
    )
    w = cute.runtime.make_fake_compact_tensor(bf16, (H,), assumed_align=16)
    stats = cute.runtime.make_fake_compact_tensor(Int32, (STATS_LEN,), assumed_align=4)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel, x, w, y, stats, Int64(1), Float32(1e-6), enable_pdl, stream, options="--enable-tvm-ffi"
    )


@functools.cache
def _dummy_stats(device) -> torch.Tensor:
    return torch.zeros(STATS_LEN, dtype=torch.int32, device=device)


def mla_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    *,
    enable_pdl: bool = False,
    fgs_consumer: bool = False,
    fgs_producer: bool = False,
    fgs_stats: Optional[torch.Tensor] = None,
    fgs_spin_limit: int = 0,
) -> torch.Tensor:
    """out = rmsnorm(x) * weight for 2-D bf16 x / out with unit last stride (row strides multiples of 8).

    Default build: ``flashinfer.norm.rmsnorm(x, weight, eps, out=out, enable_pdl=enable_pdl)`` bit for bit.
    ``fgs_consumer`` / ``fgs_producer``: see the module docstring; ``fgs_stats`` (int32 [4]) accumulates the
    consumer's reload sweeps / waiting warps; ``fgs_spin_limit`` > 0 traps after that many sweeps (diagnostic).
    """
    if x.dim() != 2 or out.shape != x.shape or x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("mla_rmsnorm needs 2-D bf16 x / out of equal shape")
    M, H = x.shape
    if x.stride(1) != 1 or out.stride(1) != 1 or x.data_ptr() % 16 or out.data_ptr() % 16:
        raise ValueError("x / out need unit last stride and 16-byte alignment")
    with_stats = fgs_consumer and fgs_stats is not None
    if with_stats and (fgs_stats.dtype != torch.int32 or fgs_stats.numel() < STATS_LEN):
        raise ValueError("fgs_stats must be an int32 tensor with at least 4 entries")
    if M == 0:
        return out
    stats = fgs_stats if with_stats else _dummy_stats(x.device)
    spin_limit = fgs_spin_limit if fgs_consumer else 0
    fn = _compiled(H, enable_pdl, fgs_consumer, fgs_producer, spin_limit, with_stats)
    fn(x, weight, out, stats, M, eps)
    return out
