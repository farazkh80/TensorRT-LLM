# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The RMSNorm kernel follows FlashInfer v0.6.18 flashinfer/norm/kernels/rmsnorm.py (RMSNormKernel, Apache-2.0,
# Copyright (c) 2025 by FlashInfer team), reduced to the single-CTA, synchronous-load case.
"""Kimi K3 decode MoE tail after the combine, in CuTe DSL:

  moe_out = bf16(f32(bf16(rmsnorm(y) @ W_up^T)) + f32(shared))

with y the combined routed output [M, 3584] bf16, ``rmsnorm`` the latent RMSNorm and W_up [7168, 3584] the latent
up-projection. Stock runs the FlashInfer CuTe RMSNorm, a cuBLASLt split-K GEMM + splitKreduce, and an ATen add.

* :func:`rmsnorm` is bitwise equal to the FlashInfer kernel: every thread owns the same 7 x 8 columns of its row
  (64 threads per row), reduces them in the same fragment order, then the warp butterfly and the two warp partials
  of the row; ``bf16(x * rstd * w)``.
* :func:`up_proj` runs the vendored FlashInfer split-K GEMM (cluster-local fp32 reduction, deterministic) with the
  add fused into its epilogue after the bf16 rounding, so the add is bitwise the ATen add of the rounded GEMM output.
  The GEMM itself differs from cuBLAS in its K-reduction order. Rows beyond 32 keep the stock cuBLAS GEMM and ATen
  add, which measured faster there than two 32-row split-K launches.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32

from ...kimi_k3_attn_res.attn_res_mix_cute import _lamport_wait, _ld_relaxed_v4, _pzo_b32

LATENT = 3584
HIDDEN = 7168
_VEC = 8
_THREADS_PER_ROW = 64
_SPLITK_ROWS = 32


def _pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


class RMSNormKernel:
    """One row per ``_THREADS_PER_ROW`` threads; ``rows_per_cta`` rows per CTA (FlashInfer uses 2 at H = 3584).

    ``fgs_consumer`` (M2d-1): reads x while its producer runs, without ``griddepcontrol.wait``; each thread re-reads
    its 16 B chunks of the row until none starts with the bf16 -0.0 sentinel (the producer stores whole chunks and
    never -0.0), falling back to waiting for the producer grid after ``fgs_max_spins`` rounds. ``fgs_producer``
    (M2d-2): dependents launch at entry and the output never holds -0.0, so it can feed a sentinel consumer.
    ``fgs_check=False`` (negative-control builds only) reads the consumer's input once without waiting.
    """

    def __init__(
        self,
        hidden: int,
        rows_per_cta: int,
        enable_pdl: bool,
        fgs_consumer: bool = False,
        fgs_producer: bool = False,
        fgs_max_spins: int = 1 << 16,
        fgs_check: bool = True,
    ):
        assert hidden % _VEC == 0
        assert not (fgs_consumer or fgs_producer) or enable_pdl, "the FGS builds launch with PDL"
        self.H = hidden
        self.rows = rows_per_cta
        self.threads = _THREADS_PER_ROW * rows_per_cta
        self.warps_per_row = _THREADS_PER_ROW // 32
        self.blocks = (hidden // _VEC + _THREADS_PER_ROW - 1) // _THREADS_PER_ROW
        self.cols_per_tile = _VEC * self.blocks * _THREADS_PER_ROW
        self.enable_pdl = enable_pdl
        self.fgs_consumer = fgs_consumer
        self.fgs_producer = fgs_producer
        self.fgs_max_spins = fgs_max_spins
        self.fgs_check = fgs_check

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        eps: Float32,
        mCnt: cute.Tensor,
        stream,
    ):
        tv_layout = cute.make_layout(
            ((_THREADS_PER_ROW, self.rows), (_VEC, self.blocks)),
            stride=((_VEC * self.rows, 1), (self.rows, self.rows * _VEC * _THREADS_PER_ROW)),
        )
        tiler_mn = (self.rows, self.cols_per_tile)
        self.kernel(mX, mW, mY, M, eps, mCnt, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows), 1, 1],
            block=[self.threads, 1, 1],
            smem=self.rows * self.warps_per_row * 4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        eps: Float32,
        mCnt: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.fgs_producer):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.enable_pdl and not self.fgs_consumer):
            cute.arch.griddepcontrol_wait()

        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(
            Float32, cute.make_layout((self.rows, self.warps_per_row)), byte_alignment=4
        )

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))
        mW_2d = cute.make_tensor(
            mW.iterator, cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        )
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        copy_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=_VEC * 16
        )
        copy_y = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mY.element_type, num_bits_per_copy=_VEC * 16
        )
        thr_x = cute.make_tiled_copy(copy_x, tv_layout, tiler_mn).get_slice(tidx)
        thr_y = cute.make_tiled_copy(copy_y, tv_layout, tiler_mn).get_slice(tidx)

        tXgX = thr_x.partition_S(gX)
        tXcX = thr_x.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)
        tWgW = thr_x.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXgO = thr_y.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)

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
        for v in cutlass.range_constexpr(tXpX.shape[0]):
            for b in cutlass.range_constexpr(tXpX.shape[2]):
                tXpX[v, 0, b] = cute.elem_less(tXcX[(0, v), 0, b][1], self.H)
        row_in_bounds = tXcX[(0, 0), 0, 0][0] < M

        tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
        if cutlass.const_expr(self.fgs_consumer):
            cute.copy(copy_x, tWgW, tWrW, pred=tXpX)
            if cutlass.const_expr(self.fgs_check):
                self._wait_published(tXgX, row_in_bounds, mCnt)
            if row_in_bounds:
                cute.copy(copy_x, tXgX, tXrX, pred=tXpX)
        else:
            if row_in_bounds:
                cute.copy(copy_x, tXgX, tXrX, pred=tXpX)
            cute.copy(copy_x, tWgW, tWrW, pred=tXpX)

        x = tXrX.load().to(Float32)
        local = (x * x).reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
        for i in cutlass.range_constexpr(5):
            local = local + cute.arch.shuffle_sync_bfly(local, offset=1 << i)
        warp = cute.arch.warp_idx()
        row = warp // self.warps_per_row
        if cute.arch.lane_idx() == 0:
            partials[row, warp % self.warps_per_row] = local
        cute.arch.barrier()
        sum_sq = partials[row, 0]
        for i in cutlass.range_constexpr(1, self.warps_per_row):
            sum_sq = sum_sq + partials[row, i]

        rstd = cute.math.rsqrt(sum_sq / Float32(self.H) + eps, fastmath=True)
        w = tWrW.load().to(Float32)
        y = x * rstd * (w + Float32(0.0))
        tXrO.store(y.to(mY.element_type))
        if cutlass.const_expr(self.fgs_producer):
            words = cute.recast_tensor(tXrO, Int32)
            for i in cutlass.range_constexpr(cute.size(words)):
                words[i] = _pzo_b32(words[i])
        if row_in_bounds:
            cute.copy(copy_y, tXrO, tXgO, pred=tXpX)
        if cutlass.const_expr(self.enable_pdl and not self.fgs_producer):
            if cutlass.const_expr(self.fgs_consumer):
                cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _wait_published(self, tXgX: cute.Tensor, row_in_bounds, mCnt: cute.Tensor):
        """Wait until none of this thread's 16 B chunks of ``tXgX`` holds the sentinel. The polls are L1-bypassing
        relaxed loads, so the plain loads that follow cannot hit a stale L1 line."""
        words, addrs = [], []
        for b in cutlass.range_constexpr(self.blocks):
            w = cute.make_rmem_tensor((_VEC // 2,), Int32)
            for i in cutlass.range_constexpr(_VEC // 2):
                w[i] = Int32(0)
            addr = tXgX[None, 0, b].iterator.toint()
            if row_in_bounds:
                vals = _ld_relaxed_v4(addr)
                for i in cutlass.range_constexpr(_VEC // 2):
                    w[i] = vals[i]
            words.append(w)
            addrs.append(addr)
        _lamport_wait(words, addrs, mCnt, self.fgs_max_spins)


class AddKernel:
    """``out = bf16(f32(a) + f32(b))`` over [M, H]; one CTA per (row, chunk), one 16 B vector per thread."""

    def __init__(self, hidden: int, chunks: int, enable_pdl: bool):
        vecs = hidden // _VEC
        assert hidden % _VEC == 0 and vecs % chunks == 0
        self.threads = vecs // chunks
        assert self.threads % 32 == 0 and self.threads <= 1024
        self.chunks = chunks
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mO: cute.Tensor, stream):
        vec = (1, _VEC)
        self.kernel(
            cute.zipped_divide(mA, vec), cute.zipped_divide(mB, vec), cute.zipped_divide(mO, vec)
        ).launch(
            grid=[mO.shape[0], self.chunks, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(self, gA: cute.Tensor, gB: cute.Tensor, gO: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        row, chunk, _ = cute.arch.block_idx()
        col = chunk * self.threads + tidx
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        a = gA[(None, (row, col))].load().to(Float32)
        b = gB[(None, (row, col))].load().to(Float32)
        gO[(None, (row, col))].store((a + b).to(BFloat16))
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


def _fake(dtype, cols):
    return cute.runtime.make_fake_compact_tensor(
        dtype, (cute.sym_int(), cols), stride_order=(1, 0), assumed_align=16
    )


@functools.lru_cache(maxsize=None)
def _compiled_norm(
    hidden: int,
    rows_per_cta: int,
    enable_pdl: bool,
    fgs_consumer: bool = False,
    fgs_producer: bool = False,
    fgs_max_spins: int = 1 << 16,
    fgs_check: bool = True,
):
    weight = cute.runtime.make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16)
    return cute.compile(
        RMSNormKernel(
            hidden, rows_per_cta, enable_pdl, fgs_consumer, fgs_producer, fgs_max_spins, fgs_check
        ),
        _fake(BFloat16, hidden),
        weight,
        _fake(BFloat16, hidden),
        Int32(1),
        Float32(1e-6),
        cute.runtime.make_fake_compact_tensor(Int32, (4,), assumed_align=16),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@functools.lru_cache(maxsize=None)
def _compiled_add(hidden: int, chunks: int, enable_pdl: bool):
    return cute.compile(
        AddKernel(hidden, chunks, enable_pdl),
        _fake(BFloat16, hidden),
        _fake(BFloat16, hidden),
        _fake(BFloat16, hidden),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _check(*tensors):
    for t in tensors:
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.data_ptr() % 16:
            raise ValueError("latent: operands must be contiguous 16-byte aligned bf16")


def default_norm_rows(m: int) -> int:
    override = os.environ.get("TLLM_K3_LATENT_NORM_ROWS")
    return int(override) if override else 2


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    rows_per_cta: int = 0,
    *,
    fgs_consumer: bool = False,
    fgs_producer: bool = False,
    fgs_counters: Optional[torch.Tensor] = None,
    fgs_max_spins: int = 1 << 16,
    fgs_check: bool = True,
) -> torch.Tensor:
    """FlashInfer-identical RMSNorm of x [M, H] bf16 into ``out``.

    ``fgs_consumer``: x is a sentinel slot written by a running producer (counters: ``make_fgs_counters``);
    ``fgs_producer``: ``out`` feeds a sentinel consumer (see :class:`RMSNormKernel`).
    """
    _check(x, weight, out)
    m, h = x.shape
    if fgs_consumer and (fgs_counters is None or fgs_counters.numel() != 4):
        raise ValueError("rmsnorm: fgs_consumer needs int32[4] fgs_counters")
    if m:
        fn = _compiled_norm(
            h,
            rows_per_cta or default_norm_rows(m),
            _pdl() or fgs_consumer or fgs_producer,
            fgs_consumer,
            fgs_producer,
            fgs_max_spins if fgs_consumer else 1 << 16,
            fgs_check or not fgs_consumer,
        )
        fn(x, weight, out, m, eps, fgs_counters if fgs_consumer else _scratch_counters(x.device))
    return out


_SCRATCH: dict = {}


def _scratch_counters(device: torch.device) -> torch.Tensor:
    if device not in _SCRATCH:
        _SCRATCH[device] = torch.zeros(4, dtype=torch.int32, device=device)
    return _SCRATCH[device]


def add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, chunks: int = 7) -> torch.Tensor:
    _check(a, b, out)
    if a.shape[0]:
        _compiled_add(a.shape[1], chunks, _pdl())(a, b, out)
    return out


# Measured on SM107 (CUDA-graph replay, ready input): (max M, (mma_m, mma_n, split_k, ab_stages)).
_UP_TACTICS = (
    (1, (128, 8, 2, 6)),
    (8, (64, 8, 1, 11)),
    (16, (64, 16, 1, 10)),
    (32, (64, 32, 1, 9)),
)


def up_tactic(m: int):
    """Split-K tactic of the latent up-projection at M rows (``TLLM_K3_LATENT_UP_TACTIC=m,n,split,stages``
    overrides)."""
    from ..splitk_gemm_fgs import SplitKTactic

    override = os.environ.get("TLLM_K3_LATENT_UP_TACTIC")
    if override:
        return SplitKTactic(*(int(v) for v in override.split(",")))
    return SplitKTactic(*next(t for max_m, t in _UP_TACTICS if m <= max_m))


def up_proj(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    addend: Optional[torch.Tensor] = None,
    *,
    fgs_consumer: bool = False,
) -> torch.Tensor:
    """``out = bf16(x @ weight^T)`` (+ ``addend`` after the rounding); x [M, 3584], weight [7168, 3584].

    ``fgs_consumer`` (M2d-2, M <= 32): x is a sentinel slot written by a running ``rmsnorm(fgs_producer=True)``; the
    split-K report_valid build re-loads activation stages that still hold -0.0.
    """
    m = x.shape[0]
    if m == 0:
        return out
    if fgs_consumer and m > _SPLITK_ROWS:
        raise ValueError(f"up_proj: the FGS consumer serves at most {_SPLITK_ROWS} rows")
    if m > _SPLITK_ROWS:
        if addend is None:
            return torch.matmul(x, weight.t(), out=out)
        return torch.add(torch.matmul(x, weight.t()), addend, out=out)
    from ..splitk_gemm_fgs import run_splitk_dense_fgs

    run_splitk_dense_fgs(
        x,
        weight.t(),
        addend,
        out,
        _pdl(),
        up_tactic(m),
        report_valid=fgs_consumer,
        retry_warp=fgs_consumer,
        pdl_wait_activation=not fgs_consumer,
        addend=addend is not None,
    )
    return out


def run(
    y: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_weight: torch.Tensor,
    shared: Optional[torch.Tensor],
    out: torch.Tensor,
    norm_out: torch.Tensor,
) -> torch.Tensor:
    """MoE tail: ``out = up_proj(rmsnorm(y)) (+ shared)``; ``norm_out`` [M, 3584] holds the normalized rows."""
    rmsnorm(y, norm_weight, eps, norm_out)
    return up_proj(norm_out, up_weight, out, shared)
