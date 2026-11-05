# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE finalize of the Kimi K3 decode expert path as a CuTe DSL kernel.

``out[r] = bf16(sum_{k = 0..15, p = e2p[r * 16 + k] >= 0} f32(topk_w[r, k]) * f32(y[p]))`` for every received row r,
zero for rows without a local pair. The sum runs in k order in fp32 with one rounding, as the TRTLLM-Gen
``finalizeKernelVecLoad``: a bf16 x bf16 product is exact in fp32, so the result does not depend on FMA contraction
and matches both that kernel and the fixed-order torch oracle bit for bit.

One CTA per (row, column chunk); each thread owns one 16 B vector of the row. Rows of ``y`` that no (row, k) maps to
(padding, other ranks' rows) are never read.
"""

from __future__ import annotations

import functools
import os

import torch

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32

from ..moe_chain import HIDDEN, TOP_K

_VEC = 8


def _pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def default_chunks(rows: int) -> int:
    """Column chunks per row (CTAs per row); threads per CTA = HIDDEN / 8 / chunks."""
    override = os.environ.get("TLLM_K3_MOE_FIN_CHUNKS")
    if override:
        return int(override)
    return 2


class FinalizeKernel:
    def __init__(self, hidden: int, chunks: int, enable_pdl: bool):
        vecs = hidden // _VEC
        assert hidden % _VEC == 0 and vecs % chunks == 0, (hidden, chunks)
        self.threads = vecs // chunks
        assert self.threads % 32 == 0 and self.threads <= 1024, self.threads
        self.chunks = chunks
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self, mY: cute.Tensor, mW: cute.Tensor, mE: cute.Tensor, mO: cute.Tensor, stream
    ):
        self.kernel(
            cute.zipped_divide(mY, (1, _VEC)),
            cute.zipped_divide(mW, (1, TOP_K)),
            cute.zipped_divide(mE, (1, TOP_K)),
            cute.zipped_divide(mO, (1, _VEC)),
        ).launch(
            grid=[mO.shape[0], self.chunks, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(self, gY: cute.Tensor, gW: cute.Tensor, gE: cute.Tensor, gO: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        row, chunk, _ = cute.arch.block_idx()
        col = chunk * self.threads + tidx
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        e = cute.make_rmem_tensor(gE[(None, (row, 0))].shape, Int32)
        e.store(gE[(None, (row, 0))].load())
        w = cute.make_rmem_tensor(gW[(None, (row, 0))].shape, BFloat16)
        w.store(gW[(None, (row, 0))].load())
        acc = cute.make_rmem_tensor(gO[(None, (row, col))].shape, Float32)
        acc.fill(0.0)
        for k in cutlass.range_constexpr(TOP_K):
            p = e[k]
            if p >= 0:
                acc.store(acc.load() + gY[(None, (p, col))].load().to(Float32) * Float32(w[k]))
        gO[(None, (row, col))].store(acc.load().to(BFloat16))


@functools.lru_cache(maxsize=None)
def _compiled(hidden: int, chunks: int, enable_pdl: bool):
    kernel = FinalizeKernel(hidden, chunks, enable_pdl)

    def fake(dtype, cols):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (cute.sym_int(), cols), stride_order=(1, 0), assumed_align=16
        )

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        fake(BFloat16, hidden),
        fake(BFloat16, TOP_K),
        fake(Int32, TOP_K),
        fake(BFloat16, hidden),
        stream,
        options="--enable-tvm-ffi",
    )


def run(y: torch.Tensor, topk_w: torch.Tensor, plan, out: torch.Tensor, chunks: int = 0) -> None:
    """``moe_chain`` finalize stage: y bf16 [P, 3584] (permuted rows), topk_w bf16 [R, 16], out bf16 [R, 3584]."""
    rows = out.shape[0]
    if rows == 0:
        return
    e2p = plan.expanded_idx_to_permuted_idx[: rows * TOP_K].view(rows, TOP_K)
    for t in (y, topk_w, e2p, out):
        if not t.is_contiguous() or t.data_ptr() % 16:
            raise ValueError("finalize: operands must be contiguous and 16-byte aligned")
    if y.shape[1] != HIDDEN or out.shape[1] != HIDDEN or topk_w.shape != (rows, TOP_K):
        raise ValueError(f"finalize: bad shapes y {tuple(y.shape)} topk_w {tuple(topk_w.shape)}")
    _compiled(HIDDEN, chunks or default_chunks(rows), _pdl())(y, topk_w, e2p, out)
