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
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from ..moe_chain import HIDDEN, TOP_K
from .sync import release_add, spin_ns, wait_at_least_short

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
    """``wait_fc2`` (fc2_done consumer): no ``griddepcontrol.wait`` before the reads. One lane per k waits for
    ``fc2_done[p // tile_n] >= fc2_expected`` and a CTA barrier orders every thread's ``y`` loads after those
    acquires; the kernel waits for the producer grid only at its end, so its completion still implies FC2's.
    ``signal`` (fin_done producer): dependents launch at entry, and after a row chunk is stored one thread adds 1 to
    ``fin_done[row]`` (release), so a row is final at ``chunks``. Every CTA has started once the dependents launch,
    so a dependent spinning on fin_done never waits for a CTA that is not resident."""

    def __init__(
        self,
        hidden: int,
        chunks: int,
        enable_pdl: bool,
        wait_fc2: bool = False,
        fc2_expected: int = 0,
        signal: bool = False,
        delay_ns: int = 0,
    ):
        vecs = hidden // _VEC
        assert hidden % _VEC == 0 and vecs % chunks == 0, (hidden, chunks)
        assert not (wait_fc2 or signal) or enable_pdl, "the FGS builds launch with PDL"
        self.threads = vecs // chunks
        assert self.threads % 32 == 0 and self.threads <= 1024, self.threads
        self.chunks = chunks
        self.enable_pdl = enable_pdl
        self.wait_fc2 = wait_fc2
        self.fc2_expected = fc2_expected
        self.signal = signal
        self.delay_ns = delay_ns

    @cute.jit
    def __call__(
        self,
        mY: cute.Tensor,
        mW: cute.Tensor,
        mE: cute.Tensor,
        mO: cute.Tensor,
        mFc2: cute.Tensor,
        mFin: cute.Tensor,
        tile_n: Int32,
        rows: Int32,
        stream,
    ):
        self.kernel(
            cute.zipped_divide(mY, (1, _VEC)),
            cute.zipped_divide(mW, (1, TOP_K)),
            cute.zipped_divide(mE, (1, TOP_K)),
            cute.zipped_divide(mO, (1, _VEC)),
            mFc2,
            mFin,
            tile_n,
        ).launch(
            grid=[rows, self.chunks, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        gY: cute.Tensor,
        gW: cute.Tensor,
        gE: cute.Tensor,
        gO: cute.Tensor,
        mFc2: cute.Tensor,
        mFin: cute.Tensor,
        tile_n: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, chunk, _ = cute.arch.block_idx()
        col = chunk * self.threads + tidx
        if cutlass.const_expr(self.signal):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.enable_pdl and not self.wait_fc2):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(not self.signal):
                cute.arch.griddepcontrol_launch_dependents()

        e = cute.make_rmem_tensor(gE[(None, (row, 0))].shape, Int32)
        e.store(gE[(None, (row, 0))].load())
        w = cute.make_rmem_tensor(gW[(None, (row, 0))].shape, BFloat16)
        w.store(gW[(None, (row, 0))].load())
        if cutlass.const_expr(self.wait_fc2):
            if tidx < TOP_K:
                p_lane = gE[((0, tidx), (row, 0))]
                if p_lane >= 0:
                    wait_at_least_short(
                        (mFc2.iterator + p_lane // tile_n).toint(), Int32(self.fc2_expected)
                    )
            cute.arch.barrier()
        acc = cute.make_rmem_tensor(gO[(None, (row, col))].shape, Float32)
        acc.fill(0.0)
        for k in cutlass.range_constexpr(TOP_K):
            p = e[k]
            if p >= 0:
                acc.store(acc.load() + gY[(None, (p, col))].load().to(Float32) * Float32(w[k]))
        if cutlass.const_expr(self.delay_ns > 0):
            if row % 3 == 0:
                spin_ns(Int64(self.delay_ns))
        gO[(None, (row, col))].store(acc.load().to(BFloat16))

        if cutlass.const_expr(self.signal):
            cute.arch.barrier()
            if tidx == 0:
                release_add((mFin.iterator + row).toint(), Int32(1))
        if cutlass.const_expr(self.wait_fc2):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(not self.signal):
                cute.arch.griddepcontrol_launch_dependents()


@functools.lru_cache(maxsize=None)
def _compiled(
    hidden: int,
    chunks: int,
    enable_pdl: bool,
    wait_fc2: bool = False,
    fc2_expected: int = 0,
    signal: bool = False,
    delay_ns: int = 0,
):
    kernel = FinalizeKernel(hidden, chunks, enable_pdl, wait_fc2, fc2_expected, signal, delay_ns)

    def fake(dtype, cols):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (cute.sym_int(), cols), stride_order=(1, 0), assumed_align=16
        )

    def fake_words():
        return cute.runtime.make_fake_compact_tensor(Int32, (cute.sym_int(),), assumed_align=4)

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        fake(BFloat16, hidden),
        fake(BFloat16, TOP_K),
        fake(Int32, TOP_K),
        fake(BFloat16, hidden),
        fake_words(),
        fake_words(),
        Int32(1),
        Int32(1),
        stream,
        options="--enable-tvm-ffi",
    )


def run(
    y: torch.Tensor,
    topk_w: torch.Tensor,
    plan,
    out: torch.Tensor,
    chunks: int = 0,
    *,
    fc2_done: Optional[torch.Tensor] = None,
    fc2_expected: int = 28,
    fin_done: Optional[torch.Tensor] = None,
    hazard_delay_ns: int = 0,
) -> None:
    """``moe_chain`` finalize stage: y bf16 [P, 3584] (permuted rows), topk_w bf16 [R, 16], out bf16 [R, 3584].

    ``fc2_done`` (int32, one word per tile_n-row tile) makes it the FC2 -> finalize consumer; ``fin_done`` (int32,
    one word per row of ``out``) the finalize -> combine producer. Both are zero-armed counter slabs owned by the
    caller. ``hazard_delay_ns`` delays the stores of every third row of the producer build (hazard tests only).
    """
    rows = out.shape[0]
    if rows == 0:
        return
    e2p = plan.expanded_idx_to_permuted_idx[: rows * TOP_K].view(rows, TOP_K)
    for t in (y, topk_w, e2p, out):
        if not t.is_contiguous() or t.data_ptr() % 16:
            raise ValueError("finalize: operands must be contiguous and 16-byte aligned")
    if y.shape[1] != HIDDEN or out.shape[1] != HIDDEN or topk_w.shape != (rows, TOP_K):
        raise ValueError(f"finalize: bad shapes y {tuple(y.shape)} topk_w {tuple(topk_w.shape)}")
    if fin_done is not None and fin_done.numel() < rows:
        raise ValueError("finalize: fin_done needs one word per row")
    chunks = chunks or default_chunks(rows)
    wait_fc2, signal = fc2_done is not None, fin_done is not None
    fn = _compiled(
        HIDDEN,
        chunks,
        _pdl() or wait_fc2 or signal,
        wait_fc2,
        fc2_expected if wait_fc2 else 0,
        signal,
        hazard_delay_ns if signal else 0,
    )
    words = plan.num_non_exiting_tiles
    fn(
        y,
        topk_w,
        e2p,
        out,
        (fc2_done if wait_fc2 else words).view(-1),
        (fin_done if signal else words).view(-1),
        plan.tile_n,
        rows,
    )
