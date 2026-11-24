# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The RMSNorm stage follows FlashInfer v0.6.18 flashinfer/norm/kernels/rmsnorm.py (RMSNormKernel, Apache-2.0,
# Copyright (c) 2025 by FlashInfer team), as in latent.py.
"""Kimi K3 decode MoE: the combine reduce fused with the latent RMSNorm, in CuTe DSL.

One CTA per local token waits for its k combine contributions in the receive inbox (the same slot counters and
completion flags as ``CuteA2A.combine``), sums them in the stock order (``comm._Reduce``), rounds the row to bf16 in
shared memory and normalizes it with the per-thread columns, fragment and reduction order of ``latent.rmsnorm``. The
output is bitwise ``latent.rmsnorm(CuteA2A.combine(y))``; the combined row never reaches global memory.

The contributions arrive as bf16 (the CuTe combine) or as e4m3 (the low-precision combine, unscaled
``cvt.rn.satfinite.e4m3x2.bf16x2`` per element, summed in the same order after an exact widening).

With ``lamport="tma"`` the contributions come from the combine's Lamport inbox (``CuteA2A.combine_push``) instead of
the counted-write receive region: each remote row lands in shared memory with one TMA load that reports the sentinel
check (``report_valid``, element 0 of each 16 B against the wire dtype's -0, which needs a push that lands whole 16 B
chunks), and a row that landed with sentinel data is re-issued. The consumer stores the sentinel back over what it
read. The push does the round duties, so this kernel leaves the round flag alone. Own rows are read from the bf16
payload (cast to e4m3 first for the low-precision combine). Dependents launch at entry.
"""

from __future__ import annotations

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from . import comm as _comm
from .comm import (
    _COUNTER_STRIDE,
    _SIDE_INBOX,
    _asm,
    _bf16x2_to_f32,
    _bf16x4_to_e4m3x4,
    _f32x2_to_bf16x2,
    _fence_fabric_acquire,
    _group8,
    _ld_b32,
    _ld_b64,
    _ld_v4,
    _ld_vol_v2,
    _ld_vol_v4,
    _spin_eq,
    _spin_ge64,
    _st_b64,
    _st_v2,
    _st_v4,
    _v,
)
from .latent import _THREADS_PER_ROW, _VEC

_THREADS = 256
_NORM_BARRIER = 1
_SMEM_CAPACITY = 227 * 1024
WIRE_DTYPES = ("bf16", "e4m3")
LAMPORT_WAITS = {"": 0, "tma": 3}
# Inner TMA box of an inbox row (bytes); the row is (_TMA_CHUNK, row_bytes // _TMA_CHUNK).
_TMA_CHUNK = 128
# Kinds of a contribution's source row.
_SRC_INBOX, _SRC_OWN, _SRC_LAMPORT = 0, 1, 2


@dsl_user_op
def _ld_v2(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_v(addr, Int64, loc, ip)],
        "ld.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc,
        ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def _ld_shared_v2(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_v(addr, Int32, loc, ip)],
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        loc,
        ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def _ld_shared_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_v(addr, Int32, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _e4m3x4_to_f32(w, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_v(w, Int32, loc, ip)],
        "{\n.reg .b16 l, h;\n.reg .b32 p, q;\n.reg .f16 a, b, c, d;\nmov.b32 {l, h}, $4;\n"
        "cvt.rn.f16x2.e4m3x2 p, l;\ncvt.rn.f16x2.e4m3x2 q, h;\nmov.b32 {a, b}, p;\nmov.b32 {c, d}, q;\n"
        "cvt.f32.f16 $0, a;\ncvt.f32.f16 $1, b;\ncvt.f32.f16 $2, c;\ncvt.f32.f16 $3, d;\n}",
        "=f,=f,=f,=f,r",
        loc,
        ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


class ReduceNormKernel:
    """``out[t] = rmsnorm(bf16(sum_k contribution_k(t)))`` for each local token t, one CTA per token.

    ``lamport`` (``LAMPORT_WAITS``) reads the Lamport inbox of ``rows`` rows per source and ``inbox_bytes`` per
    parity instead of the receive region (see the module docstring).
    """

    def __init__(
        self,
        ep: int,
        top_k: int,
        hidden: int,
        lay: tuple,
        pdl: bool,
        wire: str,
        lamport: int = 0,
        rows: int = 0,
        inbox_bytes: int = 0,
    ):
        assert top_k <= 32 and wire in WIRE_DTYPES
        assert hidden % (_VEC * _THREADS_PER_ROW) == 0
        self.ep, self.top_k, self.H, self.lay = ep, top_k, hidden, dict(lay)
        self.pdl, self.wire = pdl, wire
        self.lamport, self.rows, self.inbox_bytes = lamport, rows, inbox_bytes
        self.tma = lamport == 3
        lpc = wire == "e4m3"
        self.row_bytes = hidden * (1 if lpc else 2)
        self.sentinel_word = -0x7F7F7F80 if lpc else -0x7FFF8000
        self.pattern = (
            cpasync.SentinelPattern.PER_16BYTES_80
            if lpc
            else cpasync.SentinelPattern.PER_16BYTES_8000
        )
        self.blocks = hidden // (_VEC * _THREADS_PER_ROW)
        self.smem_bytes = 32 * 8 + 32 * 4 + 16 + 2 * 4 + hidden * 2 + 4 * 16
        if self.tma:
            assert self.row_bytes % _TMA_CHUNK == 0 and inbox_bytes == ep * rows * self.row_bytes
            self.smem_bytes += 32 * 4 + top_k * 8 + 256 + top_k * self.row_bytes
        assert self.smem_bytes <= _SMEM_CAPACITY, f"reduce_norm: {self.smem_bytes} B of smem"

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mW: cute.Tensor,
        inbox_ptr: cute.Pointer,
        ws_base: Int64,
        ws_stride: Int64,
        rank: Int32,
        n_local: Int32,
        max_tokens: Int32,
        static_max: Int32,
        timeout_ns: Int64,
        own_base: Int64,
        side_base: Int64,
        side_stride: Int64,
        eps: Float32,
        stream,
    ):
        tv_layout = cute.make_layout(
            ((_THREADS_PER_ROW, 1), (_VEC, self.blocks)),
            stride=((_VEC, 1), (1, _VEC * _THREADS_PER_ROW)),
        )
        tiler_mn = (1, self.H)
        tma_atom = tma_tensor = None
        if cutlass.const_expr(self.tma):
            # Both parities of this rank's inbox as (chunk byte, chunk, row): rows of parity b start at
            # b * ep * rows (inbox_bytes == ep * rows * row_bytes).
            chunks = self.row_bytes // _TMA_CHUNK
            m_inbox = cute.make_tensor(
                inbox_ptr,
                cute.make_layout(
                    (_TMA_CHUNK, chunks, 2 * self.ep * self.rows),
                    stride=(1, _TMA_CHUNK, self.row_bytes),
                ),
            )
            tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                m_inbox,
                cute.make_layout((_TMA_CHUNK, chunks), stride=(1, _TMA_CHUNK)),
                (_TMA_CHUNK, chunks),
            )
        self.kernel(
            mOut,
            mW,
            tma_atom,
            tma_tensor,
            ws_base,
            ws_stride,
            rank,
            n_local,
            max_tokens,
            static_max,
            timeout_ns,
            own_base,
            side_base,
            side_stride,
            eps,
            tv_layout,
            tiler_mn,
        ).launch(
            grid=[cutlass.max(n_local, Int32(1)), 1, 1],
            block=[_THREADS, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mW: cute.Tensor,
        tma_atom: Optional[cute.CopyAtom],
        mInbox: Optional[cute.Tensor],
        ws_base: Int64,
        ws_stride: Int64,
        rank: Int32,
        n_local: Int32,
        max_tokens: Int32,
        static_max: Int32,
        timeout_ns: Int64,
        own_base: Int64,
        side_base: Int64,
        side_stride: Int64,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        L = self.lay
        K = self.top_k
        tidx, _, _ = cute.arch.thread_idx()
        tok, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        smem = SmemAllocator()
        s_src = smem.allocate_tensor(Int64, cute.make_layout((32,)), byte_alignment=16)
        s_kind = smem.allocate_tensor(Int32, cute.make_layout((32,)), byte_alignment=16)
        s_n = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
        partials = smem.allocate_tensor(Float32, cute.make_layout((1, 2)), byte_alignment=16)
        sX = smem.allocate_tensor(
            BFloat16, cute.make_layout(tiler_mn, stride=(self.H, 1)), byte_alignment=16
        )
        ring = Int32(0)
        s_row = bars = tS = tG = None
        if cutlass.const_expr(self.tma):
            s_row = smem.allocate_tensor(Int32, cute.make_layout((32,)), byte_alignment=16)
            bars = smem.allocate_array(Int64, K)
            chunks = self.row_bytes // _TMA_CHUNK
            sRing = smem.allocate_tensor(  # codespell:ignore
                cutlass.Uint8,
                cute.make_layout((_TMA_CHUNK, chunks, K), stride=(1, _TMA_CHUNK, self.row_bytes)),
                byte_alignment=128,
            )
            ring = Int32(sRing.iterator.toint())  # codespell:ignore
            gIn = cute.local_tile(mInbox, (_TMA_CHUNK, chunks), (0, 0, None))
            tS, tG = cpasync.tma_partition(
                tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sRing, 0, 2),  # codespell:ignore
                cute.group_modes(gIn, 0, 2),
            )
            if warp == 0:
                cpasync.prefetch_descriptor(tma_atom)
                with cute.arch.elect_one():
                    for s in cutlass.range_constexpr(K):
                        cute.arch.mbarrier_v1_init(bars + s, 1)
                cute.arch.mbarrier_init_fence()
        my_ws = ws_base + Int64(rank) * ws_stride
        inbox = my_ws + L["combine_recv"]

        # Norm threads are 0..63; every thread builds the partition of tidx % 64 so the weight fragment can be
        # loaded before the dependency wait.
        ntid = tidx % _THREADS_PER_ROW
        copy_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=_VEC * 16
        )
        thr = cute.make_tiled_copy(copy_x, tv_layout, tiler_mn).get_slice(ntid)
        mW_2d = cute.make_tensor(
            mW.iterator, cute.prepend(mW.layout, cute.make_layout((1,), stride=(0,)))
        )
        tWgW = thr.partition_S(cute.local_tile(mW_2d, tiler_mn, (0, 0)))
        tWrW = cute.make_fragment_like(tWgW)
        cute.copy(copy_x, tWgW, tWrW)
        tXsX = thr.partition_S(cute.local_tile(sX, tiler_mn, (0, 0)))
        tXrX = cute.make_fragment_like(tXsX)
        tXgO = thr.partition_D(cute.local_tile(mOut, tiler_mn, (tok, 0)))
        tXrO = cute.make_fragment_like(tXgO)

        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        lam_inbox = Int64(0)
        parity = Int32(0)
        if cutlass.const_expr(self.lamport > 0):
            f = _ld_b32(my_ws + L["flag"])
            parity = ((f - 1) >> 1) & 1
            my_side = side_base + Int64(rank) * side_stride
            lam_inbox = my_side + _SIDE_INBOX + Int64(parity) * self.inbox_bytes
        if n_local > 0:
            if warp == 0:
                tr = Int32(-1)
                di = Int32(-1)
                if lane < K:
                    rec = Int64(tok) * K + lane
                    tr = _ld_b32(my_ws + L["topk_ranks"] + rec * 4)
                    di = _ld_b32(my_ws + L["topk_indices"] + rec * 4)
                    if cutlass.const_expr(self.lamport > 0):
                        if tr < 0:
                            di = Int32(-1)
                    else:
                        if tr >= 0:
                            if di >= 0:
                                if tr != rank:
                                    slot = Int64(tr) * static_max + di
                                    base_addr = my_ws + L["combine_baseline"] + slot * 8
                                    target = _ld_b64(base_addr) + self.row_bytes
                                    _spin_ge64(
                                        my_ws + L["combine_counters"] + slot * _COUNTER_STRIDE,
                                        target,
                                        timeout_ns,
                                    )
                                    _st_b64(base_addr, target)
                if cutlass.const_expr(self.lamport == 0):
                    _fence_fabric_acquire()
                valid = cute.arch.vote_ballot_sync(di >= 0)
                if lane == 0:
                    s_n[0] = cute.arch.popc(valid)
                if di >= 0:
                    at = cute.arch.popc(valid & ((Int32(1) << lane) - 1))
                    row_src = inbox + (Int64(tr) * max_tokens + di) * self.row_bytes
                    kind = Int32(_SRC_INBOX)
                    if cutlass.const_expr(self.lamport > 0):
                        row_src = lam_inbox + (Int64(tr) * self.rows + di) * self.row_bytes
                        kind = Int32(_SRC_LAMPORT)
                    if cutlass.const_expr(self.lamport > 0):
                        if tr == rank:
                            row_src = own_base + (Int64(rank) * max_tokens + di) * (self.H * 2)
                            kind = Int32(_SRC_OWN)
                    s_src[at] = row_src
                    s_kind[at] = kind
                    if cutlass.const_expr(self.tma):
                        row = Int32(-1)
                        if kind == _SRC_LAMPORT:
                            row = parity * (self.ep * self.rows) + tr * self.rows + di
                        s_row[at] = row
                if cutlass.const_expr(self.tma):
                    cute.arch.sync_warp()
                    count0 = cute.arch.popc(valid)
                    pending = Int32(0)
                    for s in cutlass.range(count0, unroll=1):
                        row = cute.arch.make_warp_uniform(s_row[s])
                        if row >= 0:
                            pending = pending | (Int32(1) << s)
                            self._issue(tma_atom, tS, tG, bars, s, row)
                    self._land(tma_atom, tS, tG, bars, s_row, count0, pending)
            cute.arch.sync_threads()
            # Threads load slots warp 0 waited for: each fences after the barrier.
            if cutlass.const_expr(self.lamport == 0):
                _fence_fabric_acquire()

            count = s_n[0]
            self._reduce_row(sX, s_src, s_kind, count, ring)
            cute.arch.sync_threads()

            if tidx < _THREADS_PER_ROW:
                cute.copy(copy_x, tXsX, tXrX)
                x = tXrX.load().to(Float32)
                local = (x * x).reduce(
                    cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
                )
                for i in cutlass.range_constexpr(5):
                    local = local + cute.arch.shuffle_sync_bfly(local, offset=1 << i)
                if lane == 0:
                    partials[0, warp] = local
                cute.arch.barrier(barrier_id=_NORM_BARRIER, number_of_threads=_THREADS_PER_ROW)
                sum_sq = partials[0, 0] + partials[0, 1]
                rstd = cute.math.rsqrt(sum_sq / Float32(self.H) + eps, fastmath=True)
                w = tWrW.load().to(Float32)
                y = x * rstd * (w + Float32(0.0))
                tXrO.store(y.to(BFloat16))
                cute.copy(copy_x, tXrO, tXgO)

        if cutlass.const_expr(self.lamport == 0):
            if tok == 0:
                if warp == 0:
                    flag = _ld_b32(my_ws + L["flag"])
                    if lane < self.ep:
                        _spin_eq(my_ws + L["combine_flags"] + lane * 4, flag, timeout_ns)

    @cute.jit
    def _issue(self, tma_atom, tS, tG, bars, s, row):
        """Load inbox row ``row`` into landing slot ``s`` on its barrier, with the sentinel check."""
        # The TMA copy elects its issuing thread itself; only the arrive is ours to elect.
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bars + s, self.row_bytes)
        cute.copy(
            tma_atom,
            tG[(None, row)],
            tS[(None, s)],
            tma_bar_ptr=bars + s,
            report_valid=self.pattern,
        )

    @cute.jit
    def _land(self, tma_atom, tS, tG, bars, s_row, count, pending):
        """Wait until every slot of ``pending`` landed valid; a slot that landed with sentinel data is re-issued.
        Bit s of ``phase`` is the primary-phase parity slot s completes next. Returns the re-issue count."""
        phase = Int32(0)
        retries = Int32(0)
        while pending != 0:
            for s in cutlass.range(count, unroll=1):
                bit = Int32(1) << s
                if (pending & bit) != 0:
                    complete, invalid = cute.arch.mbarrier_try_wait_primary(
                        bars + s, (phase >> s) & 1
                    )
                    if complete:
                        phase = phase ^ bit
                        if invalid:
                            retries = retries + 1
                            self._issue(
                                tma_atom, tS, tG, bars, s, cute.arch.make_warp_uniform(s_row[s])
                            )
                        else:
                            pending = pending ^ bit
        return retries

    @cute.jit
    def _reduce_row(
        self,
        sX: cute.Tensor,
        s_src: cute.Tensor,
        s_kind: cute.Tensor,
        count: Int32,
        ring: Int32,
    ):
        """Row sum of the ``count`` contributions into ``sX`` (bf16), in the order of ``comm._Reduce``."""
        tidx, _, _ = cute.arch.thread_idx()
        K = self.top_k
        lpc = self.wire == "e4m3"
        sentinel = Int32(self.sentinel_word)
        ns = (K + 7) // 8 * 8
        words = cute.recast_tensor(sX, Int32)
        a = cute.make_rmem_tensor(cute.make_layout((8, ns)), Float32)
        raw = cute.make_rmem_tensor(cute.make_layout((4, ns)), Int32)
        res = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        for v in cutlass.range(tidx, self.H // _VEC, _THREADS, unroll=1):
            a.fill(0.0)
            raw.fill(0)
            for s in cutlass.range_constexpr(ns):
                if s < count:
                    kind = s_kind[s]
                    if kind == _SRC_OWN:
                        w = _ld_v4(s_src[s] + v * 16)
                        if cutlass.const_expr(lpc):
                            raw[0, s] = _bf16x4_to_e4m3x4(w[0], w[1])
                            raw[1, s] = _bf16x4_to_e4m3x4(w[2], w[3])
                        else:
                            for q in cutlass.range_constexpr(4):
                                raw[q, s] = w[q]
                    elif kind == _SRC_LAMPORT:
                        if cutlass.const_expr(self.tma):
                            if cutlass.const_expr(lpc):
                                e = _ld_shared_v2(ring + s * self.row_bytes + v * 8)
                                raw[0, s] = e[0]
                                raw[1, s] = e[1]
                            else:
                                w = _ld_shared_v4(ring + s * self.row_bytes + v * 16)
                                for q in cutlass.range_constexpr(4):
                                    raw[q, s] = w[q]
                        elif cutlass.const_expr(lpc):
                            e = _ld_vol_v2(s_src[s] + v * 8)
                            raw[0, s] = e[0]
                            raw[1, s] = e[1]
                        else:
                            w = _ld_vol_v4(s_src[s] + v * 16)
                            for q in cutlass.range_constexpr(4):
                                raw[q, s] = w[q]
                    else:
                        if cutlass.const_expr(lpc):
                            e = _ld_v2(s_src[s] + v * 8)
                            raw[0, s] = e[0]
                            raw[1, s] = e[1]
                        else:
                            w = _ld_v4(s_src[s] + v * 16)
                            for q in cutlass.range_constexpr(4):
                                raw[q, s] = w[q]
            if cutlass.const_expr(self.lamport > 0):
                for s in cutlass.range_constexpr(ns):
                    if s < count:
                        if s_kind[s] == _SRC_LAMPORT:
                            if cutlass.const_expr(lpc):
                                addr = s_src[s] + v * 8
                                _st_v2(addr, sentinel, sentinel)
                            else:
                                addr = s_src[s] + v * 16
                                _st_v4(addr, sentinel, sentinel, sentinel, sentinel)
            for s in cutlass.range_constexpr(ns):
                if cutlass.const_expr(lpc):
                    for h in cutlass.range_constexpr(2):
                        f = _e4m3x4_to_f32(raw[h, s])
                        for e in cutlass.range_constexpr(4):
                            a[4 * h + e, s] = f[e]
                else:
                    for q in cutlass.range_constexpr(4):
                        lo, hi = _bf16x2_to_f32(raw[q, s])
                        a[2 * q, s] = lo
                        a[2 * q + 1, s] = hi
            for j in cutlass.range_constexpr(8):
                g = [_group8(a, j, 8 * i) for i in range(ns // 8)]
                res[j] = a[j, 0] + a[j, 1]
                if count > 2:
                    res[j] = g[0]
                if cutlass.const_expr(ns >= 16):
                    if count > 8:
                        res[j] = g[0] + g[1]
                if cutlass.const_expr(ns >= 32):
                    if count > 16:
                        res[j] = (g[0] + g[1]) + (g[2] + g[3])
            for q in cutlass.range_constexpr(4):
                words[0, v * 4 + q] = _f32x2_to_bf16x2(res[2 * q], res[2 * q + 1])


@functools.lru_cache(maxsize=None)
def _compiled(
    ep,
    top_k,
    hidden,
    lay,
    pdl,
    wire,
    lamport,
    rows,
    inbox_bytes,
):
    def fake(dtype, shape):
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(range(len(shape) - 1, -1, -1)), assumed_align=16
        )

    i64, i32 = Int64(0), Int32(0)
    return cute.compile(
        ReduceNormKernel(
            ep,
            top_k,
            hidden,
            lay,
            pdl,
            wire,
            lamport,
            rows,
            inbox_bytes,
        ),
        fake(BFloat16, (cute.sym_int(), hidden)),
        fake(BFloat16, (hidden,)),
        make_ptr(cutlass.Uint8, 1 << 20, cute.AddressSpace.gmem, assumed_align=128),
        i64,
        i64,
        i32,
        i32,
        i32,
        i32,
        i64,
        i64,
        i64,
        i64,
        Float32(1e-6),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def reduce_norm(
    out: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    ws_base: int,
    ws_stride: int,
    lay: tuple,
    ep: int,
    top_k: int,
    rank: int,
    n_local: int,
    max_tokens: int,
    static_max: int,
    timeout_ns: int,
    pdl: bool,
    own_base: int = 0,
    wire: str = "bf16",
    lamport: str = "",
    inbox=None,
) -> torch.Tensor:
    """Launch :class:`ReduceNormKernel` on a combine receive region (``CuteA2A`` workspace layout ``lay``), or with
    ``lamport="tma"`` on the Lamport inbox ``inbox`` (``CuteA2A.lamport_inbox()``)."""
    hidden = out.shape[-1]
    if (
        out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.shape[0] < n_local
        or norm_weight.dtype != torch.bfloat16
        or norm_weight.numel() != hidden
    ):
        raise ValueError(
            f"reduce_norm: out must be contiguous bf16 [>= {n_local}, H], weight bf16 [H]"
        )
    mode = LAMPORT_WAITS[lamport]
    if mode and inbox is None:
        raise ValueError("reduce_norm: the Lamport modes need the inbox descriptor")
    fn = _compiled(
        ep,
        top_k,
        hidden,
        lay,
        pdl,
        wire,
        mode,
        inbox.rows if mode else 0,
        inbox.inbox_bytes if mode else 0,
    )
    my_side = inbox.side_base + rank * inbox.side_stride if mode else 0
    inbox_addr = my_side + _SIDE_INBOX if mode == 3 else 1 << 20
    fn(
        out,
        norm_weight,
        make_ptr(cutlass.Uint8, inbox_addr, cute.AddressSpace.gmem, assumed_align=128),
        ws_base,
        ws_stride,
        rank,
        n_local,
        max_tokens,
        static_max,
        timeout_ns,
        own_base,
        inbox.side_base if mode else 0,
        inbox.side_stride if mode else 0,
        eps,
    )
    return out


def combine_norm(
    a2a,
    final_hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    out: Optional[torch.Tensor] = None,
    *,
    inbox: Optional[str] = None,
) -> torch.Tensor:
    """``CuteA2A.combine`` with its reduce replaced by :class:`ReduceNormKernel`: returns the latent-normed
    ``[local_tokens, hidden]`` rows (``out`` when given). Same push, flags and wire format (bf16, or e4m3 with a
    low-precision combine communicator) as the combine; ``inbox="lamport"`` pushes the data only into the Lamport inbox
    (``CuteA2A.combine_push``) and reads it with the TMA check, storing the sentinel back."""
    y, n, runtime_max = a2a._combine_args(final_hidden_states)
    hidden = y.shape[1]
    if out is None:
        out = torch.empty(n, hidden, dtype=torch.bfloat16, device=y.device)
    wire = "e4m3" if a2a.lpc else "bf16"
    pdl = _comm._pdl()
    ws_base, ws_stride, my_ws = a2a._ws()
    lay = a2a.lay.key()
    common = dict(
        ws_base=ws_base,
        ws_stride=ws_stride,
        lay=lay,
        ep=a2a.ep,
        top_k=a2a.comm.top_k,
        rank=a2a.rank,
        n_local=n,
        max_tokens=runtime_max,
        static_max=a2a.lay.max_tokens,
        timeout_ns=_comm._TIMEOUT_NS,
        own_base=y.data_ptr(),
        wire=wire,
    )
    if inbox == "lamport":
        box = a2a.combine_push(y)
        return reduce_norm(out, norm_weight, eps, pdl=True, lamport="tma", inbox=box, **common)
    if inbox is not None:
        raise ValueError(f"unknown combine inbox {inbox!r}")
    nbytes = hidden * 2
    wire_bytes = hidden if a2a.lpc else nbytes
    if a2a.ep * runtime_max * wire_bytes > a2a.lay.combine_recv_size:
        raise ValueError("combine payload exceeds the receive region")
    _comm._compile("prepare_combine", a2a.ep, wire_bytes, lay, pdl, a2a.lpc)(
        y.view(-1), my_ws, nbytes, a2a.rank, runtime_max
    )
    rows_per_warp = 2
    per_block = rows_per_warp * _comm._PUSH_WARPS
    blocks = min(32, max(1, (runtime_max + per_block - 1) // per_block))
    _comm._compile("push", a2a.ep, wire_bytes, lay, pdl, a2a.lpc)(
        y.view(-1),
        ws_base,
        ws_stride,
        nbytes,
        a2a.rank,
        runtime_max,
        a2a.lay.max_tokens,
        a2a.le_base,
        blocks,
        _comm._TIMEOUT_NS,
    )
    reduce_norm(out, norm_weight, eps, pdl=pdl, **common)
    a2a.comm.reset_state()
    return out
