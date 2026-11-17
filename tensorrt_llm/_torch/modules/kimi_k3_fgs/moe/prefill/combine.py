# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The RMSNorm stage follows FlashInfer v0.6.18 flashinfer/norm/kernels/rmsnorm.py (RMSNormKernel, Apache-2.0,
# Copyright (c) 2025 by FlashInfer team), as in latent.py.
"""Kimi K3 prefill MoE: the combine over a chunk-sized Lamport inbox, fused with the latent RMSNorm, in CuTe DSL.

The expert output of a MoE chunk (``[ep * max_tokens, hidden]`` bf16, the finalize row of every received token) goes
back to the tokens' source ranks with plain stores into a -0 armed inbox (two inboxes by round parity, sized by the
chunk cap): no prepare kernel, no counted writes, no remote loads. The consumer sums the k rows of each local token
in the stock order, rounds once to bf16 and applies the latent RMSNorm; the output is bitwise
``latent.rmsnorm(stock combine)`` except the sign of exact-zero sums (the push turns -0 into +0).

Push (``_Push``, grid (ep, blocks), one warp per row): a warp loads a received row, casts it to the wire format
(e4m3 with the low-precision combine, else bf16) and stores it into ``inbox[parity]`` of the row's source rank at
``(rank * rows + t) * wire``. CTA (source, 0) publishes the push-start flag of the stock protocol, CTA (rank, 0)
waits for every peer's and advances the round flag.

Reduce + norm (``_ReduceNorm``, one CTA per local token, one thread per 8-element vector so every source load of
the token is in flight at once; the norm runs on 64 threads with FlashInfer's thread mapping). FGS on: no wait on
the push grid; each thread re-loads a chunk of a remote row only while an element of it is the sentinel. FGS off:
the grid dependency plus the push-completion words of every CTA of every peer, then plain loads.

Re-arm, i.e. who writes the sentinel back over the rows of a round (the slot's next writer is the push two rounds
later, see the campaign notes): ``consumer`` the reduce, after the token's normed row is stored (one bulk store per
remote row from a sentinel-filled shared row; ``consumer_st`` with per-thread stores); ``producer`` the next round's
push on the receiving rank, over the other parity's rows bounded by the counts the previous push recorded.
"""

from __future__ import annotations

import functools
import os
from typing import NamedTuple, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from .. import comm as _comm
from ..comm import (
    _CACHELINE,
    _TIMEOUT_NS,
    _Layout,
    _asm,
    _bf16x2_to_f32,
    _bf16x4_to_e4m3x4,
    _bulk_commit,
    _bulk_g2s,
    _bulk_s2g,
    _bulk_wait_read,
    _mbar_expect_tx,
    _mbar_init,
    _mbar_wait,
    _e4m3x4_to_f32,
    _f32x2_to_bf16x2,
    _fence_acq_rel_sys,
    _fence_proxy_async_global,
    _fence_proxy_async_shared,
    _globaltimer,
    _ld_b32,
    _ld_v2,
    _ld_v4,
    _ld_vol_v2,
    _ld_vol_v4,
    _nanosleep,
    _sleep_ns,
    _spin_eq,
    _spin_ne,
    _st_any,
    _st_b32,
    _st_shared_v4,
    _st_sys_b32,
    _st_v2,
    _st_v4,
    _trap,
    _v,
)
from ..latent import _THREADS_PER_ROW, _VEC
from ..tail import _st_release_b32

_NORM_BARRIER = 1
_MAX_RANKS = 32
# Side buffer header: done[src][by] (u32), used[parity][src] (u32), then the two inboxes.
_HDR = 256 * 1024
_DONE = 0
_DONE_BLOCKS = 1024
_USED = 128 * 1024
PUSH_WARPS_ENV = "TLLM_K3_PC_PUSH_WARPS"
PUSH_BLOCKS_ENV = "TLLM_K3_PC_PUSH_BLOCKS"
POLL_NS_ENV = "TLLM_K3_PC_POLL_NS"
REDUCE_MIN_BLOCKS_ENV = "TLLM_K3_PC_REDUCE_MIN_BLOCKS"
CHECK_ENV = "TLLM_K3_PC_CHECK"
REARMS = ("consumer", "consumer_st", "producer", "none")
_REARM_NONE, _REARM_BULK, _REARM_ST = 0, 1, 2
_WAIT_POLL, _WAIT_DONE, _WAIT_NONE, _WAIT_PULL = 0, 1, 2, 3
PUSH_ROWS_ENV = "TLLM_K3_PC_PUSH_ROWS"
PUSH_LOAD_ENV = "TLLM_K3_PC_PUSH_LOAD"
MODES = {True: "poll", False: "done", "pull": "pull"}
FGS_WAITS = {"PrefillCombine.reduce_norm": ("recv_norm",)}
FGS_PUBLISHES = {"PrefillCombine.push": ("recv_norm",), "PrefillCombine.reduce_norm": ("rn_up",)}


@dsl_user_op
def _bulk_wait_all(*, loc=None, ip=None) -> None:
    _asm(None, [], "cp.async.bulk.wait_group 0;", "", loc, ip)


def _sentinel(lpc: bool) -> int:
    # 0xFF per e4m3 byte / 0xFFFF per bf16 half: NaN encodings the saturating / RN casts never produce.
    return -1


@dsl_user_op
def _dirty(w, lpc: bool, *, loc=None, ip=None) -> Int32:
    """1 if any element of ``w`` is the all-ones sentinel."""
    one, high = (0x01010101, 0x80808080) if lpc else (0x00010001, 0x80008000)
    return Int32(
        _asm(
            T.i32(),
            [_v(w, Int32, loc, ip)],
            f"{{\n.reg .b32 z, t;\n.reg .pred p;\nnot.b32 z, $1;\nsub.u32 t, z, {one};\nnot.b32 z, z;\n"
            f"and.b32 t, t, z;\nand.b32 t, t, {high};\nsetp.ne.u32 p, t, 0;\nselp.u32 $0, 1, 0, p;\n}}",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_shared_v4(addr, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm

    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_v(addr, Int32, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@cute.jit
def _tma_row(lpc: cutlass.Constexpr, hidden: cutlass.Constexpr, row: Int64, dst: Int64, stage: Int32, mbar: Int32, ph: Int32, lane: Int32) -> Int32:
    """One warp moves a bf16 row to the wire format at ``dst`` through TMA: bulk load onto the warp's mbarrier, the
    lanes convert in place (every lane's reads precede any write), bulk store. Returns the barrier's next phase."""
    nbytes = hidden * 2
    wire = hidden if lpc else 2 * hidden
    if lane == 0:
        _bulk_wait_read()
        _mbar_expect_tx(mbar, Int32(nbytes))
        _bulk_g2s(stage, row, Int32(nbytes), mbar)
    cute.arch.sync_warp()
    _mbar_wait(mbar, ph)
    if cutlass.const_expr(lpc):
        per_lane = wire // 16 // 32
        loads = [
            (_ld_shared_v4(stage + (lane + 32 * k) * 32), _ld_shared_v4(stage + (lane + 32 * k) * 32 + 16))
            for k in range(per_lane)
        ]
        cute.arch.sync_warp()
        for k in cutlass.range_constexpr(per_lane):
            p, q = loads[k]
            _st_shared_v4(
                stage + (lane + 32 * k) * 16,
                _bf16x4_to_e4m3x4(p[0], p[1]),
                _bf16x4_to_e4m3x4(p[2], p[3]),
                _bf16x4_to_e4m3x4(q[0], q[1]),
                _bf16x4_to_e4m3x4(q[2], q[3]),
            )
        _fence_proxy_async_shared()
    cute.arch.sync_warp()
    if lane == 0:
        _bulk_s2g(dst, stage, Int32(wire))
        _bulk_commit()
    return ph ^ 1


@cute.jit
def _poll_v2(addr: Int64, max_ns: cutlass.Constexpr, timeout_ns: Int64):
    """Re-load an 8 B chunk of e4m3 elements until none is the -0 sentinel; thousands of CTAs poll at prefill size,
    so the retries back off (32 ns doubling to ``max_ns``) to keep the L2 free for the push's traffic."""
    w0, w1 = _ld_vol_v2(addr)
    d = _dirty(w0, True) | _dirty(w1, True)
    if d != 0:
        start = _globaltimer()
        delay = Int32(32)
        while d != 0:
            if _globaltimer() - start > timeout_ns:
                _trap()
            _nanosleep(delay)
            delay = cutlass.min(delay * 2, Int32(max_ns))
            w0, w1 = _ld_vol_v2(addr)
            d = _dirty(w0, True) | _dirty(w1, True)
    return w0, w1


@cute.jit
def _poll_v4(addr: Int64, max_ns: cutlass.Constexpr, timeout_ns: Int64):
    """``_poll_v2`` for a 16 B chunk of bf16 elements."""
    w0, w1, w2, w3 = _ld_vol_v4(addr)
    d = _dirty(w0, False) | _dirty(w1, False) | _dirty(w2, False) | _dirty(w3, False)
    if d != 0:
        start = _globaltimer()
        delay = Int32(32)
        while d != 0:
            if _globaltimer() - start > timeout_ns:
                _trap()
            _nanosleep(delay)
            delay = cutlass.min(delay * 2, Int32(max_ns))
            w0, w1, w2, w3 = _ld_vol_v4(addr)
            d = _dirty(w0, False) | _dirty(w1, False) | _dirty(w2, False) | _dirty(w3, False)
    return w0, w1, w2, w3


class _Push:
    def __init__(
        self,
        ep: int,
        hidden: int,
        lpc: bool,
        lay: Tuple,
        rows: int,
        inbox_bytes: int,
        warps: int,
        bulk: bool,
        data_only: bool,
        rearm_prev: bool,
        trigger_entry: bool,
        delay_ns: int,
        pdl: bool,
        local: bool,
        rows_per_store: int,
        tma: bool = False,
    ):
        assert hidden % 512 == 0 and ep <= _MAX_RANKS
        assert not (tma and rows_per_store != 1), "the TMA row path moves one row per store"
        self.tma = tma
        self.ep, self.hidden, self.lpc, self.lay = ep, hidden, lpc, dict(lay)
        self.rows, self.inbox_bytes, self.warps = rows, inbox_bytes, warps
        self.bulk, self.data_only, self.rearm_prev = bulk, data_only, rearm_prev
        self.trigger_entry, self.delay_ns, self.pdl = trigger_entry, delay_ns, pdl
        # local: cast the received rows into this rank's own side buffer (the peers pull them after a rank
        # barrier); -0 stays -0 there since no sentinel is involved.
        self.local, self.rows_per_store = local, rows_per_store
        self.wire = hidden if lpc else 2 * hidden
        self.smem_bytes = (warps * rows_per_store * self.wire + 128) if bulk else 0
        if tma:
            self.smem_bytes = warps * (2 * hidden + 8) + 128

    def _e4(self, a0, a1):
        return _bf16x4_to_e4m3x4(a0, a1)

    def _bf(self, w):
        return w

    @cute.jit
    def __call__(
        self,
        mDummy: cute.Tensor,
        y: Int64,
        ws_base: Int64,
        ws_stride: Int64,
        side_base: Int64,
        side_stride: Int64,
        rank: Int32,
        max_tokens: Int32,
        blocks: Int32,
        timeout_ns: Int64,
        stream,
    ):
        self.kernel(y, ws_base, ws_stride, side_base, side_stride, rank, max_tokens, timeout_ns).launch(
            grid=[self.ep, blocks, 1],
            block=[self.warps * 32, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        y: Int64,
        ws_base: Int64,
        ws_stride: Int64,
        side_base: Int64,
        side_stride: Int64,
        rank: Int32,
        max_tokens: Int32,
        timeout_ns: Int64,
    ):
        L = self.lay
        wire = self.wire
        tidx, _, _ = cute.arch.thread_idx()
        source, by, _ = cute.arch.block_idx()
        _, nby, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        my_ws = ws_base + Int64(rank) * ws_stride
        my_side = side_base + Int64(rank) * side_stride
        nbytes = self.hidden * 2
        sentinel = Int32(_sentinel(self.lpc))
        stage = Int32(0)
        mbar = Int32(0)
        ph = Int32(0)
        if cutlass.const_expr(self.tma):
            smem = SmemAllocator()
            buf = smem.allocate_array(cutlass.Uint8, self.warps * 2 * self.hidden, byte_alignment=128)
            bars = smem.allocate_array(Int64, self.warps, byte_alignment=8)
            stage = Int32(buf.toint()) + warp * (2 * self.hidden)
            mbar = Int32(bars.toint()) + warp * 8
            if lane == 0:
                _mbar_init(mbar, 1)
            cute.arch.sync_threads()
        elif cutlass.const_expr(self.bulk):
            smem = SmemAllocator()
            buf = smem.allocate_array(
                cutlass.Uint8, self.warps * self.rows_per_store * wire, byte_alignment=128
            )
            stage = Int32(buf.toint()) + warp * (self.rows_per_store * wire)

        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.tma):
            _fence_proxy_async_global()
        if cutlass.const_expr(self.trigger_entry):
            cute.arch.griddepcontrol_launch_dependents()
        # The round CTA may already have advanced the flag (d -> d + 1, d odd): both give the same values.
        f = _ld_b32(my_ws + L["flag"])
        value = ((f - 1) | 1) + 1
        parity = ((f - 1) >> 1) & 1
        if cutlass.const_expr(not self.local):
            if by == 0:
                if tidx == 0:
                    _st_sys_b32(ws_base + Int64(source) * ws_stride + L["combine_flags"] + rank * 4, value)
        if source == rank:
            if by == 0:
                if warp == 0:
                    if lane < self.ep:
                        _st_b32(
                            my_side + _USED + (parity * _MAX_RANKS + lane) * 4,
                            _ld_b32(my_ws + L["send_counters"] + lane * 4),
                        )
        dst_side = side_base + Int64(source) * side_stride
        per_lane = wire // 16 // 32
        if source != rank:
            n = _spin_ne(
                my_ws + L["recv_counters"] + (parity * self.ep + source) * 4, Int32(-1), timeout_ns
            )
            box = dst_side + _HDR + Int64(parity) * self.inbox_bytes + Int64(rank) * self.rows * wire
            if cutlass.const_expr(self.local):
                box = my_side + _HDR + Int64(parity) * self.inbox_bytes + Int64(source) * self.rows * wire
            if cutlass.const_expr(self.delay_ns > 0):
                _sleep_ns(self.delay_ns)
            R = self.rows_per_store
            nchunks = (n + R - 1) // R
            for c in cutlass.range(by * self.warps + warp, nchunks, nby * self.warps, unroll=1):
                if cutlass.const_expr(self.tma):
                    ph = _tma_row(
                        self.lpc, self.hidden, y + (Int64(source) * max_tokens + c) * nbytes, box + Int64(c) * wire,
                        stage, mbar, ph, lane,
                    )
                if cutlass.const_expr(self.bulk and not self.tma):
                    if lane == 0:
                        _bulk_wait_read()
                    cute.arch.sync_warp()
                for r in cutlass.range_constexpr(0 if self.tma else R):
                    t = c * R + r
                    if t < n:
                        row = y + (Int64(source) * max_tokens + t) * nbytes
                        dst = box + Int64(t) * wire
                        # All of the lane's loads are issued before its stores (one round trip per row).
                        if cutlass.const_expr(self.lpc):
                            loads = [
                                (
                                    _ld_v4(row + (lane + 32 * k) * 32),
                                    _ld_v4(row + (lane + 32 * k) * 32 + 16),
                                )
                                for k in range(per_lane)
                            ]
                            for k in cutlass.range_constexpr(per_lane):
                                p, q = loads[k]
                                _st_any(
                                    self.bulk,
                                    dst,
                                    stage + r * wire,
                                    (lane + 32 * k) * 16,
                                    self._e4(p[0], p[1]),
                                    self._e4(p[2], p[3]),
                                    self._e4(q[0], q[1]),
                                    self._e4(q[2], q[3]),
                                )
                        else:
                            loads = [_ld_v4(row + (lane + 32 * k) * 16) for k in range(per_lane)]
                            for k in cutlass.range_constexpr(per_lane):
                                w = loads[k]
                                _st_any(
                                    self.bulk,
                                    dst,
                                    stage + r * wire,
                                    (lane + 32 * k) * 16,
                                    self._bf(w[0]),
                                    self._bf(w[1]),
                                    self._bf(w[2]),
                                    self._bf(w[3]),
                                )
                if cutlass.const_expr(self.bulk and not self.tma):
                    _fence_proxy_async_shared()
                    cute.arch.sync_warp()
                    if lane == 0:
                        _bulk_s2g(
                            box + Int64(c * R) * wire,
                            stage,
                            cutlass.min(Int32(R), n - c * R) * wire,
                        )
                        _bulk_commit()
            if cutlass.const_expr(self.rearm_prev):
                # Rows of the other parity that this rank's previous consumer read: its own next-use kernel re-arms
                # them, one round before any peer writes them again; after the data so the consumer is not delayed.
                prev = parity ^ 1
                n_prev = _ld_b32(my_side + _USED + (prev * _MAX_RANKS + source) * 4)
                box_prev = (
                    my_side + _HDR + Int64(prev) * self.inbox_bytes + Int64(source) * self.rows * wire
                )
                for t in cutlass.range(by * self.warps + warp, n_prev, nby * self.warps, unroll=1):
                    for k in cutlass.range_constexpr(per_lane):
                        _st_v4(
                            box_prev + Int64(t) * wire + (lane + 32 * k) * 16,
                            sentinel,
                            sentinel,
                            sentinel,
                            sentinel,
                        )
        if cutlass.const_expr(self.bulk):
            if lane == 0:
                if cutlass.const_expr(self.data_only):
                    _bulk_wait_read()
                else:
                    _bulk_wait_all()
                    _fence_proxy_async_global()
        cute.arch.sync_threads()
        if cutlass.const_expr(not self.data_only):
            if tidx == 0:
                _fence_acq_rel_sys()
                _st_sys_b32(dst_side + _DONE + (rank * _DONE_BLOCKS + by) * 4, value)
        if source == rank:
            if by == 0:
                if warp == 0:
                    if cutlass.const_expr(not self.local):
                        if lane < self.ep:
                            _spin_eq(my_ws + L["combine_flags"] + lane * 4, value, timeout_ns)
                        cute.arch.sync_warp()
                    if lane == 0:
                        _st_b32(my_ws + L["flag"], value)
        if cutlass.const_expr(not self.trigger_entry):
            cute.arch.griddepcontrol_launch_dependents()


class _ReduceNorm:
    def __init__(
        self,
        ep: int,
        top_k: int,
        hidden: int,
        lpc: bool,
        lay: Tuple,
        rows: int,
        inbox_bytes: int,
        wait: int,
        rearm: int,
        grid_wait: bool,
        poll_ns: int,
        min_blocks: int,
        pdl: bool,
        own_in_inbox: bool = False,
    ):
        assert top_k <= 32 and ep <= _MAX_RANKS
        assert hidden % (_VEC * _THREADS_PER_ROW) == 0
        self.ep, self.top_k, self.H, self.lpc, self.lay = ep, top_k, hidden, lpc, dict(lay)
        self.rows, self.inbox_bytes = rows, inbox_bytes
        self.wait, self.rearm, self.grid_wait, self.pdl = wait, rearm, grid_wait, pdl
        self.poll_ns, self.min_blocks = poll_ns, min_blocks
        self.own_in_inbox = own_in_inbox
        self.wire = hidden if lpc else 2 * hidden
        self.blocks = hidden // (_VEC * _THREADS_PER_ROW)
        self.threads = hidden // _VEC
        self.smem_bytes = (
            32 * 8 + 32 * 4 + 16 + 2 * 4 + hidden * 2 + 64 + (self.wire + 128 if rearm == _REARM_BULK else 0)
        )

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mW: cute.Tensor,
        ready_addr: Int64,
        ws_base: Int64,
        ws_stride: Int64,
        side_base: Int64,
        side_stride: Int64,
        rank: Int32,
        n_local: Int32,
        max_tokens: Int32,
        blocks: Int32,
        own_base: Int64,
        eps: Float32,
        timeout_ns: Int64,
        stream,
    ):
        tv_layout = cute.make_layout(
            ((_THREADS_PER_ROW, 1), (_VEC, self.blocks)),
            stride=((_VEC, 1), (1, _VEC * _THREADS_PER_ROW)),
        )
        tiler_mn = (1, self.H)
        self.kernel(
            mOut,
            mW,
            ready_addr,
            ws_base,
            ws_stride,
            side_base,
            side_stride,
            rank,
            n_local,
            max_tokens,
            blocks,
            own_base,
            eps,
            timeout_ns,
            tv_layout,
            tiler_mn,
        ).launch(
            grid=[cutlass.max(n_local, Int32(1)), 1, 1],
            block=[self.threads, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=self.min_blocks,
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mW: cute.Tensor,
        ready_addr: Int64,
        ws_base: Int64,
        ws_stride: Int64,
        side_base: Int64,
        side_stride: Int64,
        rank: Int32,
        n_local: Int32,
        max_tokens: Int32,
        blocks: Int32,
        own_base: Int64,
        eps: Float32,
        timeout_ns: Int64,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        L = self.lay
        K = self.top_k
        H = self.H
        wire = self.wire
        tidx, _, _ = cute.arch.thread_idx()
        tok, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        smem = SmemAllocator()
        s_src = smem.allocate_tensor(Int64, cute.make_layout((32,)), byte_alignment=16)
        s_own = smem.allocate_tensor(Int32, cute.make_layout((32,)), byte_alignment=16)
        s_n = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
        partials = smem.allocate_tensor(Float32, cute.make_layout((1, 2)), byte_alignment=16)
        sX = smem.allocate_tensor(
            BFloat16, cute.make_layout(tiler_mn, stride=(H, 1)), byte_alignment=16
        )
        sent = Int32(0)
        if cutlass.const_expr(self.rearm == _REARM_BULK):
            sbuf = smem.allocate_array(cutlass.Uint8, wire, byte_alignment=128)
            sent = Int32(sbuf.toint())
        my_ws = ws_base + Int64(rank) * ws_stride
        my_side = side_base + Int64(rank) * side_stride
        sentinel = Int32(_sentinel(self.lpc))

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
        if cutlass.const_expr(self.grid_wait):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.rearm == _REARM_BULK):
            for i in cutlass.range(tidx, wire // 16, self.threads, unroll=1):
                _st_shared_v4(sent + i * 16, sentinel, sentinel, sentinel, sentinel)
            _fence_proxy_async_shared()
        f = _ld_b32(my_ws + L["flag"])
        value = ((f - 1) | 1) + 1
        parity = ((f - 1) >> 1) & 1
        inbox = my_side + _HDR + Int64(parity) * self.inbox_bytes
        if cutlass.const_expr(self.wait == _WAIT_DONE):
            # The spinning threads acquire; the CTA barrier below extends the order to every thread's loads.
            for i in cutlass.range(tidx, self.ep * blocks, self.threads, unroll=1):
                src = i // blocks
                if src != rank:
                    _spin_eq(my_side + _DONE + (src * _DONE_BLOCKS + i % blocks) * 4, value, timeout_ns)
            if tidx < self.ep * blocks:
                _fence_acq_rel_sys()
        if cutlass.const_expr(self.wait == _WAIT_PULL):
            # Stock's barrier: CTA 0 publishes this rank's round value to every peer, every CTA waits for all.
            if warp == 0:
                if lane < self.ep:
                    if tok == 0:
                        _fence_acq_rel_sys()
                        _st_sys_b32(
                            ws_base + Int64(lane) * ws_stride + L["combine_flags"] + rank * 4, value
                        )
                    _spin_eq(my_ws + L["combine_flags"] + lane * 4, value, timeout_ns)
                    _fence_acq_rel_sys()
        if tok < n_local:
            if warp == 0:
                tr = Int32(-1)
                di = Int32(-1)
                if lane < K:
                    rec = Int64(tok) * K + lane
                    tr = _ld_b32(my_ws + L["topk_ranks"] + rec * 4)
                    di = _ld_b32(my_ws + L["topk_indices"] + rec * 4)
                    if tr < 0:
                        di = Int32(-1)
                valid = cute.arch.vote_ballot_sync(di >= 0)
                if lane == 0:
                    s_n[0] = cute.arch.popc(valid)
                if di >= 0:
                    at = cute.arch.popc(valid & ((Int32(1) << lane) - 1))
                    own = Int32(0)
                    row = inbox + (Int64(tr) * self.rows + di) * wire
                    if cutlass.const_expr(self.wait == _WAIT_PULL):
                        row = (
                            side_base
                            + Int64(tr) * side_stride
                            + _HDR
                            + Int64(parity) * self.inbox_bytes
                            + (Int64(rank) * self.rows + di) * wire
                        )
                    if cutlass.const_expr(not self.own_in_inbox):
                        if tr == rank:
                            row = own_base + (Int64(rank) * max_tokens + di) * (H * 2)
                            own = Int32(1)
                    s_src[at] = row
                    s_own[at] = own
        cute.arch.sync_threads()

        if tok < n_local:
            count = s_n[0]
            self._reduce_vec(sX, tidx, s_src, s_own, count, timeout_ns)
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
                rstd = cute.math.rsqrt(sum_sq / Float32(H) + eps, fastmath=True)
                w = tWrW.load().to(Float32)
                yv = x * rstd * (w + Float32(0.0))
                tXrO.store(yv.to(BFloat16))
                cute.copy(copy_x, tXrO, tXgO)
                if ready_addr != 0:
                    cute.arch.barrier(barrier_id=_NORM_BARRIER, number_of_threads=_THREADS_PER_ROW)
                    if tidx == 0:
                        _st_release_b32(ready_addr + Int64(tok) * 4, Int32(1))
            # Consumer re-arm, after the token's final output writes.
            if cutlass.const_expr(self.rearm == _REARM_BULK):
                cute.arch.sync_threads()
                if tidx == 0:
                    for s in cutlass.range(0, count, 1, unroll=1):
                        if s_own[s] == 0:
                            _bulk_s2g(s_src[s], sent, Int32(wire))
                    _bulk_commit()
            if cutlass.const_expr(self.rearm == _REARM_ST):
                cute.arch.sync_threads()
                self._rearm_st(tidx, s_src, s_own, count)
        if cutlass.const_expr(self.rearm == _REARM_BULK):
            if tidx == 0:
                _bulk_wait_all()
        # Keeps completion transitive: the next round's dispatch is ordered after the push and its round CTA.
        cute.arch.griddepcontrol_wait()

    @cute.jit
    def _reduce_vec(
        self,
        sX: cute.Tensor,
        v: Int32,
        s_src: cute.Tensor,
        s_own: cute.Tensor,
        count: Int32,
        timeout_ns: Int64,
    ):
        """Vector ``v`` (8 elements) of the row sum of the ``count`` contributions into ``sX`` (bf16), stock order.
        Every source's chunk is loaded before any Lamport chunk is checked, so only late chunks are re-polled."""
        lpc = self.lpc
        ns = (self.top_k + 7) // 8 * 8
        words = cute.recast_tensor(sX, Int32)
        raw = cute.make_rmem_tensor(cute.make_layout((4, ns)), Int32)
        res = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        raw.fill(0)
        for s in cutlass.range_constexpr(ns):
            if s < count:
                if s_own[s] != 0:
                    w = _ld_v4(s_src[s] + v * 16)
                    if cutlass.const_expr(lpc):
                        raw[0, s] = _bf16x4_to_e4m3x4(w[0], w[1])
                        raw[1, s] = _bf16x4_to_e4m3x4(w[2], w[3])
                    else:
                        for q in cutlass.range_constexpr(4):
                            raw[q, s] = w[q]
                elif cutlass.const_expr(self.wait == _WAIT_PULL and lpc):
                    e = _ld_v2(s_src[s] + v * 8)
                    raw[0, s] = e[0]
                    raw[1, s] = e[1]
                elif cutlass.const_expr(self.wait == _WAIT_PULL):
                    w = _ld_v4(s_src[s] + v * 16)
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
        if cutlass.const_expr(self.wait == _WAIT_POLL):
            for s in cutlass.range_constexpr(ns):
                if s < count:
                    if s_own[s] == 0:
                        if cutlass.const_expr(lpc):
                            if (_dirty(raw[0, s], True) | _dirty(raw[1, s], True)) != 0:
                                e0, e1 = _poll_v2(s_src[s] + v * 8, self.poll_ns, timeout_ns)
                                raw[0, s] = e0
                                raw[1, s] = e1
                        else:
                            d = (
                                _dirty(raw[0, s], False)
                                | _dirty(raw[1, s], False)
                                | _dirty(raw[2, s], False)
                                | _dirty(raw[3, s], False)
                            )
                            if d != 0:
                                w0, w1, w2, w3 = _poll_v4(s_src[s] + v * 16, self.poll_ns, timeout_ns)
                                raw[0, s] = w0
                                raw[1, s] = w1
                                raw[2, s] = w2
                                raw[3, s] = w3
        self._sum(raw, count, res)
        for q in cutlass.range_constexpr(4):
            words[0, v * 4 + q] = _f32x2_to_bf16x2(res[2 * q], res[2 * q + 1])

    def _elems(self, raw: cute.Tensor, s: int):
        if self.lpc:
            return _e4m3x4_to_f32(raw[0, s]) + _e4m3x4_to_f32(raw[1, s])
        pairs = [_bf16x2_to_f32(raw[q, s]) for q in range(4)]
        return [pairs[q][h] for q in range(4) for h in range(2)]

    def _pair(self, raw: cute.Tensor, s: int):
        a = self._elems(raw, s)
        b = self._elems(raw, s + 1)
        return [a[j] + b[j] for j in range(8)]

    @cute.jit
    def _sum(self, raw: cute.Tensor, count: Int32, res: cute.Tensor):
        """Stock ``vectorized_combine_impl`` order: pairs, then groups of 8 sources, then the group tree; absent
        sources are +0, and two sources or fewer are summed as ``a0 + a1`` alone."""
        ns = (self.top_k + 7) // 8 * 8
        p0 = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        gs = cute.make_rmem_tensor(cute.make_layout((8, ns // 8)), Float32)
        # One group of 8 sources at a time keeps the live pair sums at 4 x 8 floats.
        for k in cutlass.range_constexpr(ns // 8):
            pk = [self._pair(raw, 8 * k + 2 * i) for i in range(4)]
            for j in cutlass.range_constexpr(8):
                gs[j, k] = (pk[0][j] + pk[1][j]) + (pk[2][j] + pk[3][j])
                if k == 0:
                    p0[j] = pk[0][j]
        for j in cutlass.range_constexpr(8):
            res[j] = p0[j]
            if count > 2:
                res[j] = gs[j, 0]
            if cutlass.const_expr(ns >= 16):
                if count > 8:
                    res[j] = gs[j, 0] + gs[j, 1]
            if cutlass.const_expr(ns >= 32):
                if count > 16:
                    res[j] = (gs[j, 0] + gs[j, 1]) + (gs[j, 2] + gs[j, 3])

    @cute.jit
    def _rearm_st(self, v: Int32, s_src: cute.Tensor, s_own: cute.Tensor, count: Int32):
        ns = (self.top_k + 7) // 8 * 8
        sentinel = Int32(_sentinel(self.lpc))
        for s in cutlass.range_constexpr(ns):
            if s < count:
                if s_own[s] == 0:
                    if cutlass.const_expr(self.lpc):
                        _st_v2(s_src[s] + v * 8, sentinel, sentinel)
                    else:
                        _st_v4(s_src[s] + v * 16, sentinel, sentinel, sentinel, sentinel)


def _fake(dtype, shape):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(range(len(shape) - 1, -1, -1)), assumed_align=16
    )


@functools.lru_cache(maxsize=None)
def _compile(kind: str, *cfg):
    """Every entry takes one device tensor first: the TVM-FFI stream binding needs it to find the device."""
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    i64, i32 = Int64(0), Int32(0)
    opt = "--enable-tvm-ffi"
    if kind == "push":
        return cute.compile(
            _Push(*cfg),
            _fake(cutlass.Uint8, (cute.sym_int(),)),
            *([i64] * 5),
            i32,
            Int32(1),
            Int32(1),
            i64,
            stream,
            options=opt,
        )
    if kind == "reduce":
        hidden = cfg[2]
        return cute.compile(
            _ReduceNorm(*cfg),
            _fake(BFloat16, (cute.sym_int(), hidden)),
            _fake(BFloat16, (hidden,)),
            *([i64] * 5),
            i32,
            Int32(1),
            Int32(1),
            Int32(1),
            i64,
            Float32(1e-6),
            i64,
            stream,
            options=opt,
        )
    raise ValueError(kind)


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    v = int(os.environ.get(name, str(default)))
    if not lo <= v <= hi:
        raise ValueError(f"{name}={v}; valid: {lo}..{hi}")
    return v


class PrefillInbox(NamedTuple):
    """Consumer view of a rank's prefill inbox: rank r's side buffer starts at ``side_base + r * side_stride``;
    inbox ``p`` (p = round parity) at ``_HDR + p * inbox_bytes``; the row of (source s, slot d) at
    ``(s * rows + d) * wire``. ``done[s][by]`` (u32) at ``_DONE + (s * _DONE_BLOCKS + by) * 4`` holds the round's
    combine value once source s's push CTA ``by`` < ``blocks`` finished (not written by data-only pushes)."""

    side_base: int
    side_stride: int
    rows: int
    wire: int
    inbox_bytes: int
    lpc: bool
    blocks: int


class PrefillCombine:
    """Chunk-sized Lamport combine of a stock ``NVLinkOneSided`` round, fused with the latent RMSNorm.

    ``PrefillCombine(comm, rows)`` is collective (every rank, same ``rows`` = the chunk cap in tokens per rank): it
    allocates the symmetric side buffer with the two -0 armed inboxes. Per round, after ``comm.dispatch`` and the
    expert path produced ``y`` (bf16 ``[ep * runtime_max_tokens, hidden]``, contiguous; the finalize output or the
    in-workspace combine payload): ``push(y, fgs=..., rearm=...)`` then ``reduce_norm(norm_weight, eps, out=...)``
    (or ``combine_norm`` for both) returns the normed rows ``[local_tokens, hidden]`` bf16 and resets the
    communicator's round state. ``fgs``: True = the reduce polls the inbox and runs under the push; False = standard
    PDL plus the push's completion words; ``"pull"`` = the non-FGS form of stock's shape: the rows are cast into this
    rank's own side buffer, and the reduce pulls them from the peers after a rank barrier (no sentinels, no re-arm,
    bitwise with stock). ``rearm`` in ``REARMS``; the push and the reduce of a round must agree.
    ``hazard_delay_ns`` (tests) delays this rank's remote stores; ``check=False`` (tests) reads without polling.
    """

    def __init__(self, comm, rows: int, warps: Optional[int] = None, blocks: Optional[int] = None):
        self.comm = comm
        self.ep, self.rank = comm.ep_size, comm.ep_rank
        if self.ep > _MAX_RANKS:
            raise ValueError(f"PrefillCombine supports at most {_MAX_RANKS} ranks")
        self.lpc = bool(comm.use_low_precision_combine)
        self.lay = _Layout(comm)
        self.top_k = comm.top_k
        self.hidden = int(comm._workspace_state["hidden_size"])
        self.wire = self.hidden if self.lpc else 2 * self.hidden
        self.rows = rows
        self.inbox_bytes = _align(self.ep * rows * self.wire, _CACHELINE)
        self.warps = _env_int(PUSH_WARPS_ENV, 8, 1, 32) if warps is None else warps
        self.blocks_cap = _env_int(PUSH_BLOCKS_ENV, 64, 1, _DONE_BLOCKS) if blocks is None else blocks
        self._round: Optional[Tuple] = None
        self._alloc()

    def _alloc(self) -> None:
        from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

        self._side_mem = MnnvlMemory(self.comm.mapping, _HDR + 2 * self.inbox_bytes)
        self.side = self._side_mem.as_torch_strided_tensor(torch.uint8)
        self.arm()

    def arm(self) -> None:
        """Header zeroed, both inboxes filled with the sentinel (allocation; tests re-arm after a negative control).
        Collective: every rank's pending pushes complete before any rank fills."""
        from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

        torch.cuda.synchronize()
        MnnvlMemory.get_comm(self.comm.mapping).barrier()
        mine = self.side[self.rank]
        mine[:_HDR].zero_()
        boxes = mine[_HDR : _HDR + 2 * self.inbox_bytes]
        boxes.fill_(0xFF)
        torch.cuda.synchronize()
        MnnvlMemory.get_comm(self.comm.mapping).barrier()

    @property
    def side_bytes(self) -> int:
        return _HDR + 2 * self.inbox_bytes

    def _ws(self) -> Tuple[int, int]:
        ws = self.comm.workspace
        return ws.data_ptr(), ws.stride(0)

    def _blocks(self, max_tokens: int) -> int:
        return min(self.blocks_cap, max(1, (max_tokens + self.warps - 1) // self.warps))

    def _args(self, y: torch.Tensor):
        st = self.comm._dispatch_state
        if st.get("phase") != "dispatched":
            raise RuntimeError("PrefillCombine.push called before a successful dispatch")
        n, mt = st["local_num_tokens"], st["runtime_max_tokens_per_rank"]
        if mt > self.rows:
            raise ValueError(f"round max tokens {mt} exceeds the inbox rows {self.rows}")
        y = y.reshape(self.ep * mt, -1)
        if (
            y.shape[1] != self.hidden
            or y.dtype != torch.bfloat16
            or not y.is_contiguous()
            or y.data_ptr() % 16
        ):
            raise ValueError(f"the combine payload must be contiguous bf16 [{self.ep * mt}, {self.hidden}]")
        return y, n, mt

    def inbox(self, max_tokens: int) -> PrefillInbox:
        return PrefillInbox(
            self.side.data_ptr(),
            self.side.stride(0),
            self.rows,
            self.wire,
            self.inbox_bytes,
            self.lpc,
            self._blocks(max_tokens),
        )

    def push(
        self,
        y: torch.Tensor,
        *,
        fgs: Union[bool, str] = True,
        rearm: str = "consumer",
        data_only: Optional[bool] = None,
        bulk: bool = True,
        hazard_delay_ns: int = 0,
    ) -> PrefillInbox:
        if rearm not in REARMS:
            raise ValueError(f"rearm {rearm!r}; valid: {REARMS}")
        y, n, mt = self._args(y)
        mode = MODES[fgs]
        data_only = (mode != "done") if data_only is None else data_only
        if data_only and mode == "done":
            raise ValueError("a data-only push leaves no completion words for the FGS-off reduce")
        blocks = self._blocks(mt)
        ws_base, ws_stride = self._ws()
        _compile(
            "push",
            self.ep,
            self.hidden,
            self.lpc,
            self.lay.key(),
            self.rows,
            self.inbox_bytes,
            self.warps,
            bulk,
            data_only,
            rearm == "producer" and mode != "pull",
            mode == "poll",
            hazard_delay_ns,
            _comm._pdl() or mode != "done",
            mode == "pull",
            _env_int(PUSH_ROWS_ENV, 1, 1, 8),
            os.environ.get(PUSH_LOAD_ENV, "tma") == "tma",
        )(
            self.side[self.rank, :_CACHELINE],
            y.data_ptr(),
            ws_base,
            ws_stride,
            self.side.data_ptr(),
            self.side.stride(0),
            self.rank,
            mt,
            blocks,
            _TIMEOUT_NS,
        )
        self._round = (n, mt, y.data_ptr(), blocks, mode, data_only, rearm, False)
        return self.inbox(mt)

    def reduce_norm(
        self,
        norm_weight: torch.Tensor,
        eps: float,
        out: Optional[torch.Tensor] = None,
        *,
        fgs: Optional[Union[bool, str]] = None,
        rearm: Optional[str] = None,
        ready: Optional[torch.Tensor] = None,
        check: bool = True,
    ) -> torch.Tensor:
        if self._round is None:
            raise RuntimeError("PrefillCombine.reduce_norm needs the round's push first")
        n, mt, y_ptr, blocks, pmode, data_only, prearm, own_in_inbox = self._round
        mode = pmode if fgs is None else MODES[fgs]
        rearm = prearm if rearm is None else rearm
        if rearm not in REARMS:
            raise ValueError(f"rearm {rearm!r}; valid: {REARMS}")
        if mode == "done" and data_only:
            raise ValueError("the FGS-off reduce needs a push with completion words (data_only=False)")
        if (mode == "pull") != (pmode == "pull"):
            raise ValueError("the pull reduce needs the local cast push of the same round, and only then")
        if out is None:
            out = torch.empty(n, self.hidden, dtype=torch.bfloat16, device=norm_weight.device)
        if (
            out.dtype != torch.bfloat16
            or not out.is_contiguous()
            or out.shape[0] < n
            or out.shape[-1] != self.hidden
            or norm_weight.dtype != torch.bfloat16
            or norm_weight.numel() != self.hidden
        ):
            raise ValueError(f"out must be contiguous bf16 [>= {n}, {self.hidden}], weight bf16 [{self.hidden}]")
        if ready is not None and (ready.dtype != torch.int32 or ready.numel() < n):
            raise ValueError(f"ready must be int32 [>= {n}]")
        wait = {"poll": _WAIT_POLL, "done": _WAIT_DONE, "pull": _WAIT_PULL}[mode] if check else _WAIT_NONE
        if mode in ("poll", "done") and check and os.environ.get(CHECK_ENV, "sw") == "hw":
            from .combine_tma import reduce_norm_tma

            reduce_norm_tma(self, out, norm_weight, eps, ready, n, mt, blocks, y_ptr, rearm, own_in_inbox, wait)
            self._round = None
            self.comm.reset_state()
            return out
        rearm_mode = {
            "consumer": _REARM_BULK,
            "consumer_st": _REARM_ST,
            "producer": _REARM_NONE,
            "none": _REARM_NONE,
        }[rearm]
        if mode == "pull":
            rearm_mode = _REARM_NONE
        ws_base, ws_stride = self._ws()
        _compile(
            "reduce",
            self.ep,
            self.top_k,
            self.hidden,
            self.lpc,
            self.lay.key(),
            self.rows,
            self.inbox_bytes,
            wait,
            rearm_mode,
            mode != "poll",
            _env_int(POLL_NS_ENV, 256, 32, 1 << 20),
            _env_int(REDUCE_MIN_BLOCKS_ENV, 2, 0, 4),
            _comm._pdl() or mode != "done",
            own_in_inbox,
        )(
            out,
            norm_weight,
            ready.data_ptr() if ready is not None else 0,
            ws_base,
            ws_stride,
            self.side.data_ptr(),
            self.side.stride(0),
            self.rank,
            n,
            mt,
            blocks,
            y_ptr,
            eps,
            _TIMEOUT_NS,
        )
        self._round = None
        self.comm.reset_state()
        return out

    def combine_norm(
        self,
        y: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        out: Optional[torch.Tensor] = None,
        *,
        fgs: Union[bool, str] = True,
        rearm: str = "consumer",
        data_only: Optional[bool] = None,
        bulk: bool = True,
        ready: Optional[torch.Tensor] = None,
        check: bool = True,
        hazard_delay_ns: int = 0,
    ) -> torch.Tensor:
        self.push(y, fgs=fgs, rearm=rearm, data_only=data_only, bulk=bulk, hazard_delay_ns=hazard_delay_ns)
        return self.reduce_norm(norm_weight, eps, out, fgs=fgs, rearm=rearm, ready=ready, check=check)
