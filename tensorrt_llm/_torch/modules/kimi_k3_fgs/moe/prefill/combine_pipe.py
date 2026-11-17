# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent, pipelined consumer of the prefill Lamport inbox (``TLLM_K3_PC_CHECK=hw``, ``TLLM_K3_PC_REDUCE=pipe``).

A grid of a few CTAs per SM strides over the local tokens; each CTA has three warp roles. The producer warp keeps
``stages`` tokens of TMA row loads in flight: per token it compacts the routing records, issues one TMA tensor load
per remote row (with ``report_valid`` under FGS on: a row that landed with the 0xFF sentinel is re-issued after a
backoff, so only invalid rows are retried) and one bulk load of the own bf16 row, then hands the oldest landed stage
to the reduce warps through a full / empty mbarrier pair. The reduce warps sum the rows from shared memory in the
stock order and round once to bf16 into a double-buffered row; the two norm warps run the FlashInfer-mapped latent
RMSNorm on it, store the output and, after those writes, re-arm the token's remote rows with 16 B stores of the
sentinel. Outputs are bitwise those of ``combine._ReduceNorm``.

FGS off (the twin): the grid dependency plus the push-completion words once per kernel (CTA 0 checks them and
publishes a local round word), the same TMA loads without ``report_valid``. DSL 4.9 on SM107: blocking mbarrier
waits hang (try-wait loops throughout); the TMA ``cute.copy`` is issued by the whole producer warp, never inside
``elect_one``.
"""

from __future__ import annotations

import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from ..comm import (
    _TIMEOUT_NS,
    _asm,
    _bf16x2_to_f32,
    _bf16x4_to_e4m3x4,
    _bulk_g2s,
    _f32x2_to_bf16x2,
    _fence_acq_rel_sys,
    _fence_proxy_async_global,
    _globaltimer,
    _ld_acquire_b32,
    _ld_b32,
    _mbar_expect_tx,
    _mbar_init,
    _mbar_wait,
    _nanosleep,
    _st_shared_v4,
    _st_v4,
    _trap,
    _v,
)
from ..latent import _THREADS_PER_ROW, _VEC
from ..tail import _st_release_b32
from .combine import (
    _DONE,
    _DONE_BLOCKS,
    _HDR,
    _NORM_BARRIER,
    _REARM_BULK,
    _USED,
    _WAIT_DONE,
    _WAIT_POLL,
    _env_int,
    _ReduceNorm,
    _sentinel,
)
from .combine_tma import _ROW_BOX, _ld_shared_v2, _ld_shared_v4
from .common import num_sms

STAGES_ENV = "TLLM_K3_PC_REDUCE_STAGES"
CTAS_PER_SM_ENV = "TLLM_K3_PC_REDUCE_CTAS_PER_SM"
THREADS_ENV = "TLLM_K3_PC_REDUCE_THREADS"
DIAG_ENV = "TLLM_K3_PC_REDUCE_DIAG"
_ROUND_DONE = _USED + 256
_SMEM_PER_SM = 228 * 1024
_SMEM_PER_CTA = 227 * 1024
_SMEM_RESERVED = 1024


@dsl_user_op
def _mbar_arrive(mbar, *, loc=None, ip=None) -> None:
    _asm(None, [_v(mbar, Int32, loc, ip)], "mbarrier.arrive.shared::cta.b64 _, [$0];", "r", loc, ip)


@dsl_user_op
def _ld_relaxed_gpu_v4(addr, *, loc=None, ip=None):
    res = _asm(
        cutlass._mlir.dialects.llvm.StructType.get_literal([T.i32()] * 4),
        [_v(addr, Int64, loc, ip)],
        "ld.relaxed.gpu.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return [
        Int32(cutlass._mlir.dialects.llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@cute.jit
def _spin_eq4(addr: Int64, n: Int32, expected: Int32, timeout_ns: Int64):
    """Spin on the first ``n`` (1..4) completion words of a 16 B chunk. The peers' NVLink stores land in this GPU's
    L2, which a GPU-scope strong load reads; system-scope loads serialize at ~130 ns each (256 of them: 33 µs), and
    the caller's one ``fence.acq_rel.sys`` per CTA provides the acquire."""
    start = _globaltimer()
    pending = Int32(1)
    while pending != 0:
        w = _ld_relaxed_gpu_v4(addr)
        pending = Int32(0)
        if w[0] != expected:
            pending = Int32(1)
        if n > 1:
            if w[1] != expected:
                pending = Int32(1)
        if n > 2:
            if w[2] != expected:
                pending = Int32(1)
        if n > 3:
            if w[3] != expected:
                pending = Int32(1)
        if pending != 0:
            if _globaltimer() - start > timeout_ns:
                _trap()


class _ReduceNormPipe(_ReduceNorm):
    def __init__(self, *cfg, stages: int, cthreads: int, ctas_per_sm: int, diag: str = "none"):
        super().__init__(*cfg)
        assert self.H == 14 * 256 and (self.H // _VEC) % cthreads == 0 and cthreads % 32 == 0
        assert self.wait in (_WAIT_POLL, _WAIT_DONE)
        self.stages, self.cthreads, self.min_blocks = stages, cthreads, ctas_per_sm
        self.vpt = self.H // _VEC // cthreads
        self.slots = min(self.top_k, self.ep)
        self.ns = (self.slots + 7) // 8 * 8
        self.own_bytes = 0 if self.own_in_inbox else 2 * self.H
        self.diag = diag
        self.threads = cthreads + _THREADS_PER_ROW + 32
        self.smem_bytes = _smem_bytes(
            self.H, self.wire, self.top_k, self.slots, self.own_bytes, stages
        )

    @cute.jit
    def _row_invalid(self, bar, phase):
        done = Int32(0)
        invalid = Int32(0)
        while done == 0:
            res = cute.arch.mbarrier_try_wait_primary(bar, phase)
            done = cutlass.select_(res[0], Int32(1), Int32(0))
            invalid = cutlass.select_(res[1], Int32(1), Int32(0))
        return invalid

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mW: cute.Tensor,
        mInbox: cute.Tensor,
        ready_addr: Int64,
        ws_base: Int64,
        ws_stride: Int64,
        rank: Int32,
        n_local: Int32,
        max_tokens: Int32,
        own_base: Int64,
        inbox: Int64,
        side_base: Int64,
        side_stride: Int64,
        blocks: Int32,
        n_ctas: Int32,
        eps: Float32,
        timeout_ns: Int64,
        stream,
    ):
        tv_layout = cute.make_layout(
            ((_THREADS_PER_ROW, 1), (_VEC, self.blocks)),
            stride=((_VEC, 1), (1, _VEC * _THREADS_PER_ROW)),
        )
        tiler_mn = (1, self.H)
        row_layout = cute.make_layout(_ROW_BOX)
        tma_atom, mInbox_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mInbox, row_layout, _ROW_BOX
        )
        self.kernel(
            mOut,
            mW,
            mInbox_tma,
            tma_atom,
            ready_addr,
            ws_base,
            ws_stride,
            rank,
            n_local,
            max_tokens,
            own_base,
            inbox,
            side_base,
            side_stride,
            blocks,
            eps,
            timeout_ns,
            tv_layout,
            tiler_mn,
        ).launch(
            grid=[n_ctas, 1, 1],
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
        mInbox: cute.Tensor,
        tma_atom: cute.CopyAtom,
        ready_addr: Int64,
        ws_base: Int64,
        ws_stride: Int64,
        rank: Int32,
        n_local: Int32,
        max_tokens: Int32,
        own_base: Int64,
        inbox: Int64,
        side_base: Int64,
        side_stride: Int64,
        blocks: Int32,
        eps: Float32,
        timeout_ns: Int64,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        L = self.lay
        K = self.top_k
        H = self.H
        S = self.stages
        NS = self.slots
        wire = self.wire
        C = self.cthreads
        nbytes_own = 2 * H
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        G, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        smem = SmemAllocator()
        s_row = smem.allocate_tensor(Int32, cute.make_layout((S * K,)), byte_alignment=16)
        s_own = smem.allocate_tensor(Int32, cute.make_layout((S * K,)), byte_alignment=16)
        s_ph = smem.allocate_tensor(Int32, cute.make_layout((S * K + S,)), byte_alignment=16)
        s_n = smem.allocate_tensor(Int32, cute.make_layout((S,)), byte_alignment=16)
        s_has_own = smem.allocate_tensor(Int32, cute.make_layout((S,)), byte_alignment=16)
        partials = smem.allocate_tensor(Float32, cute.make_layout((2, 2)), byte_alignment=16)
        sX = smem.allocate_tensor(
            BFloat16, cute.make_layout((2, H), stride=(H, 1)), byte_alignment=16
        )
        bars = smem.allocate_array(Int64, S * K, byte_alignment=8)
        bars_own = smem.allocate_array(Int64, S, byte_alignment=8)
        full = smem.allocate_array(Int64, S, byte_alignment=8)
        empty = smem.allocate_array(Int64, S, byte_alignment=8)
        red = smem.allocate_array(Int64, 2, byte_alignment=8)
        free = smem.allocate_array(Int64, 2, byte_alignment=8)
        rows_buf = smem.allocate_array(cutlass.Uint8, S * NS * wire, byte_alignment=128)
        own_base_s = Int32(0)
        if cutlass.const_expr(not self.own_in_inbox):
            own_buf = smem.allocate_array(cutlass.Uint8, S * nbytes_own, byte_alignment=128)
            own_base_s = Int32(own_buf.toint())
        rows_base = Int32(rows_buf.toint())
        bars_base = Int32(bars.toint())
        bars_own_base = Int32(bars_own.toint())
        full_base = Int32(full.toint())
        empty_base = Int32(empty.toint())
        red_base = Int32(red.toint())
        free_base = Int32(free.toint())
        sX_base = Int32(sX.iterator.toint())
        dtype = cutlass.Float8E4M3FN if self.lpc else cutlass.BFloat16
        sRows = cute.make_tensor(
            cute.recast_ptr(rows_buf, dtype=dtype), cute.make_layout((256, 14, 1, S * NS))
        )
        gI = cute.local_tile(mInbox, _ROW_BOX, (0, 0, None))
        tIsI, tIgI = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sRows, 0, 3),
            cute.group_modes(gI, 0, 3),
        )
        my_ws = ws_base + Int64(rank) * ws_stride
        out_base = Int64(mOut.iterator.toint())
        sentinel = Int32(_sentinel(self.lpc))
        pattern = cpasync.SentinelPattern.PER_ELEMENT_FF

        ntid = (tidx - C) & (_THREADS_PER_ROW - 1)
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

        if tidx == 0:
            for b in cutlass.range_constexpr(S * K):
                cute.arch.mbarrier_v1_init(bars + b, 1)
            for s in cutlass.range_constexpr(S):
                _mbar_init(bars_own_base + s * 8, 1)
                _mbar_init(full_base + s * 8, 1)
                _mbar_init(empty_base + s * 8, C // 32 + 2)
            for b in cutlass.range_constexpr(2):
                _mbar_init(red_base + b * 8, C // 32)
                _mbar_init(free_base + b * 8, 1)
            cute.arch.mbarrier_init_fence()
        for i in cutlass.range(tidx, S * K + S, self.threads, unroll=1):
            s_ph[i] = Int32(0)
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.grid_wait):
            cute.arch.griddepcontrol_wait()
        f = _ld_b32(my_ws + L["flag"])
        value = ((f - 1) | 1) + 1
        parity = ((f - 1) >> 1) & 1
        if cutlass.const_expr(self.wait == _WAIT_DONE and self.diag != "nodone"):
            # CTA 0 waits for every peer's completion words (four per load), acquires once and publishes the round
            # to the other CTAs through one local word they acquire at GPU scope.
            my_side = side_base + Int64(rank) * side_stride
            if cta == 0:
                nch = (blocks + 3) // 4
                for i in cutlass.range(tidx, self.ep * nch, self.threads, unroll=1):
                    src = i // nch
                    c4 = (i % nch) * 4
                    if src != rank:
                        _spin_eq4(
                            my_side + _DONE + (src * _DONE_BLOCKS + c4) * 4,
                            cutlass.min(Int32(4), blocks - c4),
                            value,
                            timeout_ns,
                        )
                cute.arch.sync_threads()
                if tidx == 0:
                    _fence_acq_rel_sys()
                    _st_release_b32(my_side + _ROUND_DONE, value)
            else:
                if tidx == 0:
                    start = _globaltimer()
                    while _ld_acquire_b32(my_side + _ROUND_DONE) != value:
                        if _globaltimer() - start > timeout_ns:
                            _trap()
        cute.arch.sync_threads()

        ntok = Int32(0)
        if cta < n_local:
            ntok = (n_local - cta + G - 1) // G

        if warp == C // 32 + 2:
            _fence_proxy_async_global()
            # Routing records are loaded two tokens ahead so their latency hides behind two stage rounds.
            tr = Int32(-1)
            di = Int32(-1)
            tr2 = Int32(-1)
            di2 = Int32(-1)
            if ntok > 0:
                if lane < K:
                    rec = Int64(cta) * K + lane
                    tr = _ld_b32(my_ws + L["topk_ranks"] + rec * 4)
                    di = _ld_b32(my_ws + L["topk_indices"] + rec * 4)
            if ntok > 1:
                if lane < K:
                    rec = (Int64(cta) + G) * K + lane
                    tr2 = _ld_b32(my_ws + L["topk_ranks"] + rec * 4)
                    di2 = _ld_b32(my_ws + L["topk_indices"] + rec * 4)
            for i in cutlass.range(0, ntok, 1, unroll=1):
                stage = i % S
                rnd = i // S
                if i >= S - 1:
                    self._wait_stage(
                        (i - (S - 1)) % S,
                        tma_atom,
                        tIgI,
                        tIsI,
                        s_row,
                        s_own,
                        s_ph,
                        s_n,
                        s_has_own,
                        bars_base,
                        bars_own_base,
                        full_base,
                        lane,
                        timeout_ns,
                        pattern,
                    )
                if i >= S:
                    _mbar_wait(empty_base + stage * 8, (rnd - 1) & 1)
                if tr < 0:
                    di = Int32(-1)
                valid = cute.arch.vote_ballot_sync(di >= 0)
                isown = Int32(0)
                if cutlass.const_expr(not self.own_in_inbox):
                    if di >= 0:
                        if tr == rank:
                            isown = Int32(1)
                ownmask = cute.arch.vote_ballot_sync(isown != 0)
                if lane == 0:
                    s_n[stage] = cute.arch.popc(valid)
                    s_has_own[stage] = ownmask
                if di >= 0:
                    at = stage * K + cute.arch.popc(valid & ((Int32(1) << lane) - 1))
                    s_row[at] = parity * (self.ep * self.rows) + tr * self.rows + di
                    s_own[at] = isown
                if cutlass.const_expr(not self.own_in_inbox):
                    if isown != 0:
                        _mbar_expect_tx(bars_own_base + stage * 8, Int32(nbytes_own))
                        _bulk_g2s(
                            own_base_s + stage * nbytes_own,
                            own_base + (Int64(rank) * max_tokens + di) * nbytes_own,
                            Int32(nbytes_own),
                            bars_own_base + stage * 8,
                        )
                cute.arch.sync_warp()
                count = s_n[stage]
                for s in cutlass.range(0, count, 1, unroll=1):
                    if s_own[stage * K + s] == 0:
                        bar = cute.make_ptr(
                            Int64,
                            bars_base + (stage * K + s) * 8,
                            cute.AddressSpace.smem,
                            assumed_align=8,
                        )
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(bar, wire)
                        if cutlass.const_expr(self.wait == _WAIT_POLL):
                            cute.copy(
                                tma_atom,
                                tIgI[(None, s_row[stage * K + s])],
                                tIsI[(None, stage * NS + s)],
                                tma_bar_ptr=bar,
                                report_valid=pattern,
                            )
                        else:
                            cute.copy(
                                tma_atom,
                                tIgI[(None, s_row[stage * K + s])],
                                tIsI[(None, stage * NS + s)],
                                tma_bar_ptr=bar,
                            )
                tr = tr2
                di = di2
                tr2 = Int32(-1)
                di2 = Int32(-1)
                if i + 2 < ntok:
                    if lane < K:
                        rec = (Int64(cta) + Int64(i + 2) * G) * K + lane
                        tr2 = _ld_b32(my_ws + L["topk_ranks"] + rec * 4)
                        di2 = _ld_b32(my_ws + L["topk_indices"] + rec * 4)
            for j in cutlass.range(cutlass.max(ntok - (S - 1), Int32(0)), ntok, 1, unroll=1):
                self._wait_stage(
                    j % S,
                    tma_atom,
                    tIgI,
                    tIsI,
                    s_row,
                    s_own,
                    s_ph,
                    s_n,
                    s_has_own,
                    bars_base,
                    bars_own_base,
                    full_base,
                    lane,
                    timeout_ns,
                    pattern,
                )
        elif warp < C // 32:
            # Reduce warps: the stock-order row sum of a token into sX[i & 1], released to the norm warps.
            for i in cutlass.range(0, ntok, 1, unroll=1):
                stage = i % S
                buf = i & 1
                _mbar_wait(full_base + stage * 8, (i // S) & 1)
                if i >= 2:
                    _mbar_wait(free_base + buf * 8, ((i - 2) // 2) & 1)
                count = s_n[stage]
                if cutlass.const_expr(self.diag != "noreduce"):
                    for q in cutlass.range_constexpr(self.vpt):
                        self._reduce_vec_smem(
                            sX_base + buf * (2 * H),
                            tidx + q * C,
                            rows_base + stage * (NS * wire),
                            own_base_s + stage * nbytes_own,
                            s_own,
                            stage * K,
                            count,
                        )
                cute.arch.sync_warp()
                if lane == 0:
                    _mbar_arrive(red_base + buf * 8)
                    _mbar_arrive(empty_base + stage * 8)
        else:
            # Norm warps: RMSNorm of sX[i & 1] to out, then the consumer re-arm of the token's remote rows. The
            # row list is taken into lane registers so the stage is released right away.
            nt = tidx - C
            for i in cutlass.range(0, ntok, 1, unroll=1):
                stage = i % S
                tok = cta + i * G
                buf = i & 1
                _mbar_wait(red_base + buf * 8, (i // 2) & 1)
                count = s_n[stage]
                row = Int64(0)
                if lane < count:
                    row = Int64(s_row[stage * K + lane]) * wire
                    if s_own[stage * K + lane] != 0:
                        row = Int64(-1)
                cute.arch.sync_warp()
                if lane == 0:
                    _mbar_arrive(empty_base + stage * 8)
                if cutlass.const_expr(self.diag not in ("nonorm", "noreduce")):
                    self._norm_row(
                        sX_base + buf * (2 * H),
                        out_base + Int64(tok) * (2 * H),
                        nt,
                        warp - C // 32,
                        lane,
                        tWrW,
                        partials,
                        buf,
                        eps,
                        ready_addr,
                        tok,
                        free_base + buf * 8,
                    )
                else:
                    if nt == 0:
                        _mbar_arrive(free_base + buf * 8)
                if cutlass.const_expr(self.rearm == _REARM_BULK):
                    for s in cutlass.range(0, count, 1, unroll=1):
                        r = cute.arch.shuffle_sync(row, s)
                        if r >= 0:
                            for k in cutlass.range_constexpr(
                                (wire // 16 + _THREADS_PER_ROW - 1) // _THREADS_PER_ROW
                            ):
                                c = nt + k * _THREADS_PER_ROW
                                if c < wire // 16:
                                    _st_v4(
                                        inbox + r + c * 16, sentinel, sentinel, sentinel, sentinel
                                    )
        cute.arch.griddepcontrol_wait()

    @cute.jit
    def _wait_stage(
        self,
        stage,
        tma_atom,
        tIgI,
        tIsI,
        s_row,
        s_own,
        s_ph,
        s_n,
        s_has_own,
        bars_base,
        bars_own_base,
        full_base,
        lane,
        timeout_ns,
        pattern,
    ):
        """Producer warp: every row of ``stage`` landed (FGS on: and valid, re-issuing invalid rows), then full."""
        K = self.top_k
        S = self.stages
        NS = self.slots
        wire = self.wire
        count = s_n[stage]
        for s in cutlass.range(0, count, 1, unroll=1):
            idx = stage * K + s
            if s_own[idx] == 0:
                bar = cute.make_ptr(
                    Int64, bars_base + idx * 8, cute.AddressSpace.smem, assumed_align=8
                )
                ph = s_ph[idx]
                invalid = self._row_invalid(bar, ph)
                if cutlass.const_expr(self.wait == _WAIT_POLL):
                    start = _globaltimer()
                    delay = Int32(64)
                    while invalid != 0:
                        if _globaltimer() - start > timeout_ns:
                            _trap()
                        _nanosleep(delay)
                        delay = cutlass.min(delay * 2, Int32(self.poll_ns))
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(bar, wire)
                        cute.copy(
                            tma_atom,
                            tIgI[(None, s_row[idx])],
                            tIsI[(None, stage * NS + s)],
                            tma_bar_ptr=bar,
                            report_valid=pattern,
                        )
                        ph = ph ^ 1
                        invalid = self._row_invalid(bar, ph)
                if lane == 0:
                    s_ph[idx] = ph ^ 1
        if cutlass.const_expr(not self.own_in_inbox):
            if s_has_own[stage] != 0:
                ph = s_ph[S * K + stage]
                _mbar_wait(bars_own_base + stage * 8, ph)
                if lane == 0:
                    s_ph[S * K + stage] = ph ^ 1
        cute.arch.sync_warp()
        if lane == 0:
            _mbar_arrive(full_base + stage * 8)

    @cute.jit
    def _norm_row(
        self, sx, out_row, nt, nwarp, lane, tWrW, partials, buf, eps, ready_addr, tok, free_bar
    ):
        """FlashInfer's RMSNorm on 64 threads (thread t: elements 8 t + 512 j, j < 7) from the bf16 row sum in shared
        memory, element by element so that only the bf16 words stay live: the sum of squares runs in the fragment
        order of ``latent.rmsnorm``, then ``bf16((x * rstd) * (w + 0))`` with one RN cast per element. The sX buffer
        is released to the reduce warps once every norm thread holds its words."""
        H = self.H
        xw = [_ld_shared_v4(sx + nt * 16 + j * (_THREADS_PER_ROW * _VEC * 2)) for j in range(7)]
        sq = Float32(0.0)
        for j in cutlass.range_constexpr(7):
            for q in cutlass.range_constexpr(4):
                lo, hi = _bf16x2_to_f32(xw[j][q])
                sq = sq + lo * lo
                sq = sq + hi * hi
        for b in cutlass.range_constexpr(5):
            sq = sq + cute.arch.shuffle_sync_bfly(sq, offset=1 << b)
        if lane == 0:
            partials[buf, nwarp] = sq
        cute.arch.barrier(barrier_id=_NORM_BARRIER, number_of_threads=_THREADS_PER_ROW)
        if nt == 0:
            _mbar_arrive(free_bar)
        sum_sq = partials[buf, 0] + partials[buf, 1]
        rstd = cute.math.rsqrt(sum_sq / Float32(H) + eps, fastmath=True)
        for j in cutlass.range_constexpr(7):
            ys = []
            for q in cutlass.range_constexpr(4):
                lo, hi = _bf16x2_to_f32(xw[j][q])
                wlo = tWrW[8 * j + 2 * q].to(Float32)
                whi = tWrW[8 * j + 2 * q + 1].to(Float32)
                ys.append(
                    _f32x2_to_bf16x2(
                        lo * rstd * (wlo + Float32(0.0)), hi * rstd * (whi + Float32(0.0))
                    )
                )
            _st_v4(
                out_row + nt * 16 + j * (_THREADS_PER_ROW * _VEC * 2), ys[0], ys[1], ys[2], ys[3]
            )
        if ready_addr != 0:
            cute.arch.barrier(barrier_id=_NORM_BARRIER, number_of_threads=_THREADS_PER_ROW)
            if nt == 0:
                _st_release_b32(ready_addr + Int64(tok) * 4, Int32(1))

    @cute.jit
    def _sum(self, raw: cute.Tensor, count: Int32, res: cute.Tensor):
        ns = self.ns
        p0 = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        gs = cute.make_rmem_tensor(cute.make_layout((8, ns // 8)), Float32)
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
    def _reduce_vec_smem(self, sx_base, v, rows_base, own_s, s_own, lbase, count):
        lpc = self.lpc
        ns = self.ns
        raw = cute.make_rmem_tensor(cute.make_layout((2 if lpc else 4, ns)), Int32)
        res = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        raw.fill(0)
        for s in cutlass.range_constexpr(ns):
            if s < count:
                if cutlass.const_expr(not self.own_in_inbox):
                    if s_own[lbase + s] != 0:
                        w = _ld_shared_v4(own_s + v * 16)
                        if cutlass.const_expr(lpc):
                            raw[0, s] = _bf16x4_to_e4m3x4(w[0], w[1])
                            raw[1, s] = _bf16x4_to_e4m3x4(w[2], w[3])
                        else:
                            for q in cutlass.range_constexpr(4):
                                raw[q, s] = w[q]
                    elif cutlass.const_expr(lpc):
                        e = _ld_shared_v2(rows_base + s * self.wire + v * 8)
                        raw[0, s] = e[0]
                        raw[1, s] = e[1]
                    else:
                        w = _ld_shared_v4(rows_base + s * self.wire + v * 16)
                        for q in cutlass.range_constexpr(4):
                            raw[q, s] = w[q]
                elif cutlass.const_expr(lpc):
                    e = _ld_shared_v2(rows_base + s * self.wire + v * 8)
                    raw[0, s] = e[0]
                    raw[1, s] = e[1]
                else:
                    w = _ld_shared_v4(rows_base + s * self.wire + v * 16)
                    for q in cutlass.range_constexpr(4):
                        raw[q, s] = w[q]
        self._sum(raw, count, res)
        _st_shared_v4(
            sx_base + v * 16,
            _f32x2_to_bf16x2(res[0], res[1]),
            _f32x2_to_bf16x2(res[2], res[3]),
            _f32x2_to_bf16x2(res[4], res[5]),
            _f32x2_to_bf16x2(res[6], res[7]),
        )


def _smem_bytes(hidden, wire, top_k, slots, own_bytes, stages) -> int:
    lists = 4 * (3 * stages * top_k + 3 * stages + 2)
    bars = 8 * (stages * top_k + 3 * stages + 4)
    rows = stages * (slots * wire + own_bytes)
    return lists + bars + 4 * hidden + rows + 16 * 12 + 128 * 3


def geometry(pc, own_in_inbox: bool):
    """(stages, consumer threads, CTAs per SM): the deepest pipeline that fits two CTAs per SM, else one."""
    slots = min(pc.top_k, pc.ep)
    own_bytes = 0 if own_in_inbox else 2 * pc.hidden
    cthreads = _env_int(THREADS_ENV, 224, 32, 448)
    if 448 % cthreads or cthreads % 32:
        raise ValueError(f"{THREADS_ENV}={cthreads}; valid: a multiple of 32 dividing 448")
    stages_env = _env_int(STAGES_ENV, 0, 0, 8)
    ctas_env = _env_int(CTAS_PER_SM_ENV, 0, 0, 4)
    for ctas in (ctas_env,) if ctas_env else (2, 1):
        budget = min(_SMEM_PER_CTA, _SMEM_PER_SM // ctas - _SMEM_RESERVED)
        stages = stages_env
        if not stages:
            stages = 4
            while (
                stages > 2
                and _smem_bytes(pc.hidden, pc.wire, pc.top_k, slots, own_bytes, stages) > budget
            ):
                stages -= 1
        if _smem_bytes(pc.hidden, pc.wire, pc.top_k, slots, own_bytes, stages) <= budget:
            return stages, cthreads, ctas
    raise ValueError(
        f"the pipelined reduce does not fit shared memory: ep {pc.ep}, wire {pc.wire}, {STAGES_ENV}={stages_env}"
    )


_compiled = {}
_inbox_tensors = {}


def reduce_norm_pipe(
    pc, out, norm_weight, eps, ready, n, mt, blocks, y_ptr, rearm, own_in_inbox, wait
):
    rearm_mode = {"consumer": _REARM_BULK, "consumer_st": _REARM_BULK, "producer": 0, "none": 0}[
        rearm
    ]
    stages, cthreads, ctas_per_sm = geometry(pc, own_in_inbox)
    dtype = torch.float8_e4m3fn if pc.lpc else torch.bfloat16
    rows_total = 2 * pc.ep * pc.rows
    # mode 0 contiguous (CuTe's TMA convention): element (c, j, row) at row * 3584 + 256 j + c
    view = (
        pc.side[pc.rank][_HDR : _HDR + rows_total * pc.wire]
        .view(dtype)
        .view(rows_total, 14, 256)
        .permute(2, 1, 0)
    )
    key = (
        pc.ep,
        pc.top_k,
        pc.hidden,
        pc.lpc,
        pc.lay.key(),
        pc.rows,
        pc.inbox_bytes,
        rearm_mode,
        own_in_inbox,
        wait,
        stages,
        cthreads,
        ctas_per_sm,
        os.environ.get(DIAG_ENV, "none"),
        tuple(view.shape),
    )
    fn = _compiled.get(key)
    mInbox = _inbox_tensors.get(view.data_ptr())
    if mInbox is None:
        mInbox = _inbox_tensors[view.data_ptr()] = cute.runtime.from_dlpack(
            view, assumed_align=16, enable_tvm_ffi=True
        )
    if fn is None:
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kern = _ReduceNormPipe(
            pc.ep,
            pc.top_k,
            pc.hidden,
            pc.lpc,
            pc.lay.key(),
            pc.rows,
            pc.inbox_bytes,
            wait,
            rearm_mode,
            wait != _WAIT_POLL,
            _env_int("TLLM_K3_PC_POLL_NS", 256, 32, 1 << 20),
            ctas_per_sm,
            True,
            own_in_inbox,
            stages=stages,
            cthreads=cthreads,
            ctas_per_sm=ctas_per_sm,
            diag=os.environ.get(DIAG_ENV, "none"),
        )
        fake = cute.runtime.make_fake_compact_tensor
        fn = _compiled[key] = cute.compile(
            kern,
            fake(BFloat16, (cute.sym_int(), pc.hidden), stride_order=(1, 0), assumed_align=16),
            fake(BFloat16, (pc.hidden,), assumed_align=16),
            mInbox,
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(0),
            Int32(1),
            Int32(1),
            Int64(0),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(1),
            Int32(1),
            Float32(1e-6),
            Int64(0),
            stream,
            options="--enable-tvm-ffi",
        )
    ws_base, ws_stride = pc._ws()
    inbox_base = pc.side.data_ptr() + pc.rank * pc.side.stride(0) + _HDR
    n_ctas = max(1, min(n, ctas_per_sm * num_sms(out.device.index)))
    fn(
        out,
        norm_weight,
        mInbox,
        ready.data_ptr() if ready is not None else 0,
        ws_base,
        ws_stride,
        pc.rank,
        n,
        mt,
        y_ptr,
        inbox_base,
        pc.side.data_ptr(),
        pc.side.stride(0),
        blocks,
        n_ctas,
        eps,
        _TIMEOUT_NS,
    )
