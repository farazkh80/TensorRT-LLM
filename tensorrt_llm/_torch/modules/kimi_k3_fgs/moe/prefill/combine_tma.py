# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-checked consumer of the prefill Lamport inbox (``TLLM_K3_PC_CHECK=hw``): every remote row of a token is
loaded into shared memory by TMA with ``report_valid`` (0xFF per element, the inbox sentinel) on its own v1
barrier; a row that landed with the sentinel is re-issued after a backoff, so only invalid rows are retried. The
sum, the latent RMSNorm and the consumer re-arm are those of ``combine._ReduceNorm``.

DSL 4.9 on SM107: the blocking ``mbarrier_wait_primary`` never returns (try-wait loop); the TMA ``cute.copy`` is
issued by the whole warp (the atom elects internally), never inside ``elect_one``.
"""

from __future__ import annotations

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
    _bf16x4_to_e4m3x4,
    _bulk_commit,
    _bulk_s2g,
    _f32x2_to_bf16x2,
    _fence_acq_rel_sys,
    _fence_proxy_async_shared,
    _globaltimer,
    _ld_b32,
    _ld_v4,
    _nanosleep,
    _spin_eq,
    _st_shared_v4,
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
    _WAIT_DONE,
    _WAIT_POLL,
    REDUCE_MIN_BLOCKS_ENV,
    _bulk_wait_all,
    _env_int,
    _ReduceNorm,
    _sentinel,
)

_ROW_BOX = (256, 14, 1)


@dsl_user_op
def _ld_shared_v2(addr, *, loc=None, ip=None):
    res = _asm(
        cutlass._mlir.dialects.llvm.StructType.get_literal([T.i32()] * 2),
        [_v(addr, Int32, loc, ip)],
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        loc,
        ip,
    )
    return [
        Int32(cutlass._mlir.dialects.llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
        for i in range(2)
    ]


@dsl_user_op
def _ld_shared_v4(addr, *, loc=None, ip=None):
    res = _asm(
        cutlass._mlir.dialects.llvm.StructType.get_literal([T.i32()] * 4),
        [_v(addr, Int32, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
    )
    return [
        Int32(cutlass._mlir.dialects.llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


class _ReduceNormTma(_ReduceNorm):
    def __init__(self, *cfg):
        super().__init__(*cfg)
        assert self.H == 14 * 256
        self.smem_bytes += self.top_k * self.wire + 8 * self.top_k + 256

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
            row_layout,
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
        row_layout: cute.Layout,
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
        s_row = smem.allocate_tensor(Int32, cute.make_layout((32,)), byte_alignment=16)
        s_own = smem.allocate_tensor(Int32, cute.make_layout((32,)), byte_alignment=16)
        s_n = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
        partials = smem.allocate_tensor(Float32, cute.make_layout((1, 2)), byte_alignment=16)
        sX = smem.allocate_tensor(
            BFloat16, cute.make_layout(tiler_mn, stride=(H, 1)), byte_alignment=16
        )
        bars = smem.allocate_array(Int64, K, byte_alignment=8)
        rows_buf = smem.allocate_array(cutlass.Uint8, K * wire, byte_alignment=128)
        sent = Int32(0)
        if cutlass.const_expr(self.rearm == _REARM_BULK):
            sbuf = smem.allocate_array(cutlass.Uint8, wire, byte_alignment=128)
            sent = Int32(sbuf.toint())
        rows_base = Int32(rows_buf.toint())
        dtype = cutlass.Float8E4M3FN if self.lpc else cutlass.BFloat16
        sRows = cute.make_tensor(
            cute.recast_ptr(rows_buf, dtype=dtype), cute.make_layout((256, 14, 1, K))
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
        sentinel = Int32(_sentinel(self.lpc))
        pattern = cpasync.SentinelPattern.PER_ELEMENT_FF

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

        if tidx == 0:
            for s in cutlass.range_constexpr(K):
                cute.arch.mbarrier_v1_init(bars + s, 1)
            cute.arch.mbarrier_init_fence()
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
        if cutlass.const_expr(self.wait == _WAIT_DONE):
            my_side = side_base + Int64(rank) * side_stride
            for i in cutlass.range(tidx, self.ep * blocks, self.threads, unroll=1):
                src = i // blocks
                if src != rank:
                    _spin_eq(
                        my_side + _DONE + (src * _DONE_BLOCKS + i % blocks) * 4, value, timeout_ns
                    )
            if tidx < self.ep * blocks:
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
                    row = parity * (self.ep * self.rows) + tr * self.rows + di
                    if cutlass.const_expr(not self.own_in_inbox):
                        if tr == rank:
                            own = Int32(1)
                    s_row[at] = row
                    s_own[at] = own
                    s_src[at] = own_base + (Int64(rank) * max_tokens + di) * (H * 2)
        cute.arch.sync_threads()

        if tok < n_local:
            count = s_n[0]
            if warp == 0:
                for s in cutlass.range_constexpr(K):
                    if s < count:
                        if s_own[s] == 0:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(bars + s, wire)
                            if cutlass.const_expr(self.wait == _WAIT_POLL):
                                cute.copy(
                                    tma_atom,
                                    tIgI[(None, s_row[s])],
                                    tIsI[(None, s)],
                                    tma_bar_ptr=bars + s,
                                    report_valid=pattern,
                                )
                            else:
                                cute.copy(
                                    tma_atom,
                                    tIgI[(None, s_row[s])],
                                    tIsI[(None, s)],
                                    tma_bar_ptr=bars + s,
                                )
                for s in cutlass.range_constexpr(K):
                    if s < count:
                        if s_own[s] == 0:
                            phase = Int32(0)
                            invalid = self._row_invalid(bars + s, phase)
                            if cutlass.const_expr(self.wait != _WAIT_POLL):
                                invalid = Int32(0)
                            start = _globaltimer()
                            delay = Int32(64)
                            while invalid != 0:
                                if _globaltimer() - start > timeout_ns:
                                    _trap()
                                _nanosleep(delay)
                                delay = cutlass.min(delay * 2, Int32(self.poll_ns))
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive_and_expect_tx(bars + s, wire)
                                cute.copy(
                                    tma_atom,
                                    tIgI[(None, s_row[s])],
                                    tIsI[(None, s)],
                                    tma_bar_ptr=bars + s,
                                    report_valid=pattern,
                                )
                                phase = phase ^ 1
                                invalid = self._row_invalid(bars + s, phase)
            cute.arch.sync_threads()
            self._reduce_vec_smem(sX, tidx, rows_base, s_src, s_own, count)
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
            # Consumer re-arm after the token's output writes: the remote rows' global addresses.
            if cutlass.const_expr(self.rearm == _REARM_BULK):
                cute.arch.sync_threads()
                if tidx == 0:
                    for s in cutlass.range(0, count, 1, unroll=1):
                        if s_own[s] == 0:
                            _bulk_s2g(inbox + Int64(s_row[s]) * wire, sent, Int32(wire))
                    _bulk_commit()
        if cutlass.const_expr(self.rearm == _REARM_BULK):
            if tidx == 0:
                _bulk_wait_all()
        cute.arch.griddepcontrol_wait()

    @cute.jit
    def _reduce_vec_smem(self, sX, v, rows_base, s_src, s_own, count):
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
                elif cutlass.const_expr(lpc):
                    e = _ld_shared_v2(rows_base + s * self.wire + v * 8)
                    raw[0, s] = e[0]
                    raw[1, s] = e[1]
                else:
                    w = _ld_shared_v4(rows_base + s * self.wire + v * 16)
                    for q in cutlass.range_constexpr(4):
                        raw[q, s] = w[q]
        self._sum(raw, count, res)
        for q in cutlass.range_constexpr(4):
            words[0, v * 4 + q] = _f32x2_to_bf16x2(res[2 * q], res[2 * q + 1])


_compiled = {}


def reduce_norm_tma(
    pc, out, norm_weight, eps, ready, n, mt, blocks, y_ptr, rearm, own_in_inbox, wait
):
    rearm_mode = {"consumer": _REARM_BULK, "consumer_st": _REARM_BULK, "producer": 0, "none": 0}[
        rearm
    ]
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
        tuple(view.shape),
    )
    fn = _compiled.get(key)
    mInbox = cute.runtime.from_dlpack(view, assumed_align=16, enable_tvm_ffi=True)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if fn is None:
        kern = _ReduceNormTma(
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
            _env_int(REDUCE_MIN_BLOCKS_ENV, 2, 0, 4),
            True,
            own_in_inbox,
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
            Float32(1e-6),
            Int64(0),
            stream,
            options="--enable-tvm-ffi",
        )
    ws_base, ws_stride = pc._ws()
    inbox_base = pc.side.data_ptr() + pc.rank * pc.side.stride(0) + _HDR
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
        eps,
        _TIMEOUT_NS,
    )
