# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FC2 -> combine push edge of the prefill Lamport inbox (``combine.PrefillCombine``).

A persistent push kernel on ``ctas`` reserved SMs (FC2 runs on the others and triggers its dependents at entry)
walks FC2's m-tiles, waits for each tile's completion counter (``target`` (m, n) items, published by FC2 with release
semantics) and pushes the tile's finished rows to their source ranks' inboxes: a row whose token has exactly one
local pair is complete in ``out[token]`` once its m-tile is done (FC2's fused finalize), so it needs no finalize
pass. Rows of multi-pair tokens are left to the finalize and a later push. The push-start flag of the stock protocol
("this rank no longer reads its receive region") and the round-flag advance come at the end, after the grid
dependency, since FC2 still reads the dispatched rows while this kernel runs. FGS off is the same kernel after FC2
(grid dependency at entry, no counter waits, completion words for the reduce).
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.utils import SmemAllocator

from .. import comm as _comm
from ..comm import (
    _CACHELINE,
    _TIMEOUT_NS,
    _bf16x4_to_e4m3x4,
    _bulk_commit,
    _bulk_s2g,
    _bulk_wait_read,
    _fence_acq_rel_sys,
    _fence_proxy_async_global,
    _fence_proxy_async_shared,
    _globaltimer,
    _ld_acquire_b32,
    _ld_b32,
    _ld_v4,
    _mbar_init,
    _nanosleep,
    _spin_eq,
    _st_any,
    _st_b32,
    _st_sys_b32,
    _trap,
)
from .combine import (
    _DONE,
    _DONE_BLOCKS,
    _HDR,
    _MAX_RANKS,
    PUSH_LOAD_ENV,
    PUSH_WARPS_ENV,
    PrefillCombine,
    _bulk_wait_all,
    _env_int,
    _tma_row,
)
from .common import HIDDEN, TOP_K
from .fc2 import M_TILE

PUSH_CTAS_ENV = "TLLM_K3_PC_TILE_PUSH_CTAS"


@cute.jit
def _wait_counter(addr: Int64, target: Int32, timeout_ns: Int64):
    start = _globaltimer()
    delay = Int32(64)
    while _ld_acquire_b32(addr) < target:
        if _globaltimer() - start > timeout_ns:
            _trap()
        _nanosleep(delay)
        delay = cutlass.min(delay * 2, Int32(1024))


class _TilePush:
    def __init__(
        self,
        ep: int,
        hidden: int,
        lpc: bool,
        lay: tuple,
        rows: int,
        inbox_bytes: int,
        warps: int,
        target: int,
        wait_counters: bool,
        data_only: bool,
        pdl: bool,
        tma: bool = False,
    ):
        assert hidden % 512 == 0 and ep <= _MAX_RANKS
        self.ep, self.hidden, self.lpc, self.lay = ep, hidden, lpc, dict(lay)
        self.rows, self.inbox_bytes, self.warps = rows, inbox_bytes, warps
        self.target, self.wait_counters, self.data_only, self.pdl = (
            target,
            wait_counters,
            data_only,
            pdl,
        )
        self.tma = tma
        self.wire = hidden if lpc else 2 * hidden
        self.smem_bytes = warps * (2 * hidden + 8) + 128 if tma else warps * self.wire + 128

    @cute.jit
    def __call__(
        self,
        mDummy: cute.Tensor,
        out: Int64,
        p2e: Int64,
        e2p: Int64,
        limits: Int64,
        ntiles: Int64,
        counters: Int64,
        ws_base: Int64,
        ws_stride: Int64,
        side_base: Int64,
        side_stride: Int64,
        rank: Int32,
        max_tokens: Int32,
        ctas: Int32,
        timeout_ns: Int64,
        stream,
    ):
        self.kernel(
            out,
            p2e,
            e2p,
            limits,
            ntiles,
            counters,
            ws_base,
            ws_stride,
            side_base,
            side_stride,
            rank,
            max_tokens,
            timeout_ns,
        ).launch(
            grid=[ctas, 1, 1],
            block=[self.warps * 32, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.jit
    def _store_row(self, row: Int64, dst: Int64, stage: Int32, lane: Int32):
        per_lane = self.wire // 16 // 32
        if cutlass.const_expr(self.lpc):
            loads = [
                (_ld_v4(row + (lane + 32 * k) * 32), _ld_v4(row + (lane + 32 * k) * 32 + 16))
                for k in range(per_lane)
            ]
            for k in cutlass.range_constexpr(per_lane):
                p, q = loads[k]
                _st_any(
                    True,
                    dst,
                    stage,
                    (lane + 32 * k) * 16,
                    _bf16x4_to_e4m3x4(p[0], p[1]),
                    _bf16x4_to_e4m3x4(p[2], p[3]),
                    _bf16x4_to_e4m3x4(q[0], q[1]),
                    _bf16x4_to_e4m3x4(q[2], q[3]),
                )
        else:
            loads = [_ld_v4(row + (lane + 32 * k) * 16) for k in range(per_lane)]
            for k in cutlass.range_constexpr(per_lane):
                w = loads[k]
                _st_any(
                    True,
                    dst,
                    stage,
                    (lane + 32 * k) * 16,
                    w[0],
                    w[1],
                    w[2],
                    w[3],
                )

    @cute.kernel
    def kernel(
        self,
        out: Int64,
        p2e: Int64,
        e2p: Int64,
        limits: Int64,
        ntiles: Int64,
        counters: Int64,
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
        cta, _, _ = cute.arch.block_idx()
        ncta, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        my_ws = ws_base + Int64(rank) * ws_stride
        nbytes = self.hidden * 2
        smem = SmemAllocator()
        mbar = Int32(0)
        ph = Int32(0)
        if cutlass.const_expr(self.tma):
            buf = smem.allocate_array(
                cutlass.Uint8, self.warps * 2 * self.hidden, byte_alignment=128
            )
            bars = smem.allocate_array(Int64, self.warps, byte_alignment=8)
            stage = Int32(buf.toint()) + warp * (2 * self.hidden)
            mbar = Int32(bars.toint()) + warp * 8
            if lane == 0:
                _mbar_init(mbar, 1)
            cute.arch.sync_threads()
        else:
            buf = smem.allocate_array(cutlass.Uint8, self.warps * wire, byte_alignment=128)
            stage = Int32(buf.toint()) + warp * wire

        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(not self.wait_counters):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.tma):
                _fence_proxy_async_global()
        f = _ld_b32(my_ws + L["flag"])
        value = ((f - 1) | 1) + 1
        parity = ((f - 1) >> 1) & 1
        box = _HDR + Int64(parity) * self.inbox_bytes + Int64(rank) * self.rows * wire
        n_tiles = _ld_b32(ntiles)
        for m in cutlass.range(cta, n_tiles, ncta, unroll=1):
            if cutlass.const_expr(self.wait_counters):
                if tidx == 0:
                    _wait_counter(counters + Int64(m) * 4, Int32(self.target), timeout_ns)
            cute.arch.sync_threads()
            if cutlass.const_expr(self.tma and self.wait_counters):
                # FC2's generic-proxy stores of this tile, acquired above, must be visible to the TMA reads.
                _fence_proxy_async_global()
            hi = cutlass.min(_ld_b32(limits + Int64(m) * 4), (m + 1) * M_TILE)
            for p in cutlass.range(m * M_TILE + warp, hi, self.warps, unroll=1):
                expanded = _ld_b32(p2e + Int64(p) * 4)
                if expanded >= 0:
                    token = expanded // TOP_K
                    pair = Int32(0)
                    if lane < TOP_K:
                        if _ld_b32(e2p + (Int64(token) * TOP_K + lane) * 4) >= 0:
                            pair = Int32(1)
                    npairs = cute.arch.popc(cute.arch.vote_ballot_sync(pair != 0))
                    if npairs == 1:
                        # Own rows go into this rank's inbox too: the polling reduce starts under FC2 and may not
                        # read ``out`` directly.
                        source = token // max_tokens
                        slot = token - source * max_tokens
                        dst = side_base + Int64(source) * side_stride + box + Int64(slot) * wire
                        if cutlass.const_expr(self.tma):
                            ph = _tma_row(
                                self.lpc,
                                self.hidden,
                                out + Int64(token) * nbytes,
                                dst,
                                stage,
                                mbar,
                                ph,
                                lane,
                            )
                        else:
                            if lane == 0:
                                _bulk_wait_read()
                            cute.arch.sync_warp()
                            self._store_row(out + Int64(token) * nbytes, dst, stage, lane)
                            _fence_proxy_async_shared()
                            cute.arch.sync_warp()
                            if lane == 0:
                                _bulk_s2g(dst, stage, Int32(wire))
                                _bulk_commit()
        if lane == 0:
            if cutlass.const_expr(self.data_only):
                _bulk_wait_read()
            else:
                _bulk_wait_all()
                _fence_proxy_async_global()
        cute.arch.sync_threads()
        # FC2 reads the dispatched rows until it completes: only now may the peers overwrite them.
        cute.arch.griddepcontrol_wait()
        if warp == 0:
            if lane < self.ep:
                if cutlass.const_expr(not self.data_only):
                    _fence_acq_rel_sys()
                    if lane != rank:
                        _st_sys_b32(
                            side_base
                            + Int64(lane) * side_stride
                            + _DONE
                            + (rank * _DONE_BLOCKS + cta) * 4,
                            value,
                        )
                if cta == 0:
                    _st_sys_b32(
                        ws_base + Int64(lane) * ws_stride + L["combine_flags"] + rank * 4, value
                    )
                    _spin_eq(my_ws + L["combine_flags"] + lane * 4, value, timeout_ns)
            cute.arch.sync_warp()
            if cta == 0:
                if lane == 0:
                    _st_b32(my_ws + L["flag"], value)


class _RestPush(_TilePush):
    """After the finalize: the rows of tokens with two or more local pairs (the tile push skipped them), own rows
    included, into the same round's inboxes. No round duties: the tile push did them."""

    @cute.jit
    def __call__(
        self,
        mDummy: cute.Tensor,
        out: Int64,
        e2p: Int64,
        ws_base: Int64,
        ws_stride: Int64,
        side_base: Int64,
        side_stride: Int64,
        rank: Int32,
        max_tokens: Int32,
        n_tokens: Int32,
        ctas: Int32,
        stream,
    ):
        self.kernel(
            out, e2p, ws_base, ws_stride, side_base, side_stride, rank, max_tokens, n_tokens
        ).launch(
            grid=[ctas, 1, 1],
            block=[self.warps * 32, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        out: Int64,
        e2p: Int64,
        ws_base: Int64,
        ws_stride: Int64,
        side_base: Int64,
        side_stride: Int64,
        rank: Int32,
        max_tokens: Int32,
        n_tokens: Int32,
    ):
        L = self.lay
        wire = self.wire
        cta, _, _ = cute.arch.block_idx()
        ncta, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        my_ws = ws_base + Int64(rank) * ws_stride
        nbytes = self.hidden * 2
        smem = SmemAllocator()
        mbar = Int32(0)
        ph = Int32(0)
        if cutlass.const_expr(self.tma):
            buf = smem.allocate_array(
                cutlass.Uint8, self.warps * 2 * self.hidden, byte_alignment=128
            )
            bars = smem.allocate_array(Int64, self.warps, byte_alignment=8)
            stage = Int32(buf.toint()) + warp * (2 * self.hidden)
            mbar = Int32(bars.toint()) + warp * 8
            if lane == 0:
                _mbar_init(mbar, 1)
            cute.arch.sync_threads()
        else:
            buf = smem.allocate_array(cutlass.Uint8, self.warps * wire, byte_alignment=128)
            stage = Int32(buf.toint()) + warp * wire

        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.tma):
            _fence_proxy_async_global()
        cute.arch.griddepcontrol_launch_dependents()
        f = _ld_b32(my_ws + L["flag"])
        parity = ((f - 1) >> 1) & 1
        box = _HDR + Int64(parity) * self.inbox_bytes + Int64(rank) * self.rows * wire
        for token in cutlass.range(cta * self.warps + warp, n_tokens, ncta * self.warps, unroll=1):
            pair = Int32(0)
            if lane < TOP_K:
                if _ld_b32(e2p + (Int64(token) * TOP_K + lane) * 4) >= 0:
                    pair = Int32(1)
            npairs = cute.arch.popc(cute.arch.vote_ballot_sync(pair != 0))
            if npairs >= 2:
                source = token // max_tokens
                slot = token - source * max_tokens
                dst = side_base + Int64(source) * side_stride + box + Int64(slot) * wire
                if cutlass.const_expr(self.tma):
                    ph = _tma_row(
                        self.lpc,
                        self.hidden,
                        out + Int64(token) * nbytes,
                        dst,
                        stage,
                        mbar,
                        ph,
                        lane,
                    )
                else:
                    if lane == 0:
                        _bulk_wait_read()
                    cute.arch.sync_warp()
                    self._store_row(out + Int64(token) * nbytes, dst, stage, lane)
                    _fence_proxy_async_shared()
                    cute.arch.sync_warp()
                    if lane == 0:
                        _bulk_s2g(dst, stage, Int32(wire))
                        _bulk_commit()
        if lane == 0:
            _bulk_wait_read()


@functools.lru_cache(maxsize=None)
def _compile_rest(*cfg):
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    i64, i32 = Int64(0), Int32(0)
    return cute.compile(
        _RestPush(*cfg),
        cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16),
        *([i64] * 6),
        i32,
        Int32(1),
        Int32(1),
        Int32(1),
        stream,
        options="--enable-tvm-ffi",
    )


def push_rest(
    pc: PrefillCombine, out: torch.Tensor, plan, *, ctas: int = 64, warps: Optional[int] = None
) -> None:
    """After the finalize of the same round (``out`` rows of multi-pair tokens complete): push the rows of tokens
    with two or more local pairs (own rows included) into the round's inboxes; then ``pc.reduce_norm(fgs=True)``.
    Must follow ``push_tiles(fgs=True)`` of the round (which did the round duties and set the reduce's mode)."""
    if pc._round is None or pc._round[4] != "poll" or not pc._round[7]:
        raise RuntimeError("push_rest follows push_tiles(fgs=True) of the same round")
    mt = pc._round[1]
    warps = _env_int(PUSH_WARPS_ENV, 8, 1, 32) if warps is None else warps
    ws_base, ws_stride = pc._ws()
    _compile_rest(
        pc.ep,
        pc.hidden,
        pc.lpc,
        pc.lay.key(),
        pc.rows,
        pc.inbox_bytes,
        warps,
        0,
        False,
        True,
        True,
        os.environ.get(PUSH_LOAD_ENV, "tma") == "tma",
    )(
        pc.side[pc.rank, :_CACHELINE],
        out.data_ptr(),
        plan.expanded_idx_to_permuted_idx.data_ptr(),
        ws_base,
        ws_stride,
        pc.side.data_ptr(),
        pc.side.stride(0),
        pc.rank,
        mt,
        pc.ep * mt,
        ctas,
    )


@functools.lru_cache(maxsize=None)
def _compile(*cfg):
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    i64, i32 = Int64(0), Int32(0)
    return cute.compile(
        _TilePush(*cfg),
        cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16),
        *([i64] * 10),
        i32,
        Int32(1),
        Int32(1),
        i64,
        stream,
        options="--enable-tvm-ffi",
    )


def push_tiles(
    pc: PrefillCombine,
    out: torch.Tensor,
    plan,
    counters: torch.Tensor,
    *,
    fgs: bool = True,
    ctas: Optional[int] = None,
    warps: Optional[int] = None,
    tile_n: int = 256,
    data_only: Optional[bool] = None,
) -> int:
    """Push the single-pair rows of FC2's finished m-tiles into the source ranks' inboxes of ``pc``. ``out`` is FC2's
    ``fin`` output (bf16 ``[ep * runtime_max_tokens, hidden]``), ``plan`` the FC2 plan, ``counters`` int32 per m-tile
    (zeroed per round; FC2 adds one per finished (m, n) item). ``fgs``: True = launched beside FC2 (``ctas`` reserved
    SMs, FC2 with ``num_ctas = SMs - ctas`` and ``pdl_trigger="start"``), waits per tile; False = after FC2, grid
    dependency, completion words. Then ``pc.reduce_norm(..., fgs=fgs)``. Returns the grid size."""
    y, n, mt = pc._args(out)
    data_only = fgs if data_only is None else data_only
    if data_only and not fgs:
        raise ValueError("a data-only push leaves no completion words for the FGS-off reduce")
    ctas = _env_int(PUSH_CTAS_ENV, 8, 1, _DONE_BLOCKS) if ctas is None else ctas
    warps = _env_int(PUSH_WARPS_ENV, 8, 1, 32) if warps is None else warps
    if counters.dtype != torch.int32 or counters.numel() < plan.tile_idx_to_expert_idx.shape[0]:
        raise ValueError("counters must be int32 with one word per m-tile")
    ws_base, ws_stride = pc._ws()
    _compile(
        pc.ep,
        pc.hidden,
        pc.lpc,
        pc.lay.key(),
        pc.rows,
        pc.inbox_bytes,
        warps,
        HIDDEN // tile_n,
        fgs,
        data_only,
        _comm._pdl() or fgs,
        os.environ.get(PUSH_LOAD_ENV, "tma") == "tma",
    )(
        pc.side[pc.rank, :_CACHELINE],
        y.data_ptr(),
        plan.permuted_idx_to_expanded_idx.data_ptr(),
        plan.expanded_idx_to_permuted_idx.data_ptr(),
        plan.tile_idx_to_mn_limit.data_ptr(),
        plan.num_non_exiting_tiles.data_ptr(),
        counters.data_ptr(),
        ws_base,
        ws_stride,
        pc.side.data_ptr(),
        pc.side.stride(0),
        pc.rank,
        mt,
        ctas,
        _TIMEOUT_NS,
    )
    pc._round = (n, mt, y.data_ptr(), ctas, "poll" if fgs else "done", data_only, "consumer", fgs)
    return ctas
