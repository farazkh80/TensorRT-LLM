# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming finalize + push of the prefill FC2 -> combine edge (``TLLM_K3_MOE_PREFILL_PUSH=stream``).

One persistent kernel replaces the tile push, the finalize and the rest push of ``fc2_push``: its warps deal FC2's
permuted rows in index order; a token's row is shipped by the warp holding the token's largest local permuted index
(one owner per token), after acquiring the ``tile_done`` counter of every local pair's m-tile (FGS on) or the FC2
grid (FGS off). The pairs' bf16 ``y`` rows are TMA-loaded into the warp's stages and summed in k order in fp32 with the
finalize's arithmetic (one RN cast to bf16), converted to the wire format in place and TMA-stored into the token's
source inbox; single-pair rows are complete in ``out`` (FC2's fused finalize) and move as in the tile push. Own rows
go into this rank's inbox too, so the reduce runs ``own_in_inbox``. Round duties as in ``fc2_push._TilePush``.
"""

from __future__ import annotations

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from .. import comm as _comm
from ..comm import (
    _CACHELINE,
    _TIMEOUT_NS,
    _asm,
    _bf16x2_to_f32,
    _bf16x4_to_e4m3x4,
    _bulk_commit,
    _bulk_g2s,
    _bulk_s2g,
    _bulk_wait_read,
    _f32x2_to_bf16x2,
    _fence_acq_rel_sys,
    _fence_proxy_async_global,
    _fence_proxy_async_shared,
    _ld_b32,
    _mbar_expect_tx,
    _mbar_init,
    _mbar_wait,
    _spin_eq,
    _st_b32,
    _st_shared_v4,
    _st_sys_b32,
    _v,
)
from .combine import (
    _DONE,
    _DONE_BLOCKS,
    _HDR,
    _MAX_RANKS,
    PrefillCombine,
    _bulk_wait_all,
    _env_int,
    _tma_row,
)
from .combine_tma import _ld_shared_v4
from .common import HIDDEN, TOP_K, bits_f32, num_sms, st_shared_b32
from .fc2 import M_TILE
from .fc2_push import _wait_counter

CTAS_ENV = "TLLM_K3_PC_STREAM_CTAS"
WARPS_ENV = "TLLM_K3_PC_STREAM_WARPS"
ROWS_ENV = "TLLM_K3_PC_STREAM_ROWS"
CHUNK_ENV = "TLLM_K3_PC_STREAM_CHUNK"
_CHUNKS = HIDDEN // 8 // 32


@dsl_user_op
def _atom_add(addr, val, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(addr, Int64, loc, ip), _v(val, Int32, loc, ip)],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_u16(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(addr, Int64, loc, ip)],
            "{\n.reg .b16 h;\nld.global.b16 h, [$1];\ncvt.u32.u16 $0, h;\n}",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_shared_b32(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(T.i32(), [_v(addr, Int32, loc, ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip)
    )


@dsl_user_op
def _st_shared_v2(addr, w0, w1, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, Int32, loc, ip), _v(w0, Int32, loc, ip), _v(w1, Int32, loc, ip)],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
        loc,
        ip,
    )


class _StreamPush:
    def __init__(
        self,
        ep: int,
        hidden: int,
        lpc: bool,
        lay: tuple,
        rows: int,
        inbox_bytes: int,
        warps: int,
        stages: int,
        chunk: int,
        target: int,
        wait_counters: bool,
        grid_wait: bool,
        data_only: bool,
        pdl: bool,
    ):
        assert hidden == HIDDEN and ep <= _MAX_RANKS and 1 <= stages <= TOP_K
        self.ep, self.hidden, self.lpc, self.lay = ep, hidden, lpc, dict(lay)
        self.rows, self.inbox_bytes, self.warps, self.stages, self.chunk = (
            rows,
            inbox_bytes,
            warps,
            stages,
            chunk,
        )
        self.target, self.wait_counters, self.grid_wait = target, wait_counters, grid_wait
        self.data_only, self.pdl = data_only, pdl
        self.wire = hidden if lpc else 2 * hidden
        self.nbytes = 2 * hidden
        self.smem_bytes = warps * (stages * self.nbytes + 8 + 8 * TOP_K) + 128

    @cute.jit
    def __call__(
        self,
        mDummy: cute.Tensor,
        out: Int64,
        y: Int64,
        topk_w: Int64,
        p2e: Int64,
        e2p: Int64,
        limits: Int64,
        ntiles: Int64,
        counters: Int64,
        work: Int64,
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
            y,
            topk_w,
            p2e,
            e2p,
            limits,
            ntiles,
            counters,
            work,
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
    def _sum_row(
        self, y: Int64, pl: Int32, npairs: Int32, stage0: Int32, mbar: Int32, ph: Int32, lane: Int32
    ) -> Int32:
        """The owner's finalize: batches of ``stages`` pair rows TMA-loaded onto the warp's mbarrier, every lane
        accumulating its 112 elements in fp32 in k order; the result lands in stage 0 in the wire format."""
        nbytes = self.nbytes
        acc = cute.make_rmem_tensor(cute.make_layout((_CHUNKS * 8,)), Float32)
        acc.fill(0.0)
        for s0 in cutlass.range(0, npairs, self.stages, unroll=1):
            nb = cutlass.min(Int32(self.stages), npairs - s0)
            if lane == 0:
                _bulk_wait_read()
                _mbar_expect_tx(mbar, nb * nbytes)
            cute.arch.sync_warp()
            if lane < nb:
                prow = _ld_shared_b32(pl + (s0 + lane) * 4)
                _bulk_g2s(stage0 + lane * nbytes, y + Int64(prow) * nbytes, Int32(nbytes), mbar)
            _mbar_wait(mbar, ph)
            ph = ph ^ 1
            for s in cutlass.range(0, nb, 1, unroll=1):
                w = bits_f32(_ld_shared_b32(pl + 4 * TOP_K + (s0 + s) * 4))
                st = stage0 + s * nbytes
                for c in cutlass.range_constexpr(_CHUNKS):
                    v = _ld_shared_v4(st + (c * 32 + lane) * 16)
                    for q in cutlass.range_constexpr(4):
                        lo, hi = _bf16x2_to_f32(v[q])
                        acc[c * 8 + 2 * q] = acc[c * 8 + 2 * q] + lo * w
                        acc[c * 8 + 2 * q + 1] = acc[c * 8 + 2 * q + 1] + hi * w
            cute.arch.sync_warp()
        for c in cutlass.range_constexpr(_CHUNKS):
            words = [_f32x2_to_bf16x2(acc[c * 8 + 2 * q], acc[c * 8 + 2 * q + 1]) for q in range(4)]
            if cutlass.const_expr(self.lpc):
                _st_shared_v2(
                    stage0 + (c * 32 + lane) * 8,
                    _bf16x4_to_e4m3x4(words[0], words[1]),
                    _bf16x4_to_e4m3x4(words[2], words[3]),
                )
            else:
                _st_shared_v4(stage0 + (c * 32 + lane) * 16, words[0], words[1], words[2], words[3])
        _fence_proxy_async_shared()
        cute.arch.sync_warp()
        return ph

    @cute.kernel
    def kernel(
        self,
        out: Int64,
        y: Int64,
        topk_w: Int64,
        p2e: Int64,
        e2p: Int64,
        limits: Int64,
        ntiles: Int64,
        counters: Int64,
        work: Int64,
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
        nbytes = self.nbytes
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        my_ws = ws_base + Int64(rank) * ws_stride
        smem = SmemAllocator()
        buf = smem.allocate_array(
            cutlass.Uint8, self.warps * self.stages * nbytes, byte_alignment=128
        )
        bars = smem.allocate_array(Int64, self.warps, byte_alignment=8)
        lists = smem.allocate_array(Int32, self.warps * 2 * TOP_K, byte_alignment=16)
        stage0 = Int32(buf.toint()) + warp * (self.stages * nbytes)
        mbar = Int32(bars.toint()) + warp * 8
        pl = Int32(lists.toint()) + warp * (8 * TOP_K)
        if lane == 0:
            _mbar_init(mbar, 1)
        cute.arch.sync_threads()

        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.grid_wait):
            cute.arch.griddepcontrol_wait()
            _fence_proxy_async_global()
        f = _ld_b32(my_ws + L["flag"])
        value = ((f - 1) | 1) + 1
        parity = ((f - 1) >> 1) & 1
        box = _HDR + Int64(parity) * self.inbox_bytes + Int64(rank) * self.rows * wire
        n_rows = _ld_b32(ntiles) * M_TILE
        ph = Int32(0)
        base = Int32(0)
        if lane == 0:
            base = _atom_add(work, Int32(self.chunk))
        base = cute.arch.shuffle_sync(base, 0)
        while base < n_rows:
            hi = cutlass.min(base + self.chunk, n_rows)
            for p in cutlass.range(base, hi, 1, unroll=1):
                expanded = _ld_b32(p2e + Int64(p) * 4)
                if p >= _ld_b32(limits + Int64(p // M_TILE) * 4):
                    expanded = Int32(-1)
                if expanded >= 0:
                    token = expanded // TOP_K
                    pk = Int32(-1)
                    wb = Int32(0)
                    if lane < TOP_K:
                        pk = _ld_b32(e2p + (Int64(token) * TOP_K + lane) * 4)
                        wb = _ld_u16(topk_w + (Int64(token) * TOP_K + lane) * 2) << 16
                    valid = cute.arch.vote_ballot_sync(pk >= 0)
                    if cute.arch.warp_redux_sync(pk, "max") == p:
                        npairs = cute.arch.popc(valid)
                        source = token // max_tokens
                        dst = (
                            side_base
                            + Int64(source) * side_stride
                            + box
                            + Int64(token - source * max_tokens) * wire
                        )
                        if cutlass.const_expr(self.wait_counters):
                            if pk >= 0:
                                _wait_counter(
                                    counters + Int64(pk // M_TILE) * 4,
                                    Int32(self.target),
                                    timeout_ns,
                                )
                            cute.arch.sync_warp()
                            # FC2's generic-proxy stores, acquired above, must be visible to the TMA reads.
                            _fence_proxy_async_global()
                        if npairs == 1:
                            ph = _tma_row(
                                self.lpc,
                                self.hidden,
                                out + Int64(token) * nbytes,
                                dst,
                                stage0,
                                mbar,
                                ph,
                                lane,
                            )
                        else:
                            if pk >= 0:
                                at = cute.arch.popc(valid & ((Int32(1) << lane) - 1))
                                st_shared_b32(pl + at * 4, pk)
                                st_shared_b32(pl + 4 * TOP_K + at * 4, wb)
                            cute.arch.sync_warp()
                            ph = self._sum_row(y, pl, npairs, stage0, mbar, ph, lane)
                            if lane == 0:
                                _bulk_s2g(dst, stage0, Int32(wire))
                                _bulk_commit()
            if lane == 0:
                base = _atom_add(work, Int32(self.chunk))
            base = cute.arch.shuffle_sync(base, 0)
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


@functools.lru_cache(maxsize=None)
def _compile(*cfg):
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    i64, i32 = Int64(0), Int32(0)
    return cute.compile(
        _StreamPush(*cfg),
        cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16),
        *([i64] * 13),
        i32,
        Int32(1),
        Int32(1),
        i64,
        stream,
        options="--enable-tvm-ffi",
    )


def push_stream(
    pc: PrefillCombine,
    out: torch.Tensor,
    y: torch.Tensor,
    topk_w: torch.Tensor,
    plan,
    counters: torch.Tensor,
    *,
    fgs: bool = True,
    ctas: Optional[int] = None,
    warps: Optional[int] = None,
    stages: Optional[int] = None,
    chunk: Optional[int] = None,
    tile_n: int = 256,
    data_only: Optional[bool] = None,
    check: bool = True,
) -> int:
    """Finalize + push of every received row into the source ranks' inboxes of ``pc``: ``out`` is FC2's ``fin`` output
    (bf16 ``[ep * runtime_max_tokens, hidden]``, single-pair rows), ``y`` FC2's permuted output (bf16 ``[P, hidden]``,
    multi-pair rows), ``topk_w`` bf16 ``[rows, 16]``, ``plan`` the FC2 plan, ``counters`` int32 ``[ntiles_cap + 1]``
    zeroed by the caller before FC2 in every round (both modes): FC2 adds one per finished (m, n) item to
    ``counters[m]``; ``counters[ntiles_cap]`` deals the rows. ``fgs``: True = launched beside FC2 (``num_ctas = SMs -
    P``, ``pdl_trigger="start"``, ``publish=counters``), waits per row on its pairs' tiles; False = after FC2, grid
    dependency, completion words. ``ctas`` may exceed the reserved SMs (up to all SMs): the extra CTAs become resident
    as FC2 retires. ``check=False`` (diagnostic negative): neither counter waits nor the grid dependency before the
    reads. Then ``pc.reduce_norm(..., fgs=fgs)``. Returns the grid size."""
    yo, n, mt = pc._args(out)
    data_only = fgs if data_only is None else data_only
    if data_only and not fgs:
        raise ValueError("a data-only push leaves no completion words for the FGS-off reduce")
    ntiles_cap = plan.tile_idx_to_expert_idx.shape[0]
    if (
        counters.dtype != torch.int32
        or counters.numel() < ntiles_cap + 1
        or not counters.is_contiguous()
    ):
        raise ValueError(
            "counters must be contiguous int32 with one word per m-tile plus the row-dealing word"
        )
    if (
        y.dtype != torch.bfloat16
        or y.shape[1] != pc.hidden
        or not y.is_contiguous()
        or y.data_ptr() % 16
    ):
        raise ValueError("y must be contiguous bf16 [P, hidden]")
    if (
        topk_w.dtype != torch.bfloat16
        or topk_w.shape != (pc.ep * mt, TOP_K)
        or not topk_w.is_contiguous()
    ):
        raise ValueError(f"topk_w must be contiguous bf16 [{pc.ep * mt}, {TOP_K}]")
    sms = num_sms(out.device.index)
    ctas = min(_env_int(CTAS_ENV, 16, 1, _DONE_BLOCKS), sms) if ctas is None else ctas
    if not 1 <= ctas <= _DONE_BLOCKS:
        raise ValueError(f"ctas {ctas}; valid: 1..{_DONE_BLOCKS}")
    warps = _env_int(WARPS_ENV, 8, 1, 32) if warps is None else warps
    stages = _env_int(ROWS_ENV, 4, 1, TOP_K) if stages is None else stages
    chunk = _env_int(CHUNK_ENV, 8, 1, 1024) if chunk is None else chunk
    ws_base, ws_stride = pc._ws()
    _compile(
        pc.ep,
        pc.hidden,
        pc.lpc,
        pc.lay.key(),
        pc.rows,
        pc.inbox_bytes,
        warps,
        stages,
        chunk,
        HIDDEN // tile_n,
        fgs and check,
        not fgs,
        data_only,
        _comm._pdl() or fgs,
    )(
        pc.side[pc.rank, :_CACHELINE],
        yo.data_ptr(),
        y.data_ptr(),
        topk_w.data_ptr(),
        plan.permuted_idx_to_expanded_idx.data_ptr(),
        plan.expanded_idx_to_permuted_idx.data_ptr(),
        plan.tile_idx_to_mn_limit.data_ptr(),
        plan.num_non_exiting_tiles.data_ptr(),
        counters.data_ptr(),
        counters.data_ptr() + 4 * ntiles_cap,
        ws_base,
        ws_stride,
        pc.side.data_ptr(),
        pc.side.stride(0),
        pc.rank,
        mt,
        ctas,
        _TIMEOUT_NS,
    )
    pc._round = (n, mt, yo.data_ptr(), ctas, "poll" if fgs else "done", data_only, "consumer", True)
    return ctas
