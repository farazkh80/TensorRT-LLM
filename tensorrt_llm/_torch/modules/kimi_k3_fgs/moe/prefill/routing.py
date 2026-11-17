# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing of the Kimi K3 prefill MoE expert path: received top-k ids -> ``moe_chain.RoutingPlan`` at any row count.

The decode kernel (``moe/routing.py``) is one cluster of up to 8 CTAs and packs a list entry as
``(expert << 20) | (row * 16 + k)``, so it stops at 65536 received rows. This kernel keeps its row slicing and its
deterministic ranking (per-lane compaction of the local (row, k) pairs of a 32-row slice into a warp list in ascending
expanded index, ``match.any`` on the local expert, rank = running per-(expert, warp) count + ``popc(peers &
lanemask_lt)``; no atomics on the plan) over a grid of up to ``max_ctas`` CTAs, in two PDL-chained launches:

1. ``count``: each warp counts its pairs per expert; the exclusive scan over the CTA's warps and the CTA's total per
   expert go to a workspace, and the total is added to its 32-CTA group's sum. The last CTA to finish (ticket
   counter) prefixes the group sums per expert, tile-pads and scans the experts, and writes ``expert_counts``,
   ``expert_offsets``, ``total_num_padded_tokens`` and ``num_non_exiting_tiles``.
2. ``place``: each warp's first permuted row per expert is ``expert_offsets[e]`` + its group's prefix + the counts
   of the earlier CTAs of the group + ``wbase[cta, w, e]``; it re-reads its rows and writes
   ``permuted_idx_to_expanded_idx`` (and optionally ``permuted_idx_to_token_idx``, the stock cubins' map),
   ``expanded_idx_to_permuted_idx``, the padding slots and the tile maps.

The permuted order inside an expert is ascending expanded index; the plan is identical on every call. The stock routing
(``routingIndicesCoopKernel`` / ``routingIndicesOffsetsKernel``) orders pairs inside an expert by atomics, so the two
agree on the pair set per expert, the tile maps and the counters, not on the order.
"""

from __future__ import annotations

import functools
from typing import Dict, Optional, Tuple

import torch

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils import SmemAllocator

    _CUTE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CUTE_AVAILABLE = False

TOP_K = 16
_MAX_LOCAL = 256
# A warp-list entry packs (local expert << 9) | (lane * 16 + k): the pair's place in its 32-row slice.
_LOC_BITS = 9
# Row pitch of the per-warp row stage: one padding word keeps a lane-per-row access free of bank conflicts.
_PITCH = TOP_K + 1
_TRIGGERS = ("entry", "end")
# tile_idx_to_mn_limit: clipped to the tile (moe_chain.RoutingPlan, trtllm::moe_sort) or the expert's last row.
_MN_LIMITS = ("tile", "expert_end")

if _CUTE_AVAILABLE:

    def _asm(res, operands, text, constraints, loc, ip):
        return llvm.inline_asm(
            res,
            operands,
            text,
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _match_any(val, *, loc=None, ip=None) -> Int32:
        return Int32(
            _asm(
                T.i32(),
                [Int32(val).ir_value(loc=loc, ip=ip)],
                "match.any.sync.b32 $0, $1, 0xffffffff;",
                "=r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _ld_v4(addr, *, loc=None, ip=None):
        res = _asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            loc,
            ip,
        )
        return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]

    @dsl_user_op
    def _st_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)]
            + [Int32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _atom_add_acq_rel(addr, val, *, loc=None, ip=None) -> Int32:
        return Int32(
            _asm(
                T.i32(),
                [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
                "atom.acq_rel.gpu.global.add.s32 $0, [$1], $2;",
                "=r,l,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _red_add(addr, val, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
            "red.relaxed.gpu.global.add.s32 [$0], $1;",
            "l,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _trap(*, loc=None, ip=None) -> None:
        _asm(None, [], "trap;", "", loc, ip)

    @cute.jit
    def _warp_prefix(x: Int32, lane: Int32):
        """Inclusive scan of ``x`` over the warp's lanes."""
        for i in cutlass.range_constexpr(5):
            t = cute.arch.shuffle_sync_up(x, offset=1 << i)
            if lane >= (1 << i):
                x = x + t
        return x

    @cute.jit
    def _compact_rows(
        ids: cutlass.Int64,
        stage: cute.Tensor,
        items: cute.Tensor,
        row0: Int32,
        rows: Int32,
        lo: Int32,
        lane: Int32,
        num_local: cutlass.Constexpr[int],
    ) -> Int32:
        """Rows [row0, row0 + 32), one per lane: append each lane's local pairs to the warp list ``items`` after those
        of lower lanes, in k order, and leave -1 in ``stage[lane * _PITCH + k]`` for the non-local k (rows past ``rows``
        are empty); returns the list length. The rows are read with coalesced 16-byte loads through ``stage``."""
        for q in cutlass.range_constexpr(TOP_K // 4):
            idx = (q * 32 + lane) * 4
            at = idx // TOP_K * _PITCH + idx % TOP_K
            for i in cutlass.range_constexpr(4):
                stage[at + i] = Int32(-1)
            if row0 + idx // TOP_K < rows:
                w = _ld_v4(ids + (cutlass.Int64(row0) * TOP_K + idx) * 4)
                for i in cutlass.range_constexpr(4):
                    stage[at + i] = w[i]
        cute.arch.sync_warp()
        raw = cute.make_rmem_tensor(cute.make_layout((TOP_K,)), Int32)
        for k in cutlass.range_constexpr(TOP_K):
            raw[k] = stage[lane * _PITCH + k]
        cute.arch.sync_warp()
        n = Int32(0)
        for k in cutlass.range_constexpr(TOP_K):
            v = raw[k] - lo
            if v >= 0:
                if v < num_local:
                    n = n + 1
        x = _warp_prefix(n, lane)
        total = cute.arch.shuffle_sync(x, 31)
        j = x - n
        for k in cutlass.range_constexpr(TOP_K):
            v = raw[k] - lo
            local = Int32(0)
            if v >= 0:
                if v < num_local:
                    local = Int32(1)
            if local != 0:
                items[j] = (v << _LOC_BITS) | (lane * TOP_K + k)
                j = j + 1
            else:
                stage[lane * _PITCH + k] = Int32(-1)
        cute.arch.sync_warp()
        return total

    class _Config:
        """One compile-time configuration of the two launches."""

        def __init__(
            self,
            num_local: int,
            tile_n: int,
            warps: int,
            max_ctas: int,
            enable_pdl: bool,
            trigger: str,
            token_idx: bool,
            mn_limit: str,
        ):
            assert 1 <= num_local <= _MAX_LOCAL, num_local
            assert trigger in _TRIGGERS, trigger
            assert mn_limit in _MN_LIMITS, mn_limit
            self.E = num_local
            # Expert slots: a multiple of 32 (each lane of a warp owns EM / 32 of them), at least 64.
            self.EM = max(64, (num_local + 31) // 32 * 32)
            assert warps * 32 >= self.EM
            self.tile_n = tile_n
            self.W = warps
            self.B = max_ctas
            self.enable_pdl = enable_pdl
            self.trigger = trigger
            self.token_idx = token_idx
            self.clip_mn_limit = mn_limit == "tile"

        def smem_bytes(self) -> int:
            return (self.EM * self.W + 4 * self.EM + 32 + 32 * (TOP_K + _PITCH) * self.W) * 4 + 256

    class _CountKernel(_Config):
        """Launch 1: per-warp / per-CTA counts, then the last CTA prefixes the group sums and the experts."""

        @cute.jit
        def __call__(
            self,
            mIds: cute.Tensor,
            mCnt: cute.Tensor,
            mGrp: cute.Tensor,
            mGpref: cute.Tensor,
            mWBase: cute.Tensor,
            mTicket: cute.Tensor,
            mCounts: cute.Tensor,
            mOffsets: cute.Tensor,
            mTotal: cute.Tensor,
            mNumTiles: cute.Tensor,
            rows: Int32,
            lo: Int32,
            num_ctas: Int32,
            p_cap: Int32,
            stream,
        ):
            self.kernel(
                mIds,
                mCnt,
                mGrp,
                mGpref,
                mWBase,
                mTicket,
                mCounts,
                mOffsets,
                mTotal,
                mNumTiles,
                rows,
                lo,
                num_ctas,
                p_cap,
            ).launch(
                grid=[num_ctas, 1, 1],
                block=[self.W * 32, 1, 1],
                smem=self.smem_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mIds: cute.Tensor,
            mCnt: cute.Tensor,
            mGrp: cute.Tensor,
            mGpref: cute.Tensor,
            mWBase: cute.Tensor,
            mTicket: cute.Tensor,
            mCounts: cute.Tensor,
            mOffsets: cute.Tensor,
            mTotal: cute.Tensor,
            mNumTiles: cute.Tensor,
            rows: Int32,
            lo: Int32,
            num_ctas: Int32,
            p_cap: Int32,
        ):
            E = self.E
            EM = self.EM
            TN = self.tile_n
            W = self.W
            B = self.B
            CH = (B + 31) // 32
            NT = W * 32
            tidx, _, _ = cute.arch.thread_idx()
            cta, _, _ = cute.arch.block_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            lane = cute.arch.lane_idx()

            smem = SmemAllocator()
            # sBase[e, w]: pairs of expert e in warp w's rows, then the exclusive scan over warps.
            sBase = smem.allocate_tensor(Int32, cute.make_layout((EM, W)), byte_alignment=16)
            sTot = smem.allocate_tensor(Int32, cute.make_layout((EM,)), byte_alignment=16)
            sOff = smem.allocate_tensor(Int32, cute.make_layout((EM + 16,)), byte_alignment=16)
            sFlag = smem.allocate_tensor(Int32, cute.make_layout((16,)), byte_alignment=16)
            sItems = smem.allocate_tensor(
                Int32, cute.make_layout((32 * TOP_K, W)), byte_alignment=16
            )
            sStage = smem.allocate_tensor(
                Int32, cute.make_layout((32 * _PITCH, W)), byte_alignment=16
            )
            items = sItems[None, warp]
            stage = sStage[None, warp]
            ids = mIds.iterator.toint()

            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.trigger == "entry"):
                    cute.arch.griddepcontrol_launch_dependents()

            rows_per_cta = (rows + num_ctas * 32 - 1) // (num_ctas * 32) * 32
            cta_end = cutlass.min(rows, (cta + 1) * rows_per_cta)
            rows_per_warp = (rows_per_cta + NT - 1) // NT * 32
            row0 = cta * rows_per_cta + warp * rows_per_warp
            lanemask_lt = (Int32(1) << lane) - 1

            for i in cutlass.range_constexpr(EM // 32):
                sBase[lane + 32 * i, warp] = Int32(0)
            cute.arch.sync_warp()
            for s in cutlass.range(0, rows_per_warp, 32, unroll=1):
                if row0 + s < cta_end:
                    total = _compact_rows(ids, stage, items, row0 + s, cta_end, lo, lane, E)
                    for c in cutlass.range(0, total, 32, unroll=1):
                        e = Int32(E)
                        if c + lane < total:
                            e = items[c + lane] >> _LOC_BITS
                        peers = _match_any(e)
                        if e < E:
                            if (peers & lanemask_lt) == 0:
                                sBase[e, warp] = sBase[e, warp] + cute.arch.popc(peers)
                        cute.arch.sync_warp()
            cute.arch.sync_threads()

            if tidx < EM:
                run = Int32(0)
                for w in cutlass.range_constexpr(W):
                    c = sBase[tidx, w]
                    mWBase[(cta * W + w) * EM + tidx] = run
                    run = run + c
                mCnt[cta * EM + tidx] = run
                # Integer adds commute exactly: the group sum is the same whatever the arrival order.
                _red_add(mGrp.iterator.toint() + ((cta // 32) * EM + tidx) * 4, run)
            cute.arch.sync_threads()
            if tidx == 0:
                cute.arch.fence_acq_rel_gpu()
                old = _atom_add_acq_rel(mTicket.iterator.toint(), Int32(1))
                sFlag[0] = Int32(0)
                if old == num_ctas - 1:
                    sFlag[0] = Int32(1)
            cute.arch.sync_threads()

            if sFlag[0] == 1:
                cute.arch.fence_acq_rel_gpu()
                # Thread e: exclusive prefix of expert e's 32-CTA group sums; the group sums are re-zeroed for the
                # next launch (every CTA of this launch has added its count).
                if tidx < EM:
                    vals = cute.make_rmem_tensor(cute.make_layout((CH,)), Int32)
                    for g in cutlass.range_constexpr(CH):
                        v = Int32(0)
                        if g * 32 < num_ctas:
                            v = mGrp[g * EM + tidx]
                        vals[g] = v
                    run = Int32(0)
                    for g in cutlass.range_constexpr(CH):
                        if g * 32 < num_ctas:
                            mGpref[g * EM + tidx] = run
                            mGrp[g * EM + tidx] = Int32(0)
                        run = run + vals[g]
                    sTot[tidx] = run
                cute.arch.sync_threads()
                if warp == 0:
                    per = EM // 32
                    own = Int32(0)
                    for i in cutlass.range_constexpr(per):
                        own = own + (sTot[lane * per + i] + (TN - 1)) // TN * TN
                    x = _warp_prefix(own, lane)
                    run = x - own
                    for i in cutlass.range_constexpr(per):
                        sOff[lane * per + i] = run
                        run = run + (sTot[lane * per + i] + (TN - 1)) // TN * TN
                    if lane == 31:
                        sOff[EM] = x
                cute.arch.sync_threads()
                if tidx < E:
                    mCounts[tidx] = sTot[tidx]
                    mOffsets[tidx] = sOff[tidx]
                if tidx == 0:
                    total = sOff[EM]
                    if total > p_cap:
                        _trap()
                    mOffsets[E] = total
                    mTotal[0] = total
                    mNumTiles[0] = total // TN
                    mTicket[0] = Int32(0)
            if cutlass.const_expr(self.enable_pdl and self.trigger == "end"):
                cute.arch.griddepcontrol_launch_dependents()

    class _PlaceKernel(_Config):
        """Launch 2: padding slots, tile maps, and the index maps."""

        @cute.jit
        def __call__(
            self,
            mIds: cute.Tensor,
            mCnt: cute.Tensor,
            mGpref: cute.Tensor,
            mWBase: cute.Tensor,
            mCounts: cute.Tensor,
            mOffsets: cute.Tensor,
            mP2E: cute.Tensor,
            mP2T: cute.Tensor,
            mE2P: cute.Tensor,
            mT2E: cute.Tensor,
            mT2L: cute.Tensor,
            rows: Int32,
            lo: Int32,
            num_ctas: Int32,
            stream,
        ):
            self.kernel(
                mIds,
                mCnt,
                mGpref,
                mWBase,
                mCounts,
                mOffsets,
                mP2E,
                mP2T,
                mE2P,
                mT2E,
                mT2L,
                rows,
                lo,
                num_ctas,
            ).launch(
                grid=[num_ctas, 1, 1],
                block=[self.W * 32, 1, 1],
                smem=self.smem_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mIds: cute.Tensor,
            mCnt: cute.Tensor,
            mGpref: cute.Tensor,
            mWBase: cute.Tensor,
            mCounts: cute.Tensor,
            mOffsets: cute.Tensor,
            mP2E: cute.Tensor,
            mP2T: cute.Tensor,
            mE2P: cute.Tensor,
            mT2E: cute.Tensor,
            mT2L: cute.Tensor,
            rows: Int32,
            lo: Int32,
            num_ctas: Int32,
        ):
            E = self.E
            EM = self.EM
            TN = self.tile_n
            W = self.W
            NT = W * 32
            tidx, _, _ = cute.arch.thread_idx()
            cta, _, _ = cute.arch.block_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            lane = cute.arch.lane_idx()

            smem = SmemAllocator()
            sBase = smem.allocate_tensor(Int32, cute.make_layout((EM, W)), byte_alignment=16)
            sTot = smem.allocate_tensor(Int32, cute.make_layout((EM,)), byte_alignment=16)
            sPre = smem.allocate_tensor(Int32, cute.make_layout((EM,)), byte_alignment=16)
            sOff = smem.allocate_tensor(Int32, cute.make_layout((EM + 16,)), byte_alignment=16)
            sItems = smem.allocate_tensor(
                Int32, cute.make_layout((32 * TOP_K, W)), byte_alignment=16
            )
            sStage = smem.allocate_tensor(
                Int32, cute.make_layout((32 * _PITCH, W)), byte_alignment=16
            )
            items = sItems[None, warp]
            stage = sStage[None, warp]
            ids = mIds.iterator.toint()
            e2p = mE2P.iterator.toint()

            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.trigger == "entry"):
                    cute.arch.griddepcontrol_launch_dependents()

            rows_per_cta = (rows + num_ctas * 32 - 1) // (num_ctas * 32) * 32
            cta_end = cutlass.min(rows, (cta + 1) * rows_per_cta)
            rows_per_warp = (rows_per_cta + NT - 1) // NT * 32
            row0 = cta * rows_per_cta + warp * rows_per_warp
            lanemask_lt = (Int32(1) << lane) - 1

            if tidx <= EM:
                off = mOffsets[E]
                if tidx < E:
                    off = mOffsets[tidx]
                sOff[tidx] = off
            if tidx < EM:
                tot = Int32(0)
                if tidx < E:
                    tot = mCounts[tidx]
                sTot[tidx] = tot
                # Pairs of expert tidx in the CTAs before this one: the group's prefix plus the group's earlier CTAs.
                b0 = cta // 32 * 32
                pre = mGpref[(cta // 32) * EM + tidx]
                for j in cutlass.range_constexpr(32):
                    if b0 + j < cta:
                        pre = pre + mCnt[(b0 + j) * EM + tidx]
                sPre[tidx] = pre
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(EM // 32):
                e = lane + 32 * i
                sBase[e, warp] = sOff[e] + sPre[e] + mWBase[(cta * W + warp) * EM + e]
            cute.arch.sync_warp()

            for i in cutlass.range(cta * NT + tidx, E * TN, num_ctas * NT, unroll=1):
                e = i // TN
                slot = sOff[e] + sTot[e] + i % TN
                if slot < sOff[e + 1]:
                    mP2E[slot] = Int32(-1)
                    if cutlass.const_expr(self.token_idx):
                        mP2T[slot] = Int32(-1)
            num_tiles = sOff[EM] // TN
            for t in cutlass.range(cta * NT + tidx, num_tiles, num_ctas * NT, unroll=1):
                start = t * TN
                # Largest e with offsets[e] <= start; experts before it that are empty share its offset.
                e = Int32(0)
                for i in cutlass.range_constexpr(8):
                    step = 128 >> i
                    if e + step < E:
                        if sOff[e + step] <= start:
                            e = e + step
                mT2E[t] = e
                if cutlass.const_expr(self.clip_mn_limit):
                    mT2L[t] = cutlass.min(start + TN, sOff[e] + sTot[e])
                else:
                    mT2L[t] = sOff[e] + sTot[e]

            for s in cutlass.range(0, rows_per_warp, 32, unroll=1):
                if row0 + s < cta_end:
                    total = _compact_rows(ids, stage, items, row0 + s, cta_end, lo, lane, E)
                    for c in cutlass.range(0, total, 32, unroll=1):
                        e = Int32(E)
                        v = Int32(0)
                        if c + lane < total:
                            v = items[c + lane]
                            e = v >> _LOC_BITS
                        peers = _match_any(e)
                        pos = Int32(0)
                        if e < E:
                            pos = sBase[e, warp] + cute.arch.popc(peers & lanemask_lt)
                        cute.arch.sync_warp()
                        if e < E:
                            if (peers & lanemask_lt) == 0:
                                sBase[e, warp] = sBase[e, warp] + cute.arch.popc(peers)
                            local_idx = v & ((1 << _LOC_BITS) - 1)
                            row = row0 + s + local_idx // TOP_K
                            mP2E[pos] = row * TOP_K + local_idx % TOP_K
                            if cutlass.const_expr(self.token_idx):
                                mP2T[pos] = row
                            stage[local_idx // TOP_K * _PITCH + local_idx % TOP_K] = pos
                        cute.arch.sync_warp()
                    for q in cutlass.range_constexpr(TOP_K // 4):
                        idx = (q * 32 + lane) * 4
                        at = idx // TOP_K * _PITCH + idx % TOP_K
                        if row0 + s + idx // TOP_K < cta_end:
                            _st_v4(
                                e2p + (cutlass.Int64(row0 + s) * TOP_K + idx) * 4,
                                stage[at],
                                stage[at + 1],
                                stage[at + 2],
                                stage[at + 3],
                            )
                    cute.arch.sync_warp()
            if cutlass.const_expr(self.enable_pdl and self.trigger == "end"):
                cute.arch.griddepcontrol_launch_dependents()


class Workspace:
    """Per-device scratch of the two launches: ``cnt[B, E_pad]`` (pairs per CTA and expert), ``grp[B / 32, E_pad]``
    (sums over 32-CTA groups) and ``gpref`` (their exclusive prefix), ``wbase[B, W, E_pad]`` (prefix over the CTA's
    warps) and the CTA ticket. ``grp`` and the ticket are zero between launches."""

    def __init__(self, device: torch.device, em: int, warps: int, max_ctas: int):
        kw = dict(dtype=torch.int32, device=device)
        groups = (max_ctas + 31) // 32
        self.em, self.warps, self.max_ctas = em, warps, max_ctas
        self.cnt = torch.zeros(em * max_ctas, **kw)
        self.grp = torch.zeros(em * groups, **kw)
        self.gpref = torch.zeros(em * groups, **kw)
        self.wbase = torch.zeros(max_ctas * warps * em, **kw)
        self.ticket = torch.zeros(1, **kw)


_workspaces: Dict[tuple, Workspace] = {}


def _em(num_local: int) -> int:
    return max(64, (num_local + 31) // 32 * 32)


def workspace(device: torch.device, num_local: int, warps: int, max_ctas: int) -> Workspace:
    key = (device, _em(num_local), warps, max_ctas)
    ws = _workspaces.get(key)
    if ws is None:
        ws = _workspaces[key] = Workspace(device, *key[1:])
    return ws


@functools.lru_cache(maxsize=None)
def default_max_ctas(device_index: int, warps: int) -> int:
    """One resident wave: the CTAs per SM the thread count allows (shared memory allows at least as many)."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count * max(
        1, 64 // warps
    )


def num_ctas_for(rows: int, warps: int, max_ctas: int) -> int:
    """One 32-row slice per warp until ``max_ctas`` CTAs, then more slices per warp."""
    return max(1, min(max_ctas, (rows + warps * 32 - 1) // (warps * 32)))


@functools.lru_cache(maxsize=None)
def _compiled(
    num_local: int,
    tile_n: int,
    warps: int,
    max_ctas: int,
    enable_pdl: bool,
    trigger: str,
    token_idx: bool,
    mn_limit: str,
):
    cfg = (num_local, tile_n, warps, max_ctas, enable_pdl, trigger, token_idx, mn_limit)
    ids = cute.runtime.make_fake_compact_tensor(
        Int32, (cute.sym_int(), TOP_K), stride_order=(1, 0), assumed_align=16
    )

    def vec():
        return cute.runtime.make_fake_compact_tensor(Int32, (cute.sym_int(),), assumed_align=4)

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    count = cute.compile(
        _CountKernel(*cfg),
        ids,
        *[vec() for _ in range(9)],
        Int32(1),
        Int32(0),
        Int32(1),
        Int32(1),
        stream,
        options="--enable-tvm-ffi",
    )
    place = cute.compile(
        _PlaceKernel(*cfg),
        ids,
        *[vec() for _ in range(10)],
        Int32(1),
        Int32(0),
        Int32(1),
        stream,
        options="--enable-tvm-ffi",
    )
    return count, place


def run(
    topk_ids: torch.Tensor,
    local_expert_offset: int,
    num_local: int,
    plan,
    *,
    token_idx: Optional[torch.Tensor] = None,
    warps: int = 16,
    max_ctas: Optional[int] = None,
    enable_pdl: bool = False,
    trigger: str = "entry",
    mn_limit: str = "tile",
    phases: Tuple[str, ...] = ("count", "place"),
) -> None:
    """Fill ``plan`` (a ``moe_chain.RoutingPlan``) from ``topk_ids`` int32 [rows, 16] global expert ids.

    Pairs (row, k) with ``local_expert_offset <= topk_ids[row, k] < local_expert_offset + num_local`` are placed by
    local expert, in ascending ``row * 16 + k`` within an expert, each expert padded to ``plan.tile_n`` rows. Written:
    ``expanded_idx_to_permuted_idx[:rows * 16]`` (-1 for non-local pairs), ``permuted_idx_to_expanded_idx[:total]``
    (-1 on padding slots), the tile maps for tiles below ``num_non_exiting_tiles`` (mn limit clipped to the tile, as
    ``trtllm::moe_sort``), ``num_non_exiting_tiles``, ``total_num_padded_tokens``, ``expert_counts`` and
    ``expert_offsets``; later entries keep their old values. ``token_idx`` (int32, >= P entries) also receives
    ``permuted_idx_to_token_idx`` = expanded index // 16 (-1 on padding), the map the stock trtllm-gen FC cubins read.
    Two launches, no host sync, CUDA-graph safe; the process traps if the padded total exceeds the plan's
    ``permuted_idx_to_expanded_idx`` (size the plan for the worst case, ``rows * min(16, num_local)`` pairs).

    ``warps`` per CTA (16: 72 KB of shared memory, 4 CTAs per SM) and ``max_ctas`` (default: one resident wave,
    ``default_max_ctas``) size the grid (``num_ctas_for``). ``enable_pdl``: both launches wait for their producer at
    entry; ``trigger`` places ``launch_dependents`` at entry (a dependent that reads the plan must wait for the grid)
    or after the last stores. ``mn_limit="expert_end"`` writes ``expert_offsets[e] + expert_counts[e]`` unclipped
    (``moe_stages_torch.routing``; a consumer that tests ``row < mn_limit`` sees no difference). ``phases`` is a
    timing diagnostic: ``("count",)`` or ``("place",)`` runs one launch only
    (``place`` alone reuses the workspace of the previous full run).
    """
    rows = topk_ids.shape[0]
    if (
        topk_ids.dtype != torch.int32
        or tuple(topk_ids.shape[1:]) != (TOP_K,)
        or not topk_ids.is_contiguous()
    ):
        raise ValueError(
            f"routing: topk_ids must be contiguous int32 [rows, {TOP_K}], got {topk_ids.dtype} {tuple(topk_ids.shape)}"
        )
    if rows > plan.max_rows or rows * TOP_K > plan.expanded_idx_to_permuted_idx.numel():
        raise ValueError(f"routing: {rows} rows exceed the plan ({plan.max_rows} rows)")
    if not 1 <= num_local <= _MAX_LOCAL or plan.expert_counts.numel() != num_local:
        raise ValueError(
            f"routing: plan holds {plan.expert_counts.numel()} experts, asked {num_local} (at most {_MAX_LOCAL})"
        )
    if not 4 <= warps <= 32 or warps * 32 < _em(num_local):
        raise ValueError(
            f"routing: {warps} warps cannot own {num_local} experts (4..32, a lane per expert slot)"
        )
    p_cap = plan.permuted_idx_to_expanded_idx.numel()
    if plan.tile_idx_to_expert_idx.numel() * plan.tile_n < p_cap:
        p_cap = plan.tile_idx_to_expert_idx.numel() * plan.tile_n
    if token_idx is not None:
        if (
            token_idx.dtype != torch.int32
            or token_idx.numel() < plan.permuted_idx_to_expanded_idx.numel()
        ):
            raise ValueError(
                "routing: token_idx must be int32 with at least as many entries as the permuted map"
            )
    if max_ctas is None:
        max_ctas = default_max_ctas(topk_ids.device.index, warps)
    ws = workspace(topk_ids.device, num_local, warps, max_ctas)
    num_ctas = num_ctas_for(rows, warps, max_ctas)
    count, place = _compiled(
        num_local,
        plan.tile_n,
        warps,
        max_ctas,
        enable_pdl,
        trigger,
        token_idx is not None,
        mn_limit,
    )
    if "count" in phases:
        count(
            topk_ids,
            ws.cnt,
            ws.grp,
            ws.gpref,
            ws.wbase,
            ws.ticket,
            plan.expert_counts,
            plan.expert_offsets,
            plan.total_num_padded_tokens,
            plan.num_non_exiting_tiles,
            rows,
            local_expert_offset,
            num_ctas,
            p_cap,
        )
    if "place" in phases:
        place(
            topk_ids,
            ws.cnt,
            ws.gpref,
            ws.wbase,
            plan.expert_counts,
            plan.expert_offsets,
            plan.permuted_idx_to_expanded_idx,
            plan.permuted_idx_to_expanded_idx if token_idx is None else token_idx,
            plan.expanded_idx_to_permuted_idx,
            plan.tile_idx_to_expert_idx,
            plan.tile_idx_to_mn_limit,
            rows,
            local_expert_offset,
            num_ctas,
        )
