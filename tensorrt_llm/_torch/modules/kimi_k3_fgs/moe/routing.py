# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing of the Kimi K3 decode MoE expert path: received top-k ids -> ``moe_chain.RoutingPlan``.

A cluster of up to 8 CTAs (one up to 256 received rows, else one per 128 rows); CTA c and, inside it, warp w own
contiguous blocks of rows. A warp walks its block 32 rows at a time, one row per lane, reading and writing rows through
shared memory so global accesses stay coalesced. Each slice compacts its local (row, k) pairs into a warp-private list
in ascending expanded index (row * 16 + k), and the list is walked 32 pairs at a time: ``match.any`` on the local
expert groups equal experts, so the rank of a pair among the earlier pairs of its expert in the warp is a running
per-(expert, warp) count plus ``popc(peers & lanemask_lt)``. Pass 1 counts; an exclusive scan over warps, over CTAs
(the CTAs exchange their per-expert counts through distributed shared memory) and over the tile-padded expert counts
turns the counts into each (expert, warp)'s first permuted row; pass 2 places the pairs. The permuted order inside an
expert is ascending expanded index; no atomics, so the plan is identical on every call.

``tile_idx_to_mn_limit`` is ``min((t + 1) * tile_n, offsets[e] + counts[e])`` as in ``trtllm::moe_sort``.

Write order: the index maps and padding slots first, then the tile maps, then ``total_num_padded_tokens`` and
``num_non_exiting_tiles``. With ``publish`` the phases are separated by cluster barriers and fenced at GPU scope, so a
consumer that sees a tile-map entry or the tile count differ from a sentinel the reset stream armed (e.g. -1) also
sees the plan entries that phase depends on.
"""

from __future__ import annotations

import functools

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
_MAX_WARPS = 32
_MAX_LOCAL = 256
# A list entry packs (local expert << 20) | expanded index.
_IDX_BITS = 20
_MAX_ROWS = (1 << _IDX_BITS) // TOP_K
# Row pitch of the per-warp row stage: one padding word keeps a lane-per-row access free of bank conflicts.
_PITCH = TOP_K + 1

if _CUTE_AVAILABLE:

    @dsl_user_op
    def _match_any(val, *, loc=None, ip=None) -> Int32:
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [Int32(val).ir_value(loc=loc, ip=ip)],
                "match.any.sync.b32 $0, $1, 0xffffffff;",
                "=r,r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

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
    def _ld_peer_b32(smem_addr, peer, *, loc=None, ip=None) -> Int32:
        """Load an int32 from the same shared-memory offset of CTA ``peer`` of the cluster."""
        return Int32(
            _asm(
                T.i32(),
                [Int32(smem_addr).ir_value(loc=loc, ip=ip), Int32(peer).ir_value(loc=loc, ip=ip)],
                "{\n.reg .b32 a;\nmapa.shared::cluster.u32 a, $1, $2;\nld.shared::cluster.b32 $0, [a];\n}",
                "=r,r,r",
                loc,
                ip,
            )
        )

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
        x = n
        for i in cutlass.range_constexpr(5):
            t = cute.arch.shuffle_sync_up(x, offset=1 << i)
            if lane >= (1 << i):
                x = x + t
        total = cute.arch.shuffle_sync(x, 31)
        j = x - n
        row = row0 + lane
        for k in cutlass.range_constexpr(TOP_K):
            v = raw[k] - lo
            local = Int32(0)
            if v >= 0:
                if v < num_local:
                    local = Int32(1)
            if local != 0:
                items[j] = (v << _IDX_BITS) | (row * TOP_K + k)
                j = j + 1
            else:
                stage[lane * _PITCH + k] = Int32(-1)
        cute.arch.sync_warp()
        return total

    class _RoutingKernel:
        def __init__(
            self,
            num_local: int,
            tile_n: int,
            ctas: int,
            warps: int,
            enable_pdl: bool,
            publish: bool,
        ):
            assert 1 <= num_local <= _MAX_LOCAL, num_local
            self.E = num_local
            # Expert slots: a multiple of 32 (each lane of a warp owns EM / 32 of them), at least 64.
            self.EM = max(64, (num_local + 31) // 32 * 32)
            self.tile_n = tile_n
            self.C = ctas
            self.W = warps
            assert warps * 32 >= self.EM
            self.enable_pdl = enable_pdl
            self.publish = publish

        def smem_bytes(self) -> int:
            return (self.EM * self.W + 4 * self.EM + 16 + 32 * (TOP_K + _PITCH) * self.W) * 4 + 256

        @cute.jit
        def __call__(
            self,
            mIds: cute.Tensor,
            mCounts: cute.Tensor,
            mOffsets: cute.Tensor,
            mP2E: cute.Tensor,
            mE2P: cute.Tensor,
            mT2E: cute.Tensor,
            mT2L: cute.Tensor,
            mNumTiles: cute.Tensor,
            mTotal: cute.Tensor,
            rows: Int32,
            lo: Int32,
            stream,
        ):
            self.kernel(
                mIds, mCounts, mOffsets, mP2E, mE2P, mT2E, mT2L, mNumTiles, mTotal, rows, lo
            ).launch(
                grid=[self.C, 1, 1],
                block=[self.W * 32, 1, 1],
                cluster=[self.C, 1, 1] if cutlass.const_expr(self.C > 1) else None,
                smem=self.smem_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mIds: cute.Tensor,
            mCounts: cute.Tensor,
            mOffsets: cute.Tensor,
            mP2E: cute.Tensor,
            mE2P: cute.Tensor,
            mT2E: cute.Tensor,
            mT2L: cute.Tensor,
            mNumTiles: cute.Tensor,
            mTotal: cute.Tensor,
            rows: Int32,
            lo: Int32,
        ):
            E = self.E
            EM = self.EM
            TN = self.tile_n
            C = self.C
            W = self.W
            NT = W * 32
            tidx, _, _ = cute.arch.thread_idx()
            cta, _, _ = cute.arch.block_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            lane = cute.arch.lane_idx()

            smem = SmemAllocator()
            # sBase[e, w]: pairs of expert e in warp w's rows, then the first permuted row of those pairs.
            sBase = smem.allocate_tensor(Int32, cute.make_layout((EM, W)), byte_alignment=16)
            # sCount: this CTA's pairs per expert; sPre / sTot: pairs of the lower CTAs / of all CTAs.
            sCount = smem.allocate_tensor(Int32, cute.make_layout((EM,)), byte_alignment=16)
            sPre = smem.allocate_tensor(Int32, cute.make_layout((EM,)), byte_alignment=16)
            sTot = smem.allocate_tensor(Int32, cute.make_layout((EM,)), byte_alignment=16)
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
                cute.arch.griddepcontrol_launch_dependents()

            rows_per_cta = (rows + C * 32 - 1) // (C * 32) * 32
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
                            e = items[c + lane] >> _IDX_BITS
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
                    sBase[tidx, w] = run
                    run = run + c
                sCount[tidx] = run
            if cutlass.const_expr(C > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                if tidx < EM:
                    pre = Int32(0)
                    tot = Int32(0)
                    for c in cutlass.range_constexpr(C):
                        v = _ld_peer_b32(Int32(sCount.iterator.toint()) + tidx * 4, Int32(c))
                        if c < cta:
                            pre = pre + v
                        tot = tot + v
                    sPre[tidx] = pre
                    sTot[tidx] = tot
                # No CTA may leave while a peer still reads its sCount.
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            else:
                if tidx < EM:
                    sPre[tidx] = Int32(0)
                    sTot[tidx] = sCount[tidx]
            cute.arch.sync_threads()

            if warp == 0:
                per = EM // 32
                own = Int32(0)
                for i in cutlass.range_constexpr(per):
                    own = own + (sTot[lane * per + i] + (TN - 1)) // TN * TN
                x = own
                for i in cutlass.range_constexpr(5):
                    t = cute.arch.shuffle_sync_up(x, offset=1 << i)
                    if lane >= (1 << i):
                        x = x + t
                run = x - own
                for i in cutlass.range_constexpr(per):
                    sOff[lane * per + i] = run
                    run = run + (sTot[lane * per + i] + (TN - 1)) // TN * TN
                if lane == 31:
                    sOff[EM] = x
            cute.arch.sync_threads()

            for i in cutlass.range_constexpr(EM // 32):
                e = lane + 32 * i
                sBase[e, warp] = sBase[e, warp] + sOff[e] + sPre[e]
            if cta == 0:
                if tidx < E:
                    mCounts[tidx] = sTot[tidx]
                if tidx < E:
                    mOffsets[tidx] = sOff[tidx]
                if tidx == 0:
                    mOffsets[E] = sOff[E]
            for i in cutlass.range(cta * NT + tidx, E * TN, C * NT, unroll=1):
                e = i // TN
                slot = sOff[e] + sTot[e] + i % TN
                if slot < sOff[e + 1]:
                    mP2E[slot] = Int32(-1)
            cute.arch.sync_warp()

            for s in cutlass.range(0, rows_per_warp, 32, unroll=1):
                if row0 + s < cta_end:
                    total = _compact_rows(ids, stage, items, row0 + s, cta_end, lo, lane, E)
                    for c in cutlass.range(0, total, 32, unroll=1):
                        e = Int32(E)
                        v = Int32(0)
                        if c + lane < total:
                            v = items[c + lane]
                            e = v >> _IDX_BITS
                        peers = _match_any(e)
                        pos = Int32(0)
                        if e < E:
                            pos = sBase[e, warp] + cute.arch.popc(peers & lanemask_lt)
                        cute.arch.sync_warp()
                        if e < E:
                            if (peers & lanemask_lt) == 0:
                                sBase[e, warp] = sBase[e, warp] + cute.arch.popc(peers)
                            expanded = v & ((1 << _IDX_BITS) - 1)
                            mP2E[pos] = expanded
                            local_idx = expanded - (row0 + s) * TOP_K
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
            cute.arch.sync_threads()
            if cutlass.const_expr(self.publish and C > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

            total = sOff[E]
            num_tiles = total // TN
            if cutlass.const_expr(self.publish):
                cute.arch.fence_acq_rel_gpu()
            for t in cutlass.range(cta * NT + tidx, num_tiles, C * NT, unroll=1):
                start = t * TN
                # Largest e with offsets[e] <= start; experts before it that are empty share its offset.
                e = Int32(0)
                for i in cutlass.range_constexpr(8):
                    step = 128 >> i
                    if e + step < E:
                        if sOff[e + step] <= start:
                            e = e + step
                mT2E[t] = e
                mT2L[t] = cutlass.min(start + TN, sOff[e] + sTot[e])
            cute.arch.sync_threads()
            if cutlass.const_expr(self.publish and C > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            if cta == 0:
                if tidx == 0:
                    if cutlass.const_expr(self.publish):
                        cute.arch.fence_acq_rel_gpu()
                    mTotal[0] = total
                    mNumTiles[0] = num_tiles


def _shape(rows: int, num_local: int) -> tuple:
    """(cluster CTAs, warps per CTA) for ``rows`` received rows: one CTA up to 256 rows, else about 128 rows per CTA
    (a power of two up to 8); a warp per 32 rows of a CTA, a power of two, at least one lane per local expert."""
    ctas = 1
    if rows > 256:
        while ctas < 8 and ctas * 128 < rows:
            ctas *= 2
    slices = (rows + ctas * 32 - 1) // (ctas * 32)
    warps = 1
    while warps < slices and warps < _MAX_WARPS:
        warps *= 2
    em = max(64, (num_local + 31) // 32 * 32)
    return ctas, max(warps, em // 32)


@functools.lru_cache(maxsize=None)
def _compiled(num_local: int, tile_n: int, ctas: int, warps: int, enable_pdl: bool, publish: bool):
    kernel = _RoutingKernel(num_local, tile_n, ctas, warps, enable_pdl, publish)
    ids = cute.runtime.make_fake_compact_tensor(
        Int32, (cute.sym_int(), TOP_K), stride_order=(1, 0), assumed_align=16
    )

    def vec():
        return cute.runtime.make_fake_compact_tensor(Int32, (cute.sym_int(),), assumed_align=4)

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        ids,
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        Int32(1),
        Int32(0),
        stream,
        options="--enable-tvm-ffi",
    )


def run(
    topk_ids: torch.Tensor,
    local_expert_offset: int,
    num_local: int,
    plan,
    *,
    enable_pdl: bool = False,
    publish: bool = False,
) -> None:
    """Fill ``plan`` (a ``moe_chain.RoutingPlan``) from ``topk_ids`` int32 [rows, 16] global expert ids.

    Pairs (row, k) with ``local_expert_offset <= topk_ids[row, k] < local_expert_offset + num_local`` are placed by
    local expert, in ascending ``row * 16 + k`` within an expert, each expert padded to ``plan.tile_n`` rows.
    Written: ``expanded_idx_to_permuted_idx[:rows * 16]``, ``permuted_idx_to_expanded_idx[:total]`` (-1 on padding
    slots), the tile maps for tiles below ``num_non_exiting_tiles`` (mn limit clipped to the tile, as moe_sort),
    both counters, ``expert_counts`` and ``expert_offsets``; later entries keep their old values. One launch, no host
    sync, CUDA-graph safe.

    ``enable_pdl``: launch with PDL, wait for the producer of ``topk_ids`` at entry and trigger dependents at entry
    (a dependent that reads the plan must still wait for this grid, or poll a published sentinel). ``publish``:
    fence each write phase at GPU scope (see the module docstring).
    """
    rows = topk_ids.shape[0]
    if (
        topk_ids.dtype != torch.int32
        or tuple(topk_ids.shape[1:]) != (TOP_K,)
        or not topk_ids.is_contiguous()
    ):
        raise ValueError(
            f"routing: topk_ids must be contiguous int32 [rows, {TOP_K}], got {topk_ids.dtype} "
            f"{tuple(topk_ids.shape)}"
        )
    if rows > min(plan.max_rows, _MAX_ROWS) or not 1 <= num_local <= _MAX_LOCAL:
        raise ValueError(
            f"routing: {rows} rows / {num_local} local experts exceed the plan "
            f"({plan.max_rows} rows, at most {_MAX_LOCAL} experts)"
        )
    if plan.expert_counts.numel() != num_local:
        raise ValueError(
            f"routing: plan holds {plan.expert_counts.numel()} experts, not {num_local}"
        )
    fn = _compiled(num_local, plan.tile_n, *_shape(rows, num_local), enable_pdl, publish)
    fn(
        topk_ids,
        plan.expert_counts,
        plan.expert_offsets,
        plan.permuted_idx_to_expanded_idx,
        plan.expanded_idx_to_permuted_idx,
        plan.tile_idx_to_expert_idx,
        plan.tile_idx_to_mn_limit,
        plan.num_non_exiting_tiles,
        plan.total_num_padded_tokens,
        rows,
        local_expert_offset,
    )
