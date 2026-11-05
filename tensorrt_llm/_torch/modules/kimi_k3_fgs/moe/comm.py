# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE all-to-all dispatch / combine over CFT counted writes as CuTe DSL kernels.

``CuteA2A`` runs on the workspace of a stock ``NVLinkOneSided`` communicator that uses CFT counted writes and speaks
its protocol: the same workspace layout, round flag, receive-counter parity banks, per-peer and per-slot hardware byte
counters with their baselines, completion flags and per-(token, k) routing records. Its dispatch returns the views
``NVLinkOneSided.dispatch`` returns and its combine consumes the payload ``NVLinkOneSided.combine`` consumes (also the
in-workspace combine payload), so either phase can run stock or CuTe in any round.

Kernels (one stream, optional PDL as stock, ``TRTLLM_ENABLE_PDL``):

dispatch   prepare (round flag, send counters, next receive bank) and dispatch: one CTA per local token stages the
           token's payloads in shared memory with a bulk copy, routes it (one send per distinct target rank), sends it
           to itself with a bulk store and to peers with ``fabric.try_put`` counted writes into the peer's logical
           endpoint, which add the byte count to the per-source counter of the receiver. CTA 0 publishes the
           send counts, waits for every peer's count, resets the expert ids of unused receive slots to -1, and waits
           until each peer's byte counter reaches its baseline + count * bytes per token.
combine    prepare (round flag; own tokens into the receive inbox), push (per received token a counted write of the
           expert output back into its source rank's inbox, which adds to that slot's counter) and reduce (one CTA per
           local token waits for the counters of its own slots, then sums the k contributions in fp32 in stock order
           and rounds once to the payload dtype).

Differences from stock: the receive slot of a token is its order among this rank's tokens sent to that target
(counted by every CTA from the routing of the earlier tokens) instead of an atomic ticket, so dispatch output order
is deterministic (token order per source); the elected CTA is CTA 0. Received rows, counters, flags
and baselines are otherwise identical to stock, and the combine sums in stock order, so outputs match bitwise.
"""

from __future__ import annotations

import ctypes
import functools
import os
from typing import List, NamedTuple, Optional, Tuple

import torch

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils import SmemAllocator

    _CUTE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CUTE_AVAILABLE = False

ENV = "TLLM_K3_COMM"
_COUNTER_STRIDE = 256
_MAX_PAYLOADS = 4
_CACHELINE = 128
_THREADS = 256
_PUSH_WARPS = 4
_MBAR_BYTES = 64
_TIMEOUT_NS = int(os.environ.get("TLLM_K3_COMM_TIMEOUT_S", "300")) * 1_000_000_000


def enabled() -> bool:
    return os.environ.get(ENV, "") == "cute"


def _pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "0") == "1"


def _driver_fn(name: str, *argtypes):
    lib = ctypes.CDLL("libcuda.so.1")
    get = lib.cuGetProcAddress_v2
    get.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int),
    ]
    ptr, status = ctypes.c_void_p(), ctypes.c_int()
    rc = get(name.encode(), ctypes.byref(ptr), 13030, 0, ctypes.byref(status))
    if rc != 0 or not ptr.value:
        raise RuntimeError(f"{name} is not exported by the driver (rc {rc})")
    return ctypes.CFUNCTYPE(ctypes.c_int, *argtypes)(ptr.value)


def find_le_base(ep_size: int, span: int = 8192) -> int:
    """First id of the block of ``ep_size`` logical endpoints the stock CFT workspace reserved in this process.

    The stock endpoint manager is not reachable from Python; its ids are ``base + rank`` for every rank. The block is
    the only run of exactly ``ep_size`` constructed endpoint ids near the driver's next free id.
    ``TLLM_K3_COMM_LE_BASE`` overrides the search.
    """
    override = os.environ.get("TLLM_K3_COMM_LE_BASE")
    if override:
        return int(override)
    u32 = ctypes.c_uint32
    reserve = _driver_fn("cuLogicalEndpointIdReserve", ctypes.POINTER(u32), u32)
    release = _driver_fn("cuLogicalEndpointIdRelease", u32, u32)
    query = _driver_fn("cuLogicalEndpointQuery", u32, u32, ctypes.POINTER(ctypes.c_int))

    def ready(first: int, count: int) -> bool:
        if first < 0:
            return False
        st = ctypes.c_int(0)
        return query(first, count, ctypes.byref(st)) == 0 and st.value != 0

    probe = u32()
    if reserve(ctypes.byref(probe), 1) != 0:
        raise RuntimeError("cuLogicalEndpointIdReserve failed")
    release(probe.value, 1)
    nxt = probe.value
    found = [
        b
        for b in range(max(0, nxt - span), nxt + span)
        if ready(b, 1) and not ready(b - 1, 1) and ready(b, ep_size) and not ready(b + ep_size, 1)
    ]
    if len(found) != 1:
        raise RuntimeError(
            f"{ENV}: expected one block of {ep_size} logical endpoints near id {nxt}, found {found}; "
            "set TLLM_K3_COMM_LE_BASE"
        )
    return found[0]


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

    def _v(x, ty, loc, ip):
        return ty(x).ir_value(loc=loc, ip=ip)

    @dsl_user_op
    def _ld_b32(addr, *, loc=None, ip=None) -> Int32:
        return Int32(
            _asm(
                T.i32(),
                [_v(addr, Int64, loc, ip)],
                "ld.volatile.global.b32 $0, [$1];",
                "=r,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _ld_b64(addr, *, loc=None, ip=None) -> Int64:
        return Int64(
            _asm(
                T.i64(),
                [_v(addr, Int64, loc, ip)],
                "ld.volatile.global.b64 $0, [$1];",
                "=l,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _ld_sys_b32(addr, *, loc=None, ip=None) -> Int32:
        return Int32(
            _asm(
                T.i32(),
                [_v(addr, Int64, loc, ip)],
                "ld.relaxed.sys.global.b32 $0, [$1];",
                "=r,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _ld_sys_b64(addr, *, loc=None, ip=None) -> Int64:
        return Int64(
            _asm(
                T.i64(),
                [_v(addr, Int64, loc, ip)],
                "ld.relaxed.sys.global.b64 $0, [$1];",
                "=l,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _st_b32(addr, val, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [_v(addr, Int64, loc, ip), _v(val, Int32, loc, ip)],
            "st.global.b32 [$0], $1;",
            "l,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _st_b64(addr, val, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [_v(addr, Int64, loc, ip), _v(val, Int64, loc, ip)],
            "st.global.b64 [$0], $1;",
            "l,l",
            loc,
            ip,
        )

    @dsl_user_op
    def _st_sys_b32(addr, val, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [_v(addr, Int64, loc, ip), _v(val, Int32, loc, ip)],
            "st.relaxed.sys.global.b32 [$0], $1;",
            "l,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _ld_v4(addr, *, loc=None, ip=None):
        res = _asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [_v(addr, Int64, loc, ip)],
            "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            loc,
            ip,
        )
        return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]

    @dsl_user_op
    def _st_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [_v(addr, Int64, loc, ip)] + [_v(w, Int32, loc, ip) for w in (w0, w1, w2, w3)],
            "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _globaltimer(*, loc=None, ip=None) -> Int64:
        return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip))

    @dsl_user_op
    def _trap(*, loc=None, ip=None) -> None:
        _asm(None, [], "trap;", "", loc, ip)

    @dsl_user_op
    def _match_any(val, mask: int, *, loc=None, ip=None) -> Int32:
        return Int32(
            _asm(
                T.i32(),
                [_v(val, Int32, loc, ip)],
                f"match.any.sync.b32 $0, $1, {mask};",
                "=r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _mbar_init(mbar, count: int, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [_v(mbar, Int32, loc, ip)],
            f"mbarrier.init.shared::cta.b64 [$0], {count};\nfence.proxy.async.shared::cta;",
            "r",
            loc,
            ip,
        )

    @dsl_user_op
    def _mbar_expect_tx(mbar, nbytes, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [_v(mbar, Int32, loc, ip), _v(nbytes, Int32, loc, ip)],
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
            "r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _mbar_wait(mbar, parity, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [_v(mbar, Int32, loc, ip), _v(parity, Int32, loc, ip)],
            "{\n.reg .pred p;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n@!p bra W;\n}",
            "r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _bulk_g2s(dst, src, nbytes, mbar, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [
                _v(dst, Int32, loc, ip),
                _v(src, Int64, loc, ip),
                _v(nbytes, Int32, loc, ip),
                _v(mbar, Int32, loc, ip),
            ],
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
            "r,l,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _bulk_s2g(dst, src, nbytes, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [_v(dst, Int64, loc, ip), _v(src, Int32, loc, ip), _v(nbytes, Int32, loc, ip)],
            "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
            "l,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _bulk_commit_wait(*, loc=None, ip=None) -> None:
        _asm(None, [], "cp.async.bulk.commit_group;\ncp.async.bulk.wait_group 0;", "", loc, ip)

    @dsl_user_op
    def _fabric_put(le, offset, counter_offset, src, nbytes, mbar, *, loc=None, ip=None) -> None:
        """Counted write of ``nbytes`` from shared ``src`` to byte ``offset`` of endpoint ``le``; the fabric adds
        ``nbytes`` to the 8-byte counter at ``counter_offset`` of the same endpoint once the data is visible."""
        _asm(
            None,
            [
                _v(le, Int32, loc, ip),
                _v(offset, Int64, loc, ip),
                _v(counter_offset, Int64, loc, ip),
                _v(src, Int32, loc, ip),
                _v(nbytes, Int32, loc, ip),
                _v(mbar, Int32, loc, ip),
            ],
            "fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric"
            ".counted::bytes.relaxed.sys.b128 [$0, $1, $2], [$3], $4, [$5];",
            "r,l,l,r,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _fabric_submit_wait(*, loc=None, ip=None) -> None:
        _asm(None, [], "fabric.submit;\nfabric.wait.sync_restrict::reads;", "", loc, ip)

    @dsl_user_op
    def _fence_fabric_acquire(*, loc=None, ip=None) -> None:
        _asm(None, [], "fence.proxy.generic::fabric.alias.acquire.sys;", "", loc, ip)

    @dsl_user_op
    def _fence_proxy_async_global(*, loc=None, ip=None) -> None:
        _asm(None, [], "fence.proxy.async.global;", "", loc, ip)

    @dsl_user_op
    def _ld_acquire_b32(addr, *, loc=None, ip=None) -> Int32:
        return Int32(
            _asm(
                T.i32(),
                [_v(addr, Int64, loc, ip)],
                "ld.acquire.gpu.global.b32 $0, [$1];",
                "=r,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _pzo_b32(w, *, loc=None, ip=None) -> Int32:
        """Each bf16 -0.0 half of ``w`` becomes +0.0 (x + +0 is x for every other value under RN)."""
        return Int32(
            _asm(
                T.i32(),
                [_v(w, Int32, loc, ip)],
                "{\n.reg .b32 z;\nmov.b32 z, 0;\nadd.rn.bf16x2 $0, $1, z;\n}",
                "=r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _bf16x2_to_f32(w, *, loc=None, ip=None):
        res = _asm(
            llvm.StructType.get_literal([T.f32(), T.f32()]),
            [_v(w, Int32, loc, ip)],
            "{\n.reg .b32 l, h;\nshl.b32 l, $2, 16;\nand.b32 h, $2, 0xffff0000;\n"
            "mov.b32 $0, l;\nmov.b32 $1, h;\n}",
            "=f,=f,r",
            loc,
            ip,
        )
        return [
            cutlass.Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2)
        ]

    @dsl_user_op
    def _f32x2_to_bf16x2(lo, hi, *, loc=None, ip=None) -> Int32:
        return Int32(
            _asm(
                T.i32(),
                [_v(hi, cutlass.Float32, loc, ip), _v(lo, cutlass.Float32, loc, ip)],
                "cvt.rn.bf16x2.f32 $0, $1, $2;",
                "=r,f,f",
                loc,
                ip,
            )
        )

    def _group8(a, j: int, o: int):
        return ((a[j, o] + a[j, o + 1]) + (a[j, o + 2] + a[j, o + 3])) + (
            (a[j, o + 4] + a[j, o + 5]) + (a[j, o + 6] + a[j, o + 7])
        )

    @cute.jit
    def _spin_ne(addr: Int64, sentinel: Int32, timeout_ns: Int64) -> Int32:
        """Relaxed system-scope polling of an int32 until it differs from ``sentinel``; traps after the timeout."""
        start = _globaltimer()
        val = _ld_sys_b32(addr)
        while val == sentinel:
            if _globaltimer() - start > timeout_ns:
                _trap()
            val = _ld_sys_b32(addr)
        return val

    @cute.jit
    def _spin_eq(addr: Int64, expected: Int32, timeout_ns: Int64):
        start = _globaltimer()
        while _ld_sys_b32(addr) != expected:
            if _globaltimer() - start > timeout_ns:
                _trap()

    @cute.jit
    def _sleep_ns(ns: int):
        start = _globaltimer()
        while _globaltimer() - start < Int64(ns):
            pass

    @dsl_user_op
    def _nanosleep(ns, *, loc=None, ip=None) -> None:
        _asm(None, [_v(ns, Int32, loc, ip)], "nanosleep.u32 $0;", "r", loc, ip)

    @cute.jit
    def _spin_ge_acquire(addr: Int64, target: Int32, timeout_ns: Int64):
        """Acquire polling with a 32..256 ns sleep between polls: one waiter per push warp."""
        start = _globaltimer()
        delay = Int32(32)
        while _ld_acquire_b32(addr) < target:
            if _globaltimer() - start > timeout_ns:
                _trap()
            _nanosleep(delay)
            delay = cutlass.min(delay * 2, Int32(256))

    @cute.jit
    def _spin_ge64(addr: Int64, target: Int64, timeout_ns: Int64):
        start = _globaltimer()
        while _ld_sys_b64(addr) < target:
            if _globaltimer() - start > timeout_ns:
                _trap()

    class _Layout:
        """Per-rank workspace offsets of the stock layout (``moe_a2a_get_workspace_layout``)."""

        def __init__(self, comm):
            from tensorrt_llm.bindings import internal as _internal

            m = comm.moe_a2a_metainfo

            def field(name):
                return int(m[int(getattr(_internal.thop, f"MOE_A2A_{name}"))])

            self.flag = field("FLAG_VAL_OFFSET_INDEX")
            self.local_token_counter = field("LOCAL_TOKEN_COUNTER_OFFSET_INDEX")
            self.send_counters = field("SEND_COUNTERS_OFFSET_INDEX")
            self.recv_counters = field("RECV_COUNTERS_OFFSET_INDEX")
            self.dispatch_flags = field("DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX")
            self.dispatch_counters = field("DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX")
            self.dispatch_baseline = field("DISPATCH_COUNTER_BASELINE_OFFSET_INDEX")
            self.topk_ranks = field("TOPK_TARGET_RANKS_OFFSET_INDEX")
            self.topk_indices = field("TOPK_TARGET_INDICES_OFFSET_INDEX")
            self.dispatch_payload = field("DISPATCH_PAYLOAD_OFFSET_INDEX")
            self.combine_flags = field("COMBINE_COMPLETION_FLAGS_OFFSET_INDEX")
            self.combine_counters = field("COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX")
            self.combine_baseline = field("COMBINE_COUNTER_BASELINE_OFFSET_INDEX")
            self.combine_input = field("COMBINE_INPUT_OFFSET_INDEX")
            self.combine_recv = field("COMBINE_RECV_OFFSET_INDEX")
            self.combine_recv_size = field("COMBINE_RECV_SIZE_INDEX")
            self.max_tokens = field("MAX_NUM_TOKENS_INDEX")

        def key(self):
            return tuple(sorted(vars(self).items()))

    class _PrepareDispatch:
        def __init__(self, ep: int, lay: Tuple, pdl: bool):
            self.ep, self.lay, self.pdl = ep, dict(lay), pdl

        @cute.jit
        def __call__(self, mWs: cute.Tensor, stream):
            self.kernel(mWs).launch(
                grid=[1, 1, 1],
                block=[max(32, (self.ep + 31) // 32 * 32), 1, 1],
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.kernel
        def kernel(self, mWs: cute.Tensor):
            L = self.lay
            my_ws = mWs.iterator.toint()
            tidx, _, _ = cute.arch.thread_idx()
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
            if tidx == 0:
                _st_b32(my_ws + L["flag"], _ld_b32(my_ws + L["flag"]) + 1)
                _st_b32(my_ws + L["local_token_counter"], Int32(0))
            cute.arch.sync_threads()
            if tidx < self.ep:
                flag = _ld_b32(my_ws + L["flag"])
                nxt = ((flag - 1) >> 1) & 1 ^ 1
                _st_b32(my_ws + L["send_counters"] + tidx * 4, Int32(0))
                _st_b32(my_ws + L["recv_counters"] + (nxt * self.ep + tidx) * 4, Int32(-1))

    class _Dispatch:
        def __init__(
            self,
            ep: int,
            top_k: int,
            num_experts: int,
            payload_bytes: Tuple[int, ...],
            ids_payload: int,
            lay: Tuple,
            pdl: bool,
            fgs: bool,
            delay_ns: int = 0,
        ):
            assert (
                ep <= 32 and top_k <= 32 and top_k % 4 == 0 and len(payload_bytes) <= _MAX_PAYLOADS
            )
            assert all(b % 16 == 0 for b in payload_bytes)
            self.ep, self.top_k, self.num_experts = ep, top_k, num_experts
            self.pb = payload_bytes
            self.total = sum(payload_bytes)
            self.ids_payload = ids_payload
            self.lay, self.pdl = dict(lay), pdl
            self.fgs = fgs
            self.delay_ns = delay_ns

        @cute.jit
        def __call__(
            self,
            mIds: cute.Tensor,
            src0: Int64,
            src1: Int64,
            src2: Int64,
            src3: Int64,
            ws_base: Int64,
            ws_stride: Int64,
            rank: Int32,
            n_local: Int32,
            max_tokens: Int32,
            le_base: Int32,
            rcv0: Int64,
            rcv1: Int64,
            rcv2: Int64,
            rcv3: Int64,
            timeout_ns: Int64,
            stream,
        ):
            grid = cutlass.max(n_local, Int32(1))
            self.kernel(
                mIds,
                src0,
                src1,
                src2,
                src3,
                ws_base,
                ws_stride,
                rank,
                n_local,
                max_tokens,
                le_base,
                rcv0,
                rcv1,
                rcv2,
                rcv3,
                timeout_ns,
            ).launch(
                grid=[grid, 1, 1],
                block=[_THREADS, 1, 1],
                smem=2 * _MBAR_BYTES + 2 * self.top_k * 4 + 32 * 8 * 4 + 64 * 4 + self.total + 256,
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.jit
        def _target(self, e: Int32) -> Int32:
            base = self.num_experts // self.ep
            rem = self.num_experts % self.ep
            t = Int32(-1)
            if e >= 0:
                if e < self.num_experts:
                    if cutlass.const_expr(rem == 0):
                        t = e // base
                    else:
                        split = rem * (base + 1)
                        if e < split:
                            t = e // (base + 1)
                        else:
                            t = rem + (e - split) // base
            return t

        @cute.kernel
        def kernel(
            self,
            mIds: cute.Tensor,
            src0: Int64,
            src1: Int64,
            src2: Int64,
            src3: Int64,
            ws_base: Int64,
            ws_stride: Int64,
            rank: Int32,
            n_local: Int32,
            max_tokens: Int32,
            le_base: Int32,
            rcv0: Int64,
            rcv1: Int64,
            rcv2: Int64,
            rcv3: Int64,
            timeout_ns: Int64,
        ):
            L = self.lay
            ids = mIds.iterator.toint()
            EP, K = self.ep, self.top_k
            NP = len(self.pb)
            srcs = [src0, src1, src2, src3]
            rcvs = [rcv0, rcv1, rcv2, rcv3]
            tidx, _, _ = cute.arch.thread_idx()
            tok, _, _ = cute.arch.block_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            lane = cute.arch.lane_idx()

            smem = SmemAllocator()
            mbar = smem.allocate_array(Int64, 2 * _MBAR_BYTES // 8, byte_alignment=_MBAR_BYTES)
            s_rank = smem.allocate_tensor(Int32, cute.make_layout((K,)), byte_alignment=16)
            s_idx = smem.allocate_tensor(Int32, cute.make_layout((K,)), byte_alignment=16)
            s_red = smem.allocate_tensor(
                Int32, cute.make_layout((32, _THREADS // 32)), byte_alignment=16
            )
            s_cnt = smem.allocate_tensor(Int32, cute.make_layout((64,)), byte_alignment=16)
            s_nsend = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
            staging = smem.allocate_array(cutlass.Uint8, self.total, byte_alignment=128)
            bar_load = Int32(mbar.toint())
            bar_put = bar_load + _MBAR_BYTES
            stage = Int32(staging.toint())

            my_ws = ws_base + Int64(rank) * ws_stride
            elected = tok == 0

            cute.arch.griddepcontrol_wait()
            flag = _ld_b32(my_ws + L["flag"])
            parity = ((flag - 1) >> 1) & 1

            if n_local > 0:
                if tidx == 0:
                    _mbar_init(bar_load, 1)
                    off = 0
                    for p in cutlass.range_constexpr(NP):
                        _bulk_g2s(
                            stage + off,
                            srcs[p] + Int64(tok) * self.pb[p],
                            Int32(self.pb[p]),
                            bar_load,
                        )
                        off += self.pb[p]
                    _mbar_expect_tx(bar_load, Int32(self.total))

            # Per-target count of the sends of tokens [0, limit): this token's slot at target r is that count over
            # the earlier tokens. CTA 0 (first to run, no earlier tokens) counts all tokens and publishes the totals.
            limit = tok
            if elected:
                limit = n_local
            acc = Int32(0)
            for base in cutlass.range(0, limit, _THREADS, unroll=1):
                j = base + tidx
                sends = Int32(0)
                if j < limit:
                    for q in cutlass.range_constexpr(K // 4):
                        w = _ld_v4(ids + (Int64(j) * K + 4 * q) * 4)
                        for i in cutlass.range_constexpr(4):
                            tr = self._target(w[i])
                            if tr >= 0:
                                sends = sends | (Int32(1) << tr)
                for r in cutlass.range_constexpr(EP):
                    c = cute.arch.popc(cute.arch.vote_ballot_sync(((sends >> r) & 1) != 0))
                    if lane == r:
                        acc = acc + c
            s_red[lane, warp] = acc
            cute.arch.sync_threads()
            if tidx < EP:
                tot = Int32(0)
                for w in cutlass.range_constexpr(_THREADS // 32):
                    tot = tot + s_red[tidx, w]
                s_cnt[tidx] = tot
            cute.arch.sync_threads()

            if n_local > 0:
                if tidx < K:
                    e = _ld_b32(ids + (Int64(tok) * K + tidx) * 4)
                    tr = self._target(e)
                    peers = _match_any(tr, (1 << K) - 1)
                    send = Int32(0)
                    if tr >= 0:
                        if (peers & ((Int32(1) << tidx) - 1)) == 0:
                            send = Int32(1)
                    r_store = Int32(-1)
                    i_store = Int32(-1)
                    if send != 0:
                        r_store = tr
                        i_store = Int32(0)
                        if tok != 0:
                            i_store = s_cnt[tr]
                    rec = Int64(tok) * K + tidx
                    _st_b32(my_ws + L["topk_ranks"] + rec * 4, r_store)
                    _st_b32(my_ws + L["topk_indices"] + rec * 4, i_store)
                    sending = cute.arch.vote_ballot_sync(send != 0, (1 << K) - 1)
                    if send != 0:
                        at = cute.arch.popc(sending & ((Int32(1) << tidx) - 1))
                        s_rank[at] = r_store
                        s_idx[at] = i_store
                    if tidx == 0:
                        s_nsend[0] = cute.arch.popc(sending)
            cute.arch.sync_threads()

            if elected:
                if warp == 0:
                    if lane < EP:
                        _st_b32(my_ws + L["send_counters"] + lane * 4, s_cnt[lane])
                        peer_ws = ws_base + Int64(lane) * ws_stride
                        _st_sys_b32(
                            peer_ws + L["recv_counters"] + (parity * EP + rank) * 4, s_cnt[lane]
                        )

            if cutlass.const_expr(self.delay_ns > 0):
                _sleep_ns(self.delay_ns)
            if n_local > 0:
                nsend = s_nsend[0]
                has_self = Int32(0)
                has_remote = Int32(0)
                for i in cutlass.range(nsend, unroll=1):
                    if s_rank[i] == rank:
                        has_self = Int32(1)
                    else:
                        has_remote = Int32(1)
                if tidx == 0:
                    _mbar_wait(bar_load, Int32(0))
                cute.arch.sync_threads()
                if tidx == 32:
                    if has_self != 0:
                        off = 0
                        for p in cutlass.range_constexpr(NP):
                            for i in cutlass.range(nsend, unroll=1):
                                if s_rank[i] == rank:
                                    slot = Int64(rank) * max_tokens + s_idx[i]
                                    _bulk_s2g(
                                        my_ws + rcvs[p] + slot * self.pb[p],
                                        stage + off,
                                        Int32(self.pb[p]),
                                    )
                            off += self.pb[p]
                        _bulk_commit_wait()
                if tidx == 0:
                    if has_remote != 0:
                        _mbar_init(bar_put, 1)
                        counter = Int64(L["dispatch_counters"]) + Int64(rank) * _COUNTER_STRIDE
                        off = 0
                        for p in cutlass.range_constexpr(NP):
                            for i in cutlass.range(nsend, unroll=1):
                                if s_rank[i] != rank:
                                    slot = Int64(rank) * max_tokens + s_idx[i]
                                    _fabric_put(
                                        le_base + s_rank[i],
                                        rcvs[p] + slot * self.pb[p],
                                        counter,
                                        stage + off,
                                        Int32(self.pb[p]),
                                        bar_put,
                                    )
                            off += self.pb[p]
                        _fabric_submit_wait()

            cute.arch.griddepcontrol_launch_dependents()

            if elected:
                if warp == 0:
                    if lane < EP:
                        _spin_ne(
                            my_ws + L["recv_counters"] + (parity * EP + lane) * 4,
                            Int32(-1),
                            timeout_ns,
                        )
            cute.arch.sync_threads()
            if elected:
                if cutlass.const_expr(self.ids_payload >= 0):
                    for i in cutlass.range(tidx, EP * max_tokens, _THREADS, unroll=1):
                        peer = i // max_tokens
                        t = i % max_tokens
                        cnt = _ld_b32(my_ws + L["recv_counters"] + (parity * EP + peer) * 4)
                        if t >= cnt:
                            row = my_ws + rcvs[self.ids_payload] + Int64(i) * (K * 4)
                            for q in cutlass.range_constexpr(K // 4):
                                _st_v4(row + q * 16, Int32(-1), Int32(-1), Int32(-1), Int32(-1))
                            for q in cutlass.range_constexpr(K % 4):
                                _st_b32(row + (K // 4 * 4 + q) * 4, Int32(-1))
                if warp == 0:
                    if lane < EP:
                        peer_ws = ws_base + Int64(lane) * ws_stride
                        _st_sys_b32(peer_ws + L["dispatch_flags"] + rank * 4, flag)
                        if cutlass.const_expr(not self.fgs):
                            if lane != rank:
                                cnt = _ld_b32(my_ws + L["recv_counters"] + (parity * EP + lane) * 4)
                                expected = Int64(cnt) * self.total
                                if expected > 0:
                                    base_addr = my_ws + L["dispatch_baseline"] + lane * 8
                                    target = _ld_b64(base_addr) + expected
                                    _spin_ge64(
                                        my_ws
                                        + L["dispatch_counters"]
                                        + Int64(lane) * _COUNTER_STRIDE,
                                        target,
                                        timeout_ns,
                                    )
                                    _st_b64(base_addr, target)

    class _PrepareCombine:
        def __init__(self, ep: int, nbytes: int, lay: Tuple, pdl: bool):
            assert nbytes % 16 == 0
            self.ep, self.nbytes, self.lay, self.pdl = ep, nbytes, dict(lay), pdl

        @cute.jit
        def __call__(
            self,
            mY: cute.Tensor,
            my_ws: Int64,
            src_stride: Int64,
            rank: Int32,
            max_tokens: Int32,
            stream,
        ):
            self.kernel(mY, my_ws, src_stride, rank, max_tokens).launch(
                grid=[max_tokens, 1, 1], block=[_THREADS, 1, 1], stream=stream, use_pdl=self.pdl
            )

        @cute.kernel
        def kernel(
            self, mY: cute.Tensor, my_ws: Int64, src_stride: Int64, rank: Int32, max_tokens: Int32
        ):
            L = self.lay
            src = mY.iterator.toint()
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
            flag = _ld_b32(my_ws + L["flag"])
            parity = ((flag - 1) >> 1) & 1
            if bidx == 0:
                if tidx == 0:
                    _st_b32(my_ws + L["flag"], flag + 1)
            if bidx < _ld_b32(my_ws + L["recv_counters"] + (parity * self.ep + rank) * 4):
                g = Int64(rank) * max_tokens + bidx
                s = src + g * src_stride
                d = my_ws + L["combine_recv"] + g * self.nbytes
                for v in cutlass.range(tidx, self.nbytes // 16, _THREADS, unroll=1):
                    w = _ld_v4(s + v * 16)
                    _st_v4(d + v * 16, w[0], w[1], w[2], w[3])

    class _Push:
        def __init__(self, ep: int, nbytes: int, lay: Tuple, pdl: bool, fgs: bool):
            self.ep, self.nbytes, self.lay, self.pdl, self.fgs = ep, nbytes, dict(lay), pdl, fgs

        @cute.jit
        def __call__(
            self,
            mY: cute.Tensor,
            ws_base: Int64,
            ws_stride: Int64,
            src_stride: Int64,
            rank: Int32,
            max_tokens: Int32,
            static_max: Int32,
            le_base: Int32,
            blocks_per_rank: Int32,
            fin_done: Int64,
            fin_expected: Int32,
            timeout_ns: Int64,
            stream,
        ):
            self.kernel(
                mY,
                ws_base,
                ws_stride,
                src_stride,
                rank,
                max_tokens,
                static_max,
                le_base,
                fin_done,
                fin_expected,
                timeout_ns,
            ).launch(
                grid=[self.ep, blocks_per_rank, 1],
                block=[_PUSH_WARPS * 32, 1, 1],
                smem=_PUSH_WARPS * (_MBAR_BYTES + self.nbytes) + 128,
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mY: cute.Tensor,
            ws_base: Int64,
            ws_stride: Int64,
            src_stride: Int64,
            rank: Int32,
            max_tokens: Int32,
            static_max: Int32,
            le_base: Int32,
            fin_done: Int64,
            fin_expected: Int32,
            timeout_ns: Int64,
        ):
            L = self.lay
            src = mY.iterator.toint()
            tidx, _, _ = cute.arch.thread_idx()
            source, by, _ = cute.arch.block_idx()
            _, nby, _ = cute.arch.grid_dim()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            lane = cute.arch.lane_idx()
            smem = SmemAllocator()
            buf = smem.allocate_array(
                cutlass.Uint8, _PUSH_WARPS * (_MBAR_BYTES + self.nbytes), byte_alignment=128
            )
            my_ws = ws_base + Int64(rank) * ws_stride

            if cutlass.const_expr(self.fgs):
                # Row t of source block s is final once fin_done[s * max_tokens + t] counts its finalize units; the
                # round flag still holds the dispatch value (the reduce advances it).
                cute.arch.griddepcontrol_launch_dependents()
                flag = _ld_b32(my_ws + L["flag"]) + 1
            else:
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
                flag = _ld_b32(my_ws + L["flag"])
            if by == 0:
                if tidx == 0:
                    _st_sys_b32(
                        ws_base + Int64(source) * ws_stride + L["combine_flags"] + rank * 4, flag
                    )
            if source != rank:
                parity = ((flag - 1) >> 1) & 1
                n = _ld_b32(my_ws + L["recv_counters"] + (parity * self.ep + source) * 4)
                if n > 0:
                    if n <= max_tokens:
                        if lane == 0:
                            slot = Int32(buf.toint()) + warp * (_MBAR_BYTES + self.nbytes)
                            bar_load = slot
                            bar_put = slot + 8
                            stage = slot + _MBAR_BYTES
                            _mbar_init(bar_load, 1)
                            _mbar_init(bar_put, 1)
                            phase = Int32(0)
                            for t in cutlass.range(
                                by * _PUSH_WARPS + warp, n, nby * _PUSH_WARPS, unroll=1
                            ):
                                if cutlass.const_expr(self.fgs):
                                    _spin_ge_acquire(
                                        fin_done + (Int64(source) * max_tokens + t) * 4,
                                        fin_expected,
                                        timeout_ns,
                                    )
                                    _fence_proxy_async_global()
                                _bulk_g2s(
                                    stage,
                                    src + (Int64(source) * max_tokens + t) * src_stride,
                                    Int32(self.nbytes),
                                    bar_load,
                                )
                                _mbar_expect_tx(bar_load, Int32(self.nbytes))
                                _mbar_wait(bar_load, phase & 1)
                                phase = phase + 1
                                _fabric_put(
                                    le_base + source,
                                    L["combine_recv"]
                                    + (Int64(rank) * max_tokens + t) * self.nbytes,
                                    L["combine_counters"]
                                    + (Int64(rank) * static_max + t) * _COUNTER_STRIDE,
                                    stage,
                                    Int32(self.nbytes),
                                    bar_put,
                                )
                                _fabric_submit_wait()
            if cutlass.const_expr(self.fgs):
                # Completion of this grid still implies the finalize's (and, through it, the expert path's).
                cute.arch.griddepcontrol_wait()

    class _Reduce:
        """out[t] = sum over the k contributions of token t, in the stock order: two sources add directly; more
        sources add in groups of eight as ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)), absent ones as +0, and
        the groups add pairwise; fp32, one round to bf16."""

        def __init__(
            self,
            ep: int,
            top_k: int,
            hidden: int,
            lay: Tuple,
            pdl: bool,
            fgs: bool,
            pzo: bool,
            delay_ns: int = 0,
        ):
            assert top_k <= 32 and (hidden * 2) % 16 == 0
            self.ep, self.top_k, self.hidden, self.lay, self.pdl = ep, top_k, hidden, dict(lay), pdl
            self.fgs, self.pzo = fgs, pzo
            self.delay_ns = delay_ns

        @cute.jit
        def __call__(
            self,
            mOut: cute.Tensor,
            ws_base: Int64,
            ws_stride: Int64,
            rank: Int32,
            n_local: Int32,
            max_tokens: Int32,
            static_max: Int32,
            timeout_ns: Int64,
            own_base: Int64,
            stream,
        ):
            self.kernel(
                mOut,
                ws_base,
                ws_stride,
                rank,
                n_local,
                max_tokens,
                static_max,
                timeout_ns,
                own_base,
            ).launch(
                grid=[cutlass.max(n_local, Int32(1)), 1, 1],
                block=[_THREADS, 1, 1],
                smem=512,
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mOut: cute.Tensor,
            ws_base: Int64,
            ws_stride: Int64,
            rank: Int32,
            n_local: Int32,
            max_tokens: Int32,
            static_max: Int32,
            timeout_ns: Int64,
            own_base: Int64,
        ):
            L = self.lay
            out = mOut.iterator.toint()
            K = self.top_k
            nbytes = self.hidden * 2
            tidx, _, _ = cute.arch.thread_idx()
            tok, _, _ = cute.arch.block_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            lane = cute.arch.lane_idx()
            smem = SmemAllocator()
            s_src = smem.allocate_tensor(Int64, cute.make_layout((32,)), byte_alignment=16)
            s_n = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
            my_ws = ws_base + Int64(rank) * ws_stride
            inbox = my_ws + L["combine_recv"]

            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
            if n_local > 0:
                if warp == 0:
                    tr = Int32(-1)
                    di = Int32(-1)
                    if lane < K:
                        rec = Int64(tok) * K + lane
                        tr = _ld_b32(my_ws + L["topk_ranks"] + rec * 4)
                        di = _ld_b32(my_ws + L["topk_indices"] + rec * 4)
                        if tr >= 0:
                            if di >= 0:
                                if tr != rank:
                                    slot = Int64(tr) * static_max + di
                                    base_addr = my_ws + L["combine_baseline"] + slot * 8
                                    target = _ld_b64(base_addr) + nbytes
                                    _spin_ge64(
                                        my_ws + L["combine_counters"] + slot * _COUNTER_STRIDE,
                                        target,
                                        timeout_ns,
                                    )
                                    _st_b64(base_addr, target)
                    _fence_fabric_acquire()
                    valid = cute.arch.vote_ballot_sync(di >= 0)
                    if lane == 0:
                        s_n[0] = cute.arch.popc(valid)
                    if di >= 0:
                        row_src = inbox + (Int64(tr) * max_tokens + di) * nbytes
                        if cutlass.const_expr(self.fgs):
                            # The push copies no own rows: they are read from the payload itself.
                            if tr == rank:
                                row_src = own_base + (Int64(rank) * max_tokens + di) * nbytes
                        s_src[cute.arch.popc(valid & ((Int32(1) << lane) - 1))] = row_src
                cute.arch.sync_threads()
                # Threads load slots warp 0 waited for: each fences after the barrier.
                _fence_fabric_acquire()

                if cutlass.const_expr(self.delay_ns > 0):
                    _sleep_ns(self.delay_ns)
                count = s_n[0]
                dst = out + Int64(tok) * nbytes
                ns = (K + 7) // 8 * 8
                a = cute.make_rmem_tensor(cute.make_layout((8, ns)), cutlass.Float32)
                res = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Float32)
                for v in cutlass.range(tidx, nbytes // 16, _THREADS, unroll=1):
                    a.fill(0.0)
                    for s in cutlass.range_constexpr(ns):
                        if s < count:
                            w = _ld_v4(s_src[s] + v * 16)
                            for q in cutlass.range_constexpr(4):
                                lo, hi = _bf16x2_to_f32(w[q])
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
                    words = [_f32x2_to_bf16x2(res[2 * q], res[2 * q + 1]) for q in range(4)]
                    if cutlass.const_expr(self.pzo):
                        words = [_pzo_b32(w) for w in words]
                    _st_v4(dst + v * 16, *words)

            if tok == 0:
                if warp == 0:
                    flag = _ld_b32(my_ws + L["flag"])
                    if cutlass.const_expr(self.fgs):
                        # No combine prepare ran: this kernel advances the round flag to the combine value.
                        flag = flag + 1
                    if lane < self.ep:
                        _spin_eq(my_ws + L["combine_flags"] + lane * 4, flag, timeout_ns)
                    if cutlass.const_expr(self.fgs):
                        cute.arch.sync_warp()
                        if lane == 0:
                            _st_b32(my_ws + L["flag"], flag)

    def _fake(dtype):
        return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=16)

    @functools.lru_cache(maxsize=None)
    def _compile(kind: str, *cfg):
        """Every entry takes one device tensor first: the TVM-FFI stream binding needs it to find the device."""
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        i64, i32 = Int64(0), Int32(0)
        opt = "--enable-tvm-ffi"
        if kind == "prepare_dispatch":
            return cute.compile(_PrepareDispatch(*cfg), _fake(cutlass.Uint8), stream, options=opt)
        if kind == "dispatch":
            return cute.compile(
                _Dispatch(*cfg),
                _fake(Int32),
                *([i64] * 6),
                i32,
                i32,
                i32,
                i32,
                *([i64] * 5),
                stream,
                options=opt,
            )
        if kind == "prepare_combine":
            return cute.compile(
                _PrepareCombine(*cfg),
                _fake(cutlass.BFloat16),
                i64,
                i64,
                i32,
                Int32(1),
                stream,
                options=opt,
            )
        if kind == "push":
            return cute.compile(
                _Push(*cfg),
                _fake(cutlass.BFloat16),
                i64,
                i64,
                i64,
                i32,
                i32,
                i32,
                i32,
                Int32(1),
                i64,
                i32,
                i64,
                stream,
                options=opt,
            )
        if kind == "reduce":
            return cute.compile(
                _Reduce(*cfg),
                _fake(cutlass.BFloat16),
                i64,
                i64,
                i32,
                i32,
                i32,
                i32,
                i64,
                i64,
                stream,
                options=opt,
            )
        raise ValueError(kind)


class RoutingSource(NamedTuple):
    """Receive side of a dispatch whose data wait is left to the routing kernel: this rank's workspace address and
    layout, the EP shape, the rows per source block and the payload bytes per token."""

    my_ws: int
    lay: "_Layout"
    ep: int
    rank: int
    max_tokens: int
    bytes_per_token: int


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class CuteA2A:
    """CuTe DSL dispatch / combine on a stock ``NVLinkOneSided`` that uses CFT counted writes.

    ``dispatch(hidden_states, hidden_states_sf, token_selected_slots, token_final_scales, all_rank_num_tokens)``
    returns the stock tuple: per payload a ``[ep * runtime_max_tokens, width]`` view of this rank's receive region
    (rows ``src * runtime_max_tokens + j``, ``j < count[src]`` valid; expert ids of the other rows are -1).
    ``combine(final_hidden_states)`` takes the bf16 expert output ``[ep * runtime_max_tokens, hidden]`` (also the
    in-workspace view from ``get_combine_payload_tensor_in_workspace``) and returns ``[local_tokens, hidden]``.
    Both keep the stock communicator's dispatch state, so its own dispatch / combine can take either phase.

    Fine-grained-sync modes (opt-in per call; counter words are int32, zero-armed by the caller each step):
    ``dispatch(..., defer_data_wait=True)`` does not wait for the peers' data: the grid ends once the local copies,
    the counts, the sanitize and the flags are done. The data wait then belongs to the routing kernel
    (``routing.run(..., source=a2a.routing_source())``), which waits per source block and advances the per-source
    byte-counter baselines; it must run in every such round.
    ``combine(..., fin_done=c, fin_expected=x)`` skips the combine prepare: the push and the reduce launch with PDL,
    each push warp sends one row as soon as its word ``c[s * max + t] >= x`` (the finalize units of row t of source
    block s), the reduce reads this rank's own rows from the payload instead of a copy, and it advances the round
    flag. ``combine(..., out=slot)`` writes the result into ``slot`` (bf16
    [local_tokens, hidden]) with -0.0 stored as +0.0, for a sentinel-armed edge slot. ``hazard_delay_ns`` delays the
    dispatch data issue / the combine output stores, for the hazard tests only.
    """

    def __init__(self, comm, le_base: Optional[int] = None):
        if not _CUTE_AVAILABLE:
            raise ImportError("CuTe DSL is not available")
        if not comm.can_use_cft_counted_writes:
            raise ValueError(f"{ENV}=cute needs the CFT workspace of NVLinkOneSided")
        if (
            comm.use_low_precision_combine
            or getattr(comm, "_rank_mask_enabled", False)
            or comm.enable_eplb
        ):
            raise ValueError(
                f"{ENV}=cute supports bf16 combine without EPLB statistics or rank masks"
            )
        self.comm = comm
        self.ep, self.rank = comm.ep_size, comm.ep_rank
        self.lay = _Layout(comm)
        self.le_base = find_le_base(self.ep) if le_base is None else le_base
        self._source: Optional[RoutingSource] = None

    @property
    def workspace(self) -> torch.Tensor:
        return self.comm.workspace

    def _ws(self) -> Tuple[int, int, int]:
        ws = self.workspace
        return ws.data_ptr(), ws.stride(0), ws.data_ptr() + self.rank * ws.stride(0)

    def dispatch(
        self,
        hidden_states,
        hidden_states_sf,
        token_selected_slots,
        token_final_scales,
        all_rank_num_tokens: List[int],
        use_dp_padding=None,
        defer_data_wait: bool = False,
        hazard_delay_ns: int = 0,
        **kwargs,
    ):
        comm = self.comm
        if comm._dispatch_state.get("phase") == "dispatched":
            raise RuntimeError("dispatch called twice without an intervening combine")
        if kwargs.get("eplb_local_stats") is not None or kwargs.get("active_rank_mask") is not None:
            raise ValueError(f"{ENV}=cute: EPLB statistics and rank masks are not supported")
        runtime_max = max(all_rank_num_tokens)
        payloads = [hidden_states] + ([hidden_states_sf] if hidden_states_sf is not None else [])
        payloads.append(token_selected_slots)
        if token_final_scales is not None:
            payloads.append(token_final_scales)
        ids_payload = len(payloads) - (2 if token_final_scales is not None else 1)
        n = token_selected_slots.shape[0]
        top_k = comm.top_k
        assert token_selected_slots.dtype == torch.int32 and token_selected_slots.shape[1] == top_k
        assert runtime_max <= self.lay.max_tokens
        widths, nbytes, offsets = [], [], []
        off = self.lay.dispatch_payload
        for p in payloads:
            if p.dim() != 2 or p.shape[0] != n or not p.is_contiguous() or p.data_ptr() % 16:
                raise ValueError(
                    "dispatch payloads must be contiguous 16-byte aligned [tokens, width]"
                )
            widths.append(p.shape[1])
            nbytes.append(p.shape[1] * p.element_size())
            offsets.append(off)
            off = _align(off + self.ep * runtime_max * nbytes[-1], _CACHELINE)
        lay = self.lay.key()
        pdl = _pdl()
        ws_base, ws_stride, my_ws = self._ws()
        _compile("prepare_dispatch", self.ep, lay, pdl)(self.workspace[self.rank])
        fgs = defer_data_wait
        fn = _compile(
            "dispatch",
            self.ep,
            top_k,
            comm.num_experts,
            tuple(nbytes),
            ids_payload,
            lay,
            pdl,
            fgs,
            hazard_delay_ns,
        )
        srcs = [p.data_ptr() for p in payloads] + [0] * (_MAX_PAYLOADS - len(payloads))
        rcvs = offsets + [0] * (_MAX_PAYLOADS - len(offsets))
        fn(
            token_selected_slots.view(-1),
            *srcs,
            ws_base,
            ws_stride,
            self.rank,
            n,
            runtime_max,
            self.le_base,
            *rcvs,
            _TIMEOUT_NS,
        )
        self._source = (
            RoutingSource(my_ws, self.lay, self.ep, self.rank, runtime_max, sum(nbytes))
            if fgs
            else None
        )
        comm._dispatch_state.update(
            phase="dispatched",
            combine_payload_offset=self.lay.combine_input,
            local_num_tokens=n,
            runtime_max_tokens_per_rank=runtime_max,
            eplb_gathered_stats=None,
            active_rank_mask_snapshot=comm._watchdog_coordinator.capture_active_rank_mask(None),
        )
        rank_ws = self.workspace[self.rank]
        views = []
        for p, w, b, o in zip(payloads, widths, nbytes, offsets):
            views.append(
                rank_ws[o : o + self.ep * runtime_max * b]
                .view(p.dtype)
                .view(self.ep * runtime_max, w)
            )
        if hidden_states_sf is None:
            views.insert(1, None)
        if token_final_scales is None:
            views.append(None)
        return tuple(views)

    def routing_source(self) -> "RoutingSource":
        """The receive side of the last ``defer_data_wait`` dispatch, for ``routing.run(source=...)``."""
        if self._source is None:
            raise RuntimeError("routing_source() needs a dispatch with defer_data_wait=True")
        return self._source

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        fin_done: Optional[torch.Tensor] = None,
        fin_expected: int = 0,
        out: Optional[torch.Tensor] = None,
        hazard_delay_ns: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        comm = self.comm
        st = comm._dispatch_state
        if st.get("phase") != "dispatched":
            raise RuntimeError("combine called before a successful dispatch")
        n, runtime_max = st["local_num_tokens"], st["runtime_max_tokens_per_rank"]
        y = final_hidden_states.reshape(self.ep * runtime_max, -1)
        if y.dtype != torch.bfloat16 or not y.is_contiguous() or y.data_ptr() % 16:
            raise ValueError(f"{ENV}=cute combine takes a contiguous bf16 payload")
        hidden = y.shape[1]
        nbytes = hidden * 2
        if self.ep * runtime_max * nbytes > self.lay.combine_recv_size:
            raise ValueError("combine payload exceeds the receive region")
        lay = self.lay.key()
        pdl = _pdl()
        ws_base, ws_stride, my_ws = self._ws()
        pzo = out is not None
        if out is None:
            out = torch.empty(n, hidden, dtype=torch.bfloat16, device=y.device)
        elif (
            out.shape[0] < n
            or out.shape[-1] != hidden
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
        ):
            raise ValueError(f"combine out must be contiguous bf16 [>= {n}, {hidden}]")
        fgs = fin_done is not None
        if not fgs:
            _compile("prepare_combine", self.ep, nbytes, lay, pdl)(
                y.view(-1), my_ws, nbytes, self.rank, runtime_max
            )
        # FGS: one row per push warp, so each row goes out as soon as its finalize units are counted.
        rows_per_warp = 1 if fgs else 2
        blocks = min(
            32,
            max(
                1, (runtime_max + rows_per_warp * _PUSH_WARPS - 1) // (rows_per_warp * _PUSH_WARPS)
            ),
        )
        _compile("push", self.ep, nbytes, lay, pdl or fgs, fgs)(
            y.view(-1),
            ws_base,
            ws_stride,
            nbytes,
            self.rank,
            runtime_max,
            self.lay.max_tokens,
            self.le_base,
            blocks,
            fin_done.data_ptr() if fgs else 0,
            fin_expected,
            _TIMEOUT_NS,
        )
        _compile("reduce", self.ep, comm.top_k, hidden, lay, pdl or fgs, fgs, pzo, hazard_delay_ns)(
            out.view(-1),
            ws_base,
            ws_stride,
            self.rank,
            n,
            runtime_max,
            self.lay.max_tokens,
            _TIMEOUT_NS,
            y.data_ptr() if fgs else 0,
        )
        comm.reset_state()
        return out


def install(comm) -> Optional[CuteA2A]:
    """With ``TLLM_K3_COMM=cute``, route ``comm.dispatch`` / ``comm.combine`` of a CFT ``NVLinkOneSided`` through
    :class:`CuteA2A` for the batches stock runs with CFT, with at most ``TLLM_K3_MOE_MAX_ROWS`` tokens per rank and
    without EPLB statistics or rank masks; other calls keep the stock path. Strategy kwargs the scheduler passes to
    every strategy are ignored, as stock does."""
    if not enabled():
        return None
    from ..moe_chain import within_max_rows

    cute_a2a = CuteA2A(comm)
    stock_dispatch, stock_combine = comm.dispatch, comm.combine

    def plain(kwargs) -> bool:
        return kwargs.get("eplb_local_stats") is None and kwargs.get("active_rank_mask") is None

    def dispatch(
        hidden_states,
        hidden_states_sf,
        token_selected_slots,
        token_final_scales,
        all_rank_num_tokens,
        use_dp_padding=None,
        **kwargs,
    ):
        runtime_max = max(all_rank_num_tokens)
        if (
            plain(kwargs)
            and within_max_rows(runtime_max)
            and comm.use_cft_for_dispatch(runtime_max)
            and comm.use_cft_for_combine(runtime_max)
        ):
            return cute_a2a.dispatch(
                hidden_states,
                hidden_states_sf,
                token_selected_slots,
                token_final_scales,
                all_rank_num_tokens,
            )
        return stock_dispatch(
            hidden_states,
            hidden_states_sf,
            token_selected_slots,
            token_final_scales,
            all_rank_num_tokens,
            use_dp_padding,
            **kwargs,
        )

    def combine(final_hidden_states, **kwargs):
        runtime_max = comm._dispatch_state.get("runtime_max_tokens_per_rank") or 0
        if (
            plain(kwargs)
            and within_max_rows(runtime_max)
            and comm.use_cft_for_combine(runtime_max)
            and final_hidden_states.dtype == torch.bfloat16
        ):
            return cute_a2a.combine(final_hidden_states)
        return stock_combine(final_hidden_states, **kwargs)

    comm.dispatch, comm.combine = dispatch, combine
    return cute_a2a
