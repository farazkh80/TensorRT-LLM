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
           and rounds once to the payload dtype). With the low-precision combine the rows travel as e4m3 (the stock
           unscaled saturating cast per element).
lamport    combine alternative: the push stores rows with TMA stores into the peers' -0 armed inboxes (two, by round
           parity, in a separate symmetric buffer); the consumer checks each row it reads and re-arms it.

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
    from cutlass.cute.nvgpu import cpasync
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils import SmemAllocator

    _CUTE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CUTE_AVAILABLE = False

_COUNTER_STRIDE = 256
_MAX_PAYLOADS = 4
_CACHELINE = 128
_THREADS = 256
_PUSH_WARPS = 4
_MBAR_BYTES = 64
_TIMEOUT_NS = 300 * 1_000_000_000
# Lamport combine side buffer (one symmetric MNNVL allocation): a header, then two inboxes selected by the round parity.
_SIDE_INBOX = 8192
_TMA_CHUNK = 128


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
    """
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
            f"CuteA2A: expected one block of {ep_size} logical endpoints near id {nxt}, found {found}"
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

    @dsl_user_op
    def _bf16x4_to_e4m3x4(w0, w1, *, loc=None, ip=None) -> Int32:
        """Four bf16 (two words, element order kept) to four e4m3 bytes, as the stock low-precision combine casts."""
        return Int32(
            _asm(
                T.i32(),
                [_v(w0, Int32, loc, ip), _v(w1, Int32, loc, ip)],
                "{\n.reg .b16 a, b;\ncvt.rn.satfinite.e4m3x2.bf16x2 a, $1;\n"
                "cvt.rn.satfinite.e4m3x2.bf16x2 b, $2;\nmov.b32 $0, {a, b};\n}",
                "=r,r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _e4m3x4_to_f32(w, *, loc=None, ip=None):
        res = _asm(
            llvm.StructType.get_literal([T.f32()] * 4),
            [_v(w, Int32, loc, ip)],
            "{\n.reg .b16 lo, hi, f0, f1, f2, f3;\n.reg .b32 h0, h1;\nmov.b32 {lo, hi}, $4;\n"
            "cvt.rn.f16x2.e4m3x2 h0, lo;\ncvt.rn.f16x2.e4m3x2 h1, hi;\n"
            "mov.b32 {f0, f1}, h0;\nmov.b32 {f2, f3}, h1;\n"
            "cvt.f32.f16 $0, f0;\ncvt.f32.f16 $1, f1;\ncvt.f32.f16 $2, f2;\ncvt.f32.f16 $3, f3;\n}",
            "=f,=f,=f,=f,r",
            loc,
            ip,
        )
        return [
            cutlass.Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)
        ]

    def _lane_masks(lpc: bool) -> Tuple[int, int]:
        """(magnitude mask, sign mask) of the packed elements of a 32-bit word: e4m3 bytes or bf16 halves."""
        return (0x7F7F7F7F, 0x80808080) if lpc else (0x7FFF7FFF, 0x80008000)

    @dsl_user_op
    def _canon(w, lpc: bool, *, loc=None, ip=None) -> Int32:
        """Every -0 element of ``w`` becomes +0; other elements are unchanged (exact per element: the magnitude plus
        the all-ones magnitude carries into the sign bit iff the magnitude is non-zero, never across elements)."""
        mag, sgn = _lane_masks(lpc)
        return Int32(
            _asm(
                T.i32(),
                [_v(w, Int32, loc, ip)],
                f"{{\n.reg .b32 m, n;\nand.b32 m, $1, {mag};\nadd.u32 n, m, {mag};\nand.b32 n, n, {sgn};\n"
                f"and.b32 n, n, $1;\nor.b32 $0, m, n;\n}}",
                "=r,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _ld_vol_v4(addr, *, loc=None, ip=None):
        res = _asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [_v(addr, Int64, loc, ip)],
            "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            loc,
            ip,
        )
        return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]

    @dsl_user_op
    def _ld_vol_v2(addr, *, loc=None, ip=None):
        res = _asm(
            llvm.StructType.get_literal([T.i32()] * 2),
            [_v(addr, Int64, loc, ip)],
            "ld.volatile.global.v2.b32 {$0, $1}, [$2];",
            "=r,=r,l",
            loc,
            ip,
        )
        return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2)]

    @dsl_user_op
    def _st_v2(addr, w0, w1, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [_v(addr, Int64, loc, ip), _v(w0, Int32, loc, ip), _v(w1, Int32, loc, ip)],
            "st.global.v2.b32 [$0], {$1, $2};",
            "l,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _st_shared_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
        _asm(
            None,
            [_v(addr, Int32, loc, ip)] + [_v(w, Int32, loc, ip) for w in (w0, w1, w2, w3)],
            "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
            "r,r,r,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _fence_proxy_async_shared(*, loc=None, ip=None) -> None:
        _asm(None, [], "fence.proxy.async.shared::cta;", "", loc, ip)

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
                        if lane != rank:
                            cnt = _ld_b32(my_ws + L["recv_counters"] + (parity * EP + lane) * 4)
                            expected = Int64(cnt) * self.total
                            if expected > 0:
                                base_addr = my_ws + L["dispatch_baseline"] + lane * 8
                                target = _ld_b64(base_addr) + expected
                                _spin_ge64(
                                    my_ws + L["dispatch_counters"] + Int64(lane) * _COUNTER_STRIDE,
                                    target,
                                    timeout_ns,
                                )
                                _st_b64(base_addr, target)

    class _PrepareCombine:
        """Advances the round flag and copies this rank's own rows into the receive inbox (``nbytes`` per row; e4m3
        cast of the bf16 row with ``lpc``)."""

        def __init__(self, ep: int, nbytes: int, lay: Tuple, pdl: bool, lpc: bool = False):
            assert nbytes % 16 == 0
            self.ep, self.nbytes, self.lay, self.pdl, self.lpc = ep, nbytes, dict(lay), pdl, lpc

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
                    if cutlass.const_expr(self.lpc):
                        a = _ld_v4(s + v * 32)
                        b = _ld_v4(s + v * 32 + 16)
                        _st_v4(
                            d + v * 16,
                            _bf16x4_to_e4m3x4(a[0], a[1]),
                            _bf16x4_to_e4m3x4(a[2], a[3]),
                            _bf16x4_to_e4m3x4(b[0], b[1]),
                            _bf16x4_to_e4m3x4(b[2], b[3]),
                        )
                    else:
                        w = _ld_v4(s + v * 16)
                        _st_v4(d + v * 16, w[0], w[1], w[2], w[3])

    class _Push:
        def __init__(self, ep: int, nbytes: int, lay: Tuple, pdl: bool, lpc: bool = False):
            assert not lpc or nbytes % 512 == 0
            self.ep, self.nbytes, self.lay, self.pdl = ep, nbytes, dict(lay), pdl
            self.lpc = lpc

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
                n = _spin_ne(
                    my_ws + L["recv_counters"] + (parity * self.ep + source) * 4,
                    Int32(-1),
                    timeout_ns,
                )
                if n > 0:
                    if n <= max_tokens:
                        if cutlass.const_expr(self.lpc):
                            slot = Int32(buf.toint()) + warp * (_MBAR_BYTES + self.nbytes)
                            bar_put = slot + 8
                            stage = slot + _MBAR_BYTES
                            if lane == 0:
                                _mbar_init(bar_put, 1)
                            cute.arch.sync_warp()
                            for t in cutlass.range(
                                by * _PUSH_WARPS + warp, n, nby * _PUSH_WARPS, unroll=1
                            ):
                                row = src + (Int64(source) * max_tokens + t) * src_stride
                                per_lane = self.nbytes // 16 // 32
                                loads = [
                                    (
                                        _ld_v4(row + (lane + 32 * k) * 32),
                                        _ld_v4(row + (lane + 32 * k) * 32 + 16),
                                    )
                                    for k in range(per_lane)
                                ]
                                for k in cutlass.range_constexpr(per_lane):
                                    a, b = loads[k]
                                    _st_shared_v4(
                                        stage + (lane + 32 * k) * 16,
                                        _bf16x4_to_e4m3x4(a[0], a[1]),
                                        _bf16x4_to_e4m3x4(a[2], a[3]),
                                        _bf16x4_to_e4m3x4(b[0], b[1]),
                                        _bf16x4_to_e4m3x4(b[2], b[3]),
                                    )
                                _fence_proxy_async_shared()
                                cute.arch.sync_warp()
                                if lane == 0:
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
                                cute.arch.sync_warp()
                        elif lane == 0:
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
            lpc: bool = False,
        ):
            assert top_k <= 32 and hidden % 16 == 0
            self.ep, self.top_k, self.hidden, self.lay, self.pdl = ep, top_k, hidden, dict(lay), pdl
            self.lpc = lpc

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
            s_own = smem.allocate_tensor(Int32, cute.make_layout((32,)), byte_alignment=16)
            s_n = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
            my_ws = ws_base + Int64(rank) * ws_stride
            inbox = my_ws + L["combine_recv"]
            wire = self.hidden if self.lpc else nbytes

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
                                    target = _ld_b64(base_addr) + wire
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
                        row_src = inbox + (Int64(tr) * max_tokens + di) * wire
                        own = Int32(0)
                        at = cute.arch.popc(valid & ((Int32(1) << lane) - 1))
                        s_src[at] = row_src
                        s_own[at] = own
                cute.arch.sync_threads()
                # Threads load slots warp 0 waited for: each fences after the barrier.
                _fence_fabric_acquire()

                count = s_n[0]
                dst = out + Int64(tok) * nbytes
                ns = (K + 7) // 8 * 8
                a = cute.make_rmem_tensor(cute.make_layout((8, ns)), cutlass.Float32)
                res = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Float32)
                raw = cute.make_rmem_tensor(cute.make_layout((2, ns)), Int32)
                for v in cutlass.range(tidx, nbytes // 16, _THREADS, unroll=1):
                    a.fill(0.0)
                    if cutlass.const_expr(self.lpc):
                        # Every source's chunk is loaded before any is widened, so the k loads overlap.
                        for s in cutlass.range_constexpr(ns):
                            if s < count:
                                pair = _ld_v2(s_src[s] + v * 8)
                                raw[0, s] = pair[0]
                                raw[1, s] = pair[1]
                    for s in cutlass.range_constexpr(ns):
                        if s < count:
                            if cutlass.const_expr(self.lpc):
                                for h in cutlass.range_constexpr(2):
                                    f = _e4m3x4_to_f32(raw[h, s])
                                    for q in cutlass.range_constexpr(4):
                                        a[4 * h + q, s] = f[q]
                            else:
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
                    _st_v4(dst + v * 16, *words)

            if tok == 0:
                if warp == 0:
                    flag = _ld_b32(my_ws + L["flag"])
                    if lane < self.ep:
                        _spin_eq(my_ws + L["combine_flags"] + lane * 4, flag, timeout_ns)

    class _LamportPush:
        """Combine push into the peers' -0 armed inboxes (see ``CuteA2A.combine_push``): grid (ep, blocks), one warp
        per row; a warp stages the converted row in shared memory and stores it to the peer with a TMA tensor store.
        CTA (rank, 0) also does the round's bookkeeping: publish / wait the push-start flags and advance the round
        flag."""

        def __init__(
            self,
            ep: int,
            hidden: int,
            lpc: bool,
            lay: Tuple,
            pdl: bool,
            rows: int,
            inbox_bytes: int,
        ):
            assert hidden % 512 == 0
            self.ep, self.hidden, self.lpc, self.lay, self.pdl = ep, hidden, lpc, dict(lay), pdl
            self.rows = rows
            self.wire = hidden if lpc else 2 * hidden
            assert self.wire % _TMA_CHUNK == 0 and inbox_bytes == ep * rows * self.wire

        @cute.jit
        def __call__(
            self,
            mY: cute.Tensor,
            ws_base: Int64,
            ws_stride: Int64,
            rank: Int32,
            max_tokens: Int32,
            blocks: Int32,
            timeout_ns: Int64,
            inbox_ptr: cute.Pointer,
            side_stride32: Int32,
            stream,
        ):
            # Every rank's two inboxes as (chunk byte, chunk, row, rank); rows of parity b start at b * ep * rows.
            chunks = self.wire // _TMA_CHUNK
            m_side = cute.make_tensor(
                inbox_ptr,
                cute.make_layout(
                    (_TMA_CHUNK, chunks, 2 * self.ep * self.rows, self.ep),
                    stride=(1, _TMA_CHUNK, self.wire, side_stride32),
                ),
            )
            tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                m_side,
                cute.make_layout((_TMA_CHUNK, chunks), stride=(1, _TMA_CHUNK)),
                (_TMA_CHUNK, chunks),
            )
            self.kernel(
                mY,
                tma_atom,
                tma_tensor,
                ws_base,
                ws_stride,
                rank,
                max_tokens,
                timeout_ns,
            ).launch(
                grid=[self.ep, blocks, 1],
                block=[_PUSH_WARPS * 32, 1, 1],
                smem=_PUSH_WARPS * self.wire + 128,
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mY: cute.Tensor,
            tma_atom: cute.CopyAtom,
            mSide: cute.Tensor,
            ws_base: Int64,
            ws_stride: Int64,
            rank: Int32,
            max_tokens: Int32,
            timeout_ns: Int64,
        ):
            L = self.lay
            src = mY.iterator.toint()
            tidx, _, _ = cute.arch.thread_idx()
            source, by, _ = cute.arch.block_idx()
            _, nby, _ = cute.arch.grid_dim()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            lane = cute.arch.lane_idx()
            my_ws = ws_base + Int64(rank) * ws_stride
            nbytes = self.hidden * 2
            smem = SmemAllocator()
            chunks = self.wire // _TMA_CHUNK
            sRing = smem.allocate_tensor(  # codespell:ignore
                cutlass.Uint8,
                cute.make_layout(
                    (_TMA_CHUNK, chunks, _PUSH_WARPS), stride=(1, _TMA_CHUNK, self.wire)
                ),
                byte_alignment=128,
            )
            stage = Int32(sRing.iterator.toint()) + warp * self.wire  # codespell:ignore
            gS = cute.local_tile(mSide, (_TMA_CHUNK, chunks), (0, 0, None, None))
            tS, tG = cpasync.tma_partition(
                tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sRing, 0, 2),  # codespell:ignore
                cute.group_modes(gS, 0, 2),
            )
            if warp == 0:
                cpasync.prefetch_descriptor(tma_atom)

            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
            # The round CTA may already have advanced the flag (d -> d + 1, d odd): both give the same values.
            f = _ld_b32(my_ws + L["flag"])
            value = ((f - 1) | 1) + 1
            parity = ((f - 1) >> 1) & 1
            if by == 0:
                if tidx == 0:
                    _st_sys_b32(
                        ws_base + Int64(source) * ws_stride + L["combine_flags"] + rank * 4, value
                    )
            if source != rank:
                cnt_addr = my_ws + L["recv_counters"] + (parity * self.ep + source) * 4
                n = _spin_ne(cnt_addr, Int32(-1), timeout_ns)
                for t in cutlass.range(by * _PUSH_WARPS + warp, n, nby * _PUSH_WARPS, unroll=1):
                    row = src + (Int64(source) * max_tokens + t) * nbytes
                    # The stage is read by the warp's previous store; whichever lane issued it waits.
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    cute.arch.sync_warp()
                    # All of the lane's loads are issued before its remote stores (one round trip per row).
                    per_lane = self.wire // 16 // 32
                    if cutlass.const_expr(self.lpc):
                        loads = [
                            (
                                _ld_v4(row + (lane + 32 * k) * 32),
                                _ld_v4(row + (lane + 32 * k) * 32 + 16),
                            )
                            for k in range(per_lane)
                        ]
                        for k in cutlass.range_constexpr(per_lane):
                            a, b = loads[k]
                            _st_shared_v4(
                                stage + (lane + 32 * k) * 16,
                                _canon(_bf16x4_to_e4m3x4(a[0], a[1]), True),
                                _canon(_bf16x4_to_e4m3x4(a[2], a[3]), True),
                                _canon(_bf16x4_to_e4m3x4(b[0], b[1]), True),
                                _canon(_bf16x4_to_e4m3x4(b[2], b[3]), True),
                            )
                    else:
                        loads = [_ld_v4(row + (lane + 32 * k) * 16) for k in range(per_lane)]
                        for k in cutlass.range_constexpr(per_lane):
                            w = loads[k]
                            _st_shared_v4(
                                stage + (lane + 32 * k) * 16,
                                _canon(w[0], False),
                                _canon(w[1], False),
                                _canon(w[2], False),
                                _canon(w[3], False),
                            )
                    _fence_proxy_async_shared()
                    cute.arch.sync_warp()
                    # The TMA store elects its issuing lane itself.
                    cute.copy(
                        tma_atom,
                        tS[(None, warp)],
                        tG[(None, parity * (self.ep * self.rows) + rank * self.rows + t, source)],
                    )
                    cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_threads()
            if source == rank:
                if by == 0:
                    if warp == 0:
                        flag_addr = my_ws + L["combine_flags"] + lane * 4
                        if lane < self.ep:
                            _spin_eq(flag_addr, value, timeout_ns)
                        cute.arch.sync_warp()
                        if lane == 0:
                            _st_b32(my_ws + L["flag"], value)

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
                stream,
                options=opt,
            )
        if kind == "lamport_push":
            return cute.compile(
                _LamportPush(*cfg),
                _fake(cutlass.BFloat16),
                *([i64] * 2),
                i32,
                i32,
                Int32(1),
                i64,
                make_ptr(cutlass.Uint8, 1 << 20, cute.AddressSpace.gmem, assumed_align=128),
                i32,
                stream,
                options=opt,
            )
        raise ValueError(kind)


class LamportInbox(NamedTuple):
    """Consumer view of the Lamport combine inbox (``CuteA2A.combine_push``). Rank r's side buffer starts at
    ``side_base + r * side_stride``; inbox ``b`` (b = round parity) at ``_SIDE_INBOX + b * inbox_bytes``; the row of
    (source rank s, slot d) at ``(s * rows + d) * wire``; ``wire`` bytes per row, e4m3 (``lpc``) or bf16. The inbox
    is armed with the wire dtype's -0."""

    side_base: int
    side_stride: int
    rows: int
    wire: int
    inbox_bytes: int
    lpc: bool


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class CuteA2A:
    """CuTe DSL dispatch / combine on a stock ``NVLinkOneSided`` that uses CFT counted writes.

    ``dispatch(hidden_states, hidden_states_sf, token_selected_slots, token_final_scales, all_rank_num_tokens)``
    returns the stock tuple: per payload a ``[ep * runtime_max_tokens, width]`` view of this rank's receive region
    (rows ``src * runtime_max_tokens + j``, ``j < count[src]`` valid; expert ids of the other rows are -1).
    ``combine(final_hidden_states)`` takes the bf16 expert output ``[ep * runtime_max_tokens, hidden]`` (also the
    in-workspace view from ``get_combine_payload_tensor_in_workspace``) and returns ``[local_tokens, hidden]``.
    Both keep the stock communicator's dispatch state, so its own dispatch / combine can take either phase. With a
    low-precision combine communicator the rows travel as e4m3 (the stock saturating cast of each bf16 element) and
    the sum is the stock one. ``combine_push`` (needs ``lamport_rows``) pushes into the peers' -0 armed Lamport
    inboxes instead; see ``LamportInbox``.
    """

    def __init__(self, comm, le_base: Optional[int] = None, lamport_rows: int = 0):
        if not _CUTE_AVAILABLE:
            raise ImportError("CuTe DSL is not available")
        if not comm.can_use_cft_counted_writes:
            raise ValueError("CuteA2A needs the CFT workspace of NVLinkOneSided")
        if getattr(comm, "_rank_mask_enabled", False) or comm.enable_eplb:
            raise ValueError("CuteA2A supports no EPLB statistics or rank masks")
        self.comm = comm
        self.ep, self.rank = comm.ep_size, comm.ep_rank
        self.lpc = bool(comm.use_low_precision_combine)
        self.lay = _Layout(comm)
        self.le_base = find_le_base(self.ep) if le_base is None else le_base
        self.side: Optional[torch.Tensor] = None
        self.lamport_rows = lamport_rows
        if lamport_rows > 0:
            self._alloc_side(lamport_rows)

    def _alloc_side(self, rows: int) -> None:
        """One symmetric allocation per rank: header, then two inboxes armed with the wire dtype's -0."""
        from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

        hidden = int(self.comm._workspace_state["hidden_size"])
        self.hidden = hidden
        self.wire = hidden if self.lpc else 2 * hidden
        self.inbox_bytes = _align(self.ep * rows * self.wire, _CACHELINE)
        self._side_mem = MnnvlMemory(self.comm.mapping, _SIDE_INBOX + 2 * self.inbox_bytes)
        self.side = self._side_mem.as_torch_strided_tensor(torch.uint8)
        mine = self.side[self.rank]
        mine[:_SIDE_INBOX].zero_()
        boxes = mine[_SIDE_INBOX : _SIDE_INBOX + 2 * self.inbox_bytes]
        if self.lpc:
            boxes.fill_(0x80)
        else:
            boxes.view(torch.int16).fill_(-32768)
        torch.cuda.synchronize()
        MnnvlMemory.get_comm(self.comm.mapping).barrier()

    @property
    def workspace(self) -> torch.Tensor:
        return self.comm.workspace

    def _ws_head(self) -> torch.Tensor:
        # The prepare kernel reads only the row's base pointer; a whole row sized for a serving max_num_tokens can
        # exceed 2 GiB, which fails the launcher's int32 shape check.
        return self.workspace[self.rank, :_CACHELINE]

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
        **kwargs,
    ):
        comm = self.comm
        if comm._dispatch_state.get("phase") == "dispatched":
            raise RuntimeError("dispatch called twice without an intervening combine")
        if kwargs.get("eplb_local_stats") is not None or kwargs.get("active_rank_mask") is not None:
            raise ValueError("CuteA2A: EPLB statistics and rank masks are not supported")
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
        _compile("prepare_dispatch", self.ep, lay, pdl)(self._ws_head())
        fn = _compile(
            "dispatch",
            self.ep,
            top_k,
            comm.num_experts,
            tuple(nbytes),
            ids_payload,
            lay,
            pdl,
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

    def _combine_args(self, final_hidden_states: torch.Tensor):
        st = self.comm._dispatch_state
        if st.get("phase") != "dispatched":
            raise RuntimeError("combine called before a successful dispatch")
        n, runtime_max = st["local_num_tokens"], st["runtime_max_tokens_per_rank"]
        y = final_hidden_states.reshape(self.ep * runtime_max, -1)
        if y.dtype != torch.bfloat16 or not y.is_contiguous() or y.data_ptr() % 16:
            raise ValueError("CuteA2A combine takes a contiguous bf16 payload")
        return y, n, runtime_max

    def combine(
        self,
        final_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        y, n, runtime_max = self._combine_args(final_hidden_states)
        hidden = y.shape[1]
        lpc = self.lpc
        nbytes = hidden * 2
        wire = hidden if lpc else nbytes
        if self.ep * runtime_max * wire > self.lay.combine_recv_size:
            raise ValueError("combine payload exceeds the receive region")
        lay = self.lay.key()
        pdl = _pdl()
        ws_base, ws_stride, my_ws = self._ws()
        out = torch.empty(n, hidden, dtype=torch.bfloat16, device=y.device)
        _compile("prepare_combine", self.ep, wire, lay, pdl, lpc)(
            y.view(-1), my_ws, nbytes, self.rank, runtime_max
        )
        rows_per_warp = 2
        blocks = min(
            32,
            max(
                1, (runtime_max + rows_per_warp * _PUSH_WARPS - 1) // (rows_per_warp * _PUSH_WARPS)
            ),
        )
        _compile(
            "push",
            self.ep,
            wire,
            lay,
            pdl,
            lpc,
        )(
            y.view(-1),
            ws_base,
            ws_stride,
            nbytes,
            self.rank,
            runtime_max,
            self.lay.max_tokens,
            self.le_base,
            blocks,
            _TIMEOUT_NS,
        )
        _compile(
            "reduce",
            self.ep,
            self.comm.top_k,
            hidden,
            lay,
            pdl,
            lpc,
        )(
            out.view(-1),
            ws_base,
            ws_stride,
            self.rank,
            n,
            runtime_max,
            self.lay.max_tokens,
            _TIMEOUT_NS,
        )
        self.comm.reset_state()
        return out

    def _lamport_blocks(self, runtime_max: int) -> int:
        # At most 4 push blocks per source rank: each warp then pushes more rows, and the push leaves SMs to the
        # shared-expert down launched beside it. Every rank must use the same value.
        return min(4, max(1, (runtime_max + _PUSH_WARPS - 1) // _PUSH_WARPS))

    def lamport_inbox(self) -> LamportInbox:
        """The consumer view of this rank's Lamport inbox for the current round."""
        if self.side is None:
            raise RuntimeError("the Lamport inbox needs CuteA2A(..., lamport_rows=N)")
        return LamportInbox(
            self.side.data_ptr(),
            self.side.stride(0),
            self.lamport_rows,
            self.wire,
            self.inbox_bytes,
            self.lpc,
        )

    def combine_push(self, final_hidden_states: torch.Tensor) -> LamportInbox:
        """Push each received row back to its source rank's inbox: a warp per row loads the bf16 expert output (after
        the producer grid), casts it to the wire format, turns -0 into +0 and stores the row into ``inbox[parity]`` of
        the source with a TMA tensor store. A CTA exits once its reads of the payload are done: it neither waits for its
        remote stores to land nor publishes completion words, for a consumer that checks every row it reads. The round
        CTA waits for the peers' push-start flags. Own rows stay in the payload. The consumer (a fused kernel following
        ``LamportInbox``) must read this round's rows, re-arm what it read and ``griddepcontrol.wait`` before it
        exits."""
        y, n, runtime_max = self._combine_args(final_hidden_states)
        if self.side is None:
            raise RuntimeError("the Lamport inbox needs CuteA2A(..., lamport_rows=N)")
        if runtime_max > self.lamport_rows or y.shape[1] != self.hidden:
            raise ValueError(
                f"Lamport combine takes at most {self.lamport_rows} rows of {self.hidden}"
            )
        side_stride = self.side.stride(0)
        if side_stride >= 1 << 31:
            raise ValueError(
                f"Lamport TMA store: side stride {side_stride} exceeds the descriptor's int32"
            )
        ws_base, ws_stride, my_ws = self._ws()
        blocks = self._lamport_blocks(runtime_max)
        _compile(
            "lamport_push",
            self.ep,
            self.hidden,
            self.lpc,
            self.lay.key(),
            _pdl(),
            self.lamport_rows,
            self.inbox_bytes,
        )(
            y.view(-1),
            ws_base,
            ws_stride,
            self.rank,
            runtime_max,
            blocks,
            _TIMEOUT_NS,
            make_ptr(
                cutlass.Uint8,
                self.side.data_ptr() + _SIDE_INBOX,
                cute.AddressSpace.gmem,
                assumed_align=128,
            ),
            side_stride,
        )
        box = self.lamport_inbox()
        self.comm.reset_state()
        return box


def install(comm, lamport_rows: int = 0) -> Optional[CuteA2A]:
    """Route ``comm.dispatch`` / ``comm.combine`` of a CFT ``NVLinkOneSided`` through :class:`CuteA2A` for the
    batches stock runs with CFT, with at most ``moe_chain.MAX_TOKENS_PER_RANK`` tokens per rank and without EPLB
    statistics or rank masks; other calls keep the stock path. Strategy kwargs the scheduler passes to every strategy
    are ignored, as stock does. ``lamport_rows`` > 0 allocates the Lamport combine inbox (collective: every rank must
    pass the same value)."""
    from ..moe_chain import within_max_rows

    cute_a2a = CuteA2A(comm, lamport_rows=lamport_rows)
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
