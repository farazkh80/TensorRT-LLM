# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Counter-slab primitives of the MoE expert-path edges: a producer publishes a unit with ``release_add`` after a CTA
barrier that orders its threads' stores; a consumer spins with ``wait_at_least`` before reading the unit's data."""

import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(result_type, operands, text, constraints, loc, ip):
    return llvm.inline_asm(
        result_type,
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
def ld_acquire(addr: cutlass.Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def release_add(addr: cutlass.Int64, val: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def nanosleep(ns: Int32, *, loc=None, ip=None) -> None:
    _asm(None, [Int32(ns).ir_value(loc=loc, ip=ip)], "nanosleep.u32 $0;", "r", loc, ip)


@dsl_user_op
def atom_add_acq_rel(addr: cutlass.Int64, val: Int32, *, loc=None, ip=None) -> Int32:
    """``atom.acq_rel.gpu.add``: returns the old value."""
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
def ld_cg_v4(addr: cutlass.Int64, *, loc=None, ip=None):
    """Four int32 at a 16-byte aligned address, read at L2 (no L1 allocation)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_cg_b32(addr: cutlass.Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def bits_to_f32(bits: Int32, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(bits).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_relaxed_sys_b32(addr: cutlass.Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.sys.global.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_relaxed_gpu_b32(addr: cutlass.Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            loc,
            ip,
        )
    )


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None) -> None:
    _asm(None, [], "fence.proxy.async.global;", "", loc, ip)


@dsl_user_op
def ld_relaxed_sys_b64(addr: cutlass.Int64, *, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(
        _asm(
            T.i64(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.sys.global.b64 $0, [$1];",
            "=l,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_volatile_b32(addr: cutlass.Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_volatile_b64(addr: cutlass.Int64, *, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(
        _asm(
            T.i64(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.b64 $0, [$1];",
            "=l,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def st_b64(addr: cutlass.Int64, val: cutlass.Int64, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int64(val).ir_value(loc=loc, ip=ip)],
        "st.global.b64 [$0], $1;",
        "l,l",
        loc,
        ip,
    )


@dsl_user_op
def fence_fabric_acquire(*, loc=None, ip=None) -> None:
    """Makes NVLink counted writes that a preceding counter read observed visible to this thread's generic loads."""
    _asm(None, [], "fence.proxy.generic::fabric.alias.acquire.sys;", "", loc, ip)


@dsl_user_op
def match_any(val: Int32, *, loc=None, ip=None) -> Int32:
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


@cute.jit
def wait_ne_sys(addr: cutlass.Int64, sentinel: Int32) -> Int32:
    """Spin until the 32-bit word at ``addr`` (written over NVLink) differs from ``sentinel``; returns it."""
    seen = ld_relaxed_sys_b32(addr)
    while seen == sentinel:
        nanosleep(Int32(64))
        seen = ld_relaxed_sys_b32(addr)
    return seen


@cute.jit
def wait_ge64(addr: cutlass.Int64, target: cutlass.Int64):
    """Spin until the 64-bit byte counter at ``addr`` (written over NVLink) reaches ``target``."""
    seen = ld_relaxed_sys_b64(addr)
    while seen < target:
        nanosleep(Int32(64))
        seen = ld_relaxed_sys_b64(addr)


@cute.jit
def wait_at_least(addr: cutlass.Int64, expected: Int32):
    """Spin until the counter at ``addr`` reaches ``expected``; the last read is an acquire."""
    delay = Int32(32)
    seen = ld_acquire(addr)
    while seen < expected:
        nanosleep(delay)
        delay = cutlass.min(delay * 2, Int32(512))
        seen = ld_acquire(addr)


@cute.jit
def wait_at_least_short(addr: cutlass.Int64, expected: Int32):
    """``wait_at_least`` with the backoff capped at 128 ns, for a wait on the critical path."""
    delay = Int32(32)
    seen = ld_acquire(addr)
    while seen < expected:
        nanosleep(delay)
        delay = cutlass.min(delay * 2, Int32(128))
        seen = ld_acquire(addr)


@dsl_user_op
def globaltimer(*, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip))


@cute.jit
def spin_ns(ns: cutlass.Int64):
    """Busy-wait ``ns`` nanoseconds (hazard tests only)."""
    start = globaltimer()
    while globaltimer() - start < ns:
        nanosleep(Int32(1000))


# The FC1 -> FC2 counters cover the first EARLY_TILES token tiles only. The FC2 CTAs that start in FC1's tail take
# those tiles first; FC2 units of later tiles wait for the FC1 grid instead, which saves FC1 a fence per item.
EARLY_TILES = 8


DIAG_ENV = "TLLM_K3_MOE_FGS_DIAG"
DIAG_EDGES_ENV = "TLLM_K3_MOE_FGS_DIAG_EDGES"
DIAG_MODES = ("oracle", "ready")


def fgs_diag(edge: str) -> str:
    """Decision build of ``edge``'s consumer, diagnostic only (``TLLM_K3_MOE_FGS_DIAG``, applied to every enabled
    edge or to the comma list in ``TLLM_K3_MOE_FGS_DIAG_EDGES``): ``oracle`` launches the consumer as its FGS build
    does but skips its waits (wrong numerics: the upper bound of the edge); ``ready`` makes the FGS build wait for its
    producer grid first (launched as the FUSED-OFF build launches that consumer: in stream order, or with PDL plus a
    grid wait), so its waits find the input complete (the cost of the edge). Empty when off for this edge."""
    mode = os.environ.get(DIAG_ENV, "")
    if mode and mode not in DIAG_MODES:
        raise ValueError(f"{DIAG_ENV}={mode!r}; valid: {', '.join(DIAG_MODES)}")
    only = os.environ.get(DIAG_EDGES_ENV, "")
    if mode and only and edge not in only.split(","):
        return ""
    return mode
