# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Counter-slab primitives of the MoE expert-path edges: a producer publishes a unit with ``release_add`` after a CTA
barrier that orders its threads' stores; a consumer spins with ``wait_at_least`` before reading the unit's data."""

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
