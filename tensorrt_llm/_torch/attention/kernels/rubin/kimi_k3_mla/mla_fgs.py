# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-grained-sync (Lamport) building blocks shared by the Kimi K3 MLA kernels.

An FGS edge publishes through a slot that the framework fills with the sentinel (-0.0: bf16 0x8000, fp32
0x80000000) before the producer runs. A producer never stores the sentinel as data (positive-zero override,
``bf16x2_pzo`` / ``canon_bf16x2`` / ``canon_f32``) and triggers its dependents on entry. A consumer is launched with
PDL, skips ``griddepcontrol.wait`` and treats any element still equal to the sentinel as not written yet.

Consumer protocol for plain (non-TMA) loads, used by the elementwise kernels:

* The first load of a 16-B chunk is ``ld.global.cg`` (L2, no L1 allocation). Reloads are ``ld.relaxed.gpu`` and are
  never merged with or hoisted above an earlier load of the same address.
* Every element of a chunk is checked, so a producer that writes a chunk with narrower stores is still safe.
* The wait is warp-uniform: ``poll_chunks`` runs only when some lane of the warp holds a sentinel, and the whole
  warp loops (nanosleep backoff, reload of the pending chunks only) until no lane does. It must be called from
  warp-converged code; lanes without work pass chunks that hold no sentinel.
* Deadlock freedom comes from the PDL chain: every kernel of an FGS chain triggers on entry, so all CTAs of every
  upstream producer are resident or done before a consumer CTA can spin. ``spin_limit`` (diagnostic) traps after that
  many sweeps instead of spinning forever.

Stats (optional int32 tensor, 4 entries, accumulated with one atomic per warp that waited):
[0] reload sweeps, [1] warps that waited, [2] chunks reloaded, [3] unused.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

BF16_SENTINEL = 0x8000
F32_SENTINEL = -0x80000000
SPIN_NS_MIN = 32
SPIN_NS_MAX = 1024
STATS_LEN = 4


def _v4(asm, addr, side_effects, constraints, loc, ip):
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        asm,
        constraints,
        has_side_effects=side_effects,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_cg_v4(addr, *, loc=None, ip=None):
    """16-byte ``ld.global.cg`` as four int32 (first load of an FGS chunk)."""
    return _v4(
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];", addr, False, "=r,=r,=r,=r,l", loc, ip
    )


@dsl_user_op
def ld_relaxed_v4(addr, *, loc=None, ip=None):
    """16-byte ``ld.relaxed.gpu.global`` as four int32 (reload of an FGS chunk)."""
    return _v4(
        "ld.relaxed.gpu.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        addr,
        True,
        "=r,=r,=r,=r,l,~{memory}",
        loc,
        ip,
    )


@dsl_user_op
def st_global_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    """One 16-byte ``st.global.v4.b32``."""
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)]
        + [cutlass.Int32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


_PZO_SW = (
    "{ .reg .b16 l, h; .reg .pred p; cvt.rn.bf16x2.f32 $0, $1, $2; mov.b32 {l, h}, $0; "
    "setp.eq.b16 p, l, 0x8000; selp.b16 l, 0, l, p; "
    "setp.eq.b16 p, h, 0x8000; selp.b16 h, 0, h, p; mov.b32 $0, {l, h}; }"
)
_PZO_HW = "cvt.rn.pzo.bf16x2.f32 $0, $1, $2;"


@dsl_user_op
def bf16x2_pzo(lo, hi, hw, *, loc=None, ip=None):
    """Round two f32 to a packed bf16x2 (``lo`` in the low half) like ``cvt.rn``, with -0.0 -> +0.0."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            _PZO_HW if hw else _PZO_SW,
            "=r,f,f",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def canon_bf16x2(w, *, loc=None, ip=None):
    """A packed bf16x2 word with each -0.0 half replaced by +0.0; every other bit pattern is kept."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 l, h; .reg .pred p; mov.b32 {l, h}, $1; "
            "setp.eq.b16 p, l, 0x8000; selp.b16 l, 0, l, p; "
            "setp.eq.b16 p, h, 0x8000; selp.b16 h, 0, h, p; mov.b32 $0, {l, h}; }",
            "=r,r",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def canon_f32(x, *, loc=None, ip=None):
    """``x`` with -0.0 replaced by +0.0; every other bit pattern (NaN payloads included) is kept."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "{ .reg .pred p; .reg .b32 b; mov.b32 b, $1; setp.eq.b32 p, b, 0x80000000; "
            "selp.b32 b, 0, b, p; mov.b32 $0, b; }",
            "=f,f",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x2_has_sentinel(w, *, loc=None, ip=None):
    """1 when either bf16 half of ``w`` is -0.0, else 0 (int32)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 l, h; .reg .pred p, q; mov.b32 {l, h}, $1; "
            "setp.eq.b16 p, l, 0x8000; setp.eq.or.b16 q, h, 0x8000, p; selp.b32 $0, 1, 0, q; }",
            "=r,r",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def trap(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "trap;", "", has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip
    )


def words_pending(words, base, f32: bool):
    """Int32 1 when any element of the 4 int32 words ``words[base:base+4]`` is the sentinel."""
    acc = cutlass.Int32(0)
    for i in range(4):
        w = words[base + i]
        if f32:
            acc = acc | cutlass.Int32(w == cutlass.Int32(F32_SENTINEL))
        else:
            acc = acc | bf16x2_has_sentinel(w)
    return acc


@cute.jit
def load_chunks(words: cute.Tensor, addrs: cute.Tensor, n: cutlass.Constexpr, active: cutlass.Boolean):
    """First load of ``n`` 16-B chunks into the int32 rmem tensor ``words`` (4 per chunk); inactive lanes load
    nothing and hold zeros, so they never count as pending."""
    for c in cutlass.range_constexpr(4 * n):
        words[c] = cutlass.Int32(0)
    if active:
        for c in cutlass.range_constexpr(n):
            w = ld_cg_v4(addrs[c])
            for i in cutlass.range_constexpr(4):
                words[4 * c + i] = w[i]


@cute.jit
def poll_chunks(
    words: cute.Tensor,
    addrs: cute.Tensor,
    n: cutlass.Constexpr,
    f32: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    stats: cute.Tensor,
    with_stats: cutlass.Constexpr,
):
    """Warp-uniform wait until no lane's ``n`` chunks hold a sentinel; reloads only the pending chunks.

    ``words`` int32 rmem [4 * n] (already loaded), ``addrs`` int64 rmem [n]. Returns the number of sweeps.
    """
    pending = cutlass.Int32(0)
    for c in cutlass.range_constexpr(n):
        pending = pending | words_pending(words, 4 * c, f32)
    sweeps = cutlass.Int32(0)
    reloads = cutlass.Int32(0)
    if cute.arch.vote_any_sync(pending != cutlass.Int32(0)):
        sleep_ns = cutlass.Int32(SPIN_NS_MIN)
        again = cutlass.Boolean(True)
        while again:
            nanosleep(sleep_ns)
            sleep_ns = cutlass.min(sleep_ns * 2, cutlass.Int32(SPIN_NS_MAX))
            pending = cutlass.Int32(0)
            for c in cutlass.range_constexpr(n):
                if words_pending(words, 4 * c, f32) != cutlass.Int32(0):
                    w = ld_relaxed_v4(addrs[c])
                    for i in cutlass.range_constexpr(4):
                        words[4 * c + i] = w[i]
                    reloads = reloads + 1
                    pending = pending | words_pending(words, 4 * c, f32)
            sweeps = sweeps + 1
            if cutlass.const_expr(spin_limit > 0):
                if sweeps > cutlass.Int32(spin_limit):
                    trap()
            again = cute.arch.vote_any_sync(pending != cutlass.Int32(0))
        if cutlass.const_expr(with_stats):
            reloads = cute.arch.warp_reduction_sum(reloads)
            if cute.arch.lane_idx() == 0:
                cute.arch.atomic_add(stats.iterator, sweeps)
                cute.arch.atomic_add(stats.iterator + 1, cutlass.Int32(1))
                cute.arch.atomic_add(stats.iterator + 2, reloads)
    return sweeps
