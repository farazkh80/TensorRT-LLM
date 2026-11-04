# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
#
# Derived from FlashInfer v0.6.18 (39c024e2b5)
# flashinfer/gemm/kernels/dense_bf16_gemm_direct.py, Apache-2.0.
"""Register-prefetch BF16 GEMM for low-M decode shapes, with optional fine-grained-sync roles (SM107).

Each CTA computes ``outputs_per_block`` output columns for ``rows_per_block`` rows over the whole K: every thread
prefetches its slice of the weight rows into registers, streams the activation rows through registers, and the
partial dot products are reduced across the warp and then across warps through shared memory. Activations are read
with plain 16 B vector loads (no TMA), so the consumer check is done in software.

Compile-time consumer roles:

- ``check_valid``: an activation element counts as written once it is not the bf16 -0.0 sentinel. With
  ``check_elements`` (default) every element of a 16 B chunk is checked, so a chunk its producer wrote in pieces
  (several 2 B stores, e.g. the ``pzo`` producer below) is complete when accepted; without it only element 0, the
  rule of the TMA ``report_valid`` consumers (``heads`` checks element 0 only). The first loads are the default build's
  (L1-allocating) loads; a warp that sees an unwritten chunk sleeps (exponential backoff) and reloads with
  ``ld.relaxed.gpu``, which an L1 line holding the sentinel cannot serve. Within the kernel's lifetime an
  activation chunk only goes from the sentinel to its final value, so a cached line holds one or the other. Waits
  are warp-uniform; there is no fallback (an input that is never published hangs the kernel). ``check_mode``:

  - ``rows``: every row's chunks are loaded (the default build's loads) and checked before the FMAs; reloaded
    chunks are used directly. One round trip; holds all rows' activations in registers.
  - ``row``: each row's chunks are checked as they are loaded; reloaded chunks are used directly.
  - ``heads``: after the weight loads are issued, element 0 of every chunk of the CTA's rows is loaded (4 B,
    L1-allocating) until written, then the default body runs; its activation loads hit the lines the check
    brought in. After a reload, ``fence.acq_rel.gpu`` keeps those loads from stale lines.
  - ``heads_before_b``: before the weight loads, every chunk of the CTA's rows is loaded (16 B with
    ``check_elements``, else 4 B heads) and checked, then the default body runs. With several rows, ``row`` splits
    the rows' FMAs into separate blocks, so the weight registers stay live instead of being consumed by interleaved
    rows (spills at 4 outputs x 14 K tiles); this mode keeps the default body's register use there.
- ``relaxed_first`` (diagnostic, ``row``): the first activation loads are ``ld.relaxed.gpu`` as well.
- ``spin_limit`` (diagnostic): after that many reload rounds a warp (or a chunk-head CTA) stops waiting, uses what it
  holds and counts a give-up, so a test reports a stuck wait instead of hanging.
- ``pdl_wait_activation``: ``griddepcontrol.wait`` before the weight and activation loads (the default build's
  only wait). Without it the weight rows stream from launch.

Compile-time producer roles (the output is an FGS consumer's input):

- ``pzo``: the output is narrowed with ``cvt.rn.pzo`` (-0.0 stored as +0.0).
- ``early_trigger``: ``griddepcontrol.launch_dependents`` at kernel entry instead of after the stores.

A 16 B output chunk spans ``G = 8 / outputs_per_block`` CTAs, each storing 2 B per element, so a consumer that
checks element 0 only (TMA ``report_valid``, KdaDecodeT1, the mixes) could accept it before the rest has landed.
``pzo`` alone suits a consumer that checks every element (``check_elements``); for the others:

- ``group8``: each CTA computes a whole chunk, 8 columns, as ``G`` consecutive default-build column groups (same
  per-column summation order, bit-identical), and one thread per row stores the chunk with one 16 B store. No
  cross-CTA wait; the grid is ``G`` times smaller. Needs ``pzo`` and N % 8 == 0.
- ``head_last``: CTAs keep their columns, but are mapped so that the chunk head is computed by the group's highest
  CTA index (dispatched last); that CTA stores its other elements, waits until the chunk's remaining elements are
  no longer -0.0 (``ld.relaxed.gpu``; the output must be armed with -0.0 for the order to mean anything), then
  stores element 0; the store depends on the loop exit, so it is performed after the rest of the chunk was read
  back from L2. Needs ``pzo`` (a published -0.0 would never end the wait) and N % 8 == 0.
- ``gather``: the default grid and per-CTA work, launched as clusters of ``G`` CTAs along N (one cluster per chunk
  column). Each CTA's thread 0 stores its results into the cluster leader's shared memory, every thread arrives on
  the cluster barrier (release) and the other CTAs exit; the leader waits on the barrier (acquire) and stores each
  row's chunk with one 16 B store. Exited threads do not hold the barrier, and the leader, whose shared memory the
  others write, cannot exit before all of them arrived. Needs ``pzo`` and N % 8 == 0.

The kernel never writes sentinels; the caller owns sentinel initialization of the activation buffer.
"""

from __future__ import annotations

import dataclasses
import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch as _torch
from cutlass import Int32, Int64, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

_VECTOR_WIDTH = 8
_SUPPORTED_BLOCK_SIZES = (32, 64, 96, 128, 192, 256, 384)
_SUPPORTED_OUTPUTS_PER_BLOCK = (1, 2, 4)
_MAX_M = 32
_COMPILE_OPTIONS = "--ptxas-options -maxrregcount=64"

#: BF16 elements per 16 B output chunk.
_CHUNK = 8
_BF16_NEG_ZERO = 0x8000
_SPIN_NS_MIN = 32
_SPIN_SHIFT_MAX = 6

#: 32-bit weight registers per thread above which a multi-row consumer validates before loading weights.
_HEADS_BEFORE_B_REGS = 128
_CHECK_MODES = ("rows", "row", "heads", "heads_before_b")

#: CTAs per SM of the default build (its register count, CuTe DSL 4.9, sm_107a) at the Kimi K3 decode tactics:
#: (K, (block_size, outputs_per_block, rows_per_block)) -> CTAs. The FGS roles cost a few registers, which at
#: these counts drops a CTA per SM; FGS builds of a multi-wave grid request the default build's occupancy.
_DEFAULT_CTAS_PER_SM = {
    (7168, (128, 1, 1)): 8,
    (7168, (128, 1, 2)): 5,
    (7168, (128, 1, 4)): 3,
    (7168, (128, 2, 1)): 5,
    (7168, (64, 4, 2)): 6,
    (12288, (256, 2, 1)): 3,
}

#: Slots of the retry counter: activation reload rounds summed over warps, waits given up (``spin_limit``).
RETRY_COUNTER_SLOTS = 2


@dataclasses.dataclass(frozen=True)
class DirectTactic:
    """One direct-kernel specialization."""

    block_size: int
    outputs_per_block: int
    rows_per_block: int


def _default_rows_per_block(m: int) -> int:
    if m <= 8:
        return m
    return next(rows for rows in (8, 4, 2, 1) if m % rows == 0)


def validate_tactic(tactic: DirectTactic, m: int, n: int, k: int) -> None:
    """Reject a direct tactic that cannot serve ``(m, n, k)``."""
    if tactic.block_size not in _SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"unsupported block_size={tactic.block_size}")
    if tactic.outputs_per_block not in _SUPPORTED_OUTPUTS_PER_BLOCK:
        raise ValueError(f"unsupported outputs_per_block={tactic.outputs_per_block}")
    if not 1 <= m <= _MAX_M:
        raise ValueError(f"direct GEMM requires 1 <= M <= {_MAX_M}, got {m}")
    if not 1 <= tactic.rows_per_block <= m or m % tactic.rows_per_block:
        raise ValueError(f"rows_per_block={tactic.rows_per_block} must divide M={m}")
    if n <= 0 or n % tactic.outputs_per_block:
        raise ValueError(f"N={n} must be divisible by outputs_per_block={tactic.outputs_per_block}")
    k_tile = tactic.block_size * _VECTOR_WIDTH
    if k <= 0 or k % k_tile:
        raise ValueError(f"K={k} must be divisible by {k_tile}")


def default_tactic(m: int, n: int, k: int) -> DirectTactic:
    """Choose the measured register-prefetch fallback tactic."""
    block_size = next(
        (block for block in (256, 192, 128, 96, 64, 32) if k % (block * _VECTOR_WIDTH) == 0),
        None,
    )
    if block_size is None:
        raise ValueError("direct GEMM requires a supported 16-byte K tiling")
    outputs_per_block = next(outputs for outputs in (2, 1) if n % outputs == 0)
    tactic = DirectTactic(block_size, outputs_per_block, _default_rows_per_block(m))
    validate_tactic(tactic, m, n, k)
    return tactic


@dsl_user_op
def _ld_relaxed_v4(addr, offset: int, reload, *, loc=None, ip=None):
    """16 B ``ld.relaxed.gpu.global`` at ``addr + offset`` bytes as four int32; a ``reload`` is never merged with or
    hoisted above an earlier load of the address."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        f"ld.relaxed.gpu.global.v4.b32 {{$0, $1, $2, $3}}, [$4+{offset}];",
        "=r,=r,=r,=r,l,~{memory}" if reload else "=r,=r,=r,=r,l",
        has_side_effects=reload,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ld_head(addr, offset: int, relaxed: bool, *, loc=None, ip=None):
    """4 B load of a chunk head, L1-allocating or ``ld.relaxed.gpu``; never reordered with other memory accesses."""
    op = "ld.relaxed.gpu.global.b32" if relaxed else "ld.global.b32"
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            f"{op} $0, [$1+{offset}];",
            "=r,l,~{memory}",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_chunk(addr, offset: int, relaxed: bool, *, loc=None, ip=None):
    """16 B chunk load, L1-allocating or ``ld.relaxed.gpu``; never reordered with other memory accesses."""
    op = "ld.relaxed.gpu.global.v4.b32" if relaxed else "ld.global.v4.b32"
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        f"{op} {{$0, $1, $2, $3}}, [$4+{offset}];",
        "=r,=r,=r,=r,l,~{memory}",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _st_global_v4(addr, words, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)] + [Int32(w).ir_value(loc=loc, ip=ip) for w in words],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_cluster_b16(smem_addr, peer, bits, *, loc=None, ip=None):
    """Store the low 16 bits of ``bits`` at this CTA's shared address ``smem_addr`` of cluster rank ``peer``."""
    llvm.inline_asm(
        None,
        [Int32(v).ir_value(loc=loc, ip=ip) for v in (smem_addr, peer, bits)],
        "{ .reg .u32 ra; .reg .b16 lo, hi; mapa.shared::cluster.u32 ra, $0, $1; mov.b32 {lo, hi}, $2; "
        "st.shared::cluster.b16 [ra], lo; }",
        "r,r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _copy_shared_to_global_v4(gaddr, saddr, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(gaddr).ir_value(loc=loc, ip=ip), Int32(saddr).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 r<4>; ld.shared.v4.b32 {r0, r1, r2, r3}, [$1]; "
        "st.global.v4.b32 [$0], {r0, r1, r2, r3}; }",
        "l,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _min_s16x2(acc, words, *, loc=None, ip=None):
    """Lane-wise signed 16-bit minimum of ``acc`` and the four words of a chunk."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(v).ir_value(loc=loc, ip=ip) for v in (acc, *words)],
            "{ .reg .b32 t, u; min.s16x2 t, $1, $2; min.s16x2 u, $3, $4; min.s16x2 t, t, u; "
            "min.s16x2 $0, t, $5; }",
            "=r,r,r,r,r,r",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _has_sentinel(lanes_min):
    """Whether a lane-wise s16 minimum holds the bf16 -0.0 sentinel: 0x8000 is the smallest s16 value, so the
    minimum over a set of elements is 0x8000 exactly when one of them is."""
    return ((lanes_min & 0xFFFF) == _BF16_NEG_ZERO) | (
        ((lanes_min >> 16) & 0xFFFF) == _BF16_NEG_ZERO
    )


#: Lane-wise s16 minimum start value (0x7FFF7FFF).
_LANES_MAX = 0x7FFF7FFF


def _chunk_bad(acc, words, elements: bool):
    """Fold a chunk into the running check: the lane minimum with ``elements``, else whether element 0 is the
    sentinel (``acc`` is then a Boolean)."""
    if elements:
        return _min_s16x2(acc, words)
    return acc | ((words[0] & 0xFFFF) == _BF16_NEG_ZERO)


@dsl_user_op
def _nanosleep(ns, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _backoff(rounds):
    """Sleep before reload round ``rounds`` + 1: ``_SPIN_NS_MIN`` doubled per round, capped at 2 us."""
    _nanosleep(Int32(_SPIN_NS_MIN) << cute.min(rounds, Int32(_SPIN_SHIFT_MAX)))


@dsl_user_op
def _fma_f32_bf16(a, b, c, *, loc=None, ip=None):
    """``a * b + c`` with bf16 ``a`` / ``b`` and an f32 accumulator, the default build's FMA."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                llvm.bitcast(T.i16(), cutlass.BFloat16(a).ir_value(loc=loc, ip=ip), loc=loc, ip=ip),
                llvm.bitcast(T.i16(), cutlass.BFloat16(b).ir_value(loc=loc, ip=ip), loc=loc, ip=ip),
                cutlass.Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32.bf16 $0, $1, $2, $3;",
            "=f,h,h,f",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_bits(lo, hi, pzo: bool, *, loc=None, ip=None):
    """``cvt.rn`` of two f32 to a packed bf16x2 (``lo`` in the low half; ``.pzo``: -0.0 -> +0.0)."""
    cvt = "cvt.rn.pzo.bf16x2.f32" if pzo else "cvt.rn.bf16x2.f32"
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            f"{cvt} $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16_bits(x, pzo: bool, *, loc=None, ip=None):
    """``cvt.rn`` of one f32 to bf16 (``.pzo``: -0.0 -> +0.0), returned in the low 16 bits."""
    cvt = "cvt.rn.pzo.bf16x2.f32" if pzo else "cvt.rn.bf16x2.f32"
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            f"{cvt} $0, $1, $1;",
            "=r,f",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class DirectFgsGemmKernel:
    """K-specialized direct GEMM with whole-mainloop vector prefetch and optional FGS roles."""

    def __init__(
        self,
        *,
        element_type,
        num_rows: int,
        k_extent: int,
        tactic: DirectTactic,
        use_pdl: bool,
        check_valid: bool = False,
        relaxed_first: bool = False,
        check_mode: str = "row",
        check_elements: bool = True,
        pdl_wait_activation: bool = True,
        pzo: bool = False,
        early_trigger: bool = False,
        head_last: bool = False,
        group8: bool = False,
        gather: bool = False,
        min_ctas_per_sm: int = 1,
        spin_limit: int = 0,
    ) -> None:
        validate_tactic(tactic, num_rows, tactic.outputs_per_block, k_extent)
        if check_mode not in _CHECK_MODES:
            raise ValueError(f"check_mode must be one of {_CHECK_MODES}, got {check_mode!r}")
        if relaxed_first and (not check_valid or check_mode != "row"):
            raise ValueError("relaxed_first applies to the row check only")
        if (head_last or group8 or gather) and not pzo:
            raise ValueError("head_last, group8 and gather need pzo")
        if head_last + group8 + gather > 1:
            raise ValueError("head_last, group8 and gather are alternatives")
        self.element_type = element_type
        self.num_rows = num_rows
        self.rows_per_block = tactic.rows_per_block
        self.k_extent = k_extent
        self.block_size = tactic.block_size
        self.outputs_per_block = tactic.outputs_per_block
        self.vector_width = _VECTOR_WIDTH
        self.use_pdl = use_pdl
        self.num_warps = tactic.block_size // cute.arch.WARP_SIZE
        self.num_k_tiles = k_extent // (tactic.block_size * _VECTOR_WIDTH)
        # Byte distance between a thread's consecutive activation chunks of one row (K is contiguous).
        self.k_tile_bytes = tactic.block_size * _VECTOR_WIDTH * element_type.width // 8
        self.check_valid = check_valid
        self.relaxed_first = relaxed_first
        self.check_mode = check_mode
        self.check_elements = check_elements
        self.pdl_wait_activation = pdl_wait_activation
        self.pzo = pzo
        self.early_trigger = early_trigger and use_pdl
        self.head_last = head_last
        self.group8 = group8
        self.gather = gather
        self.group = _CHUNK // tactic.outputs_per_block
        self.min_ctas_per_sm = min_ctas_per_sm
        self.spin_limit = spin_limit

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        retry_count: cute.Tensor,
        stream: _cuda.CUstream,
    ) -> None:
        n = cute.size(gB, mode=[0])
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        if const_expr(self.group8):
            self.kernel_group8(gA, gB, gC, retry_count, copy_a, copy_b).launch(
                grid=[cute.ceil_div(n, _CHUNK), self.num_rows // self.rows_per_block, 1],
                block=[self.block_size, 1, 1],
                smem=self.group * self.rows_per_block * self.outputs_per_block * self.num_warps * 4,
                stream=stream,
                use_pdl=self.use_pdl,
                min_blocks_per_mp=self.min_ctas_per_sm,
            )
        else:
            smem = self.rows_per_block * self.outputs_per_block * self.num_warps * 4
            if const_expr(self.gather):
                smem = (smem + 15) // 16 * 16 + self.rows_per_block * _CHUNK * 2
            self.kernel(gA, gB, gC, retry_count, copy_a, copy_b).launch(
                grid=[
                    cute.ceil_div(n, self.outputs_per_block),
                    self.num_rows // self.rows_per_block,
                    1,
                ],
                block=[self.block_size, 1, 1],
                cluster=[self.group, 1, 1] if const_expr(self.gather) else None,
                smem=smem,
                stream=stream,
                use_pdl=self.use_pdl,
                min_blocks_per_mp=self.min_ctas_per_sm,
            )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        retry_count: cute.Tensor,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        if const_expr(self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, block_m, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        num_rows: cutlass.Constexpr = self.rows_per_block
        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        vector_width: cutlass.Constexpr = self.vector_width
        block_size: cutlass.Constexpr = self.block_size
        num_warps: cutlass.Constexpr = self.num_warps
        num_k_tiles: cutlass.Constexpr = self.num_k_tiles

        acc = cute.make_rmem_tensor(
            cute.make_layout((num_rows, outputs_per_block), stride=(outputs_per_block, 1)),
            cutlass.Float32,
        )
        acc.fill(0.0)

        if const_expr(self.use_pdl and self.pdl_wait_activation):
            cute.arch.griddepcontrol_wait()

        n_base = block_idx * outputs_per_block
        if const_expr(self.head_last):
            rank = block_idx % self.group
            n_base = (block_idx - rank + (self.group - 1 - rank)) * outputs_per_block
        m_base = block_m * num_rows
        gA_vec = cute.logical_divide(gA, (None, vector_width))
        gB_vec = cute.logical_divide(gB, (None, vector_width))
        tA_all = cute.logical_divide(gA_vec, (None, (None, block_size)))
        tB_all = cute.logical_divide(gB_vec, (None, (None, block_size)))
        tA = tA_all[None, (None, (tidx, None))]
        if const_expr(self.check_valid and self.check_mode == "heads_before_b"):
            self._wait_heads_written(tA, m_base, retry_count)

        b_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (outputs_per_block, num_k_tiles, vector_width),
                stride=(num_k_tiles * vector_width, vector_width, 1),
            ),
            self.element_type,
        )
        for ni in cutlass.range_constexpr(outputs_per_block):
            tB = tB_all[n_base + ni, (None, (tidx, None))]
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                cute.copy(copy_b, tB[None, k_tile], b_regs[ni, k_tile, None])
        if const_expr(self.check_valid and self.check_mode == "heads"):
            self._wait_heads_written(tA, m_base, retry_count)

        rows_check = self.check_valid and self.check_mode == "rows"
        if const_expr(rows_check):
            a_rows = cute.make_rmem_tensor(
                cute.make_layout(
                    (num_rows, num_k_tiles, vector_width),
                    stride=(num_k_tiles * vector_width, vector_width, 1),
                ),
                self.element_type,
            )
            self._load_rows_checked(copy_a, tA, m_base, a_rows, retry_count)
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_k_tiles, vector_width), stride=(vector_width, 1)),
            self.element_type,
        )
        per_row_check = self.check_valid and self.check_mode == "row"
        for mi in cutlass.range_constexpr(num_rows):
            if const_expr(per_row_check):
                self._load_a_checked(copy_a, tA, m_base + mi, a_regs, retry_count)
            elif const_expr(rows_check):
                pass
            else:
                for k_tile in cutlass.range_constexpr(num_k_tiles):
                    cute.copy(
                        copy_a,
                        tA[m_base + mi, None, k_tile],
                        a_regs[k_tile, None],
                    )
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                for vi in cutlass.range_constexpr(vector_width):
                    if const_expr(per_row_check):
                        # The reload loops split rows into basic blocks; converting B to f32 once would then be
                        # shared across rows and keep it live in f32 registers instead of fused into each FMA.
                        for ni in cutlass.range_constexpr(outputs_per_block):
                            acc[mi, ni] = _fma_f32_bf16(
                                a_regs[k_tile, vi], b_regs[ni, k_tile, vi], acc[mi, ni]
                            )
                    else:
                        if const_expr(rows_check):
                            a_value = a_rows[mi, k_tile, vi].to(cutlass.Float32)
                        else:
                            a_value = a_regs[k_tile, vi].to(cutlass.Float32)
                        for ni in cutlass.range_constexpr(outputs_per_block):
                            acc[mi, ni] = acc[mi, ni] + a_value * b_regs[ni, k_tile, vi].to(
                                cutlass.Float32
                            )

        for mi in cutlass.range_constexpr(num_rows):
            for ni in cutlass.range_constexpr(outputs_per_block):
                acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])

        smem_layout = cute.make_layout(
            (num_rows, outputs_per_block, num_warps),
            stride=(outputs_per_block * num_warps, num_warps, 1),
        )
        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(cutlass.Float32, smem_layout, byte_alignment=16)
        if const_expr(self.gather):
            # Row mi of the leader's buffer is output chunk (m_base + mi, the leader's n_base .. + 8).
            chunk_rows = smem.allocate_tensor(
                cutlass.Int16,
                cute.make_layout((num_rows, _CHUNK), stride=(_CHUNK, 1)),
                byte_alignment=16,
            )
            rank = block_idx % self.group
        with cute.arch.elect_one():
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    partials[mi, ni, warp_idx] = acc[mi, ni]

        cute.arch.sync_threads()
        if tidx == 0:
            if const_expr(self.head_last):
                is_head = block_idx % self.group == self.group - 1
                head_bits = cute.make_rmem_tensor((num_rows,), Int32)
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    total = cutlass.Float32(0.0)
                    for warp in cutlass.range_constexpr(num_warps):
                        total = total + partials[mi, ni, warp]
                    if const_expr(self.gather):
                        _st_cluster_b16(
                            (
                                chunk_rows.iterator + (mi * _CHUNK + rank * outputs_per_block + ni)
                            ).toint(),
                            0,
                            _bf16_bits(total, True),
                        )
                    elif const_expr(self.pzo):
                        out_bits = _bf16_bits(total, True)
                        if const_expr(self.head_last and ni == 0):
                            head_bits[mi] = out_bits
                            if not is_head:
                                cute.recast_tensor(gC, cutlass.Int16)[m_base + mi, n_base] = (
                                    cutlass.Int16(out_bits)
                                )
                        else:
                            cute.recast_tensor(gC, cutlass.Int16)[m_base + mi, n_base + ni] = (
                                cutlass.Int16(out_bits)
                            )
                    else:
                        gC[m_base + mi, n_base + ni] = total.to(self.element_type)
            if const_expr(self.head_last):
                if is_head:
                    self._wait_chunk_tails(gC, m_base, n_base, retry_count)
                    for mi in cutlass.range_constexpr(num_rows):
                        cute.recast_tensor(gC, cutlass.Int16)[m_base + mi, n_base] = cutlass.Int16(
                            head_bits[mi]
                        )
        if const_expr(self.gather):
            cute.arch.cluster_arrive()
            if rank == 0:
                cute.arch.cluster_wait()
                if tidx < num_rows:
                    _copy_shared_to_global_v4(
                        (gC.iterator + cute.crd2idx((m_base + tidx, n_base), gC.layout)).toint(),
                        (chunk_rows.iterator + tidx * _CHUNK).toint(),
                    )

        if const_expr(self.use_pdl and not self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def kernel_group8(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        retry_count: cute.Tensor,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        if const_expr(self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, block_m, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        num_rows: cutlass.Constexpr = self.rows_per_block
        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        vector_width: cutlass.Constexpr = self.vector_width
        block_size: cutlass.Constexpr = self.block_size
        num_warps: cutlass.Constexpr = self.num_warps
        num_k_tiles: cutlass.Constexpr = self.num_k_tiles
        groups: cutlass.Constexpr = self.group

        if const_expr(self.use_pdl and self.pdl_wait_activation):
            cute.arch.griddepcontrol_wait()

        m_base = block_m * num_rows
        gA_vec = cute.logical_divide(gA, (None, vector_width))
        gB_vec = cute.logical_divide(gB, (None, vector_width))
        tA_all = cute.logical_divide(gA_vec, (None, (None, block_size)))
        tB_all = cute.logical_divide(gB_vec, (None, (None, block_size)))
        tA = tA_all[None, (None, (tidx, None))]

        a_rows = cute.make_rmem_tensor(
            cute.make_layout(
                (num_rows, num_k_tiles, vector_width),
                stride=(num_k_tiles * vector_width, vector_width, 1),
            ),
            self.element_type,
        )
        if const_expr(self.check_valid):
            self._load_rows_checked(copy_a, tA, m_base, a_rows, retry_count)
        else:
            for mi in cutlass.range_constexpr(num_rows):
                for k_tile in cutlass.range_constexpr(num_k_tiles):
                    cute.copy(copy_a, tA[m_base + mi, None, k_tile], a_rows[mi, k_tile, None])

        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(
                (groups, num_rows, outputs_per_block, num_warps),
                stride=(
                    num_rows * outputs_per_block * num_warps,
                    outputs_per_block * num_warps,
                    num_warps,
                    1,
                ),
            ),
            byte_alignment=16,
        )
        for g in cutlass.range_constexpr(groups):
            n_base = block_idx * _CHUNK + g * outputs_per_block
            acc = cute.make_rmem_tensor(
                cute.make_layout((num_rows, outputs_per_block), stride=(outputs_per_block, 1)),
                cutlass.Float32,
            )
            acc.fill(0.0)
            b_regs = cute.make_rmem_tensor(
                cute.make_layout(
                    (outputs_per_block, num_k_tiles, vector_width),
                    stride=(num_k_tiles * vector_width, vector_width, 1),
                ),
                self.element_type,
            )
            for ni in cutlass.range_constexpr(outputs_per_block):
                tB = tB_all[n_base + ni, (None, (tidx, None))]
                for k_tile in cutlass.range_constexpr(num_k_tiles):
                    cute.copy(copy_b, tB[None, k_tile], b_regs[ni, k_tile, None])
            # The activations are shared by all groups; a shared f32 conversion would defeat the fused bf16 FMA.
            for mi in cutlass.range_constexpr(num_rows):
                for k_tile in cutlass.range_constexpr(num_k_tiles):
                    for vi in cutlass.range_constexpr(vector_width):
                        for ni in cutlass.range_constexpr(outputs_per_block):
                            acc[mi, ni] = _fma_f32_bf16(
                                a_rows[mi, k_tile, vi], b_regs[ni, k_tile, vi], acc[mi, ni]
                            )
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])
            with cute.arch.elect_one():
                for mi in cutlass.range_constexpr(num_rows):
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        partials[g, mi, ni, warp_idx] = acc[mi, ni]

        cute.arch.sync_threads()
        if tidx < num_rows:
            words = []
            for j in cutlass.range_constexpr(_CHUNK // 2):
                pair = []
                for c in (2 * j, 2 * j + 1):
                    total = cutlass.Float32(0.0)
                    for warp in cutlass.range_constexpr(num_warps):
                        total = (
                            total
                            + partials[c // outputs_per_block, tidx, c % outputs_per_block, warp]
                        )
                    pair.append(total)
                words.append(_bf16x2_bits(pair[0], pair[1], True))
            _st_global_v4(
                (
                    gC.iterator + cute.crd2idx((m_base + tidx, block_idx * _CHUNK), gC.layout)
                ).toint(),
                words,
            )

        if const_expr(self.use_pdl and not self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _keep_waiting(self, pending, rounds):
        if const_expr(self.spin_limit):
            pending = pending & (rounds < self.spin_limit)
        return pending

    @cute.jit
    def _count(self, retry_count: cute.Tensor, rounds, bad):
        """Lane 0 adds the warp's reload rounds and, if a ``spin_limit`` wait ended unwritten, one give-up."""
        stuck = cute.arch.vote_any_sync(bad)
        if cute.arch.lane_idx() == 0:
            if rounds > 0:
                cute.arch.atomic_add(retry_count.iterator, rounds)
            if stuck:
                cute.arch.atomic_add(retry_count.iterator + 1, Int32(1))

    @cute.jit
    def _chunk_unwritten(
        self, addr, offset: cutlass.Constexpr[int], relaxed: cutlass.Constexpr[bool]
    ):
        if const_expr(self.check_elements):
            return _has_sentinel(_min_s16x2(Int32(_LANES_MAX), _ld_chunk(addr, offset, relaxed)))
        return (_ld_head(addr, offset, relaxed) & 0xFFFF) == _BF16_NEG_ZERO

    def _check_start(self):
        return Int32(_LANES_MAX) if self.check_elements else cutlass.Boolean(False)

    def _check_done(self, acc):
        return _has_sentinel(acc) if self.check_elements else acc

    @cute.jit
    def _wait_heads_written(self, tA: cute.Tensor, m_base, retry_count: cute.Tensor):
        """Return once every activation chunk this warp reads in rows ``m_base ..`` is written (element 0 only
        without ``check_elements``)."""
        num_rows: cutlass.Constexpr = self.rows_per_block
        num_k_tiles: cutlass.Constexpr = self.num_k_tiles
        k_stride: cutlass.Constexpr = self.k_tile_bytes
        bad = cutlass.Boolean(False)
        for mi in cutlass.range_constexpr(num_rows):
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                bad = bad | self._chunk_unwritten(
                    tA[m_base + mi, None, 0].iterator.toint(), k_tile * k_stride, False
                )
        pending = cute.arch.vote_any_sync(bad)
        rounds = Int32(0)
        while pending:
            _backoff(rounds)
            rounds = rounds + 1
            bad = cutlass.Boolean(False)
            for mi in cutlass.range_constexpr(num_rows):
                for k_tile in cutlass.range_constexpr(num_k_tiles):
                    bad = bad | self._chunk_unwritten(
                        tA[m_base + mi, None, 0].iterator.toint(), k_tile * k_stride, True
                    )
            pending = self._keep_waiting(cute.arch.vote_any_sync(bad), rounds)
        if rounds > 0:
            # The default body's loads must not hit the lines the first pass cached with the sentinel.
            cute.arch.fence_acq_rel_gpu()
            self._count(retry_count, rounds, bad)

    @cute.jit
    def _load_a_checked(
        self,
        copy_a: cute.CopyAtom,
        tA: cute.Tensor,
        row,
        a_regs: cute.Tensor,
        retry_count: cute.Tensor,
    ):
        """Load this thread's chunks of activation ``row`` into ``a_regs`` and return once every lane of the warp
        holds written ones."""
        num_k_tiles: cutlass.Constexpr = self.num_k_tiles
        words = cute.recast_tensor(a_regs, Int32)
        acc = self._check_start()
        for k_tile in cutlass.range_constexpr(num_k_tiles):
            if const_expr(self.relaxed_first):
                first = _ld_relaxed_v4(
                    tA[row, None, 0].iterator.toint(), k_tile * self.k_tile_bytes, False
                )
                for j in cutlass.range_constexpr(4):
                    words[k_tile, j] = first[j]
            else:
                cute.copy(copy_a, tA[row, None, k_tile], a_regs[k_tile, None])
            acc = _chunk_bad(acc, [words[k_tile, j] for j in range(4)], self.check_elements)
        bad = self._check_done(acc)
        pending = cute.arch.vote_any_sync(bad)
        rounds = Int32(0)
        while pending:
            _backoff(rounds)
            rounds = rounds + 1
            acc = self._check_start()
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                reloaded = _ld_relaxed_v4(
                    tA[row, None, 0].iterator.toint(), k_tile * self.k_tile_bytes, True
                )
                for j in cutlass.range_constexpr(4):
                    words[k_tile, j] = reloaded[j]
                acc = _chunk_bad(acc, reloaded, self.check_elements)
            bad = self._check_done(acc)
            pending = self._keep_waiting(cute.arch.vote_any_sync(bad), rounds)
        self._count(retry_count, rounds, bad)

    @cute.jit
    def _load_rows_checked(
        self,
        copy_a: cute.CopyAtom,
        tA: cute.Tensor,
        m_base,
        a_rows: cute.Tensor,
        retry_count: cute.Tensor,
    ):
        """``_load_a_checked`` for all of the CTA's rows at once, before any FMA."""
        num_rows: cutlass.Constexpr = self.rows_per_block
        num_k_tiles: cutlass.Constexpr = self.num_k_tiles
        words = cute.recast_tensor(a_rows, Int32)
        acc = self._check_start()
        for mi in cutlass.range_constexpr(num_rows):
            for k_tile in cutlass.range_constexpr(num_k_tiles):
                cute.copy(copy_a, tA[m_base + mi, None, k_tile], a_rows[mi, k_tile, None])
                acc = _chunk_bad(acc, [words[mi, k_tile, j] for j in range(4)], self.check_elements)
        bad = self._check_done(acc)
        pending = cute.arch.vote_any_sync(bad)
        rounds = Int32(0)
        while pending:
            _backoff(rounds)
            rounds = rounds + 1
            acc = self._check_start()
            for mi in cutlass.range_constexpr(num_rows):
                for k_tile in cutlass.range_constexpr(num_k_tiles):
                    reloaded = _ld_relaxed_v4(
                        tA[m_base + mi, None, 0].iterator.toint(), k_tile * self.k_tile_bytes, True
                    )
                    for j in cutlass.range_constexpr(4):
                        words[mi, k_tile, j] = reloaded[j]
                    acc = _chunk_bad(acc, reloaded, self.check_elements)
            bad = self._check_done(acc)
            pending = self._keep_waiting(cute.arch.vote_any_sync(bad), rounds)
        self._count(retry_count, rounds, bad)

    @cute.jit
    def _wait_chunk_tails(self, gC: cute.Tensor, m_base, n_base, retry_count: cute.Tensor):
        """Spin until elements ``outputs_per_block .. 7`` of the output chunks at (rows ``m_base ..``, ``n_base``)
        are no longer -0.0 (stored by the group's other CTAs). One thread."""
        num_rows: cutlass.Constexpr = self.rows_per_block
        first_tail: cutlass.Constexpr = self.outputs_per_block
        bad = cutlass.Boolean(True)
        pending = cutlass.Boolean(True)
        rounds = Int32(0)
        while pending:
            bad = cutlass.Boolean(False)
            for mi in cutlass.range_constexpr(num_rows):
                chunk = _ld_relaxed_v4(
                    (gC.iterator + cute.crd2idx((m_base + mi, n_base), gC.layout)).toint(), 0, True
                )
                for e in cutlass.range_constexpr(first_tail, _CHUNK):
                    bad = bad | (((chunk[e // 2] >> (16 * (e % 2))) & 0xFFFF) == _BF16_NEG_ZERO)
            pending = self._keep_waiting(bad, rounds)
            if pending:
                _backoff(rounds)
                rounds = rounds + 1
        if const_expr(self.spin_limit):
            if bad:
                cute.arch.atomic_add(retry_count.iterator + 1, Int32(1))


def _from_dlpack_static(tensor: _torch.Tensor):
    # K is specialized and the row stride must retain its 16-byte divisibility
    # for the verifier to accept vectorized G2R copies.
    return from_dlpack(tensor, assumed_align=32)


def _make_compile_repr_tensors(dtype, m: int, n: int, k: int):
    return tuple(
        _from_dlpack_static(tensor)
        for tensor in (
            _torch.empty((m, k), dtype=dtype, device="cuda"),
            _torch.empty((n, k), dtype=dtype, device="cuda"),
            _torch.empty((m, n), dtype=dtype, device="cuda"),
        )
    )


def _retry_tensor(retry_counter: _torch.Tensor):
    return from_dlpack(retry_counter, assumed_align=4)


@functools.cache
def _get_compiled_direct_kernel(
    dtype,
    m: int,
    n: int,
    k: int,
    tactic: DirectTactic,
    use_pdl: bool,
    check_valid: bool,
    relaxed_first: bool,
    check_mode: str,
    check_elements: bool,
    pdl_wait_activation: bool,
    pzo: bool,
    early_trigger: bool,
    head_last: bool,
    group8: bool,
    gather: bool,
    min_ctas_per_sm: int,
    spin_limit: int,
):
    if dtype != _torch.bfloat16:
        raise ValueError(f"direct GEMM supports BF16; got {dtype}")
    kernel = DirectFgsGemmKernel(
        element_type=cutlass.BFloat16,
        num_rows=m,
        k_extent=k,
        tactic=tactic,
        use_pdl=use_pdl,
        check_valid=check_valid,
        relaxed_first=relaxed_first,
        check_mode=check_mode,
        check_elements=check_elements,
        pdl_wait_activation=pdl_wait_activation,
        pzo=pzo,
        early_trigger=early_trigger,
        head_last=head_last,
        group8=group8,
        gather=gather,
        min_ctas_per_sm=min_ctas_per_sm,
        spin_limit=spin_limit,
    )
    tensors = _make_compile_repr_tensors(dtype, m, n, k)
    retry = _retry_tensor(_torch.zeros(RETRY_COUNTER_SLOTS, dtype=_torch.int32, device="cuda"))
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(kernel, *tensors, retry, stream, options=_COMPILE_OPTIONS)


def _validate_runtime_tensors(a, b, out, tactic: DirectTactic):
    if any(not isinstance(tensor, _torch.Tensor) for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be torch tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("direct GEMM accepts only 2D tensors")
    if a.device.type != "cuda" or b.device != a.device or out.device != a.device:
        raise ValueError("a, b, and out must be on the same CUDA device")
    if a.dtype != _torch.bfloat16 or b.dtype != a.dtype or out.dtype != a.dtype:
        raise ValueError("a, b, and out must share BF16 dtype")
    if not a.is_contiguous() or not b.T.is_contiguous() or not out.is_contiguous():
        raise ValueError("direct GEMM requires row-major A/out and column-major B")
    if any(tensor.data_ptr() % 32 for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be 32-byte aligned")

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}")
    n = b.shape[1]
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    validate_tactic(tactic, m, n, k)
    return m, n, k


def default_check_mode(tactic: DirectTactic, k: int) -> str:
    """``rows`` (one round trip, the reloaded data is used directly: no fence), or ``heads_before_b`` for multi-row
    tactics whose weight registers exceed ``_HEADS_BEFORE_B_REGS``."""
    b_regs = tactic.outputs_per_block * k // (tactic.block_size * _VECTOR_WIDTH) * 4
    return (
        "heads_before_b" if tactic.rows_per_block > 1 and b_regs > _HEADS_BEFORE_B_REGS else "rows"
    )


@functools.cache
def _sm_count(device_index: int) -> int:
    return _torch.cuda.get_device_properties(device_index).multi_processor_count


def default_min_ctas_per_sm(
    tactic: DirectTactic, m: int, n: int, k: int, device_index: int, group8: bool = False
) -> int:
    """The default build's occupancy for an FGS build whose grid needs more than one wave at it, else 1."""
    ctas = _DEFAULT_CTAS_PER_SM.get((k, dataclasses.astuple(tactic)), 1)
    grid = -(-n // (_CHUNK if group8 else tactic.outputs_per_block)) * (m // tactic.rows_per_block)
    return ctas if grid > ctas * _sm_count(device_index) else 1


_SCRATCH_RETRY_COUNTERS: dict[int, _torch.Tensor] = {}


def _scratch_retry_counter(device: _torch.device) -> _torch.Tensor:
    index = device.index if device.index is not None else _torch.cuda.current_device()
    if index not in _SCRATCH_RETRY_COUNTERS:
        _SCRATCH_RETRY_COUNTERS[index] = _torch.zeros(
            RETRY_COUNTER_SLOTS, dtype=_torch.int32, device=device
        )
    return _SCRATCH_RETRY_COUNTERS[index]


def run_direct_dense_fgs(
    a,
    b,
    out,
    pdl: bool,
    tactic: DirectTactic,
    *,
    check_valid: bool = True,
    relaxed_first: bool = False,
    check_mode: str | None = None,
    check_elements: bool = True,
    pdl_wait_activation: bool = False,
    retry_counter: _torch.Tensor | None = None,
    pzo: bool = False,
    early_trigger: bool = False,
    head_last: bool = False,
    group8: bool = False,
    gather: bool = False,
    min_ctas_per_sm: int | None = None,
    spin_limit: int = 0,
):
    """Run direct ``A[M,K] @ B[K,N]`` with the ``mm_bf16`` layouts; ``a`` is the FGS-checked operand.

    ``retry_counter`` (int32, ``RETRY_COUNTER_SLOTS`` elements) accumulates the activation reload rounds and the
    waits ``spin_limit`` (diagnostic) gave up. With every
    role off (``check_valid=False``, ``pdl_wait_activation=True``) this is FlashInfer's ``run_direct_dense``.
    ``check_mode=None`` takes ``default_check_mode``; ``min_ctas_per_sm`` (launch bound) ``None`` takes
    ``default_min_ctas_per_sm`` when any role is on and 1 (the default build's) otherwise.
    """
    m, n, k = _validate_runtime_tensors(a, b, out, tactic)
    if not check_valid:
        check_mode = "row"
    elif check_mode is None:
        check_mode = default_check_mode(tactic, k)
    if min_ctas_per_sm is None:
        roles = check_valid or pzo or early_trigger or head_last or group8 or gather
        min_ctas_per_sm = (
            default_min_ctas_per_sm(tactic, m, n, k, a.get_device(), group8) if roles else 1
        )
    if (head_last or group8 or gather) and n % _CHUNK:
        raise ValueError(f"head_last / group8 / gather need N % {_CHUNK} == 0, got N={n}")
    if retry_counter is None:
        retry_counter = _scratch_retry_counter(a.device)
    elif (
        retry_counter.dtype != _torch.int32
        or retry_counter.numel() != RETRY_COUNTER_SLOTS
        or retry_counter.device != a.device
    ):
        raise ValueError(f"retry_counter must be int32[{RETRY_COUNTER_SLOTS}] on {a.device}")
    compiled = _get_compiled_direct_kernel(
        a.dtype,
        m,
        n,
        k,
        tactic,
        pdl,
        check_valid,
        relaxed_first,
        check_mode,
        check_elements,
        pdl_wait_activation,
        pzo,
        early_trigger,
        head_last,
        group8,
        gather,
        min_ctas_per_sm,
        spin_limit,
    )
    tensors = tuple(_from_dlpack_static(tensor) for tensor in (a, b.T, out))
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(*tensors, _retry_tensor(retry_counter), stream)
    return out


__all__ = [
    "DirectTactic",
    "RETRY_COUNTER_SLOTS",
    "default_check_mode",
    "default_min_ctas_per_sm",
    "default_tactic",
    "run_direct_dense_fgs",
    "validate_tactic",
]
