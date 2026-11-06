# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode MoE router as one CuTe DSL kernel (SM107).

Replaces the stock sequence cast h -> fp32, tf32 router GEMM (cuBLASLt split-K + reduce), ``noaux_tc`` top-16
(``deepseek_v3_topk_kernel``) and the routing-weight cast:

  logits[t, e] = tf32 GEMM of bf16 h[t, :] (exact in tf32) and the gate weight W[e, :] (fp32, rounded to tf32 on
                 load), fp32 accumulate
  score = sigmoid(logits) = 0.5 * tanh(0.5 * x) + 0.5; key = score + e_score_correction_bias
  ids[t, k] = the k-th largest key (ties: lower expert first), weights[t, k] =
  bf16(f32(f64(score[ids[t, k]]) * scale / (f64(sum over k of score) + 1e-20)))

ids [T, 16] int32 and weights [T, 16] bf16 are what the dispatch sends. From 2 tokens on the logits are bitwise
equal to the stock cuBLASLt kernel's: the K axis is cut into the same ``split`` contiguous slices, each slice
accumulates in one tcgen05 accumulator in K order, and the slices are summed in slice order. Everything after the
logits matches the stock top-k bit for bit (same key order, warp butterfly for the normalizer, f64 division).

Grid: 14 clusters (expert tiles of 64) of ``split`` CTAs (K slices). CTA (tile, slice) runs a swap-AB kind::tf32
GEMM, MMA M = 64 experts, N = ``n_mma`` >= tokens (rows past the token count are TMA zero fill). The weight slice
streams by TMA from kernel entry; the bf16 activation is TMA-loaded in chunks of several K stages (after
``griddepcontrol.wait``, or at once and validated per 16 B against the -0 sentinel with ``mix_router``) and widened
to fp32 in shared memory by warps 0-3 (exact). The cluster exchanges partials through DSMEM so that rank q sums
tokens q, q + split, ... in slice order and writes the tile's top-16 (key, expert, score) per token. A per-token
counter elects the tile that completes a token (the last of 14 release adds) to merge the 14 x 16 candidates and
store the row. No CTA waits on a CTA outside its cluster, so the kernel cannot deadlock at any residency; the
counters return to zero in every launch.
"""

from __future__ import annotations

import os
import struct

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from .sync import spin_ns

HIDDEN = 7168
NUM_EXPERTS = 896
TOP_K = 16
M_TILE = 64
E_TILES = NUM_EXPERTS // M_TILE
K_STAGE = 32
K_TILES = HIDDEN // K_STAGE
CANDS = E_TILES * TOP_K
MAX_TOKENS = 128
N_MMA = (16, 32, 64, 128)
SMEM_ARCH = "sm_107"
SMEM_BUDGET = 227 * 1024
INT_MIN = -(2**31)


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


def _i32(v, loc, ip):
    return cutlass.Int32(v).ir_value(loc=loc, ip=ip)


def _i64(v, loc, ip):
    return cutlass.Int64(v).ir_value(loc=loc, ip=ip)


def _f32(v, loc, ip):
    return cutlass.Float32(v).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _f32_bits(x, *, loc=None, ip=None):
    return cutlass.Int32(_asm(T.i32(), [_f32(x, loc, ip)], "mov.b32 $0, $1;", "=r,f", loc, ip))


@dsl_user_op
def _bits_f32(x, *, loc=None, ip=None):
    return cutlass.Float32(_asm(T.f32(), [_i32(x, loc, ip)], "mov.b32 $0, $1;", "=f,r", loc, ip))


@dsl_user_op
def _tmem_ld16(addr, *, loc=None, ip=None):
    """tcgen05.ld 32x32b.x16: thread i of warp w gets TMEM lane 32 * w + i, columns addr .. addr + 15."""
    regs = ", ".join(f"${i}" for i in range(16))
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 16),
        [_i32(addr, loc, ip)],
        f"tcgen05.ld.sync.aligned.32x32b.x16.b32 {{{regs}}}, [$16];\ntcgen05.wait::ld.sync.aligned;",
        ",".join(["=f"] * 16 + ["r"]),
        loc,
        ip,
    )
    return [
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(16)
    ]


@dsl_user_op
def _tcgen05_fence(kind: str, *, loc=None, ip=None):
    _asm(None, [], f"tcgen05.fence::{kind}_thread_sync;", "", loc, ip)


@dsl_user_op
def _st_global_v4i(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip)] + [_i32(x, loc, ip) for x in v],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def _ld_global_cg_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_i64(addr, loc, ip)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ld_global_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(T.f32(), [_i64(addr, loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", loc, ip)
    )


@dsl_user_op
def _st_global_b32(addr, v, *, loc=None, ip=None):
    _asm(None, [_i64(addr, loc, ip), _i32(v, loc, ip)], "st.global.b32 [$0], $1;", "l,r", loc, ip)


@dsl_user_op
def _st_global_f32(addr, v, *, loc=None, ip=None):
    _asm(None, [_i64(addr, loc, ip), _f32(v, loc, ip)], "st.global.f32 [$0], $1;", "l,f", loc, ip)


@dsl_user_op
def _ld_shared_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_i32(addr, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _st_shared_v4(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip)] + [_i32(x, loc, ip) for x in v],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def _ld_shared_b32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(T.i32(), [_i32(addr, loc, ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip)
    )


@dsl_user_op
def _st_shared_b32(addr, v, *, loc=None, ip=None):
    _asm(None, [_i32(addr, loc, ip), _i32(v, loc, ip)], "st.shared.b32 [$0], $1;", "r,r", loc, ip)


@dsl_user_op
def _atom_add_acq_rel(addr, v, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_i64(addr, loc, ip), _i32(v, loc, ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _red_release_add(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(v, loc, ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def _redux_max(v, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(T.i32(), [_i32(v, loc, ip)], "redux.sync.max.s32 $0, $1, 0xffffffff;", "=r,r", loc, ip)
    )


@dsl_user_op
def _shfl_idx(v, lane, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_i32(v, loc, ip), _i32(lane, loc, ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _shfl_bfly_f32(v, offset: int, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_f32(v, loc, ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, 0xffffffff;",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip), _f32(c, loc, ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _route_weight_bits(score, norm, scale_bits: int, *, loc=None, ip=None):
    """bf16 bits of f32(f64(score) * scale / (f64(norm) + 1e-20)), as the stock top-k and cast compute it."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_f32(score, loc, ip), _f32(norm, loc, ip)],
            "{ .reg .f64 a, b, c; .reg .f32 f; .reg .b16 h; cvt.f64.f32 a, $1; "
            f"mul.rn.f64 a, a, 0d{scale_bits:016x}; cvt.f64.f32 b, $2; add.rn.f64 b, b, 0d3bc79ca10c924223; "
            "div.rn.f64 c, a, b; cvt.rn.f32.f64 f, c; cvt.rn.bf16.f32 h, f; cvt.u32.u16 $0, h; }",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _div_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_bits(x, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_f32(x, loc, ip)],
            "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.u32.u16 $0, h; }",
            "=r,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _globaltimer_lo(*, loc=None, ip=None):
    return cutlass.Int32(_asm(T.i32(), [], "mov.u32 $0, %globaltimer_lo;", "=r", loc, ip))


@dsl_user_op
def _bar_sync(barrier_id: int, num_threads: int, *, loc=None, ip=None):
    _asm(None, [], f"barrier.sync {barrier_id}, {num_threads};", "", loc, ip)


@dsl_user_op
def _fence_proxy_async_shared(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc, ip)


class _Barrier:
    def __init__(self, barrier_id: int, num_threads: int):
        self.barrier_id = barrier_id
        self.num_threads = num_threads

    def arrive_and_wait(self):
        _bar_sync(self.barrier_id, self.num_threads)

    def wait_unaligned(self):
        _bar_sync(self.barrier_id, self.num_threads)


@cute.jit
def _sigmoid(x: cutlass.Float32) -> cutlass.Float32:
    """0.5 * tanh(0.5 * x) + 0.5 with the libm tanh and a fused multiply-add, bitwise as the stock top-k."""
    return _fma_rn(
        cutlass.Float32(0.5), cute.math.tanh(_mul_rn(cutlass.Float32(0.5), x)), cutlass.Float32(0.5)
    )


@cute.jit
def _sortable(x: cutlass.Float32) -> cutlass.Int32:
    """Signed-int image of ``x`` with the order of the stock top-k key's value word (-0 below +0)."""
    b = _f32_bits(x)
    return cutlass.select_(b >= 0, b, b ^ cutlass.Int32(0x7FFFFFFF))


@dsl_user_op
def _st_cluster_f32(addr, rank, v, *, loc=None, ip=None):
    """Store to the shared-memory address ``addr`` of cluster CTA ``rank``."""
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(rank, loc, ip), _f32(v, loc, ip)],
        "{ .reg .b32 ra; mapa.shared::cluster.u32 ra, $0, $1; st.shared::cluster.f32 [ra], $2; }",
        "r,r,f",
        loc,
        ip,
    )


@dsl_user_op
def _ld_shared_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(T.f32(), [_i32(addr, loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc, ip)
    )


@dsl_user_op
def _poll_v4(addr, *, loc=None, ip=None):
    """Re-read 16 B at a global address until its first bf16 is not the -0 sentinel."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_i64(addr, loc, ip)],
        "{ .reg .pred p; .reg .b32 lo; L_poll_${:uid}: ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4]; "
        "and.b32 lo, $0, 0xffff; setp.eq.u32 p, lo, 0x8000; @p nanosleep.u32 32; @p bra L_poll_${:uid}; }",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _cp_async_16(smem_addr, addr, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(smem_addr, loc, ip), _i64(addr, loc, ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        loc,
        ip,
    )


@dsl_user_op
def _cp_async_wait_all(*, loc=None, ip=None):
    _asm(None, [], "cp.async.commit_group;\ncp.async.wait_group 0;", "", loc, ip)


@dsl_user_op
def _cluster_sync(*, loc=None, ip=None):
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;",
        "",
        loc,
        ip,
    )


@dsl_user_op
def _syncwarp(*, loc=None, ip=None):
    _asm(None, [], "bar.warp.sync 0xffffffff;", "", loc, ip)


class RouterKernel:
    """One cluster per expert tile, one CTA per K slice (cluster rank = slice). Warps 0-3: activation widening,
    then the epilogue and top-k; warp 4: MMA; warp 5: TMA."""

    def __init__(
        self,
        n_mma: int,
        split: int,
        use_pdl: bool,
        write_logits: bool,
        delay_ns: int,
        scale: float,
        timeline: bool = False,
        x_check: bool = False,
        x_wait: bool = True,
        smem_budget: int = SMEM_BUDGET,
        balanced: tuple = None,
    ):
        if n_mma not in N_MMA:
            raise ValueError(f"n_mma must be one of {N_MMA}")
        if K_TILES % split or split > 8:
            raise ValueError(f"split must divide {K_TILES} and fit a portable cluster (<= 8)")
        self.m_tile = M_TILE
        self.e_tiles = E_TILES
        self.cands = self.e_tiles * TOP_K
        self.timeline = timeline
        self.x_check = x_check
        self.x_wait = x_wait
        self.balanced = balanced
        self.n_mma = n_mma
        self.split = split
        self.k_iters = K_TILES // split
        self.use_pdl = use_pdl
        self.write_logits = write_logits
        self.delay_ns = delay_ns
        self.scale_bits = int.from_bytes(struct.pack(">d", scale), "big")
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 192
        self.work_threads = 128
        self.work_barrier = _Barrier(1, self.work_threads)
        self.tmem_alloc_barrier = _Barrier(2, self.work_threads + 32)
        self.w_stage_bytes = self.m_tile * K_STAGE * 4
        # The activation moves in chunks of x_atoms K stages (one TMA, one widening pass and one handoff each), so the
        # per-stage handoff cost does not throttle the weight stream.
        self.x_atoms = {16: 4, 32: 4, 64: 2, 128: 1}[n_mma]
        self.x_chunks = self.k_iters // self.x_atoms
        self.kc = self.x_atoms * K_STAGE
        self.x_chunk_bytes = n_mma * self.kc * 4
        self.xs_chunk_bytes = n_mma * self.kc * 2
        # 16 B activation pieces per chunk; every widening thread owns the same number.
        self.pieces = n_mma * self.kc // 8
        assert self.pieces % self.work_threads == 0
        # Batched re-copy of the chunks loaded before the producer stored them; pays at 16 tokens, not at 64.
        self.x_refresh = x_check and n_mma <= 32
        # Tokens per cluster rank: rank q reduces and ranks tokens q, q + split, ...
        self.t_own = (n_mma + split - 1) // split
        mailbox_bytes = split * self.t_own * self.m_tile * 4
        budget = (
            min(utils.get_smem_capacity_in_bytes(SMEM_ARCH), smem_budget) - mailbox_bytes - 4096
        )
        # Activation chunks are re-loaded only as the MMA frees them, so their ring must cover the load latency as
        # deeply as the weight ring: split the budget for equal depth in K stages.
        x_stage_bytes = (self.x_chunk_bytes + self.xs_chunk_bytes) // self.x_atoms
        depth = budget // (self.w_stage_bytes + x_stage_bytes)
        self.x_stages = max(1, min(self.x_chunks, (depth + self.x_atoms - 1) // self.x_atoms))
        fixed = self.x_stages * (self.x_chunk_bytes + self.xs_chunk_bytes)
        self.stages = max(2, min((budget - fixed) // self.w_stage_bytes, self.k_iters))
        self.off_x = self.stages * self.w_stage_bytes
        self.off_xs = self.off_x + self.x_stages * self.x_chunk_bytes
        self.mailbox_off = self.off_xs + self.x_stages * self.xs_chunk_bytes
        self.ring_bytes = self.mailbox_off + mailbox_bytes
        # After the mainloop the weight ring holds the owned tokens' keys and scores, the merge flags and one
        # candidate table per warp.
        self.merge_off = (self.t_own * self.m_tile * 8 + 4 * self.t_own + 15) // 16 * 16
        assert self.merge_off + 4 * self.cands * 16 <= self.off_x
        self.tmem_cols = max(32, n_mma)

    @cute.jit
    def _stamp(self, tl_addr, bidx, slot: int, cond):
        if cutlass.const_expr(self.timeline):
            if cond:
                _st_global_b32(tl_addr + cutlass.Int64((bidx * 8 + slot) * 4), _globaltimer_lo())

    def _tiled_mma(self):
        return sm100_utils.make_trivial_tiled_mma(
            cutlass.TFloat32,
            cutlass.TFloat32,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            (self.m_tile, self.n_mma),
        )

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        num_tokens: cutlass.Int32,
        x_addr: cutlass.Int64,
        bias_addr: cutlass.Int64,
        ids_addr: cutlass.Int64,
        wts_addr: cutlass.Int64,
        ready_addr: cutlass.Int64,
        logits_addr: cutlass.Int64,
        cand_addr: cutlass.Int64,
        cnt_addr: cutlass.Int64,
        tl_addr: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        n = self.n_mma
        tiled_mma = self._tiled_mma()
        mma_tiler = (self.m_tile, n, K_STAGE)
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler, cutlass.TFloat32, self.stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, (self.m_tile, n, self.kc), cutlass.TFloat32, self.x_stages
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )

        w = cute.make_tensor(
            w_ptr,
            cute.make_layout((NUM_EXPERTS, HIDDEN, 1), stride=(HIDDEN, 1, NUM_EXPERTS * HIDDEN)),
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        tma_atom_w, tma_w = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w,
            cute.slice_(a_layout, (None, None, None, 0)),
            mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        x = cute.make_tensor(x_ptr, cute.make_layout((num_tokens, HIDDEN), stride=(HIDDEN, 1)))
        tma_atom_x, tma_x = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            x,
            cute.make_layout((n, self.kc), stride=(self.kc, 1)),
            (n, self.kc),
        )

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            cv_mbar: cute.struct.MemRange[cutlass.Int64, self.x_stages * 2]
            xin_mbar: cute.struct.MemRange[cutlass.Int64, self.x_stages]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            ring: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.ring_bytes], 1024]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_w,
            tma_w,
            tma_atom_x,
            tma_x,
            num_tokens,
            x_addr,
            bias_addr,
            ids_addr,
            wts_addr,
            ready_addr,
            logits_addr,
            cand_addr,
            cnt_addr,
            tl_addr,
            cluster_layout_vmnk,
            a_layout,
            b_layout,
        ).launch(
            grid=[self.e_tiles * self.split, 1, 1],
            block=[self.threads, 1, 1],
            cluster=(self.split, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_w: cute.CopyAtom,
        mW: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mX: cute.Tensor,
        num_tokens: cutlass.Int32,
        x_addr: cutlass.Int64,
        bias_addr: cutlass.Int64,
        ids_addr: cutlass.Int64,
        wts_addr: cutlass.Int64,
        ready_addr: cutlass.Int64,
        logits_addr: cutlass.Int64,
        cand_addr: cutlass.Int64,
        cnt_addr: cutlass.Int64,
        tl_addr: cutlass.Int64,
        cluster_layout_vmnk: cute.Layout,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
    ):
        n = self.n_mma
        k_iters = self.k_iters
        stages = self.stages
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tile = bidx // self.split
        split = bidx % self.split
        lane = tidx % 32

        self._stamp(tl_addr, bidx, 0, tidx == 0)
        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_x)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ring = storage.ring.data_ptr()
        ring_addr = ring.toint()

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.w_stage_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        cv_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.cv_mbar.data_ptr(),
            num_stages=self.x_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.work_threads // 32
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.work_threads),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
        )
        tmem.allocate(self.tmem_cols)
        xin = storage.xin_mbar.data_ptr()
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.x_stages):
                    cute.arch.mbarrier_init(xin + i, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        sW = cute.make_tensor(
            cute.recast_ptr(ring, a_layout.inner, dtype=cutlass.TFloat32), a_layout.outer
        )
        sX = cute.make_tensor(
            cute.recast_ptr(ring + self.off_x, b_layout.inner, dtype=cutlass.TFloat32),
            b_layout.outer,
        )
        sXs = cute.make_tensor(
            cute.recast_ptr(ring + self.off_xs, dtype=cutlass.BFloat16),
            cute.make_layout((n, self.kc, self.x_stages), stride=(self.kc, 1, n * self.kc)),
        )

        gW = cute.local_tile(
            mW, cute.slice_((self.m_tile, n, K_STAGE), (None, 0, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgW = thr_mma.partition_A(gW)
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            cute.make_layout(1),
            cute.group_modes(sW, 0, 3),
            cute.group_modes(tCgW, 0, 3),
        )
        gX = cute.local_tile(mX, (n, self.kc), (0, None))
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sXs, 0, 2),
            cute.group_modes(gX, 0, 2),
        )
        tCrW = tiled_mma.make_fragment_A(sW)
        tCrX = tiled_mma.make_fragment_B(sX)
        acc_shape = tiled_mma.partition_shape_C((self.m_tile, n))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))

        k0 = split * k_iters
        n_pre = min(stages, k_iters)

        # TMA warp: the weight slice streams from kernel entry. Activation chunk c goes to slot c % x_stages once the
        # MMA has released the slot's previous chunk; without ``x_check`` only after the grid dependency (the
        # activation is the predecessor's output), with it right away (the widening warps validate it).
        if warp == self.tma_warp:
            x0 = split * self.x_chunks
            cv_empty = cv_pipeline.sync_object_empty
            state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, stages)
            if cutlass.const_expr(not self.x_wait):
                for c in cutlass.range_constexpr(self.x_stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(xin + c, self.xs_chunk_bytes)
                    cute.copy(
                        tma_atom_x, tXgX[(None, x0 + c)], tXsX[(None, c)], tma_bar_ptr=xin + c
                    )
            for i in cutlass.range_constexpr(n_pre):
                ab_pipeline.producer_acquire(state)
                cute.copy(
                    tma_atom_w,
                    tWgW[(None, tile, k0 + i, 0)],
                    tWsW[(None, state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(state),
                )
                state.advance()
            if cutlass.const_expr(self.x_wait):
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
                for c in cutlass.range_constexpr(self.x_stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(xin + c, self.xs_chunk_bytes)
                    cute.copy(
                        tma_atom_x, tXgX[(None, x0 + c)], tXsX[(None, c)], tma_bar_ptr=xin + c
                    )
            if cutlass.const_expr(self.use_pdl):
                if cutlass.const_expr(not self.x_wait):
                    cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            next_x = cutlass.Int32(self.x_stages)
            for i in cutlass.range(n_pre, k_iters, unroll=1):
                if (next_x < self.x_chunks) & (next_x * self.x_atoms <= i):
                    slot = next_x % self.x_stages
                    cv_empty.wait(slot, ((next_x // self.x_stages) & 1) ^ 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(xin + slot, self.xs_chunk_bytes)
                    cute.copy(
                        tma_atom_x,
                        tXgX[(None, x0 + next_x)],
                        tXsX[(None, slot)],
                        tma_bar_ptr=xin + slot,
                    )
                    next_x = next_x + 1
                ab_pipeline.producer_acquire(state)
                cute.copy(
                    tma_atom_w,
                    tWgW[(None, tile, k0 + i, 0)],
                    tWsW[(None, state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(state),
                )
                state.advance()
            while next_x < self.x_chunks:
                slot = next_x % self.x_stages
                cv_empty.wait(slot, ((next_x // self.x_stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(xin + slot, self.xs_chunk_bytes)
                cute.copy(
                    tma_atom_x,
                    tXgX[(None, x0 + next_x)],
                    tXsX[(None, slot)],
                    tma_bar_ptr=xin + slot,
                )
                next_x = next_x + 1
            ab_pipeline.producer_tail(state)
            _cluster_sync()

        if warp == self.mma_warp:
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(acc_ptr, tCtAcc_fake.layout)[(None, None, None, 0)]
            ab_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, stages)
            cv_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.x_stages
            )
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            acc_pipeline.producer_acquire(acc_state)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            num_kblocks = cute.size(tCrW, mode=[2])
            for _c in cutlass.range(self.x_chunks, unroll=1):
                cv_pipeline.consumer_wait(cv_state)
                for a in cutlass.range_constexpr(self.x_atoms):
                    ab_pipeline.consumer_wait(ab_state)
                    for kb in cutlass.range_constexpr(num_kblocks):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrW[(None, None, kb, ab_state.index)],
                            tCrX[(None, None, a * num_kblocks + kb, cv_state.index)],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_pipeline.consumer_release(ab_state)
                    ab_state.advance()
                cv_pipeline.consumer_release(cv_state)
                cv_state.advance()
            acc_pipeline.producer_commit(acc_state)
            acc_state.advance()
            # The epilogue releases the accumulator before the cluster barrier.
            _cluster_sync()
            acc_pipeline.producer_tail(acc_state)

        if warp < 4:
            self._work(
                tidx,
                warp,
                lane,
                tile,
                split,
                num_tokens,
                ring_addr,
                xin,
                x_addr,
                ab_pipeline,
                cv_pipeline,
                acc_pipeline,
                tmem,
                bias_addr,
                ids_addr,
                wts_addr,
                ready_addr,
                logits_addr,
                cand_addr,
                cnt_addr,
                tl_addr,
                bidx,
            )

    @cute.jit
    def _work(
        self,
        tidx,
        warp,
        lane,
        tile,
        split,
        num_tokens,
        ring_addr,
        xin,
        x_addr,
        ab_pipeline,
        cv_pipeline,
        acc_pipeline,
        tmem,
        bias_addr,
        ids_addr,
        wts_addr,
        ready_addr,
        logits_addr,
        cand_addr,
        cnt_addr,
        tl_addr,
        bidx,
    ):
        n = self.n_mma
        split_count = self.split
        t_own = self.t_own

        # The MMA warp waits for the allocation with these warps, so they meet it before the mainloop.
        tmem.wait_for_alloc()
        acc_ptr = tmem.retrieve_ptr(self.acc_dtype)
        # Reduction role: thread r < m_tile owns expert row r of the tile.
        r = tidx
        reducer = r < self.m_tile
        expert = tile * self.m_tile + cutlass.min(r, self.m_tile - 1)
        bias = _ld_global_f32(bias_addr + expert * 4)
        # M = 64 accumulator: row 16 * w + l sits in TMEM lane 32 * w + l for l < 16.
        acc_row = 16 * warp + lane
        acc_valid = lane < 16

        # Widen each activation chunk bf16 -> fp32 into the SW128 K-major B operand: K atom a (32 fp32) of row r
        # sits at a * n * 128 + r * 128, its 16 B piece j at piece j ^ (r % 8). With ``x_check`` a 16 B piece whose
        # first element is the -0 sentinel was loaded before its producer stored it; the thread re-reads it from
        # global until it is valid (the producer writes each 16 B piece whole).
        xs_base = ring_addr + self.off_xs
        x_base = ring_addr + self.off_x
        pieces = self.pieces
        per_row = self.kc // 8
        passes = pieces // self.work_threads
        cv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.x_stages)
        xin_phase = cutlass.Int32(0)
        refreshed = cutlass.Int32(0)
        for c in cutlass.range(self.x_chunks, unroll=1):
            slot = c % self.x_stages
            cute.arch.mbarrier_wait(xin + slot, xin_phase)
            if slot == self.x_stages - 1:
                xin_phase = xin_phase ^ 1
            self._stamp(tl_addr, bidx, 1, (tidx == 0) & (c == 0))
            cv_pipeline.producer_acquire(cv_state)
            k_base = (split * self.x_chunks + c) * self.kc
            if cutlass.const_expr(self.x_refresh):
                if (refreshed == 0) & (c < self.x_stages):
                    if self._stale(tidx, xs_base + slot * self.xs_chunk_bytes, num_tokens) != 0:
                        self._refresh_x(tidx, c, xin, xs_base, x_addr, split, num_tokens)
                        refreshed = cutlass.Int32(1)
            for p in cutlass.range_constexpr(passes):
                q = tidx + p * self.work_threads
                if q < pieces:
                    row = q // per_row
                    kk = (q % per_row) * 8
                    src = _ld_shared_v4(
                        xs_base + slot * self.xs_chunk_bytes + (row * self.kc + kk) * 2
                    )
                    if cutlass.const_expr(self.x_check):
                        if ((src[0] & 0xFFFF) == 0x8000) & (row < num_tokens):
                            src = _poll_v4(x_addr + cutlass.Int64((row * HIDDEN + k_base + kk) * 2))
                    wide = []
                    for v in src:
                        wide.append(v << 16)
                        wide.append(v & cutlass.Int32(-65536))
                    dst = x_base + slot * self.x_chunk_bytes + (kk // 32) * n * 128 + row * 128
                    sw = row & 7
                    j0 = (kk % 32) // 4
                    _st_shared_v4(dst + ((j0 ^ sw) << 4), wide[0:4])
                    _st_shared_v4(dst + (((j0 + 1) ^ sw) << 4), wide[4:8])
            # Each thread orders its generic-proxy stores for the MMA; one arrive per warp.
            _fence_proxy_async_shared()
            _syncwarp()
            if lane == 0:
                cv_pipeline.producer_commit(cv_state)
            cv_state.advance()

        self._stamp(tl_addr, bidx, 7, tidx == 0)

        # Exchange through DSMEM: column t of this slice's partial goes to rank t % split, which owns token t, in its
        # mailbox [slice][token / split][expert row] fp32.
        acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        acc_pipeline.consumer_wait(acc_state)
        _tcgen05_fence("after")
        self._stamp(tl_addr, bidx, 2, tidx == 0)
        if cutlass.const_expr(self.use_pdl and not self.x_wait):
            # A previous launch's consumers may still read the outputs.
            cute.arch.griddepcontrol_wait()
        tmem_row = acc_ptr.toint() + ((32 * warp) << 16)
        mailbox = ring_addr + self.mailbox_off
        m = self.m_tile
        for c in cutlass.range_constexpr(n // 16):
            if c * 16 < num_tokens:
                v = _tmem_ld16(tmem_row + 16 * c)
                for j in cutlass.range_constexpr(16):
                    t = 16 * c + j
                    if (t < num_tokens) & acc_valid:
                        slot = mailbox + ((split * t_own + t // split_count) * m + acc_row) * 4
                        _st_cluster_f32(slot, cutlass.Int32(t % split_count), v[j])
        _tcgen05_fence("before")
        acc_pipeline.consumer_release(acc_state)
        _cluster_sync()
        tmem.relinquish_alloc_permit()
        self.work_barrier.arrive_and_wait()
        tmem.free(acc_ptr)
        self._stamp(tl_addr, bidx, 3, tidx == 0)

        # Owned tokens: sum the slices in slice order, then key = sortable(sigmoid + bias), score kept.
        s_base = ring_addr
        sig_base = ring_addr + t_own * m * 4
        mine = (num_tokens - split + split_count - 1) // split_count
        if reducer:
            for tl in cutlass.range(mine, unroll=1):
                acc = _ld_shared_f32(mailbox + (tl * m + r) * 4)
                for s in cutlass.range_constexpr(1, split_count):
                    acc = _add_rn(acc, _ld_shared_f32(mailbox + ((s * t_own + tl) * m + r) * 4))
                if cutlass.const_expr(self.write_logits):
                    t = tl * split_count + split
                    _st_global_f32(logits_addr + cutlass.Int64((t * NUM_EXPERTS + expert) * 4), acc)
                score = _sigmoid(acc)
                _st_shared_b32(s_base + (tl * m + r) * 4, _sortable(_add_rn(score, bias)))
                _st_shared_b32(sig_base + (tl * m + r) * 4, _f32_bits(score))
        self.work_barrier.arrive_and_wait()

        # Tile top-16 per owned token by rank counting, one thread per key (rank = keys above, ties to the lower
        # expert); rank k < 16 is stored at candidate slot k, so each tile's list is descending.
        per_pass = self.work_threads // m
        rr = tidx % m
        for tl in cutlass.range(tidx // m, mine, per_pass, unroll=1):
            t = tl * split_count + split
            my_s = _ld_shared_b32(s_base + (tl * m + rr) * 4)
            rank = cutlass.Int32(0)
            for j4 in cutlass.range_constexpr(m // 4):
                v = _ld_shared_v4(s_base + (tl * m + 4 * j4) * 4)
                for q in cutlass.range_constexpr(4):
                    above = (v[q] > my_s) | ((v[q] == my_s) & (rr > 4 * j4 + q))
                    rank = rank + cutlass.select_(above, cutlass.Int32(1), cutlass.Int32(0))
            if rank < TOP_K:
                slot = cand_addr + cutlass.Int64(((t * self.cands + tile * TOP_K + rank) * 4) * 4)
                _st_global_v4i(
                    slot,
                    [
                        my_s,
                        65535 - (tile * m + rr),
                        _ld_shared_b32(sig_base + (tl * m + rr) * 4),
                        cutlass.Int32(0),
                    ],
                )
        self.work_barrier.arrive_and_wait()

        self._stamp(tl_addr, bidx, 4, tidx == 0)
        # One count per token: the release add of the tile that completes a token (e_tiles adds) elects it to merge.
        flags = ring_addr + t_own * m * 8
        if tidx < mine:
            t = tidx * split_count + split
            old = _atom_add_acq_rel(cnt_addr + cutlass.Int64(t * 4), cutlass.Int32(1))
            last = old == self.e_tiles - 1
            if last:
                _st_global_b32(cnt_addr + cutlass.Int64(t * 4), cutlass.Int32(0))
            _st_shared_b32(
                flags + tidx * 4, cutlass.select_(last, cutlass.Int32(1), cutlass.Int32(0))
            )
        self.work_barrier.arrive_and_wait()
        self._stamp(tl_addr, bidx, 5, tidx == 0)
        for tl in cutlass.range(warp, mine, 4, unroll=1):
            if _ld_shared_b32(flags + tl * 4) == 1:
                self._merge(
                    tl * split_count + split,
                    num_tokens,
                    lane,
                    ring_addr + self.merge_off + warp * self.cands * 16,
                    ids_addr,
                    wts_addr,
                    ready_addr,
                    cand_addr,
                )
                self._stamp(tl_addr, bidx, 6, lane == 0)

    @cute.jit
    def _balanced_id(self, t, num_tokens, lane):
        """Lane k < 16: expert k of token t in the benchmark's Balanced routing, the ascending sort of
        (g % ep) * per_rank + (g // ep) % per_rank over g = (dp_rank * T + t) * 16 + 0 .. 15."""
        dp_rank, ep = self.balanced
        per_rank = NUM_EXPERTS // ep
        g = (dp_rank * num_tokens + t) * TOP_K + lane
        e = cutlass.select_(
            lane < TOP_K, (g % ep) * per_rank + (g // ep) % per_rank, cutlass.Int32(0x7FFFFFFF)
        )
        rank = cutlass.Int32(0)
        for j in cutlass.range_constexpr(TOP_K):
            ej = _shfl_idx(e, cutlass.Int32(j))
            below = (ej < e) | ((ej == e) & (lane > j))
            rank = rank + cutlass.select_(below, cutlass.Int32(1), cutlass.Int32(0))
        out = cutlass.Int32(0)
        for j in cutlass.range_constexpr(TOP_K):
            ej = _shfl_idx(e, cutlass.Int32(j))
            out = cutlass.select_(_shfl_idx(rank, cutlass.Int32(j)) == lane, ej, out)
        return out

    @cute.jit
    def _stale(self, tidx, xs_chunk, num_tokens):
        """Nonzero when one of this thread's pieces of the staged chunk still holds the -0 sentinel."""
        per_row = self.kc // 8
        stale = cutlass.Int32(0)
        for p in cutlass.range_constexpr(self.pieces // self.work_threads):
            q = tidx + p * self.work_threads
            row = q // per_row
            v = _ld_shared_b32(xs_chunk + (row * self.kc + (q % per_row) * 8) * 2)
            if ((v & 0xFFFF) == 0x8000) & (row < num_tokens):
                stale = stale | 1
        return stale

    @cute.jit
    def _refresh_x(self, tidx, c0, xin, xs_base, x_addr, split, num_tokens):
        """Chunks c0 .. x_stages - 1 were loaded at entry, before the producer stored them: wait for this thread's
        stale pieces of chunk c0, then re-copy all its pieces of those chunks per round trip until none is stale."""
        per_row = self.kc // 8
        xs_chunk = xs_base + c0 * self.xs_chunk_bytes
        for p in cutlass.range_constexpr(self.pieces // self.work_threads):
            q = tidx + p * self.work_threads
            row = q // per_row
            kk = (q % per_row) * 8
            v = _ld_shared_b32(xs_chunk + (row * self.kc + kk) * 2)
            if ((v & 0xFFFF) == 0x8000) & (row < num_tokens):
                k_base = (split * self.x_chunks + c0) * self.kc
                _poll_v4(x_addr + cutlass.Int64((row * HIDDEN + k_base + kk) * 2))
        # The entry loads of the later chunks must land before the copies overwrite their slots.
        for c in cutlass.range(c0 + 1, self.x_stages, unroll=1):
            cute.arch.mbarrier_wait(xin + c, 0)
        stale = cutlass.Int32(1)
        while stale != 0:
            for c in cutlass.range(c0, self.x_stages, unroll=1):
                k_base = (split * self.x_chunks + c) * self.kc
                for p in cutlass.range_constexpr(self.pieces // self.work_threads):
                    q = tidx + p * self.work_threads
                    row = q // per_row
                    kk = (q % per_row) * 8
                    if row < num_tokens:
                        _cp_async_16(
                            xs_base + c * self.xs_chunk_bytes + (row * self.kc + kk) * 2,
                            x_addr + cutlass.Int64((row * HIDDEN + k_base + kk) * 2),
                        )
            _cp_async_wait_all()
            stale = cutlass.Int32(0)
            for c in cutlass.range(c0, self.x_stages, unroll=1):
                stale = stale | self._stale(tidx, xs_base + c * self.xs_chunk_bytes, num_tokens)

    @cute.jit
    def _merge(self, t, num_tokens, lane, table, ids_addr, wts_addr, ready_addr, cand_addr):
        """One warp: token t's top-16 as a merge of the tiles' descending candidate lists (lane i owns tile i's list
        in the warp's shared table [tile][position] of (key, 65535 - expert, score) and pops its head when it wins a
        round), then the stock normalization; lane 0 stores ids / bf16 weights with 16 B stores and raises ready[t]
        (release) when ``ready_addr`` is set."""
        m = self.m_tile
        live = lane < self.e_tiles
        if live:
            row = cand_addr + cutlass.Int64((t * self.cands + lane * TOP_K) * 16)
            for k in cutlass.range_constexpr(TOP_K):
                _st_shared_v4(table + (lane * TOP_K + k) * 16, _ld_global_cg_v4(row + 16 * k))
        _syncwarp()
        head_s = cutlass.Int32(INT_MIN)
        head_l = cutlass.Int32(-1)
        if live:
            head_s = _ld_shared_b32(table + lane * TOP_K * 16)
            head_l = _ld_shared_b32(table + lane * TOP_K * 16 + 4)
        pos = cutlass.Int32(0)
        my_rec = cutlass.Int32(0)
        for k in cutlass.range_constexpr(TOP_K):
            hs = _redux_max(head_s)
            hl = _redux_max(cutlass.select_(head_s == hs, head_l, cutlass.Int32(-1)))
            win = (65535 - hl) // m
            my_rec = cutlass.select_(lane == k, win * TOP_K + _shfl_idx(pos, win), my_rec)
            if cutlass.const_expr(k < TOP_K - 1):
                if lane == win:
                    pos = pos + 1
                    head_s = cutlass.Int32(INT_MIN)
                    head_l = cutlass.Int32(-1)
                    if pos < TOP_K:
                        head_s = _ld_shared_b32(table + (lane * TOP_K + pos) * 16)
                        head_l = _ld_shared_b32(table + (lane * TOP_K + pos) * 16 + 4)
        my_l = _ld_shared_b32(table + my_rec * 16 + 4)
        my_sig = _ld_shared_b32(table + my_rec * 16 + 8)
        if cutlass.const_expr(self.delay_ns > 0):
            if t == 0:
                spin_ns(cutlass.Int64(self.delay_ns))
        score = cutlass.select_(lane < TOP_K, _bits_f32(my_sig), cutlass.Float32(0.0))
        norm = score
        for off in (16, 8, 4, 2, 1):
            norm = _add_rn(norm, _shfl_bfly_f32(norm, off))
        norm = _bits_f32(_shfl_idx(_f32_bits(norm), cutlass.Int32(0)))
        expert_id = 65535 - my_l
        if cutlass.const_expr(self.balanced is not None):
            expert_id = self._balanced_id(t, num_tokens, lane)
        wbits = self._weight_bits(_bits_f32(my_sig), norm)
        partner = _shfl_idx(wbits, lane + 1)
        wpair = wbits | (partner << 16)
        ids = []
        wts = []
        for kk in cutlass.range_constexpr(TOP_K):
            ids.append(_shfl_idx(expert_id, cutlass.Int32(kk)))
        for kk in cutlass.range_constexpr(TOP_K // 2):
            wts.append(_shfl_idx(wpair, cutlass.Int32(2 * kk)))
        if lane == 0:
            row_ids = ids_addr + cutlass.Int64(t * TOP_K * 4)
            row_wts = wts_addr + cutlass.Int64(t * TOP_K * 2)
            for q in cutlass.range_constexpr(4):
                _st_global_v4i(row_ids + 16 * q, ids[4 * q : 4 * q + 4])
            for q in cutlass.range_constexpr(2):
                _st_global_v4i(row_wts + 16 * q, wts[4 * q : 4 * q + 4])
            if ready_addr != 0:
                _red_release_add(ready_addr + cutlass.Int64(t * 4), cutlass.Int32(1))

    @cute.jit
    def _weight_bits(self, score, norm):
        """bf16 bits of f32(f64(score) * scale / (f64(norm) + 1e-20)). With scale 1 and norm >= 2^-10 the 1e-20 is
        below half an ulp of norm in f64 and rounding the f64 quotient to f32 equals the f32 quotient (53 >= 2 * 24
        + 2), so a single div.rn.f32 gives the same bits."""
        out = cutlass.Int32(0)
        if cutlass.const_expr(self.scale_bits == 0x3FF0000000000000):
            if norm >= cutlass.Float32(2.0**-10):
                out = _bf16_bits(_div_rn(score, norm))
            else:
                out = _route_weight_bits(score, norm, self.scale_bits)
        else:
            out = _route_weight_bits(score, norm, self.scale_bits)
        return out


FGS_WAITS = {"run": ("mix_router",)}
FGS_PUBLISHES = {"run": ("router_send",)}


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def n_mma_for(tokens: int) -> int:
    return next(n for n in N_MMA if tokens <= n)


def default_split(tokens: int) -> int:
    """K slices per expert tile: the split-K partition of the stock cuBLASLt kernel for this token count, so the
    logits are bitwise equal to it (7 slices of 1024 up to 16 tokens, 8 of 896 above; one token runs a cuBLAS GEMV,
    which no K partition reproduces)."""
    return 7 if tokens <= 16 else 8


class Workspace:
    """Per-device scratch, allocated once: tile candidates [T, 14 x 16, 4] int32 and the per-token merge counters
    (zero at rest; every launch returns them to zero)."""

    def __init__(self, device: torch.device):
        self.cand = torch.empty(MAX_TOKENS * CANDS * 4, dtype=torch.int32, device=device)
        self.counters = torch.zeros(MAX_TOKENS, dtype=torch.int32, device=device)


_workspaces = {}
_compiled = {}


def workspace(device: torch.device) -> Workspace:
    ws = _workspaces.get(device.index)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("router workspace must be created before CUDA graph capture")
        ws = _workspaces[device.index] = Workspace(device)
    return ws


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    routed_scaling_factor: float = 1.0,
    fgs=None,
    enable_pdl: bool = None,
    logits: torch.Tensor = None,
    split: int = None,
    timeline: torch.Tensor = None,
    smem_budget: int = None,
    balanced: tuple = None,
) -> None:
    """ids [T, 16] int32, weights [T, 16] bf16 = Kimi K3 noaux_tc top-16 of x [T, 7168] bf16 with the fp32 gate
    weight [896, 7168] and e_score_correction_bias [896] fp32; T <= 128. One kernel on the current stream, no
    allocation after the first call per device.

    ``fgs`` (``moe_chain.StageSync``): ``waits["mix_router"]`` makes x the -0-armed slot of a running producer: the
    activation loads start without the grid dependency and every 16 B piece whose first element is -0 is re-read
    until its producer has stored it (``diag="oracle"`` / ``hazard_nowait`` skip the check, ``diag="ready"`` keeps
    the grid dependency). ``publishes["router_send"].tensor`` (int32 [>= T], zero-armed by the owner)
    gets +1 per token with release semantics after the token's ids / weights are stored; its ``hazard_delay_ns``
    delays token 0's row (hazard tests). ``logits`` (fp32 [T, 896]) receives the router logits and ``timeline``
    (int32 [CTAs, 8]) per-CTA %globaltimer_lo stamps of the kernel's phases (tests only). ``smem_budget`` (or
    ``TLLM_K3_ROUTER_SMEM_KB``) caps the shared memory per CTA, so the kernel can share SMs with a concurrent one.
    ``balanced=(dp_rank, ep_size)`` (benchmarks only) stores the layer-wise benchmark's Balanced expert ids instead
    of the selected ones, with the selected tokens' weights, before ``router_send``."""
    tokens = x.shape[0]
    if not 1 <= tokens <= MAX_TOKENS:
        raise ValueError(f"router serves 1 .. {MAX_TOKENS} tokens, got {tokens}")
    if x.dtype != torch.bfloat16 or tuple(x.shape) != (tokens, HIDDEN) or not x.is_contiguous():
        raise ValueError(f"x must be contiguous bf16 [T, {HIDDEN}]")
    if (
        weight.dtype != torch.float32
        or tuple(weight.shape) != (NUM_EXPERTS, HIDDEN)
        or not weight.is_contiguous()
    ):
        raise ValueError(f"weight must be contiguous fp32 [{NUM_EXPERTS}, {HIDDEN}]")
    if bias.dtype != torch.float32 or bias.numel() != NUM_EXPERTS:
        raise ValueError(f"bias must be fp32 [{NUM_EXPERTS}]")
    if ids.dtype != torch.int32 or tuple(ids.shape) != (tokens, TOP_K) or not ids.is_contiguous():
        raise ValueError("ids must be contiguous int32 [T, 16]")
    if (
        weights.dtype != torch.bfloat16
        or tuple(weights.shape) != (tokens, TOP_K)
        or not weights.is_contiguous()
    ):
        raise ValueError("weights must be contiguous bf16 [T, 16]")
    for name, t in (("ids", ids), ("weights", weights)):
        if t.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")
    if logits is not None and (
        logits.dtype != torch.float32 or tuple(logits.shape) != (tokens, NUM_EXPERTS)
    ):
        raise ValueError("logits must be fp32 [T, 896]")
    ready, delay_ns, x_check, x_wait = None, 0, False, True
    if fgs is not None:
        unknown = (set(fgs.waits) - set(FGS_WAITS["run"])) | (
            set(fgs.publishes) - set(FGS_PUBLISHES["run"])
        )
        if unknown:
            raise ValueError(f"router: no side for {sorted(unknown)}")
        wait = fgs.waits.get("mix_router")
        if wait is not None:
            if wait.tensor.data_ptr() != x.data_ptr():
                raise ValueError("router: mix_router waits on x itself (the armed slot)")
            x_check = not (wait.diag == "oracle" or wait.hazard_nowait)
            x_wait = wait.diag == "ready"
        edge = fgs.publishes.get("router_send")
        if edge is not None:
            ready, delay_ns = edge.tensor, edge.hazard_delay_ns
            if ready.dtype != torch.int32 or ready.numel() < tokens or not ready.is_contiguous():
                raise ValueError("router_send needs a contiguous int32 counter per token")
    if balanced is not None:
        balanced = (int(balanced[0]), int(balanced[1]))
        if balanced[1] < 1 or NUM_EXPERTS % balanced[1] or balanced[0] < 0:
            raise ValueError(
                f"balanced routing needs (dp_rank >= 0, ep_size dividing {NUM_EXPERTS})"
            )
    pdl = _use_pdl() if enable_pdl is None else enable_pdl
    split = split or default_split(tokens)
    ws = workspace(x.device)
    n = n_mma_for(tokens)
    key = (
        n,
        split,
        pdl,
        logits is not None,
        delay_ns,
        routed_scaling_factor,
        timeline is not None,
        x_check,
        x_wait,
        smem_budget or int(os.environ.get("TLLM_K3_ROUTER_SMEM_KB", "227")) * 1024,
        balanced,
    )
    args = (
        make_ptr(cutlass.TFloat32, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        cutlass.Int32(tokens),
        cutlass.Int64(x.data_ptr()),
        cutlass.Int64(bias.data_ptr()),
        cutlass.Int64(ids.data_ptr()),
        cutlass.Int64(weights.data_ptr()),
        cutlass.Int64(0 if ready is None else ready.data_ptr()),
        cutlass.Int64(0 if logits is None else logits.data_ptr()),
        cutlass.Int64(ws.cand.data_ptr()),
        cutlass.Int64(ws.counters.data_ptr()),
        cutlass.Int64(0 if timeline is None else timeline.data_ptr()),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled = _compiled.get(key)
    if compiled is None:
        compiled = _compiled[key] = cute.compile(RouterKernel(*key), *args, stream)
    compiled(*args, stream)
