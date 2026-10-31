# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA output gate: y = rmsnorm(o) * w * sigmoid(z) over 128-wide (token, head) rows.

Same math as ``rms_norm_gated_token_major(..., gate_activation="sigmoid")``: fp32 sum of
squares over the row, rstd = rsqrt(mean + eps), y = o * rstd * w * sigmoid(z), rounded once
to BF16.

o is ``[T * H, 128]`` with 16-byte aligned rows. z is a ``[T, H, 128]`` view whose
(H, 128) block is contiguous per token and whose token stride is arbitrary (a
column slice of the fused qkvg projection). Row r reads z at token r // H, head r % H.

Sixteen lanes own one row (eight BF16 channels, one 16-byte load per operand);
each thread keeps ``rows_per_thread`` rows in flight before reducing.

``fgs_consumer`` reads o from a fine-grained-sync slot instead of waiting for the
producer grid: each 16-byte chunk of o counts as written once its element 0 is not
the BF16 -0.0 sentinel (the producer writes every chunk with one 16-byte store and
never stores -0.0). Unwritten chunks are reloaded with ``ld.relaxed.gpu`` after an
exponential ``nanosleep`` backoff; after ``max_spins`` reloads a thread falls back
to ``griddepcontrol.wait`` (producer grid complete) and takes the value it then
reads. Optional ``stats`` (int32 [2]) accumulates reloads and fallbacks.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

F32 = cutlass.Float32
BF16 = cutlass.BFloat16

HEAD_DIM = 128
VEC = 8
LANES_PER_ROW = HEAD_DIM // VEC
ROWS_PER_WARP_STEP = 32 // LANES_PER_ROW
BF16_NEG_ZERO = 0x8000
SPIN_NS_MIN = 32
SPIN_NS_MAX = 2048


@dsl_user_op
def _ld_relaxed_v4(addr, reload, *, loc=None, ip=None):
    """16-byte ``ld.relaxed.gpu.global`` (not served from a stale L1 line) as four int32;
    a ``reload`` is never merged with or hoisted above an earlier load of the address."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.relaxed.gpu.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}" if reload else "=r,=r,=r,=r,l",
        has_side_effects=reload,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ld_cg_v4(addr, *, loc=None, ip=None):
    """16-byte ``ld.global.cg`` (L2 only) as four int32."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _globaltimer(*, loc=None, ip=None):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nanosleep(ns, *, loc=None, ip=None):
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


class GatedRmsNorm:
    def __init__(
        self,
        heads: int,
        rows_per_thread: int,
        warps: int,
        use_pdl: bool,
        fgs_consumer: bool = False,
        max_spins: int = 0,
        with_stats: bool = False,
        persistent_ctas: int = 0,
        min_blocks: int | None = None,
        weak_first_load: bool = False,
        with_timeline: bool = False,
    ):
        self.heads = heads
        self.rows_per_thread = rows_per_thread
        self.warps = warps
        self.use_pdl = use_pdl
        self.rows_per_cta = warps * ROWS_PER_WARP_STEP * rows_per_thread
        assert not fgs_consumer or use_pdl
        assert fgs_consumer or not (with_stats or persistent_ctas or weak_first_load)
        self.fgs_consumer = fgs_consumer
        self.max_spins = max_spins
        self.with_stats = with_stats
        # persistent_ctas > 0: at most that many CTAs, each walking row blocks in order (the
        # producer's token order). min_blocks bounds registers for the retry loop (default: a
        # full SM of threads, as the default build reaches). weak_first_load issues the first
        # load of a chunk as ld.global.cg instead of ld.relaxed.gpu; reloads stay relaxed.
        self.persistent_ctas = persistent_ctas
        if min_blocks is None:
            min_blocks = 1024 // (warps * 32) if fgs_consumer else 0
        self.min_blocks = min_blocks
        self.weak_first_load = weak_first_load
        self.with_timeline = with_timeline

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        z: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        eps: F32,
        stats: cute.Tensor | None,
        timeline: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        rows = x.shape[0]
        grid = (rows + self.rows_per_cta - 1) // self.rows_per_cta
        if cutlass.const_expr(self.persistent_ctas > 0):
            grid = cute.min(grid, self.persistent_ctas)
        self.kernel(x, z, weight, out, eps, stats, timeline).launch(
            grid=[grid, 1, 1],
            block=[self.warps * 32, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=self.min_blocks,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        z: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        eps: F32,
        stats: cute.Tensor | None,
        timeline: cute.Tensor | None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        col = (lane % LANES_PER_ROW) * VEC
        row0 = (bidx * self.warps + warp) * (ROWS_PER_WARP_STEP * self.rows_per_thread) + (
            lane // LANES_PER_ROW
        )
        rows = x.shape[0]

        copy_x = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF16, num_bits_per_copy=128)
        copy_w = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            weight.element_type,
            num_bits_per_copy=VEC * weight.element_type.width,
        )

        w_reg = cute.make_rmem_tensor((VEC,), weight.element_type)
        cute.copy(copy_w, cute.local_tile(weight, (VEC,), (col // VEC,)), w_reg)

        if cutlass.const_expr(self.use_pdl and not self.fgs_consumer):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.with_timeline):
            if tidx == 0:
                cute.arch.atomic_min(timeline.iterator, _globaltimer())

        spins = cutlass.Int32(0)
        waited = cutlass.Int32(0)
        if cutlass.const_expr(self.persistent_ctas > 0):
            grid_x, _, _ = cute.arch.grid_dim()
            blocks = (rows + self.rows_per_cta - 1) // self.rows_per_cta
            for block in cutlass.range(bidx, blocks, grid_x):
                block_row0 = (block * self.warps + warp) * (
                    ROWS_PER_WARP_STEP * self.rows_per_thread
                ) + (lane // LANES_PER_ROW)
                spins, waited = self._block(
                    x, z, out, eps, w_reg, copy_x, block_row0, col, spins, waited
                )
        else:
            spins, waited = self._block(x, z, out, eps, w_reg, copy_x, row0, col, spins, waited)

        if cutlass.const_expr(self.with_stats):
            if spins > 0:
                cute.arch.atomic_add(stats.iterator, spins)
            if waited > 0:
                cute.arch.atomic_add(stats.iterator + 1, waited)
        if cutlass.const_expr(self.with_timeline):
            cute.arch.barrier()
            if tidx == 0:
                cute.arch.atomic_max(timeline.iterator + 1, _globaltimer())
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _block(
        self,
        x: cute.Tensor,
        z: cute.Tensor,
        out: cute.Tensor,
        eps: F32,
        w_reg: cute.Tensor,
        copy_x,
        row0,
        col,
        spins,
        waited,
    ):
        """This thread's rows from ``row0``; returns the updated (reloads, fallback) counts."""
        rows = x.shape[0]
        x_regs = []
        z_regs = []
        for i in cutlass.range_constexpr(self.rows_per_thread):
            r = row0 + i * ROWS_PER_WARP_STEP
            x_reg = cute.make_rmem_tensor((VEC,), BF16)
            z_reg = cute.make_rmem_tensor((VEC,), BF16)
            x_reg.fill(0.0)
            z_reg.fill(0.0)
            if r < rows:
                tok = r // self.heads
                head = r - tok * self.heads
                x_chunk = cute.local_tile(x[r, None], (VEC,), (col // VEC,))
                if cutlass.const_expr(self.fgs_consumer):
                    x_bits = cute.recast_tensor(x_reg, cutlass.Int32)
                    if cutlass.const_expr(self.weak_first_load):
                        words = _ld_cg_v4(x_chunk.iterator.toint())
                    else:
                        words = _ld_relaxed_v4(x_chunk.iterator.toint(), False)
                    for w in cutlass.range_constexpr(4):
                        x_bits[w] = words[w]
                else:
                    cute.copy(copy_x, x_chunk, x_reg)
                cute.copy(copy_x, cute.local_tile(z[tok, head, None], (VEC,), (col // VEC,)), z_reg)
            x_regs.append(x_reg)
            z_regs.append(z_reg)

        if cutlass.const_expr(self.fgs_consumer):
            for i in cutlass.range_constexpr(self.rows_per_thread):
                r = row0 + i * ROWS_PER_WARP_STEP
                if r < rows:
                    x_bits = cute.recast_tensor(x_regs[i], cutlass.Int32)
                    x_addr = cute.local_tile(x[r, None], (VEC,), (col // VEC,)).iterator.toint()
                    # After a fallback wait a sentinel loaded before it is reloaded once.
                    pending = (x_bits[0] & 0xFFFF) == BF16_NEG_ZERO
                    sleep_ns = cutlass.Int32(SPIN_NS_MIN)
                    while pending:
                        if waited == 0:
                            if spins >= self.max_spins:
                                # The producer grid has completed: its value is final.
                                cute.arch.griddepcontrol_wait()
                                waited = cutlass.Int32(1)
                            else:
                                _nanosleep(sleep_ns)
                                sleep_ns = cute.min(sleep_ns * 2, cutlass.Int32(SPIN_NS_MAX))
                        spins = spins + 1
                        words = _ld_relaxed_v4(x_addr, True)
                        for w in cutlass.range_constexpr(4):
                            x_bits[w] = words[w]
                        pending = ((x_bits[0] & 0xFFFF) == BF16_NEG_ZERO) & (waited == 0)

        for i in cutlass.range_constexpr(self.rows_per_thread):
            r = row0 + i * ROWS_PER_WARP_STEP
            sum_sq = F32(0.0)
            for c in cutlass.range_constexpr(VEC):
                v = x_regs[i][c].to(F32)
                sum_sq += v * v
            sum_sq = cute.arch.warp_reduction_sum(sum_sq, threads_in_group=LANES_PER_ROW)
            rstd = cute.math.rsqrt(sum_sq * (1.0 / HEAD_DIM) + eps)
            y_reg = cute.make_rmem_tensor((VEC,), BF16)
            for c in cutlass.range_constexpr(VEC):
                gate = cute.math.rcp(
                    cute.math.exp(-z_regs[i][c].to(F32), fastmath=True) + 1.0,
                    approx=True,
                    ftz=True,
                )
                y = x_regs[i][c].to(F32) * rstd * w_reg[c].to(F32) * gate
                y_reg[c] = y.to(BF16)
            if r < rows:
                cute.copy(copy_x, y_reg, cute.local_tile(out[r, None], (VEC,), (col // VEC,)))
        return spins, waited


_TORCH_TO_CUTE = {torch.bfloat16: BF16, torch.float32: F32}


@cache
def _compile(
    heads: int,
    weight_dtype: torch.dtype,
    rows_per_thread: int,
    warps: int,
    use_pdl: bool,
    fgs_consumer: bool = False,
    max_spins: int = 0,
    with_stats: bool = False,
    persistent_ctas: int = 0,
    min_blocks: int | None = None,
    weak_first_load: bool = False,
    with_timeline: bool = False,
):
    rows, tokens = cute.sym_int(), cute.sym_int()
    x = make_fake_tensor(
        BF16, (rows, HEAD_DIM), stride=(cute.sym_int64(divisibility=VEC), 1), assumed_align=16
    )
    z = make_fake_tensor(
        BF16,
        (tokens, heads, HEAD_DIM),
        stride=(cute.sym_int64(divisibility=VEC), HEAD_DIM, 1),
        assumed_align=16,
    )
    w_dtype = _TORCH_TO_CUTE[weight_dtype]
    weight = make_fake_tensor(
        w_dtype, (HEAD_DIM,), stride=(1,), assumed_align=VEC * w_dtype.width // 8
    )
    out = make_fake_tensor(
        BF16, (rows, HEAD_DIM), stride=(cute.sym_int64(divisibility=VEC), 1), assumed_align=16
    )
    stats = None
    if with_stats:
        stats = make_fake_tensor(cutlass.Int32, (2,), stride=(1,), assumed_align=8)
    timeline = None
    if with_timeline:
        timeline = make_fake_tensor(cutlass.Int64, (2,), stride=(1,), assumed_align=16)
    op = GatedRmsNorm(
        heads,
        rows_per_thread,
        warps,
        use_pdl,
        fgs_consumer,
        max_spins,
        with_stats,
        persistent_ctas,
        min_blocks,
        weak_first_load,
        with_timeline,
    )
    return cute.compile(
        op,
        x,
        z,
        weight,
        out,
        F32(1e-5),
        stats,
        timeline,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _pick_config(rows: int, device_index: int) -> tuple[int, int]:
    """(rows_per_thread, warps): enough CTAs to cover every SM, deep rows when large."""
    if rows >= 64 * _num_sms(device_index) * 4:
        return 4, 4
    if rows >= 16 * _num_sms(device_index):
        return 2, 4
    return 1, 2


def _rows_ok(t: torch.Tensor) -> bool:
    return (
        t.dtype == torch.bfloat16
        and t.stride(1) == 1
        and t.stride(0) % VEC == 0
        and t.data_ptr() % 16 == 0
    )


def supports(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> bool:
    """Whether ``kda_gated_rmsnorm`` accepts these layouts (otherwise it raises)."""
    if x.ndim != 2 or z.ndim != 3 or x.shape[1] != HEAD_DIM or z.shape[2] != HEAD_DIM:
        return False
    if (
        z.shape[0] * z.shape[1] != x.shape[0]
        or weight.dtype not in _TORCH_TO_CUTE
        or weight.shape != (HEAD_DIM,)
    ):
        return False
    if not _rows_ok(x) or (out is not None and (out.shape != x.shape or not _rows_ok(out))):
        return False
    return (
        z.dtype == torch.bfloat16
        and z.stride(2) == 1
        and z.stride(1) == HEAD_DIM
        and z.stride(0) % VEC == 0
        and z.data_ptr() % 16 == 0
    )


def kda_gated_rmsnorm(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
    *,
    config: tuple[int, int] | None = None,
    use_pdl: bool | None = None,
    fgs_consumer: bool = False,
    fgs_max_spins: int = 1 << 16,
    fgs_stats: torch.Tensor | None = None,
    fgs_ctas: int | None = None,
    fgs_timeline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sigmoid-gated RMSNorm of ``x [T*H, 128]`` with ``z [T, H, 128]``; returns ``[T*H, 128]``.

    ``out`` may be any ``[T*H, 128]`` BF16 view with 16-byte aligned rows (for example the
    caller's graph-owned core buffer), which removes the separate copy.

    ``fgs_consumer`` reads ``x`` as a fine-grained-sync slot (see the module docstring):
    the kernel launches with PDL and does not wait for the producer grid; ``x`` must be
    armed with -0.0 before the producer runs and ``out`` must not alias it.
    ``fgs_stats`` (int32 [2], zeroed by the caller) accumulates reloads and fallbacks.
    ``fgs_ctas`` caps the grid at that many CTAs, each walking row blocks in token order,
    so the consumer can trail the producer on the SMs it leaves free instead of filling
    every SM with waiting CTAs. ``fgs_timeline`` (int64 [2], caller sets [0] to INT64_MAX
    and [1] to 0) receives the first CTA start and the last CTA end (``%globaltimer`` ns).
    """
    if not supports(x, z, weight, out):
        raise ValueError(
            "kda_gated_rmsnorm needs BF16 x/out [T*H, 128] and z [T, H, 128] with 16-byte aligned rows"
        )
    if out is None:
        out = torch.empty_like(x)
    rows = x.shape[0]
    heads = z.shape[1]
    if rows == 0:
        return out
    device_index = x.device.index
    rows_per_thread, warps = config or _pick_config(rows, device_index)
    grid = -(-rows // (warps * ROWS_PER_WARP_STEP * rows_per_thread))
    if use_pdl is None:
        use_pdl = grid < _num_sms(device_index)
    if fgs_consumer:
        if out.data_ptr() == x.data_ptr():
            raise ValueError("kda_gated_rmsnorm: fgs_consumer needs out separate from x")
        use_pdl = True
    with_stats = fgs_consumer and fgs_stats is not None
    if with_stats and (fgs_stats.dtype != torch.int32 or fgs_stats.numel() < 2):
        raise ValueError("kda_gated_rmsnorm: fgs_stats must be an int32 tensor of 2 elements")
    with_timeline = fgs_timeline is not None
    if with_timeline and (fgs_timeline.dtype != torch.int64 or fgs_timeline.numel() < 2):
        raise ValueError("kda_gated_rmsnorm: fgs_timeline must be an int64 tensor of 2 elements")
    fn = _compile(
        heads,
        weight.dtype,
        rows_per_thread,
        warps,
        bool(use_pdl),
        fgs_consumer,
        fgs_max_spins if fgs_consumer else 0,
        with_stats,
        (fgs_ctas or 0) if fgs_consumer else 0,
        None,
        False,
        with_timeline,
    )
    fn(x, z, weight, out, eps, fgs_stats if with_stats else None, fgs_timeline)
    return out
