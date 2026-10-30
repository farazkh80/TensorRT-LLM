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
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

F32 = cutlass.Float32
BF16 = cutlass.BFloat16

HEAD_DIM = 128
VEC = 8
LANES_PER_ROW = HEAD_DIM // VEC
ROWS_PER_WARP_STEP = 32 // LANES_PER_ROW


class GatedRmsNorm:
    def __init__(self, heads: int, rows_per_thread: int, warps: int, use_pdl: bool):
        self.heads = heads
        self.rows_per_thread = rows_per_thread
        self.warps = warps
        self.use_pdl = use_pdl
        self.rows_per_cta = warps * ROWS_PER_WARP_STEP * rows_per_thread

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        z: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        eps: F32,
        stream: cuda.CUstream,
    ):
        rows = x.shape[0]
        grid = (rows + self.rows_per_cta - 1) // self.rows_per_cta
        self.kernel(x, z, weight, out, eps).launch(
            grid=[grid, 1, 1],
            block=[self.warps * 32, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        z: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        eps: F32,
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

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

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
                cute.copy(copy_x, cute.local_tile(x[r, None], (VEC,), (col // VEC,)), x_reg)
                cute.copy(copy_x, cute.local_tile(z[tok, head, None], (VEC,), (col // VEC,)), z_reg)
            x_regs.append(x_reg)
            z_regs.append(z_reg)

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

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


_TORCH_TO_CUTE = {torch.bfloat16: BF16, torch.float32: F32}


@cache
def _compile(
    heads: int, weight_dtype: torch.dtype, rows_per_thread: int, warps: int, use_pdl: bool
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
    op = GatedRmsNorm(heads, rows_per_thread, warps, use_pdl)
    return cute.compile(
        op,
        x,
        z,
        weight,
        out,
        F32(1e-5),
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


def kda_gated_rmsnorm(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
    *,
    config: tuple[int, int] | None = None,
    use_pdl: bool | None = None,
) -> torch.Tensor:
    """Sigmoid-gated RMSNorm of ``x [T*H, 128]`` with ``z [T, H, 128]``; returns ``[T*H, 128]``.

    ``out`` may be any ``[T*H, 128]`` BF16 view with 16-byte aligned rows (for example the
    caller's graph-owned core buffer), which removes the separate copy.
    """
    if x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
        raise TypeError("x and z must be BF16")
    if weight.dtype not in _TORCH_TO_CUTE or weight.shape != (HEAD_DIM,):
        raise TypeError("weight must be a [128] BF16 or FP32 tensor")
    rows = x.shape[0]
    tokens, heads, _ = z.shape
    if x.shape != (rows, HEAD_DIM) or z.shape[2] != HEAD_DIM or tokens * heads != rows:
        raise ValueError(f"x {tuple(x.shape)} does not match z {tuple(z.shape)}")
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != torch.bfloat16:
        raise ValueError("out must be a BF16 tensor shaped like x")
    for name, t in (("x", x), ("out", out)):
        if t.stride(1) != 1 or t.stride(0) % VEC or t.data_ptr() % 16:
            raise ValueError(f"{name} rows must be contiguous and 16-byte aligned")
    if z.stride(2) != 1 or z.stride(1) != HEAD_DIM or z.stride(0) % VEC or z.data_ptr() % 16:
        raise ValueError("z must have contiguous (H, 128) blocks with 16-byte aligned tokens")
    weight = weight.contiguous()
    if rows == 0:
        return out
    device_index = x.device.index
    rows_per_thread, warps = config or _pick_config(rows, device_index)
    grid = -(-rows // (warps * ROWS_PER_WARP_STEP * rows_per_thread))
    if use_pdl is None:
        use_pdl = grid < _num_sms(device_index)
    fn = _compile(heads, weight.dtype, rows_per_thread, warps, bool(use_pdl))
    fn(x, z, weight, out, eps)
    return out
