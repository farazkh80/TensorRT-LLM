# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from FlashInfer flashinfer/kda_kernels/fused_kda_decode.py (v0.6.18),
# Copyright (c) 2026 by FlashInfer team, licensed under the Apache License 2.0.
"""Fused Kimi K3 KDA T=1 decode on the TensorRT-LLM cache pools (CuTe DSL).

One launch per decode step does what ``trtllm::kda_decode`` does with
``update_conv_cache``, lower-bound gate, beta sigmoid and the output gate:

  conv4 + conv-cache shift, SiLU (fp32), q/k L2 norm (eps 1e-6, q scaled by
  128^-0.5), decay = exp(lb * sigmoid(exp(A_log) * (g + dt_bias))),
  beta = sigmoid(b), delta rule on the fp32 V-first state
  (h = S * decay; u = (v - h.k) * beta; S = h + k u; o = S.q), and
  y = o * rsqrt(mean(o^2) + eps) * w * sigmoid(z) stored as BF16.

Pools are the live TRT-LLM ones, addressed by ``state_indices`` (any slot >= 0
is live, including 0; a negative slot skips both cache updates and writes zeros):
  conv_state [slots, 3 * H * 128, 3] BF16, width innermost, oldest tap first;
  state      [slots, H, 128 (V), 128 (K)] FP32.

Grid (H * v_split, B) with ``v_split`` CTAs per (request, head) forming a
cluster. Each CTA owns 128 / v_split state rows; 16 / v_split warps hold eight
complete rows each in registers (lane l owns keys 4l..4l+3). With v_split > 1
the RMSNorm sum of squares is exchanged through distributed shared memory, and
only cluster rank 0 writes the conv cache.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

F32 = cutlass.Float32
BF16 = cutlass.BFloat16

D = 128
ROWS_PER_WARP = 8
CONV_ITEMS = 3 * D // 4
SETUP_ITEMS = CONV_ITEMS + D
L2_EPS = 1.0e-6


def _aligned(tensor, alignment):
    ptr = tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(ptr.dtype, ptr.toint(), ptr.memspace, assumed_align=alignment),
        tensor.layout,
    )


@dsl_user_op
def _load_peer_f32(smem_addr, peer, *, loc=None, ip=None):
    """Load the f32 at this CTA's shared address ``smem_addr`` from cluster rank ``peer``."""
    return F32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip),
                cutlass.Int32(peer).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .u32 ra; mapa.shared::cluster.u32 ra, $1, $2; ld.shared::cluster.f32 $0, [ra]; }",
            "=f,r,r",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _sigmoid(x):
    return 1.0 / (1.0 + cute.math.exp(-x, fastmath=True))


class KdaDecodeT1:
    def __init__(
        self,
        heads: int,
        v_split: int,
        lower_bound: float,
        q_scale: float,
        norm_eps: float,
        early_state: bool,
        use_pdl: bool,
        stage_smem: bool,
        min_blocks: int,
    ):
        assert v_split in (1, 2, 4)
        self.heads = heads
        self.v_split = v_split
        self.warps = 16 // v_split
        self.threads = self.warps * 32
        self.rows_per_cta = D // v_split
        self.lower_bound = lower_bound
        self.q_scale = q_scale
        self.norm_eps = norm_eps
        self.early_state = early_state
        self.use_pdl = use_pdl
        self.stage_smem = stage_smem
        # Resident CTAs per SM requested from the register allocator (0: no bound).
        self.min_blocks = min_blocks
        self.smem_bytes = 4096 + (self.rows_per_cta * D * 4 if stage_smem else 0)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        conv_state: cute.Tensor,
        raw_gate: cute.Tensor,
        raw_beta: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state_indices: cute.Tensor,
        state: cute.Tensor,
        output_gate: cute.Tensor,
        norm_weight: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        H = self.heads
        d = H * D
        rows = state_indices.shape[0]
        # [elem, lane, section(q/k/v), head, row]
        x_v = cute.make_tensor(
            x.iterator, cute.make_layout((4, 32, 3, H, rows), stride=(1, 4, d, D, x.stride[0]))
        )
        # [elem, lane, tap, section, head]
        w_v = cute.make_tensor(
            weight.iterator,
            cute.make_layout((4, 32, 4, 3, H), stride=(1, 4, d, 4 * d, D)),
        )
        # Twelve contiguous BF16 per lane: four channels x three taps, as three 4-vectors.
        cs_v = cute.make_tensor(
            conv_state.iterator,
            cute.make_layout(
                (4, 3, 32, 3, H, conv_state.shape[0]),
                stride=(1, 4, 12, 3 * d, 3 * D, conv_state.stride[0]),
            ),
        )
        g_v = cute.make_tensor(
            raw_gate.iterator, cute.make_layout((D, H, rows), stride=(1, D, raw_gate.stride[0]))
        )
        dt_v = cute.make_tensor(dt_bias.iterator, cute.make_layout((D, H), stride=(1, D)))
        b_v = cute.make_tensor(
            raw_beta.iterator, cute.make_layout((H, rows), stride=(1, raw_beta.stride[0]))
        )
        # [elem, lane, v row, head, slot]
        st_v = cute.make_tensor(
            state.iterator,
            cute.make_layout(
                (4, 32, D, H, state.shape[0]), stride=(1, 4, D, D * D, state.stride[0])
            ),
        )
        og_v = cute.make_tensor(
            output_gate.iterator,
            cute.make_layout((D, H, rows), stride=(1, D, output_gate.stride[0])),
        )
        out_v = cute.make_tensor(
            output.iterator,
            cute.make_layout((4, 32, H, rows), stride=(1, 4, D, output.stride[0])),
        )
        cluster = [self.v_split, 1, 1] if self.v_split > 1 else None
        self.kernel(
            x_v, w_v, cs_v, g_v, b_v, A_log, dt_v, state_indices, st_v, og_v, norm_weight, out_v
        ).launch(
            grid=[H * self.v_split, rows, 1],
            block=[self.threads, 1, 1],
            cluster=cluster,
            smem=self.smem_bytes,
            stream=stream,
            min_blocks_per_mp=self.min_blocks,
            use_pdl=self.use_pdl,
        )

    def _load_state(self, state, state_s, state_regs, lane, warp, row_base, head, slot):
        if self.stage_smem:
            copy_g2s = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
                F32,
                num_bits_per_copy=128,
            )
        else:
            copy_g2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), F32, num_bits_per_copy=128)
        for r in range(ROWS_PER_WARP):
            local_row = warp * ROWS_PER_WARP + r
            src = _aligned(state[(None, lane, row_base + local_row, head, slot)], 16)
            if self.stage_smem:
                cute.copy(copy_g2s, src, _aligned(state_s[(None, lane, local_row)], 16))
            else:
                cute.copy(copy_g2r, src, state_regs[(None, r)])
        if self.stage_smem:
            cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        conv_state: cute.Tensor,
        raw_gate: cute.Tensor,
        raw_beta: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state_indices: cute.Tensor,
        state: cute.Tensor,
        output_gate: cute.Tensor,
        norm_weight: cute.Tensor,
        output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, row, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        warp = cute.arch.make_warp_uniform(warp)
        head = bidx // self.v_split
        rank = bidx - head * self.v_split
        row_base = rank * self.rows_per_cta

        smem = SmemAllocator()
        mixed = smem.allocate_tensor(F32, cute.make_layout((3 * D,)), byte_alignment=16)
        rec_out = smem.allocate_tensor(F32, cute.make_layout((D,)), byte_alignment=16)
        decay_s = smem.allocate_tensor(F32, cute.make_layout((D,)), byte_alignment=16)
        gate_s = smem.allocate_tensor(F32, cute.make_layout((D,)), byte_alignment=16)
        normw_s = smem.allocate_tensor(F32, cute.make_layout((D,)), byte_alignment=16)
        scalars = smem.allocate_tensor(F32, cute.make_layout((4,)), byte_alignment=16)

        copy_f32x4 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), F32, num_bits_per_copy=128)
        copy_bf16x4 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF16, num_bits_per_copy=64)

        slot_req = state_indices[row]
        is_live = slot_req >= 0
        slot = cute.max(slot_req, cutlass.Int32(0))

        # Each thread stages its own eight state row slices (registers, or shared memory
        # through cp.async so they stay in flight during the setup stage without
        # holding registers); only the issuing thread reads them back.
        state_regs = cute.make_rmem_tensor((4, ROWS_PER_WARP), F32)
        state_s = None
        if cutlass.const_expr(self.stage_smem):
            state_s = smem.allocate_tensor(
                F32,
                cute.make_layout((4, 32, self.rows_per_cta), stride=(1, 4, D)),
                byte_alignment=16,
            )
        if cutlass.const_expr(self.early_state):
            self._load_state(state, state_s, state_regs, lane, warp, row_base, head, slot)
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(not self.early_state):
            self._load_state(state, state_s, state_regs, lane, warp, row_base, head, slot)

        # Setup: 96 conv items (4 channels of q, k or v each) and 128 gate items.
        for it in cutlass.range_constexpr((SETUP_ITEMS + self.threads - 1) // self.threads):
            item = tidx + it * self.threads
            if item < CONV_ITEMS:
                cl = item % 32
                sec = item // 32
                w_regs = cute.make_rmem_tensor((4, 4), F32)
                for tap in cutlass.range_constexpr(4):
                    cute.copy(
                        copy_f32x4,
                        _aligned(weight[(None, cl, tap, sec, head)], 16),
                        w_regs[(None, tap)],
                    )
                hist = cute.make_rmem_tensor((4, 3), BF16)
                for j in cutlass.range_constexpr(3):
                    cute.copy(
                        copy_bf16x4,
                        _aligned(conv_state[(None, j, cl, sec, head, slot)], 8),
                        hist[(None, j)],
                    )
                cur = cute.make_rmem_tensor((4,), BF16)
                cute.copy(copy_bf16x4, _aligned(x[(None, cl, sec, head, row)], 8), cur)
                silu = cute.make_rmem_tensor((4,), F32)
                new_hist = cute.make_rmem_tensor((4, 3), BF16)
                for e in cutlass.range_constexpr(4):
                    # Lane-local element (channel e, tap t) sits at 3 * e + t.
                    acc = hist[(3 * e) % 4, (3 * e) // 4].to(F32) * w_regs[e, 0]
                    acc = acc + hist[(3 * e + 1) % 4, (3 * e + 1) // 4].to(F32) * w_regs[e, 1]
                    acc = acc + hist[(3 * e + 2) % 4, (3 * e + 2) // 4].to(F32) * w_regs[e, 2]
                    acc = acc + cur[e].to(F32) * w_regs[e, 3]
                    silu[e] = acc * _sigmoid(acc)
                    new_hist[(3 * e) % 4, (3 * e) // 4] = hist[(3 * e + 1) % 4, (3 * e + 1) // 4]
                    new_hist[(3 * e + 1) % 4, (3 * e + 1) // 4] = hist[
                        (3 * e + 2) % 4, (3 * e + 2) // 4
                    ]
                    new_hist[(3 * e + 2) % 4, (3 * e + 2) // 4] = cur[e]
                cute.copy(
                    copy_f32x4,
                    silu,
                    _aligned(
                        cute.make_tensor(
                            mixed.iterator + (sec * D + 4 * cl), cute.make_layout((4,))
                        ),
                        16,
                    ),
                )
                if is_live:
                    if rank == 0:
                        for j in cutlass.range_constexpr(3):
                            cute.copy(
                                copy_bf16x4,
                                new_hist[(None, j)],
                                _aligned(conv_state[(None, j, cl, sec, head, slot)], 8),
                            )
            elif item < SETUP_ITEMS:
                c = item - CONV_ITEMS
                a = cute.math.exp(A_log[head], fastmath=True)
                g = raw_gate[c, head, row].to(F32) + dt_bias[c, head]
                decay_s[c] = cute.math.exp(self.lower_bound * _sigmoid(a * g), fastmath=True)
                gate_s[c] = _sigmoid(output_gate[c, head, row].to(F32))
                normw_s[c] = norm_weight[c]
                if c == 0:
                    scalars[0] = _sigmoid(raw_beta[head, row].to(F32))

        cute.arch.barrier()

        q = cute.make_rmem_tensor((4,), F32)
        k = cute.make_rmem_tensor((4,), F32)
        decay = cute.make_rmem_tensor((4,), F32)
        for e in cutlass.range_constexpr(4):
            q[e] = mixed[4 * lane + e]
            k[e] = mixed[D + 4 * lane + e]
            decay[e] = decay_s[4 * lane + e]
        q_sq = F32(0.0)
        k_sq = F32(0.0)
        for e in cutlass.range_constexpr(4):
            q_sq += q[e] * q[e]
            k_sq += k[e] * k[e]
        q_sq = cute.arch.warp_reduction_sum(q_sq)
        k_sq = cute.arch.warp_reduction_sum(k_sq)
        q_factor = cute.math.rsqrt(q_sq + L2_EPS) * self.q_scale
        k_factor = cute.math.rsqrt(k_sq + L2_EPS)
        for e in cutlass.range_constexpr(4):
            q[e] = q[e] * q_factor
            k[e] = k[e] * k_factor
        beta = scalars[0]

        if cutlass.const_expr(self.stage_smem):
            cute.arch.cp_async_wait_group(0)
        for r in cutlass.range_constexpr(ROWS_PER_WARP):
            local_row = warp * ROWS_PER_WARP + r
            v_row = row_base + local_row
            if cutlass.const_expr(self.stage_smem):
                cute.autovec_copy(state_s[(None, lane, local_row)], state_regs[(None, r)])
            hk = F32(0.0)
            for e in cutlass.range_constexpr(4):
                state_regs[e, r] = state_regs[e, r] * decay[e]
                hk += state_regs[e, r] * k[e]
            hk = cute.arch.warp_reduction_sum(hk)
            u = (mixed[2 * D + v_row] - hk) * beta
            hq = F32(0.0)
            for e in cutlass.range_constexpr(4):
                state_regs[e, r] = state_regs[e, r] + k[e] * u
                hq += state_regs[e, r] * q[e]
            if is_live:
                cute.copy(
                    copy_f32x4,
                    state_regs[(None, r)],
                    _aligned(state[(None, lane, v_row, head, slot)], 16),
                )
            hq = cute.arch.warp_reduction_sum(hq)
            if lane == 0:
                rec_out[local_row] = hq

        cute.arch.barrier()

        # Gated RMSNorm: lane l of warp 0 owns this CTA's rows 4l..4l+3.
        lanes_used = self.rows_per_cta // 4
        o_regs = cute.make_rmem_tensor((4,), F32)
        sum_sq = F32(0.0)
        if tidx < 32:
            for e in cutlass.range_constexpr(4):
                o_regs[e] = F32(0.0)
                if lane < lanes_used:
                    o_regs[e] = rec_out[4 * lane + e]
                sum_sq += o_regs[e] * o_regs[e]
            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
        if cutlass.const_expr(self.v_split > 1):
            if tidx == 0:
                scalars[1] = sum_sq
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if tidx < 32:
                total = F32(0.0)
                for peer in cutlass.range_constexpr(self.v_split):
                    total += _load_peer_f32((scalars.iterator + 1).toint(), peer)
                sum_sq = total
        if tidx < lanes_used:
            rstd = cute.math.rsqrt(sum_sq * (1.0 / D) + self.norm_eps)
            y = cute.make_rmem_tensor((4,), BF16)
            for e in cutlass.range_constexpr(4):
                ch = row_base + 4 * lane + e
                y[e] = (o_regs[e] * rstd * normw_s[ch] * gate_s[ch]).to(BF16)
                if slot_req < 0:
                    y[e] = BF16(0.0)
            cute.copy(
                copy_bf16x4,
                y,
                _aligned(output[(None, rank * lanes_used + lane, head, row)], 8),
            )
        if cutlass.const_expr(self.v_split > 1):
            # Keep this CTA's shared memory alive until every peer has read it.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


@cache
def _compile(
    heads: int,
    v_split: int,
    lower_bound: float,
    q_scale: float,
    norm_eps: float,
    early_state: bool,
    use_pdl: bool,
    stage_smem: bool,
    min_blocks: int,
):
    d = heads * D
    rows, slots, conv_slots = cute.sym_int(), cute.sym_int(), cute.sym_int()

    def compact(dtype, shape, align=16):
        return make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=align
        )

    args = (
        make_fake_tensor(
            BF16, (rows, 3 * d), stride=(cute.sym_int64(divisibility=4), 1), assumed_align=8
        ),
        compact(F32, (3, 4, d)),
        make_fake_tensor(
            BF16,
            (conv_slots, 3 * d, 3),
            stride=(cute.sym_int64(divisibility=4), 3, 1),
            assumed_align=8,
        ),
        make_fake_tensor(BF16, (rows, d), stride=(cute.sym_int64(), 1), assumed_align=2),
        make_fake_tensor(BF16, (rows, heads), stride=(cute.sym_int64(), 1), assumed_align=2),
        compact(F32, (heads,), 4),
        compact(F32, (d,)),
        compact(cutlass.Int32, (rows,), 4),
        make_fake_tensor(
            F32,
            (slots, heads, D, D),
            stride=(cute.sym_int64(divisibility=4), D * D, D, 1),
            assumed_align=16,
        ),
        make_fake_tensor(BF16, (rows, heads, D), stride=(cute.sym_int64(), D, 1), assumed_align=2),
        compact(F32, (D,)),
        make_fake_tensor(
            BF16, (rows, heads, D), stride=(cute.sym_int64(divisibility=4), D, 1), assumed_align=8
        ),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    op = KdaDecodeT1(
        heads, v_split, lower_bound, q_scale, norm_eps, early_state, use_pdl, stage_smem, min_blocks
    )
    return cute.compile(op, *args, options="--enable-tvm-ffi")


@cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def pick_v_split(batch: int, heads: int, device_index: int) -> int:
    """Split V rows across a cluster until the grid covers the SMs."""
    sms = _num_sms(device_index)
    for v_split in (1, 2):
        if batch * heads * v_split >= sms:
            return v_split
    return 4


def pack_conv_weight(w_q_t: torch.Tensor, w_k_t: torch.Tensor, w_v_t: torch.Tensor) -> torch.Tensor:
    """``trtllm::kda_decode``'s tap-major BF16 ``[4, H*128]`` conv weights -> FP32 ``[3, 4, H*128]``."""
    return torch.stack((w_q_t, w_k_t, w_v_t)).float().contiguous()


def kda_decode_t1(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
    *,
    lower_bound: float,
    norm_eps: float,
    scale: float = D**-0.5,
    v_split: int | None = None,
    stage_smem: bool = True,
    min_blocks: int | None = None,
    early_state: bool = False,
    use_pdl: bool = True,
) -> torch.Tensor:
    """Fused T=1 KDA decode; updates ``conv_state`` and ``state`` rows in place.

    x [B, 3*H*128] (q | k | v projection columns, any row stride), conv_weight FP32
    [3, 4, H*128] (``pack_conv_weight``), conv_state BF16 [slots, 3*H*128, 3] with
    strides (s, 3, 1), raw_gate BF16 [B, H*128], raw_beta BF16 [B, H] (any row stride),
    A_log FP32 [H], dt_bias FP32 [H*128], state_indices int32 [B], state FP32
    [slots, H, 128, 128], output_gate BF16 [B, H, 128] (any token stride),
    norm_weight FP32 [128], output BF16 [B, H, 128].
    """
    B, three_d = x.shape
    heads = three_d // (3 * D)
    d = heads * D
    if B == 0:
        return output
    if lower_bound is None or not lower_bound < 0:
        raise ValueError("kda_decode_t1 implements the negative lower-bound gate only")
    checks = (
        (
            x.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.stride(0) % 4 == 0
            and x.data_ptr() % 8 == 0,
            "x",
        ),
        (
            conv_weight.dtype == torch.float32
            and conv_weight.shape == (3, 4, d)
            and conv_weight.is_contiguous(),
            "conv_weight",
        ),
        (
            conv_state.dtype == torch.bfloat16
            and conv_state.shape[1:] == (3 * d, 3)
            and conv_state.stride()[1:] == (3, 1)
            and conv_state.stride(0) % 4 == 0
            and conv_state.data_ptr() % 8 == 0,
            "conv_state",
        ),
        (
            raw_gate.dtype == torch.bfloat16
            and raw_gate.shape == (B, d)
            and raw_gate.stride(1) == 1,
            "raw_gate",
        ),
        (
            raw_beta.dtype == torch.bfloat16
            and raw_beta.shape == (B, heads)
            and raw_beta.stride(1) == 1,
            "raw_beta",
        ),
        (
            A_log.dtype == torch.float32 and A_log.shape == (heads,) and A_log.is_contiguous(),
            "A_log",
        ),
        (
            dt_bias.dtype == torch.float32 and dt_bias.shape == (d,) and dt_bias.is_contiguous(),
            "dt_bias",
        ),
        (
            state_indices.dtype == torch.int32
            and state_indices.shape == (B,)
            and state_indices.is_contiguous(),
            "state_indices",
        ),
        (
            state.dtype == torch.float32
            and state.shape[1:] == (heads, D, D)
            and state.stride()[1:] == (D * D, D, 1)
            and state.stride(0) % 4 == 0
            and state.data_ptr() % 16 == 0,
            "state",
        ),
        (
            output_gate.dtype == torch.bfloat16
            and output_gate.shape == (B, heads, D)
            and output_gate.stride()[1:] == (D, 1),
            "output_gate",
        ),
        (
            norm_weight.dtype == torch.float32
            and norm_weight.shape == (D,)
            and norm_weight.is_contiguous(),
            "norm_weight",
        ),
        (
            output.dtype == torch.bfloat16
            and output.shape == (B, heads, D)
            and output.stride()[1:] == (D, 1)
            and output.stride(0) % 4 == 0
            and output.data_ptr() % 8 == 0,
            "output",
        ),
    )
    for ok, name in checks:
        if not ok:
            raise ValueError(f"kda_decode_t1: unsupported {name} layout or dtype")
    if v_split is None:
        v_split = pick_v_split(B, heads, x.device.index)
    if min_blocks is None:
        min_blocks = 1024 // (512 // v_split)
    fn = _compile(
        heads,
        v_split,
        float(lower_bound),
        float(scale),
        float(norm_eps),
        early_state,
        use_pdl,
        stage_smem,
        min_blocks,
    )
    fn(
        x,
        conv_weight,
        conv_state,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_indices,
        state,
        output_gate,
        norm_weight,
        output,
    )
    return output
