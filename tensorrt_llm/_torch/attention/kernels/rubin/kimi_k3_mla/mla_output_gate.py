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
"""Kimi K3 MLA output gate ``o * sigmoid(g)`` as one CuTe DSL elementwise kernel.

Rounding matches ATen's ``o * g.sigmoid()`` (two kernels):
s = bf16(1 / (1 + expf(-float(g)))) with IEEE expf / division, then out = bf16(float(o) * float(s)).
The product of two bf16 values is exact in fp32, so the only rounding steps are the two bf16 casts.

With ``fast`` the sigmoid is first evaluated with ex2.approx / rcp.approx. For g > -80 that value is within
2^-17 (relative) of the IEEE one, about 60 fp32 ulps, so both round to the same bf16 unless the fast value lies
within 256 ulps of a bf16 rounding midpoint; those elements (about 0.8%), g <= -80 (the rcp.approx flush range)
and NaN fall back to the IEEE formula. The bf16 result is therefore identical to the exact path.
"""

from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import make_ptr

THREADS = 256
LOG2E = 1.4426950408889634
MIDPOINT_GUARD_ULPS = 256


@cute.jit
def _sigmoid_bf16_as_f32(g: cutlass.Float32, fast: cutlass.Constexpr) -> cutlass.Float32:
    s = cutlass.Float32(0.0)
    if cutlass.const_expr(fast):
        s = cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute.math.exp2(g * cutlass.Float32(-LOG2E), fastmath=True)
        )
        bits = cutlass.Vector.from_elements((s,), cutlass.Float32).bitcast(cutlass.Int32)[0]
        off = (bits & cutlass.Int32(0xFFFF)) - cutlass.Int32(0x8000)
        if (
            g <= cutlass.Float32(-80.0)
            or g != g
            or (
                off <= cutlass.Int32(MIDPOINT_GUARD_ULPS)
                and off >= cutlass.Int32(-MIDPOINT_GUARD_ULPS)
            )
        ):
            s = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-g, fastmath=False))
    else:
        s = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-g, fastmath=False))
    return s.to(cutlass.BFloat16).to(cutlass.Float32)


def _gate_chunk(o_arr, g_arr, out_arr, idx, vec, fast):
    o = o_arr.load(idx, vec, alignment=2 * vec).to(cutlass.Float32)
    g = g_arr.load(idx, vec, alignment=2 * vec).to(cutlass.Float32)
    res = []
    for e in range(vec):
        res.append(o[e] * _sigmoid_bf16_as_f32(g[e], fast))
    out_arr.store(
        cutlass.Vector.from_elements(tuple(res), dtype=cutlass.Float32).to(cutlass.BFloat16),
        idx=idx,
        vector_size=vec,
        alignment=2 * vec,
    )


@cute.kernel
def _output_gate_kernel(
    mO: cute.Tensor,
    mG: cute.Tensor,
    mOut: cute.Tensor,
    num_vecs: cutlass.Int64,
    vec: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    fast: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    o_arr = cutlass.make_array_view(mO)
    g_arr = cutlass.make_array_view(mG)
    out_arr = cutlass.make_array_view(mOut)
    first = cutlass.Int64(bidx) * cutlass.Int64(THREADS * unroll) + cutlass.Int64(tidx)
    if cutlass.Int64(bidx + 1) * cutlass.Int64(THREADS * unroll) <= num_vecs:
        # Full CTA: issue every load of the thread before any math so `unroll` vectors per operand are in flight.
        os = []
        gs = []
        for u in cutlass.range_constexpr(unroll):
            idx = (first + cutlass.Int64(u * THREADS)) * cutlass.Int64(vec)
            os.append(o_arr.load(idx, vec, alignment=2 * vec))
            gs.append(g_arr.load(idx, vec, alignment=2 * vec))
        for u in cutlass.range_constexpr(unroll):
            idx = (first + cutlass.Int64(u * THREADS)) * cutlass.Int64(vec)
            o = os[u].to(cutlass.Float32)
            g = gs[u].to(cutlass.Float32)
            res = []
            for e in cutlass.range_constexpr(vec):
                res.append(o[e] * _sigmoid_bf16_as_f32(g[e], fast))
            out_arr.store(
                cutlass.Vector.from_elements(tuple(res), dtype=cutlass.Float32).to(
                    cutlass.BFloat16
                ),
                idx=idx,
                vector_size=vec,
                alignment=2 * vec,
            )
    else:
        for u in cutlass.range_constexpr(unroll):
            v = first + cutlass.Int64(u * THREADS)
            if v < num_vecs:
                _gate_chunk(o_arr, g_arr, out_arr, v * cutlass.Int64(vec), vec, fast)


@cute.jit
def _output_gate_host(
    o_ptr: cute.Pointer,
    g_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    stream: cuda.CUstream,
    num_vecs: cutlass.Int64,
    num_ctas: cutlass.Int32,
    vec: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    fast: cutlass.Constexpr,
):
    layout = cute.make_layout(
        (num_ctas, THREADS * unroll * vec), stride=(THREADS * unroll * vec, 1)
    )
    _output_gate_kernel(
        cute.make_tensor(o_ptr, layout),
        cute.make_tensor(g_ptr, layout),
        cute.make_tensor(out_ptr, layout),
        num_vecs,
        vec,
        unroll,
        fast,
    ).launch(grid=(num_ctas, 1, 1), block=(THREADS, 1, 1), stream=stream)


# (elements per access, accesses per thread per operand, fast sigmoid with exact fallback).
# Decode-sized inputs are latency bound and favour one narrow access per thread; prefill inputs
# stream from HBM and favour wide, unrolled accesses.
DECODE_CONFIG = (8, 1, False)
PREFILL_CONFIG = (16, 2, True)
DECODE_MAX_NUMEL = 64 * 96 * 128
_COMPILED: dict[tuple, Callable] = {}


def _ptrs(align, *tensors):
    return [
        make_ptr(cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)
        for t in tensors
    ]


def supported(o: torch.Tensor, g: torch.Tensor) -> bool:
    return (
        o.dtype == torch.bfloat16
        and g.dtype == torch.bfloat16
        and o.shape == g.shape
        and o.is_cuda
        and o.is_contiguous()
        and g.is_contiguous()
        and o.numel() % 16 == 0
        and o.data_ptr() % 32 == 0
        and g.data_ptr() % 32 == 0
    )


def mla_output_gate(
    o: torch.Tensor,
    g: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    config: Optional[tuple] = None,
) -> torch.Tensor:
    """out = o * sigmoid(g), bf16, bit-matching ``o * g.sigmoid()``. ``out`` may alias ``o``."""
    if not supported(o, g):
        raise ValueError("mla_output_gate needs contiguous, 32B-aligned bf16 o / g of equal shape")
    if out is None:
        out = torch.empty_like(o)
    elif not (
        out.shape == o.shape
        and out.dtype == o.dtype
        and out.is_contiguous()
        and out.data_ptr() % 32 == 0
    ):
        raise ValueError("out must be a contiguous, 32B-aligned bf16 tensor shaped like o")
    if config is None:
        config = DECODE_CONFIG if o.numel() <= DECODE_MAX_NUMEL else PREFILL_CONFIG
    vec, unroll, fast = config
    num_vecs = o.numel() // vec
    if num_vecs == 0:
        return out
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    ptrs = _ptrs(2 * vec, o, g, out)
    fn = _COMPILED.get(config)
    if fn is None:
        with BaseDSL.enable_pyir():
            fn = cute.compile(
                _output_gate_host,
                *ptrs,
                stream,
                cutlass.Int64(1),
                cutlass.Int32(1),
                vec,
                unroll,
                fast,
            )
        _COMPILED[config] = fn
    fn(*ptrs, stream, cutlass.Int64(num_vecs), cutlass.Int32(-(-num_vecs // (THREADS * unroll))))
    return out
