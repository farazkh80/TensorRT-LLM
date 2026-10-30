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

Rounding follows the two ATen kernels it replaces (``g.sigmoid()`` then ``o * s``):
s = bf16(1 / (1 + expf(-float(g)))) with IEEE expf / division, then out = bf16(float(o) * float(s)).
The product of two bf16 values is exact in fp32, so the only rounding steps are the two bf16 casts.
"""

from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import make_ptr

THREADS = 256
VEC = 8  # bf16 elements per 16-byte access


@cute.kernel
def _output_gate_kernel(mO: cute.Tensor, mG: cute.Tensor, mOut: cute.Tensor, num_vecs: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    o_arr = cutlass.make_array_view(mO)
    g_arr = cutlass.make_array_view(mG)
    out_arr = cutlass.make_array_view(mOut)
    vec = cutlass.Int64(bidx) * cutlass.Int64(THREADS) + cutlass.Int64(tidx)
    if vec < num_vecs:
        idx = vec * cutlass.Int64(VEC)
        o = o_arr.load(idx, VEC, alignment=16).to(cutlass.Float32)
        g = g_arr.load(idx, VEC, alignment=16).to(cutlass.Float32)
        res = []
        for e in cutlass.range_constexpr(VEC):
            s = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-g[e], fastmath=False))
            res.append(o[e] * s.to(cutlass.BFloat16).to(cutlass.Float32))
        out_arr.store(
            cutlass.Vector.from_elements(tuple(res), dtype=cutlass.Float32).to(cutlass.BFloat16),
            idx=idx,
            vector_size=VEC,
            alignment=16,
        )


@cute.jit
def _output_gate_host(
    o_ptr: cute.Pointer,
    g_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    stream: cuda.CUstream,
    num_vecs: cutlass.Int64,
    num_ctas: cutlass.Int32,
):
    layout = cute.make_layout((num_ctas, THREADS * VEC), stride=(THREADS * VEC, 1))
    _output_gate_kernel(
        cute.make_tensor(o_ptr, layout),
        cute.make_tensor(g_ptr, layout),
        cute.make_tensor(out_ptr, layout),
        num_vecs,
    ).launch(grid=(num_ctas, 1, 1), block=(THREADS, 1, 1), stream=stream)


_COMPILED: list[Callable] = []


def _ptrs(*tensors):
    return [
        make_ptr(cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
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
        and o.numel() % VEC == 0
        and o.data_ptr() % 16 == 0
        and g.data_ptr() % 16 == 0
    )


def mla_output_gate(o: torch.Tensor, g: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """out = o * sigmoid(g), bf16, bit-matching ``o * g.sigmoid()``. ``out`` may alias ``o``."""
    if not supported(o, g):
        raise ValueError("mla_output_gate needs contiguous, 16B-aligned bf16 o / g of equal shape")
    if out is None:
        out = torch.empty_like(o)
    elif not (out.shape == o.shape and out.dtype == o.dtype and out.is_contiguous() and out.data_ptr() % 16 == 0):
        raise ValueError("out must be a contiguous, 16B-aligned bf16 tensor shaped like o")
    num_vecs = o.numel() // VEC
    if num_vecs == 0:
        return out
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    ptrs = _ptrs(o, g, out)
    if not _COMPILED:
        with BaseDSL.enable_pyir():
            _COMPILED.append(
                cute.compile(_output_gate_host, *ptrs, stream, cutlass.Int64(1), cutlass.Int32(1))
            )
    _COMPILED[0](*ptrs, stream, cutlass.Int64(num_vecs), cutlass.Int32(-(-num_vecs // THREADS)))
    return out
