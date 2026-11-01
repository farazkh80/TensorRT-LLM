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

``fgs_producer`` publishes the output to a fine-grained-sync consumer that validates each 16-B chunk by its first
element (bf16 -0.0 = not written yet): every 16-B chunk is one ``st.global.v4``, -0.0 is stored as +0.0, and the
grid triggers its dependents on entry and waits for its own inputs (PDL). The output buffer is the framework's
sentinel-initialized slot; the kernel never writes sentinels.

``fgs_consumer`` reads o and g from fine-grained-sync slots instead of waiting for their producers: no
``griddepcontrol.wait``; each 16-B chunk is loaded with ``ld.global.cg`` and, while any of its elements is the -0.0
sentinel, reloaded by the warp-uniform wait of ``mla_fgs.poll_chunks``. Both modes can be on together.
"""

from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import make_ptr

from tensorrt_llm._utils import get_sm_version

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


try:
    from .mla_fgs import STATS_LEN, bf16x2_pzo, ld_cg_v8, load_chunks, poll_chunks, st_global_v4
except ImportError:
    from mla_fgs import STATS_LEN, bf16x2_pzo, ld_cg_v8, load_chunks, poll_chunks, st_global_v4


def _store_out(out_arr, out_base, idx, res, vec, producer, hw_pzo):
    if cutlass.const_expr(producer):
        # One 16-B store per chunk: the consumer checks only the chunk's first element.
        for c in range(vec // 8):
            w = [bf16x2_pzo(res[8 * c + 2 * p], res[8 * c + 2 * p + 1], hw_pzo) for p in range(4)]
            addr = out_base + (idx + cutlass.Int64(8 * c)) * cutlass.Int64(2)
            st_global_v4(addr, w[0], w[1], w[2], w[3])
    else:
        out_arr.store(
            cutlass.Vector.from_elements(tuple(res), dtype=cutlass.Float32).to(cutlass.BFloat16),
            idx=idx,
            vector_size=vec,
            alignment=2 * vec,
        )


def _gate_chunk(o_arr, g_arr, out_arr, out_base, idx, vec, fast, producer, hw_pzo):
    o = o_arr.load(idx, vec, alignment=2 * vec).to(cutlass.Float32)
    g = g_arr.load(idx, vec, alignment=2 * vec).to(cutlass.Float32)
    res = []
    for e in range(vec):
        res.append(o[e] * _sigmoid_bf16_as_f32(g[e], fast))
    _store_out(out_arr, out_base, idx, res, vec, producer, hw_pzo)


@cute.jit
def _gate_consumer(
    mO: cute.Tensor,
    mG: cute.Tensor,
    mOut: cute.Tensor,
    mStats: cute.Tensor,
    out_base: cutlass.Int64,
    first: cutlass.Int64,
    num_vecs: cutlass.Int64,
    vec: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    fast: cutlass.Constexpr,
    producer: cutlass.Constexpr,
    hw_pzo: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    with_stats: cutlass.Constexpr,
):
    out_arr = cutlass.make_array_view(mOut)
    cpv: cutlass.Constexpr = vec // 8
    n: cutlass.Constexpr = 2 * unroll * cpv
    words = cute.make_rmem_tensor((4 * n,), cutlass.Int32)
    addrs = cute.make_rmem_tensor((n,), cutlass.Int64)
    o_base = mO.iterator.toint()
    g_base = mG.iterator.toint()
    last = num_vecs - cutlass.Int64(1)
    # Lanes past the end re-read the last vector (it is published like any other) and store nothing, so the
    # warp-uniform wait needs no per-lane predicate.
    for u in cutlass.range_constexpr(unroll):
        v = cutlass.min(first + cutlass.Int64(u * THREADS), last)
        for c in cutlass.range_constexpr(cpv):
            off = (v * cutlass.Int64(vec) + cutlass.Int64(8 * c)) * cutlass.Int64(2)
            addrs[(2 * u) * cpv + c] = o_base + off
            addrs[(2 * u + 1) * cpv + c] = g_base + off
    if cutlass.const_expr(cpv == 2):
        # 32-byte accesses load both chunks of a vector at once, as the default build does.
        for k in cutlass.range_constexpr(n // 2):
            w = ld_cg_v8(addrs[2 * k])
            for i in cutlass.range_constexpr(8):
                words[8 * k + i] = w[i]
    else:
        load_chunks(words, addrs, n, cutlass.Boolean(True))
    poll_chunks(words, addrs, n, False, spin_limit, mStats, with_stats)
    for u in cutlass.range_constexpr(unroll):
        v = first + cutlass.Int64(u * THREADS)
        if v < num_vecs:
            ow = []
            gw = []
            for i in cutlass.range_constexpr(4 * cpv):
                ow.append(words[4 * (2 * u) * cpv + i])
                gw.append(words[4 * (2 * u + 1) * cpv + i])
            o = cutlass.Vector.from_elements(tuple(ow), cutlass.Int32).bitcast(cutlass.BFloat16)
            g = cutlass.Vector.from_elements(tuple(gw), cutlass.Int32).bitcast(cutlass.BFloat16)
            res = []
            for e in cutlass.range_constexpr(vec):
                res.append(
                    o[e].to(cutlass.Float32) * _sigmoid_bf16_as_f32(g[e].to(cutlass.Float32), fast)
                )
            _store_out(out_arr, out_base, v * cutlass.Int64(vec), res, vec, producer, hw_pzo)


@cute.kernel
def _output_gate_kernel(
    mO: cute.Tensor,
    mG: cute.Tensor,
    mOut: cute.Tensor,
    mStats: cute.Tensor,
    num_vecs: cutlass.Int64,
    vec: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    fast: cutlass.Constexpr,
    producer: cutlass.Constexpr,
    hw_pzo: cutlass.Constexpr,
    consumer: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    with_stats: cutlass.Constexpr,
):
    if cutlass.const_expr(producer):
        cute.arch.griddepcontrol_launch_dependents()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    out_base = cutlass.Int64(0)
    if cutlass.const_expr(producer):
        out_base = mOut.iterator.toint()
    first = cutlass.Int64(bidx) * cutlass.Int64(THREADS * unroll) + cutlass.Int64(tidx)
    if cutlass.const_expr(consumer):
        _gate_consumer(
            mO, mG, mOut, mStats, out_base, first, num_vecs, vec, unroll, fast, producer, hw_pzo,
            spin_limit, with_stats,
        )  # fmt: skip
    else:
        o_arr = cutlass.make_array_view(mO)
        g_arr = cutlass.make_array_view(mG)
        out_arr = cutlass.make_array_view(mOut)
        if cutlass.const_expr(producer):
            cute.arch.griddepcontrol_wait()
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
                _store_out(out_arr, out_base, idx, res, vec, producer, hw_pzo)
        else:
            for u in cutlass.range_constexpr(unroll):
                v = first + cutlass.Int64(u * THREADS)
                if v < num_vecs:
                    _gate_chunk(
                        o_arr,
                        g_arr,
                        out_arr,
                        out_base,
                        v * cutlass.Int64(vec),
                        vec,
                        fast,
                        producer,
                        hw_pzo,
                    )


@cute.jit
def _output_gate_host(
    o_ptr: cute.Pointer,
    g_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
    stream: cuda.CUstream,
    num_vecs: cutlass.Int64,
    num_ctas: cutlass.Int32,
    vec: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    fast: cutlass.Constexpr,
    producer: cutlass.Constexpr,
    hw_pzo: cutlass.Constexpr,
    consumer: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    with_stats: cutlass.Constexpr,
):
    layout = cute.make_layout(
        (num_ctas, THREADS * unroll * vec), stride=(THREADS * unroll * vec, 1)
    )
    _output_gate_kernel(
        cute.make_tensor(o_ptr, layout),
        cute.make_tensor(g_ptr, layout),
        cute.make_tensor(out_ptr, layout),
        cute.make_tensor(stats_ptr, cute.make_layout((STATS_LEN,))),
        num_vecs,
        vec,
        unroll,
        fast,
        producer,
        hw_pzo,
        consumer,
        spin_limit,
        with_stats,
    ).launch(
        grid=(num_ctas, 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        use_pdl=producer or consumer,
    )


# (elements per access, accesses per thread per operand, fast sigmoid with exact fallback).
# Decode-sized inputs are latency bound and favour one narrow access per thread; prefill inputs
# stream from HBM and favour wide, unrolled accesses.
DECODE_CONFIG = (8, 1, False)
PREFILL_CONFIG = (16, 2, True)
DECODE_MAX_NUMEL = 64 * 96 * 128
# fgs_producer stores each 16-B chunk of an access separately.
PRODUCER_PREFILL_CONFIG = PREFILL_CONFIG
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
    *,
    fgs_producer: bool = False,
    fgs_consumer: bool = False,
    fgs_stats: Optional[torch.Tensor] = None,
    fgs_spin_limit: int = 0,
) -> torch.Tensor:
    """out = o * sigmoid(g), bf16, bit-matching ``o * g.sigmoid()``. ``out`` may alias ``o``.

    With ``fgs_producer`` the result is ``o * g.sigmoid()`` with -0.0 stored as +0.0, ``out`` is required (the
    consumer's slot) and must not overlap ``o`` or ``g``.
    With ``fgs_consumer`` o and g are fine-grained-sync slots (``out`` must not overlap them); ``fgs_stats``
    (int32 [4]) accumulates the wait statistics of ``mla_fgs.poll_chunks`` and ``fgs_spin_limit`` > 0 traps after
    that many reload sweeps (diagnostic).
    """
    if not supported(o, g):
        raise ValueError("mla_output_gate needs contiguous, 32B-aligned bf16 o / g of equal shape")
    if fgs_producer and out is None:
        raise ValueError("mla_output_gate: fgs_producer needs the consumer's out slot")
    if (fgs_producer or fgs_consumer) and out is not None:
        nbytes = out.numel() * out.element_size()
        for src in (o, g):
            if (
                out.data_ptr() < src.data_ptr() + nbytes
                and src.data_ptr() < out.data_ptr() + nbytes
            ):
                raise ValueError("mla_output_gate: fgs out must not overlap o or g")
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
        prefill = PRODUCER_PREFILL_CONFIG if fgs_producer else PREFILL_CONFIG
        config = DECODE_CONFIG if o.numel() <= DECODE_MAX_NUMEL else prefill
    vec, unroll, fast = config
    with_stats = fgs_consumer and fgs_stats is not None
    if with_stats and (fgs_stats.dtype != torch.int32 or fgs_stats.numel() < STATS_LEN):
        raise ValueError("fgs_stats must be an int32 tensor with at least 4 entries")
    num_vecs = o.numel() // vec
    if num_vecs == 0:
        return out
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    ptrs = _ptrs(2 * vec, o, g, out)
    stats_ptr = make_ptr(
        cutlass.Int32,
        fgs_stats.data_ptr() if with_stats else out.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    hw_pzo = fgs_producer and get_sm_version() == 107
    spin_limit = fgs_spin_limit if fgs_consumer else 0
    key = (config, fgs_producer, hw_pzo, fgs_consumer, spin_limit, with_stats)
    fn = _COMPILED.get(key)
    if fn is None:
        with BaseDSL.enable_pyir():
            fn = cute.compile(
                _output_gate_host,
                *ptrs,
                stats_ptr,
                stream,
                cutlass.Int64(1),
                cutlass.Int32(1),
                vec,
                unroll,
                fast,
                fgs_producer,
                hw_pzo,
                fgs_consumer,
                spin_limit,
                with_stats,
            )
        _COMPILED[key] = fn
    fn(
        *ptrs,
        stats_ptr,
        stream,
        cutlass.Int64(num_vecs),
        cutlass.Int32(-(-num_vecs // (THREADS * unroll))),
    )
    return out
