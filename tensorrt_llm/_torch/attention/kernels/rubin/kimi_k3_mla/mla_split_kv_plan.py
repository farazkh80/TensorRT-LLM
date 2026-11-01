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
"""Per-request split-KV plan of the CuTe DSL MLA decode, as one single-CTA CuTe DSL kernel.

Same plan as ``cute_dsl_mla._mla_split_kv_plan_kernel`` (and its host mirror ``plan_mla_split_kv``): every request has
``tiles = max(ceil(kv_len / kv_tile), 1)`` KV tiles; the longest request is cut into ``k`` equal pieces of
``max(ceil(longest / k), min_tiles)`` tiles for the ``k`` in [1, max_pieces] with the smallest wave-quantized makespan
``ceil(sum(splits) * rows_per_request / wave_clusters) * (piece + piece_cost)`` (ties keep the smaller ``k``), and
request i gets ``clamp(ceil(tiles_i / piece), 1, max_split)`` splits.
"""

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import make_ptr

THREADS = 256
MAX_REQUESTS = 1024
REQUESTS_PER_THREAD = MAX_REQUESTS // THREADS
WARPS = THREADS // 32
KV_TILE = 128
MIN_TILES = 4
PIECE_COST_TILES = 60
MAX_PIECES = 32
INT32_MAX = 2**31 - 1


@cute.kernel
def _split_kv_plan_kernel(
    mLens: cute.Tensor,
    mPlan: cute.Tensor,
    num_requests: cutlass.Int32,
    rows_per_request: cutlass.Int32,
    wave_clusters: cutlass.Int32,
    max_split: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // cutlass.Int32(32)
    lane = tidx - warp * cutlass.Int32(32)
    red = cutlass.Array(cutlass.Int32, WARPS, space=cutlass.AddressSpace.smem, alignment=16)

    tiles = []
    for j in cutlass.range_constexpr(REQUESTS_PER_THREAD):
        r = tidx + cutlass.Int32(j * THREADS)
        t = cutlass.Int32(0)
        if r < num_requests:
            kv = cutlass.Int32(mLens[r])
            if kv < cutlass.Int32(0):
                kv = cutlass.Int32(0)
            t = (kv + cutlass.Int32(KV_TILE - 1)) // cutlass.Int32(KV_TILE)
            if t < cutlass.Int32(1):
                t = cutlass.Int32(1)
        tiles.append(t)

    longest = tiles[0]
    for j in cutlass.range_constexpr(1, REQUESTS_PER_THREAD):
        if tiles[j] > longest:
            longest = tiles[j]
    for off in (16, 8, 4, 2, 1):
        other = cute.arch.shuffle_sync_bfly(longest, offset=off)
        if other > longest:
            longest = other
    if lane == cutlass.Int32(0):
        red[warp] = longest
    cute.arch.sync_threads()
    longest = red[0]
    for w in cutlass.range_constexpr(1, WARPS):
        if red[w] > longest:
            longest = red[w]
    cute.arch.sync_threads()

    best_piece = longest
    best_cost = cutlass.Int32(INT32_MAX)
    for pieces in cutlass.range(1, MAX_PIECES + 1, 1, unroll=1):
        piece = (longest + pieces - cutlass.Int32(1)) // pieces
        if piece < cutlass.Int32(MIN_TILES):
            piece = cutlass.Int32(MIN_TILES)
        total = cutlass.Int32(0)
        for j in cutlass.range_constexpr(REQUESTS_PER_THREAD):
            s = (tiles[j] + piece - cutlass.Int32(1)) // piece
            if s < cutlass.Int32(1):
                s = cutlass.Int32(1)
            if s > max_split:
                s = max_split
            if tidx + cutlass.Int32(j * THREADS) < num_requests:
                total = total + s
        for off in (16, 8, 4, 2, 1):
            total = total + cute.arch.shuffle_sync_bfly(total, offset=off)
        if lane == cutlass.Int32(0):
            red[warp] = total
        cute.arch.sync_threads()
        total = red[0]
        for w in cutlass.range_constexpr(1, WARPS):
            total = total + red[w]
        cute.arch.sync_threads()
        clusters = total * rows_per_request
        waves = (clusters + wave_clusters - cutlass.Int32(1)) // wave_clusters
        cost = waves * (piece + cutlass.Int32(PIECE_COST_TILES))
        if cost < best_cost:
            best_cost = cost
            best_piece = piece

    for j in cutlass.range_constexpr(REQUESTS_PER_THREAD):
        r = tidx + cutlass.Int32(j * THREADS)
        if r < num_requests:
            s = (tiles[j] + best_piece - cutlass.Int32(1)) // best_piece
            if s < cutlass.Int32(1):
                s = cutlass.Int32(1)
            if s > max_split:
                s = max_split
            mPlan[r] = s


@cute.jit
def _split_kv_plan_host(
    lens_ptr: cute.Pointer,
    plan_ptr: cute.Pointer,
    stream: cuda.CUstream,
    num_requests: cutlass.Int32,
    rows_per_request: cutlass.Int32,
    wave_clusters: cutlass.Int32,
    max_split: cutlass.Int32,
):
    layout = cute.make_layout((num_requests,), stride=(1,))
    _split_kv_plan_kernel(
        cute.make_tensor(lens_ptr, layout),
        cute.make_tensor(plan_ptr, layout),
        num_requests,
        rows_per_request,
        wave_clusters,
        max_split,
    ).launch(grid=(1, 1, 1), block=(THREADS, 1, 1), stream=stream)


_COMPILED: list[Callable] = []


def launch_mla_split_kv_plan(
    cache_seqs: torch.Tensor,
    plan: torch.Tensor,
    *,
    num_requests: int,
    rows_per_request: int,
    wave_clusters: int,
    max_split: int = 32,
) -> None:
    """Write the per-request split counts for ``cache_seqs`` into ``plan`` (one single-CTA launch)."""
    if not 0 < num_requests <= MAX_REQUESTS:
        raise ValueError(f"split plan supports 1..{MAX_REQUESTS} requests")
    for name, tensor in (("cache_seqs", cache_seqs), ("plan", plan)):
        if (
            tensor.dtype != torch.int32
            or tensor.ndim != 1
            or tensor.stride(0) != 1
            or tensor.numel() < num_requests
            or not tensor.is_cuda
        ):
            raise ValueError(f"{name} must be a contiguous CUDA int32 vector of >= {num_requests}")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    ptrs = [
        make_ptr(cutlass.Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
        for t in (cache_seqs, plan)
    ]
    if not _COMPILED:
        one = cutlass.Int32(1)
        with BaseDSL.enable_pyir():
            _COMPILED.append(cute.compile(_split_kv_plan_host, *ptrs, stream, one, one, one, one))
    _COMPILED[0](
        *ptrs,
        stream,
        cutlass.Int32(num_requests),
        cutlass.Int32(max(rows_per_request, 1)),
        cutlass.Int32(max(wave_clusters, 1)),
        cutlass.Int32(max(max_split, 1)),
    )
