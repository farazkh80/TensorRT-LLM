# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded-grid CuTe DSL fill of a fine-grained-sync sentinel word.

Each thread issues aligned 128-bit stores of one repeated 32-bit word over a
grid-stride loop, so the grid size bounds how many SMs a reset can occupy
independently of the buffer size. Sizes and the grid are runtime values: one
compile per (threads, unroll) serves every buffer.
"""

from typing import Callable, Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import make_ptr

_VECTOR_BYTES = 16
_COMPILED: Dict[Tuple[int, int], Callable] = {}


@cute.kernel
def _fill_kernel(
    m_words: cute.Tensor,
    num_vecs: cutlass.Int64,
    unrolled_end: cutlass.Int64,
    rounds: cutlass.Int32,
    word: cutlass.Int32,
    num_ctas: cutlass.Int32,
    threads: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
):
    # Every thread runs the same number of unchecked rounds below unrolled_end (a multiple of
    # stride * unroll); fewer than stride * unroll vectors remain, at most `unroll` per thread.
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    words = cutlass.make_array_view(m_words)
    stride = cutlass.Int64(num_ctas) * cutlass.Int64(threads)
    first = cutlass.Int64(bidx) * cutlass.Int64(threads) + cutlass.Int64(tidx)
    for r in cutlass.range(0, rounds, 1, unroll=1):
        round_first = cutlass.Int64(r) * stride * cutlass.Int64(unroll) + first
        fill = cutlass.Vector.from_elements((word, word, word, word), dtype=cutlass.Int32)
        for u in cutlass.range_constexpr(unroll):
            idx = (round_first + cutlass.Int64(u) * stride) * cutlass.Int64(4)
            words.store(fill, idx=idx, vector_size=4, alignment=_VECTOR_BYTES)
    fill = cutlass.Vector.from_elements((word, word, word, word), dtype=cutlass.Int32)
    for u in cutlass.range_constexpr(unroll):
        v = unrolled_end + cutlass.Int64(u) * stride + first
        if v < num_vecs:
            words.store(fill, idx=v * cutlass.Int64(4), vector_size=4, alignment=_VECTOR_BYTES)


@cute.jit
def _fill_host(
    words_ptr: cute.Pointer,
    num_words: cutlass.Int32,
    num_vecs: cutlass.Int64,
    unrolled_end: cutlass.Int64,
    rounds: cutlass.Int32,
    word: cutlass.Int32,
    num_ctas: cutlass.Int32,
    stream: cuda.CUstream,
    threads: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
):
    m_words = cute.make_tensor(words_ptr, cute.make_layout((num_words,), stride=(1,)))
    _fill_kernel(m_words, num_vecs, unrolled_end, rounds, word, num_ctas, threads, unroll).launch(
        grid=(num_ctas, 1, 1), block=(threads, 1, 1), stream=stream
    )


def _words_ptr(address: int):
    return make_ptr(cutlass.Int32, address, cute.AddressSpace.gmem, assumed_align=_VECTOR_BYTES)


def compile_sentinel_fill(threads: int = 256, unroll: int = 4) -> Callable:
    """Compile (once per configuration) the fill; call it before graph capture."""
    key = (threads, unroll)
    fn = _COMPILED.get(key)
    if fn is None:
        if threads % 32 or not 32 <= threads <= 1024:
            raise ValueError(f"threads must be a warp multiple in [32, 1024], got {threads}")
        probe = torch.empty(4, dtype=torch.int32, device="cuda")
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        with BaseDSL.enable_pyir():
            fn = cute.compile(
                _fill_host,
                _words_ptr(probe.data_ptr()),
                cutlass.Int32(4),
                cutlass.Int64(1),
                cutlass.Int64(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(1),
                stream,
                threads,
                unroll,
            )
        _COMPILED[key] = fn
    return fn


def sentinel_fill_(
    tensor: torch.Tensor, word: int, max_ctas: int = 16, threads: int = 256, unroll: int = 4
) -> torch.Tensor:
    """Fill a contiguous, 16-byte-aligned tensor with the repeated 32-bit ``word``.

    A byte tail shorter than one vector falls back to ``torch`` and must be a
    whole number of elements of the word's 16-bit or 32-bit lane.
    """
    if not tensor.is_contiguous():
        raise ValueError("sentinel_fill_ needs a contiguous tensor")
    if tensor.data_ptr() % _VECTOR_BYTES:
        raise ValueError("sentinel_fill_ needs a 16-byte-aligned tensor")
    nbytes = tensor.numel() * tensor.element_size()
    num_vecs = nbytes // _VECTOR_BYTES
    if num_vecs:
        if num_vecs * 4 >= 2**31:
            raise ValueError(f"sentinel_fill_ supports up to 8 GiB, got {nbytes} bytes")
        fn = compile_sentinel_fill(threads, unroll)
        num_ctas = max(1, min(max_ctas, -(-num_vecs // (threads * unroll))))
        chunk = num_ctas * threads * unroll
        rounds = num_vecs // chunk
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        fn(
            _words_ptr(tensor.data_ptr()),
            cutlass.Int32(num_vecs * 4),
            cutlass.Int64(num_vecs),
            cutlass.Int64(rounds * chunk),
            cutlass.Int32(rounds),
            cutlass.Int32(word),
            cutlass.Int32(num_ctas),
            stream,
        )
    tail = nbytes - num_vecs * _VECTOR_BYTES
    if tail:
        flat = tensor.view(-1).view(torch.uint8)[num_vecs * _VECTOR_BYTES :]
        lane_bytes = tensor.element_size()
        if lane_bytes not in (2, 4):
            raise ValueError(f"sentinel_fill_ supports 2- and 4-byte elements, got {lane_bytes}")
        lane_raw = word.to_bytes(4, "little", signed=True)[:lane_bytes]
        lane = int.from_bytes(lane_raw, "little", signed=True)
        flat.view(torch.int16 if lane_bytes == 2 else torch.int32).fill_(lane)
    return tensor
