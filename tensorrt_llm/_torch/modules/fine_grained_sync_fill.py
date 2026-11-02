# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded-grid CuTe DSL fill of a fine-grained-sync sentinel word.

Each thread issues aligned 128-bit stores of one repeated 32-bit word over a
grid-stride loop, so the grid size bounds how many SMs a reset can occupy
independently of the buffer size. Sizes and the grid are runtime values: one
compile per (threads, unroll) serves every buffer.
"""

from typing import Callable, Dict, List, Sequence, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import make_ptr

_VECTOR_BYTES = 16
# Spans one multi-span launch fills; longer span lists take several launches.
MAX_SPANS = 4
_COMPILED: Dict[Tuple[int, int], Callable] = {}
_COMPILED_SPANS: Dict[Tuple[int, int], Callable] = {}
_DUMMY: Dict[int, torch.Tensor] = {}


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


@cute.jit
def _fill_span(
    m_words: cute.Tensor,
    num_vecs: cutlass.Int64,
    unrolled_end: cutlass.Int64,
    rounds: cutlass.Int32,
    word: cutlass.Int32,
    first: cutlass.Int64,
    stride: cutlass.Int64,
    unroll: cutlass.Constexpr,
):
    words = cutlass.make_array_view(m_words)
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


@cute.kernel
def _fill_spans_kernel(
    m0: cute.Tensor,
    n0: cutlass.Int64,
    e0: cutlass.Int64,
    r0: cutlass.Int32,
    w0: cutlass.Int32,
    m1: cute.Tensor,
    n1: cutlass.Int64,
    e1: cutlass.Int64,
    r1: cutlass.Int32,
    w1: cutlass.Int32,
    m2: cute.Tensor,
    n2: cutlass.Int64,
    e2: cutlass.Int64,
    r2: cutlass.Int32,
    w2: cutlass.Int32,
    m3: cute.Tensor,
    n3: cutlass.Int64,
    e3: cutlass.Int64,
    r3: cutlass.Int32,
    w3: cutlass.Int32,
    num_ctas: cutlass.Int32,
    threads: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    stride = cutlass.Int64(num_ctas) * cutlass.Int64(threads)
    first = cutlass.Int64(bidx) * cutlass.Int64(threads) + cutlass.Int64(tidx)
    _fill_span(m0, n0, e0, r0, w0, first, stride, unroll)
    _fill_span(m1, n1, e1, r1, w1, first, stride, unroll)
    _fill_span(m2, n2, e2, r2, w2, first, stride, unroll)
    _fill_span(m3, n3, e3, r3, w3, first, stride, unroll)


@cute.jit
def _fill_spans_host(
    p0: cute.Pointer,
    c0: cutlass.Int32,
    n0: cutlass.Int64,
    e0: cutlass.Int64,
    r0: cutlass.Int32,
    w0: cutlass.Int32,
    p1: cute.Pointer,
    c1: cutlass.Int32,
    n1: cutlass.Int64,
    e1: cutlass.Int64,
    r1: cutlass.Int32,
    w1: cutlass.Int32,
    p2: cute.Pointer,
    c2: cutlass.Int32,
    n2: cutlass.Int64,
    e2: cutlass.Int64,
    r2: cutlass.Int32,
    w2: cutlass.Int32,
    p3: cute.Pointer,
    c3: cutlass.Int32,
    n3: cutlass.Int64,
    e3: cutlass.Int64,
    r3: cutlass.Int32,
    w3: cutlass.Int32,
    num_ctas: cutlass.Int32,
    stream: cuda.CUstream,
    threads: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
):
    _fill_spans_kernel(
        cute.make_tensor(p0, cute.make_layout((c0,), stride=(1,))),
        n0,
        e0,
        r0,
        w0,
        cute.make_tensor(p1, cute.make_layout((c1,), stride=(1,))),
        n1,
        e1,
        r1,
        w1,
        cute.make_tensor(p2, cute.make_layout((c2,), stride=(1,))),
        n2,
        e2,
        r2,
        w2,
        cute.make_tensor(p3, cute.make_layout((c3,), stride=(1,))),
        n3,
        e3,
        r3,
        w3,
        num_ctas,
        threads,
        unroll,
    ).launch(grid=(num_ctas, 1, 1), block=(threads, 1, 1), stream=stream)


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


def _dummy_words() -> torch.Tensor:
    device = torch.cuda.current_device()
    dummy = _DUMMY.get(device)
    if dummy is None:
        dummy = _DUMMY[device] = torch.empty(4, dtype=torch.int32, device="cuda")
    return dummy


def _span_args(address: int, num_vecs: int, chunk: int, word: int) -> List:
    rounds = num_vecs // chunk
    return [
        _words_ptr(address),
        cutlass.Int32(max(4, num_vecs * 4)),
        cutlass.Int64(num_vecs),
        cutlass.Int64(rounds * chunk),
        cutlass.Int32(rounds),
        cutlass.Int32(word),
    ]


def compile_sentinel_fill_spans(threads: int = 256, unroll: int = 4) -> Callable:
    """Compile (once per configuration) the multi-span fill; call it before graph capture."""
    key = (threads, unroll)
    fn = _COMPILED_SPANS.get(key)
    if fn is None:
        if threads % 32 or not 32 <= threads <= 1024:
            raise ValueError(f"threads must be a warp multiple in [32, 1024], got {threads}")
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = _span_args(_dummy_words().data_ptr(), 0, 1, 0) * MAX_SPANS
        with BaseDSL.enable_pyir():
            fn = cute.compile(_fill_spans_host, *args, cutlass.Int32(1), stream, threads, unroll)
        _COMPILED_SPANS[key] = fn
    return fn


def sentinel_fill_spans_(
    spans: Sequence[Tuple[torch.Tensor, int]],
    max_ctas: int = 1 << 30,
    threads: int = 256,
    unroll: int = 4,
) -> None:
    """Fill each contiguous, 16-byte-aligned ``(tensor, word)`` span; ``MAX_SPANS`` per launch.

    The launch's grid is sized by the spans' total vectors (capped at ``max_ctas``)
    and every CTA strides over every span. Byte tails fall back to ``torch``.
    """
    vector_spans = []
    for tensor, word in spans:
        if not tensor.is_contiguous() or tensor.data_ptr() % _VECTOR_BYTES:
            raise ValueError("sentinel_fill_spans_ needs contiguous, 16-byte-aligned tensors")
        nbytes = tensor.numel() * tensor.element_size()
        num_vecs = nbytes // _VECTOR_BYTES
        if num_vecs * 4 >= 2**31:
            raise ValueError(f"sentinel_fill_spans_ supports up to 8 GiB per span, got {nbytes}")
        if num_vecs:
            vector_spans.append((tensor.data_ptr(), num_vecs, word))
        if nbytes % _VECTOR_BYTES:
            sentinel_fill_(
                tensor.view(-1)[num_vecs * _VECTOR_BYTES // tensor.element_size() :], word
            )
    if not vector_spans:
        return
    fn = compile_sentinel_fill_spans(threads, unroll)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    for start in range(0, len(vector_spans), MAX_SPANS):
        group = vector_spans[start : start + MAX_SPANS]
        total = sum(num_vecs for _, num_vecs, _ in group)
        num_ctas = max(1, min(max_ctas, -(-total // (threads * unroll))))
        chunk = num_ctas * threads * unroll
        args = []
        for address, num_vecs, word in group:
            args += _span_args(address, num_vecs, chunk, word)
        for _ in range(MAX_SPANS - len(group)):
            args += _span_args(_dummy_words().data_ptr(), 0, chunk, 0)
        fn(*args, cutlass.Int32(num_ctas), stream)
