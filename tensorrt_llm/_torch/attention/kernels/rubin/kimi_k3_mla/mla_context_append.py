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
"""Kimi K3 MLA context latent append: one CuTe DSL launch for a ragged context batch on a BF16 latent cache.

For the identity (NoPE) rope table, writes what ``applyMLARopeAndAssignQKVKernelOptContext`` writes besides Q:

  cache row (b, kv_len[b] - len_b + u) = [compressed_kv[t] | k_pe[t]]     t = cu_seqlens[b] + u, u < len_b
  k[t, h, nope:] = k_pe[t]                                                 every head h

Q is left alone: the stock kernel rewrites q[..., nope:] with the same values. Every write is a copy, so the result
is bit-identical to stock, except that the stock fp32 identity rotation can turn a -0.0 into +0.0.

Grid (ctas, B): one 16-byte vector per thread; per token the first latent/8 + rope/8 vectors are the cache row and
the next H * rope/8 the K rope columns.

Fine-grained sync (Lamport) modes, both opt-in and combinable:
  fgs_consumer  compressed_kv and k_pe are FGS slots: launched with PDL, no ``griddepcontrol.wait``; the step
                inputs (cu_seqlens, kv_lens, page table) are read first, then each 16-B input chunk is loaded with
                ``ld.global.cg`` and, while it holds the -0.0 sentinel, reloaded by the warp-uniform wait of
                ``mla_fgs.poll_chunks``.
  fgs_producer  K's rope columns and the cache rows are published to FGS consumers (the context FMHA): dependents
                are triggered on entry and every 16-B chunk is one ``st.global.v4`` with -0.0 stored as +0.0.
In these modes a thread of the K-rope role loads one k_pe chunk and stores it for ``FGS_HEADS_PER_THREAD`` heads.
"""

from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import make_ptr

try:
    from .mla_fgs import STATS_LEN, canon_bf16x2, load_chunks, poll_chunks, st_global_v4
    from .mla_latent_append import kv_pool_geometry
except ImportError:
    from mla_fgs import STATS_LEN, canon_bf16x2, load_chunks, poll_chunks, st_global_v4
    from mla_latent_append import kv_pool_geometry

THREADS = 256
VEC = 8  # bf16 elements per 16-byte access
FGS_HEADS_PER_THREAD = 8


@cute.jit
def _append_fgs(
    mCkv: cute.Tensor,
    mKpe: cute.Tensor,
    mK: cute.Tensor,
    mPool: cute.Tensor,
    mPt: cute.Tensor,
    mKvLens: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    mStats: cute.Tensor,
    pt_stride: cutlass.Int32,
    page_offset: cutlass.Int32,
    ckv_stride: cutlass.Int64,
    kpe_stride: cutlass.Int64,
    k_tok_stride: cutlass.Int64,
    k_head_stride: cutlass.Int64,
    pool_row_stride: cutlass.Int64,
    num_heads: cutlass.Constexpr,
    latent: cutlass.Constexpr,
    rope: cutlass.Constexpr,
    nope: cutlass.Constexpr,
    page_size: cutlass.Constexpr,
    consumer: cutlass.Constexpr,
    producer: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    with_stats: cutlass.Constexpr,
):
    """FGS mode: one 16-B source chunk per thread (a cache-row chunk, or a k_pe chunk stored for
    FGS_HEADS_PER_THREAD heads); addresses computed before the input wait, lanes without work stay converged."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, b, _ = cute.arch.block_idx()
    hpt: cutlass.Constexpr = FGS_HEADS_PER_THREAD
    rope_vecs: cutlass.Constexpr = rope // VEC
    latent_vecs: cutlass.Constexpr = latent // VEC
    row_vecs: cutlass.Constexpr = (latent + rope) // VEC
    token_vecs: cutlass.Constexpr = row_vecs + (num_heads // hpt) * rope_vecs

    gid = bidx * cutlass.Int32(THREADS) + tidx
    u = gid // cutlass.Int32(token_vecs)
    v = gid - u * cutlass.Int32(token_vecs)
    start = cutlass.Int32(mCuSeqlens[b])
    seq_len = cutlass.Int32(mCuSeqlens[b + 1]) - start
    t = start + u
    src = cutlass.Int64(0)
    dst = cutlass.Int64(0)
    row = cutlass.Boolean(False)
    active = cutlass.Boolean(False)
    if u < seq_len:
        if v < cutlass.Int32(row_vecs):
            pos = cutlass.Int32(mKvLens[b]) - seq_len + u
            if pos >= cutlass.Int32(0):
                page = (
                    cutlass.Int32(mPt[b * pt_stride + pos // cutlass.Int32(page_size)])
                    + page_offset
                )
                dst = mPool.iterator.toint() + cutlass.Int64(2) * (
                    (
                        cutlass.Int64(page) * cutlass.Int64(page_size)
                        + cutlass.Int64(pos % cutlass.Int32(page_size))
                    )
                    * pool_row_stride
                    + cutlass.Int64(v * cutlass.Int32(VEC))
                )
                if v < cutlass.Int32(latent_vecs):
                    src = mCkv.iterator.toint() + cutlass.Int64(2) * (
                        cutlass.Int64(t) * ckv_stride + cutlass.Int64(v * cutlass.Int32(VEC))
                    )
                else:
                    src = mKpe.iterator.toint() + cutlass.Int64(2) * (
                        cutlass.Int64(t) * kpe_stride
                        + cutlass.Int64((v - cutlass.Int32(latent_vecs)) * cutlass.Int32(VEC))
                    )
                row = cutlass.Boolean(True)
                active = cutlass.Boolean(True)
        else:
            w = v - cutlass.Int32(row_vecs)
            hg = w // cutlass.Int32(rope_vecs)
            j = w - hg * cutlass.Int32(rope_vecs)
            src = mKpe.iterator.toint() + cutlass.Int64(2) * (
                cutlass.Int64(t) * kpe_stride + cutlass.Int64(j * cutlass.Int32(VEC))
            )
            dst = mK.iterator.toint() + cutlass.Int64(2) * (
                cutlass.Int64(t) * k_tok_stride
                + cutlass.Int64(hg * cutlass.Int32(hpt)) * k_head_stride
                + cutlass.Int64(nope)
                + cutlass.Int64(j * cutlass.Int32(VEC))
            )
            active = cutlass.Boolean(True)
    words = cute.make_rmem_tensor((4,), cutlass.Int32)
    addrs = cute.make_rmem_tensor((1,), cutlass.Int64)
    addrs[0] = src
    if cutlass.const_expr(not consumer):
        cute.arch.griddepcontrol_wait()
    load_chunks(words, addrs, 1, active)
    if cutlass.const_expr(consumer):
        poll_chunks(words, addrs, 1, False, spin_limit, mStats, with_stats)
    if cutlass.const_expr(producer):
        for i in cutlass.range_constexpr(4):
            words[i] = canon_bf16x2(words[i])
    if active:
        if row:
            st_global_v4(dst, words[0], words[1], words[2], words[3])
        else:
            for h in cutlass.range_constexpr(hpt):
                st_global_v4(
                    dst + cutlass.Int64(2 * h) * k_head_stride,
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )


@cute.kernel
def _context_append_kernel(
    mCkv: cute.Tensor,
    mKpe: cute.Tensor,
    mK: cute.Tensor,
    mPool: cute.Tensor,
    mPt: cute.Tensor,
    mKvLens: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    mStats: cute.Tensor,
    pt_stride: cutlass.Int32,
    page_offset: cutlass.Int32,
    ckv_stride: cutlass.Int64,
    kpe_stride: cutlass.Int64,
    k_tok_stride: cutlass.Int64,
    k_head_stride: cutlass.Int64,
    pool_row_stride: cutlass.Int64,
    num_heads: cutlass.Constexpr,
    latent: cutlass.Constexpr,
    rope: cutlass.Constexpr,
    nope: cutlass.Constexpr,
    page_size: cutlass.Constexpr,
    consumer: cutlass.Constexpr,
    producer: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    with_stats: cutlass.Constexpr,
):
    if cutlass.const_expr(producer):
        cute.arch.griddepcontrol_launch_dependents()
    if cutlass.const_expr(consumer or producer):
        _append_fgs(
            mCkv, mKpe, mK, mPool, mPt, mKvLens, mCuSeqlens, mStats, pt_stride, page_offset, ckv_stride,
            kpe_stride, k_tok_stride, k_head_stride, pool_row_stride, num_heads, latent, rope, nope, page_size,
            consumer, producer, spin_limit, with_stats,
        )  # fmt: skip
    else:
        _append_default(
            mCkv, mKpe, mK, mPool, mPt, mKvLens, mCuSeqlens, pt_stride, page_offset, ckv_stride, kpe_stride,
            k_tok_stride, k_head_stride, pool_row_stride, num_heads, latent, rope, nope, page_size,
        )  # fmt: skip


@cute.jit
def _append_default(
    mCkv: cute.Tensor,
    mKpe: cute.Tensor,
    mK: cute.Tensor,
    mPool: cute.Tensor,
    mPt: cute.Tensor,
    mKvLens: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    pt_stride: cutlass.Int32,
    page_offset: cutlass.Int32,
    ckv_stride: cutlass.Int64,
    kpe_stride: cutlass.Int64,
    k_tok_stride: cutlass.Int64,
    k_head_stride: cutlass.Int64,
    pool_row_stride: cutlass.Int64,
    num_heads: cutlass.Constexpr,
    latent: cutlass.Constexpr,
    rope: cutlass.Constexpr,
    nope: cutlass.Constexpr,
    page_size: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, b, _ = cute.arch.block_idx()
    rope_vecs: cutlass.Constexpr = rope // VEC
    latent_vecs: cutlass.Constexpr = latent // VEC
    row_vecs: cutlass.Constexpr = (latent + rope) // VEC
    token_vecs: cutlass.Constexpr = row_vecs + num_heads * rope_vecs

    ckv = cutlass.make_array_view(mCkv)
    kpe = cutlass.make_array_view(mKpe)
    k_arr = cutlass.make_array_view(mK)
    pool = cutlass.make_array_view(mPool)

    gid = bidx * cutlass.Int32(THREADS) + tidx
    u = gid // cutlass.Int32(token_vecs)
    v = gid - u * cutlass.Int32(token_vecs)
    start = cutlass.Int32(mCuSeqlens[b])
    seq_len = cutlass.Int32(mCuSeqlens[b + 1]) - start
    if u < seq_len:
        t = start + u
        if v < cutlass.Int32(row_vecs):
            pos = cutlass.Int32(mKvLens[b]) - seq_len + u
            if pos >= cutlass.Int32(0):
                page = (
                    cutlass.Int32(mPt[b * pt_stride + pos // cutlass.Int32(page_size)])
                    + page_offset
                )
                dst = (
                    cutlass.Int64(page) * cutlass.Int64(page_size)
                    + cutlass.Int64(pos % cutlass.Int32(page_size))
                ) * pool_row_stride + cutlass.Int64(v * cutlass.Int32(VEC))
                if v < cutlass.Int32(latent_vecs):
                    pool.store(
                        ckv.load(
                            cutlass.Int64(t) * ckv_stride + cutlass.Int64(v * cutlass.Int32(VEC)),
                            VEC,
                            alignment=16,
                        ),
                        idx=dst,
                        vector_size=VEC,
                        alignment=16,
                    )
                else:
                    pool.store(
                        kpe.load(
                            cutlass.Int64(t) * kpe_stride
                            + cutlass.Int64((v - cutlass.Int32(latent_vecs)) * cutlass.Int32(VEC)),
                            VEC,
                            alignment=16,
                        ),
                        idx=dst,
                        vector_size=VEC,
                        alignment=16,
                    )
        else:
            w = v - cutlass.Int32(row_vecs)
            h = w // cutlass.Int32(rope_vecs)
            j = w - h * cutlass.Int32(rope_vecs)
            k_arr.store(
                kpe.load(
                    cutlass.Int64(t) * kpe_stride + cutlass.Int64(j * cutlass.Int32(VEC)),
                    VEC,
                    alignment=16,
                ),
                idx=cutlass.Int64(t) * k_tok_stride
                + cutlass.Int64(h) * k_head_stride
                + cutlass.Int64(nope)
                + cutlass.Int64(j * cutlass.Int32(VEC)),
                vector_size=VEC,
                alignment=16,
            )


@cute.jit
def _context_append_host(
    ckv_ptr: cute.Pointer,
    kpe_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    pool_ptr: cute.Pointer,
    pt_ptr: cute.Pointer,
    kv_lens_ptr: cute.Pointer,
    cu_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
    stream: cuda.CUstream,
    num_tokens: cutlass.Int32,
    batch: cutlass.Int32,
    ctas: cutlass.Int32,
    pt_stride: cutlass.Int32,
    page_offset: cutlass.Int32,
    pool_rows: cutlass.Int32,
    ckv_stride: cutlass.Int64,
    kpe_stride: cutlass.Int64,
    k_tok_stride: cutlass.Int64,
    k_head_stride: cutlass.Int64,
    pool_row_stride: cutlass.Int64,
    num_heads: cutlass.Constexpr,
    latent: cutlass.Constexpr,
    rope: cutlass.Constexpr,
    nope: cutlass.Constexpr,
    page_size: cutlass.Constexpr,
    consumer: cutlass.Constexpr,
    producer: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    with_stats: cutlass.Constexpr,
):
    mCkv = cute.make_tensor(ckv_ptr, cute.make_layout((num_tokens, latent), stride=(ckv_stride, 1)))
    mKpe = cute.make_tensor(kpe_ptr, cute.make_layout((num_tokens, rope), stride=(kpe_stride, 1)))
    mK = cute.make_tensor(
        k_ptr,
        cute.make_layout(
            (num_tokens, num_heads, nope + rope), stride=(k_tok_stride, k_head_stride, 1)
        ),
    )
    mPool = cute.make_tensor(
        pool_ptr, cute.make_layout((pool_rows, latent + rope), stride=(pool_row_stride, 1))
    )
    mPt = cute.make_tensor(pt_ptr, cute.make_layout((batch * pt_stride,), stride=(1,)))
    mKvLens = cute.make_tensor(kv_lens_ptr, cute.make_layout((batch,), stride=(1,)))
    mCuSeqlens = cute.make_tensor(cu_ptr, cute.make_layout((batch + 1,), stride=(1,)))
    mStats = cute.make_tensor(stats_ptr, cute.make_layout((STATS_LEN,), stride=(1,)))
    _context_append_kernel(
        mCkv,
        mKpe,
        mK,
        mPool,
        mPt,
        mKvLens,
        mCuSeqlens,
        mStats,
        pt_stride,
        page_offset,
        ckv_stride,
        kpe_stride,
        k_tok_stride,
        k_head_stride,
        pool_row_stride,
        num_heads,
        latent,
        rope,
        nope,
        page_size,
        consumer,
        producer,
        spin_limit,
        with_stats,
    ).launch(
        grid=(ctas, batch, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        use_pdl=consumer or producer,
    )


_COMPILED: dict[tuple, Callable] = {}


def _ptrs(addrs):
    bf16, i32 = cutlass.BFloat16, cutlass.Int32
    kinds = (bf16, bf16, bf16, bf16, i32, i32, i32, i32)
    aligns = (16, 16, 16, 16, 4, 4, 4, 4)
    return [
        make_ptr(k, a, cute.AddressSpace.gmem, assumed_align=al)
        for k, a, al in zip(kinds, addrs, aligns)
    ]


def _compiled(key: tuple, addrs, stream) -> Callable:
    fn = _COMPILED.get(key)
    if fn is None:
        num_heads, latent, rope, nope, page_size, consumer, producer, spin_limit, with_stats = key
        one32, eight64 = cutlass.Int32(1), cutlass.Int64(8)
        with BaseDSL.enable_pyir():
            fn = cute.compile(
                _context_append_host,
                *_ptrs(addrs),
                stream,
                *([one32] * 6),
                *([eight64] * 5),
                num_heads,
                latent,
                rope,
                nope,
                page_size,
                consumer,
                producer,
                spin_limit,
                with_stats,
            )
        _COMPILED[key] = fn
    return fn


def mla_context_append(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    k: torch.Tensor,
    kv_pool: torch.Tensor,
    page_table_rows: torch.Tensor,
    kv_lens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    fgs_consumer: bool = False,
    fgs_producer: bool = False,
    fgs_stats: Optional[torch.Tensor] = None,
    fgs_spin_limit: int = 0,
) -> None:
    """Append a context batch's latent rows to the paged cache and broadcast k_pe into K's rope columns.

    compressed_kv [T, latent] and k_pe [T, rope] (row-strided views, unit last stride, bf16); k [T, H, nope + rope]
    bf16 with unit last stride (its rope columns are overwritten). kv_pool: this layer's ``get_buffers`` view.
    page_table_rows: ``block_offsets[pool, ctx rows, 0, :]`` [B, max_blocks] int32. kv_lens: int32 [B], KV length
    including this chunk. cu_seqlens: int32 [B + 1], this chunk's token offsets. max_seqlen: host max of the lengths.
    fgs_consumer / fgs_producer: see the module docstring; ``fgs_stats`` (int32 [4]) accumulates the consumer's wait
    statistics and ``fgs_spin_limit`` > 0 traps after that many reload sweeps (diagnostic).
    """
    fgs = fgs_consumer or fgs_producer
    num_tokens, latent = compressed_kv.shape
    rope = k_pe.shape[-1]
    _, num_heads, head_dim = k.shape
    nope = head_dim - rope
    batch = page_table_rows.shape[0]
    if num_tokens == 0 or batch == 0:
        return
    for name, t in (("compressed_kv", compressed_kv), ("k_pe", k_pe), ("k", k)):
        if t.dtype != torch.bfloat16 or t.stride(-1) != 1 or t.data_ptr() % 16:
            raise ValueError(f"{name} must be bf16, unit last stride, 16B aligned")
    strides = (compressed_kv.stride(0), k_pe.stride(0), k.stride(0), k.stride(1), nope)
    if any(s % VEC for s in strides) or latent % VEC or rope % VEC:
        raise ValueError("row strides, widths and the nope offset must be multiples of 8 elements")
    if page_table_rows.dtype != torch.int32 or page_table_rows.stride(1) != 1:
        raise ValueError("page_table_rows must be int32 with unit last stride")
    for name, t, n in (("kv_lens", kv_lens, batch), ("cu_seqlens", cu_seqlens, batch + 1)):
        if t.dtype != torch.int32 or not t.is_contiguous() or t.numel() < n:
            raise ValueError(f"{name} must be a contiguous int32 tensor with {n} entries")

    base, pool_rows, row_stride, layer_in_pool, _ = kv_pool_geometry(kv_pool)
    page_size = kv_pool.shape[2]
    if (
        kv_pool.dtype != torch.bfloat16
        or base % 16
        or row_stride % VEC
        or kv_pool.shape[-1] < latent + rope
    ):
        raise ValueError(
            "kv_pool must be a bf16, 16B-aligned pool with rows of at least latent + rope"
        )

    if fgs and num_heads % FGS_HEADS_PER_THREAD:
        raise ValueError(f"FGS modes need a multiple of {FGS_HEADS_PER_THREAD} heads")
    with_stats = fgs_consumer and fgs_stats is not None
    if with_stats and (fgs_stats.dtype != torch.int32 or fgs_stats.numel() < STATS_LEN):
        raise ValueError("fgs_stats must be an int32 tensor with at least 4 entries")
    head_vecs = num_heads // FGS_HEADS_PER_THREAD if fgs else num_heads
    token_vecs = (latent + rope) // VEC + head_vecs * (rope // VEC)
    ctas = -(-max_seqlen * token_vecs // THREADS)
    addrs = (
        compressed_kv.data_ptr(),
        k_pe.data_ptr(),
        k.data_ptr(),
        base,
        page_table_rows.data_ptr(),
        kv_lens.data_ptr(),
        cu_seqlens.data_ptr(),
        fgs_stats.data_ptr() if with_stats else kv_lens.data_ptr(),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    spin_limit = fgs_spin_limit if fgs_consumer else 0
    key = (
        num_heads,
        latent,
        rope,
        nope,
        page_size,
        fgs_consumer,
        fgs_producer,
        spin_limit,
        with_stats,
    )
    fn = _compiled(key, addrs, stream)
    i32, i64 = cutlass.Int32, cutlass.Int64
    fn(
        *_ptrs(addrs),
        stream,
        i32(num_tokens),
        i32(batch),
        i32(ctas),
        i32(page_table_rows.stride(0)),
        i32(layer_in_pool),
        i32(pool_rows),
        i64(compressed_kv.stride(0)),
        i64(k_pe.stride(0)),
        i64(k.stride(0)),
        i64(k.stride(1)),
        i64(row_stride),
    )
