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
"""Kimi K3 MLA decode latent append: one CuTe DSL launch for the BF16 latent cache.

For a BF16 cache, the identity (NoPE) rope table, no helix and a uniform query length per request, computes what
``torch.cat([compressed_kv, k_pe])`` + ``torch.ops.trtllm.mla_rope_generation`` + the decode page-table add do:

  fused_q[t, h, latent:] = q_pe[t, h]
  cache row (b, kv_len[b] - seq_len_q + u) = [compressed_kv[t] | k_pe[t]]
  page_table_out[j, b] = page_table[b, j] + layer_in_pool
  cu_q_seqlens[i] = i * H * seq_len_q, cu_kv_seqlens = exclusive scan of kv_len, fmha_scheduler_counter = 0

Every write is a copy, so the result is bit-identical to the stock sequence, except that the stock identity
rotation computes x*1 - y*0 in fp32 and can turn a -0.0 in q_pe / k_pe into +0.0.

CTA roles (blockIdx.x ranges, 256 threads, one 16-byte vector or one int per thread):
  [0, ctas_q)        q_pe -> fused_q[..., latent:]
  [.., +ctas_kv)     latent rows -> paged cache (row = (page_table[b][p / P] + page_offset) * P + p % P)
  [.., +ctas_pt)     page_table_out (when write_pt)
  last (n_meta = 1)  warp 0: cu_q_seqlens, cu_kv_seqlens (warp scan), scheduler counter (when write_meta)

Fine-grained sync (Lamport) modes, both opt-in and combinable:
  fgs_consumer  q_pe, compressed_kv and k_pe are FGS slots: launched with PDL, no ``griddepcontrol.wait``; the
                step inputs (kv_lens, page table) are read first, then every 16-B input chunk is loaded with
                ``ld.global.cg`` and, while it holds the -0.0 sentinel, reloaded by the warp-uniform wait of
                ``mla_fgs.poll_chunks``.
  fgs_producer  fused_q[..., latent:] and the cache rows are published to FGS consumers: dependents are triggered
                on entry and every 16-B chunk is one ``st.global.v4`` with -0.0 stored as +0.0 (with
                ``fgs_consumer`` the copied chunks already hold no -0.0, so they are stored as loaded).
  rearm_ckv     (with fgs_consumer) the latent append is the last reader of compressed_kv: each thread stores the
                -0.0 sentinel over the compressed_kv chunk it copied, after its own load, so the slot is armed for its
                next producer without the framework's fill. ``rearm_delay_ns`` > 0 sleeps that long before the
                re-arm (hazard tests only).
In either mode the page table is not written (the decode kernel prefetches it before its first wait, so it must be
complete before the step: ``mla_decode_page_table``).
"""

from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import make_ptr

try:
    from .mla_fgs import (
        BF16_SENTINEL,
        STATS_LEN,
        canon_bf16x2,
        load_chunks,
        poll_chunks,
        sleep_ns,
        st_global_v4,
    )
except ImportError:
    from mla_fgs import (
        BF16_SENTINEL,
        STATS_LEN,
        canon_bf16x2,
        load_chunks,
        poll_chunks,
        sleep_ns,
        st_global_v4,
    )

THREADS = 256
VEC = 8  # bf16 elements per 16-byte access
SENTINEL_X2 = (BF16_SENTINEL | (BF16_SENTINEL << 16)) - (1 << 32)


@cute.jit
def _write_page_table(
    mPtIn: cute.Tensor,
    mPtOut: cute.Tensor,
    batch: cutlass.Int32,
    first_cta: cutlass.Int32,
    max_blocks: cutlass.Int32,
    pt_in_stride: cutlass.Int32,
    page_offset: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    e = (bidx - first_cta) * cutlass.Int32(THREADS) + tidx
    if e < batch * max_blocks:
        b = e // max_blocks
        j = e - b * max_blocks
        mPtOut[e] = cutlass.Int32(mPtIn[b * pt_in_stride + j]) + page_offset


@cute.jit
def _write_meta(
    mKvLens: cute.Tensor,
    mCuQ: cute.Tensor,
    mCuKv: cute.Tensor,
    mCounter: cute.Tensor,
    batch: cutlass.Int32,
    seq_len_q: cutlass.Int32,
    num_heads: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < cutlass.Int32(32):
        if tidx == cutlass.Int32(0):
            mCounter[0] = cutlass.Int32(0)
        carry = cutlass.Int32(0)
        # cu_kv_seqlens has batch + 1 entries.
        n_chunks = (batch + cutlass.Int32(32)) // cutlass.Int32(32)
        for c in cutlass.range(0, n_chunks, 1, unroll=1):
            i = c * cutlass.Int32(32) + tidx
            x = cutlass.Int32(0)
            if i < batch:
                x = cutlass.Int32(mKvLens[i])
            incl = x
            for off in (1, 2, 4, 8, 16):
                y = cute.arch.shuffle_sync_up(incl, off)
                if tidx >= cutlass.Int32(off):
                    incl = incl + y
            if i <= batch:
                mCuKv[i] = carry + incl - x
                mCuQ[i] = i * seq_len_q * cutlass.Int32(num_heads)
            carry = carry + cute.arch.shuffle_sync(incl, 31)


@cute.jit
def _append_fgs(
    mQpe: cute.Tensor,
    mFusedQ: cute.Tensor,
    mCkv: cute.Tensor,
    mKpe: cute.Tensor,
    mPool: cute.Tensor,
    mPtIn: cute.Tensor,
    mKvLens: cute.Tensor,
    mStats: cute.Tensor,
    num_tokens: cutlass.Int32,
    seq_len_q: cutlass.Int32,
    ctas_q: cutlass.Int32,
    pt_in_stride: cutlass.Int32,
    page_offset: cutlass.Int32,
    q_tok_stride: cutlass.Int64,
    q_head_stride: cutlass.Int64,
    ckv_stride: cutlass.Int64,
    kpe_stride: cutlass.Int64,
    pool_row_stride: cutlass.Int64,
    num_heads: cutlass.Constexpr,
    latent: cutlass.Constexpr,
    rope: cutlass.Constexpr,
    page_size: cutlass.Constexpr,
    consumer: cutlass.Constexpr,
    producer: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    with_stats: cutlass.Constexpr,
    rearm_ckv: cutlass.Constexpr,
    rearm_delay_ns: cutlass.Constexpr,
):
    """The q_pe and cache-row roles in FGS mode: one 16-B chunk per thread, source / destination byte addresses
    computed (step inputs read) before the input wait; lanes without a chunk stay converged for the wait."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    rope_vecs: cutlass.Constexpr = rope // VEC
    latent_vecs: cutlass.Constexpr = latent // VEC
    row_vecs: cutlass.Constexpr = (latent + rope) // VEC
    q_row: cutlass.Constexpr = latent + rope
    src = cutlass.Int64(0)
    dst = cutlass.Int64(0)
    active = cutlass.Boolean(False)
    is_ckv = cutlass.Boolean(False)
    if bidx < ctas_q:
        gid = bidx * cutlass.Int32(THREADS) + tidx
        if gid < num_tokens * cutlass.Int32(num_heads * rope_vecs):
            row = gid // cutlass.Int32(rope_vecs)
            v = gid - row * cutlass.Int32(rope_vecs)
            t = row // cutlass.Int32(num_heads)
            h = row - t * cutlass.Int32(num_heads)
            src = mQpe.iterator.toint() + cutlass.Int64(2) * (
                cutlass.Int64(t) * q_tok_stride
                + cutlass.Int64(h) * q_head_stride
                + cutlass.Int64(v * cutlass.Int32(VEC))
            )
            dst = mFusedQ.iterator.toint() + cutlass.Int64(2) * (
                cutlass.Int64(row) * cutlass.Int64(q_row)
                + cutlass.Int64(latent)
                + cutlass.Int64(v * cutlass.Int32(VEC))
            )
            active = cutlass.Boolean(True)
    else:
        gid = (bidx - ctas_q) * cutlass.Int32(THREADS) + tidx
        if gid < num_tokens * cutlass.Int32(row_vecs):
            t = gid // cutlass.Int32(row_vecs)
            v = gid - t * cutlass.Int32(row_vecs)
            b = t // seq_len_q
            pos = cutlass.Int32(mKvLens[b]) - seq_len_q + (t - b * seq_len_q)
            if pos >= cutlass.Int32(0):
                page = (
                    cutlass.Int32(mPtIn[b * pt_in_stride + pos // cutlass.Int32(page_size)])
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
                    is_ckv = cutlass.Boolean(True)
                else:
                    src = mKpe.iterator.toint() + cutlass.Int64(2) * (
                        cutlass.Int64(t) * kpe_stride
                        + cutlass.Int64((v - cutlass.Int32(latent_vecs)) * cutlass.Int32(VEC))
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
    if active:
        # A chunk that passed the consumer wait holds no -0.0 element, so only the producer-only build canonicalizes.
        if cutlass.const_expr(producer and not consumer):
            st_global_v4(
                dst,
                canon_bf16x2(words[0]),
                canon_bf16x2(words[1]),
                canon_bf16x2(words[2]),
                canon_bf16x2(words[3]),
            )
        else:
            st_global_v4(dst, words[0], words[1], words[2], words[3])
        if cutlass.const_expr(rearm_ckv):
            # Each compressed_kv chunk has exactly one reader thread, and its load completed above.
            if is_ckv:
                if cutlass.const_expr(rearm_delay_ns > 0):
                    sleep_ns(rearm_delay_ns)
                s = cutlass.Int32(SENTINEL_X2)
                st_global_v4(src, s, s, s, s)


@cute.kernel
def _latent_append_kernel(
    mQpe: cute.Tensor,
    mFusedQ: cute.Tensor,
    mCkv: cute.Tensor,
    mKpe: cute.Tensor,
    mPool: cute.Tensor,
    mPtIn: cute.Tensor,
    mKvLens: cute.Tensor,
    mPtOut: cute.Tensor,
    mCuQ: cute.Tensor,
    mCuKv: cute.Tensor,
    mCounter: cute.Tensor,
    mStats: cute.Tensor,
    num_tokens: cutlass.Int32,
    seq_len_q: cutlass.Int32,
    batch: cutlass.Int32,
    ctas_q: cutlass.Int32,
    ctas_kv: cutlass.Int32,
    ctas_pt: cutlass.Int32,
    max_blocks: cutlass.Int32,
    pt_in_stride: cutlass.Int32,
    page_offset: cutlass.Int32,
    q_tok_stride: cutlass.Int64,
    q_head_stride: cutlass.Int64,
    ckv_stride: cutlass.Int64,
    kpe_stride: cutlass.Int64,
    pool_row_stride: cutlass.Int64,
    num_heads: cutlass.Constexpr,
    latent: cutlass.Constexpr,
    rope: cutlass.Constexpr,
    page_size: cutlass.Constexpr,
    write_pt: cutlass.Constexpr,
    write_meta: cutlass.Constexpr,
    consumer: cutlass.Constexpr,
    producer: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    with_stats: cutlass.Constexpr,
    rearm_ckv: cutlass.Constexpr,
    rearm_delay_ns: cutlass.Constexpr,
):
    if cutlass.const_expr(producer):
        cute.arch.griddepcontrol_launch_dependents()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    rope_vecs: cutlass.Constexpr = rope // VEC
    latent_vecs: cutlass.Constexpr = latent // VEC
    row_vecs: cutlass.Constexpr = (latent + rope) // VEC
    q_row: cutlass.Constexpr = latent + rope

    qpe = cutlass.make_array_view(mQpe)
    fused_q = cutlass.make_array_view(mFusedQ)
    ckv = cutlass.make_array_view(mCkv)
    kpe = cutlass.make_array_view(mKpe)
    pool = cutlass.make_array_view(mPool)

    if cutlass.const_expr(consumer or producer):
        if bidx < ctas_q + ctas_kv:
            _append_fgs(
                mQpe, mFusedQ, mCkv, mKpe, mPool, mPtIn, mKvLens, mStats, num_tokens, seq_len_q, ctas_q,
                pt_in_stride, page_offset, q_tok_stride, q_head_stride, ckv_stride, kpe_stride, pool_row_stride,
                num_heads, latent, rope, page_size, consumer, producer, spin_limit, with_stats, rearm_ckv,
                rearm_delay_ns,
            )  # fmt: skip
        else:
            if cutlass.const_expr(write_meta):
                _write_meta(mKvLens, mCuQ, mCuKv, mCounter, batch, seq_len_q, num_heads)
    else:
        if bidx < ctas_q:
            gid = bidx * cutlass.Int32(THREADS) + tidx
            if gid < num_tokens * cutlass.Int32(num_heads * rope_vecs):
                row = gid // cutlass.Int32(rope_vecs)
                v = gid - row * cutlass.Int32(rope_vecs)
                t = row // cutlass.Int32(num_heads)
                h = row - t * cutlass.Int32(num_heads)
                src = (
                    cutlass.Int64(t) * q_tok_stride
                    + cutlass.Int64(h) * q_head_stride
                    + cutlass.Int64(v * cutlass.Int32(VEC))
                )
                dst = (
                    cutlass.Int64(row) * cutlass.Int64(q_row)
                    + cutlass.Int64(latent)
                    + cutlass.Int64(v * cutlass.Int32(VEC))
                )
                fused_q.store(
                    qpe.load(src, VEC, alignment=16), idx=dst, vector_size=VEC, alignment=16
                )
        elif bidx < ctas_q + ctas_kv:
            gid = (bidx - ctas_q) * cutlass.Int32(THREADS) + tidx
            if gid < num_tokens * cutlass.Int32(row_vecs):
                t = gid // cutlass.Int32(row_vecs)
                v = gid - t * cutlass.Int32(row_vecs)
                b = t // seq_len_q
                pos = cutlass.Int32(mKvLens[b]) - seq_len_q + (t - b * seq_len_q)
                if pos >= cutlass.Int32(0):
                    page = (
                        cutlass.Int32(mPtIn[b * pt_in_stride + pos // cutlass.Int32(page_size)])
                        + page_offset
                    )
                    dst = (
                        cutlass.Int64(page) * cutlass.Int64(page_size)
                        + cutlass.Int64(pos % cutlass.Int32(page_size))
                    ) * pool_row_stride + cutlass.Int64(v * cutlass.Int32(VEC))
                    if v < cutlass.Int32(latent_vecs):
                        pool.store(
                            ckv.load(
                                cutlass.Int64(t) * ckv_stride
                                + cutlass.Int64(v * cutlass.Int32(VEC)),
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
                                + cutlass.Int64(
                                    (v - cutlass.Int32(latent_vecs)) * cutlass.Int32(VEC)
                                ),
                                VEC,
                                alignment=16,
                            ),
                            idx=dst,
                            vector_size=VEC,
                            alignment=16,
                        )
        elif bidx < ctas_q + ctas_kv + ctas_pt:
            if cutlass.const_expr(write_pt):
                _write_page_table(
                    mPtIn, mPtOut, batch, ctas_q + ctas_kv, max_blocks, pt_in_stride, page_offset
                )
        else:
            if cutlass.const_expr(write_meta):
                _write_meta(mKvLens, mCuQ, mCuKv, mCounter, batch, seq_len_q, num_heads)


@cute.jit
def _latent_append_host(
    q_pe_ptr: cute.Pointer,
    fused_q_ptr: cute.Pointer,
    ckv_ptr: cute.Pointer,
    kpe_ptr: cute.Pointer,
    pool_ptr: cute.Pointer,
    pt_in_ptr: cute.Pointer,
    kv_lens_ptr: cute.Pointer,
    pt_out_ptr: cute.Pointer,
    cu_q_ptr: cute.Pointer,
    cu_kv_ptr: cute.Pointer,
    counter_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
    stream: cuda.CUstream,
    num_tokens: cutlass.Int32,
    seq_len_q: cutlass.Int32,
    batch: cutlass.Int32,
    ctas_q: cutlass.Int32,
    ctas_kv: cutlass.Int32,
    ctas_pt: cutlass.Int32,
    n_meta: cutlass.Int32,
    max_blocks: cutlass.Int32,
    pt_in_stride: cutlass.Int32,
    page_offset: cutlass.Int32,
    pool_rows: cutlass.Int32,
    q_tok_stride: cutlass.Int64,
    q_head_stride: cutlass.Int64,
    ckv_stride: cutlass.Int64,
    kpe_stride: cutlass.Int64,
    pool_row_stride: cutlass.Int64,
    num_heads: cutlass.Constexpr,
    latent: cutlass.Constexpr,
    rope: cutlass.Constexpr,
    page_size: cutlass.Constexpr,
    write_pt: cutlass.Constexpr,
    write_meta: cutlass.Constexpr,
    consumer: cutlass.Constexpr,
    producer: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    with_stats: cutlass.Constexpr,
    rearm_ckv: cutlass.Constexpr,
    rearm_delay_ns: cutlass.Constexpr,
):
    mQpe = cute.make_tensor(
        q_pe_ptr,
        cute.make_layout((num_tokens, num_heads, rope), stride=(q_tok_stride, q_head_stride, 1)),
    )
    mFusedQ = cute.make_tensor(
        fused_q_ptr,
        cute.make_layout(
            (num_tokens, num_heads, latent + rope),
            stride=(num_heads * (latent + rope), latent + rope, 1),
        ),
    )
    mCkv = cute.make_tensor(ckv_ptr, cute.make_layout((num_tokens, latent), stride=(ckv_stride, 1)))
    mKpe = cute.make_tensor(kpe_ptr, cute.make_layout((num_tokens, rope), stride=(kpe_stride, 1)))
    mPool = cute.make_tensor(
        pool_ptr, cute.make_layout((pool_rows, latent + rope), stride=(pool_row_stride, 1))
    )
    mPtIn = cute.make_tensor(pt_in_ptr, cute.make_layout((batch * pt_in_stride,), stride=(1,)))
    mKvLens = cute.make_tensor(kv_lens_ptr, cute.make_layout((batch,), stride=(1,)))
    mPtOut = cute.make_tensor(pt_out_ptr, cute.make_layout((batch * max_blocks,), stride=(1,)))
    mCuQ = cute.make_tensor(cu_q_ptr, cute.make_layout((batch + 1,), stride=(1,)))
    mCuKv = cute.make_tensor(cu_kv_ptr, cute.make_layout((batch + 1,), stride=(1,)))
    mCounter = cute.make_tensor(counter_ptr, cute.make_layout((1,), stride=(1,)))
    mStats = cute.make_tensor(stats_ptr, cute.make_layout((STATS_LEN,), stride=(1,)))
    _latent_append_kernel(
        mQpe,
        mFusedQ,
        mCkv,
        mKpe,
        mPool,
        mPtIn,
        mKvLens,
        mPtOut,
        mCuQ,
        mCuKv,
        mCounter,
        mStats,
        num_tokens,
        seq_len_q,
        batch,
        ctas_q,
        ctas_kv,
        ctas_pt,
        max_blocks,
        pt_in_stride,
        page_offset,
        q_tok_stride,
        q_head_stride,
        ckv_stride,
        kpe_stride,
        pool_row_stride,
        num_heads,
        latent,
        rope,
        page_size,
        write_pt,
        write_meta,
        consumer,
        producer,
        spin_limit,
        with_stats,
        rearm_ckv,
        rearm_delay_ns,
    ).launch(
        grid=(ctas_q + ctas_kv + ctas_pt + n_meta, 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        use_pdl=consumer or producer,
    )


_COMPILED: dict[tuple, Callable] = {}


def _ptr(dtype, address: int, align: int) -> cute.Pointer:
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def _ptrs(addrs):
    bf16, i32 = cutlass.BFloat16, cutlass.Int32
    kinds = (bf16, bf16, bf16, bf16, bf16, i32, i32, i32, i32, i32, i32, i32)
    aligns = (16, 16, 16, 16, 16, 4, 4, 4, 4, 4, 4, 4)
    return [_ptr(k, a, al) for k, a, al in zip(kinds, addrs, aligns)]


def _compiled(key: tuple, addrs, stream) -> Callable:
    fn = _COMPILED.get(key)
    if fn is None:
        (num_heads, latent, rope, page_size, write_pt, write_meta, consumer, producer, spin_limit,
         with_stats, rearm_ckv, rearm_delay_ns) = key  # fmt: skip
        one32, eight64 = cutlass.Int32(1), cutlass.Int64(8)
        with BaseDSL.enable_pyir():
            fn = cute.compile(
                _latent_append_host,
                *_ptrs(addrs),
                stream,
                *([one32] * 11),
                *([eight64] * 5),
                num_heads,
                latent,
                rope,
                page_size,
                write_pt,
                write_meta,
                consumer,
                producer,
                spin_limit,
                with_stats,
                rearm_ckv,
                rearm_delay_ns,
            )
        _COMPILED[key] = fn
    return fn


def kv_pool_geometry(kv_pool: torch.Tensor) -> tuple[int, int, int, int, bool]:
    """(base address, total rows, row stride in elements, layer_in_pool, interleaved) of the paged latent pool.

    ``kv_pool`` is ``kv_cache_manager.get_buffers(layer_idx)``, ``[pages, 1, page_size, 1, dim]``.
    Mirrors ``CuteDslMlaFmha._run_mla_decode``: a V1 per-layer view over an interleaved pool is
    addressed as the packed combined-slot pool with this layer's slot added to every page index.
    """
    if (
        kv_pool.dim() != 5
        or kv_pool.shape[1] != 1
        or kv_pool.shape[3] != 1
        or kv_pool.stride(-1) != 1
    ):
        raise ValueError(
            f"expected a [pages, 1, page_size, 1, dim] pool, got {tuple(kv_pool.shape)}"
        )
    packed = 1
    for size in kv_pool.shape[1:]:
        packed *= size
    block_stride = kv_pool.stride(0)
    layers_in_pool = block_stride // packed if packed else 1
    if layers_in_pool > 1 and block_stride == layers_in_pool * packed:
        layer_in_pool = kv_pool.storage_offset() // packed
        base = kv_pool.data_ptr() - kv_pool.storage_offset() * kv_pool.element_size()
        pages = kv_pool.shape[0] * layers_in_pool
        page_stride = packed
    else:
        layers_in_pool = 1
        layer_in_pool = 0
        base = kv_pool.data_ptr()
        pages = kv_pool.shape[0]
        page_stride = block_stride
    page_size = kv_pool.shape[2]
    row_stride = kv_pool.stride(2)
    if page_stride != page_size * row_stride:
        raise ValueError(
            f"pool page stride {page_stride} != page_size {page_size} * row stride {row_stride}"
        )
    return base, pages * page_size, row_stride, layer_in_pool, layers_in_pool > 1


def mla_latent_append(
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    fused_q: torch.Tensor,
    kv_pool: torch.Tensor,
    page_table_rows: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    page_table_out: Optional[torch.Tensor] = None,
    cu_q_seqlens: Optional[torch.Tensor] = None,
    cu_kv_seqlens: Optional[torch.Tensor] = None,
    fmha_scheduler_counter: Optional[torch.Tensor] = None,
    fgs_consumer: bool = False,
    fgs_producer: bool = False,
    fgs_stats: Optional[torch.Tensor] = None,
    fgs_spin_limit: int = 0,
    fgs_rearm_ckv: bool = False,
    fgs_rearm_delay_ns: int = 0,
) -> Optional[torch.Tensor]:
    """Append this step's latent rows to the paged cache and copy q_pe into fused_q.

    q_pe [T, H, rope] (strided view, unit last stride), compressed_kv [T, latent] (normalized),
    k_pe [T, rope], fused_q [T, H, latent + rope] contiguous, all bf16.
    kv_pool: this layer's ``get_buffers`` view. page_table_rows: ``block_offsets[pool, gen rows, 0, :]``
    [B, max_blocks] int32. kv_lens: int32 [B], KV length including this step's tokens.
    page_table_out: [max_blocks, B] int32 with strides (1, max_blocks); when the pool is
    interleaved it receives page_table_rows^T + layer_in_pool (the decode kernel's page table).
    cu_q_seqlens / cu_kv_seqlens ([>= B + 1] int32) and fmha_scheduler_counter ([1] 32-bit) are
    written only when all three are given (the trtllm-gen decode reads them, the CuTe one does not).

    Returns the page table the CuTe MLA decode kernel consumes ([max_blocks, B]), or None in the FGS modes
    (``fgs_consumer`` / ``fgs_producer``, see the module docstring), which never write it: the caller provides it
    before the step (``mla_decode_page_table``). ``fgs_stats`` (int32 [4]) accumulates the consumer's wait
    statistics; ``fgs_spin_limit`` > 0 traps after that many reload sweeps (diagnostic). ``fgs_rearm_ckv``
    (needs ``fgs_consumer``) re-arms every compressed_kv chunk after it is read; ``fgs_rearm_delay_ns`` delays that
    (hazard tests).
    """
    if fgs_rearm_ckv and not fgs_consumer:
        raise ValueError("fgs_rearm_ckv re-arms an FGS input slot: it needs fgs_consumer")
    fgs = fgs_consumer or fgs_producer
    num_tokens, num_heads, rope = q_pe.shape
    latent = compressed_kv.shape[-1]
    batch, max_blocks = page_table_rows.shape
    if num_tokens == 0 or batch == 0:
        return None if fgs else page_table_rows.transpose(0, 1)
    if num_tokens % batch:
        raise ValueError("every request must contribute the same number of query tokens")
    seq_len_q = num_tokens // batch
    for name, t in (
        ("q_pe", q_pe),
        ("compressed_kv", compressed_kv),
        ("k_pe", k_pe),
        ("fused_q", fused_q),
    ):
        if t.dtype != torch.bfloat16 or t.stride(-1) != 1 or t.data_ptr() % 16:
            raise ValueError(f"{name} must be bf16, unit last stride, 16B aligned")
    if not fused_q.is_contiguous() or tuple(fused_q.shape) != (
        num_tokens,
        num_heads,
        latent + rope,
    ):
        raise ValueError("fused_q must be a contiguous [T, H, latent + rope] tensor")
    strides = (q_pe.stride(0), q_pe.stride(1), compressed_kv.stride(0), k_pe.stride(0))
    if any(s % VEC for s in strides) or latent % VEC or rope % VEC:
        raise ValueError("row strides and widths must be multiples of 8 elements")
    if page_table_rows.dtype != torch.int32 or page_table_rows.stride(1) != 1:
        raise ValueError("page_table_rows must be int32 with unit last stride")
    if kv_lens.dtype != torch.int32 or not kv_lens.is_contiguous() or kv_lens.numel() < batch:
        raise ValueError("kv_lens must be a contiguous int32 [B] tensor")

    base, pool_rows, row_stride, layer_in_pool, interleaved = kv_pool_geometry(kv_pool)
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

    write_pt = interleaved and not fgs
    if fgs:
        result = None
    elif write_pt:
        if page_table_out is None:
            page_table_out = torch.empty_strided(
                (max_blocks, batch), (1, max_blocks), dtype=torch.int32, device=q_pe.device
            )
        elif page_table_out.dtype != torch.int32 or page_table_out.stride() != (1, max_blocks):
            raise ValueError(
                "page_table_out must be int32 [max_blocks, B] with strides (1, max_blocks)"
            )
        result = page_table_out
    else:
        result = page_table_rows.transpose(0, 1)
    write_meta = (
        cu_q_seqlens is not None
        and cu_kv_seqlens is not None
        and fmha_scheduler_counter is not None
    )
    if write_meta and (cu_q_seqlens.numel() < batch + 1 or cu_kv_seqlens.numel() < batch + 1):
        raise ValueError("cu_q_seqlens / cu_kv_seqlens need batch + 1 entries")
    with_stats = fgs_consumer and fgs_stats is not None
    if with_stats and (fgs_stats.dtype != torch.int32 or fgs_stats.numel() < STATS_LEN):
        raise ValueError("fgs_stats must be an int32 tensor with at least 4 entries")

    ctas_q = -(-num_tokens * num_heads * (rope // VEC) // THREADS)
    ctas_kv = -(-num_tokens * ((latent + rope) // VEC) // THREADS)
    ctas_pt = -(-batch * max_blocks // THREADS) if write_pt else 0
    n_meta = 1 if write_meta else 0
    dummy = kv_lens.data_ptr()
    addrs = (
        q_pe.data_ptr(),
        fused_q.data_ptr(),
        compressed_kv.data_ptr(),
        k_pe.data_ptr(),
        base,
        page_table_rows.data_ptr(),
        kv_lens.data_ptr(),
        page_table_out.data_ptr() if write_pt else dummy,
        cu_q_seqlens.data_ptr() if write_meta else dummy,
        cu_kv_seqlens.data_ptr() if write_meta else dummy,
        fmha_scheduler_counter.data_ptr() if write_meta else dummy,
        fgs_stats.data_ptr() if with_stats else dummy,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    spin_limit = fgs_spin_limit if fgs_consumer else 0
    rearm_delay = fgs_rearm_delay_ns if fgs_rearm_ckv else 0
    key = (num_heads, latent, rope, page_size, write_pt, write_meta, fgs_consumer, fgs_producer, spin_limit,
           with_stats, fgs_rearm_ckv, rearm_delay)  # fmt: skip
    fn = _compiled(key, addrs, stream)
    i32, i64 = cutlass.Int32, cutlass.Int64
    fn(
        *_ptrs(addrs),
        stream,
        i32(num_tokens),
        i32(seq_len_q),
        i32(batch),
        i32(ctas_q),
        i32(ctas_kv),
        i32(ctas_pt),
        i32(n_meta),
        i32(max_blocks),
        i32(page_table_rows.stride(0)),
        i32(layer_in_pool),
        i32(pool_rows),
        i64(q_pe.stride(0)),
        i64(q_pe.stride(1)),
        i64(compressed_kv.stride(0)),
        i64(k_pe.stride(0)),
        i64(row_stride),
    )
    return result


def mla_decode_page_table(
    kv_pool: torch.Tensor,
    page_table_rows: torch.Tensor,
    page_table_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The decode kernel's page table ([max_blocks, B], strides (1, max_blocks)) for this layer's pool view:
    page_table_rows^T + layer_in_pool for an interleaved pool, the transposed view otherwise.

    What ``mla_latent_append`` writes in its default mode; the FGS modes need it before the step.
    """
    batch, max_blocks = page_table_rows.shape
    base, _, _, layer_in_pool, interleaved = kv_pool_geometry(kv_pool)
    if not interleaved or batch == 0:
        return page_table_rows.transpose(0, 1)
    if page_table_rows.dtype != torch.int32 or page_table_rows.stride(1) != 1:
        raise ValueError("page_table_rows must be int32 with unit last stride")
    if page_table_out is None:
        page_table_out = torch.empty_strided(
            (max_blocks, batch), (1, max_blocks), dtype=torch.int32, device=page_table_rows.device
        )
    elif page_table_out.dtype != torch.int32 or page_table_out.stride() != (1, max_blocks):
        raise ValueError(
            "page_table_out must be int32 [max_blocks, B] with strides (1, max_blocks)"
        )
    pt = page_table_rows.data_ptr()
    addrs = (base, base, base, base, base, pt, pt, page_table_out.data_ptr(), pt, pt, pt, pt)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (1, 8, 8, kv_pool.shape[2], True, False, False, False, 0, False, False, 0)
    fn = _compiled(key, addrs, stream)
    i32, i64 = cutlass.Int32, cutlass.Int64
    fn(
        *_ptrs(addrs),
        stream,
        i32(0),
        i32(1),
        i32(batch),
        i32(0),
        i32(0),
        i32(-(-batch * max_blocks // THREADS)),
        i32(0),
        i32(max_blocks),
        i32(page_table_rows.stride(0)),
        i32(layer_in_pool),
        i32(1),
        *([i64(8)] * 5),
    )
    return page_table_out
