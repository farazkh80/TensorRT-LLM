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
"""Kimi K3 attention-residual mix as a CuTe DSL kernel (sm_100 family, incl. SM107).

Computes, per token, over the N = K + 1 candidates ``v_0..v_{K-1} = bank[0..K-1], v_K = prefix``::

    q[h]    = f32(rms_w[h]) * f32(res_w[h])
    sigma_n = rsqrt(sum_h v_n[h]^2 / H + eps)
    logit_n = sigma_n * sum_h v_n[h] * q[h]
    w_n     = exp2((logit_n - max_n logit_n) * log2(e))
    mix[h]  = bf16((sum_n w_n * v_n[h]) * (1 / sum_n w_n))

which is ``torch.ops.trtllm.attn_res_fwd`` without its discarded diagnostics. Compile-time options:

* ``fuse_add``: the last candidate is ``bf16(f32(prefix) + f32(addend))``, also stored to ``updated``
  (bitwise equal to the ATen bf16 add).
* ``fuse_norm``: the output is the RMSNorm of the bf16 mix instead of the mix. ``norm_mode="flashinfer"``
  rounds once, ``bf16(x * rstd * w)``, like the FlashInfer RMSNorm on the unfused path;
  ``norm_mode="kimi"`` rounds twice, ``bf16(f32(bf16(x * rstd)) * w)``, like ``KimiK3RMSNorm`` and the
  C++ fused per-token kernels.

``AttnResMixKernel`` (few tokens): one token per cluster of ``cluster_n`` CTAs, each CTA owning
``H / cluster_n`` columns; every thread owns ``vecs_per_thread`` 16-byte vectors per candidate and keeps all N
of them in registers, so a token is one round of N independent loads. The per-candidate sums (2N floats) are
reduced once per token (warp shuffle, then shared memory, then DSMEM across the cluster). A grid smaller than
the token count makes the kernel persistent: each cluster strides over tokens with double-buffered reduction
slots.
``AttnResMixPipeKernel`` (many tokens) streams the rows through a shared-memory ring instead.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    _CUTE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CUTE_AVAILABLE = False

_VEC = 8
_LOG2E = 1.4426950408889634
_MAX_CANDIDATES = 12


def is_attn_res_mix_cute_available() -> bool:
    return (
        _CUTE_AVAILABLE
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
    )


if _CUTE_AVAILABLE:
    # mapa / st.async helpers follow FlashInfer's norm utils (Apache-2.0).
    @dsl_user_op
    def _map_to_peer(smem_ptr: cute.Pointer, peer: Int32, *, loc=None, ip=None) -> Int32:
        ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [ptr_i32, peer.ir_value()],
                "mapa.shared::cluster.u32 $0, $1, $2;",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _store_remote_f32(
        val: Float32,
        smem_ptr: cute.Pointer,
        mbar_ptr: cute.Pointer,
        peer: Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        remote_ptr = _map_to_peer(smem_ptr, peer, loc=loc, ip=ip).ir_value()
        remote_mbar = _map_to_peer(mbar_ptr, peer, loc=loc, ip=ip).ir_value()
        llvm.inline_asm(
            None,
            [remote_ptr, val.ir_value(loc=loc, ip=ip), remote_mbar],
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
            "r,f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    class AttnResMixKernel:
        def __init__(
            self,
            num_candidates: int,
            hidden_size: int = 7168,
            cluster_n: int = 1,
            vecs_per_thread: int = 1,
            fuse_add: bool = False,
            fuse_norm: bool = False,
            norm_mode: str = "flashinfer",
            enable_pdl: bool = False,
        ):
            assert 2 <= num_candidates <= _MAX_CANDIDATES
            assert norm_mode in ("flashinfer", "kimi")
            cols_per_thread = _VEC * vecs_per_thread
            assert hidden_size % (cluster_n * cols_per_thread) == 0
            self.N = num_candidates
            self.H = hidden_size
            self.C = cluster_n
            self.V = vecs_per_thread
            self.threads = hidden_size // (cluster_n * cols_per_thread)
            assert self.threads % 32 == 0 and self.threads <= 1024, self.threads
            self.warps = self.threads // 32
            # Reduction slots hold one row of partials per (cluster CTA, warp); the final pass reads one
            # slot per lane, so a cluster may have at most 32 warps in total.
            self.slots = self.warps * cluster_n
            assert self.slots <= 32, self.slots
            self.nstat = 2 * num_candidates
            self.slot_stride = self.nstat + 1  # odd stride: conflict-free per-lane slot reads
            self.fuse_add = fuse_add
            self.fuse_norm = fuse_norm
            self.norm_kimi = norm_mode == "kimi"
            self.enable_pdl = enable_pdl

        def smem_bytes(self) -> int:
            nbytes = 2 * self.slots * self.slot_stride * 4
            if self.fuse_norm:
                nbytes += 2 * self.slots * 4
            nbytes += 4 * 8 + 64
            return nbytes

        @cute.jit
        def __call__(
            self,
            mB: cute.Tensor,
            mP: cute.Tensor,
            mA: cute.Tensor,
            mR: cute.Tensor,
            mG: cute.Tensor,
            mW: cute.Tensor,
            mO: cute.Tensor,
            mU: cute.Tensor,
            M: Int32,
            grid_x: Int32,
            eps: Float32,
            eps_out: Float32,
            stream,
        ):
            vec = (1, _VEC)
            self.kernel(
                cute.zipped_divide(mB, vec),
                cute.zipped_divide(mP, vec),
                cute.zipped_divide(mA, vec),
                cute.zipped_divide(mR, vec),
                cute.zipped_divide(mG, vec),
                cute.zipped_divide(mW, vec),
                cute.zipped_divide(mO, vec),
                cute.zipped_divide(mU, vec),
                M,
                eps,
                eps_out,
            ).launch(
                grid=[grid_x, self.C, 1],
                block=[self.threads, 1, 1],
                cluster=[1, self.C, 1] if cutlass.const_expr(self.C > 1) else None,
                smem=self.smem_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )

        @cute.jit
        def _allreduce(
            self,
            vals: cute.Tensor,
            count: cutlass.Constexpr[int],
            stride: cutlass.Constexpr[int],
            buf: cute.Tensor,
            mbar: cute.Pointer,
            parity: Int32,
            phase: Int32,
            rank: Int32,
        ):
            """Sum ``vals[0:count]`` over every thread of the cluster; every thread gets the totals."""
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            for k in cutlass.range_constexpr(count):
                v = vals[k]
                for i in cutlass.range_constexpr(5):
                    v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << i)
                vals[k] = v
            slot = rank * self.warps + warp
            if cutlass.const_expr(self.C == 1):
                if lane == 0:
                    for k in cutlass.range_constexpr(count):
                        buf[parity, slot * stride + k] = vals[k]
                cute.arch.barrier()
            else:
                if cute.arch.thread_idx()[0] == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar + parity, self.slots * count * 4)
                if lane < self.C:
                    for k in cutlass.range_constexpr(count):
                        _store_remote_f32(
                            vals[k],
                            buf.iterator + cute.crd2idx((parity, slot * stride + k), buf.layout),
                            mbar + parity,
                            lane,
                        )
                cute.arch.mbarrier_wait(mbar + parity, phase)
            for k in cutlass.range_constexpr(count):
                v = Float32(0.0)
                if lane < self.slots:
                    v = buf[parity, lane * stride + k]
                for i in cutlass.range_constexpr(5):
                    v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << i)
                vals[k] = v

        @cute.kernel
        def kernel(
            self,
            gB: cute.Tensor,
            gP: cute.Tensor,
            gA: cute.Tensor,
            gR: cute.Tensor,
            gG: cute.Tensor,
            gW: cute.Tensor,
            gO: cute.Tensor,
            gU: cute.Tensor,
            M: Int32,
            eps: Float32,
            eps_out: Float32,
        ):
            N = self.N
            K = N - 1
            H = self.H
            V = self.V
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            gdim, _, _ = cute.arch.grid_dim()

            rank = Int32(0)
            if cutlass.const_expr(self.C > 1):
                rank = cute.arch.block_idx_in_cluster()
            col0 = bidy * (self.threads * V) + tidx

            smem = cutlass.utils.SmemAllocator()
            stat_cols = self.slots * self.slot_stride
            stat_buf = smem.allocate_tensor(
                Float32, cute.make_layout((2, stat_cols), stride=(stat_cols, 1)), byte_alignment=16
            )
            norm_buf = stat_buf
            if cutlass.const_expr(self.fuse_norm):
                norm_buf = smem.allocate_tensor(
                    Float32,
                    cute.make_layout((2, self.slots), stride=(self.slots, 1)),
                    byte_alignment=16,
                )
            mbar = smem.allocate_array(cutlass.Int64, num_elems=4)

            if cutlass.const_expr(self.C > 1):
                if tidx == 0:
                    for i in cutlass.range_constexpr(4):
                        cute.arch.mbarrier_init(mbar + i, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.cluster_arrive_relaxed()

            q = []
            w_out = []
            for v in cutlass.range_constexpr(V):
                col = col0 + v * self.threads
                q.append(
                    gR[(None, (0, col))].load().to(Float32)
                    * gG[(None, (0, col))].load().to(Float32)
                )
                if cutlass.const_expr(self.fuse_norm):
                    w_out.append(gW[(None, (0, col))].load().to(Float32))

            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_wait()

            stats = cute.make_rmem_tensor(cute.make_layout((self.nstat,)), Float32)
            nstats = cute.make_rmem_tensor(cute.make_layout((1,)), Float32)
            it = Int32(0)
            for tok in cutlass.range(bidx, M, gdim, unroll=1):
                parity = it & 1
                phase = (it >> 1) & 1

                xs = []
                for n in cutlass.range_constexpr(K):
                    row = n * M + tok
                    xs.append([gB[(None, (row, col0 + v * self.threads))].load() for v in range(V)])
                last = [gP[(None, (tok, col0 + v * self.threads))].load() for v in range(V)]
                if cutlass.const_expr(self.fuse_add):
                    for v in cutlass.range_constexpr(V):
                        col = col0 + v * self.threads
                        a = gA[(None, (tok, col))].load()
                        last[v] = (last[v].to(Float32) + a.to(Float32)).to(BFloat16)
                        gU[(None, (tok, col))].store(last[v])
                xs.append(last)

                for n in cutlass.range_constexpr(N):
                    ss = Float32(0.0)
                    dot = Float32(0.0)
                    for v in cutlass.range_constexpr(V):
                        xf = xs[n][v].to(Float32)
                        ss = ss + (xf * xf).reduce(
                            cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
                        )
                        dot = dot + (xf * q[v]).reduce(
                            cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
                        )
                    stats[n] = ss
                    stats[N + n] = dot

                if cutlass.const_expr(self.C > 1):
                    if it == 0:
                        cute.arch.cluster_wait()
                self._allreduce(
                    stats, self.nstat, self.slot_stride, stat_buf, mbar, parity, phase, rank
                )

                logits = []
                for n in cutlass.range_constexpr(N):
                    rs = cute.math.rsqrt(stats[n] / Float32(H) + eps, fastmath=True)
                    logits.append(stats[N + n] * rs)
                mx = logits[0]
                for n in cutlass.range_constexpr(1, N):
                    mx = cutlass.max(mx, logits[n])
                wts = []
                wsum = Float32(0.0)
                for n in cutlass.range_constexpr(N):
                    wn = cute.math.exp2((logits[n] - mx) * Float32(_LOG2E), fastmath=True)
                    wts.append(wn)
                    wsum = wsum + wn
                inv_s = Float32(1.0) / wsum

                mix = []
                for v in cutlass.range_constexpr(V):
                    acc = xs[0][v].to(Float32) * wts[0]
                    for n in cutlass.range_constexpr(1, N):
                        acc = acc + xs[n][v].to(Float32) * wts[n]
                    mix.append((acc * inv_s).to(BFloat16))

                if cutlass.const_expr(self.fuse_norm):
                    ss = Float32(0.0)
                    for v in cutlass.range_constexpr(V):
                        mf = mix[v].to(Float32)
                        ss = ss + (mf * mf).reduce(
                            cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
                        )
                    nstats[0] = ss
                    self._allreduce(nstats, 1, 1, norm_buf, mbar + 2, parity, phase, rank)
                    rstd = cute.math.rsqrt(nstats[0] / Float32(H) + eps_out, fastmath=True)
                    for v in cutlass.range_constexpr(V):
                        col = col0 + v * self.threads
                        mf = mix[v].to(Float32)
                        if cutlass.const_expr(self.norm_kimi):
                            y = ((mf * rstd).to(BFloat16).to(Float32) * w_out[v]).to(BFloat16)
                        else:
                            y = (mf * rstd * w_out[v]).to(BFloat16)
                        gO[(None, (tok, col))].store(y)
                else:
                    for v in cutlass.range_constexpr(V):
                        gO[(None, (tok, col0 + v * self.threads))].store(mix[v])
                it = it + 1

            if cutlass.const_expr(self.C > 1):
                if it == 0:
                    cute.arch.cluster_wait()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

    @dsl_user_op
    def _bulk_g2s(
        dst: cute.Pointer, src: cute.Pointer, mbar: cute.Pointer, nbytes: int, *, loc=None, ip=None
    ) -> None:
        llvm.inline_asm(
            None,
            [
                dst.toint(loc=loc, ip=ip).ir_value(),
                src.toint(loc=loc, ip=ip).ir_value(),
                Int32(nbytes).ir_value(loc=loc, ip=ip),
                mbar.toint(loc=loc, ip=ip).ir_value(),
            ],
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
            "r,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    def _split_candidates(num_candidates: int, max_chunk: int):
        nchunks = -(-num_candidates // max_chunk)
        sizes = [
            num_candidates // nchunks + (1 if i < num_candidates % nchunks else 0)
            for i in range(nchunks)
        ]
        chunks, start = [], 0
        for size in sizes:
            chunks.append(list(range(start, start + size)))
            start += size
        return chunks

    class AttnResMixPipeKernel:
        """Many-token variant: persistent CTAs fed by a bulk-copy shared-memory ring.

        The last warp issues one ``cp.async.bulk`` per (token, row) into a ``stages``-deep ring of H-wide
        slots, so HBM reads run ``stages`` rows ahead of the consumers whatever N is. Consumer warps take the
        candidates in chunks of at most ``max_chunk`` rows, copy them to registers, release the slots and fold
        each chunk into an online softmax (the rescaling of stock online_v2).

        Lane 0 of every warp posts its warp's chunk sums to a reduction slot; lanes 0..2c-1 then sum one
        column each over the warps and broadcast. Without ``overlap`` a named barrier separates post and
        collect. With ``overlap`` the post arrives on a per-slot mbarrier and the warp loads and reduces the
        next chunk before collecting the previous one; a warp can then be at most two reductions ahead of
        another, so four slots make slot reuse safe. ``overlap`` keeps two chunks live and needs the registers.
        """

        RED_SLOTS = 4

        def __init__(
            self,
            num_candidates: int,
            hidden_size: int = 7168,
            vecs_per_thread: int = 2,
            stages: int = 12,
            max_chunk: int = 3,
            fuse_add: bool = False,
            fuse_norm: bool = False,
            norm_mode: str = "flashinfer",
            enable_pdl: bool = False,
            overlap: bool = False,
        ):
            assert 2 <= num_candidates <= _MAX_CANDIDATES
            assert norm_mode in ("flashinfer", "kimi")
            assert hidden_size % (_VEC * vecs_per_thread) == 0
            self.N = num_candidates
            self.H = hidden_size
            self.V = vecs_per_thread
            self.stages = stages
            self.consumers = hidden_size // (_VEC * vecs_per_thread)
            assert self.consumers % 32 == 0 and self.consumers + 32 <= 1024, self.consumers
            self.cwarps = self.consumers // 32
            self.threads = self.consumers + 32
            self.fuse_add = fuse_add
            self.fuse_norm = fuse_norm
            self.norm_kimi = norm_mode == "kimi"
            self.enable_pdl = enable_pdl
            self.overlap = overlap
            self.rows_per_token = num_candidates + (1 if fuse_add else 0)
            self.chunks = _split_candidates(num_candidates, max_chunk)
            self.max_stat = 2 * max(len(c) for c in self.chunks)
            self.slot_stride = self.max_stat + 1
            self.reductions_per_token = len(self.chunks) + (1 if fuse_norm else 0)

        def smem_bytes(self) -> int:
            red = self.RED_SLOTS * self.cwarps * self.slot_stride * 4
            return self.stages * self.H * 2 + red + (2 * self.stages + self.RED_SLOTS) * 8 + 256

        @cute.jit
        def __call__(
            self,
            mB: cute.Tensor,
            mP: cute.Tensor,
            mA: cute.Tensor,
            mR: cute.Tensor,
            mG: cute.Tensor,
            mW: cute.Tensor,
            mO: cute.Tensor,
            mU: cute.Tensor,
            M: Int32,
            grid_x: Int32,
            eps: Float32,
            eps_out: Float32,
            stream,
        ):
            vec = (1, _VEC)
            self.kernel(
                mB,
                mP,
                mA,
                cute.zipped_divide(mR, vec),
                cute.zipped_divide(mG, vec),
                cute.zipped_divide(mW, vec),
                cute.zipped_divide(mO, vec),
                cute.zipped_divide(mU, vec),
                M,
                eps,
                eps_out,
            ).launch(
                grid=[grid_x, 1, 1],
                block=[self.threads, 1, 1],
                smem=self.smem_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )

        @cute.jit
        def _post(
            self,
            vals: cute.Tensor,
            count: cutlass.Constexpr[int],
            buf: cute.Tensor,
            red_bar: cute.Pointer,
            j: Int32,
        ):
            """Post this warp's sums of ``vals[0:count]`` for reduction ``j``."""
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            for k in cutlass.range_constexpr(count):
                v = vals[k]
                for i in cutlass.range_constexpr(5):
                    v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << i)
                vals[k] = v
            if lane == 0:
                slot = j % self.RED_SLOTS
                for k in cutlass.range_constexpr(count):
                    buf[slot, warp * self.slot_stride + k] = vals[k]
                if cutlass.const_expr(self.overlap):
                    cute.arch.mbarrier_arrive(red_bar + slot)
            if cutlass.const_expr(not self.overlap):
                cute.arch.barrier(barrier_id=1, number_of_threads=self.consumers)

        @cute.jit
        def _collect(
            self,
            vals: cute.Tensor,
            count: cutlass.Constexpr[int],
            buf: cute.Tensor,
            red_bar: cute.Pointer,
            j: Int32,
        ):
            """Wait for reduction ``j`` and leave its CTA totals in ``vals[0:count]``."""
            lane = cute.arch.lane_idx()
            slot = j % self.RED_SLOTS
            if cutlass.const_expr(self.overlap):
                cute.arch.mbarrier_wait(red_bar + slot, (j // self.RED_SLOTS) & 1)
            total = Float32(0.0)
            if lane < count:
                for w in cutlass.range_constexpr(self.cwarps):
                    total = total + buf[slot, w * self.slot_stride + lane]
            for k in cutlass.range_constexpr(count):
                vals[k] = cute.arch.shuffle_sync(total, offset=k)

        @cute.kernel
        def kernel(
            self,
            mB: cute.Tensor,
            mP: cute.Tensor,
            mA: cute.Tensor,
            gR: cute.Tensor,
            gG: cute.Tensor,
            gW: cute.Tensor,
            gO: cute.Tensor,
            gU: cute.Tensor,
            M: Int32,
            eps: Float32,
            eps_out: Float32,
        ):
            N = self.N
            K = N - 1
            H = self.H
            V = self.V
            S = self.stages
            R = self.rows_per_token
            RED = self.reductions_per_token
            NCH = len(self.chunks)
            row_bytes = H * 2
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            gdim, _, _ = cute.arch.grid_dim()
            lane = cute.arch.lane_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

            smem = cutlass.utils.SmemAllocator()
            ring = smem.allocate_tensor(
                BFloat16, cute.make_layout((S, H), stride=(H, 1)), byte_alignment=128
            )
            red_cols = self.cwarps * self.slot_stride
            red_buf = smem.allocate_tensor(
                Float32,
                cute.make_layout((self.RED_SLOTS, red_cols), stride=(red_cols, 1)),
                byte_alignment=16,
            )
            full = smem.allocate_array(cutlass.Int64, num_elems=S)
            empty = smem.allocate_array(cutlass.Int64, num_elems=S)
            red_bar = smem.allocate_array(cutlass.Int64, num_elems=self.RED_SLOTS)
            if tidx == 0:
                for s in cutlass.range_constexpr(S):
                    cute.arch.mbarrier_init(full + s, 1)
                    cute.arch.mbarrier_init(empty + s, self.cwarps)
                for s in cutlass.range_constexpr(self.RED_SLOTS):
                    cute.arch.mbarrier_init(red_bar + s, self.cwarps)
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()

            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_wait()

            if warp == self.cwarps:
                if lane == 0:
                    it = Int32(0)
                    for tok in cutlass.range(bidx, M, gdim, unroll=1):
                        for i in cutlass.range_constexpr(R):
                            r = it * R + i
                            slot = r % S
                            cute.arch.mbarrier_wait(empty + slot, ((r // S) & 1) ^ 1)
                            cute.arch.mbarrier_arrive_and_expect_tx(full + slot, row_bytes)
                            if cutlass.const_expr(i < K):
                                src = (
                                    mB.iterator
                                    + (cutlass.Int64(i) * cutlass.Int64(M) + cutlass.Int64(tok)) * H
                                )
                            elif cutlass.const_expr(i == K):
                                src = mP.iterator + cutlass.Int64(tok) * H
                            else:
                                src = mA.iterator + cutlass.Int64(tok) * H
                            _bulk_g2s(ring.iterator + slot * H, src, full + slot, row_bytes)
                        it = it + 1
            else:
                sv = cute.zipped_divide(ring, (1, _VEC))
                q = []
                for v in cutlass.range_constexpr(V):
                    col = tidx + v * self.consumers
                    qv = cute.make_rmem_tensor(cute.make_layout((1, _VEC)), Float32)
                    qv.store(
                        gR[(None, (0, col))].load().to(Float32)
                        * gG[(None, (0, col))].load().to(Float32)
                    )
                    q.append(qv)
                w_out = []
                if cutlass.const_expr(self.fuse_norm):
                    for v in cutlass.range_constexpr(V):
                        wv = cute.make_rmem_tensor(cute.make_layout((1, _VEC)), Float32)
                        wv.store(gW[(None, (0, tidx + v * self.consumers))].load().to(Float32))
                        w_out.append(wv)
                acc = [
                    cute.make_rmem_tensor(cute.make_layout((1, _VEC)), Float32) for _ in range(V)
                ]
                xf = cute.make_rmem_tensor(cute.make_layout((1, _VEC)), Float32)
                stats = [
                    cute.make_rmem_tensor(cute.make_layout((self.max_stat,)), Float32),
                    cute.make_rmem_tensor(cute.make_layout((self.max_stat,)), Float32),
                ]
                it = Int32(0)
                for tok in cutlass.range(bidx, M, gdim, unroll=1):
                    m_run = Float32(0.0)
                    s_run = Float32(0.0)
                    pending = []

                    for c in cutlass.range_constexpr(NCH + 1):
                        if cutlass.const_expr(c < NCH):
                            cand = self.chunks[c]
                            nc = len(cand)
                            st = stats[c % 2]
                            xs = []
                            for n in cand:
                                r = it * R + n
                                slot = r % S
                                cute.arch.mbarrier_wait(full + slot, (r // S) & 1)
                                vals = [
                                    sv[(None, (slot, tidx + v * self.consumers))].load()
                                    for v in range(V)
                                ]
                                if cutlass.const_expr(self.fuse_add and n == K):
                                    slot_a = (r + 1) % S
                                    cute.arch.mbarrier_wait(full + slot_a, ((r + 1) // S) & 1)
                                    adds = [
                                        sv[(None, (slot_a, tidx + v * self.consumers))].load()
                                        for v in range(V)
                                    ]
                                    cute.arch.sync_warp()
                                    if lane == 0:
                                        cute.arch.mbarrier_arrive(empty + slot_a)
                                    for v in cutlass.range_constexpr(V):
                                        vals[v] = (vals[v].to(Float32) + adds[v].to(Float32)).to(
                                            BFloat16
                                        )
                                        gU[(None, (tok, tidx + v * self.consumers))].store(vals[v])
                                cute.arch.sync_warp()
                                if lane == 0:
                                    cute.arch.mbarrier_arrive(empty + slot)
                                xs.append(vals)
                            for jj in cutlass.range_constexpr(nc):
                                ss = (Float32(0.0), Float32(0.0))
                                dot = (Float32(0.0), Float32(0.0))
                                for v in cutlass.range_constexpr(V):
                                    xf.store(xs[jj][v].to(Float32))
                                    for i in cutlass.range_constexpr(0, _VEC, 2):
                                        pair = (xf[i], xf[i + 1])
                                        ss = cute.arch.fma_packed_f32x2(pair, pair, ss)
                                        dot = cute.arch.fma_packed_f32x2(
                                            pair, (q[v][i], q[v][i + 1]), dot
                                        )
                                st[jj] = ss[0] + ss[1]
                                st[nc + jj] = dot[0] + dot[1]
                            self._post(st, 2 * nc, red_buf, red_bar, it * RED + c)
                            pending.append((c, nc, xs))
                            if cutlass.const_expr(not self.overlap):
                                c_done = c
                            else:
                                c_done = c - 1
                        else:
                            c_done = NCH - 1 if self.overlap else -1

                        if cutlass.const_expr(c_done >= 0 and c_done == pending[0][0]):
                            cd, nc, xs = pending.pop(0)
                            st = stats[cd % 2]
                            self._collect(st, 2 * nc, red_buf, red_bar, it * RED + cd)
                            logits = []
                            for jj in cutlass.range_constexpr(nc):
                                rs = cute.math.rsqrt(st[jj] / Float32(H) + eps, fastmath=True)
                                logits.append(st[nc + jj] * rs)
                            m_new = logits[0]
                            for jj in cutlass.range_constexpr(1, nc):
                                m_new = cutlass.max(m_new, logits[jj])
                            if cutlass.const_expr(cd > 0):
                                m_new = cutlass.max(m_new, m_run)
                            wts = []
                            wsum = Float32(0.0)
                            for jj in cutlass.range_constexpr(nc):
                                wj = cute.math.exp2(
                                    (logits[jj] - m_new) * Float32(_LOG2E), fastmath=True
                                )
                                wts.append(wj)
                                wsum = wsum + wj
                            corr = Float32(0.0)
                            if cutlass.const_expr(cd > 0):
                                corr = cute.math.exp2(
                                    (m_run - m_new) * Float32(_LOG2E), fastmath=True
                                )
                            for v in cutlass.range_constexpr(V):
                                for jj in cutlass.range_constexpr(nc):
                                    xf.store(xs[jj][v].to(Float32))
                                    for i in cutlass.range_constexpr(0, _VEC, 2):
                                        pair = (xf[i], xf[i + 1])
                                        wpair = (wts[jj], wts[jj])
                                        if cutlass.const_expr(jj == 0 and cd == 0):
                                            out = cute.arch.mul_packed_f32x2(pair, wpair)
                                        elif cutlass.const_expr(jj == 0):
                                            old = cute.arch.mul_packed_f32x2(
                                                (acc[v][i], acc[v][i + 1]), (corr, corr)
                                            )
                                            out = cute.arch.fma_packed_f32x2(pair, wpair, old)
                                        else:
                                            out = cute.arch.fma_packed_f32x2(
                                                pair, wpair, (acc[v][i], acc[v][i + 1])
                                            )
                                        acc[v][i] = out[0]
                                        acc[v][i + 1] = out[1]
                            if cutlass.const_expr(cd == 0):
                                s_run = wsum
                            else:
                                s_run = s_run * corr + wsum
                            m_run = m_new

                    inv_s = Float32(1.0) / s_run
                    mix = [(acc[v].load() * inv_s).to(BFloat16) for v in range(V)]
                    if cutlass.const_expr(self.fuse_norm):
                        st = stats[0]
                        ss = (Float32(0.0), Float32(0.0))
                        for v in cutlass.range_constexpr(V):
                            xf.store(mix[v].to(Float32))
                            for i in cutlass.range_constexpr(0, _VEC, 2):
                                pair = (xf[i], xf[i + 1])
                                ss = cute.arch.fma_packed_f32x2(pair, pair, ss)
                        st[0] = ss[0] + ss[1]
                        self._post(st, 1, red_buf, red_bar, it * RED + RED - 1)
                        self._collect(st, 1, red_buf, red_bar, it * RED + RED - 1)
                        rstd = cute.math.rsqrt(st[0] / Float32(H) + eps_out, fastmath=True)
                        for v in cutlass.range_constexpr(V):
                            col = tidx + v * self.consumers
                            mf = mix[v].to(Float32)
                            w = w_out[v].load()
                            if cutlass.const_expr(self.norm_kimi):
                                y = ((mf * rstd).to(BFloat16).to(Float32) * w).to(BFloat16)
                            else:
                                y = (mf * rstd * w).to(BFloat16)
                            gO[(None, (tok, col))].store(y)
                    else:
                        for v in cutlass.range_constexpr(V):
                            gO[(None, (tok, tidx + v * self.consumers))].store(mix[v])
                    it = it + 1

            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()


# ("reg", cluster_n, vecs_per_thread) or ("pipe", vecs_per_thread, stages, max_chunk[, overlap]).
# Tuned on a 212-SM SM107: the register kernel wins up to 64 tokens, pipe V4 S14 c3 above that for every N.
SMALL_M_CONFIG = ("reg", 7, 1)
MID_M_CONFIG = ("reg", 4, 1)
PIPE_CONFIG = ("pipe", 4, 14, 3)


def default_config(num_tokens: int) -> Tuple:
    if num_tokens <= 16:
        return SMALL_M_CONFIG
    if num_tokens <= 64:
        return MID_M_CONFIG
    return PIPE_CONFIG


def _normalize_config(config: Tuple) -> Tuple:
    if len(config) == 2:
        return ("reg",) + tuple(config)
    if config[0] == "pipe" and len(config) == 4:
        return tuple(config) + (False,)
    return tuple(config)


@functools.lru_cache(maxsize=None)
def _compiled(
    num_candidates: int,
    hidden_size: int,
    config: Tuple,
    fuse_add: bool,
    fuse_norm: bool,
    norm_mode: str,
    enable_pdl: bool,
):
    if config[0] == "pipe":
        _, vecs, stages, max_chunk, overlap = config
        kernel = AttnResMixPipeKernel(
            num_candidates,
            hidden_size,
            vecs,
            stages,
            max_chunk,
            fuse_add,
            fuse_norm,
            norm_mode,
            enable_pdl,
            overlap,
        )
    else:
        _, cluster_n, vecs = config
        kernel = AttnResMixKernel(
            num_candidates, hidden_size, cluster_n, vecs, fuse_add, fuse_norm, norm_mode, enable_pdl
        )
    act = cute.runtime.make_fake_compact_tensor(
        BFloat16, (cute.sym_int(), hidden_size), stride_order=(1, 0), assumed_align=16
    )
    bank = cute.runtime.make_fake_compact_tensor(
        BFloat16, (cute.sym_int(), hidden_size), stride_order=(1, 0), assumed_align=16
    )
    wgt = cute.runtime.make_fake_compact_tensor(
        BFloat16, (1, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        bank,
        act,
        act,
        wgt,
        wgt,
        wgt,
        act,
        act,
        Int32(1),
        Int32(1),
        Float32(1e-5),
        Float32(1e-5),
        stream,
        options="--enable-tvm-ffi",
    )


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def attn_res_mix(
    prefix: torch.Tensor,
    bank: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    *,
    addend: Optional[torch.Tensor] = None,
    out_norm_weight: Optional[torch.Tensor] = None,
    out_norm_eps: float = 1e-5,
    norm_mode: str = "flashinfer",
    config: Optional[Tuple] = None,
    grid_x: int = 0,
    enable_pdl: bool = False,
    out: Optional[torch.Tensor] = None,
    updated: Optional[torch.Tensor] = None,
):
    """Attention-residual mix (+ optional fused add / output RMSNorm).

    prefix: bf16 [M, H]; bank: bf16 [K, M, H] (or anything viewable as [K * M, H]) with K = N - 1 >= 1;
    res_weight / rms_weight / out_norm_weight: bf16 [H].
    ``config`` defaults to :func:`default_config`. ``grid_x`` = number of token clusters (CTAs for
    "pipe") launched; 0 means one per token for "reg" and one per SM for "pipe".
    Returns ``out`` or, with ``addend``, ``(updated_prefix, out)``.
    """
    M, H = prefix.shape
    K = bank.numel() // (M * H)
    N = K + 1
    if not 2 <= N <= _MAX_CANDIDATES or bank.numel() != K * M * H:
        raise ValueError(
            f"attn_res_mix: unsupported bank shape {tuple(bank.shape)} for prefix [{M}, {H}]"
        )
    tensors = [prefix, bank, res_weight, rms_weight, addend, out_norm_weight, out, updated]
    for t in tensors:
        if t is not None and (
            t.dtype != torch.bfloat16 or not t.is_contiguous() or t.data_ptr() % 16
        ):
            raise ValueError("attn_res_mix: tensors must be contiguous 16-byte aligned bf16")
    config = _normalize_config(config or default_config(M))
    fuse_add = addend is not None
    fuse_norm = out_norm_weight is not None
    fn = _compiled(N, H, config, fuse_add, fuse_norm, norm_mode, enable_pdl)
    if grid_x <= 0:
        grid_x = min(M, _num_sms(prefix.device.index)) if config[0] == "pipe" else M
    if out is None:
        out = torch.empty_like(prefix)
    if fuse_add and updated is None:
        updated = torch.empty_like(prefix)
    r = res_weight.reshape(1, H)
    fn(
        bank.reshape(K * M, H),
        prefix,
        addend if fuse_add else prefix,
        r,
        rms_weight.reshape(1, H),
        out_norm_weight.reshape(1, H) if fuse_norm else r,
        out,
        updated if fuse_add else out,
        M,
        grid_x,
        eps,
        out_norm_eps,
    )
    return (updated, out) if fuse_add else out
