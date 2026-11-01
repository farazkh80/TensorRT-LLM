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
"""Fine-grained-sync (Lamport) modes of the Blackwell / Rubin MLA decode (split-KV kernel + LSE reduction).

``KimiK3MlaDecodeFgs`` subclasses ``BlackwellMultiHeadLatentAttentionForwardFP16``; with every flag off it launches
the base kernels unchanged. Fixed-Q decode only (no ``is_var_q``, no softmax stats).

``fgs_consumer`` (split-KV kernel, edges fused_q -> attention and current cache row -> attention):
  * PDL launch without ``griddepcontrol.wait``. The page table and ``cache_seqs`` are step-level inputs and are
    trusted; KV history is trusted and its TMA loads start at once.
  * Q (fused_q latent + rope columns) is loaded with TMA ``report_valid`` into a V1 full barrier. The MMA waits on
    the conditional phase, so it only consumes valid Q. Warp 11 (idle in the base kernel) is the retry warp: in
    the leader CTA it waits on the primary phase; when the load completed with a sentinel chunk it re-arms the
    barrier, re-issues its half of the 2-CTA load into the same stage and tells the peer CTA (DSMEM message) to
    re-issue its half, because only the leader's barrier receives the validity report. The load warp cannot do
    this itself: it blocks in ``producer_acquire`` once the KV stages are full while the MMA waits for Q.
  * The current cache rows (positions ``K - seq_len_q .. K - 1``, written this step with 16-B stores) are polled by
    the page-table warp, every element of the 1152-B row against the bf16 sentinel (``mla_fgs.poll_chunks``),
    before it publishes the page-table stage of the KV tile that holds them; the load warp issues that tile's TMA
    loads only after consuming the stage. Rows past ``K`` in the same page are never checked (they may hold
    anything, including -0.0). Both CTAs of a cluster poll, since each loads part of the tile.

``fgs_producer`` (split-KV kernel, edges -> reduction and, for rows planned onto one split, -> W_UV): triggers
dependents on entry; the bf16 output of single-split rows is converted with ``cvt.rn.pzo`` and stored as 16-B
chunks. How fp32 partials are published depends on ``fgs_partials``:
  * "lse" (default): the partial LSE of each (row, split) is the flag. The split kernel stores partial O, every
    correction thread runs ``fence.acq_rel.gpu``, the CTA's correction warps meet at a named barrier, then the
    canonical LSE is stored with ``st.relaxed.gpu``. Only the LSE entries need the sentinel.
  * "all": partial O and LSE are canonicalized and every element is a flag (the whole read set needs the sentinel).

``fgs_reduction`` (reduction kernel, both sides): PDL launch without ``griddepcontrol.wait``, trigger on entry.
Warp 0 waits (warp-uniform) until no partial LSE it reads is the fp32 sentinel; in "lse" mode the loads are
``ld.relaxed.gpu`` followed by ``fence.acq_rel.gpu`` (acquire pattern) before the CTA barrier that precedes the O
loads, which are ``ld.global.cg``; in "all" mode every O element is checked as well. Each thread reduces 4
consecutive D elements (same per-element summation order as the base kernel) and even lanes store 16-B chunks of
8 bf16 (``cvt.rn.pzo``), because the W_UV TMA ``report_valid`` load checks only element 0 of a chunk.

``pdl_early``: kernels without their FGS consumer flag are launched with PDL, trigger on entry and wait on entry.

Framework contract (sentinel = -0.0; bf16 0x8000, fp32 0x80000000):
  * fused_q (all 576 columns of the step's rows) holds the sentinel before W_UK / latent append write it.
  * The current cache rows of every MLA layer hold the sentinel before the latent append writes them
    (``fill_current_rows_sentinel``).
  * "lse": the partial LSE entries read by the reduction (rows < H, splits < the request's split count) hold the
    sentinel before the split kernel writes them (``fill_lse_sentinel``; bytes ``lse_flag_bytes``). The partial O
    region is never reset. "all": the whole read set, O and LSE (``fill_partials_sentinel``;
    ``partial_read_set_bytes``).
  * The attention output (W_UV input) holds the sentinel before the split kernel / reduction write it.
  * No producer of these buffers emits -0.0.
"""

from types import SimpleNamespace
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

try:
    from tensorrt_llm._torch.attention.kernels.blackwell.mla.mla_decode_fp16 import (
        BlackwellMultiHeadLatentAttentionForwardFP16,
    )
    from tensorrt_llm._torch.attention.kernels.blackwell.mla.mla_helpers import (
        MLAStaticTileSchedulerParams,
        create_mla_static_tile_scheduler,
    )
except ImportError:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp16 import (
        BlackwellMultiHeadLatentAttentionForwardFP16,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_helpers import (
        MLAStaticTileSchedulerParams,
        create_mla_static_tile_scheduler,
    )

try:
    from .mla_fgs import (
        BF16_SENTINEL,
        F32_SENTINEL,
        SPIN_NS_MAX,
        SPIN_NS_MIN,
        bf16x2_pzo,
        canon_f32,
        load_chunks,
        nanosleep,
        poll_chunks,
        st_global_v4,
        trap,
    )
except ImportError:
    from mla_fgs import (
        BF16_SENTINEL,
        F32_SENTINEL,
        SPIN_NS_MAX,
        SPIN_NS_MIN,
        bf16x2_pzo,
        canon_f32,
        load_chunks,
        nanosleep,
        poll_chunks,
        st_global_v4,
        trap,
    )

_Base = BlackwellMultiHeadLatentAttentionForwardFP16

LATENT_DIM = 512
ROPE_DIM = 64
ROW_CHUNKS = (LATENT_DIM + ROPE_DIM) * 2 // 16
STATS_LEN = 16
# Stats layout (int32): [0] Q re-issues, [1] clusters that re-issued Q, [2:5] current-row poll (sweeps, warps
# that waited, chunks reloaded), [8:11] reduction partial-O poll, [11:14] reduction partial-LSE poll.
_STATS_Q = 0
_STATS_ROW = 2
_STATS_RED_O = 8
_STATS_RED_LSE = 11
_MSG_DONE = 0
_MSG_REISSUE = 1
_MSG_BYTES = 16
_EXTRA_SMEM = 128
_RED_UNROLL = 4
_FILL_LSE_THREADS = 256


@cute.struct
class _FgsSharedStorage:
    # [0] message full (peer side, completed by the leader's st.async), [1] message empty (leader side, 32 peer
    # lanes arrive after reading).
    msg_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    msg: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]


@dsl_user_op
def _ld_cg_b32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.b32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_relaxed_b32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_b32(addr, w, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_relaxed_f32(addr, x, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "st.relaxed.gpu.global.f32 [$0], $1;",
        "l,f,~{memory}",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _as_f32(w, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.bitcast(T.f32(), cutlass.Int32(w).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@cute.jit
def _poll_words(
    bits: cute.Tensor,
    addrs: cute.Tensor,
    n: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    stats: cute.Tensor,
    with_stats: cutlass.Constexpr,
):
    """Warp-uniform wait until no lane's ``n`` fp32 words (int32 rmem ``bits``, already loaded) is the sentinel.

    Lanes without work hold non-sentinel bits. Same protocol and stats layout as ``mla_fgs.poll_chunks`` for 4-B
    elements: reloads are ``ld.relaxed.gpu`` of the pending words only.
    """
    pending = cutlass.Int32(0)
    for i in cutlass.range_constexpr(n):
        pending = pending | cutlass.Int32(bits[i] == cutlass.Int32(F32_SENTINEL))
    sweeps = cutlass.Int32(0)
    reloads = cutlass.Int32(0)
    if cute.arch.vote_any_sync(pending != cutlass.Int32(0)):
        sleep_ns = cutlass.Int32(SPIN_NS_MIN)
        again = cutlass.Boolean(True)
        while again:
            nanosleep(sleep_ns)
            sleep_ns = cutlass.min(sleep_ns * 2, cutlass.Int32(SPIN_NS_MAX))
            pending = cutlass.Int32(0)
            for i in cutlass.range_constexpr(n):
                if bits[i] == cutlass.Int32(F32_SENTINEL):
                    bits[i] = _ld_relaxed_b32(addrs[i])
                    reloads = reloads + 1
                    pending = pending | cutlass.Int32(bits[i] == cutlass.Int32(F32_SENTINEL))
            sweeps = sweeps + 1
            if cutlass.const_expr(spin_limit > 0):
                if sweeps > cutlass.Int32(spin_limit):
                    trap()
            again = cute.arch.vote_any_sync(pending != cutlass.Int32(0))
        if cutlass.const_expr(with_stats):
            reloads = cute.arch.warp_reduction_sum(reloads)
            if cute.arch.lane_idx() == 0:
                cute.arch.atomic_add(stats.iterator, sweeps)
                cute.arch.atomic_add(stats.iterator + 1, cutlass.Int32(1))
                cute.arch.atomic_add(stats.iterator + 2, reloads)
    return sweeps


class _PdlLaunch:
    def __init__(self, launcher, extra_smem: int = 0):
        self._launcher = launcher
        self._extra_smem = extra_smem

    def launch(self, **kwargs):
        if self._extra_smem:
            kwargs["smem"] = kwargs["smem"] + self._extra_smem
        return self._launcher.launch(use_pdl=True, **kwargs)


class _NoLaunch:
    def launch(self, **kwargs):
        return None


class KimiK3MlaDecodeFgs(BlackwellMultiHeadLatentAttentionForwardFP16):
    """MLA decode with opt-in fine-grained-sync consumer / producer modes (module docstring)."""

    def __init__(
        self,
        *args,
        fgs_consumer: bool = False,
        fgs_producer: bool = False,
        fgs_reduction: bool = False,
        fgs_partials: str = "lse",
        pdl_early: bool = False,
        fgs_stats: bool = False,
        spin_limit: int = 0,
        launch: tuple = ("split", "reduction"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fgs_consumer = fgs_consumer
        self.fgs_producer = fgs_producer
        self.fgs_reduction = fgs_reduction
        self.fgs_partials = fgs_partials
        self.pdl_early = pdl_early
        self.fgs_stats = fgs_stats
        self.spin_limit = spin_limit
        self.fgs_launch = frozenset(launch)
        self._fgs_split = fgs_consumer or fgs_producer or pdl_early
        self._fgs_red = fgs_reduction or pdl_early
        if self._fgs_split or self._fgs_red:
            if self.is_var_q or self.emit_softmax_stats:
                raise ValueError(
                    "FGS MLA decode supports fixed-Q decode without softmax stats only"
                )
            if self.is_persistent and fgs_consumer:
                raise ValueError("FGS MLA decode consumer supports the non-persistent grid only")
        if fgs_partials not in ("lse", "all"):
            raise ValueError(f"fgs_partials must be 'lse' or 'all', got {fgs_partials!r}")
        if fgs_consumer and getattr(pipeline, "PipelineTmaUmmaRetry", None) is None:
            raise ValueError(
                "fgs_consumer needs a CuTe DSL with TMA report_valid (PipelineTmaUmmaRetry)"
            )
        if fgs_reduction and "split" in self.fgs_launch and not fgs_producer:
            raise ValueError(
                "fgs_reduction reads sentinel-checked partials: the split kernel needs fgs_producer"
            )
        if fgs_stats and not (fgs_consumer or fgs_reduction):
            raise ValueError(
                "fgs_stats counts consumer waits: enable fgs_consumer or fgs_reduction"
            )

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        kv_bounds: Optional[cute.Tensor],
        cum_seq_lens_q: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.run_fgs(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table,
            o,
            lse,
            workspace,
            split_kv,
            cache_seqs,
            kv_bounds,
            cum_seq_lens_q,
            block_split_kvs,
            softmax_scale,
            output_scale,
            None,
            stream,
        )

    @cute.jit
    def run_fgs(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        kv_bounds: Optional[cute.Tensor],
        cum_seq_lens_q: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        fgs_stats: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        """``__call__`` with an optional int32 stats tensor of ``STATS_LEN`` entries (needs ``fgs_stats``)."""
        stats = fgs_stats
        if cutlass.const_expr(not self.fgs_stats):
            stats = None
        self._fgs_c_latent = c_latent
        self._fgs_stats_tensor = stats
        self._run(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table,
            o,
            lse,
            None,
            workspace,
            split_kv,
            cache_seqs,
            kv_bounds,
            cum_seq_lens_q,
            block_split_kvs,
            softmax_scale,
            output_scale,
            stream,
        )

    def split_kv_kernel(self, *args):
        if "split" not in self.fgs_launch:
            return _NoLaunch()
        if not self._fgs_split:
            return _Base.split_kv_kernel(self, *args)
        extra = _EXTRA_SMEM if self.fgs_consumer else 0
        return _PdlLaunch(
            self._fgs_split_kv_kernel(*args, self._fgs_c_latent, self._fgs_stats_tensor), extra
        )

    def reduction_kernel(self, *args):
        if "reduction" not in self.fgs_launch:
            return _NoLaunch()
        if not self._fgs_red:
            return _Base.reduction_kernel(self, *args)
        return _PdlLaunch(self._fgs_reduction_kernel(*args, self._fgs_stats_tensor))

    def issue_q_tma_load(self, *args):
        if self.fgs_consumer:
            return self._fgs_issue_q_tma_load(*args)
        return _Base.issue_q_tma_load(self, *args)

    def load_tma_qk_one_k_tile(self, *args, **kwargs):
        if self.fgs_consumer:
            return self._fgs_load_tma_qk_one_k_tile(*args, **kwargs)
        return _Base.load_tma_qk_one_k_tile(self, *args, **kwargs)

    def epilogue(self, *args):
        if self.fgs_producer:
            return self._fgs_epilogue(*args)
        return _Base.epilogue(self, *args)

    @staticmethod
    def partial_read_set_bytes(num_heads: int, splits_per_request) -> int:
        """Bytes of the split-KV partial workspace the reduction reads (rows < H, D + LSE, fp32).

        ``splits_per_request`` lists each request's effective split count; requests with one split are stored
        directly by the split kernel and read nothing.
        """
        return sum(num_heads * s * (LATENT_DIM + 1) * 4 for s in splits_per_request if s > 1)

    @staticmethod
    def lse_flag_bytes(num_heads: int, splits_per_request) -> int:
        """Bytes ``fill_lse_sentinel`` writes (the partial LSE entries the reduction reads)."""
        return sum(num_heads * s * 4 for s in splits_per_request if s > 1)

    @cute.kernel
    def _fgs_split_kv_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: Optional[cute.CopyAtom],
        mQL: cute.Tensor,
        tma_atom_q_rope: Optional[cute.CopyAtom],
        mQR: cute.Tensor,
        tma_atom_c_latent: Optional[cute.CopyAtom],
        mCL: cute.Tensor,
        tma_atom_c_rope: Optional[cute.CopyAtom],
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: Optional[cute.CopyAtom],
        mCLT: cute.Tensor,
        mPT: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        kv_bounds: Optional[cute.Tensor],
        cum_seq_lens_q: Optional[cute.Tensor],
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        q_latent_smem_layout_staged: cute.ComposedLayout,
        q_rope_smem_layout_staged: cute.ComposedLayout,
        kc_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        kc_smem_layout_for_tma: cute.ComposedLayout,
        vc_smem_layout_for_tma: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: MLAStaticTileSchedulerParams,
        SharedStorage: cutlass.Constexpr,
        mCLraw: cute.Tensor,
        mStats: Optional[cute.Tensor],
    ):
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(not self.fgs_consumer):
            cute.arch.griddepcontrol_wait()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        if warp_idx == self.mma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_q_latent)
            cpasync.prefetch_descriptor(tma_atom_q_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent)
            cpasync.prefetch_descriptor(tma_atom_c_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        fgs_storage = None
        if cutlass.const_expr(self.fgs_consumer):
            fgs_storage = smem.allocate(_FgsSharedStorage)

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_ptr_sync_bar,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        if cutlass.const_expr(self.fgs_consumer):
            load_q_pipeline = pipeline.PipelineTmaUmmaRetry.create(
                barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
                num_stages=self.load_q_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group_retry=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=self.tma_copy_q_bytes,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            load_q_pipeline = self.make_and_init_load_qkv_pipeline(
                storage.load_q_mbar_ptr.data_ptr(),
                cta_layout_vmnk,
                self.load_q_stage,
                self.tma_copy_q_bytes,
            )
        load_kv_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_kv_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_kv_stage,
            self.tma_copy_kc_bytes,
        )
        mma_s_pipeline = self.make_and_init_mma_s_pipeline(
            storage.mma_s_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        p_mma_pipeline = self.make_and_init_p_mma_pipeline(
            storage.p_mma_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        p_cor_pipeline = self.make_and_init_p_cor_pipeline(storage.p_cor_mbar_ptr.data_ptr())
        mma_o_pipeline = self.make_and_init_mma_o_pipeline(
            storage.mma_o_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        load_pt_pipeline = self.make_and_init_load_pt_pipeline(storage.load_pt_mbar_ptr.data_ptr())

        if cutlass.const_expr(self.fgs_consumer):
            if warp_idx == self.empty_warp_ids[0]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(fgs_storage.msg_mbar.data_ptr(), 1)
                    cute.arch.mbarrier_init(
                        fgs_storage.msg_mbar.data_ptr() + 1, self.threads_per_warp
                    )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)

        sQ = storage.smem_q_latent.get_tensor(
            q_latent_smem_layout_staged.outer, swizzle=q_latent_smem_layout_staged.inner
        )
        sQ_rope = storage.smem_q_rope.get_tensor(
            q_rope_smem_layout_staged.outer, swizzle=q_rope_smem_layout_staged.inner
        )
        sKC = storage.smem_kc.get_tensor(
            kc_smem_layout_staged.outer, swizzle=kc_smem_layout_staged.inner
        )
        sKC_for_tma = storage.smem_kc.get_tensor(
            kc_smem_layout_for_tma.outer, swizzle=kc_smem_layout_for_tma.inner
        )
        sVC_ptr = cute.recast_ptr(sKC.iterator, vc_smem_layout_staged.inner)
        sVC = cute.make_tensor(sVC_ptr, vc_smem_layout_staged.outer)
        sVC_for_tma = cute.make_tensor(sVC_ptr, vc_smem_layout_for_tma.outer)
        sP = storage.smem_p.get_tensor(
            p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner
        )
        sPT = storage.smem_page_table.get_tensor(
            cute.make_layout((self.mma_qk_tiler[1] // 2, self.load_pt_stage))
        )
        softmax_smem_exchange = storage.softmax_smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )
        epilogue_smem_exchange = storage.epilogue_smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

        if warp_idx >= self.empty_warp_ids[0] and warp_idx <= self.empty_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            if cutlass.const_expr(self.fgs_consumer):
                if warp_idx == self.empty_warp_ids[0]:
                    retry_params = SimpleNamespace(
                        is_leader_cta=is_leader_cta,
                        mma_tile_coord_v=mma_tile_coord_v,
                        load_q_pipeline=load_q_pipeline,
                        msg_mbar=fgs_storage.msg_mbar.data_ptr(),
                        msg=fgs_storage.msg.data_ptr(),
                        tile_sched_params=tile_sched_params,
                        split_kv=split_kv,
                        cache_seqs=cache_seqs,
                        cum_seq_lens_q=cum_seq_lens_q,
                        block_split_kvs=block_split_kvs,
                        tiled_mma_qk=tiled_mma_qk,
                        tma_atom_q_latent=tma_atom_q_latent,
                        tma_atom_q_rope=tma_atom_q_rope,
                        mQL=mQL,
                        mQR=mQR,
                        sQ=sQ,
                        sQ_rope=sQ_rope,
                        mStats=mStats,
                    )
                    self._fgs_q_retry_warp(retry_params)
        if warp_idx == self.load_pt_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            load_pt_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_pt_stage
            )
            tile_sched = create_mla_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv, _, _, _ = self.get_k_tile_count(
                    split_kv, cache_seqs, cum_seq_lens_q, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    load_pt_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        load_pt_pipeline=load_pt_pipeline,
                        mPT=mPT,
                        sPT=sPT,
                        tidx=tidx,
                        page_size=mCL.shape[0],
                        K=cache_seqs[blk_coord[2]],
                        mCLraw=mCLraw,
                        mStats=mStats,
                    )
                    if cutlass.const_expr(self.fgs_consumer):
                        load_pt_producer_state = self._fgs_load_page_table(
                            load_pt_common_params, k_index, k_tile_count, load_pt_producer_state
                        )
                    else:
                        load_pt_producer_state = self.load_page_table(
                            load_pt_common_params, k_index, k_tile_count, load_pt_producer_state
                        )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            load_pt_pipeline.producer_tail(load_pt_producer_state)
        if warp_idx == self.load_tma_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            load_q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_q_stage
            )
            load_kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_kv_stage
            )
            load_pt_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_pt_stage
            )
            load_pt_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_pt_stage
            )
            tile_sched = create_mla_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv, q_begin, _, _ = self.get_k_tile_count(
                    split_kv, cache_seqs, cum_seq_lens_q, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    tma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        q_begin=q_begin,
                        load_q_pipeline=load_q_pipeline,
                        load_kv_pipeline=load_kv_pipeline,
                        mPT=mPT,
                        sPT=sPT,
                        load_pt_pipeline=load_pt_pipeline,
                        K=cache_seqs[blk_coord[2]],
                    )
                    tma_qk_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        tma_atom_q_latent=tma_atom_q_latent,
                        tma_atom_q_rope=tma_atom_q_rope,
                        tma_atom_c_latent=tma_atom_c_latent,
                        tma_atom_c_rope=tma_atom_c_rope,
                        mQL=mQL,
                        mQR=mQR,
                        mCL=mCL,
                        mKR=mKR,
                        sQ=sQ,
                        sQ_rope=sQ_rope,
                        sKC=sKC_for_tma,
                    )
                    tma_pv_params = SimpleNamespace(
                        tiled_mma_pv=tiled_mma_pv,
                        tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
                        mCL=mCL,
                        mKR=mKR,
                        mCLT=mCLT,
                        sVC=sVC_for_tma,
                    )
                    (
                        load_q_producer_state,
                        load_kv_producer_state,
                        load_pt_consumer_state,
                        load_pt_release_state,
                    ) = self.load_tma(
                        tma_common_params,
                        tma_qk_params,
                        tma_pv_params,
                        k_index,
                        k_tile_count,
                        load_q_producer_state,
                        load_kv_producer_state,
                        load_pt_consumer_state,
                        load_pt_release_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            load_q_pipeline.producer_tail(load_q_producer_state)
            load_kv_pipeline.producer_tail(load_kv_producer_state)

        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            load_q_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_q_stage
            )
            load_kv_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_kv_stage
            )
            mma_s_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_s_stage
            )
            p_mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_mma_stage
            )
            mma_o_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_o_stage
            )
            tile_sched = create_mla_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv, _, _, _ = self.get_k_tile_count(
                    split_kv, cache_seqs, cum_seq_lens_q, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    mma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_q_pipeline=load_q_pipeline,
                        load_kv_pipeline=load_kv_pipeline,
                        tmem_ptr=tmem_ptr,
                        is_leader_cta=is_leader_cta,
                        L=mCL.shape[1],
                    )
                    mma_qk_params = SimpleNamespace(
                        mma_s_pipeline=mma_s_pipeline,
                        sQ=sQ,
                        sQ_rope=sQ_rope,
                        sKC=sKC,
                    )
                    mma_pv_params = SimpleNamespace(
                        p_mma_pipeline=p_mma_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                        sP=sP,
                        sVC=sVC,
                    )
                    (
                        tiled_mma_qk,
                        tiled_mma_pv,
                        load_q_consumer_state,
                        load_kv_consumer_state,
                        mma_s_producer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    ) = self.mma(
                        mma_common_params,
                        mma_qk_params,
                        mma_pv_params,
                        k_tile_count,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        load_q_consumer_state,
                        load_kv_consumer_state,
                        mma_s_producer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mma_s_pipeline.producer_tail(mma_s_producer_state)
            mma_o_pipeline.producer_tail(mma_o_producer_state)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.softmax_reg_num)
            mma_s_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_s_stage
            )
            p_mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_mma_stage
            )
            p_cor_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_cor_stage
            )
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tile_sched = create_mla_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv, q_begin, q_len, _ = self.get_k_tile_count(
                    split_kv, cache_seqs, cum_seq_lens_q, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=softmax_smem_exchange,
                        mAccO=mAccO,
                        mO=mO,
                        K=cache_seqs[blk_coord[2]],
                        kv_bounds=kv_bounds,
                        q_begin=q_begin,
                        q_len=q_len,
                        L=mCL.shape[1],
                        tmem_ptr=tmem_ptr,
                        tidx=tidx,
                        p_cor_pipeline=p_cor_pipeline,
                    )
                    compute_softmax_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        sP=sP,
                        mma_s_pipeline=mma_s_pipeline,
                        p_mma_pipeline=p_mma_pipeline,
                        softmax_scale_log2=softmax_scale_log2,
                    )
                    mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state = self.compute(
                        compute_common_params,
                        compute_softmax_params,
                        k_index=k_index,
                        k_tile_count=k_tile_count,
                        mma_s_consumer_state=mma_s_consumer_state,
                        p_mma_producer_state=p_mma_producer_state,
                        p_cor_producer_state=p_cor_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            p_cor_pipeline.producer_tail(p_cor_producer_state)

        if warp_idx >= self.correction_warp_ids[0] and warp_idx <= self.correction_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.correction_reg_num)
            p_cor_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tile_sched = create_mla_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv, q_begin, q_len, valid_q_rows = (
                    self.get_k_tile_count(
                        split_kv, cache_seqs, cum_seq_lens_q, block_split_kvs, blk_coord
                    )
                )
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=epilogue_smem_exchange,
                        mAccO=mAccO,
                        mO=mO,
                        K=cache_seqs[blk_coord[2]],
                        kv_bounds=kv_bounds,
                        q_begin=q_begin,
                        q_len=q_len,
                        L=mCL.shape[1],
                        H=valid_q_rows,
                        tmem_ptr=tmem_ptr,
                        tidx=tidx,
                        tiled_mma_pv=tiled_mma_pv,
                        p_cor_pipeline=p_cor_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                    )
                    compute_epilogue_params = SimpleNamespace(
                        output_scale=output_scale,
                        softmax_scale_log2=softmax_scale_log2,
                        mAccLSE=mAccLSE,
                        mLSE=mLSE,
                    )
                    if cutlass.const_expr(self.is_var_split_kv and mAccO is not None):
                        direct_common_params = SimpleNamespace(
                            **{**vars(compute_common_params), "mAccO": None}
                        )
                        direct_epilogue_params = SimpleNamespace(
                            **{**vars(compute_epilogue_params), "mAccLSE": None}
                        )
                        if local_split_kv == 1:
                            p_cor_consumer_state, mma_o_consumer_state = self.correction(
                                direct_common_params,
                                direct_epilogue_params,
                                k_tile_count=k_tile_count,
                                p_cor_consumer_state=p_cor_consumer_state,
                                mma_o_consumer_state=mma_o_consumer_state,
                            )
                        else:
                            p_cor_consumer_state, mma_o_consumer_state = self.correction(
                                compute_common_params,
                                compute_epilogue_params,
                                k_tile_count=k_tile_count,
                                p_cor_consumer_state=p_cor_consumer_state,
                                mma_o_consumer_state=mma_o_consumer_state,
                            )
                    else:
                        p_cor_consumer_state, mma_o_consumer_state = self.correction(
                            compute_common_params,
                            compute_epilogue_params,
                            k_tile_count=k_tile_count,
                            p_cor_consumer_state=p_cor_consumer_state,
                            mma_o_consumer_state=mma_o_consumer_state,
                        )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        return

    @cute.jit
    def _fgs_q_partitions(
        self, p: SimpleNamespace, v: cutlass.Int32, q_tile: cutlass.Int32, batch: cutlass.Int32
    ):
        """Q latent / rope TMA partitions of CTA ``v`` of the pair for one (q tile, request); see ``load_tma``."""
        gQL = cute.flat_divide(p.mQL, cute.select(self.mma_qk_tiler, mode=[0, 2]))
        gQR = cute.flat_divide(p.mQR, cute.select(self.mma_qk_rope_tiler, mode=[0, 2]))
        thr_mma_qk = p.tiled_mma_qk.get_slice(v % cute.size(p.tiled_mma_qk.thr_id))
        tSgQL = thr_mma_qk.partition_A(gQL)
        tSgQR = thr_mma_qk.partition_A(gQR)
        tQsQ, tQLgQL_mkl = cpasync.tma_partition(
            p.tma_atom_q_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(p.sQ, 0, 3),
            cute.group_modes(tSgQL, 0, 3),
        )
        tQsQ_rope, tQRgQR_mkl = cpasync.tma_partition(
            p.tma_atom_q_rope,
            0,
            cute.make_layout(1),
            cute.group_modes(p.sQ_rope, 0, 3),
            cute.group_modes(tSgQR, 0, 3),
        )
        tQLgQL = tQLgQL_mkl[None, q_tile, None, batch]
        tQRgQR = tQRgQR_mkl[None, q_tile, None, batch]
        return tQLgQL, tQsQ, tQRgQR, tQsQ_rope

    @cute.jit
    def _fgs_reissue_q(self, p: SimpleNamespace, q_tile: cutlass.Int32, batch: cutlass.Int32):
        tQLgQL, tQsQ, tQRgQR, tQsQ_rope = self._fgs_q_partitions(
            p, p.mma_tile_coord_v, q_tile, batch
        )
        self._fgs_issue_q_tma_load(
            p.tma_atom_q_latent,
            tQLgQL,
            tQsQ,
            p.tma_atom_q_rope,
            tQRgQR,
            tQsQ_rope,
            p.load_q_pipeline.sync_object_full.get_barrier(0),
        )

    @cute.jit
    def _fgs_q_retry_warp(self, p: SimpleNamespace):
        # The 2-CTA Q load completes (and reports validity) only on the leader's full barrier. Leader: wait for each
        # attempt's primary phase; valid -> release the stage (second arrival on both CTAs' empty barriers) and
        # send DONE; invalid -> send REISSUE, re-arm the full barrier and re-issue its own half. Peer: for each work
        # tile with K tiles, re-issue its half on REISSUE until DONE. Messages ping-pong on msg_mbar (one in flight);
        # the leader waits for the last acknowledgement before exiting so no remote arrive targets an exited CTA.
        if p.is_leader_cta:
            self._fgs_q_retry_leader(p)
        else:
            self._fgs_q_retry_peer(p)

    @cute.jit
    def _fgs_q_retry_leader(self, p: SimpleNamespace):
        msg_full = p.msg_mbar
        msg_empty = p.msg_mbar + 1
        peer = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) ^ 1
        q_pipe = p.load_q_pipeline
        q_full = q_pipe.sync_object_full.get_barrier(0)
        msg = p.msg
        tile_sched = create_mla_static_tile_scheduler(
            p.tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        q_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_q_stage
        )
        empty_phase = cutlass.Int32(1)
        sent = cutlass.Int32(0)
        reissues = cutlass.Int32(0)
        while work_tile.is_valid_tile:
            blk_coord = work_tile.tile_idx
            k_tile_count = self.get_k_tile_count(
                p.split_kv, p.cache_seqs, p.cum_seq_lens_q, p.block_split_kvs, blk_coord
            )[1]
            if k_tile_count > 0:
                sleep_ns = cutlass.Int32(SPIN_NS_MIN)
                pending = cutlass.Boolean(True)
                while pending:
                    invalid = cute.arch.mbarrier_wait_primary(q_full, q_state.phase)[1]
                    code = cutlass.Int32(_MSG_DONE)
                    if invalid:
                        code = cutlass.Int32(_MSG_REISSUE)
                        reissues = reissues + 1
                        if cutlass.const_expr(self.spin_limit > 0):
                            if reissues > cutlass.Int32(self.spin_limit):
                                trap()
                        nanosleep(sleep_ns)
                        sleep_ns = cutlass.min(sleep_ns * 2, cutlass.Int32(SPIN_NS_MAX))
                    cute.arch.mbarrier_wait(msg_empty, empty_phase)
                    empty_phase = empty_phase ^ 1
                    sent = sent + 1
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            msg_full, _MSG_BYTES, peer, relaxed=True, scope=_cluster_scope()
                        )
                        cute.arch.store_async_dsmem(
                            msg,
                            (code, blk_coord[1], blk_coord[2], cutlass.Int32(0)),
                            msg_full,
                            peer,
                        )
                    if invalid:
                        q_pipe.retry_expect_tx(q_state, self.tma_copy_q_bytes)
                        self._fgs_reissue_q(p, blk_coord[1], blk_coord[2])
                    else:
                        q_pipe.retry_release(q_state)
                        pending = cutlass.Boolean(False)
                    q_state.advance()
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        if sent > 0:
            cute.arch.mbarrier_wait(msg_empty, empty_phase)
        if cutlass.const_expr(self.fgs_stats):
            if reissues > 0:
                if cute.arch.lane_idx() == 0:
                    cute.arch.atomic_add(p.mStats.iterator + _STATS_Q, reissues)
                    cute.arch.atomic_add(p.mStats.iterator + _STATS_Q + 1, cutlass.Int32(1))

    @cute.jit
    def _fgs_q_retry_peer(self, p: SimpleNamespace):
        msg_full = p.msg_mbar
        msg_empty = p.msg_mbar + 1
        peer = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) ^ 1
        tile_sched = create_mla_static_tile_scheduler(
            p.tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        msg = p.msg
        full_phase = cutlass.Int32(0)
        while work_tile.is_valid_tile:
            blk_coord = work_tile.tile_idx
            k_tile_count = self.get_k_tile_count(
                p.split_kv, p.cache_seqs, p.cum_seq_lens_q, p.block_split_kvs, blk_coord
            )[1]
            if k_tile_count > 0:
                pending = cutlass.Boolean(True)
                while pending:
                    cute.arch.mbarrier_wait(msg_full, full_phase)
                    full_phase = full_phase ^ 1
                    code = msg[0]
                    q_tile = msg[1]
                    batch = msg[2]
                    _fence_release_cluster_smem()
                    cute.arch.mbarrier_arrive(msg_empty, peer, relaxed=True, scope=_cluster_scope())
                    if code == cutlass.Int32(_MSG_REISSUE):
                        self._fgs_reissue_q(p, q_tile, batch)
                    else:
                        pending = cutlass.Boolean(False)
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

    @cute.jit
    def _fgs_issue_q_tma_load(
        self,
        tma_atom_q_latent: cute.CopyAtom,
        tQLgQL: cute.Tensor,
        tQsQ: cute.Tensor,
        tma_atom_q_rope: cute.CopyAtom,
        tQRgQR: cute.Tensor,
        tQsQ_rope: cute.Tensor,
        tma_bar_ptr,
    ):
        for i in cutlass.range(self.iterations_qk_latent):
            cute.copy(
                tma_atom_q_latent,
                tQLgQL[None, i],
                tQsQ[None, (i, 0)],
                tma_bar_ptr=tma_bar_ptr,
                report_valid=True,
            )
        for i in cutlass.range(self.iterations_qk_rope):
            cute.copy(
                tma_atom_q_rope,
                tQRgQR[None, i],
                tQsQ_rope[None, i],
                tma_bar_ptr=tma_bar_ptr,
                report_valid=True,
            )

    @cute.jit
    def _fgs_poll_cache_row(
        self, common_params: SimpleNamespace, row: cutlass.Int32, lane: cutlass.Int32
    ):
        """Warp-uniform wait until no element of cache row ``row`` of this request is the bf16 sentinel."""
        n = (ROW_CHUNKS + self.threads_per_warp - 1) // self.threads_per_warp
        mCLraw = common_params.mCLraw
        page = common_params.mPTfull[row // self.page_size, common_params.blk_coord[2]]
        elem = cutlass.Int64(page) * cutlass.Int64(mCLraw.stride[2]) + cutlass.Int64(
            row % self.page_size
        ) * cutlass.Int64(mCLraw.stride[0])
        row_addr = mCLraw.iterator.toint() + elem * (mCLraw.element_type.width // 8)
        words = cute.make_rmem_tensor(cute.make_layout(4 * n), cutlass.Int32)
        addrs = cute.make_rmem_tensor(cute.make_layout(n), cutlass.Int64)
        for j in cutlass.range_constexpr(n):
            chunk = lane + j * self.threads_per_warp
            # Lanes past the row re-check chunk 0, so every lane has n live chunks.
            if chunk >= ROW_CHUNKS:
                chunk = cutlass.Int32(0)
            addrs[j] = row_addr + cutlass.Int64(chunk) * 16
        load_chunks(words, addrs, n, cutlass.Boolean(True))
        if cutlass.const_expr(self.fgs_stats):
            row_stats = cute.make_tensor(
                common_params.mStats.iterator + _STATS_ROW, cute.make_layout(4)
            )
            poll_chunks(words, addrs, n, False, self.spin_limit, row_stats, True)
        else:
            poll_chunks(words, addrs, n, False, self.spin_limit, words, False)

    @cute.jit
    def _fgs_load_page_table(
        self,
        common_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_pt_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        # ``load_page_table`` plus the current-row poll before the stage of the tile holding the row is published.
        mPT = common_params.mPT[None, common_params.blk_coord[2]]
        common_params.mPTfull = common_params.mPT
        page_per_tile = self.mma_qk_tiler[1] // self.page_size
        tidx = common_params.tidx % self.threads_per_warp
        load_pt_pipeline = common_params.load_pt_pipeline
        while k_tile_count > 0:
            load_pt_pipeline.producer_acquire(load_pt_producer_state)
            elem_per_thread = cute.ceil_div(page_per_tile, self.threads_per_warp)
            atom_async_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
                cutlass.Int32,
                num_bits_per_copy=cutlass.Int32.width,
            )
            mPT_for_copy = cute.flat_divide(mPT, (1,))
            sPT_for_copy = cute.flat_divide(common_params.sPT, (1,))
            for i in range(elem_per_thread):
                idx = i * self.threads_per_warp + tidx
                if cute.elem_less(k_index * page_per_tile + idx, mPT.shape[0]) and cute.elem_less(
                    idx, page_per_tile
                ):
                    cute.copy(
                        atom_async_copy,
                        mPT_for_copy[None, k_index * page_per_tile + idx],
                        sPT_for_copy[None, idx, load_pt_producer_state.index],
                    )
                else:
                    sPT_for_copy[None, idx, load_pt_producer_state.index].fill(0)
            tile_lo = k_index * self.mma_qk_tiler[1]
            row_lo = cutlass.max(common_params.K - self.seq_len_q, tile_lo)
            row_hi = cutlass.min(common_params.K, tile_lo + self.mma_qk_tiler[1])
            for row in cutlass.range(row_lo, row_hi):
                self._fgs_poll_cache_row(common_params, row, tidx)
            if row_lo < row_hi:
                cute.arch.fence_proxy("async.global")
            load_pt_pipeline.producer_commit(load_pt_producer_state)
            load_pt_producer_state.advance()
            k_index += 1
            k_tile_count -= 1
        return load_pt_producer_state

    @cute.jit
    def _fgs_load_tma_qk_one_k_tile(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_q_producer_state: pipeline.PipelineState,
        load_kv_producer_state: pipeline.PipelineState,
        load_pt_consumer_state: pipeline.PipelineState,
        load_q: bool,
    ):
        # ``load_tma_qk_one_k_tile`` plus a proxy fence before the TMA loads of a tile that holds current rows: the
        # page-table warp observed them with generic loads before publishing this stage.
        page_per_tile = (
            self.mma_qk_tiler[1] // self.page_size + qk_params.tiled_mma_qk.thr_id.shape - 1
        ) // (qk_params.tiled_mma_qk.thr_id.shape)
        common_params.load_pt_pipeline.consumer_wait(load_pt_consumer_state)
        page_table_stage = load_pt_consumer_state.index
        load_pt_consumer_state.advance()
        tile_lo = k_index * self.mma_qk_tiler[1]
        if tile_lo + self.mma_qk_tiler[1] > common_params.K - self.seq_len_q:
            cute.arch.fence_proxy("async.global")
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile), cutlass.Int32)
        for i in cutlass.range_constexpr(page_per_tile):
            k_idx[i] = (
                common_params.sPT[0, page_table_stage]
                if self.mma_qk_tiler[1] // self.page_size == 1
                else common_params.sPT[
                    i + common_params.blk_coord[0] * page_per_tile, page_table_stage
                ]
            )
        if cutlass.const_expr(load_q):
            common_params.load_q_pipeline.producer_acquire(load_q_producer_state)
            tma_bar_ptr = common_params.load_q_pipeline.producer_get_barrier(load_q_producer_state)
            self.issue_q_tma_load(
                qk_params.tma_atom_q_latent,
                qk_params.tQLgQL,
                qk_params.tQsQ,
                qk_params.tma_atom_q_rope,
                qk_params.tQRgQR,
                qk_params.tQsQ_rope,
                tma_bar_ptr,
            )
            load_q_producer_state.advance()
        load_kv_pipeline = common_params.load_kv_pipeline
        tma_bar_ptr = load_kv_pipeline.producer_get_barrier(load_kv_producer_state)
        for i in cutlass.range(self.iterations_qk_latent):
            tma_bar_ptr = load_kv_pipeline.producer_get_barrier(load_kv_producer_state)
            load_kv_pipeline.producer_acquire(load_kv_producer_state)
            for k in cutlass.range(page_per_tile):
                cute.copy(
                    qk_params.tma_atom_c_latent,
                    qk_params.tCLgCL[None, i, k_idx[k]],
                    qk_params.tKCsKC[None, k, 0, load_kv_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
            load_kv_producer_state.advance()
        for i in cutlass.range(self.iterations_qk_rope):
            tma_bar_ptr = load_kv_pipeline.producer_get_barrier(load_kv_producer_state)
            load_kv_pipeline.producer_acquire(load_kv_producer_state)
            for k in cutlass.range(page_per_tile):
                cute.copy(
                    qk_params.tma_atom_c_rope,
                    qk_params.tKRgKR[None, i, k_idx[k]],
                    qk_params.tKCsKC[None, k, 0, load_kv_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
            load_kv_producer_state.advance()
        return load_q_producer_state, load_kv_producer_state, load_pt_consumer_state

    @cute.jit
    def _fgs_lse_tile(self, common_params: SimpleNamespace, epilogue_params: SimpleNamespace):
        """(gLSE, cLSE): this CTA's 64-row LSE tile (final or partial) and its coordinates, as in ``epilogue``."""
        cta_rows = self.mma_pv_tiler[0] // self.cluster_shape_mnk[0]
        coord = (common_params.blk_coord[0], common_params.blk_coord[1], common_params.blk_coord[2])
        mLSE = epilogue_params.mLSE
        if cutlass.const_expr(epilogue_params.mAccLSE is not None):
            mLSE = epilogue_params.mAccLSE[None, common_params.blk_coord[3], None, None]
        gLSE = cute.local_tile(mLSE, (cta_rows, 1, 1), coord, (1, 1, 1))
        cLSE = cute.local_tile(
            cute.make_identity_tensor(mLSE.shape), (cta_rows, 1, 1), coord, (1, 1, 1)
        )
        return gLSE, cLSE

    @cute.jit
    def _fgs_epilogue(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
    ) -> pipeline.PipelineState:
        # ``epilogue`` (fixed Q, no softmax stats). bf16 output: cvt.rn.pzo, whole 16-B chunks. Partials:
        #   "all": O and LSE canonicalized (the reduction checks every element);
        #   "lse": plain O stores; every correction thread fences its O stores (fence.acq_rel.gpu), the 128 threads
        #   meet at a named barrier, then thread r < 64 publishes the canonical LSE of tile row r with st.relaxed.gpu.
        #   A reduction that acquires a non-sentinel LSE therefore sees all of this CTA's partial O.
        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        lse_flag = common_params.mAccO is not None and self.fgs_partials == "lse"
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_sum
            self.epilogue_exchange_sync_bar.arrive_and_wait()
            row_sum = (
                row_sum
                + common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ]
            )
        common_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)
        lse = cute.math.log2(row_sum, fastmath=True) + epilogue_params.softmax_scale_log2 * row_max
        gLSE, cLSE = self._fgs_lse_tile(common_params, epilogue_params)
        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                self._tmem_load_partition(common_params, common_params.tiled_mma_pv, iter_n)
            )
            cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
            for i in cutlass.range(cute.size(tTR_rAcc), vectorize=True, unroll_full=True):
                tTR_rAcc[i] = (
                    tTR_rAcc[i] * epilogue_params.output_scale * cute.arch.rcp_approx(row_sum)
                )
            tR2G_rO_src = None
            if cutlass.const_expr(common_params.mAccO is None):
                tR2G_rO_src = cute.make_fragment_like(tTR_gO, self.o_dtype)
                tR2G_rO_src.store(tTR_rAcc.load().to(self.o_dtype, pzo=True))
            else:
                if cutlass.const_expr(not lse_flag):
                    for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                        tTR_rAcc[i] = canon_f32(tTR_rAcc[i])
                tR2G_rO_src = tTR_rAcc
            if cute.elem_less(tTR_cO[0][0], common_params.H):
                cute.autovec_copy(
                    tR2G_rO_src,
                    tTR_gO,
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                )
            if cutlass.const_expr(not lse_flag and self.warps_in_n == 2):
                if cute.elem_less(cLSE[tidx][0], common_params.H):
                    gLSE[tidx] = canon_f32(lse)
        if cutlass.const_expr(lse_flag):
            cute.arch.fence_acq_rel_gpu()
            self.epilogue_exchange_sync_bar.arrive_and_wait()
            if tidx < cute.size(gLSE):
                if cute.elem_less(cLSE[tidx][0], common_params.H):
                    _st_relaxed_f32(
                        (gLSE.iterator + cute.crd2idx(tidx, gLSE.layout)).toint(), canon_f32(lse)
                    )
        cute.arch.fence_view_async_tmem_load()
        common_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
        mma_o_consumer_state.advance()
        return mma_o_consumer_state

    @cute.kernel
    def _fgs_reduction_kernel(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        kv_bounds: Optional[cute.Tensor],
        cum_seq_lens_q: Optional[cute.Tensor],
        block_split_kvs: cute.Tensor,
        mStats: Optional[cute.Tensor],
    ):
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(not self.fgs_reduction):
            cute.arch.griddepcontrol_wait()
        check_lse = self.fgs_reduction
        check_o = self.fgs_reduction and self.fgs_partials == "all"
        lse_flag = self.fgs_reduction and self.fgs_partials == "lse"
        with_stats = self.fgs_stats and self.fgs_reduction

        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % self.threads_per_warp
        d_tile_idx = bidx % self.reducer_d_tiles
        head_idx = bidx // self.reducer_d_tiles

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.reducer_max_splits * self.acc_dtype.width // 8, 16)
        smem_lse_scale = cute.make_tensor(
            cute.recast_ptr(storage, dtype=self.acc_dtype),
            cute.make_layout(self.reducer_max_splits),
        )

        flat_q_row = bidy * self.num_heads + head_idx
        q_tile = flat_q_row >> 7
        q_tile_row = flat_q_row & 127
        local_split_kv = split_kv
        if cutlass.const_expr(self.is_var_split_kv):
            local_split_kv = cutlass.max(cutlass.min(block_split_kvs[bidz], split_kv), 1)
        k_tile_total = cute.ceil_div(cache_seqs[bidz], self.mma_qk_tiler[1])
        k_tile_per_cta = cutlass.max(cute.ceil_div(k_tile_total, local_split_kv), 1)
        local_split_kv = cute.ceil_div(k_tile_total, k_tile_per_cta)

        reduce_row = cutlass.Boolean(True)
        if cutlass.const_expr(self.is_var_split_kv):
            reduce_row = local_split_kv != 1
        if reduce_row:
            gLSE = mAccLSE[q_tile_row, None, q_tile, bidz]
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 0:
                lse_per_thread = cute.ceil_div(self.reducer_max_splits, self.threads_per_warp)
                local_lse = cute.make_rmem_tensor(cute.make_layout(lse_per_thread), self.lse_dtype)
                bits = cute.make_rmem_tensor(cute.make_layout(lse_per_thread), cutlass.Int32)
                addrs = cute.make_rmem_tensor(cute.make_layout(lse_per_thread), cutlass.Int64)
                for i in cutlass.range_constexpr(lse_per_thread):
                    split_kv_idx = tidx + i * self.threads_per_warp
                    addrs[i] = (gLSE.iterator + cute.crd2idx(split_kv_idx, gLSE.layout)).toint()
                    bits[i] = cutlass.Int32(0)
                    if split_kv_idx < local_split_kv:
                        if cutlass.const_expr(lse_flag):
                            bits[i] = _ld_relaxed_b32(addrs[i])
                        else:
                            bits[i] = _ld_cg_b32(addrs[i])
                if cutlass.const_expr(check_lse):
                    if cutlass.const_expr(with_stats):
                        lse_stats = cute.make_tensor(
                            mStats.iterator + _STATS_RED_LSE, cute.make_layout(4)
                        )
                        _poll_words(bits, addrs, lse_per_thread, self.spin_limit, lse_stats, True)
                    else:
                        _poll_words(bits, addrs, lse_per_thread, self.spin_limit, bits, False)
                if cutlass.const_expr(lse_flag):
                    # Acquire pattern (relaxed loads that saw every published LSE, then a fence): the producer's
                    # partial O stores are visible to this CTA's O loads after the barrier below.
                    cute.arch.fence_acq_rel_gpu()
                lse_max = -self.lse_dtype.inf
                for i in cutlass.range_constexpr(lse_per_thread):
                    split_kv_idx = tidx + i * self.threads_per_warp
                    local_lse[i] = -self.lse_dtype.inf
                    if split_kv_idx < local_split_kv:
                        local_lse[i] = _as_f32(bits[i])
                    lse_max = cute.arch.fmax(lse_max, local_lse[i])
                lse_max = cute.arch.warp_reduction_max(lse_max)
                lse_max = lse_max if lse_max != -self.lse_dtype.inf else 0.0
                sum_lse = 0.0
                for i in cutlass.range_constexpr(lse_per_thread):
                    sum_lse += cute.math.exp2(local_lse[i] - lse_max, fastmath=True)
                sum_lse = cute.arch.warp_reduction_sum(sum_lse)
                global_lse = (
                    lse_max + cute.math.log2(sum_lse, fastmath=True)
                    if not sum_lse == self.lse_dtype(0.0) or sum_lse != sum_lse
                    else self.lse_dtype.inf
                )
                if d_tile_idx == 0 and tidx == 0:
                    mLSE[head_idx, bidy, bidz] = canon_f32(global_lse)
                for i in cutlass.range_constexpr(lse_per_thread):
                    split_kv_idx = tidx + i * self.threads_per_warp
                    if cute.elem_less(split_kv_idx, local_split_kv):
                        smem_lse_scale[split_kv_idx] = cute.math.exp2(
                            local_lse[i] - global_lse, fastmath=True
                        )

            pipeline.sync(barrier_id=4)

            # Thread t reduces D elements [4t, 4t + 4) of this band, splits in order (same per-element order as the
            # base kernel); lane pairs then form one 16-B chunk of 8 bf16 that the even lane stores. Splits are
            # loaded and checked _RED_UNROLL at a time; slots past the last split re-load split g and are not added.
            gAccO = mAccO[q_tile_row, None, None, q_tile, bidz]
            e0 = d_tile_idx * self.reducer_d_tile + tidx * 4
            active = tidx * 4 < self.reducer_d_tile
            rAccO = cute.make_rmem_tensor(cute.make_layout(4), self.acc_dtype)
            rAccO.fill(0.0)
            words = cute.make_rmem_tensor(cute.make_layout(4 * _RED_UNROLL), cutlass.Int32)
            chunk_addr = cute.make_rmem_tensor(cute.make_layout(_RED_UNROLL), cutlass.Int64)
            for g in cutlass.range(0, local_split_kv, _RED_UNROLL, unroll=1):
                for u in cutlass.range_constexpr(_RED_UNROLL):
                    split_idx = g + u
                    if split_idx >= local_split_kv:
                        split_idx = g
                    chunk_addr[u] = (
                        gAccO.iterator + cute.crd2idx((split_idx, e0), gAccO.layout)
                    ).toint()
                load_chunks(words, chunk_addr, _RED_UNROLL, active)
                if cutlass.const_expr(check_o):
                    if cutlass.const_expr(with_stats):
                        o_stats = cute.make_tensor(
                            mStats.iterator + _STATS_RED_O, cute.make_layout(4)
                        )
                        poll_chunks(
                            words, chunk_addr, _RED_UNROLL, True, self.spin_limit, o_stats, True
                        )
                    else:
                        poll_chunks(
                            words, chunk_addr, _RED_UNROLL, True, self.spin_limit, words, False
                        )
                for u in cutlass.range_constexpr(_RED_UNROLL):
                    if g + u < local_split_kv:
                        for j in cutlass.range_constexpr(4):
                            rAccO[j] += _as_f32(words[4 * u + j]) * smem_lse_scale[g + u]
            w0 = bf16x2_pzo(rAccO[0], rAccO[1], True)
            w1 = bf16x2_pzo(rAccO[2], rAccO[3], True)
            n0 = cute.arch.shuffle_sync_down(w0, 1)
            n1 = cute.arch.shuffle_sync_down(w1, 1)
            if active and lane % 2 == 0:
                out_addr = (
                    mO.iterator + cute.crd2idx((head_idx, e0, bidy, bidz), mO.layout)
                ).toint()
                st_global_v4(out_addr, w0, w1, n0, n1)
        return


def _cluster_scope():
    from cutlass._mlir.dialects.nvvm import MBarrierScopeKind

    return MBarrierScopeKind.CLUSTER


def _fence_release_cluster_smem():
    from cutlass.experimental.primitives.nvvm_wrapper import MemOrder, fence_sync_restrict

    fence_sync_restrict(MemOrder.RELEASE)


@cute.kernel
def _fill_partials_kernel(
    ws_base: cutlass.Int64,
    cache_seqs: cute.Tensor,
    block_split_kvs: Optional[cute.Tensor],
    split_kv: cutlass.Int32,
    kv_tile: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    seq_q_tiles: cutlass.Int32,
    batch: cutlass.Int32,
):
    # Workspace layout of ``initialize_workspace`` (acc_o then acc_lse, fp32): acc_o[row, split, d, s, b] at
    # ((((b * S + s) * rows + row) * split_kv + split) * D + d), acc_lse[row, split, s, b] after all of acc_o.
    tidx, _, _ = cute.arch.thread_idx()
    h, bs, _ = cute.arch.block_idx()
    b = bs // seq_q_tiles
    local = split_kv
    if cutlass.const_expr(block_split_kvs is not None):
        local = cutlass.max(cutlass.min(block_split_kvs[b], split_kv), 1)
    k_tiles = (cache_seqs[b] + kv_tile - 1) // kv_tile
    per_cta = cutlass.max((k_tiles + local - 1) // local, 1)
    local = (k_tiles + per_cta - 1) // per_cta
    if cutlass.const_expr(block_split_kvs is not None):
        if local == 1:
            local = cutlass.Int32(0)
    lse_off = (
        cutlass.Int64(batch)
        * cutlass.Int64(seq_q_tiles * rows)
        * cutlass.Int64(split_kv)
        * LATENT_DIM
    )
    for split in cutlass.range(local):
        row = cutlass.Int64(bs * rows + h) * cutlass.Int64(split_kv) + cutlass.Int64(split)
        st_global_v4(
            ws_base + (row * LATENT_DIM + cutlass.Int64(tidx) * 4) * 4,
            F32_SENTINEL,
            F32_SENTINEL,
            F32_SENTINEL,
            F32_SENTINEL,
        )
        if tidx == 0:
            _st_global_b32(ws_base + (lse_off + row) * 4, F32_SENTINEL)


@cute.jit
def _fill_partials_host(
    ws_base: cutlass.Int64,
    cache_seqs: cute.Tensor,
    block_split_kvs: Optional[cute.Tensor],
    split_kv: cutlass.Int32,
    heads: cutlass.Int32,
    seq_q_tiles: cutlass.Int32,
    batch: cutlass.Int32,
    stream: cuda.CUstream,
    kv_tile: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    _fill_partials_kernel(
        ws_base, cache_seqs, block_split_kvs, split_kv, kv_tile, rows, seq_q_tiles, batch
    ).launch(grid=(heads, batch * seq_q_tiles, 1), block=(LATENT_DIM // 4, 1, 1), stream=stream)


@cute.kernel
def _fill_lse_kernel(
    ws_base: cutlass.Int64,
    cache_seqs: cute.Tensor,
    block_split_kvs: Optional[cute.Tensor],
    split_kv: cutlass.Int32,
    heads: cutlass.Int32,
    kv_tile: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    seq_q_tiles: cutlass.Int32,
    batch: cutlass.Int32,
):
    # acc_lse[row, split, s, b] (fp32) follows acc_o in the workspace; see ``_fill_partials_kernel``.
    tidx, _, _ = cute.arch.thread_idx()
    bs, _, _ = cute.arch.block_idx()
    b = bs // seq_q_tiles
    local = split_kv
    if cutlass.const_expr(block_split_kvs is not None):
        local = cutlass.max(cutlass.min(block_split_kvs[b], split_kv), 1)
    k_tiles = (cache_seqs[b] + kv_tile - 1) // kv_tile
    per_cta = cutlass.max((k_tiles + local - 1) // local, 1)
    local = (k_tiles + per_cta - 1) // per_cta
    read = local > 0
    if cutlass.const_expr(block_split_kvs is not None):
        read = read & (local != 1)
    if read:
        lse_base = (
            ws_base
            + (
                cutlass.Int64(batch)
                * cutlass.Int64(seq_q_tiles * rows)
                * cutlass.Int64(split_kv)
                * LATENT_DIM
            )
            * 4
        )
        tile_base = cutlass.Int64(bs) * cutlass.Int64(rows) * cutlass.Int64(split_kv)
        for idx in cutlass.range(tidx, heads * local, _FILL_LSE_THREADS):
            row = idx // local
            split = idx - row * local
            _st_global_b32(
                lse_base
                + (tile_base + cutlass.Int64(row) * cutlass.Int64(split_kv) + cutlass.Int64(split))
                * 4,
                F32_SENTINEL,
            )


@cute.jit
def _fill_lse_host(
    ws_base: cutlass.Int64,
    cache_seqs: cute.Tensor,
    block_split_kvs: Optional[cute.Tensor],
    split_kv: cutlass.Int32,
    heads: cutlass.Int32,
    seq_q_tiles: cutlass.Int32,
    batch: cutlass.Int32,
    stream: cuda.CUstream,
    kv_tile: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    _fill_lse_kernel(
        ws_base, cache_seqs, block_split_kvs, split_kv, heads, kv_tile, rows, seq_q_tiles, batch
    ).launch(grid=(batch * seq_q_tiles, 1, 1), block=(_FILL_LSE_THREADS, 1, 1), stream=stream)


@cute.kernel
def _fill_rows_kernel(
    pool_base: cutlass.Int64,
    row_stride: cutlass.Int64,
    page_stride: cutlass.Int64,
    page_table: cute.Tensor,
    cache_seqs: cute.Tensor,
    page_size: cutlass.Constexpr,
    seq_len_q: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    b, u, _ = cute.arch.block_idx()
    row = cache_seqs[b] - seq_len_q + u
    if row >= 0 and tidx < ROW_CHUNKS:
        page = page_table[row // page_size, b]
        elem = cutlass.Int64(page) * page_stride + cutlass.Int64(row % page_size) * row_stride
        sentinel = cutlass.Int32((BF16_SENTINEL | (BF16_SENTINEL << 16)) - (1 << 32))
        st_global_v4(
            pool_base + elem * 2 + cutlass.Int64(tidx) * 16, sentinel, sentinel, sentinel, sentinel
        )


@cute.jit
def _fill_rows_host(
    pool_base: cutlass.Int64,
    row_stride: cutlass.Int64,
    page_stride: cutlass.Int64,
    page_table: cute.Tensor,
    cache_seqs: cute.Tensor,
    batch: cutlass.Int32,
    stream: cuda.CUstream,
    page_size: cutlass.Constexpr,
    seq_len_q: cutlass.Constexpr,
):
    _fill_rows_kernel(
        pool_base, row_stride, page_stride, page_table, cache_seqs, page_size, seq_len_q
    ).launch(grid=(batch, seq_len_q, 1), block=(96, 1, 1), stream=stream)


_FILL_CACHE = {}


def _current_stream():
    import torch

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _from_torch(t, leading_dim=0, align=4):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t, assumed_align=align).mark_layout_dynamic(leading_dim=leading_dim)


def fill_partials_sentinel(
    split_workspace,
    num_heads: int,
    seq_q_tiles: int,
    cache_seqs,
    split_kv: int,
    block_split_kvs=None,
    kv_tile: int = 128,
    rows: int = 128,
):
    """Sentinel-fill the split-KV partials (O and LSE) the reduction reads (``fgs_partials="all"``; one launch,
    one CTA per row and request, graph-capturable).

    ``split_workspace`` is the split region the decode kernel receives as ``workspace`` (its acc_o / acc_lse);
    ``cache_seqs`` / ``block_split_kvs`` are the step's KV lengths and split plan, as passed to the decode.
    """
    batch = int(cache_seqs.shape[0])
    key = ("partials", block_split_kvs is None, kv_tile, rows)
    args = [
        cutlass.Int64(split_workspace.data_ptr()),
        _from_torch(cache_seqs),
        None if block_split_kvs is None else _from_torch(block_split_kvs),
        cutlass.Int32(split_kv),
        cutlass.Int32(num_heads),
        cutlass.Int32(seq_q_tiles),
        cutlass.Int32(batch),
        _current_stream(),
    ]
    fn = _FILL_CACHE.get(key)
    if fn is None:
        fn = cute.compile(_fill_partials_host, *args, kv_tile, rows)
        _FILL_CACHE[key] = fn
    fn(*args)


def fill_lse_sentinel(
    split_workspace,
    num_heads: int,
    seq_q_tiles: int,
    cache_seqs,
    split_kv: int,
    block_split_kvs=None,
    kv_tile: int = 128,
    rows: int = 128,
):
    """Sentinel-fill only the partial LSE entries the reduction will read (``fgs_partials="lse"``; one launch,
    one CTA per request and q tile, graph-capturable). Same arguments as ``fill_partials_sentinel``."""
    batch = int(cache_seqs.shape[0])
    key = ("lse", block_split_kvs is None, kv_tile, rows)
    args = [
        cutlass.Int64(split_workspace.data_ptr()),
        _from_torch(cache_seqs),
        None if block_split_kvs is None else _from_torch(block_split_kvs),
        cutlass.Int32(split_kv),
        cutlass.Int32(num_heads),
        cutlass.Int32(seq_q_tiles),
        cutlass.Int32(batch),
        _current_stream(),
    ]
    fn = _FILL_CACHE.get(key)
    if fn is None:
        fn = cute.compile(_fill_lse_host, *args, kv_tile, rows)
        _FILL_CACHE[key] = fn
    fn(*args)


def fill_current_rows_sentinel(c_latent, page_table, cache_seqs, seq_len_q: int):
    """Sentinel-fill this step's cache rows (positions ``K - seq_len_q .. K - 1`` of each request).

    ``c_latent`` / ``page_table`` / ``cache_seqs`` are exactly the decode kernel's inputs for one layer
    (``c_latent`` = [page_size, 512, num_pages] view of the pool, rows of 576 bf16).
    """
    batch = int(cache_seqs.shape[0])
    page_size = int(c_latent.shape[0])
    key = ("rows", page_size, seq_len_q)
    args = [
        cutlass.Int64(c_latent.data_ptr()),
        cutlass.Int64(c_latent.stride(0)),
        cutlass.Int64(c_latent.stride(2)),
        _from_torch(page_table, leading_dim=0),
        _from_torch(cache_seqs),
        cutlass.Int32(batch),
        _current_stream(),
    ]
    fn = _FILL_CACHE.get(key)
    if fn is None:
        fn = cute.compile(_fill_rows_host, *args, page_size, seq_len_q)
        _FILL_CACHE[key] = fn
    fn(*args)
