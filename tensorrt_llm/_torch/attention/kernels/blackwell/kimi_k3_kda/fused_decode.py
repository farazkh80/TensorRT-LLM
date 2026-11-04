# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from FlashInfer flashinfer/kda_kernels/fused_kda_decode.py (v0.6.18),
# Copyright (c) 2026 by FlashInfer team, licensed under the Apache License 2.0.
"""Fused Kimi K3 KDA T=1 decode on the TensorRT-LLM cache pools (CuTe DSL).

One launch per decode step does what ``trtllm::kda_decode`` does with
``update_conv_cache``, lower-bound gate, beta sigmoid and the output gate:

  conv4 + conv-cache shift, SiLU (fp32), q/k L2 norm (eps 1e-6, q scaled by
  128^-0.5), decay = exp(lb * sigmoid(exp(A_log) * (g + dt_bias))),
  beta = sigmoid(b), delta rule on the fp32 V-first state
  (h = S * decay; u = (v - h.k) * beta; S = h + k u; o = S.q), and
  y = o * rsqrt(mean(o^2) + eps) * w * sigmoid(z) stored as BF16.

Pools are the live TRT-LLM ones, addressed by ``state_indices`` (any slot >= 0
is live, including 0; a negative slot skips both cache updates and writes zeros):
  conv_state [slots, 3 * H * 128, 3] BF16, width innermost, oldest tap first;
  state      [slots, H, 128 (V), 128 (K)] FP32.

Grid (H * v_split, B) with ``v_split`` CTAs per (request, head) forming a
cluster. Each CTA owns 128 / v_split state rows split evenly over its warps;
lane l owns keys 4l..4l+3 of every row of its warp. With v_split > 1
the RMSNorm sum of squares is exchanged through distributed shared memory, and
only cluster rank 0 writes the conv cache.

``fgs_producer`` builds the fine-grained-sync producer of the core output for a
consumer that validates its input per 16 B (TMA report_valid, bf16 -0.0
sentinel): work items are issued in head groups (all rows of heads
[g, g + head_group) before the next group; a 1-D grid, or a one-wave persistent
grid looping over the items), PDL dependents are released at kernel entry, and
each 16-B chunk of the output is written by one 16-B store with -0.0 mapped to
+0.0. Values are otherwise bit-identical to the default build.

``fgs_consumer`` reads the projection inputs (x and the output gate from qkvg, g, beta)
from fine-grained-sync slots instead of waiting for the PDL primary grid: the state
rows are prefetched first, then every input is read as 16-B chunks with coherent
``ld.relaxed.gpu`` loads, and a chunk counts as written once its element 0 is not the
BF16 -0.0 sentinel (its producer writes each chunk with one 16-B store and never
stores -0.0). The wait is warp-uniform: while any lane of a setup warp holds an
unwritten chunk, the whole warp sleeps (exponential backoff) and reloads all of its
chunks, and leaves the loop together. After ``max_spins`` rounds the warp falls back to
``griddepcontrol.wait`` and takes the values it then reads. Both modes combine: the
consumer of qkvg / g / beta is then also the producer of the core output, also as a
one-wave persistent grid, where every item validates its own inputs. With ``late_gate``
the consumer validates the output gate only in the epilogue (warp 0, right before the
gated RMSNorm), so the state update does not wait for it.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

F32 = cutlass.Float32
BF16 = cutlass.BFloat16

D = 128
CONV_ITEMS = 3 * D // 4
SETUP_ITEMS = CONV_ITEMS + D
L2_EPS = 1.0e-6
BF16_NEG_ZERO = 0x8000
SPIN_NS_MIN = 32
SPIN_NS_MAX = 2048


def cta_smem_bytes(warps: int, ring: int) -> int:
    """Dynamic shared memory of one CTA: 4 KB of vectors plus ``ring`` staged rows per warp."""
    return 4096 + warps * ring * D * 4


def _aligned(tensor, alignment):
    ptr = tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(ptr.dtype, ptr.toint(), ptr.memspace, assumed_align=alignment),
        tensor.layout,
    )


@dsl_user_op
def _load_peer_f32(smem_addr, peer, *, loc=None, ip=None):
    """Load the f32 at this CTA's shared address ``smem_addr`` from cluster rank ``peer``."""
    return F32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip),
                cutlass.Int32(peer).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .u32 ra; mapa.shared::cluster.u32 ra, $1, $2; ld.shared::cluster.f32 $0, [ra]; }",
            "=f,r,r",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


_PZO_SW = (
    "{ .reg .b16 l, h; .reg .pred p; cvt.rn.bf16x2.f32 $0, $1, $2; mov.b32 {l, h}, $0; "
    "setp.eq.b16 p, l, 0x8000; selp.b16 l, 0, l, p; "
    "setp.eq.b16 p, h, 0x8000; selp.b16 h, 0, h, p; mov.b32 $0, {l, h}; }"
)
# The same conversion in one instruction (sm_107).
_PZO_HW = "cvt.rn.pzo.bf16x2.f32 $0, $1, $2;"


@dsl_user_op
def _bf16x2_pzo(lo, hi, hw, *, loc=None, ip=None):
    """Round two f32 to a packed bf16x2 (``lo`` in the low half) like ``cvt.rn``, with -0.0 -> +0.0."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [F32(hi).ir_value(loc=loc, ip=ip), F32(lo).ir_value(loc=loc, ip=ip)],
            _PZO_HW if hw else _PZO_SW,
            "=r,f,f",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_relaxed_v4(addr, reload, *, loc=None, ip=None):
    """16-B ``ld.relaxed.gpu.global`` (never served from a stale L1 line) as four int32; a
    ``reload`` is never merged with or hoisted above an earlier load of the address."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.relaxed.gpu.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}" if reload else "=r,=r,=r,=r,l",
        has_side_effects=reload,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _bf16_at(w0, w1, w2, w3, e, *, loc=None, ip=None):
    """FP32 value of BF16 element ``e`` (0..7) of the 16-B chunk held in words w0..w3."""
    return F32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Int32(v).ir_value(loc=loc, ip=ip) for v in (w0, w1, w2, w3, e)],
            "{ .reg .pred p; .reg .b32 a, b, s; "
            "and.b32 s, $5, 2; setp.ne.b32 p, s, 0; selp.b32 a, $2, $1, p; selp.b32 b, $4, $3, p; "
            "and.b32 s, $5, 4; setp.ne.b32 p, s, 0; selp.b32 a, b, a, p; "
            "and.b32 s, $5, 1; setp.ne.b32 p, s, 0; and.b32 b, a, 0xFFFF0000; shl.b32 a, a, 16; "
            "selp.b32 $0, b, a, p; }",
            "=f,r,r,r,r,r",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nanosleep(ns, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_global_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)]
        + [cutlass.Int32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _sigmoid(x):
    return 1.0 / (1.0 + cute.math.exp(-x, fastmath=True))


class KdaDecodeT1:
    def __init__(
        self,
        heads: int,
        v_split: int,
        lower_bound: float,
        q_scale: float,
        norm_eps: float,
        early_state: bool,
        use_pdl: bool,
        reg_rows: int,
        min_blocks: int,
        warps: int,
        ring: int,
        early_trigger: bool = False,
        store16: bool = False,
        hw_pzo: bool = False,
        head_group: int = 0,
        persistent_ctas: int = 0,
        fgs_consumer: bool = False,
        max_spins: int = 0,
        with_stats: bool = False,
        head_split: int = 1,
        prefetch_next: bool = False,
        smem_carveout: int | None = None,
        late_gate: bool = False,
        with_head_order: bool = False,
    ):
        assert v_split in (1, 2, 4)
        self.heads = heads
        self.v_split = v_split
        self.warps = warps
        self.threads = self.warps * 32
        self.rows_per_cta = D // v_split
        self.rows_per_warp = self.rows_per_cta // warps
        assert self.rows_per_warp * warps == self.rows_per_cta and self.threads >= 4 * 32
        # The first reg_rows rows of each warp load straight into registers. The rest stream
        # through a per-thread ring of `ring` shared-memory row slots filled by cp.async, one
        # commit group per row; a slot is refilled with the row `ring` ahead once its row is
        # consumed, so up to `ring` rows per thread stay in flight.
        assert 0 <= reg_rows <= self.rows_per_warp
        self.reg_rows = reg_rows
        self.smem_rows = self.rows_per_warp - reg_rows
        self.ring = min(ring, self.smem_rows)
        assert self.smem_rows == 0 or self.ring >= 1
        self.lower_bound = lower_bound
        self.q_scale = q_scale
        self.norm_eps = norm_eps
        self.early_state = early_state
        self.use_pdl = use_pdl
        # Resident CTAs per SM requested from the register allocator (0: no bound).
        self.min_blocks = min_blocks
        self.smem_bytes = cta_smem_bytes(warps, self.ring)
        # The fgs_producer parts, separable for A/B: launch_dependents at entry; one 16-B store
        # per chunk with -0.0 -> +0.0; head_group > 0: a 1-D grid over (row, head) items, all
        # rows of head_group heads before the next group (1: rows fastest, heads: heads fastest);
        # persistent_ctas > 0: at most that many CTAs (one wave) loop over those items.
        self.early_trigger = early_trigger
        self.store16 = store16
        self.hw_pzo = hw_pzo
        assert head_group == 0 or heads % head_group == 0
        self.head_group = head_group
        # head_split > 1 interleaves the head order over that many equal head ranges (the K
        # slices of a split-K consumer): heads 0, H/s, 2H/s, ..., 1, H/s + 1, ...
        assert head_split == 1 or (head_group > 0 and heads % head_split == 0)
        self.head_split = head_split
        # with_head_order: group positions map to heads through a device permutation instead.
        assert not with_head_order or (head_group > 0 and head_split == 1)
        self.with_head_order = with_head_order
        assert persistent_ctas % v_split == 0 and (persistent_ctas == 0 or head_group > 0)
        self.persistent = persistent_ctas > 0
        self.persistent_ctas = persistent_ctas
        assert not (self.persistent and early_state)
        # prefetch_next: a warp issues the ring rows of its next persistent item right after its
        # last state row of the current one.
        assert not prefetch_next or (self.persistent and self.smem_rows > 0)
        self.prefetch_next = prefetch_next
        self.smem_carveout = smem_carveout
        # Conv items fill warps 0-2 and gate items the next four, so every setup warp has one
        # role per iteration and the validity wait stays warp-uniform.
        assert CONV_ITEMS % 32 == 0 and SETUP_ITEMS % 32 == 0 and self.threads % 32 == 0
        assert not fgs_consumer or (use_pdl and not early_state)
        assert fgs_consumer or not with_stats
        assert fgs_consumer or not late_gate
        self.late_gate = late_gate
        self.fgs_consumer = fgs_consumer
        self.max_spins = max_spins
        self.with_stats = with_stats

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        conv_state: cute.Tensor,
        raw_gate: cute.Tensor,
        raw_beta: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state_indices: cute.Tensor,
        state: cute.Tensor,
        output_gate: cute.Tensor,
        norm_weight: cute.Tensor,
        output: cute.Tensor,
        stats: cute.Tensor | None,
        head_order: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        H = self.heads
        d = H * D
        rows = state_indices.shape[0]
        # [elem, lane, section(q/k/v), head, row]
        x_v = cute.make_tensor(
            x.iterator, cute.make_layout((4, 32, 3, H, rows), stride=(1, 4, d, D, x.stride[0]))
        )
        # [elem, lane, tap, section, head]
        w_v = cute.make_tensor(
            weight.iterator,
            cute.make_layout((4, 32, 4, 3, H), stride=(1, 4, d, 4 * d, D)),
        )
        # Twelve contiguous BF16 per lane: four channels x three taps, as three 4-vectors.
        cs_v = cute.make_tensor(
            conv_state.iterator,
            cute.make_layout(
                (4, 3, 32, 3, H, conv_state.shape[0]),
                stride=(1, 4, 12, 3 * d, 3 * D, conv_state.stride[0]),
            ),
        )
        g_v = cute.make_tensor(
            raw_gate.iterator, cute.make_layout((D, H, rows), stride=(1, D, raw_gate.stride[0]))
        )
        dt_v = cute.make_tensor(dt_bias.iterator, cute.make_layout((D, H), stride=(1, D)))
        b_v = cute.make_tensor(
            raw_beta.iterator, cute.make_layout((H, rows), stride=(1, raw_beta.stride[0]))
        )
        # [elem, lane, v row, head, slot]
        st_v = cute.make_tensor(
            state.iterator,
            cute.make_layout(
                (4, 32, D, H, state.shape[0]), stride=(1, 4, D, D * D, state.stride[0])
            ),
        )
        og_v = cute.make_tensor(
            output_gate.iterator,
            cute.make_layout((D, H, rows), stride=(1, D, output_gate.stride[0])),
        )
        # [elem, lane (default) or 16-B chunk (store16), head, row]
        vec = 8 if self.store16 else 4
        out_v = cute.make_tensor(
            output.iterator,
            cute.make_layout((vec, D // vec, H, rows), stride=(1, vec, D, output.stride[0])),
        )
        cluster = [self.v_split, 1, 1] if self.v_split > 1 else None
        grid = [H * self.v_split, rows, 1]
        if cutlass.const_expr(self.head_group > 0):
            grid = [rows * (H * self.v_split), 1, 1]
        if cutlass.const_expr(self.persistent):
            grid = [cute.min(rows * (H * self.v_split), self.persistent_ctas), 1, 1]
        carveout = (
            {} if self.smem_carveout is None else dict(preferred_smem_carveout=self.smem_carveout)
        )
        self.kernel(
            x_v,
            w_v,
            cs_v,
            g_v,
            b_v,
            A_log,
            dt_v,
            state_indices,
            st_v,
            og_v,
            norm_weight,
            out_v,
            stats,
            head_order,
        ).launch(
            grid=grid,
            block=[self.threads, 1, 1],
            cluster=cluster,
            smem=self.smem_bytes,
            stream=stream,
            min_blocks_per_mp=self.min_blocks,
            use_pdl=self.use_pdl,
            **carveout,
        )

    def _issue_ring_row(self, state, state_s, lane, warp, row_base, head, slot, j):
        """cp.async staged row j of this thread's warp into ring slot j % ring and commit it."""
        copy_g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
            F32,
            num_bits_per_copy=128,
        )
        r = self.reg_rows + j
        src = _aligned(
            state[(None, lane, row_base + warp * self.rows_per_warp + r, head, slot)], 16
        )
        cute.copy(
            copy_g2s, src, _aligned(state_s[(None, lane, warp * self.ring + j % self.ring)], 16)
        )
        cute.arch.cp_async_commit_group()

    def _load_state(self, state, state_s, state_regs, lane, warp, row_base, head, slot):
        copy_g2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), F32, num_bits_per_copy=128)
        for r in range(self.reg_rows):
            src = _aligned(
                state[(None, lane, row_base + warp * self.rows_per_warp + r, head, slot)], 16
            )
            cute.copy(copy_g2r, src, state_regs[(None, r)])
        if not self.prefetch_next:
            self._issue_ring(state, state_s, lane, warp, row_base, head, slot)

    def _issue_ring(self, state, state_s, lane, warp, row_base, head, slot):
        for j in range(self.ring):
            self._issue_ring_row(state, state_s, lane, warp, row_base, head, slot, j)

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        conv_state: cute.Tensor,
        raw_gate: cute.Tensor,
        raw_beta: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state_indices: cute.Tensor,
        state: cute.Tensor,
        output_gate: cute.Tensor,
        norm_weight: cute.Tensor,
        output: cute.Tensor,
        stats: cute.Tensor | None,
        head_order: cute.Tensor | None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        warp = cute.arch.make_warp_uniform(warp)
        if cutlass.const_expr(self.early_trigger):
            # The consumer validates the output per 16 B instead of waiting for this grid.
            cute.arch.griddepcontrol_launch_dependents()

        smem = SmemAllocator()
        mixed = smem.allocate_tensor(F32, cute.make_layout((3 * D,)), byte_alignment=16)
        rec_out = smem.allocate_tensor(F32, cute.make_layout((D,)), byte_alignment=16)
        decay_s = smem.allocate_tensor(F32, cute.make_layout((D,)), byte_alignment=16)
        gate_s = smem.allocate_tensor(F32, cute.make_layout((D,)), byte_alignment=16)
        normw_s = smem.allocate_tensor(F32, cute.make_layout((D,)), byte_alignment=16)
        scalars = smem.allocate_tensor(F32, cute.make_layout((4,)), byte_alignment=16)
        state_s = None
        if cutlass.const_expr(self.smem_rows > 0):
            state_s = smem.allocate_tensor(
                F32,
                cute.make_layout((4, 32, self.warps * self.ring), stride=(1, 4, D)),
                byte_alignment=16,
            )
        args = (
            x,
            weight,
            conv_state,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_indices,
            state,
            output_gate,
            norm_weight,
            output,
            mixed,
            rec_out,
            decay_s,
            gate_s,
            normw_s,
            scalars,
            state_s,
            tidx,
            lane,
            warp,
            stats,
        )

        if cutlass.const_expr(self.persistent):
            # The FGS consumer validates each item's inputs in the item itself.
            if cutlass.const_expr(self.use_pdl and not self.fgs_consumer):
                cute.arch.griddepcontrol_wait()
            grid_x, _, _ = cute.arch.grid_dim()
            clusters = grid_x // self.v_split
            cluster = bx // self.v_split
            rank = bx - cluster * self.v_split
            rows = state_indices.shape[0]
            units = rows * self.heads
            if cutlass.const_expr(self.prefetch_next):
                first_row, first_head = self._unit_coords(cluster, rows, head_order)
                self._issue_ring(
                    state,
                    state_s,
                    lane,
                    warp,
                    rank * self.rows_per_cta,
                    first_head,
                    cute.max(state_indices[first_row], cutlass.Int32(0)),
                )
            for unit in cutlass.range(cluster, units, clusters):
                row, head = self._unit_coords(unit, rows, head_order)
                next_unit = unit + clusters
                next_row, next_head = row, head
                if cutlass.const_expr(self.prefetch_next):
                    next_row, next_head = self._unit_coords(
                        cute.min(next_unit, units - 1), rows, head_order
                    )
                self._item(
                    *args,
                    row,
                    head,
                    rank,
                    self.fgs_consumer,
                    next_unit < units,
                    next_row,
                    next_head,
                )
                if cutlass.const_expr(self.v_split == 1):
                    # The next item rewrites the shared vectors warp 0 is still reading.
                    cute.arch.barrier()
        else:
            if cutlass.const_expr(self.head_group > 0):
                unit = bx // self.v_split
                rank = bx - unit * self.v_split
                row, head = self._unit_coords(unit, state_indices.shape[0], head_order)
            else:
                row = by
                head = bx // self.v_split
                rank = bx - head * self.v_split
            self._item(*args, row, head, rank, True, False, row, head)
        if cutlass.const_expr(self.use_pdl and not self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()

    def _unit_coords(self, unit, rows, head_order):
        """(row, head) of work item ``unit`` in head-group order."""
        per_group = rows * self.head_group
        group = unit // per_group
        rest = unit - group * per_group
        row = rest // self.head_group
        head = group * self.head_group + rest - row * self.head_group
        if self.head_split > 1:
            step = head // self.head_split
            head = (head - step * self.head_split) * (self.heads // self.head_split) + step
        if self.with_head_order:
            head = head_order[head]
        return row, head

    @cute.jit
    def _item(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        conv_state: cute.Tensor,
        raw_gate: cute.Tensor,
        raw_beta: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state_indices: cute.Tensor,
        state: cute.Tensor,
        output_gate: cute.Tensor,
        norm_weight: cute.Tensor,
        output: cute.Tensor,
        mixed: cute.Tensor,
        rec_out: cute.Tensor,
        decay_s: cute.Tensor,
        gate_s: cute.Tensor,
        normw_s: cute.Tensor,
        scalars: cute.Tensor,
        state_s,
        tidx,
        lane,
        warp,
        stats,
        row,
        head,
        rank,
        wait_inputs: cutlass.Constexpr[bool],
        has_next,
        next_row,
        next_head,
    ):
        """One (request row, head, cluster rank) work item; ``wait_inputs`` places the PDL wait.
        With ``prefetch_next`` the ring rows of the next item (``has_next``) are issued after the
        last state row."""
        row_base = rank * self.rows_per_cta
        copy_f32x4 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), F32, num_bits_per_copy=128)
        copy_bf16x4 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF16, num_bits_per_copy=64)

        slot_req = state_indices[row]
        is_live = slot_req >= 0
        slot = cute.max(slot_req, cutlass.Int32(0))

        # Only the thread that stages a state row reads it back, so staging needs no CTA barrier.
        state_regs = cute.make_rmem_tensor((4, self.rows_per_warp), F32)
        # The FGS consumer prefetches the state (no producer writes it) and validates its inputs
        # per 16-B chunk instead of waiting for the primary grid.
        prefetch = wait_inputs and (self.early_state or self.fgs_consumer)
        if cutlass.const_expr(prefetch):
            self._load_state(state, state_s, state_regs, lane, warp, row_base, head, slot)
        if cutlass.const_expr(wait_inputs and self.use_pdl and not self.fgs_consumer):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(not prefetch):
            self._load_state(state, state_s, state_regs, lane, warp, row_base, head, slot)
        spins = cutlass.Int32(0)
        waited = cutlass.Int32(0)
        lanes_used = self.rows_per_cta // 4
        gate_words = cute.make_rmem_tensor((4,), cutlass.Int32)
        gate_addr = cutlass.Int64(0)
        if cutlass.const_expr(self.late_gate):
            # Warp 0 issues its output-gate load now and checks it only in the epilogue. Lanes 2i
            # and 2i+1 share the chunk of channels 8i..8i+7; idle lanes load the CTA's first one.
            if tidx < 32:
                ch0 = row_base + 4 * cute.min(lane, cutlass.Int32(lanes_used - 1))
                gate_addr = (output_gate[(None, head, row)].iterator + (ch0 - ch0 % 8)).toint()
                first = _ld_relaxed_v4(gate_addr, True)
                for j in cutlass.range_constexpr(4):
                    gate_words[j] = first[j]

        # Setup: 96 conv items (4 channels of q, k or v each) and 128 gate items. Conv items
        # are always the first iteration (threads >= 128), so tidx < CONV_ITEMS owns one.
        new_hist = cute.make_rmem_tensor((4, 3), BF16)
        for it in cutlass.range_constexpr((SETUP_ITEMS + self.threads - 1) // self.threads):
            item = tidx + it * self.threads
            if item < CONV_ITEMS:
                cl = item % 32
                sec = item // 32
                w_regs = cute.make_rmem_tensor((4, 4), F32)
                for tap in cutlass.range_constexpr(4):
                    cute.copy(
                        copy_f32x4,
                        _aligned(weight[(None, cl, tap, sec, head)], 16),
                        w_regs[(None, tap)],
                    )
                hist = cute.make_rmem_tensor((4, 3), BF16)
                for j in cutlass.range_constexpr(3):
                    cute.copy(
                        copy_bf16x4,
                        _aligned(conv_state[(None, j, cl, sec, head, slot)], 8),
                        hist[(None, j)],
                    )
                cur = cute.make_rmem_tensor((4,), BF16)
                if cutlass.const_expr(self.fgs_consumer):
                    # Lanes 2i and 2i+1 share the 16-B chunk of channels 8i..8i+7.
                    words = cute.make_rmem_tensor((4,), cutlass.Int32)
                    addr = x[(None, cl - cl % 2, sec, head, row)].iterator.toint()
                    spins, waited = self._fgs_wait(words, 1, addr, addr, addr, spins, waited)
                    odd = cl % 2 == 1
                    cur_bits = cute.recast_tensor(cur, cutlass.Int32)
                    cur_bits[0] = cutlass.Int32(cutlass.select_(odd, words[2], words[0]))
                    cur_bits[1] = cutlass.Int32(cutlass.select_(odd, words[3], words[1]))
                else:
                    cute.copy(copy_bf16x4, _aligned(x[(None, cl, sec, head, row)], 8), cur)
                silu = cute.make_rmem_tensor((4,), F32)
                for e in cutlass.range_constexpr(4):
                    # Lane-local element (channel e, tap t) sits at 3 * e + t.
                    acc = hist[(3 * e) % 4, (3 * e) // 4].to(F32) * w_regs[e, 0]
                    acc = acc + hist[(3 * e + 1) % 4, (3 * e + 1) // 4].to(F32) * w_regs[e, 1]
                    acc = acc + hist[(3 * e + 2) % 4, (3 * e + 2) // 4].to(F32) * w_regs[e, 2]
                    acc = acc + cur[e].to(F32) * w_regs[e, 3]
                    silu[e] = acc * _sigmoid(acc)
                    new_hist[(3 * e) % 4, (3 * e) // 4] = hist[(3 * e + 1) % 4, (3 * e + 1) // 4]
                    new_hist[(3 * e + 1) % 4, (3 * e + 1) // 4] = hist[
                        (3 * e + 2) % 4, (3 * e + 2) // 4
                    ]
                    new_hist[(3 * e + 2) % 4, (3 * e + 2) // 4] = cur[e]
                cute.copy(
                    copy_f32x4,
                    silu,
                    _aligned(
                        cute.make_tensor(
                            mixed.iterator + (sec * D + 4 * cl), cute.make_layout((4,))
                        ),
                        16,
                    ),
                )
            elif item < SETUP_ITEMS:
                c = item - CONV_ITEMS
                a = cute.math.exp(A_log[head], fastmath=True)
                if cutlass.const_expr(self.late_gate):
                    words = cute.make_rmem_tensor((8,), cutlass.Int32)
                    spins, waited = self._fgs_wait(
                        words,
                        2,
                        (raw_gate[(None, head, row)].iterator + (c - c % 8)).toint(),
                        (raw_beta[(None, row)].iterator + (head - head % 8)).toint(),
                        0,
                        spins,
                        waited,
                    )
                    g = _bf16_at(words[0], words[1], words[2], words[3], c % 8) + dt_bias[c, head]
                    decay_s[c] = cute.math.exp(self.lower_bound * _sigmoid(a * g), fastmath=True)
                    normw_s[c] = norm_weight[c]
                    if c == 0:
                        scalars[0] = _sigmoid(
                            _bf16_at(words[4], words[5], words[6], words[7], head % 8)
                        )
                elif cutlass.const_expr(self.fgs_consumer):
                    # Eight lanes share each 16-B chunk of g and of the output gate; every lane
                    # also checks the chunk holding this head's beta.
                    words = cute.make_rmem_tensor((12,), cutlass.Int32)
                    spins, waited = self._fgs_wait(
                        words,
                        3,
                        (raw_gate[(None, head, row)].iterator + (c - c % 8)).toint(),
                        (output_gate[(None, head, row)].iterator + (c - c % 8)).toint(),
                        (raw_beta[(None, row)].iterator + (head - head % 8)).toint(),
                        spins,
                        waited,
                    )
                    g = _bf16_at(words[0], words[1], words[2], words[3], c % 8) + dt_bias[c, head]
                    decay_s[c] = cute.math.exp(self.lower_bound * _sigmoid(a * g), fastmath=True)
                    gate_s[c] = _sigmoid(_bf16_at(words[4], words[5], words[6], words[7], c % 8))
                    normw_s[c] = norm_weight[c]
                    if c == 0:
                        scalars[0] = _sigmoid(
                            _bf16_at(words[8], words[9], words[10], words[11], head % 8)
                        )
                else:
                    g = raw_gate[c, head, row].to(F32) + dt_bias[c, head]
                    decay_s[c] = cute.math.exp(self.lower_bound * _sigmoid(a * g), fastmath=True)
                    gate_s[c] = _sigmoid(output_gate[c, head, row].to(F32))
                    normw_s[c] = norm_weight[c]
                    if c == 0:
                        scalars[0] = _sigmoid(raw_beta[head, row].to(F32))

        if cutlass.const_expr(self.with_stats):
            if lane == 0:
                if spins > 0:
                    cute.arch.atomic_add(stats.iterator, spins)
                if waited > 0:
                    cute.arch.atomic_add(stats.iterator + 1, waited)

        if cutlass.const_expr(self.v_split > 1):
            # Every CTA of the cluster reads the old conv window; rank 0 overwrites it only
            # after all of them have passed this point.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()
        if tidx < CONV_ITEMS:
            if is_live:
                if rank == 0:
                    cl = tidx % 32
                    sec = tidx // 32
                    for j in cutlass.range_constexpr(3):
                        cute.copy(
                            copy_bf16x4,
                            new_hist[(None, j)],
                            _aligned(conv_state[(None, j, cl, sec, head, slot)], 8),
                        )

        q = cute.make_rmem_tensor((4,), F32)
        k = cute.make_rmem_tensor((4,), F32)
        decay = cute.make_rmem_tensor((4,), F32)
        for e in cutlass.range_constexpr(4):
            q[e] = mixed[4 * lane + e]
            k[e] = mixed[D + 4 * lane + e]
            decay[e] = decay_s[4 * lane + e]
        q_sq = F32(0.0)
        k_sq = F32(0.0)
        for e in cutlass.range_constexpr(4):
            q_sq += q[e] * q[e]
            k_sq += k[e] * k[e]
        q_sq = cute.arch.warp_reduction_sum(q_sq)
        k_sq = cute.arch.warp_reduction_sum(k_sq)
        q_factor = cute.math.rsqrt(q_sq + L2_EPS) * self.q_scale
        k_factor = cute.math.rsqrt(k_sq + L2_EPS)
        for e in cutlass.range_constexpr(4):
            q[e] = q[e] * q_factor
            k[e] = k[e] * k_factor
        beta = scalars[0]

        for r in cutlass.range_constexpr(self.rows_per_warp):
            local_row = warp * self.rows_per_warp + r
            v_row = row_base + local_row
            j = r - self.reg_rows
            if cutlass.const_expr(j >= 0):
                # Rows committed after row j: the rest of the ring, capped at the last row.
                cute.arch.cp_async_wait_group(min(self.ring - 1, self.smem_rows - 1 - j))
                cute.autovec_copy(
                    state_s[(None, lane, warp * self.ring + j % self.ring)], state_regs[(None, r)]
                )
            hk = F32(0.0)
            for e in cutlass.range_constexpr(4):
                state_regs[e, r] = state_regs[e, r] * decay[e]
                hk += state_regs[e, r] * k[e]
            if cutlass.const_expr(0 <= j and j + self.ring < self.smem_rows):
                # Refill only after the slot's row has been consumed into registers.
                self._issue_ring_row(
                    state, state_s, lane, warp, row_base, head, slot, j + self.ring
                )
            hk = cute.arch.warp_reduction_sum(hk)
            u = (mixed[2 * D + v_row] - hk) * beta
            hq = F32(0.0)
            for e in cutlass.range_constexpr(4):
                state_regs[e, r] = state_regs[e, r] + k[e] * u
                hq += state_regs[e, r] * q[e]
            if is_live:
                cute.copy(
                    copy_f32x4,
                    state_regs[(None, r)],
                    _aligned(state[(None, lane, v_row, head, slot)], 16),
                )
            hq = cute.arch.warp_reduction_sum(hq)
            if lane == 0:
                rec_out[local_row] = hq

        if cutlass.const_expr(self.prefetch_next):
            # Every ring slot of this warp has been read back, and only the next item itself
            # writes the state rows it reads.
            if has_next:
                self._issue_ring(
                    state,
                    state_s,
                    lane,
                    warp,
                    row_base,
                    next_head,
                    cute.max(state_indices[next_row], cutlass.Int32(0)),
                )

        cute.arch.barrier()

        # Gated RMSNorm: lane l of warp 0 owns this CTA's rows 4l..4l+3.
        o_regs = cute.make_rmem_tensor((4,), F32)
        sum_sq = F32(0.0)
        if tidx < 32:
            for e in cutlass.range_constexpr(4):
                o_regs[e] = F32(0.0)
                if lane < lanes_used:
                    o_regs[e] = rec_out[4 * lane + e]
                sum_sq += o_regs[e] * o_regs[e]
            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
        if cutlass.const_expr(self.v_split > 1):
            if tidx == 0:
                scalars[1] = sum_sq
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if tidx < 32:
                total = F32(0.0)
                for peer in cutlass.range_constexpr(self.v_split):
                    total += _load_peer_f32((scalars.iterator + 1).toint(), peer)
                sum_sq = total
        gate_r = cute.make_rmem_tensor((4,), F32)
        if cutlass.const_expr(self.late_gate):
            if tidx < 32:
                gate_spins, gate_waited = self._fgs_wait(
                    gate_words,
                    1,
                    gate_addr,
                    gate_addr,
                    gate_addr,
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    loaded=True,
                )
                ch0 = row_base + 4 * cute.min(lane, cutlass.Int32(lanes_used - 1))
                for e in cutlass.range_constexpr(4):
                    gate_r[e] = _sigmoid(
                        _bf16_at(
                            gate_words[0],
                            gate_words[1],
                            gate_words[2],
                            gate_words[3],
                            (ch0 + e) % 8,
                        )
                    )
                if cutlass.const_expr(self.with_stats):
                    if lane == 0:
                        if gate_spins > 0:
                            cute.arch.atomic_add(stats.iterator, gate_spins)
                        if gate_waited > 0:
                            cute.arch.atomic_add(stats.iterator + 1, gate_waited)
        if cutlass.const_expr(self.store16):
            # Even lanes store their own four channels and their odd neighbour's as one 16-B chunk.
            if tidx < 32:
                rstd = cute.math.rsqrt(sum_sq * (1.0 / D) + self.norm_eps)
                yf = cute.make_rmem_tensor((8,), F32)
                for e in cutlass.range_constexpr(4):
                    yf[e] = F32(0.0)
                    if lane < lanes_used:
                        ch = row_base + 4 * lane + e
                        if cutlass.const_expr(self.late_gate):
                            yf[e] = o_regs[e] * rstd * normw_s[ch] * gate_r[e]
                        else:
                            yf[e] = o_regs[e] * rstd * normw_s[ch] * gate_s[ch]
                for e in cutlass.range_constexpr(4):
                    yf[4 + e] = cute.arch.shuffle_sync_down(yf[e], 1)
                if lane < lanes_used:
                    if lane % 2 == 0:
                        packed = cute.make_rmem_tensor((4,), cutlass.Int32)
                        for p in cutlass.range_constexpr(4):
                            packed[p] = _bf16x2_pzo(yf[2 * p], yf[2 * p + 1], self.hw_pzo)
                            if slot_req < 0:
                                packed[p] = cutlass.Int32(0)
                        chunk = output[(None, rank * (lanes_used // 2) + lane // 2, head, row)]
                        _store_global_v4(
                            chunk.iterator.toint(), packed[0], packed[1], packed[2], packed[3]
                        )
        else:
            if tidx < lanes_used:
                rstd = cute.math.rsqrt(sum_sq * (1.0 / D) + self.norm_eps)
                y = cute.make_rmem_tensor((4,), BF16)
                for e in cutlass.range_constexpr(4):
                    ch = row_base + 4 * lane + e
                    if cutlass.const_expr(self.late_gate):
                        y[e] = (o_regs[e] * rstd * normw_s[ch] * gate_r[e]).to(BF16)
                    else:
                        y[e] = (o_regs[e] * rstd * normw_s[ch] * gate_s[ch]).to(BF16)
                    if slot_req < 0:
                        y[e] = BF16(0.0)
                cute.copy(
                    copy_bf16x4,
                    y,
                    _aligned(output[(None, rank * lanes_used + lane, head, row)], 8),
                )
        if cutlass.const_expr(self.v_split > 1):
            # Keep this CTA's shared memory alive until every peer has read it.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def _fgs_wait(
        self,
        words: cute.Tensor,
        chunks: cutlass.Constexpr[int],
        a0,
        a1,
        a2,
        spins,
        waited,
        loaded: cutlass.Constexpr[bool] = False,
    ):
        """Load ``chunks`` 16-B chunks at a0..a2 into ``words`` (``loaded``: they already hold a
        first load) and return once every lane of the warp holds written ones; returns
        (spins + reload rounds, waited)."""
        addrs = (a0, a1, a2)
        bad = cutlass.Boolean(False)
        for i in cutlass.range_constexpr(chunks):
            if cutlass.const_expr(not loaded):
                first = _ld_relaxed_v4(addrs[i], False)
                for j in cutlass.range_constexpr(4):
                    words[4 * i + j] = first[j]
            bad = bad | ((words[4 * i] & 0xFFFF) == BF16_NEG_ZERO)
        pending = cute.arch.vote_any_sync(bad)
        sleep_ns = cutlass.Int32(SPIN_NS_MIN)
        rounds = cutlass.Int32(0)
        while pending:
            if waited > 0 or rounds >= self.max_spins:
                # The primary grid has completed: the values read next are final.
                cute.arch.griddepcontrol_wait()
                waited = cutlass.Int32(1)
            else:
                _nanosleep(sleep_ns)
                sleep_ns = cute.min(sleep_ns * 2, cutlass.Int32(SPIN_NS_MAX))
            rounds = rounds + 1
            bad = cutlass.Boolean(False)
            for i in cutlass.range_constexpr(chunks):
                w = _ld_relaxed_v4(addrs[i], True)
                for j in cutlass.range_constexpr(4):
                    words[4 * i + j] = w[j]
                bad = bad | ((words[4 * i] & 0xFFFF) == BF16_NEG_ZERO)
            pending = cute.arch.vote_any_sync(bad) & (waited == 0)
        return spins + rounds, waited


@cache
def _compile(
    heads: int,
    v_split: int,
    lower_bound: float,
    q_scale: float,
    norm_eps: float,
    early_state: bool,
    use_pdl: bool,
    reg_rows: int,
    min_blocks: int,
    warps: int,
    ring: int,
    early_trigger: bool = False,
    store16: bool = False,
    hw_pzo: bool = False,
    head_group: int = 0,
    persistent_ctas: int = 0,
    fgs_consumer: bool = False,
    max_spins: int = 0,
    with_stats: bool = False,
    head_split: int = 1,
    prefetch_next: bool = False,
    smem_carveout: int | None = None,
    max_regs: int = 0,
    late_gate: bool = False,
    with_head_order: bool = False,
):
    d = heads * D
    rows, slots, conv_slots = cute.sym_int(), cute.sym_int(), cute.sym_int()

    def compact(dtype, shape, align=16):
        return make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=align
        )

    args = (
        make_fake_tensor(
            BF16, (rows, 3 * d), stride=(cute.sym_int64(divisibility=4), 1), assumed_align=8
        ),
        compact(F32, (3, 4, d)),
        make_fake_tensor(
            BF16,
            (conv_slots, 3 * d, 3),
            stride=(cute.sym_int64(divisibility=4), 3, 1),
            assumed_align=8,
        ),
        make_fake_tensor(BF16, (rows, d), stride=(cute.sym_int64(), 1), assumed_align=2),
        make_fake_tensor(BF16, (rows, heads), stride=(cute.sym_int64(), 1), assumed_align=2),
        compact(F32, (heads,), 4),
        compact(F32, (d,)),
        compact(cutlass.Int32, (rows,), 4),
        make_fake_tensor(
            F32,
            (slots, heads, D, D),
            stride=(cute.sym_int64(divisibility=4), D * D, D, 1),
            assumed_align=16,
        ),
        make_fake_tensor(BF16, (rows, heads, D), stride=(cute.sym_int64(), D, 1), assumed_align=2),
        compact(F32, (D,)),
        make_fake_tensor(
            BF16,
            (rows, heads, D),
            stride=(cute.sym_int64(divisibility=8 if store16 else 4), D, 1),
            assumed_align=16 if store16 else 8,
        ),
        make_fake_tensor(cutlass.Int32, (2,), stride=(1,), assumed_align=8) if with_stats else None,
        make_fake_tensor(cutlass.Int32, (heads,), stride=(1,), assumed_align=4)
        if with_head_order
        else None,
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    op = KdaDecodeT1(
        heads,
        v_split,
        lower_bound,
        q_scale,
        norm_eps,
        early_state,
        use_pdl,
        reg_rows,
        min_blocks,
        warps,
        ring,
        early_trigger,
        store16,
        hw_pzo,
        head_group,
        persistent_ctas,
        fgs_consumer,
        max_spins,
        with_stats,
        head_split,
        prefetch_next,
        smem_carveout,
        late_gate,
        with_head_order,
    )
    options = "--enable-tvm-ffi"
    if max_regs:
        options += f" --ptxas-options '--maxrregcount={max_regs}'"
    return cute.compile(op, *args, options=options)


def pick_config(batch: int) -> dict:
    """Launch configuration for a decode batch at H=96, tuned on Rubin (228 KB shared memory per SM)."""
    if batch < 4:
        return dict(v_split=1, warps=16, ring=4, min_blocks=2)
    return dict(v_split=1, warps=8, ring=8, min_blocks=4)


@cache
def smem_per_sm(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_multiprocessor


def staging_plan(
    device_index: int, v_split: int, warps: int, stage_smem: bool
) -> tuple[int, int, int]:
    """(reg_rows, ring, min_blocks): the deepest shared-memory ring that lets 1024 / threads
    CTAs share an SM, or every row in registers without staging."""
    rows_per_warp = D // v_split // warps
    target = max(1, 1024 // (32 * warps))
    if not stage_smem:
        return rows_per_warp, 0, target
    budget = smem_per_sm(device_index)
    while target > 1:
        # 1 KB per CTA is reserved by the runtime.
        if budget // target >= cta_smem_bytes(warps, 1) + 1024:
            break
        target -= 1
    ring = (budget // target - 1024 - cta_smem_bytes(warps, 0)) // (warps * D * 4)
    return 0, max(1, min(rows_per_warp, ring)), target


def pack_conv_weight(w_q_t: torch.Tensor, w_k_t: torch.Tensor, w_v_t: torch.Tensor) -> torch.Tensor:
    """``trtllm::kda_decode``'s tap-major BF16 ``[4, H*128]`` conv weights -> FP32 ``[3, 4, H*128]``."""
    return torch.stack((w_q_t, w_k_t, w_v_t)).float().contiguous()


def unsupported(
    x,
    conv_weight,
    conv_state,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    state_indices,
    state,
    output_gate,
    norm_weight,
    output,
) -> str | None:
    """Name of the first ``kda_decode_t1`` argument with an unsupported layout or dtype, else None."""
    if x.ndim != 2 or x.shape[1] % (3 * D):
        return "x"
    B, three_d = x.shape
    heads = three_d // (3 * D)
    d = heads * D
    checks = (
        (
            x.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.stride(0) % 4 == 0
            and x.data_ptr() % 8 == 0,
            "x",
        ),
        (
            conv_weight.dtype == torch.float32
            and conv_weight.shape == (3, 4, d)
            and conv_weight.is_contiguous(),
            "conv_weight",
        ),
        (
            conv_state.dtype == torch.bfloat16
            and conv_state.shape[1:] == (3 * d, 3)
            and conv_state.stride()[1:] == (3, 1)
            and conv_state.stride(0) % 4 == 0
            and conv_state.data_ptr() % 8 == 0,
            "conv_state",
        ),
        (
            raw_gate.dtype == torch.bfloat16
            and raw_gate.shape == (B, d)
            and raw_gate.stride(1) == 1,
            "raw_gate",
        ),
        (
            raw_beta.dtype == torch.bfloat16
            and raw_beta.shape == (B, heads)
            and raw_beta.stride(1) == 1,
            "raw_beta",
        ),
        (
            A_log.dtype == torch.float32 and A_log.shape == (heads,) and A_log.is_contiguous(),
            "A_log",
        ),
        (
            dt_bias.dtype == torch.float32 and dt_bias.shape == (d,) and dt_bias.is_contiguous(),
            "dt_bias",
        ),
        (
            state_indices.dtype == torch.int32
            and state_indices.shape == (B,)
            and state_indices.is_contiguous(),
            "state_indices",
        ),
        (
            state.dtype == torch.float32
            and state.shape[1:] == (heads, D, D)
            and state.stride()[1:] == (D * D, D, 1)
            and state.stride(0) % 4 == 0
            and state.data_ptr() % 16 == 0,
            "state",
        ),
        (
            output_gate.dtype == torch.bfloat16
            and output_gate.shape == (B, heads, D)
            and output_gate.stride()[1:] == (D, 1),
            "output_gate",
        ),
        (
            norm_weight.dtype == torch.float32
            and norm_weight.shape == (D,)
            and norm_weight.is_contiguous(),
            "norm_weight",
        ),
        (
            output.dtype == torch.bfloat16
            and output.shape == (B, heads, D)
            and output.stride()[1:] == (D, 1)
            and output.stride(0) % 4 == 0
            and output.data_ptr() % 8 == 0,
            "output",
        ),
    )
    for ok, name in checks:
        if not ok:
            return name
    return None


def kda_decode_t1(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
    *,
    lower_bound: float,
    norm_eps: float,
    scale: float = D**-0.5,
    v_split: int | None = None,
    warps: int | None = None,
    stage_smem: bool | None = None,
    reg_rows: int | None = None,
    ring: int | None = None,
    min_blocks: int | None = None,
    early_state: bool = False,
    use_pdl: bool = True,
    fgs_producer: bool = False,
    fgs_persistent: bool = False,
    fgs_head_group: int | None = None,
    fgs_ctas: int | None = None,
    fgs_consumer: bool = False,
    fgs_max_spins: int = 1 << 16,
    fgs_stats: torch.Tensor | None = None,
    fgs_head_split: int = 1,
    fgs_prefetch_next: bool = False,
    fgs_smem_carveout: int | None = None,
    max_regs: int = 0,
    fgs_late_gate: bool = False,
    fgs_head_order: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused T=1 KDA decode; updates ``conv_state`` and ``state`` rows in place.

    x [B, 3*H*128] (q | k | v projection columns, any row stride), conv_weight FP32
    [3, 4, H*128] (``pack_conv_weight``), conv_state BF16 [slots, 3*H*128, 3] with
    strides (s, 3, 1), raw_gate BF16 [B, H*128], raw_beta BF16 [B, H] (any row stride),
    A_log FP32 [H], dt_bias FP32 [H*128], state_indices int32 [B], state FP32
    [slots, H, 128, 128], output_gate BF16 [B, H, 128] (any token stride),
    norm_weight FP32 [128], output BF16 [B, H, 128].

    With ``fgs_producer`` the output is published for a fine-grained-sync consumer
    that reads it as BF16 [B, H*128] (row stride ``output.stride(0)``, a multiple of
    8 elements; base 16-B aligned) and checks element 0 of every 16-B chunk for the
    -0.0 sentinel: each chunk (8 channels of one head) is written exactly once by one
    16-B store, never as -0.0 (rows with a negative slot publish +0.0), and work
    items are issued for all rows of ``fgs_head_group`` heads before the next group
    (default H: heads fastest, the default build's order; 1: rows fastest, which costs
    2-5 % at B 16-64 on Rubin). The kernel never
    writes the sentinel; resetting the buffer to -0.0 between steps is the caller's
    job, as is launching the consumer with PDL (dependents are released at kernel
    entry). ``fgs_persistent`` launches at most one wave (``fgs_ctas``, default every
    resident CTA slot: SMs x ``min_blocks``) that loops over the items in that order,
    so the entry trigger releases the consumer at once instead of after the last wave
    has started. ``fgs_head_split`` > 1 interleaves the head order over that many equal
    head ranges (heads 0, H/s, ..., 1, H/s + 1, ...), the order in which a split-K
    consumer with that split reads its K slices concurrently. ``fgs_prefetch_next``
    (persistent) issues the next item's staged state rows as soon as a warp has read
    its last row of the current item. ``fgs_smem_carveout`` is the launch's preferred
    shared-memory carveout (percent): a persistent grid that leaves room for a consumer
    CTA on its SMs needs the consumer's carveout, not the default one sized for this
    kernel alone. ``max_regs`` caps registers per thread (``--maxrregcount``) and
    derives ``min_blocks`` from it unless given: above 64 registers at 8 warps, at most
    3 CTAs fit an SM's 64 K registers.

    ``fgs_consumer`` reads x, the output gate, raw_gate and raw_beta as fine-grained-sync
    slots (see the module docstring) instead of waiting for the PDL primary grid, also in
    a persistent grid (every item checks its own inputs). Each of
    them must be armed with -0.0 before its producer runs, have 16-B aligned rows whose
    stride is a multiple of 8 elements, and be published by one 16-B store per 16-B
    chunk, never as -0.0. The state and conv pools and ``state_indices`` are read before
    any input is validated, so they must be complete when the kernel launches.
    ``fgs_head_order`` (int32 [H] on the device, a permutation of the heads; producer only, in
    place of ``fgs_head_split``) is the head of each head-group position: e.g. the heads a
    producer of the inputs publishes first, first.

    ``fgs_late_gate`` (consumer only, ignored otherwise) validates the output gate in the
    epilogue instead of the setup, after the state update. Warps
    that still see a sentinel after ``fgs_max_spins`` reload rounds wait for the primary
    grid and take the values they then read. ``fgs_stats`` (int32 [2], zeroed by the
    caller) accumulates reload rounds and fallbacks per setup warp.
    """
    B = x.shape[0]
    heads = x.shape[1] // (3 * D)
    if B == 0:
        return output
    if lower_bound is None or not lower_bound < 0:
        raise ValueError("kda_decode_t1 implements the negative lower-bound gate only")
    reason = unsupported(
        x,
        conv_weight,
        conv_state,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_indices,
        state,
        output_gate,
        norm_weight,
        output,
    )
    if reason is not None:
        raise ValueError(f"kda_decode_t1: unsupported {reason} layout or dtype")
    if fgs_producer and (output.stride(0) % 8 or output.data_ptr() % 16):
        raise ValueError("kda_decode_t1: fgs_producer needs 16-B aligned output rows")
    if fgs_persistent and not fgs_producer:
        raise ValueError("kda_decode_t1: fgs_persistent needs fgs_producer")
    if fgs_prefetch_next and not fgs_persistent:
        raise ValueError("kda_decode_t1: fgs_prefetch_next needs fgs_persistent")
    if fgs_consumer:
        if early_state or not use_pdl:
            raise ValueError("kda_decode_t1: fgs_consumer needs PDL and no early_state")
        for name, t in (
            ("x", x),
            ("output_gate", output_gate),
            ("raw_gate", raw_gate),
            ("raw_beta", raw_beta),
        ):
            if t.stride(0) % 8 or t.data_ptr() % 16:
                raise ValueError(f"kda_decode_t1: fgs_consumer needs 16-B aligned {name} rows")
    with_stats = fgs_consumer and fgs_stats is not None
    if with_stats and (fgs_stats.dtype != torch.int32 or fgs_stats.numel() < 2):
        raise ValueError("kda_decode_t1: fgs_stats must be an int32 tensor of 2 elements")
    if fgs_head_group is None:
        fgs_head_group = heads
    if fgs_producer and heads % fgs_head_group:
        raise ValueError("kda_decode_t1: fgs_head_group must divide the head count")
    if fgs_head_order is not None and (
        not fgs_producer
        or fgs_head_split != 1
        or fgs_head_order.dtype != torch.int32
        or fgs_head_order.shape != (heads,)
        or not fgs_head_order.is_contiguous()
        or fgs_head_order.device != x.device
    ):
        raise ValueError(
            "kda_decode_t1: fgs_head_order needs fgs_producer, no fgs_head_split and an int32 [H] "
            "tensor on the input's device"
        )
    if fgs_head_split != 1 and (not fgs_producer or heads % fgs_head_split):
        raise ValueError(
            "kda_decode_t1: fgs_head_split needs fgs_producer and must divide the head count"
        )
    min_blocks_arg = min_blocks
    if all(v is None for v in (v_split, warps, stage_smem, reg_rows, ring, min_blocks)):
        cfg = pick_config(B)
        v_split, warps, ring, min_blocks = (
            cfg["v_split"],
            cfg["warps"],
            cfg["ring"],
            cfg["min_blocks"],
        )
    if v_split is None:
        v_split = 1
    if stage_smem is None:
        stage_smem = True
    if warps is None:
        warps = 16 // v_split
    device_index = x.device.index
    plan_reg_rows, plan_ring, plan_min_blocks = staging_plan(
        device_index, v_split, warps, stage_smem
    )
    if reg_rows is None:
        reg_rows = plan_reg_rows
    if ring is None:
        ring = plan_ring
    if min_blocks is None:
        min_blocks = plan_min_blocks
    if max_regs and min_blocks_arg is None:
        min_blocks = max(1, 65536 // (32 * warps * max_regs))
    ring = min(ring, D // v_split // warps - reg_rows)
    if max(min_blocks, 1) * (cta_smem_bytes(warps, ring) + 1024) > smem_per_sm(device_index):
        raise ValueError(
            f"kda_decode_t1: {min_blocks} CTAs per SM do not fit the SM's shared memory"
        )
    if fgs_prefetch_next and ring < 1:
        raise ValueError("kda_decode_t1: fgs_prefetch_next needs staged state rows (ring >= 1)")
    persistent_ctas = 0
    if fgs_persistent:
        if early_state:
            raise ValueError("kda_decode_t1: fgs_persistent does not support early_state")
        persistent_ctas = fgs_ctas or (
            torch.cuda.get_device_properties(device_index).multi_processor_count
            * max(min_blocks, 1)
        )
        persistent_ctas -= persistent_ctas % v_split
        if persistent_ctas <= 0:
            raise ValueError("kda_decode_t1: fgs_ctas must hold at least one cluster")
    fn = _compile(
        heads,
        v_split,
        float(lower_bound),
        float(scale),
        float(norm_eps),
        early_state,
        use_pdl,
        reg_rows,
        min_blocks,
        warps,
        ring,
        early_trigger=fgs_producer,
        store16=fgs_producer,
        hw_pzo=fgs_producer,
        head_group=fgs_head_group if fgs_producer else 0,
        persistent_ctas=persistent_ctas,
        fgs_consumer=fgs_consumer,
        max_spins=fgs_max_spins if fgs_consumer else 0,
        with_stats=with_stats,
        head_split=fgs_head_split if fgs_producer else 1,
        prefetch_next=fgs_prefetch_next,
        smem_carveout=fgs_smem_carveout,
        max_regs=max_regs,
        late_gate=fgs_late_gate and fgs_consumer,
        with_head_order=fgs_head_order is not None,
    )
    fn(
        x,
        conv_weight,
        conv_state,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_indices,
        state,
        output_gate,
        norm_weight,
        output,
        fgs_stats if with_stats else None,
        fgs_head_order,
    )
    return output
