# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA output gate ``out = o * sigmoid(g)`` with TMA tile loads (prefill, SM107).

o, g, out: ``[T, N]`` bf16, viewed as ``[T * N / 256, 256]``; a CTA gates a 32 x 256 tile (8192 elements): one TMA
load per operand into shared memory, one mbarrier, 256 threads x 32 elements. The sigmoid is ``mla_output_gate``'s
(fast path with the exact fallback), so the result is bit-identical to ``o * g.sigmoid()``.

``pdl``: launched with PDL, ``griddepcontrol.wait`` before the loads, trigger after the stores (the FGS-off twin).
``fgs_consumer``: no grid wait; o and g are loaded with ``report_valid`` (bf16 -0.0 per 16 B chunk, element 0) and
a tile that landed invalid is re-issued until valid; after ``fgs_max_spins`` invalid landings the CTA waits for the
grid once and reloads plainly. Dependents are triggered at entry.
``fgs_counters``: after its stores each CTA adds 1 to ``counters[first token row // fgs_counter_rows]``
(``red.release.gpu``); ``fgs_counter_rows * N`` must be a multiple of the tile so no CTA straddles a block.
"""

from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import SmemAllocator

try:
    from .mla_fgs import STATS_LEN
    from .mla_output_gate import _sigmoid_bf16_as_f32
except ImportError:
    from mla_fgs import STATS_LEN
    from mla_output_gate import _sigmoid_bf16_as_f32

THREADS = 256
TILE_ROWS = 32
TILE_COLS = 256
TILE_ELEMS = TILE_ROWS * TILE_COLS
VEC = 16
PER_THREAD = TILE_ELEMS // THREADS
TILE_BYTES = TILE_ELEMS * 2


@dsl_user_op
def _release_add(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _Gate:
    def __init__(self, fast: bool, pdl: bool, consumer: bool, max_spins: int, with_stats: bool, publish: bool):
        self.fast = fast
        self.pdl = pdl
        self.consumer = consumer
        self.max_spins = max_spins
        self.with_stats = with_stats
        self.publish = publish

        @cute.struct
        class SharedStorage:
            bar: cute.struct.MemRange[cutlass.Int64, 1]
            sO: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, TILE_ELEMS], 128]
            sG: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, TILE_ELEMS], 128]

        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mG: cute.Tensor,
        mOut: cute.Tensor,
        stats_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        cnt_block: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        tile = (TILE_ROWS, TILE_COLS)
        op = cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.make_layout((TILE_ROWS, TILE_COLS, 1), stride=(TILE_COLS, 1, TILE_ELEMS))
        smem_tile = cute.slice_(smem_layout, (None, None, 0))
        tma_o, mO_tma = cpasync.make_tiled_tma_atom(op, mO, smem_tile, tile)
        tma_g, mG_tma = cpasync.make_tiled_tma_atom(op, mG, smem_tile, tile)
        rows = mO.shape[0]
        num_ctas = (rows + TILE_ROWS - 1) // TILE_ROWS
        self.kernel(tma_o, mO_tma, tma_g, mG_tma, mOut, stats_ptr, cnt_ptr, cnt_block, smem_layout).launch(
            grid=(num_ctas, 1, 1),
            block=(THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.pdl or self.consumer,
        )

    @cute.jit
    def _load(self, tma_o, tOgO, tOsO, tma_g, tGgG, tGsG, bar, bidx, check: cutlass.Constexpr):
        # Whole-warp: the TMA copy elects its issuing lane itself.
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, 2 * TILE_BYTES)
        if cutlass.const_expr(check):
            cute.copy(tma_o, tOgO[(None, bidx, 0)], tOsO[(None, 0)], tma_bar_ptr=bar, report_valid=True)
            cute.copy(tma_g, tGgG[(None, bidx, 0)], tGsG[(None, 0)], tma_bar_ptr=bar, report_valid=True)
        else:
            cute.copy(tma_o, tOgO[(None, bidx, 0)], tOsO[(None, 0)], tma_bar_ptr=bar)
            cute.copy(tma_g, tGgG[(None, bidx, 0)], tGsG[(None, 0)], tma_bar_ptr=bar)

    @cute.kernel
    def kernel(
        self,
        tma_o: cute.CopyAtom,
        mO: cute.Tensor,
        tma_g: cute.CopyAtom,
        mG: cute.Tensor,
        mOut: cute.Tensor,
        stats_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        cnt_block: cutlass.Int64,
        smem_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rows = mO.shape[0]
        if cutlass.const_expr(self.consumer):
            cute.arch.griddepcontrol_launch_dependents()
        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        bar = storage.bar.data_ptr()
        if tidx == 0:
            cpasync.prefetch_descriptor(tma_o)
            cpasync.prefetch_descriptor(tma_g)
            if cutlass.const_expr(self.consumer):
                cute.arch.mbarrier_v1_init(bar, 1)
            else:
                cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        sO = storage.sO.get_tensor(smem_layout)
        sG = storage.sG.get_tensor(smem_layout)
        gO = cute.local_tile(mO, (TILE_ROWS, TILE_COLS), (None, None))
        gG = cute.local_tile(mG, (TILE_ROWS, TILE_COLS), (None, None))
        tOsO, tOgO = cpasync.tma_partition(
            tma_o, 0, cute.make_layout(1), cute.group_modes(sO, 0, 2), cute.group_modes(gO, 0, 2)
        )
        tGsG, tGgG = cpasync.tma_partition(
            tma_g, 0, cute.make_layout(1), cute.group_modes(sG, 0, 2), cute.group_modes(gG, 0, 2)
        )
        if cutlass.const_expr(self.pdl and not self.consumer):
            cute.arch.griddepcontrol_wait()
        if warp_idx == 0:
            self._load(tma_o, tOgO, tOsO, tma_g, tGgG, tGsG, bar, bidx, self.consumer)

        phase = cutlass.Int32(0)
        retries = cutlass.Int32(0)
        if cutlass.const_expr(self.consumer):
            # Every thread observes each landing and the CTA barrier keeps the re-issue behind the slowest observer,
            # so no thread can miss a phase. A sentinel tile is re-issued with report_valid up to max_spins times,
            # then once plainly after the grid wait (the producer is complete by then).
            pending = cutlass.Int32(1)
            while pending != 0:
                complete = cutlass.Int32(0)
                invalid = cutlass.Int32(0)
                while complete == 0:
                    res = cute.arch.mbarrier_try_wait_primary(bar, phase)
                    complete = cutlass.select_(res[0], cutlass.Int32(1), cutlass.Int32(0))
                    invalid = cutlass.select_(res[1], cutlass.Int32(1), cutlass.Int32(0))
                phase = phase ^ 1
                cute.arch.barrier()
                if invalid != 0:
                    retries = retries + 1
                    if warp_idx == 0:
                        if retries > self.max_spins:
                            cute.arch.griddepcontrol_wait()
                            self._load(tma_o, tOgO, tOsO, tma_g, tGgG, tGsG, bar, bidx, False)
                        else:
                            self._load(tma_o, tOgO, tOsO, tma_g, tGgG, tGsG, bar, bidx, True)
                else:
                    pending = cutlass.Int32(0)
        else:
            done = cutlass.Int32(0)
            while done == 0:
                done = cutlass.select_(cute.arch.mbarrier_try_wait(bar, phase), cutlass.Int32(1), cutlass.Int32(0))

        so = cutlass.make_array_view(cute.make_tensor(storage.sO.data_ptr(), cute.make_layout((TILE_ELEMS,))))
        sg = cutlass.make_array_view(cute.make_tensor(storage.sG.data_ptr(), cute.make_layout((TILE_ELEMS,))))
        out = cutlass.make_array_view(mOut)
        row = tidx // (TILE_COLS // PER_THREAD)
        col0 = (tidx % (TILE_COLS // PER_THREAD)) * PER_THREAD
        grow = cutlass.Int64(bidx) * TILE_ROWS + row
        if grow < cutlass.Int64(rows):
            sidx = row * TILE_COLS + col0
            gidx = grow * TILE_COLS + col0
            for c in cutlass.range_constexpr(PER_THREAD // VEC):
                o = so.load(sidx + c * VEC, VEC, alignment=2 * VEC).to(cutlass.Float32)
                g = sg.load(sidx + c * VEC, VEC, alignment=2 * VEC).to(cutlass.Float32)
                res = []
                for e in cutlass.range_constexpr(VEC):
                    res.append(o[e] * _sigmoid_bf16_as_f32(g[e], self.fast))
                out.store(
                    cutlass.Vector.from_elements(tuple(res), dtype=cutlass.Float32).to(cutlass.BFloat16),
                    idx=gidx + c * VEC,
                    vector_size=VEC,
                    alignment=2 * VEC,
                )
        if cutlass.const_expr(self.publish):
            cute.arch.barrier()
            if tidx == 0:
                block = (cutlass.Int64(bidx) * TILE_ELEMS) // cnt_block
                _release_add(cnt_ptr.toint() + block * 4, cutlass.Int32(1))
        if cutlass.const_expr(self.with_stats):
            if tidx == 0 and retries > 0:
                stats = cute.make_tensor(stats_ptr, cute.make_layout((STATS_LEN,)))
                cute.arch.atomic_add(stats.iterator, retries)
                cute.arch.atomic_add(stats.iterator + 1, cutlass.Int32(1))
        if cutlass.const_expr(self.pdl and not self.consumer):
            cute.arch.griddepcontrol_launch_dependents()


_COMPILED: dict[tuple, Callable] = {}


def supported(o: torch.Tensor, g: torch.Tensor) -> bool:
    return (
        o.dtype == torch.bfloat16
        and g.dtype == torch.bfloat16
        and o.shape == g.shape
        and o.is_cuda
        and o.is_contiguous()
        and g.is_contiguous()
        and o.dim() == 2
        and o.numel() % TILE_COLS == 0
        and o.data_ptr() % 32 == 0
        and g.data_ptr() % 32 == 0
    )


def mla_output_gate_tma(
    o: torch.Tensor,
    g: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    fast: bool = True,
    pdl: bool = False,
    fgs_consumer: bool = False,
    fgs_max_spins: int = 1 << 16,
    fgs_stats: Optional[torch.Tensor] = None,
    fgs_counters: Optional[torch.Tensor] = None,
    fgs_counter_rows: int = 0,
) -> torch.Tensor:
    """out = o * sigmoid(g) for 2-D bf16 o / g (see the module docstring); ``out`` must not alias o or g."""
    if not supported(o, g):
        raise ValueError("mla_output_gate_tma needs contiguous 32B-aligned 2-D bf16 o / g, numel % 256 == 0")
    if out is None:
        out = torch.empty_like(o)
    elif out.shape != o.shape or out.dtype != o.dtype or not out.is_contiguous() or out.data_ptr() % 32:
        raise ValueError("out must be a contiguous, 32B-aligned bf16 tensor shaped like o")
    T, N = o.shape
    rows = T * N // TILE_COLS
    publish = fgs_counters is not None
    cnt_block = 1
    if publish:
        if fgs_counter_rows <= 0 or (fgs_counter_rows * N) % TILE_ELEMS:
            raise ValueError("fgs_counter_rows * N must be a multiple of the 8192-element tile")
        if fgs_counters.dtype != torch.int32 or not fgs_counters.is_contiguous():
            raise ValueError("fgs_counters must be a contiguous int32 tensor")
        if fgs_counters.numel() < -(-T // fgs_counter_rows):
            raise ValueError("fgs_counters needs a word per row block")
        cnt_block = fgs_counter_rows * N
    with_stats = fgs_consumer and fgs_stats is not None
    if with_stats and (fgs_stats.dtype != torch.int32 or fgs_stats.numel() < STATS_LEN):
        raise ValueError(f"fgs_stats must be an int32 tensor with at least {STATS_LEN} entries")
    if rows == 0:
        return out
    max_spins = fgs_max_spins if fgs_consumer else 0
    key = (fast, pdl, fgs_consumer, max_spins, with_stats, publish)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def arg2d(t):
        return from_dlpack(t.view(rows, TILE_COLS), assumed_align=32).mark_layout_dynamic(leading_dim=1)

    def arg1d(t):
        return from_dlpack(t.view(-1), assumed_align=32).mark_layout_dynamic()

    def iptr(t):
        return make_ptr(cutlass.Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)

    args = (
        arg2d(o),
        arg2d(g),
        arg1d(out),
        iptr(fgs_stats if with_stats else out),
        iptr(fgs_counters if publish else out),
        cutlass.Int64(cnt_block),
        stream,
    )
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_Gate(fast, pdl, fgs_consumer, max_spins, with_stats, publish), *args)
        _COMPILED[key] = fn
    fn(*args)
    return out
