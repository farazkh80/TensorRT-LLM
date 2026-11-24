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
"""TMA data-driven sync for the Kimi K3 MLA side kernels (norms, latent append, output gate, LSE reduction).

Consumer: the CTA's input tile is loaded by TMA into shared memory with ``report_valid`` (the -0.0 sentinel of the
dtype in element 0 of each 16-B chunk) on a v1 mbarrier. One warp arms the barrier, issues the loads and waits on the
primary phase with ``mbarrier.try_wait``; a completion that reports a sentinel chunk re-arms the barrier and re-issues
the same loads into the same smem. No thread touches the input in global memory. With ``check=False`` the same loads
run without ``report_valid`` (the data is complete: the consumer waited for its producer grid).

Producer: the output tile is written to shared memory in the global layout and leaves through one TMA store, so every
16-B chunk lands whole; the issuing warp waits for the bulk group before the smem is reused or the kernel exits.

A row tile of a 2-D [rows, cols] tensor is viewed as (rows, cols / box, box) with ``box`` the widest power of two
<= 256 dividing ``cols``, so one TMA instruction covers the whole tile (every box extent <= 256).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32
from cutlass.cute.nvgpu import cpasync

try:
    from .mla_fgs import trap
except ImportError:
    from mla_fgs import trap

BF16_PATTERN = cpasync.SentinelPattern.PER_16BYTES_8000
F32_PATTERN = cpasync.SentinelPattern.PER_16BYTES_80000000
SMEM_ALIGN = 128


def inner_box(cols: int) -> int:
    box = 256
    while cols % box:
        box //= 2
    return box


def tile_layout(rows: int, cols: int) -> cute.Layout:
    """Static smem layout of a dense (rows, cols) row tile: (rows, cols / box, box, 1 stage)."""
    box = inner_box(cols)
    return cute.make_layout((rows, cols // box, box, 1), stride=(cols, box, 1, rows * cols))


def tile_shape(rows: int, cols: int):
    box = inner_box(cols)
    return (rows, cols // box, box)


@cute.jit
def make_row_tile_atom(
    op: cutlass.Constexpr, mX: cute.Tensor, rows: cutlass.Constexpr, cols: cutlass.Constexpr
):
    """TMA atom and TMA tensor for (rows, cols) tiles of the 2-D [M, cols] tensor ``mX`` (unit last stride);
    ``op`` is a ``CopyBulkTensorTileG2SOp`` (load) or ``CopyBulkTensorTileS2GOp`` (store)."""
    box = inner_box(cols)
    g3 = cute.make_tensor(
        mX.iterator,
        cute.make_layout((mX.shape[0], cols // box, box), stride=(mX.stride[0], box, 1)),
    )
    atom, tma_tensor = cpasync.make_tiled_tma_atom(
        op, g3, tile_layout(rows, cols), tile_shape(rows, cols)
    )
    return atom, tma_tensor


@cute.jit
def partition_row_tiles(
    atom, tma_tensor: cute.Tensor, sX: cute.Tensor, rows: cutlass.Constexpr, cols: cutlass.Constexpr
):
    """(tS, tG): row tile ``i`` of the TMA tensor is ``tG[(None, i)]``, the smem stage is ``tS[(None, 0)]``."""
    gX = cute.local_tile(tma_tensor, tile_shape(rows, cols), (None, 0, 0))
    tS, tG = cpasync.tma_partition(
        atom, 0, cute.make_layout(1), cute.group_modes(sX, 0, 3), cute.group_modes(gX, 0, 3)
    )
    return tS, tG


@cute.jit
def init_barrier(bar: cute.Pointer, v1: cutlass.Constexpr):
    """One warp: initialize ``bar`` (v1 when it receives report_valid loads) and fence; the caller syncs the CTA."""
    with cute.arch.elect_one():
        if cutlass.const_expr(v1):
            cute.arch.mbarrier_v1_init(bar, 1)
        else:
            cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()


@cute.jit
def checked_load(
    bar: cute.Pointer,
    tx_bytes: Int32,
    issue: cutlass.Constexpr,
    check: cutlass.Constexpr,
    spin_limit: cutlass.Constexpr,
    phase: Int32,
):
    """One warp: arm ``bar`` for ``tx_bytes``, call ``issue()`` (the TMA loads onto ``bar``), wait for the primary
    phase ``phase``; while the completion reports a sentinel chunk, re-arm and re-issue. Returns (re-issues, the
    phase to wait for next). ``check=False``: one plain completion wait."""
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(bar, tx_bytes)
    issue()
    retries = Int32(0)
    pending = Boolean(True)
    if cutlass.const_expr(check):
        while pending:
            complete, invalid = cute.arch.mbarrier_try_wait_primary(bar, phase)
            if complete:
                phase = phase ^ 1
                if invalid:
                    retries = retries + 1
                    if cutlass.const_expr(spin_limit > 0):
                        if retries > Int32(spin_limit):
                            trap()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(bar, tx_bytes)
                    issue()
                else:
                    pending = Boolean(False)
    else:
        while pending:
            complete = cute.arch.mbarrier_try_wait(bar, phase)
            if complete:
                phase = phase ^ 1
                pending = Boolean(False)
    return retries, phase


@cute.jit
def store_tile(atom, tS: cute.Tensor, tG: cute.Tensor):
    """One warp, after every writer of the smem tile fenced (``fence_proxy("async.shared")``) and the CTA synced:
    TMA-store the tile and wait for the bulk group (the smem may be reused or freed afterwards)."""
    cute.copy(atom, tS, tG)
    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0)
