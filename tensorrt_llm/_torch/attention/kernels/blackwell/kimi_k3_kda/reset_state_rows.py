# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero the recurrent-state pool rows of fresh prefill requests (CuTe DSL).

Same contract as ``reset_recurrent_state_rows(pool, slots, has_init)`` without a
convolution pool: row ``slots[i]`` is zeroed when ``has_init[i]`` is false; slots
outside ``[0, pool rows)`` are ignored. Grid (row chunks, requests); a CTA whose
request keeps its state exits after reading two scalars.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

THREADS = 256
VECS_PER_THREAD = 4

_TORCH_TO_CUTE = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16}


class ResetStateRows:
    def __init__(self, row_elems: int, dtype):
        self.vec = 128 // dtype.width
        assert row_elems % self.vec == 0
        self.row_elems = row_elems
        self.dtype = dtype
        self.elems_per_cta = THREADS * VECS_PER_THREAD * self.vec

    @cute.jit
    def __call__(
        self, pool: cute.Tensor, slots: cute.Tensor, has_init: cute.Tensor, stream: cuda.CUstream
    ):
        chunks = (self.row_elems + self.elems_per_cta - 1) // self.elems_per_cta
        self.kernel(pool, slots, has_init).launch(
            grid=[chunks, slots.shape[0], 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, pool: cute.Tensor, slots: cute.Tensor, has_init: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        chunk, req, _ = cute.arch.block_idx()
        slot = slots[req]
        if not has_init[req]:
            if slot >= 0:
                if slot < pool.shape[0]:
                    zero = cute.make_rmem_tensor((self.vec,), self.dtype)
                    zero.fill(0.0)
                    store = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128
                    )
                    row = cute.zipped_divide(pool[slot, None], (self.vec,))
                    base = chunk * (THREADS * VECS_PER_THREAD) + tidx
                    for u in cutlass.range_constexpr(VECS_PER_THREAD):
                        v = base + u * THREADS
                        if v * self.vec < self.row_elems:
                            cute.copy(store, zero, row[(None, v)])


@cache
def _compile(row_elems: int, dtype: torch.dtype):
    cdtype = _TORCH_TO_CUTE[dtype]
    vec = 128 // cdtype.width
    pool = make_fake_tensor(
        cdtype,
        (cute.sym_int(), row_elems),
        stride=(cute.sym_int64(divisibility=vec), 1),
        assumed_align=16,
    )
    n = cute.sym_int()
    slots = make_fake_tensor(cutlass.Int32, (n,), stride=(1,), assumed_align=4)
    has_init = make_fake_tensor(cutlass.Boolean, (n,), stride=(1,), assumed_align=1)
    return cute.compile(
        ResetStateRows(row_elems, cdtype),
        pool,
        slots,
        has_init,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def supports(pool: torch.Tensor, slots: torch.Tensor, has_init: torch.Tensor) -> bool:
    """Whether ``reset_state_rows`` accepts these tensors (otherwise it raises)."""
    if (
        pool.dtype not in _TORCH_TO_CUTE
        or pool.ndim < 2
        or pool.shape[0] == 0
        or not pool[0].is_contiguous()
    ):
        return False
    vec = 16 // pool.element_size()
    row_elems = pool[0].numel()
    return (
        row_elems % vec == 0
        and pool.stride(0) % vec == 0
        and pool.stride(0) >= row_elems
        and pool.data_ptr() % 16 == 0
        and slots.dtype == torch.int32
        and slots.ndim == 1
        and slots.is_contiguous()
        and has_init.dtype == torch.bool
        and has_init.shape == slots.shape
        and has_init.is_contiguous()
    )


def reset_state_rows(pool: torch.Tensor, slots: torch.Tensor, has_init: torch.Tensor) -> None:
    """Zero ``pool[slots[i]]`` for every request ``i`` with ``has_init[i] == False``."""
    if not supports(pool, slots, has_init):
        raise ValueError(
            "reset_state_rows needs a 16-byte aligned FP32/BF16 pool with contiguous rows"
        )
    if slots.numel() == 0:
        return
    rows = pool.as_strided((pool.shape[0], pool[0].numel()), (pool.stride(0), 1))
    _compile(rows.shape[1], pool.dtype)(rows, slots, has_init)
