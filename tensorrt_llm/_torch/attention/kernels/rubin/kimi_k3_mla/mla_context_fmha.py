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
"""Kimi K3 MLA context attention (qk 192 / v 128, causal, bf16, varlen) with the CuTe DSL FMHA.

q, k: [T, H, 192]; v, o: [T, H, 128]; cu_seqlens: int32 [B + 1] on device, shared by q and kv (no cached
prefix). The kernel is the Blackwell CuTe DSL FMHA in ``context_fmha`` (st.global epilogue) with SM107 tuning
applied through ``tuning``.
"""

import functools
import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32

try:
    from .context_fmha.fmha import BlackwellFusedMultiHeadAttentionForward
    from .context_fmha.helpers import fmha_helpers as fmha_utils
except ImportError:
    from context_fmha.fmha import BlackwellFusedMultiHeadAttentionForward
    from context_fmha.helpers import fmha_helpers as fmha_utils

HEAD_DIM_QK = 192
HEAD_DIM_V = 128

# SM107 (328 KiB smem): the st.global epilogue frees sO, so four K/V stages of a D=192 tile fit next to the
# two Q stages. The correction warps mostly wait on O, so registers move from them to the softmax warps, which
# release P to the MMA. rescale_threshold 0 rescales O whenever a row max grows: with the default 8, P is taken
# against a stale max, so the dominant P can be up to 2^8 and is rounded in bf16 instead of being exactly 1.
DEFAULT_TUNING = (
    ("kv_stage", 4),
    ("num_regs_softmax", 200),
    ("num_regs_correction", 80),
    ("num_regs_other", 32),
    ("rescale_threshold", 0.0),
)


@functools.cache
def _compile(num_heads: int, tuning: tuple):
    is_persistent = dict(tuning).get("persistent", False)
    fmha = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler=(128, 128),
        head_dim=(HEAD_DIM_QK, HEAD_DIM_V),
        is_persistent=is_persistent,
        mask_type=fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE,
        enable_ex2_emulation=False,
        enable_skip_correction=True,
        use_tma_store=False,
        tuning=dict(tuning),
    )
    sym_s_q, sym_s_k, sym_seq = cute.sym_int(), cute.sym_int(), cute.sym_int()
    bf16 = cutlass.BFloat16
    order = (4, 3, 2, 1, 0)
    q = make_fake_compact_tensor(
        bf16, (1, sym_s_q, num_heads, 1, HEAD_DIM_QK), stride_order=order, assumed_align=16
    )
    k = make_fake_compact_tensor(
        bf16, (1, sym_s_k, num_heads, 1, HEAD_DIM_QK), stride_order=order, assumed_align=16
    )
    v = make_fake_compact_tensor(
        bf16, (1, sym_s_k, num_heads, 1, HEAD_DIM_V), stride_order=order, assumed_align=16
    )
    o = make_fake_compact_tensor(
        bf16, (1, sym_s_q, num_heads, 1, HEAD_DIM_V), stride_order=order, assumed_align=32
    )
    cu_q = make_fake_compact_tensor(Int32, (sym_seq,), assumed_align=16)
    cu_k = make_fake_compact_tensor(Int32, (sym_seq,), assumed_align=16)
    return cute.compile(
        fmha,
        q,
        k,
        v,
        o,
        (1, 1, 1, 1, num_heads, num_heads, HEAD_DIM_QK, HEAD_DIM_V),
        cu_q,
        cu_k,
        None,
        None,
        1.0,
        1.0,
        1.0,
        None,
        None,
        Int32(0),
        None,
        None,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        False,
        options="--enable-tvm-ffi --opt-level 2",
    )


def supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor) -> bool:
    return (
        all(t.dtype == torch.bfloat16 and t.is_cuda and t.is_contiguous() for t in (q, k, v, o))
        and q.dim() == 3
        and q.shape[-1] == HEAD_DIM_QK
        and k.shape == q.shape
        and v.shape == (*q.shape[:2], HEAD_DIM_V)
        and o.shape == v.shape
        and o.data_ptr() % 32 == 0
        and all(t.data_ptr() % 16 == 0 for t in (q, k, v))
    )


def mla_context_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
    tuning: Optional[dict] = None,
) -> None:
    """Causal attention of every sequence in ``cu_seqlens`` onto itself; writes o in place."""
    if not supported(q, k, v, o):
        raise ValueError("mla_context_fmha needs contiguous bf16 q/k [T, H, 192], v/o [T, H, 128]")
    if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be a contiguous int32 tensor")
    total, num_heads, _ = q.shape
    key = tuple(sorted((tuning if tuning is not None else dict(DEFAULT_TUNING)).items()))
    kernel = _compile(num_heads, key)
    q5 = q.view(1, total, num_heads, 1, HEAD_DIM_QK)
    k5 = k.view(1, total, num_heads, 1, HEAD_DIM_QK)
    v5 = v.view(1, total, num_heads, 1, HEAD_DIM_V)
    o5 = o.view(1, total, num_heads, 1, HEAD_DIM_V)
    problem = (
        cu_seqlens.numel() - 1,
        max_seqlen,
        total,
        max_seqlen,
        num_heads,
        num_heads,
        HEAD_DIM_QK,
        HEAD_DIM_V,
    )
    kernel(
        q5,
        k5,
        v5,
        o5,
        problem,
        cu_seqlens,
        cu_seqlens,
        None,
        None,
        Float32(softmax_scale * math.log2(math.e)),
        Float32(softmax_scale),
        Float32(1.0),
        None,
        None,
        Int32(0),
        None,
        None,
        False,
    )
