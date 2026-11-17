# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 prefill router logits without the fp32 activation cast.

Stock (``KimiK3MoEGate.compute_logits``): ``F.linear(h.float(), W.float())`` under the image's TF32 override, i.e. a
full fp32 copy of the activations (179 µs per layer at 8192 tokens) and a cuBLAS tf32 GEMM. The gate weight comes
from a bf16 checkpoint, so its fp32 values are bf16-exact; bf16 activations are exact in tf32 too. Hence a bf16 x bf16
tcgen05 GEMM with an fp32 accumulator and fp32 output (the Step 0 SM107 persistent GEMM, ``kimi_k3_cute_gemm``'s
``rubin`` backend) forms the same exact products and differs from the stock logits only by the fp32 accumulation
order. The top-k that follows (``noaux_tc_op``) is unchanged, so its ids agree with stock except at near-ties, which
the test counts.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

# M tiles of 256 (2-CTA), N tiles of 128: 896 experts = 7 N tiles.
DEFAULT_TACTIC = ("base", True, (256, 128), (2, 1), 0, 1)
_bf16_weights: Dict[Tuple[int, int], torch.Tensor] = {}


def bf16_weight(weight: torch.Tensor) -> torch.Tensor:
    """The gate weight as bf16; an fp32 weight must hold bf16-exact values (checked once per tensor)."""
    if weight.dtype == torch.bfloat16:
        return weight
    key = (weight.data_ptr(), weight._version)
    w = _bf16_weights.get(key)
    if w is None:
        w = weight.detach().to(torch.bfloat16)
        if not torch.equal(w.to(torch.float32), weight.detach()):
            raise ValueError("router: the fp32 gate weight is not bf16-exact; the bf16 GEMM would change the logits")
        _bf16_weights.clear()
        _bf16_weights[key] = w
    return w


def logits(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    tactic: tuple = DEFAULT_TACTIC,
) -> torch.Tensor:
    """fp32 ``[M, E] = x[M, K] @ weight[E, K]^T`` for bf16 ``x`` and a bf16-exact ``weight`` (fp32 or bf16)."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLBf16RubinGemmRunner

    if x.dtype != torch.bfloat16 or x.dim() != 2:
        raise ValueError("router: x must be bf16 [M, K]")
    w = bf16_weight(weight)
    m, k = x.shape
    if out is None:
        out = torch.empty(m, w.shape[0], dtype=torch.float32, device=x.device)
    elif out.dtype != torch.float32 or tuple(out.shape) != (m, w.shape[0]) or not out.is_contiguous():
        raise ValueError(f"router: out must be contiguous fp32 [{m}, {w.shape[0]}]")
    if m == 0:
        return out
    CuteDSLBf16RubinGemmRunner(output_dtype=torch.float32)([x.contiguous(), w, out], tactic=tactic)
    return out
