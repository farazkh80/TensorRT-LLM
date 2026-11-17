# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 prefill latent up-projection with the residual adds folded into the GEMM epilogue:

    out = bf16(f32(prefix) + f32(bf16(f32(bf16(latent @ W_up^T)) + f32(shared))))

latent [T, 3584] (the routed experts' combined and normalized output), W_up [7168, 3584], shared [T, 7168] (the
shared expert), prefix [T, 7168] (the layer's running residual; optional). Stock: cuBLAS GEMM, ATen add, ATen add.
The folded adds keep the stock rounding sequence, so for equal GEMM values the output is bitwise the stock one.
"""

from typing import Optional

import torch

from .... import kimi_k3_cute_gemm
from ...dense_gemm_fgs import run_gemm_fgs


def latent_up(
    latent: torch.Tensor,
    w_up: torch.Tensor,
    shared: torch.Tensor,
    prefix: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    m, k = latent.shape
    n = w_up.shape[0]
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=latent.device)
    _, tactic = kimi_k3_cute_gemm._select(kimi_k3_cute_gemm._GEMM_TABLE.get((n, k)), m)
    run_gemm_fgs(
        latent.detach().unsqueeze(0),
        w_up.detach().unsqueeze(0),
        out.unsqueeze(0),
        tactic or kimi_k3_cute_gemm._FGS_LARGE_TACTIC,
        report_valid=False,
        retry_warp=False,
        wait_input=True,
        use_pdl=kimi_k3_cute_gemm._pdl(),
        fold=(shared.unsqueeze(0), None if prefix is None else prefix.unsqueeze(0)),
    )
    return out


def reference(latent: torch.Tensor, w_up: torch.Tensor, shared: torch.Tensor, prefix: Optional[torch.Tensor]):
    """Stock path: cuBLAS GEMM + ATen adds."""
    y = torch.matmul(latent, w_up.t()) + shared
    return y if prefix is None else prefix + y
