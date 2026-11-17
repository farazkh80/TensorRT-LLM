# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 prefill shared expert: ``h = situ(x @ Wg^T, x @ Wu^T)`` as one token-M GEMM with the SiTU-GLU epilogue,
then ``y = h @ Wd^T``.

x [T, 7168], W_gate_up [12288, 7168] = [Wg; Wu] (the module's fused ``gate_up_proj`` weight), h [T, 6144],
W_down [7168, 6144], y [T, 7168]. Stock runs cuBLAS -> Triton ``situ_and_mul`` -> cuBLAS.

The GEMM's epilogue sees one (128, 32) subtile of the 256-column CTA tile at a time, so the weight rows are
interleaved per 32 output columns (32 gate rows, then their 32 up rows); consecutive subtiles are (gate, up) of
the same columns and the epilogue stores ``situ`` of the pair into the half-width output. The math is the decode
kernel's ``_situ`` (the stock Triton instruction sequence on the bf16-rounded GEMM values), so for equal GEMM
values the output is bitwise the stock one; the GEMM itself differs from cuBLAS in its K-reduction order.
"""

import torch

from .... import kimi_k3_cute_gemm
from ...dense_gemm_fgs import run_gemm_fgs

HIDDEN = 7168
INTER = 6144
# Epilogue subtile columns of the rubin 2-CTA 256x256 tactic for a 16-bit output (4096 elements / 128 rows).
EPI_N = 32


def interleave_gate_up(w: torch.Tensor) -> torch.Tensor:
    """``[Wg; Wu]`` [2I, K] -> rows ordered (32 gate, 32 up) per 32 output columns. Done once per weight by the
    caller (a module keeps the result); not cached here, a freed weight's address can be reused."""
    rows, k = w.shape
    inter = rows // 2
    if rows % (2 * EPI_N):
        raise ValueError(f"gate_up weight rows {rows} must be a multiple of {2 * EPI_N}")
    wg = w[:inter].reshape(inter // EPI_N, EPI_N, k)
    wu = w[inter:].reshape(inter // EPI_N, EPI_N, k)
    return torch.stack((wg, wu), dim=1).reshape(rows, k).contiguous()


def gate_up(x: torch.Tensor, w_interleaved: torch.Tensor, cap: float, lin: float) -> torch.Tensor:
    """``situ(x @ Wg^T, x @ Wu^T)`` [T, I] from the interleaved weight (:func:`interleave_gate_up`)."""
    m, k = x.shape
    n = w_interleaved.shape[0]
    inter = n // 2
    cta_m, cta_n = kimi_k3_cute_gemm.rubin_tile(n, k, m)
    if cta_n % (2 * EPI_N) or 4096 // cta_m != EPI_N:
        raise ValueError(f"shared gate_up: epilogue tile {cta_m}x{EPI_N} does not pair the interleaved rows")
    # One spare row lets a [T, 2I]-shaped view over the half-width buffer stand in for the GEMM's unused C.
    h = torch.empty(m + 1, inter, dtype=torch.bfloat16, device=x.device)
    c_virtual = h.as_strided((1, m, n), (m * inter, inter, 1))
    _, tactic = kimi_k3_cute_gemm._select(kimi_k3_cute_gemm._GEMM_TABLE.get((n, k)), m)
    run_gemm_fgs(
        x.detach().unsqueeze(0),
        w_interleaved.unsqueeze(0),
        c_virtual,
        tactic or kimi_k3_cute_gemm._FGS_LARGE_TACTIC,
        report_valid=False,
        retry_warp=False,
        wait_input=True,
        use_pdl=kimi_k3_cute_gemm._pdl(),
        glu=(h[:m].unsqueeze(0), cap, lin),
    )
    return h[:m]


def down(h: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """``h @ Wd^T`` on the rubin 2-CTA 256x256 build; (7168, 6144) has no table entry."""
    return kimi_k3_cute_gemm.linear_fgs(
        h, w_down, torch.empty(h.shape[0], w_down.shape[0], dtype=torch.bfloat16, device=h.device)
    )


def forward(x: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor, cap: float, lin: float) -> torch.Tensor:
    return down(gate_up(x, interleave_gate_up(w_gate_up), cap, lin), w_down)


def reference(x: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor, cap: float, lin: float):
    """Stock path: cuBLAS gate_up, Triton ``situ_and_mul``, cuBLAS down; returns (h, y)."""
    from .... import situ  # noqa: F401  (registers trtllm::situ_and_mul)

    gu = torch.matmul(x, w_gate_up.t())
    h = torch.ops.trtllm.situ_and_mul(gu, cap, lin)
    return h, torch.matmul(h, w_down.t())
