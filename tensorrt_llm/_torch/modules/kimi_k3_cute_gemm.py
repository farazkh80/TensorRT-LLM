# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL BF16 GEMM / BMM dispatch for the Kimi K3 attention-core projections on SM107.

Each projection shape (N, K) maps to a list of (max_m, backend, tactic) entries measured at the Kimi K3 workload
points; the first entry whose max_m covers the runtime M is used. Backends:

- ``rubin``: the in-tree SM107 persistent GEMM (``CuteDSLBf16RubinGemmRunner``), or the batched runner when an operand
  is a strided view (the GEMM runner would copy it).
- ``fi_direct`` / ``fi_splitk``: FlashInfer's low-M CuTe DSL GEMMs (M <= 32 per launch; split-K runs M <= 64 as two
  launches). The in-tree split-K reduces bf16 partials in the output and loses precision, so no table entry uses it.

A shape without an entry runs the ``rubin`` backend under the TRT-LLM autotuner. Non-BF16 operands keep the
cuBLAS matmul; empty problems launch nothing.
"""

import functools
import os
from typing import Dict, List, Optional, Tuple

import torch

_ENV = "TLLM_K3_CUTE_GEMM"
_FI_MAX_M = 32

# Measured on VR200 (212 SMs): decode M <= 64 as CUDA-graph replays with PDL, prefill M = 8192 as idle-gapped single
# launches (back-to-back multi-ms GEMMs throttle the clock and mislead the autotuner). (N, K) -> [(max_m, backend,
# tactic)], ascending max_m.
_GEMM_TABLE: Dict[Tuple[int, int], List[Tuple[int, str, Optional[tuple]]]] = {
    (224, 7168): [
        (1, "fi_direct", (128, 1, 1)),
        (2, "fi_direct", (128, 1, 2)),
        (4, "fi_direct", (128, 1, 4)),
        (8, "fi_splitk", (64, 8, 4, 12)),
        (16, "fi_splitk", (64, 8, 4, 11)),
        (32, "fi_splitk", (64, 8, 4, 11)),
        (64, "fi_splitk", (64, 8, 4, 11)),
        (1 << 30, "rubin", ("base", True, (256, 128), (4, 2), 0, 1)),
    ],
    (2112, 7168): [
        (1, "fi_splitk", (64, 8, 4, 11)),
        (2, "fi_splitk", (64, 8, 4, 12)),
        (4, "fi_splitk", (64, 8, 4, 11)),
        (8, "fi_splitk", (64, 8, 4, 10)),
        (16, "fi_splitk", (64, 16, 4, 10)),
        (32, "fi_splitk", (64, 32, 4, 8)),
        (64, "rubin", ("base", False, (64, 32), (1, 1), 0, 1)),
        (1 << 30, "rubin", ("base", True, (256, 256), (2, 1), 0, 1)),
    ],
    (7168, 12288): [
        (1, "fi_direct", (256, 2, 1)),
        (2, "fi_splitk", (128, 8, 3, 6)),
        (4, "fi_splitk", (128, 8, 3, 6)),
        (8, "fi_splitk", (128, 8, 3, 6)),
        (16, "fi_splitk", (64, 16, 3, 10)),
        (32, "rubin", ("base", False, (64, 64), (1, 2), 0, 1)),
        (64, "rubin", ("base", False, (64, 64), (1, 2), 0, 1)),
        (1 << 30, "rubin", ("base", True, (256, 256), (2, 1), 0, 1)),
    ],
    (8192, 7168): [
        (1 << 30, "rubin", ("base", True, (256, 256), (2, 4), 0, 1)),
    ],
    (12288, 128): [
        (1, "fi_splitk", (128, 8, 1, 5)),
        (2, "fi_splitk", (128, 8, 1, 4)),
        (4, "fi_splitk", (128, 8, 1, 6)),
        (8, "fi_splitk", (128, 8, 1, 6)),
        (16, "fi_splitk", (128, 16, 1, 3)),
        (32, "fi_splitk", (128, 16, 1, 1)),
        (64, "rubin", ("base", False, (64, 64), (1, 1), 0, 1)),
        (1 << 30, "rubin", ("base", True, (256, 256), (2, 1), 0, 1)),
    ],
    (12288, 512): [
        (1 << 30, "rubin", ("base", True, (256, 128), (4, 1), 0, 1)),
    ],
    (12288, 7168): [
        (1, "fi_direct", (128, 2, 1)),
        (2, "fi_splitk", (128, 8, 2, 6)),
        (4, "fi_splitk", (128, 8, 2, 6)),
        (8, "fi_splitk", (64, 8, 1, 11)),
        (16, "fi_splitk", (64, 16, 1, 9)),
        (32, "fi_splitk", (64, 32, 1, 9)),
        (64, "rubin", ("base", False, (64, 64), (1, 1), 0, 1)),
        (1 << 30, "rubin", ("base", True, (256, 256), (2, 1), 0, 1)),
    ],
    (18432, 1536): [
        (1, "fi_splitk", (128, 8, 1, 6)),
        (2, "fi_splitk", (128, 8, 1, 6)),
        (4, "fi_splitk", (128, 8, 1, 6)),
        (8, "fi_splitk", (128, 8, 1, 6)),
        (16, "fi_splitk", (128, 16, 1, 6)),
        (32, "fi_splitk", (128, 32, 1, 5)),
        (64, "rubin", ("base", False, (64, 128), (1, 1), 0, 1)),
        (1 << 30, "rubin", ("base", True, (256, 256), (2, 1), 0, 1)),
    ],
    (49152, 7168): [
        (1, "fi_direct", (128, 1, 1)),
        (2, "fi_direct", (64, 4, 2)),
        (4, "fi_splitk", (128, 16, 1, 6)),
        (8, "fi_splitk", (128, 16, 1, 6)),
        (16, "rubin", ("base", False, (128, 128), (1, 1), 0, 1)),
        (32, "rubin", ("base", False, (128, 128), (1, 1), 0, 1)),
        (64, "rubin", ("base", False, (128, 128), (1, 1), 0, 1)),
        (1 << 30, "rubin", ("base", True, (256, 256), (2, 1), 0, 1)),
    ],
}
# (B, N, K) -> [(max_m, backend, tactic)].
_BMM_TABLE: Dict[Tuple[int, int, int], List[Tuple[int, str, Optional[tuple]]]] = {
    (96, 128, 512): [
        (1, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (2, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (4, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (8, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (16, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (32, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (64, "rubin", ("base", False, (64, 128), (1, 1), 0)),
    ],
    (96, 512, 128): [
        (1, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (2, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (4, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (8, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (16, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (32, "rubin", ("base", False, (64, 128), (1, 1), 0)),
        (64, "rubin", ("base", False, (64, 128), (1, 1), 0)),
    ],
}


@functools.lru_cache(maxsize=None)
def enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


@functools.lru_cache(maxsize=None)
def _pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


@functools.lru_cache(maxsize=None)
def _gemm_runner(out_dtype: torch.dtype):
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLBf16RubinGemmRunner

    return CuteDSLBf16RubinGemmRunner(output_dtype=out_dtype)


@functools.lru_cache(maxsize=None)
def _bmm_runner():
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLBf16RubinBmmRunner

    return CuteDSLBf16RubinBmmRunner()


def _usable(backend: str, tactic: Optional[tuple], m: int) -> bool:
    if backend == "fi_direct":
        return m <= _FI_MAX_M and m % tactic[2] == 0
    if backend == "fi_splitk":
        return m <= 2 * _FI_MAX_M
    return True


def _select(entries, m: int):
    for max_m, backend, tactic in entries or ():
        if m <= max_m and _usable(backend, tactic, m):
            return backend, tactic
    return "rubin", None


def _row_major(t: torch.Tensor) -> bool:
    return t.stride(-1) == 1 and t.stride(0) == t.shape[-1]


def linear(
    x: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``out[M, N] = x[M, K] @ weight[N, K]^T`` in BF16 with a CuTe DSL kernel.

    ``x`` and ``out`` may be row-strided views (K / N innermost); ``weight`` must be K-major.
    """
    m, k = x.shape
    n = weight.shape[0]
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return torch.matmul(x, weight.t(), out=out)
    # DLPack export rejects tensors that require grad (nn.Linear parameters); detach() is a view.
    x, weight = x.detach(), weight.detach()
    if out is None:
        out = x.new_empty(m, n)
    if m == 0 or n == 0:
        return out
    backend, tactic = _select(_GEMM_TABLE.get((n, k)), m)
    dense = _row_major(x) and _row_major(out)
    if backend == "fi_direct" and dense:
        from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic, run_direct_dense

        return run_direct_dense(x, weight.t(), out, _pdl(), DirectTactic(*tactic))
    if backend == "fi_splitk" and dense:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
            SplitKTactic,
            run_splitk_dense,
        )

        # Rows beyond the kernel's M limit run as a second launch on the next row block.
        for r in range(0, m, _FI_MAX_M):
            run_splitk_dense(
                x[r : r + _FI_MAX_M],
                weight.t(),
                None,
                out[r : r + _FI_MAX_M],
                _pdl(),
                SplitKTactic(*tactic),
            )
        return out
    if not dense:
        # The batched runner takes strided A / C views directly but has no split-K; its tactics are the 5-field
        # base form.
        bmm_tactic = None
        if backend == "rubin" and tactic is not None and tactic[0] == "base" and tactic[5] == 1:
            bmm_tactic = tactic[:5]
        _run_bmm(x.unsqueeze(0), weight.unsqueeze(0), out.unsqueeze(0), bmm_tactic)
        return out
    if tactic is None:
        torch.ops.trtllm.cute_dsl_bf16_gemm_rubin(x, weight, out)
    else:
        _gemm_runner(out.dtype)([x, weight, out], tactic=tactic)
    return out


def bmm(x: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``out[B, M, N] = x[B, M, K] @ weight[B, N, K]^T`` in BF16; any operand may be a strided view with K / N
    innermost."""
    b, m, k = x.shape
    n = weight.shape[1]
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return torch.bmm(x, weight.transpose(1, 2), out=out)
    x, weight = x.detach(), weight.detach()
    if out is None:
        out = x.new_empty(b, m, n)
    if b == 0 or m == 0 or n == 0:
        return out
    _, tactic = _select(_BMM_TABLE.get((b, n, k)), m)
    _run_bmm(x, weight, out, tactic)
    return out


def _run_bmm(x, weight, out, tactic):
    if tactic is None:
        torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(x, weight, out)
    else:
        _bmm_runner()([x, weight, out], tactic=tactic)
