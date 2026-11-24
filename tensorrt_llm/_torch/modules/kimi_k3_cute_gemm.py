# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL BF16 GEMM / BMM dispatch for the Kimi K3 attention-core projections on SM107.

Each projection shape (N, K) maps to a list of (max_m, backend, tactic) entries measured at the Kimi K3 workload
points; the first entry whose max_m covers the runtime M is used. Backends:

- ``rubin``: the in-tree SM107 persistent GEMM (``CuteDSLBf16RubinGemmRunner``), or the batched runner when an operand
  is a strided view (the GEMM runner would copy it).
- ``fi_direct`` / ``fi_splitk``: FlashInfer's low-M CuTe DSL GEMMs (M <= 32 per launch; split-K runs M <= 64 as two
  launches). The in-tree split-K reduces bf16 partials in the output and loses precision, so no table entry uses it.
- ``vsplitk``: the FlashInfer split-K kernel vendored in ``kimi_k3_fgs``, requesting only its tactic's shared memory
  so two CTAs share an SM (bit-exact with ``fi_splitk`` at the same tile and split). Entries carry
  ``(vendored tactic, fi_splitk tactic)``; a CuTe DSL older than 4.9 runs the second.

An operand that is a view of a live edge slot (``kimi_k3_fgs.edges``) selects the fine-grained-sync build of the
table entry's kernel at its tactic: an input slot the report_valid consumer (no ``griddepcontrol.wait``; the slot is
consumed), an output slot the producer roles (.pzo, trigger at entry, tile stored by TMA). ``rubin`` runs
``kimi_k3_fgs.dense_gemm_fgs``, split-K entries ``kimi_k3_fgs.splitk_gemm_fgs``, ``fi_direct`` rows
``kimi_k3_fgs.direct_gemm_fgs`` at the table tactic (dense, 32 B aligned operands, N % 8 == 0 for an output slot),
else FlashInfer's default split-K tactic; a strided input at split-K rows runs the ``rubin`` build at
``_FGS_SMALL_TACTIC``. The direct consumer checks element 0 of each 16 B chunk in software (the TMA report_valid
rule, sound because every producer of a direct GEMM's input slot stores whole chunks); a direct producer stores whole
16 B chunks (``_DIRECT_AUTO_STORE``).
The ``rubin`` producer build of the KDA qkvg projection computes the q / k / v tiles of heads 0 .. h1 - 1 first
(``dense_gemm_fgs`` ``n_order``; same tiles, bit-exact): h1 = 70 at 9..16 rows (the rows the KDA core's late-gate,
head-major producer order runs at).

A shape without an entry runs the ``rubin`` backend under the TRT-LLM autotuner. Non-BF16 operands keep the
cuBLAS matmul; empty problems launch nothing.
"""

import functools
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

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
        (1 << 30, "rubin", ("base", True, (256, 128), (2, 2), 0, 1)),
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
        (2, "vsplitk", ((128, 8, 3, 5), (128, 8, 3, 6))),
        (4, "vsplitk", ((128, 8, 3, 5), (128, 8, 3, 6))),
        (8, "vsplitk", ((128, 8, 3, 5), (128, 8, 3, 6))),
        (16, "vsplitk", ((64, 16, 3, 5), (64, 16, 3, 10))),
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
        (1 << 30, "rubin", ("base", True, (256, 256), (4, 1), 0, 1)),
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
        (1 << 30, "rubin", ("base", True, (256, 128), (2, 1), 0)),
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
# (B, N, K) -> persistent-grid cap (CTAs) of the fine-grained-sync BMM. The MLA decode W_UK / W_UV launches make the
# decode step faster at 132 CTAs than at one per SM; each output tile is still computed by one CTA (bit-exact).
_FGS_BMM_MAX_CTAS: Dict[Tuple[int, int, int], int] = {
    (96, 128, 512): 132,
    (96, 512, 128): 132,
}


@functools.lru_cache(maxsize=None)
def _pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


@functools.lru_cache(maxsize=None)
def fgs_dsl() -> bool:
    """Whether the CuTe DSL builds the vendored split-K kernel (v1 mbarriers, launch carveout: 4.9)."""
    import cutlass.cute as cute

    return hasattr(cute.arch, "mbarrier_v1_init")


# Chunk-safe store of the direct producer by output width N (ready input): cluster gather on the single-wave bfa grid,
# group8 on the multi-wave ones.
_DIRECT_AUTO_STORE = {224: "cluster"}


class _Roles(NamedTuple):
    """Sync roles of one launch: the consumer check (TMA report_valid + retry warp, or the direct GEMM's software
    check), the grid wait before the activation loads, the producer build (.pzo, whole-chunk stores) and the
    trigger at kernel entry."""

    check: bool
    wait: bool
    arm: bool
    early: bool


def _roles(consume: bool, produce: bool) -> _Roles:
    return _Roles(check=consume, wait=not consume, arm=produce, early=produce)


# KDA qkvg projection (N, K) and its heads (q | k | v | gate sections of one-tile heads).
_QKVG = (49152, 7168)
_QKVG_HEADS = 96


# Head-ordered qkvg tiles: (min rows, max rows, h1).
_QKVG_ORDER = (9, 16, 70)


def _qkvg_order(m: int) -> int:
    low, high, h1 = _QKVG_ORDER
    return h1 if low <= m <= high else 0


# Preferred smem carveout (percent) of the vendored split-K launches.
_VSPLITK_CARVEOUT = 100
# Report_valid consumer tactic per launch rows (<= 8, <= 16, <= 32) and whether it requests only its footprint.
_FGS_TACTICS = (
    (8, (128, 8, 3, 5), True),
    (16, (64, 16, 3, 5), True),
    (32, (64, 32, 3, 8), False),
)
# Fine-grained-sync ``rubin`` builds where the table has no ``rubin`` tactic: decode rows (also strided inputs at
# split-K rows, which the stock path autotunes) and prefill rows.
_FGS_SMALL_TACTIC = ("base", False, (64, 128), (1, 1), 0, 1)
_FGS_LARGE_TACTIC = ("base", True, (256, 256), (2, 1), 0, 1)
# (N, K) -> [(max_m, split-K tactic)], ascending max_m: the fine-grained-sync build's tactic in place of the table's
# ``fi_splitk`` tactic (MLA decode g_proj). Applied only over a split-K 1 table tactic: a different split changes the
# reduction order.
_FGS_SPLITK_TACTICS: Dict[Tuple[int, int], List[Tuple[int, tuple]]] = {
    (12288, 7168): [(16, (128, 16, 1, 4)), (32, (128, 32, 1, 4))],
}


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
    if backend in ("fi_splitk", "vsplitk"):
        return m <= 2 * _FI_MAX_M
    return True


def _select(entries, m: int):
    for max_m, backend, tactic in entries or ():
        if m <= max_m and _usable(backend, tactic, m):
            return backend, tactic
    return "rubin", None


def _row_major(t: torch.Tensor) -> bool:
    return t.stride(-1) == 1 and t.stride(0) == t.shape[-1]


def _fgs_consume(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, roles: _Roles) -> None:
    """o_proj on a live edge slot: report_valid loads, retry warp, no griddepcontrol.wait."""
    from .kimi_k3_fgs.splitk_gemm_fgs import SplitKTactic, run_splitk_dense_fgs

    for r in range(0, x.shape[0], _FI_MAX_M):
        rows = min(_FI_MAX_M, x.shape[0] - r)
        _, tactic, exact = next(t for t in _FGS_TACTICS if rows <= t[0])
        run_splitk_dense_fgs(
            x[r : r + rows],
            weight.t(),
            None,
            out[r : r + rows],
            _pdl(),
            SplitKTactic(*tactic),
            report_valid=roles.check,
            retry_warp=roles.check,
            pdl_wait_activation=roles.wait,
            exact_smem=exact,
            smem_carveout=_VSPLITK_CARVEOUT if exact else None,
        )


def _fgs_leases(*tensors):
    """The live edge lease each tensor is a view of (None each when no edge is live)."""
    from .kimi_k3_fgs import edges as fgs_edges

    return tuple(fgs_edges.lease_containing(t) for t in tensors)


def _fgs_rubin(
    x3,
    w3,
    out3,
    tactic,
    roles: _Roles,
    max_ctas: Optional[int] = None,
    n_order: Optional[Tuple[int, int]] = None,
) -> None:
    from .kimi_k3_fgs.dense_gemm_fgs import run_gemm_fgs

    if tactic is None:
        tactic = _FGS_SMALL_TACTIC if x3.shape[1] <= 64 else _FGS_LARGE_TACTIC
    run_gemm_fgs(
        x3,
        w3,
        out3,
        tactic,
        report_valid=roles.check,
        retry_warp=roles.check,
        wait_input=roles.wait,
        pzo=roles.arm,
        early_trigger=roles.early,
        use_pdl=_pdl(),
        max_ctas=max_ctas,
        n_order=n_order,
    )


def _fgs_direct(x, weight, out, tactic, roles: _Roles, produce: str) -> None:
    """``produce``: "" (no output slot), "group8" or "cluster"."""
    from .kimi_k3_fgs.direct_gemm_fgs import DirectTactic, run_direct_dense_fgs

    run_direct_dense_fgs(
        x,
        weight.t(),
        out,
        _pdl(),
        DirectTactic(*tactic),
        check_valid=roles.check,
        pdl_wait_activation=roles.wait,
        pzo=bool(produce),
        early_trigger=roles.early,
        group8=produce == "group8",
        gather=produce == "cluster",
    )


def _fgs_linear(x, weight, out, consume: bool, produce: bool) -> None:
    """``linear`` on edge-slot operands: the fine-grained-sync build of the table entry's kernel."""
    from .kimi_k3_fgs.splitk_gemm_fgs import SplitKTactic, default_tactic, run_splitk_dense_fgs

    m, k = x.shape
    n = weight.shape[0]
    dense = _row_major(x) and _row_major(out)
    roles = _roles(consume, produce)
    backend, tactic = _select(_GEMM_TABLE.get((n, k)), m)
    if backend == "fi_splitk" and tactic[2] == 1:
        fgs_tactic = next(
            (t for max_m, t in _FGS_SPLITK_TACTICS.get((n, k), ()) if m <= max_m), None
        )
        if fgs_tactic is not None:
            tactic = fgs_tactic
    direct_store = _DIRECT_AUTO_STORE.get(n, "group8") if roles.arm else ""
    if (
        backend == "fi_direct"
        and dense
        and x.data_ptr() % 32 == 0
        and out.data_ptr() % 32 == 0
        and (not roles.arm or n % 8 == 0)
    ):
        _fgs_direct(x, weight, out, tactic, roles, direct_store)
        return
    # The decode o_proj consumer at the table's split-K / direct rows; its rubin rows (17 and up) take the table's
    # GEMM below, since a split-K build would change their reduction order.
    if consume and not produce and dense and n == 7168 and k == 12288 and backend != "rubin":
        _fgs_consume(x, weight, out, roles)
        return
    if backend == "rubin" or not dense:
        if backend != "rubin":
            tactic = _FGS_SMALL_TACTIC
        n_order = None
        if roles.arm and backend == "rubin" and (n, k) == _QKVG:
            h1 = _qkvg_order(m)
            if h1 > 0:
                n_order = (_QKVG_HEADS, h1)
        _fgs_rubin(
            x.unsqueeze(0),
            weight.unsqueeze(0),
            out.unsqueeze(0),
            tactic,
            roles,
            n_order=n_order,
        )
        return
    exact = backend == "vsplitk"
    for r in range(0, m, _FI_MAX_M):
        rows = min(_FI_MAX_M, m - r)
        if backend == "fi_direct":
            launch_tactic = default_tactic(rows, n, k)
        else:
            launch_tactic = SplitKTactic(*(tactic[0] if exact else tactic))
        run_splitk_dense_fgs(
            x[r : r + rows],
            weight.t(),
            None,
            out[r : r + rows],
            _pdl(),
            launch_tactic,
            report_valid=roles.check,
            retry_warp=roles.check,
            pdl_wait_activation=roles.wait,
            exact_smem=exact,
            smem_carveout=_VSPLITK_CARVEOUT if exact else None,
            pzo=roles.arm,
            early_trigger=roles.early,
            tma_store=roles.arm,
        )


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
    dense = _row_major(x) and _row_major(out)
    in_lease, out_lease = _fgs_leases(x, out)
    if in_lease is not None or out_lease is not None:
        from .kimi_k3_fgs import edges as fgs_edges

        _fgs_linear(x, weight, out, in_lease is not None, out_lease is not None)
        if in_lease is not None:
            fgs_edges.consumed(in_lease)
        return out
    backend, tactic = _select(_GEMM_TABLE.get((n, k)), m)
    if backend == "vsplitk" and not fgs_dsl():
        backend, tactic = "fi_splitk", tactic[1]
    if backend == "fi_direct" and dense:
        from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic, run_direct_dense

        return run_direct_dense(x, weight.t(), out, _pdl(), DirectTactic(*tactic))
    if backend == "vsplitk" and dense:
        from .kimi_k3_fgs.splitk_gemm_fgs import SplitKTactic, run_splitk_dense_fgs

        for r in range(0, m, _FI_MAX_M):
            run_splitk_dense_fgs(
                x[r : r + _FI_MAX_M],
                weight.t(),
                None,
                out[r : r + _FI_MAX_M],
                _pdl(),
                SplitKTactic(*tactic[0]),
                report_valid=False,
                retry_warp=False,
                pdl_wait_activation=True,
                exact_smem=True,
                smem_carveout=_VSPLITK_CARVEOUT,
            )
        return out
    if backend == "vsplitk":
        backend, tactic = "fi_splitk", tactic[1]
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
    if (
        m > 64
        and x.stride(0) == 0
        and weight.is_contiguous()
        and out.transpose(0, 1).is_contiguous()
    ):
        # A broadcast over the batch into a [M, B, N] output (MLA prefill kv_b): one GEMM on the stacked weights
        # beats the batched kernel.
        linear(x[0], weight.view(b * n, k), out.transpose(0, 1).view(m, b * n))
        return out
    _, tactic = _select(_BMM_TABLE.get((b, n, k)), m)
    in_lease, out_lease = _fgs_leases(x, out)
    if in_lease is not None or out_lease is not None:
        from .kimi_k3_fgs import edges as fgs_edges

        roles = _roles(in_lease is not None, out_lease is not None)
        _fgs_rubin(x, weight, out, tactic, roles, _FGS_BMM_MAX_CTAS.get((b, n, k)))
        if in_lease is not None:
            fgs_edges.consumed(in_lease)
        return out
    _run_bmm(x, weight, out, tactic)
    return out


def _run_bmm(x, weight, out, tactic):
    if tactic is None:
        torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(x, weight, out)
    else:
        _bmm_runner()([x, weight, out], tactic=tactic)
