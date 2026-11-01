# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode KDA layer as one PDL chain of fine-grained-sync edges (``TLLM_K3_FGS=chain``).

Every kernel runs on the main stream with its original grid; each edge tensor is an :mod:`edges` slot, so each
kernel is a report_valid / Lamport consumer of its inputs and an early-triggering, -0.0-free producer of its output:

  pre-attention mix --x_norm (2 readers)--> [f_a | b] GEMM --bfa (2)--> f_b GEMM --g--> KdaDecodeT1
                                        \\-> qkvg GEMM --qkvg--> KdaDecodeT1 (q | k | v and the output gate)
  KdaDecodeT1 --core_o--> o_proj --attn_o--> pre-MoE mix (consumer only: the MoE input is a plain buffer)

[f_a | b] and f_b launch ahead of qkvg, or behind it with ``TLLM_K3_FGS=chain_after`` (or
``TLLM_K3_FGS_CHAIN_BFA=after``). Outputs that are not slots (the mixes' ``out`` / ``updated``) come from
:func:`buffer`: a consumer may run while its producers still read their inputs, so those outputs must not be blocks
the caching allocator could hand back.
"""

import os
from typing import Dict, Optional, Tuple

import torch

from .. import kimi_k3_cute_gemm
from . import edges

X_NORM = "kda.x_norm"
BFA = "kda.bfa"
GATE = "kda.g"
QKVG = "kda.qkvg"
CORE = "kda.core_o"
ATTN_O = "kda.attn_o"


def bfa_first() -> bool:
    value = os.environ.get("TLLM_K3_FGS_CHAIN_BFA") or ("after" if edges.mode() == "chain_after" else "before")
    if value not in ("before", "after"):
        raise ValueError(f"TLLM_K3_FGS_CHAIN_BFA={value!r}; valid: before, after")
    return value == "before"


def declare(layer_idx: int, hidden: int, bfa_width: int, proj_size: int, qkvg_width: int) -> None:
    """Declare the chain edges of one KDA layer (``CORE`` is declared with the ``lamport`` edges)."""
    if not edges.chain():
        return
    for name, width in (
        (X_NORM, hidden),
        (BFA, bfa_width),
        (GATE, proj_size),
        (QKVG, qkvg_width),
        (ATTN_O, hidden),
    ):
        edges.declare(name, width, layer_idx)


def acquire(name: str, rows: int, readers: int = 1):
    return edges.acquire(name, rows, readers=readers) if edges.chain() else None


def project(
    x2d: torch.Tensor,
    bfa_weight: torch.Tensor,
    f_b_weight: torch.Tensor,
    qkvg_weight: torch.Tensor,
    head_dim: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]]:
    """The decode projections into fresh slots on the current stream: ``(qkvg, bfa, g, leases)``, or None (and
    nothing launched) when the chain does not cover this batch. KdaDecodeT1 must ``consume`` the leases."""
    rows = x2d.shape[0]
    leases = []
    for name, readers in ((BFA, 2), (GATE, 1), (QKVG, 1)):
        lease = acquire(name, rows, readers)
        if lease is None:
            for held in leases:
                edges.release(held)
            return None
        leases.append(lease)
    bfa, g, qkvg = (lease.tensor for lease in leases)

    def gates():
        kimi_k3_cute_gemm.linear(x2d, bfa_weight, bfa)
        kimi_k3_cute_gemm.linear(bfa[:, :head_dim], f_b_weight, g)

    if bfa_first():
        gates()
        kimi_k3_cute_gemm.linear(x2d, qkvg_weight, qkvg)
    else:
        kimi_k3_cute_gemm.linear(x2d, qkvg_weight, qkvg)
        gates()
    return qkvg, bfa, g, tuple(leases)


def consume(leases) -> None:
    """One reader of each lease was launched on the current stream."""
    for lease in leases:
        edges.consumed(lease)


def o_proj(core2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``core2d @ weight^T`` into an ``ATTN_O`` slot when the chain covers the batch (consumed by the pre-MoE mix)."""
    lease = acquire(ATTN_O, core2d.shape[0])
    return kimi_k3_cute_gemm.linear(core2d, weight, None if lease is None else lease.tensor)


def buffer(owner, name: str, like: torch.Tensor) -> torch.Tensor:
    """A persistent tensor shaped like ``like``, owned by ``owner`` (one per name and shape)."""
    bufs: Dict[tuple, torch.Tensor] = owner.__dict__.setdefault("_k3_chain_bufs", {})
    key = (name, tuple(like.shape), like.dtype, like.device)
    if key not in bufs:
        bufs[key] = torch.empty_like(like, memory_format=torch.contiguous_format)
    return bufs[key]


def mix_counters(owner, device) -> torch.Tensor:
    """Persistent int32[4] counters of the pre-MoE mix consumer (attn_res_mix ``make_fgs_counters`` layout)."""
    bufs: Dict[tuple, torch.Tensor] = owner.__dict__.setdefault("_k3_chain_bufs", {})
    key = ("mix_counters", device)
    if key not in bufs:
        bufs[key] = torch.zeros(4, dtype=torch.int32, device=device)
    return bufs[key]
