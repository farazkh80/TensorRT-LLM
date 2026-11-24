# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode KDA layer as one PDL chain of fine-grained-sync edges.

Every kernel runs on the main stream with its original grid; each edge tensor is an :mod:`edges` slot, so each
kernel is a report_valid / Lamport consumer of its inputs and an early-triggering, -0.0-free producer of its output:

  pre-attention mix --x_norm (2 readers)--> qkvg GEMM --qkvg--> KdaDecodeT1 (q | k | v and the output gate)
                                        \\-> [f_a | b] GEMM --bfa (2)--> f_b GEMM --g--> KdaDecodeT1
  KdaDecodeT1 --core_o--> o_proj --attn_o--> pre-MoE mix (consumer only: the MoE input is a plain buffer)

[f_a | b] and f_b launch behind qkvg. Outputs that are not slots (the mixes' ``out`` / ``updated``) come from
:func:`buffer`: a consumer may run while its producers still read their inputs, so those outputs must not be blocks
the caching allocator could hand back.

:func:`core_kwargs` runs the decode KdaDecodeT1 producer of ``CORE`` at 9..16 rows as one resident wave of
``_CORE_CPS`` CTAs per SM, with the chained core waiting for the output gate only in its epilogue and taking first
the heads qkvg publishes first. :func:`core_sync_kwargs` adds the chained core's sync form: TMA ``report_valid``
loads in the consumer and a TMA store of the core output per item in the producer.
"""

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


def declare(layer_idx: int, hidden: int, bfa_width: int, proj_size: int, qkvg_width: int) -> None:
    """Declare the chain edges of one KDA layer (``CORE`` is declared with the o_proj-input edges)."""
    if not edges.chain() or edges.MIN_ROWS > edges.MAX_ROWS:
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
    kimi_k3_cute_gemm.linear(x2d, qkvg_weight, qkvg)
    kimi_k3_cute_gemm.linear(x2d, bfa_weight, bfa)
    kimi_k3_cute_gemm.linear(bfa[:, :head_dim], f_b_weight, g)
    return qkvg, bfa, g, tuple(leases)


# The chain's decode core at these rows runs persistent, paired with the FGS qkvg order of ``_CORE_FIRST`` heads: one
# wave of 4 CTAs per SM holds the core at <= 8 rows, and o_proj's 8-row tile does not fit beside it.
_CORE_ROWS = (9, 16)
_CORE_CPS = 4
_CORE_FIRST = 70


_head_orders: Dict[tuple, torch.Tensor] = {}


def head_order(heads: int, first: int, split: int, device) -> torch.Tensor:
    """int32 [heads] head of each core group position: heads ``[0, first)`` then the rest, each part interleaved over
    ``split`` equal head ranges (``h mod (heads / split)`` major). Built on the first (eager) call."""
    key = (heads, first, split, str(device))
    if key not in _head_orders:
        span = heads // split
        order = [
            h
            for part in (range(first), range(first, heads))
            for h in sorted(part, key=lambda h: (h % span, h // span))
        ]
        _head_orders[key] = torch.tensor(order, dtype=torch.int32, device=device)
    return _head_orders[key]


def _o_proj_split(rows: int, heads: int) -> int:
    """Split-K of the chained decode o_proj at ``rows``: the split-K FGS consumer (``_fgs_consume``) where the table
    runs a split-K kernel, else the ``rubin`` build at split-K 1 (3 if unknown)."""
    table = getattr(kimi_k3_cute_gemm, "_GEMM_TABLE", {}).get((7168, heads * 128))
    if table is not None and kimi_k3_cute_gemm._select(table, rows)[0] == "rubin":
        return 1
    tactics = getattr(kimi_k3_cute_gemm, "_FGS_TACTICS", ())
    return next((t[1][2] for t in tactics if rows <= t[0]), 3)


def core_kwargs(rows: int, heads: int = 96, lease=None) -> Dict[str, object]:
    """Extra ``kda_decode_t1`` kwargs of the decode core when it produces a ``CORE`` slot (empty outside
    ``_CORE_ROWS``): a persistent grid of ``_CORE_CPS`` CTAs per SM that leaves one o_proj CTA room on each SM,
    publishing all rows of one head of each of o_proj's split-K slices per group; the heads qkvg publishes first,
    first; the late output-gate wait (applies only to the consumer build)."""
    if not edges.chain() or not _CORE_ROWS[0] <= rows <= _CORE_ROWS[1]:
        return {}
    split = _o_proj_split(rows, heads)
    device = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return dict(
        fgs_head_group=split,
        fgs_head_order=head_order(heads, _CORE_FIRST, split, device),
        fgs_smem_carveout=kimi_k3_cute_gemm._VSPLITK_CARVEOUT,
        fgs_persistent=True,
        fgs_ctas=_CORE_CPS * sms,
        fgs_prefetch_next=True,
        max_regs=0,
        fgs_late_gate=True,
    )


def bfa_on_main_stream() -> bool:
    return False


def core_sync_kwargs(rows: int, consumer: bool, producer: bool) -> Dict[str, object]:
    """``kda_decode_t1`` kwargs of the decode core's sync form: TMA ``report_valid`` loads in the consumer, a TMA
    store of the core output in the producer."""
    kwargs: Dict[str, object] = {}
    if producer:
        kwargs["fgs_producer"] = True
        kwargs["fgs_tma_store"] = True
    if consumer:
        kwargs["fgs_consumer"] = True
        kwargs["fgs_input_check"] = "tma"
    return kwargs


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


def mix_owner(layer, layer_idx: int, rows: int):
    """Owner of a decoder layer's chained-mix buffers: the layer itself."""
    return layer


def mix_counters(owner, device) -> torch.Tensor:
    """Persistent int32[4] counters of the pre-MoE mix consumer (attn_res_mix ``make_fgs_counters`` layout)."""
    bufs: Dict[tuple, torch.Tensor] = owner.__dict__.setdefault("_k3_chain_bufs", {})
    key = ("mix_counters", device)
    if key not in bufs:
        bufs[key] = torch.zeros(4, dtype=torch.int32, device=device)
    return bufs[key]
