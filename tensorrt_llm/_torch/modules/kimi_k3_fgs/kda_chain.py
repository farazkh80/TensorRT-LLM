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

With ``TLLM_K3_FGS_PREFILL=1`` a prefill of ``TLLM_K3_FGS_PREFILL_MIN_ROWS`` to ``_MAX_ROWS`` tokens runs as

  qkvg GEMM --p_qkvg--> KdaConv | [f_a | b], f_b, BT16 prep (plain) | chain_dv2 --p_o--> gated norm
  --p_core--> o_proj --p_attn_o--> pre-MoE mix (pipe consumer)

``TLLM_K3_FGS_PREFILL_EDGES`` (comma list of pk2, pk7, pk8, pk9; default all) keeps only the listed edges: a dropped
edge is a plain boundary and its slot is neither declared nor reset.

The pre-attention mix (pipe kernel, no producer mode) and BT16 prep (no consumer mode) are plain boundaries. The
gated norm reads the gate columns of the p_qkvg slot unchecked: chain_dv2 is a plain launch, so the qkvg GEMM has
completed when the norm starts.

``TLLM_K3_FGS_KDA_CORE=persistent`` (off by default) runs the decode KdaDecodeT1 producer of ``CORE`` as one resident
wave that leaves an o_proj CTA room on every SM (:func:`core_kwargs`).
"""

import atexit
import collections
import functools
import os
from typing import Dict, Optional, Tuple

import torch

from ....logger import logger
from .. import kimi_k3_cute_gemm
from . import edges

X_NORM = "kda.x_norm"
BFA = "kda.bfa"
GATE = "kda.g"
QKVG = "kda.qkvg"
CORE = "kda.core_o"
ATTN_O = "kda.attn_o"
P_QKVG = "kda.p_qkvg"
P_O = "kda.p_o"
P_CORE = "kda.p_core"
P_ATTN_O = "kda.p_attn_o"

PREFILL_MIN_ROWS = int(os.environ.get("TLLM_K3_FGS_PREFILL_MIN_ROWS", "1024"))
PREFILL_MAX_ROWS = int(os.environ.get("TLLM_K3_FGS_PREFILL_MAX_ROWS", "8256"))


def bfa_first() -> bool:
    value = os.environ.get("TLLM_K3_FGS_CHAIN_BFA") or (
        "after" if edges.mode() == "chain_after" else "before"
    )
    if value not in ("before", "after"):
        raise ValueError(f"TLLM_K3_FGS_CHAIN_BFA={value!r}; valid: before, after")
    return value == "before"


_PREFILL_EDGE_IDS = {"pk2": P_QKVG, "pk7": P_O, "pk8": P_CORE, "pk9": P_ATTN_O}


def prefill() -> bool:
    return edges.chain() and os.environ.get("TLLM_K3_FGS_PREFILL", "0") == "1"


def prefill_edges() -> Tuple[str, ...]:
    """The prefill edges ``TLLM_K3_FGS_PREFILL_EDGES`` keeps, in chain order."""
    value = os.environ.get("TLLM_K3_FGS_PREFILL_EDGES", ",".join(_PREFILL_EDGE_IDS))
    ids = {v.strip().lower() for v in value.split(",") if v.strip()}
    unknown = ids - set(_PREFILL_EDGE_IDS)
    if unknown:
        raise ValueError(
            f"TLLM_K3_FGS_PREFILL_EDGES: unknown {sorted(unknown)}; valid: {', '.join(_PREFILL_EDGE_IDS)}"
        )
    return tuple(name for edge_id, name in _PREFILL_EDGE_IDS.items() if edge_id in ids)


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
    if prefill():
        widths = {P_QKVG: qkvg_width, P_O: proj_size, P_CORE: proj_size, P_ATTN_O: hidden}
        for name in prefill_edges():
            edges.declare(
                name, widths[name], layer_idx, max_rows=PREFILL_MAX_ROWS, min_rows=PREFILL_MIN_ROWS
            )


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


class Prefill:
    """One KDA layer's prefill slots, from :func:`prefill_begin` to the gated norm. A dropped edge has no slot."""

    def __init__(self, leases: Dict[str, object]):
        self.leases = leases

    def slot(self, name: str) -> Optional[torch.Tensor]:
        lease = self.leases.get(name)
        return None if lease is None else lease.tensor

    def consumed(self, name: str) -> None:
        if name in self.leases:
            edges.consumed(self.leases[name])


def prefill_begin(rows: int) -> Optional[Prefill]:
    """The p_qkvg (read by the conv and the gated norm), p_o and p_core slots of one prefill, or None (nothing held)."""
    if not prefill():
        return None
    leases = {}
    kept = prefill_edges()
    for name, readers in ((P_QKVG, 2), (P_O, 1), (P_CORE, 1)):
        if name not in kept:
            continue
        lease = edges.acquire(
            name, rows, readers=readers, window=(PREFILL_MIN_ROWS, PREFILL_MAX_ROWS)
        )
        if lease is None:
            for held in leases.values():
                edges.release(held)
            return None
        leases[name] = lease
    return Prefill(leases)


# Prefill layer-forwards by outcome ("on" or the refusal), logged every PREFILL_LOG_EVERY forwards.
prefill_outcomes: collections.Counter = collections.Counter()
PREFILL_LOG_EVERY = 20


def note_prefill(layer_idx: int, refusal: Optional[str]) -> None:
    """Count one prefill layer-forward; log each new outcome once and the totals periodically."""
    if not prefill_outcomes:
        atexit.register(
            lambda: logger.info(
                f"TLLM_K3_FGS prefill chain layer-forwards: {dict(prefill_outcomes)}"
            )
        )
    outcome = "on" if refusal is None else f"off: {refusal}"
    if outcome not in prefill_outcomes:
        logger.info(f"TLLM_K3_FGS prefill chain, first {outcome!r} at KDA layer {layer_idx}")
    prefill_outcomes[outcome] += 1
    if sum(prefill_outcomes.values()) % PREFILL_LOG_EVERY == 0:
        logger.info(f"TLLM_K3_FGS prefill chain layer-forwards: {dict(prefill_outcomes)}")


CORE_ENV = "TLLM_K3_FGS_KDA_CORE"
_CORE_KEYS = ("rows", "cps", "ctas", "hg", "split", "pf", "regs", "carveout")
_SM_REGS = 65536
_CORE_THREADS = 256


@functools.lru_cache(maxsize=None)
def _core_knob() -> Optional[Dict[str, object]]:
    """``TLLM_K3_FGS_KDA_CORE``: empty / ``0`` off; ``persistent`` with optional ``key=value`` overrides, comma
    separated (e.g. ``persistent,cps=2``):

      rows=lo:hi   decode rows it applies to (9:16: one 4-CTA/SM wave holds the core at <= 8 rows, and o_proj's
                   8-row tile does not fit beside it)
      cps=n        core CTAs per SM (3); ctas=n sets the grid instead
      split=s      head order interleaved over o_proj's split-K slices (default: o_proj's split at the rows)
      hg=g         heads per publication group (default ``split``: one head of every K slice)
      pf=0|1       prefetch the next item's state rows (1)
      regs=r       register cap per thread (0: none); default the smallest above 64 (the 4-CTA bound) that keeps a
                   (cps+1)-th core CTA off the SM, so the free slot is left to o_proj
      carveout=p   preferred smem carveout, percent (default o_proj's; -1: the kernel's own)
    """
    value = os.environ.get(CORE_ENV, "").strip()
    if value in ("", "0"):
        return None
    knob: Dict[str, object] = {}
    for item in (v.strip() for v in value.split(",")):
        if item in ("persistent", "1"):
            continue
        key, sep, val = item.partition("=")
        if not sep or key not in _CORE_KEYS:
            raise ValueError(f"{CORE_ENV}: bad item {item!r}; keys: {', '.join(_CORE_KEYS)}")
        knob[key] = tuple(int(v) for v in val.split(":")) if key == "rows" else int(val)
    knob.setdefault("rows", (9, 16))
    if len(knob["rows"]) == 1:
        knob["rows"] = knob["rows"] * 2
    knob.setdefault("cps", 3)
    knob.setdefault("pf", 1)
    cap = (_SM_REGS // ((knob["cps"] + 1) * _CORE_THREADS) // 8 + 1) * 8
    knob.setdefault("regs", min(255, cap) if cap > 64 else 0)
    knob.setdefault("carveout", getattr(kimi_k3_cute_gemm, "_VSPLITK_CARVEOUT", 100))
    return knob


_core_logged = set()


def _o_proj_split(rows: int) -> int:
    """Split-K of the decode o_proj FGS consumer at ``rows`` (``kimi_k3_cute_gemm._fgs_consume``; 3 if unknown)."""
    tactics = getattr(kimi_k3_cute_gemm, "_FGS_TACTICS", ())
    return next((t[1][2] for t in tactics if rows <= t[0]), 3)


def core_kwargs(rows: int) -> Dict[str, object]:
    """Extra ``kda_decode_t1`` kwargs of the decode core when it produces a ``CORE`` slot (empty unless
    ``TLLM_K3_FGS_KDA_CORE`` covers ``rows``): a persistent grid of ``cps`` CTAs per SM that leaves one o_proj CTA
    room on each SM, publishing all rows of one head of each of o_proj's split-K slices per group."""
    knob = _core_knob()
    if knob is None or not knob["rows"][0] <= rows <= knob["rows"][1]:
        return {}
    split = knob.get("split") or _o_proj_split(rows)
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    kwargs = dict(
        fgs_persistent=True,
        fgs_ctas=knob.get("ctas") or knob["cps"] * sms,
        fgs_head_group=knob.get("hg") or split,
        fgs_head_split=split,
        fgs_prefetch_next=bool(knob["pf"]),
        fgs_smem_carveout=None if knob["carveout"] < 0 else knob["carveout"],
        max_regs=knob["regs"],
    )
    if rows not in _core_logged:
        _core_logged.add(rows)
        logger.info(f"{CORE_ENV} rows={rows}: {kwargs}")
    return kwargs


def consume(leases) -> None:
    """One reader of each lease was launched on the current stream."""
    for lease in leases:
        edges.consumed(lease)


def o_proj(core2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``core2d @ weight^T`` into an ``ATTN_O`` (decode) or ``P_ATTN_O`` (prefill) slot when the chain covers the
    batch (consumed by the pre-MoE mix)."""
    rows = core2d.shape[0]
    if prefill() and PREFILL_MIN_ROWS <= rows <= PREFILL_MAX_ROWS:
        lease = (
            edges.acquire(P_ATTN_O, rows, window=(PREFILL_MIN_ROWS, PREFILL_MAX_ROWS))
            if P_ATTN_O in prefill_edges()
            else None
        )
    else:
        lease = acquire(ATTN_O, rows)
    return kimi_k3_cute_gemm.linear(core2d, weight, None if lease is None else lease.tensor)


def buffer(owner, name: str, like: torch.Tensor) -> torch.Tensor:
    """A persistent tensor shaped like ``like``, owned by ``owner`` (one per name and shape)."""
    bufs: Dict[tuple, torch.Tensor] = owner.__dict__.setdefault("_k3_chain_bufs", {})
    key = (name, tuple(like.shape), like.dtype, like.device)
    if key not in bufs:
        bufs[key] = torch.empty_like(like, memory_format=torch.contiguous_format)
    return bufs[key]


_SHARED_OWNERS = (type("PrefillMixBuffers", (), {})(), type("PrefillMixBuffers", (), {})())


def mix_owner(layer, layer_idx: int, rows: int):
    """Owner of a decoder layer's chained-mix buffers: the layer itself for decode rows; for prefill rows one of two
    owners shared by alternating layers (a layer's ``updated`` is read by the next layer only, and the MoE between
    them is a plain boundary)."""
    return layer if rows < PREFILL_MIN_ROWS else _SHARED_OWNERS[layer_idx % 2]


def mix_counters(owner, device) -> torch.Tensor:
    """Persistent int32[4] counters of the pre-MoE mix consumer (attn_res_mix ``make_fgs_counters`` layout)."""
    bufs: Dict[tuple, torch.Tensor] = owner.__dict__.setdefault("_k3_chain_bufs", {})
    key = ("mix_counters", device)
    if key not in bufs:
        bufs[key] = torch.zeros(4, dtype=torch.int32, device=device)
    return bufs[key]
