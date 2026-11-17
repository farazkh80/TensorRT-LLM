# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill routed-expert path: routing -> token-M FC1 -> FC2 (finalize fused) -> finalize of multi-pair rows.

``TLLM_K3_MOE_PREFILL_ROUTING`` picks the routing stage: ``cute`` (``moe.prefill.routing``, default) or ``torch``
(the sync-free reference of the moe_chain routing contract). The permuted buffers are sized for
``TLLM_K3_MOE_PREFILL_PAIRS_PER_ROW`` local (row, k) pairs per received row (default twice the rank-balanced
16 / EP: 2 at EP16, 8 at EP4); the CuTe routing traps on a routing with more local pairs, the torch one drops them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from .. import finalize
from . import fc1, fc2

HIDDEN = 3584
INTER = 3072
TOP_K = 16
SF_VEC = 32
TILE = 128
ROUTING_ENV = "TLLM_K3_MOE_PREFILL_ROUTING"
PAIRS_ENV = "TLLM_K3_MOE_PREFILL_PAIRS_PER_ROW"


@dataclass
class Plan:
    tile_n: int
    max_rows: int
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    permuted_idx_to_expanded_idx: torch.Tensor
    expanded_idx_to_permuted_idx: torch.Tensor
    tile_idx_to_expert_idx: torch.Tensor
    tile_idx_to_mn_limit: torch.Tensor
    num_non_exiting_tiles: torch.Tensor
    total_num_padded_tokens: torch.Tensor


class Workspace:
    def __init__(self, device: torch.device, max_rows: int, num_local: int, ep: int):
        pairs = int(os.environ.get(PAIRS_ENV, 0)) or min(TOP_K, 2 * -(-TOP_K // ep))
        p = (max_rows * min(pairs, TOP_K) + num_local * (TILE - 1) + TILE - 1) // TILE * TILE
        kw = dict(dtype=torch.int32, device=device)
        self.max_rows = max_rows
        self.plan = Plan(
            TILE,
            max_rows,
            torch.zeros(num_local, **kw),
            torch.zeros(num_local + 1, **kw),
            torch.full((p,), -1, **kw),
            torch.full((max_rows * TOP_K,), -1, **kw),
            torch.zeros(p // TILE, **kw),
            torch.zeros(p // TILE, **kw),
            torch.zeros(1, **kw),
            torch.zeros(1, **kw),
        )
        self.act = torch.empty(p, INTER, dtype=torch.float8_e4m3fn, device=device)
        self.act_sf = torch.empty(p, INTER // SF_VEC, dtype=torch.uint8, device=device)
        self.y = torch.empty(p, HIDDEN, dtype=torch.bfloat16, device=device)
        self.tile_done = torch.zeros(p // TILE, dtype=torch.int32, device=device)


_workspaces: Dict[Tuple[torch.device, int, int], Workspace] = {}


def workspace(device: torch.device, rows: int, num_local: int, ep: int) -> Workspace:
    key = (device, num_local, ep)
    ws = _workspaces.get(key)
    if ws is None or ws.max_rows < rows:
        _workspaces.pop(key, None)
        ws = _workspaces[key] = Workspace(device, rows, num_local, ep)
    return ws


def routing_torch(topk_ids: torch.Tensor, lo: int, num_local: int, plan: Plan) -> None:
    """Local (row, k) pairs grouped by local expert in ascending expanded index, each expert padded to tile_n."""
    dev = topk_ids.device
    tile = plan.tile_n
    ids = topk_ids.reshape(-1).long()
    n = ids.numel()
    local = (ids >= lo) & (ids < lo + num_local)
    le = torch.where(local, ids - lo, torch.full_like(ids, num_local))
    counts = torch.zeros(num_local + 1, dtype=torch.long, device=dev).scatter_add_(0, le, torch.ones_like(le))
    counts = counts[:num_local]
    padded = (counts + tile - 1) // tile * tile
    offsets = torch.zeros(num_local + 1, dtype=torch.long, device=dev)
    offsets[1:] = padded.cumsum(0)
    order = torch.argsort(le * n + torch.arange(n, device=dev))
    le_sorted = le[order]
    starts = torch.zeros(num_local + 1, dtype=torch.long, device=dev)
    starts[1:] = counts.cumsum(0)
    rank = torch.arange(n, device=dev) - starts[le_sorted.clamp(max=num_local)]
    p_cap = plan.permuted_idx_to_expanded_idx.numel()
    perm = torch.where(
        le_sorted < num_local,
        offsets[le_sorted.clamp(max=num_local - 1)] + rank,
        torch.full_like(rank, p_cap),
    ).clamp(max=p_cap)
    p2e = torch.full((p_cap + 1,), -1, dtype=torch.long, device=dev)
    p2e.scatter_(0, perm, order)
    e2p = torch.full((n,), -1, dtype=torch.long, device=dev)
    e2p.scatter_(0, order, torch.where(perm < p_cap, perm, torch.full_like(perm, -1)))
    t_cap = plan.tile_idx_to_expert_idx.numel()
    tile_start = torch.arange(t_cap, device=dev) * tile
    t_exp = torch.searchsorted(offsets[1:], tile_start, right=True).clamp(max=num_local - 1)
    plan.expert_counts.copy_(counts)
    plan.expert_offsets.copy_(offsets)
    plan.permuted_idx_to_expanded_idx.copy_(p2e[:p_cap])
    plan.expanded_idx_to_permuted_idx.fill_(-1)
    plan.expanded_idx_to_permuted_idx[:n].copy_(e2p)
    plan.tile_idx_to_expert_idx.copy_(t_exp)
    plan.tile_idx_to_mn_limit.copy_(torch.minimum(tile_start + tile, offsets[t_exp] + counts[t_exp]))
    plan.num_non_exiting_tiles.copy_((offsets[num_local:] // tile).clamp(max=t_cap))
    plan.total_num_padded_tokens.copy_(offsets[num_local:].clamp(max=p_cap))


def _routing(name: str):
    if name == "torch":
        return routing_torch
    if name == "cute":
        from . import routing

        return routing.run
    raise ValueError(f"{ROUTING_ENV}: unknown routing {name!r} (torch | cute)")


def run(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    weights,
    out: torch.Tensor,
    ep: int,
    *,
    fc2_ctas: Optional[int] = None,
    after_fc2: Optional[Callable] = None,
    after_finalize: Optional[Callable] = None,
) -> Workspace:
    """Writes ``out`` [R, 3584] bf16: per received row, the top-k weighted sum of its local experts' outputs.

    ``after_fc2(out, plan, tile_done)`` is launched right after FC2, which then publishes one count per finished
    (m-tile, n-tile) into ``tile_done``, triggers its dependents at entry and runs on ``fc2_ctas`` CTAs (the FC2 ->
    combine push edge); ``after_finalize(out, plan)`` is launched after the finalize."""
    rows = x.shape[0]
    assert x.shape == (rows, HIDDEN) and x.dtype == torch.float8_e4m3fn, (x.shape, x.dtype)
    assert x_sf.shape == (rows, HIDDEN // SF_VEC) and x_sf.dtype == torch.uint8, (x_sf.shape, x_sf.dtype)
    assert topk_ids.shape == (rows, TOP_K) and topk_ids.dtype == torch.int32
    assert topk_w.shape == (rows, TOP_K) and topk_w.dtype == torch.bfloat16
    assert out.shape == (rows, HIDDEN) and out.dtype == torch.bfloat16 and out.is_contiguous()
    ws = workspace(x.device, rows, weights.num_local, ep)
    plan = ws.plan
    _routing(os.environ.get(ROUTING_ENV, "cute"))(
        topk_ids, weights.local_expert_offset, weights.num_local, plan
    )
    if after_fc2 is not None:
        ws.tile_done.zero_()
    fc1.run(x, x_sf, weights, plan, ws.act, ws.act_sf)
    # FC2 writes the rows with one local pair; finalize writes the others (zero where no pair is local).
    if after_fc2 is None:
        fc2.run(ws.act, ws.act_sf, weights, plan, ws.y, fin=(out, topk_w))
    else:
        fc2.run(
            ws.act,
            ws.act_sf,
            weights,
            plan,
            ws.y,
            fin=(out, topk_w),
            publish=ws.tile_done,
            pdl_trigger="start",
            num_ctas=fc2_ctas,
        )
        after_fc2(out, plan, ws.tile_done)
    finalize.run(ws.y, topk_w, plan, out, skip_single=True)
    if after_finalize is not None:
        after_finalize(out, plan)
    return ws
