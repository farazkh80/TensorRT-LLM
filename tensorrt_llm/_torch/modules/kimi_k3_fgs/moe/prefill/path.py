# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill routed-expert path: routing -> token-M FC1 -> FC2 (finalize fused) -> finalize of multi-pair rows.

``TLLM_K3_MOE_PREFILL_ROUTING`` picks the routing stage: ``cute`` (``moe.prefill.routing``, default) or ``torch``
(the sync-free reference of the moe_chain routing contract). The permuted buffers are sized for
``TLLM_K3_MOE_PREFILL_PAIRS_PER_ROW`` local (row, k) pairs per received row (default twice the rank-balanced
16 / EP: 2 at EP16, 8 at EP4); the CuTe routing traps on a routing with more local pairs, the torch one drops them.
``TLLM_K3_MOE_PREFILL_PUSH`` (``push_mode()``) names the FC2 -> combine push the caller wires: ``tiles``
(``fc2_push``: tile push, finalize, rest push) or ``stream`` (``stream_push``: finalize fused into one push kernel).
``TLLM_K3_MOE_PREFILL_FC1_STORE`` / ``TLLM_K3_MOE_PREFILL_FC2_STORE`` (``v4`` | ``tma``) pick the kernels' output
store path. ``TLLM_K3_MOE_PREFILL_FC12`` is the FC1 -> FC2 edge: ``off`` (FC2 waits for the FC1 grid, standard PDL),
``B`` (act slots armed 0x80, FC2 checks each 16 B chunk's first byte by TMA ``report_valid``, scales per element
against 0xFF), ``A`` (slots armed 0xFF, every element checked) or ``counter`` (FC1 publishes per (m, n) item, FC2
acquires per m-tile). B and A keep two slot pairs that alternate per use; FC1 re-arms the other pair.
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
FC1_TILE_N = 256
ROUTING_ENV = "TLLM_K3_MOE_PREFILL_ROUTING"
PAIRS_ENV = "TLLM_K3_MOE_PREFILL_PAIRS_PER_ROW"
PUSH_ENV = "TLLM_K3_MOE_PREFILL_PUSH"
PUSH_MODES = ("tiles", "stream")
FC1_STORE_ENV = "TLLM_K3_MOE_PREFILL_FC1_STORE"
FC2_STORE_ENV = "TLLM_K3_MOE_PREFILL_FC2_STORE"
STORES = ("v4", "tma")
FC12_ENV = "TLLM_K3_MOE_PREFILL_FC12"
FC12_MODES = ("off", "B", "A", "counter")
FC12_SENTINEL = {"B": "80", "A": "ff"}


def _choice(env: str, default: str, choices) -> str:
    value = os.environ.get(env, default)
    if value not in choices:
        raise ValueError(f"{env}: unknown value {value!r} ({' | '.join(choices)})")
    return value


def push_mode() -> str:
    return _choice(PUSH_ENV, "tiles", PUSH_MODES)


def fc1_store() -> str:
    return _choice(FC1_STORE_ENV, "v4", STORES)


def fc2_store() -> str:
    return _choice(FC2_STORE_ENV, "v4", STORES)


def fc12_mode() -> str:
    return _choice(FC12_ENV, "off", FC12_MODES)


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
        self.fc12 = fc12_mode()
        if self.fc12 in FC12_SENTINEL:
            fill = 0x80 if self.fc12 == "B" else 0xFF
            self.slots = [
                (
                    torch.full((p, INTER), fill, dtype=torch.uint8, device=device).view(
                        torch.float8_e4m3fn
                    ),
                    torch.full((p, INTER // SF_VEC), 0xFF, dtype=torch.uint8, device=device),
                )
                for _ in range(2)
            ]
            # Rows FC1 wrote into each slot in its last use: what the next FC1 on the other slot re-arms.
            self.rows_used = torch.zeros(2, **kw)
            self.use = 0
        else:
            self.act = torch.empty(p, INTER, dtype=torch.float8_e4m3fn, device=device)
            self.act_sf = torch.empty(p, INTER // SF_VEC, dtype=torch.uint8, device=device)
        if self.fc12 == "counter":
            self.fc1_done = torch.zeros(p // TILE, **kw)
        self.y = torch.empty(p, HIDDEN, dtype=torch.bfloat16, device=device)
        # One word per m-tile plus the streaming push's row-dealing word, zeroed together per round.
        self.tile_done = torch.zeros(p // TILE + 1, dtype=torch.int32, device=device)


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
    counts = torch.zeros(num_local + 1, dtype=torch.long, device=dev).scatter_add_(
        0, le, torch.ones_like(le)
    )
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
    plan.tile_idx_to_mn_limit.copy_(
        torch.minimum(tile_start + tile, offsets[t_exp] + counts[t_exp])
    )
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
    stream: Optional[Callable] = None,
    stream_fgs: bool = True,
) -> Workspace:
    """Writes ``out`` [R, 3584] bf16: per received row, the top-k weighted sum of its local experts' outputs.

    ``after_fc2(out, plan, tile_done)`` is launched right after FC2, which then publishes one count per finished
    (m-tile, n-tile) into ``tile_done``, triggers its dependents at entry and runs on ``fc2_ctas`` CTAs (the FC2 ->
    combine push edge); ``after_finalize(out, plan)`` is launched after the finalize.
    ``stream(out, y, topk_w, plan, tile_done)`` replaces the finalize (``stream_push.push_stream``: the finalize is
    fused into the push, so ``out`` keeps only the single-pair rows): with ``stream_fgs`` FC2 runs as for
    ``after_fc2``, else plain on the full grid (the FGS-off twin)."""
    rows = x.shape[0]
    assert x.shape == (rows, HIDDEN) and x.dtype == torch.float8_e4m3fn, (x.shape, x.dtype)
    assert x_sf.shape == (rows, HIDDEN // SF_VEC) and x_sf.dtype == torch.uint8, (
        x_sf.shape,
        x_sf.dtype,
    )
    assert topk_ids.shape == (rows, TOP_K) and topk_ids.dtype == torch.int32
    assert topk_w.shape == (rows, TOP_K) and topk_w.dtype == torch.bfloat16
    assert out.shape == (rows, HIDDEN) and out.dtype == torch.bfloat16 and out.is_contiguous()
    ws = workspace(x.device, rows, weights.num_local, ep)
    plan = ws.plan
    _routing(os.environ.get(ROUTING_ENV, "cute"))(
        topk_ids, weights.local_expert_offset, weights.num_local, plan
    )
    publish = after_fc2 is not None or (stream is not None and stream_fgs)
    if publish or stream is not None:
        ws.tile_done.zero_()
    fc1_kw = dict(store=fc1_store())
    # FC2 writes the rows with one local pair; finalize writes the others (zero where no pair is local).
    fc2_kw = dict(store=fc2_store(), fin=(out, topk_w))
    lamport = ws.fc12 in FC12_SENTINEL
    if lamport:
        # The pair this use writes alternates; FC1 re-arms the other pair over the rows its last use wrote (that
        # pair's reader, the previous FC2, completed before FC1's grid wait returned), and FC2 checks its loads
        # instead of waiting for the FC1 grid.
        s = ws.use & 1
        ws.use += 1
        act, act_sf = ws.slots[s]
        prev_act, prev_sf = ws.slots[1 - s]
        sentinel = FC12_SENTINEL[ws.fc12]
        fc1_kw.update(sentinel=sentinel, rearm=(prev_act, prev_sf, ws.rows_used[1 - s : 2 - s]))
        fc2_kw.update(sync="lamport", sentinel=sentinel)
    else:
        act, act_sf = ws.act, ws.act_sf
        if ws.fc12 == "counter":
            ws.fc1_done.zero_()
            fc1_kw.update(counters=(ws.fc1_done, ws.fc1_done))
            fc2_kw.update(sync="counter", counters=(ws.fc1_done, 2 * INTER // FC1_TILE_N))
    if publish:
        fc2_kw.update(publish=ws.tile_done, pdl_trigger="start", num_ctas=fc2_ctas)
    fc1.run(x, x_sf, weights, plan, act, act_sf, tile_n=FC1_TILE_N, **fc1_kw)
    if lamport:
        ws.rows_used[s : s + 1].copy_(plan.total_num_padded_tokens)
    fc2.run(act, act_sf, weights, plan, ws.y, **fc2_kw)
    if after_fc2 is not None:
        after_fc2(out, plan, ws.tile_done)
    if stream is not None:
        stream(out, ws.y, topk_w, plan, ws.tile_done)
        return ws
    finalize.run(ws.y, topk_w, plan, out, skip_single=True)
    if after_finalize is not None:
        after_finalize(out, plan)
    return ws
