# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode MoE expert path in CuTe DSL (``TLLM_K3_MOE=cute``): routing -> FC1 -> FC2 -> finalize.

It replaces only the TRTLLM-Gen runner call inside the routed experts' ``run_moe``; quantize, dispatch and combine
stay stock. Per call, on one rank, with R = ep_size * max_tokens_per_rank received rows:

  inputs (stock dispatch)  x [R, 3584] e4m3, x_sf [R, 112] UE8M0 (linear), topk_ids [R, 16] int32 global ids
                           (-1 on unused receive slots), topk_w [R, 16] bf16
  routing                  RoutingPlan of the (row, k) pairs whose expert is local, grouped by local expert in
                           ascending expanded index (row * 16 + k), each expert padded to ``tile_n`` rows
  FC1                      act [P, 3072] e4m3 + act_sf [P, 96] UE8M0 (linear), in permuted-row order
  FC2                      y [P, 3584] bf16, in permuted-row order
  finalize                 out[r] = bf16(sum over k = 0..15 in order, fp32, of topk_w[r, k] * y[e2p[r * 16 + k]])
                           for every r < R (zero when no k is local), written into the combine payload buffer

Weights are the backend's TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers, read in place. Every buffer is allocated once per
device for the largest R seen outside graph capture, so CUDA-graph replays and FGS sentinels see fixed addresses.
"""

import importlib
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

ENV = "TLLM_K3_MOE"
HIDDEN = 3584
INTER = 3072
TOP_K = 16
SF_VEC = 32
STAGES = ("routing", "fc1", "fc2", "finalize")


def enabled() -> bool:
    return os.environ.get(ENV, "") == "cute"


def tile_n_for(rows: int) -> int:
    """Token rows per FC tile (the N of the swap-AB MMA) for R received rows."""
    override = os.environ.get("TLLM_K3_MOE_TILE_N")
    if override:
        return int(override)
    return 8 if rows <= 512 else 16 if rows <= 1024 else 32


@dataclass(frozen=True)
class ExpertWeights:
    """This rank's experts [local_expert_offset, local_expert_offset + num_local) in the TRTLLM-Gen layout:
    w3_w1_weight uint8 [E, 6144, 1792] (rows [up; gate] interleaved per row pair, then shuffled in 128-row epilogue
    tiles), w3_w1_weight_scale uint8 [E, 6144, 112] (same row order, 128x4 SF-atom interleaved), w2_weight uint8
    [E, 3584, 1536], w2_weight_scale uint8 [E, 3584, 96]. SiTU-GLU: gate_cap * tanh(g / gate_cap) * sigmoid(g) *
    linear_cap * tanh(u / linear_cap)."""

    w3_w1_weight: torch.Tensor
    w3_w1_weight_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor
    local_expert_offset: int
    num_local: int
    gate_cap: float
    linear_cap: float

    @classmethod
    def from_backend(cls, backend) -> "ExpertWeights":
        act = backend.activation
        return cls(
            backend.w3_w1_weight,
            backend.w3_w1_weight_scale,
            backend.w2_weight,
            backend.w2_weight_scale,
            int(backend.slot_start),
            int(backend.expert_size_per_partition),
            float(act.gate_softcap),
            float(act.linear_softcap),
        )


@dataclass
class RoutingPlan:
    """Routing output, sized for ``max_rows`` received rows (``trtllm::moe_sort`` names and meanings). Each local
    (row, k) pair appears exactly once; FC1 / FC2 / finalize results do not depend on the order within an expert.
    Kernels read only the tile maps, the two index maps and the two counters; padding slots of
    ``permuted_idx_to_expanded_idx`` are guarded by ``tile_idx_to_mn_limit``, and slots past
    ``total_num_padded_tokens`` and tiles past ``num_non_exiting_tiles`` are not read.

    expert_counts                 int32 [E]          local (row, k) pairs per local expert
    expert_offsets                int32 [E + 1]      first permuted row of each expert (multiples of tile_n)
    permuted_idx_to_expanded_idx  int32 [P]          expanded index row * 16 + k
    expanded_idx_to_permuted_idx  int32 [rows * 16]  -1 where the expert is not local or the row is unused
    tile_idx_to_expert_idx        int32 [T]          local expert of each tile_n-row tile, tiles in expert order
    tile_idx_to_mn_limit          int32 [T]          expert_offsets[e] + expert_counts[e] of the tile's expert
    num_non_exiting_tiles         int32 [1]
    total_num_padded_tokens       int32 [1]          expert_offsets[E]
    """

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


def max_permuted(max_rows: int, num_local: int, tile_n: int) -> int:
    # Each row has at most min(16, E) local experts; each expert adds at most tile_n - 1 padding rows.
    return max_rows * min(TOP_K, num_local) + num_local * (tile_n - 1)


class Workspace:
    """Per-device buffers of the expert path; layers run in stream order and share them."""

    def __init__(self, device: torch.device, max_rows: int, num_local: int, tile_n: int):
        p = max_permuted(max_rows, num_local, tile_n)
        t = p // tile_n
        kw = dict(dtype=torch.int32, device=device)
        self.device = device
        self.max_rows = max_rows
        self.num_local = num_local
        self.tile_n = tile_n
        self.plan = RoutingPlan(
            tile_n,
            max_rows,
            torch.zeros(num_local, **kw),
            torch.zeros(num_local + 1, **kw),
            torch.full((p,), -1, **kw),
            torch.full((max_rows * TOP_K,), -1, **kw),
            torch.zeros(t, **kw),
            torch.zeros(t, **kw),
            torch.zeros(1, **kw),
            torch.zeros(1, **kw),
        )
        self.act = torch.empty(p, INTER, dtype=torch.float8_e4m3fn, device=device)
        self.act_sf = torch.empty(p, INTER // SF_VEC, dtype=torch.uint8, device=device)
        self.y = torch.empty(p, HIDDEN, dtype=torch.bfloat16, device=device)


_workspaces: Dict[tuple, Workspace] = {}


def workspace(device: torch.device, rows: int, num_local: int, tile_n: int) -> Workspace:
    key = (device, num_local, tile_n)
    ws = _workspaces.get(key)
    if ws is None or ws.max_rows < rows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{ENV}=cute: no workspace for {rows} rows (tile_n {tile_n}) before graph capture; "
                "run the batch eagerly first"
            )
        ws = _workspaces[key] = Workspace(device, rows, num_local, tile_n)
    return ws


def _stage(name: str) -> Callable:
    return importlib.import_module(f"{__package__}.moe.{name}").run


def run_expert_path(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    weights: ExpertWeights,
    out: torch.Tensor,
    stages: Optional[Dict[str, Callable]] = None,
    ws: Optional[Workspace] = None,
) -> Workspace:
    """Run routing -> FC1 -> FC2 -> finalize on the current stream; writes ``out`` [R, 3584] bf16.

    ``stages`` overrides any of ``STAGES`` (tests plug reference or candidate kernels in per stage); the rest come
    from ``kimi_k3_fgs.moe.<stage>.run``. Stage signatures:
      routing(topk_ids, local_expert_offset, num_local, plan)
      fc1(x, x_sf, weights, plan, act, act_sf)
      fc2(act, act_sf, weights, plan, y)
      finalize(y, topk_w, plan, out)
    """
    rows = x.shape[0]
    assert x.shape == (rows, HIDDEN) and x.dtype == torch.float8_e4m3fn, (x.shape, x.dtype)
    assert x_sf.shape == (rows, HIDDEN // SF_VEC) and x_sf.dtype == torch.uint8, x_sf.dtype
    assert topk_ids.shape == (rows, TOP_K) and topk_ids.dtype == torch.int32
    assert topk_w.shape == (rows, TOP_K) and topk_w.dtype == torch.bfloat16
    assert out.shape == (rows, HIDDEN) and out.dtype == torch.bfloat16 and out.is_contiguous()
    stages = stages or {}
    run = {name: stages.get(name) or _stage(name) for name in STAGES}
    if ws is None:
        ws = workspace(x.device, rows, weights.num_local, tile_n_for(rows))
    plan = ws.plan
    run["routing"](topk_ids, weights.local_expert_offset, weights.num_local, plan)
    run["fc1"](x, x_sf, weights, plan, ws.act, ws.act_sf)
    run["fc2"](ws.act, ws.act_sf, weights, plan, ws.y)
    run["finalize"](ws.y, topk_w, plan, out)
    return ws


def install(routed_experts) -> bool:
    """Route ``routed_experts``' TRTLLM-Gen W4A8 MXFP4 x MXFP8 ``run_moe`` through :func:`run_expert_path` when
    ``TLLM_K3_MOE=cute``. Calls the path cannot take (no combine payload buffer, unfinalized output, internal
    routing) keep the stock runner."""
    if not enabled():
        return False
    backend = routed_experts.backend
    w31 = getattr(backend, "w3_w1_weight", None)
    w31_s = getattr(backend, "w3_w1_weight_scale", None)
    if (
        w31 is None
        or w31.dtype != torch.uint8
        or tuple(w31.shape[1:]) != (2 * INTER, HIDDEN // 2)
        or tuple(w31_s.shape[1:]) != (2 * INTER, HIDDEN // SF_VEC)
        or not hasattr(backend.activation, "gate_softcap")
    ):
        raise ValueError(
            f"{ENV}=cute needs the TRTLLM-Gen MXFP4 x MXFP8 SiTU experts, got {type(backend).__name__}"
        )
    stock_run_moe = backend.run_moe

    def run_moe(ctx, *, workspace=None):
        plan = ctx.comm_plan
        out = None if plan is None else plan.moe_output
        if (
            out is None
            or not ctx.do_finalize
            or ctx.token_selected_experts is None
            or ctx.x_sf is None
        ):
            return stock_run_moe(ctx, workspace=workspace)
        if ctx.x_ready_event is not None:
            ctx.x_ready_event.wait()
        topk_w = ctx.token_final_scales
        if topk_w.dtype != torch.bfloat16:
            topk_w = topk_w.to(torch.bfloat16)
        run_expert_path(
            ctx.x,
            ctx.x_sf.view(torch.uint8),
            ctx.token_selected_experts,
            topk_w,
            ExpertWeights.from_backend(backend),
            out,
        )
        return out

    backend.run_moe = run_moe
    return True
