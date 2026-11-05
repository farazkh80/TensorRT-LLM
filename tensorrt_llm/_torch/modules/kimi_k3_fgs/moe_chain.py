# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode MoE expert path in CuTe DSL (``TLLM_K3_MOE=cute``): routing -> FC1 -> FC2 -> finalize, the CuTe
all-to-all (``TLLM_K3_COMM=cute``), the post-combine latent tail (``TLLM_K3_MOE_TAIL=cute``) and the fine-grained-sync
edges between them (``TLLM_K3_MOE_FGS``). Every switch is off by default.

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
import inspect
import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from . import edges as k3_edges

ENV = "TLLM_K3_MOE"
TAIL_ENV = "TLLM_K3_MOE_TAIL"
COMM_ENV = "TLLM_K3_COMM"
HIDDEN = 3584
INTER = 3072
TOP_K = 16
SF_VEC = 32
STAGES = ("routing", "fc1", "fc2", "finalize")
FGS_ENV = "TLLM_K3_MOE_FGS"
EDGES = ("disp_route", "route_fc1", "fc1_fc2", "fc2_fin", "fin_push", "comb_norm", "norm_up")
# Counter-slab edges (int32, zero-armed every step) and sentinel-slot edges (bf16, -0 armed).
_COUNTERS = {
    "route_fc1": "moe.plan_ready",
    "fc1_fc2": "moe.fc1_done",
    "fc2_fin": "moe.fc2_done",
    "fin_push": "moe.fin_done",
}
_COMM_EDGES = frozenset(("disp_route", "fin_push", "comb_norm"))
COMBINED = "moe.combined"
LATENT_NORM = "moe.latent_norm"
# FC2 K stages of 256 columns, each fed by 4 FC1 units of 64 columns.
FC1_K_STAGES = INTER // 256
# Rows per rank the expert-path edges serve; the tail edges stop where the split-K up-projection consumer does.
FGS_MAX_ROWS = int(os.environ.get("TLLM_K3_MOE_FGS_MAX_ROWS", "64"))
TAIL_FGS_MAX_ROWS = 32
# Hazard tests only: the CuTe combine and routing delay their FGS-published stores by this many ns.
_HAZARD_DELAY_NS = int(os.environ.get("TLLM_K3_MOE_HAZARD_DELAY_NS", "0"))


def enabled() -> bool:
    return os.environ.get(ENV, "") == "cute"


def tail_enabled() -> bool:
    return os.environ.get(TAIL_ENV, "") == "cute"


def _bucket_rows(rows: int) -> int:
    """The largest R that ``tile_n_for`` maps to the same tile_n as ``rows``."""
    return 512 if rows <= 512 else 1024 if rows <= 1024 else rows


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
    tile_idx_to_mn_limit          int32 [T]          min((t + 1) * tile_n, expert_offsets[e] + expert_counts[e])
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
    stage_kwargs: Optional[Dict[str, dict]] = None,
) -> Workspace:
    """Run routing -> FC1 -> FC2 -> finalize on the current stream; writes ``out`` [R, 3584] bf16.

    ``stages`` overrides any of ``STAGES`` (tests plug reference or candidate kernels in per stage); the rest come
    from ``kimi_k3_fgs.moe.<stage>.run``. Stage signatures:
      routing(topk_ids, local_expert_offset, num_local, plan)
      fc1(x, x_sf, weights, plan, act, act_sf)
      fc2(act, act_sf, weights, plan, y)
      finalize(y, topk_w, plan, out)
    ``stage_kwargs`` adds keyword arguments per stage (the FGS edge builds).
    """
    rows = x.shape[0]
    assert x.shape == (rows, HIDDEN) and x.dtype == torch.float8_e4m3fn, (x.shape, x.dtype)
    assert x_sf.shape == (rows, HIDDEN // SF_VEC) and x_sf.dtype == torch.uint8, x_sf.dtype
    assert topk_ids.shape == (rows, TOP_K) and topk_ids.dtype == torch.int32
    assert topk_w.shape == (rows, TOP_K) and topk_w.dtype == torch.bfloat16
    assert out.shape == (rows, HIDDEN) and out.dtype == torch.bfloat16 and out.is_contiguous()
    stages = stages or {}
    kw = stage_kwargs or {}
    run = {name: stages.get(name) or _stage(name) for name in STAGES}
    if ws is None:
        ws = workspace(x.device, rows, weights.num_local, tile_n_for(rows))
    plan = ws.plan
    run["routing"](
        topk_ids, weights.local_expert_offset, weights.num_local, plan, **kw.get("routing", {})
    )
    run["fc1"](x, x_sf, weights, plan, ws.act, ws.act_sf, **kw.get("fc1", {}))
    run["fc2"](ws.act, ws.act_sf, weights, plan, ws.y, **kw.get("fc2", {}))
    run["finalize"](ws.y, topk_w, plan, out, **kw.get("finalize", {}))
    return ws


def _pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def fgs_edges() -> frozenset:
    """The M2 edges ``TLLM_K3_MOE_FGS`` turns on (comma list of ``EDGES``, or ``all``)."""
    value = os.environ.get(FGS_ENV, "")
    names = set(EDGES) if value == "all" else {v for v in value.split(",") if v}
    unknown = names - set(EDGES)
    if unknown:
        raise ValueError(
            f"{FGS_ENV}: unknown edge(s) {sorted(unknown)}; edges are {', '.join(EDGES)}"
        )
    return frozenset(names)


def _accepts(fn: Callable, *names: str) -> bool:
    params = inspect.signature(fn).parameters
    return all(n in params for n in names)


def _missing_sides(edges: frozenset, moe_on: bool, comm_on: bool, tail_on: bool) -> List[str]:
    """Every requested edge needs both its kernel sides and the stages it connects; an edge that cannot run is an
    error rather than a silent M1 arm."""
    from .moe import comm, finalize, latent, routing

    def stage_run(name):
        try:
            return _stage(name)
        except (ImportError, AttributeError):
            return None

    fc1, fc2 = stage_run("fc1"), stage_run("fc2")
    a2a_d, a2a_c = comm.CuteA2A.dispatch, comm.CuteA2A.combine
    need = {
        "disp_route": [
            (moe_on, ENV),
            (comm_on, f"{comm.ENV}=cute"),
            (_accepts(a2a_d, "defer_data_wait"), "dispatch defer_data_wait"),
            (_accepts(routing.run, "source"), "routing source"),
        ],
        "route_fc1": [
            (moe_on, ENV),
            (_accepts(routing.run, "ready"), "routing ready"),
            (fc1 is not None and _accepts(fc1, "plan_ready"), "FC1 plan_ready consumer"),
        ],
        "fc1_fc2": [
            (moe_on, ENV),
            (fc1 is not None and _accepts(fc1, "fc1_done"), "FC1 fc1_done producer"),
            (fc2 is not None and _accepts(fc2, "fc1_done"), "FC2 fc1_done consumer"),
        ],
        "fc2_fin": [
            (moe_on, ENV),
            (fc2 is not None and _accepts(fc2, "fc2_done"), "FC2 fc2_done producer"),
            (_accepts(finalize.run, "fc2_done"), "finalize fc2_done consumer"),
        ],
        "fin_push": [
            (moe_on, ENV),
            (comm_on, f"{comm.ENV}=cute"),
            (_accepts(finalize.run, "fin_done"), "finalize fin_done producer"),
            (_accepts(a2a_c, "fin_done"), "combine fin_done consumer"),
        ],
        "comb_norm": [
            (comm_on, f"{comm.ENV}=cute"),
            (tail_on, TAIL_ENV),
            (_accepts(a2a_c, "out"), "combine out"),
            (_accepts(latent.rmsnorm, "fgs_consumer"), "rmsnorm fgs_consumer"),
        ],
        "norm_up": [
            (tail_on, TAIL_ENV),
            (_accepts(latent.rmsnorm, "fgs_producer"), "rmsnorm fgs_producer"),
            (_accepts(latent.up_proj, "fgs_consumer"), "up_proj fgs_consumer"),
        ],
    }
    return [f"{edge}: {what}" for edge in sorted(edges) for ok, what in need[edge] if not ok]


def _counter_words(edge: str, tiles: int, ep: int) -> int:
    """int32 words of ``edge``'s counter slab for a plan of ``tiles`` token tiles (the kernels size their checks by the
    workspace's plan, which can exceed the current forward's)."""
    return {"route_fc1": 1, "fc1_fc2": tiles * FC1_K_STAGES, "fc2_fin": tiles, "fin_push": ep}[edge]


def _max_tiles(rows: int, num_local: int) -> int:
    tile_n = tile_n_for(rows)
    return max_permuted(rows, num_local, tile_n) // tile_n


class _Call:
    """One forward of one MoE layer: the edges it runs and their leases (``edges.acquire*``)."""

    def __init__(self, layer: "_Layer", rows: int, local_rows: int):
        self.rows = rows
        self.leases: Dict[str, object] = {}
        self.source = None
        counters = layer.edges & set(_COUNTERS)
        if counters and 1 <= rows <= FGS_MAX_ROWS:
            r = layer.ep * rows
            ws = workspace(layer.device, r, layer.num_local, tile_n_for(r))
            tiles = ws.plan.tile_idx_to_expert_idx.numel()
            for edge in counters:
                words = _counter_words(edge, tiles, layer.ep)
                if words > layer.words[edge]:
                    k3_edges.debug(
                        f"moe.{edge}",
                        rows,
                        f"refused: plan of {tiles} tiles above the declared slab",
                    )
                    continue
                lease = k3_edges.acquire_counters(
                    _COUNTERS[edge], rows, words, window=(1, FGS_MAX_ROWS)
                )
                if lease is not None:
                    self.leases[edge] = lease
        if "comb_norm" in layer.edges and local_rows:
            lease = k3_edges.acquire(COMBINED, local_rows, window=(1, TAIL_FGS_MAX_ROWS))
            if lease is not None:
                self.leases["comb_norm"] = lease
        if "disp_route" in layer.edges and k3_edges.slots() and 1 <= rows <= FGS_MAX_ROWS:
            self.leases["disp_route"] = None

    def on(self, edge: str) -> bool:
        return edge in self.leases

    def slab(self, edge: str) -> torch.Tensor:
        return self.leases[edge].tensor.view(-1)

    def consumed(self, edge: str) -> None:
        lease = self.leases.get(edge)
        if lease is not None:
            k3_edges.consumed(lease)


class _Layer:
    """The expert-path state of one MoE layer: its backend, its CuteA2A (``TLLM_K3_COMM=cute``), its M2 edges and the
    forward in flight between dispatch, run_moe and combine."""

    def __init__(self, routed_experts, edges: frozenset, a2a):
        self.backend = routed_experts.backend
        self.edges = edges
        self.a2a = a2a
        self.num_local = int(self.backend.expert_size_per_partition)
        self.ep = a2a.ep if a2a is not None else 1
        self.call: Optional[_Call] = None
        layer_idx = getattr(routed_experts, "layer_idx", None)
        # A workspace serves every R of its tile_n bucket, so a plan can be as large as the bucket's largest R.
        tiles = max(
            _max_tiles(_bucket_rows(self.ep * rows), self.num_local)
            for rows in range(1, FGS_MAX_ROWS + 1)
        )
        self.words = {edge: _counter_words(edge, tiles, self.ep) for edge in edges & set(_COUNTERS)}
        for edge, words in self.words.items():
            k3_edges.declare_counters(
                _COUNTERS[edge], words, layer_idx, max_rows=FGS_MAX_ROWS, min_rows=1
            )
        for edge, name in (("comb_norm", COMBINED), ("norm_up", LATENT_NORM)):
            if edge in edges:
                k3_edges.declare(name, HIDDEN, layer_idx, max_rows=TAIL_FGS_MAX_ROWS, min_rows=1)

    @property
    def device(self) -> torch.device:
        return self.backend.w3_w1_weight.device

    def stage_kwargs(self, call: Optional[_Call], max_tokens: int) -> Dict[str, dict]:
        kw = {"routing": {"enable_pdl": _pdl()}, "fc1": {}, "fc2": {}, "finalize": {}}
        if call is None:
            return kw
        if call.on("disp_route"):
            kw["routing"]["source"] = call.source
        if call.on("route_fc1"):
            kw["routing"]["ready"] = call.slab("route_fc1")
            if _HAZARD_DELAY_NS:
                kw["routing"]["hazard_delay_ns"] = _HAZARD_DELAY_NS
            kw["fc1"]["plan_ready"] = call.slab("route_fc1")
        if call.on("fc1_fc2"):
            kw["fc1"]["fc1_done"] = kw["fc2"]["fc1_done"] = call.slab("fc1_fc2")
        if call.on("fc2_fin"):
            kw["fc2"]["fc2_done"] = kw["finalize"]["fc2_done"] = call.slab("fc2_fin")
            kw["finalize"]["fc2_expected"] = HIDDEN // 128
        if call.on("fin_push"):
            kw["finalize"]["fin_done"] = call.slab("fin_push")
            kw["finalize"]["max_tokens"] = max_tokens
        return kw

    def run_moe(self, ctx, out: torch.Tensor) -> None:
        call, self.call = self.call, None
        if call is None and self.a2a is None and self.edges:
            rows = max(ctx.all_rank_num_tokens) if ctx.all_rank_num_tokens else ctx.x.shape[0]
            call = _Call(self, rows, 0)
        topk_w = ctx.token_final_scales
        if topk_w.dtype != torch.bfloat16:
            topk_w = topk_w.to(torch.bfloat16)
        max_tokens = ctx.x.shape[0] // self.ep
        run_expert_path(
            ctx.x,
            ctx.x_sf.view(torch.uint8),
            ctx.token_selected_experts,
            topk_w,
            ExpertWeights.from_backend(self.backend),
            out,
            stage_kwargs=self.stage_kwargs(call, max_tokens),
        )
        if call is not None:
            for edge in ("route_fc1", "fc1_fc2", "fc2_fin"):
                call.consumed(edge)
            if call.on("fin_push") or call.on("comb_norm"):
                self.call = call

    def abandon(self) -> None:
        """The stock runner takes this forward: its producers never run, so the edges fall back to M1 (released
        unwritten; the step reset re-arms them)."""
        call, self.call = self.call, None
        if call is not None:
            for edge in call.leases:
                call.consumed(edge)

    def install_comm(self, comm) -> None:
        """Wrap the CuteA2A dispatch / combine that ``moe.comm.install`` put on ``comm`` so a forward on the CuTe
        path carries this layer's edge arguments; anything else goes to those wrappers unchanged."""
        m1_dispatch, m1_combine = comm.dispatch, comm.combine
        a2a = self.a2a

        def engaged(runtime_max: int, kwargs) -> bool:
            return (
                kwargs.get("eplb_local_stats") is None
                and kwargs.get("active_rank_mask") is None
                and comm.use_cft_for_dispatch(runtime_max)
                and comm.use_cft_for_combine(runtime_max)
            )

        def dispatch(
            hidden_states,
            hidden_states_sf,
            token_selected_slots,
            token_final_scales,
            all_rank_num_tokens,
            use_dp_padding=None,
            **kwargs,
        ):
            self.call = None
            runtime_max = max(all_rank_num_tokens)
            if not engaged(runtime_max, kwargs):
                return m1_dispatch(
                    hidden_states,
                    hidden_states_sf,
                    token_selected_slots,
                    token_final_scales,
                    all_rank_num_tokens,
                    use_dp_padding,
                    **kwargs,
                )
            call = _Call(self, runtime_max, hidden_states.shape[0])
            out = a2a.dispatch(
                hidden_states,
                hidden_states_sf,
                token_selected_slots,
                token_final_scales,
                all_rank_num_tokens,
                defer_data_wait=call.on("disp_route"),
            )
            if call.on("disp_route"):
                call.source = a2a.routing_source()
            self.call = call
            return out

        def combine(final_hidden_states, **kwargs):
            call, self.call = self.call, None
            if call is None or not (call.on("fin_push") or call.on("comb_norm")):
                return m1_combine(final_hidden_states, **kwargs)
            fin_kw = {}
            if call.on("fin_push"):
                rows = final_hidden_states.numel() // HIDDEN
                fin_kw = {
                    "fin_done": call.slab("fin_push"),
                    "fin_expected": call.rows * finalize_chunks(rows),
                }
            out = call.leases["comb_norm"].tensor if call.on("comb_norm") else None
            if _HAZARD_DELAY_NS:
                fin_kw["hazard_delay_ns"] = _HAZARD_DELAY_NS
            y = a2a.combine(final_hidden_states, out=out, **fin_kw)
            call.consumed("fin_push")
            if out is not None:
                _pending_tail.lease = call.leases["comb_norm"]
            return y

        comm.dispatch, comm.combine = dispatch, combine


def finalize_chunks(rows: int) -> int:
    from .moe import finalize

    return finalize.default_chunks(rows)


_pending_tail = threading.local()


def _check_weights(backend) -> None:
    """The in-place weight layout the stage kernels read; checked at the first call because the backend creates its
    weights after the model is constructed."""
    w31, w31_s = backend.w3_w1_weight, backend.w3_w1_weight_scale
    w2, w2_s = backend.w2_weight, backend.w2_weight_scale
    expected = (
        (w31, torch.uint8, (2 * INTER, HIDDEN // 2)),
        (w31_s, torch.uint8, (2 * INTER, HIDDEN // SF_VEC)),
        (w2, torch.uint8, (HIDDEN, INTER // 2)),
        (w2_s, torch.uint8, (HIDDEN, INTER // SF_VEC)),
    )
    for t, dtype, shape in expected:
        if t.dtype != dtype or tuple(t.shape[1:]) != shape:
            raise ValueError(
                f"{ENV}=cute needs the TRTLLM-Gen MXFP4 x MXFP8 expert buffers [E, {shape[0]}, {shape[1]}] "
                f"{dtype}; got {tuple(t.shape)} {t.dtype} on {type(backend).__name__}"
            )


def install(routed_experts) -> bool:
    """Opt-in switches on one MoE layer, each default off:

    - ``TLLM_K3_MOE=cute``: the routed experts' TRTLLM-Gen W4A8 MXFP4 x MXFP8 ``run_moe`` runs
      :func:`run_expert_path`; calls the path cannot take (no combine payload buffer, unfinalized output, internal
      routing) keep the stock runner.
    - ``TLLM_K3_COMM=cute``: dispatch / combine through ``moe.comm.CuteA2A``.
    - ``TLLM_K3_MOE_FGS=<edges>|all``: the M2 edges; each needs both of its kernel sides and the stages it connects.
    """
    edges = fgs_edges()
    moe_on, comm_on, tail_on = enabled(), os.environ.get(COMM_ENV, "") == "cute", tail_enabled()
    if not (moe_on or comm_on or edges):
        return False
    from .moe import comm as k3_comm

    missing = _missing_sides(edges, moe_on, comm_on, tail_on)
    if missing:
        raise ValueError(f"{FGS_ENV}: cannot run {'; '.join(missing)}")
    if edges and not k3_edges.slots():
        raise ValueError(f"{FGS_ENV} needs edge slots ({k3_edges.ENV}=chain|chain_after|lamport)")
    backend = routed_experts.backend
    if moe_on and not hasattr(getattr(backend, "activation", None), "gate_softcap"):
        raise ValueError(f"{ENV}=cute needs the SiTU experts, got {type(backend).__name__}")
    comm = getattr(routed_experts, "comm", None)
    a2a = k3_comm.install(comm) if comm_on and comm is not None else None
    if a2a is None and edges & _COMM_EDGES:
        raise ValueError(
            f"{FGS_ENV}: {sorted(edges & _COMM_EDGES)} need the CuTe all-to-all (attention DP)"
        )
    layer = _Layer(routed_experts, edges, a2a)
    if a2a is not None and edges & _COMM_EDGES:
        layer.install_comm(comm)
    if not moe_on:
        return True
    stock_run_moe = backend.run_moe
    checked = []

    def run_moe(ctx, *, workspace=None):
        plan = ctx.comm_plan
        out = None if plan is None else plan.moe_output
        if (
            out is None
            or not ctx.do_finalize
            or ctx.token_selected_experts is None
            or ctx.x_sf is None
        ):
            layer.abandon()
            return stock_run_moe(ctx, workspace=workspace)
        if not checked:
            _check_weights(backend)
            checked.append(True)
        if ctx.x_ready_event is not None:
            ctx.x_ready_event.wait()
        layer.run_moe(ctx, out)
        return out

    backend.run_moe = run_moe
    return True


def run_tail(y: torch.Tensor, norm, up_proj, shared: torch.Tensor) -> torch.Tensor:
    """Post-combine latent tail (``TLLM_K3_MOE_TAIL=cute``): ``up_proj(norm(y)) + shared`` with the routed + shared
    add fused into the up-projection epilogue. ``y`` is the combine output [M, 3584]; it runs after the shared-expert
    join because ``shared`` is read in that epilogue. With ``comb_norm`` the norm polls the combine's slot; with
    ``norm_up`` it writes a slot the up-projection consumes as it lands."""
    from .moe import latent

    combined = getattr(_pending_tail, "lease", None)
    _pending_tail.lease = None
    m = y.shape[0]
    norm_lease = None
    if "norm_up" in fgs_edges() and m:
        norm_lease = k3_edges.acquire(LATENT_NORM, m, window=(1, TAIL_FGS_MAX_ROWS))
    if combined is None and norm_lease is None:
        norm_out = torch.empty_like(y)
        out = torch.empty_like(shared)
        return latent.run(
            y, norm.weight, float(norm.variance_epsilon), up_proj.weight, shared, out, norm_out
        )
    consumer = combined is not None and k3_edges.lease_containing(y) is combined
    norm_out = norm_lease.tensor if norm_lease is not None else torch.empty_like(y)
    latent.rmsnorm(
        y,
        norm.weight,
        float(norm.variance_epsilon),
        norm_out,
        fgs_consumer=consumer,
        fgs_producer=norm_lease is not None,
        fgs_counters=_fgs_counters(y.device) if consumer else None,
    )
    if combined is not None:
        k3_edges.consumed(combined)
    out = torch.empty_like(shared)
    latent.up_proj(norm_out, up_proj.weight, out, shared, fgs_consumer=norm_lease is not None)
    if norm_lease is not None:
        k3_edges.consumed(norm_lease)
    return out


_counters: Dict[torch.device, torch.Tensor] = {}


def _fgs_counters(device: torch.device) -> torch.Tensor:
    """int32[4] statistics of the sentinel consumers (waited warps, re-reads, fallbacks), never reset."""
    if device not in _counters:
        _counters[device] = torch.zeros(4, dtype=torch.int32, device=device)
    return _counters[device]
