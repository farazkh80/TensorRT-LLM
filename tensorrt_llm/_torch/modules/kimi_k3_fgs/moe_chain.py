# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode MoE expert path in CuTe DSL: routing -> FC1 -> FC2 -> finalize, the CuTe
all-to-all, the post-combine latent tail and the fine-grained-sync edges between them (the recipe's ``moe.*``
edges).

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
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch

from ..multi_stream_utils import do_multi_stream
from . import edges as k3_edges
from . import moe_streams
from . import recipe as k3_recipe

HIDDEN = 3584
INTER = 3072
TOP_K = 16
NUM_EXPERTS = 896
SF_VEC = 32
# The MoE edges a recipe can list (``recipe.py`` validates ``moe.<edge>`` names against EDGES + V2_EDGES).
EDGES = ("fc1_fc2",)
_COMM_EDGES = frozenset(("recv_norm",))
# FC1 -> FC2 (Lamport): FC1 writes act / act_sf into the FC1_ACT / FC1_SF slots (armed 0x80 / 0xFF) by TMA store,
# FC2 checks each act chunk with a TMA report_valid load, and the edge workspaces' reset stream re-arms the rows FC2
# reports.
FC1_ACT = "moe.fc1_act"
FC1_SF = "moe.fc1_sf"
# Rows per rank the expert-path edges serve (the fc1_fc2 / recv_norm windows end at 128); the tail edges stop where
# the split-K up-projection consumer does.
FGS_MAX_ROWS = 128
TAIL_FGS_MAX_ROWS = 32
# Forwards with more tokens on the fullest rank, such as context steps, keep the stock kernels.
MAX_TOKENS_PER_RANK = 128
# The fused router serves up to this many tokens per rank; the stock gate GEMM runs above.
ROUTER_MAX_ROWS = 64


V2_EDGES = ("mix_latent", "recv_norm")
# Edges whose producer is the pre-MoE mix: it writes the MoE input into a -0-armed slot (``mix_out``).
_MIX_EDGES = frozenset(("mix_latent",))
MIX_X = "moe.x"
MODEL_HIDDEN = 7168
MIX_MAX_ROWS = 64


@dataclass(frozen=True)
class EdgeSync:
    """One edge's sync object as a fused kernel sees it: ``kind`` ``slot`` (the -0-armed mix slot the consumer reads)
    or ``lamport`` (the FC1 -> FC2 slots; ``tensor`` is the int32 [1] word FC2 stores its padded rows into)."""

    edge: str
    kind: str
    tensor: Optional[torch.Tensor]


@dataclass(frozen=True)
class StageSync:
    """The edges one call of a fused kernel consumes (``waits``) and produces (``publishes``); empty maps or no
    ``StageSync`` at all is the FUSED-OFF build."""

    waits: Mapping[str, EdgeSync]
    publishes: Mapping[str, EdgeSync]


def _resolve(entry: str):
    """The function behind ``module:attr[.attr]`` in ``kimi_k3_fgs.moe``, or None when it does not exist yet."""
    module, _, attr = entry.partition(":")
    try:
        obj = importlib.import_module(f"{__package__}.moe.{module}")
    except ImportError:
        return None
    for part in attr.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


@dataclass
class Fc1Slots:
    """The Lamport FC1 -> FC2 slot pair of one MoE forward: ``act`` / ``act_sf`` (rows past the workspace's are never
    touched) and the int32 [1] word FC2 stores its padded rows into."""

    act: torch.Tensor
    act_sf: torch.Tensor
    rows_used: torch.Tensor


def _max_tokens_per_rank() -> int:
    return MAX_TOKENS_PER_RANK


def within_max_rows(tokens_per_rank: int) -> bool:
    """Whether a forward with this many tokens on its fullest rank runs the CuTe path (expert path, all-to-all and
    tail); larger forwards, such as context steps, keep the stock kernels and never grow the workspace."""
    return tokens_per_rank <= _max_tokens_per_rank()


def run_shared(runtime, x: torch.Tensor) -> Optional[torch.Tensor]:
    """The shared expert as the CuTe gate/up + down pair (``shared_expert.config``) up to its row cap, or None for
    the stock module. The down GEMM still waits for the shared-down event ``moe_streams.arm_shared_down`` armed on
    ``down_proj``, which this path bypasses. The kernels share per-device scratch: one stream at a time per device."""
    from .moe import shared_expert

    mlp = runtime.shared_experts
    gate_up, down = mlp.gate_up_proj, mlp.down_proj
    m = x.shape[0]
    cap = shared_expert.config(m)
    why = (
        "above the CuTe shared expert's rows"
        if cap is None
        else "tensor-parallel shared expert"
        if gate_up.tp_size != 1 or down.tp_size != 1
        else f"{gate_up.weight.dtype} / {down.weight.dtype} weights"
        if gate_up.weight.dtype != torch.bfloat16 or down.weight.dtype != torch.bfloat16
        else ""
    )
    if why:
        return None
    # The scratch must exist before graph capture; the model may be built on the meta device, so size it at the
    # eager warm-up forwards rather than at install.
    if not torch.cuda.is_current_stream_capturing():
        shared_expert.reserve(x.device, cap)
    return shared_expert.run(
        x.contiguous(),
        gate_up.weight,
        down.weight,
        ctas=cap,
        before_down=lambda: moe_streams.wait_shared_down(down),
    )


def tail_enabled() -> bool:
    """The post-combine latent tail runs on the CuTe path (fused combine reduce + norm, split latent up)."""
    return True


def _tile_bounds(num_experts: int) -> tuple:
    # The tile follows the mean rows per expert, R * TOP_K / num_experts: R <= 512 / 1024 at 896 experts.
    return 512 * num_experts // NUM_EXPERTS, 1024 * num_experts // NUM_EXPERTS


def _bucket_rows(rows: int, num_experts: int, max_rows: int) -> int:
    """The largest R up to ``max_rows`` that ``tile_n_for`` maps to the same tile_n as ``rows``."""
    r8, r16 = _tile_bounds(num_experts)
    return max(rows, min(r8 if rows <= r8 else r16 if rows <= r16 else max_rows, max_rows))


def tile_n_for(rows: int, num_experts: int = NUM_EXPERTS) -> int:
    """Token rows per FC tile (the N of the swap-AB MMA) for R received rows routed over ``num_experts``."""
    r8, r16 = _tile_bounds(num_experts)
    return 8 if rows <= r8 else 16 if rows <= r16 else 32


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


def workspace(
    device: torch.device, rows: int, num_local: int, tile_n: int, alloc_rows: int = 0
) -> Workspace:
    """The workspace serving ``rows``; a new one is sized for ``max(rows, alloc_rows)`` rows. A reallocation frees the
    buffers that CUDA graphs captured earlier still use, so callers size the first allocation for the largest R they
    can request with this tile_n."""
    key = (device, num_local, tile_n)
    ws = _workspaces.get(key)
    if ws is None or ws.max_rows < rows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"CuTe MoE path: no workspace for {rows} rows (tile_n {tile_n}) before graph capture; "
                "run the batch eagerly first"
            )
        ws = _workspaces[key] = Workspace(device, max(rows, alloc_rows), num_local, tile_n)
    return ws


def _stage(name: str) -> Callable:
    if name == "fc2_v2":
        return _resolve("fc2:run_v2")
    return importlib.import_module(f"{__package__}.moe.{name}").run


def expert_stages(tokens_per_rank: Optional[int] = None) -> tuple:
    """The expert-path stages: routing, FC1, and FC2 with the finalize fused (``fc2.run_v2``)."""
    return ("routing", "fc1", "fc2_v2")


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
    tokens_per_rank: Optional[int] = None,
    act_bufs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Workspace:
    """Run routing -> FC1 -> FC2 + finalize on the current stream; writes ``out`` [R, 3584] bf16.

    ``stages`` overrides any of the stages (tests plug reference or candidate kernels in per stage); the rest come
    from ``kimi_k3_fgs.moe``. Stage signatures:
      routing(topk_ids, local_expert_offset, num_local, plan)
      fc1(x, x_sf, weights, plan, act, act_sf)
      fc2_v2(act, act_sf, weights, plan, topk_w, out)            FC2 with the finalize fused
    ``stage_kwargs`` adds keyword arguments per stage (the FGS edge builds). ``act_bufs`` replaces the workspace's
    ``act`` / ``act_sf`` (the Lamport FC1 -> FC2 slots).
    """
    rows = x.shape[0]
    assert x.shape == (rows, HIDDEN) and x.dtype == torch.float8_e4m3fn, (x.shape, x.dtype)
    assert x_sf.shape == (rows, HIDDEN // SF_VEC) and x_sf.dtype == torch.uint8, x_sf.dtype
    assert topk_ids.shape == (rows, TOP_K) and topk_ids.dtype == torch.int32
    assert topk_w.shape == (rows, TOP_K) and topk_w.dtype == torch.bfloat16
    assert out.shape == (rows, HIDDEN) and out.dtype == torch.bfloat16 and out.is_contiguous()
    stages = stages or {}
    kw = stage_kwargs or {}
    names = expert_stages(tokens_per_rank)
    run = {name: stages.get(name) or _stage(name) for name in names}
    if ws is None:
        ws = workspace(x.device, rows, weights.num_local, tile_n_for(rows))
    plan = ws.plan
    act, act_sf = act_bufs if act_bufs is not None else (ws.act, ws.act_sf)
    run["routing"](
        topk_ids, weights.local_expert_offset, weights.num_local, plan, **kw.get("routing", {})
    )
    run["fc1"](x, x_sf, weights, plan, act, act_sf, **kw.get("fc1", {}))
    run["fc2_v2"](act, act_sf, weights, plan, topk_w, out, y=ws.y, **kw.get("fc2", {}))
    return ws


def _pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def fgs_edges() -> frozenset:
    """The MoE FGS edges the recipe lists (``recipe.moe_edge_names``); none without one."""
    return k3_recipe.moe_edge_names() or frozenset()


def configure_windows(windows: Mapping[str, Tuple[int, int]]) -> None:
    """Size the mix slot for the recipe's MoE edge spans (``recipe.install``)."""
    global MIX_MAX_ROWS
    for edge, (_, high) in windows.items():
        if edge in _MIX_EDGES:
            MIX_MAX_ROWS = max(MIX_MAX_ROWS, high)


def _edge_rows(edge: str) -> tuple:
    return k3_recipe.span((f"moe.{edge}",)) or (1, 0)


class _Call:
    """One forward of one MoE layer: the edges it runs and their leases (``edges.acquire*``)."""

    def __init__(self, layer: "_Layer", rows: int):
        self.rows = rows
        self.leases: Dict[str, object] = {}
        self.fc1: Optional[Fc1Slots] = None
        if layer.fc1_fc2_lamport and 1 <= rows <= FGS_MAX_ROWS:
            self._acquire_fc1_slots(layer, rows)

    def _acquire_fc1_slots(self, layer: "_Layer", rows: int) -> None:
        low, high = _edge_rows("fc1_fc2")
        if not (low <= rows <= high and k3_recipe.moe_on("fc1_fc2", rows)):
            return
        p = layer.workspace(layer.ep * rows).act.shape[0]
        act = k3_edges.acquire_bytes(FC1_ACT, rows, p, window=(1, FGS_MAX_ROWS))
        sf = (
            None
            if act is None
            else k3_edges.acquire_bytes(FC1_SF, rows, p, window=(1, FGS_MAX_ROWS))
        )
        if sf is None:
            if act is not None:
                k3_edges.consumed(act)
            return
        if act.slot != sf.slot:
            raise RuntimeError(f"{FC1_ACT} slot {act.slot} and {FC1_SF} slot {sf.slot} must pair")
        self.leases["fc1_fc2"] = act
        self.leases["fc1_sf"] = sf
        self.fc1 = Fc1Slots(act.tensor.view(torch.float8_e4m3fn), sf.tensor, act.bound)

    def fc1_slots(self, rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The Lamport FC1 -> FC2 slots as the workspace's ``rows``-row ``act`` / ``act_sf``."""
        act, sf = self.fc1.act, self.fc1.act_sf
        if act.shape[0] < rows:
            raise RuntimeError(f"{FC1_ACT}: a {act.shape[0]}-row slot for a {rows}-row workspace")
        return act[:rows], sf[:rows]

    def on(self, edge: str) -> bool:
        return edge in self.leases

    def consumed(self, edge: str) -> None:
        lease = self.leases.get(edge)
        if lease is not None:
            k3_edges.consumed(lease)


class _Layer:
    """The expert-path state of one MoE layer: its backend, its CuteA2A, its FGS edges and the
    forward in flight between dispatch, run_moe and combine."""

    def __init__(self, routed_experts, edges: frozenset, a2a):
        self.backend = routed_experts.backend
        self.edges = edges
        self.a2a = a2a
        self.num_local = int(self.backend.expert_size_per_partition)
        self.num_experts = int(self.backend.num_slots)
        # Received rows R are ep * tokens on the fullest rank, at most the cap on the CuTe path; the stock
        # all-to-all delivers the same R, so the plan is sized from its ep too.
        comm_ep = getattr(getattr(routed_experts, "comm", None), "ep_size", 1)
        self.ep = a2a.ep if a2a is not None else comm_ep
        self.max_rows = comm_ep * _max_tokens_per_rank()
        self.call: Optional[_Call] = None
        # (ids, weights, x, x_sf) of the forward ``entry`` ran, until the scheduler takes it; None: stock entry.
        self.pending: Optional[tuple] = None
        # Benchmark-only hook, inert unless the layer-wise benchmark sets ``routed_experts._k3_balanced_routing``
        # (Balanced ids computed in the kernel) or ``_k3_routing_table`` (``(rows, device) -> int32 [rows, 16]``, the
        # benchmark's hot ids): the fused router then writes those ids into the pending buffers, which the benchmark
        # keeps.
        self.balanced_ids = False
        self.entry_kernels: Tuple[str, ...] = ()
        layer_idx = getattr(routed_experts, "layer_idx", None)
        if edges & _MIX_EDGES:
            k3_edges.declare(MIX_X, MODEL_HIDDEN, layer_idx, max_rows=MIX_MAX_ROWS, min_rows=1)
        self.fc1_fc2_lamport = "fc1_fc2" in edges
        if self.fc1_fc2_lamport:
            # The workspace's act rows for any R the edge serves (the tile_n bucket's largest R).
            slot_rows = max(
                max_permuted(
                    _bucket_rows(self.ep * rows, self.num_experts, self.max_rows),
                    self.num_local,
                    tile_n_for(self.ep * rows, self.num_experts),
                )
                for rows in range(1, FGS_MAX_ROWS + 1)
            )
            # The framework arms a 1-byte edge by dtype: the E4M3 act slots with -0, the uint8 scale slots with 0xFF.
            for name, width, dtype, shared in (
                (FC1_ACT, INTER, torch.float8_e4m3fn, None),
                (FC1_SF, INTER // SF_VEC, torch.uint8, FC1_ACT),
            ):
                k3_edges.declare_bytes(
                    name,
                    width,
                    dtype,
                    slot_rows,
                    layer_idx,
                    max_rows=FGS_MAX_ROWS,
                    policy=None,
                    bounded=shared is None,
                    bounds_from=shared,
                )
        self.tail_fused = a2a is not None
        self.inbox = "recv_norm" in edges and self.tail_fused
        self.entry_events = None
        # The routed-expert latent RMSNorm, for the fused combine; set by ``entry`` at every MoE forward.
        self.tail_norm = None

    @property
    def device(self) -> torch.device:
        return self.backend.w3_w1_weight.device

    def workspace(self, rows: int) -> Workspace:
        return workspace(
            self.device,
            rows,
            self.num_local,
            tile_n_for(rows, self.num_experts),
            _bucket_rows(rows, self.num_experts, self.max_rows),
        )

    def stage_kwargs(self, call: Optional[_Call]) -> Dict[str, dict]:
        """Keyword arguments of the expert-path stages: FC2 (finalize fused) takes ``fgs=StageSync`` and
        ``enable_pdl``; with ``fc1_fc2`` FC1 runs its Lamport build and FC2 waits on the slots."""
        kw = {"routing": {"enable_pdl": _pdl()}, "fc1": {}, "fc2": {"enable_pdl": _pdl()}}
        sync = StageSync({}, {})
        if call is not None and call.on("fc1_fc2") and self.fc1_fc2_lamport:
            kw["fc1"]["lamport"] = True
            sync.waits["fc1_fc2"] = EdgeSync("fc1_fc2", "lamport", call.fc1.rows_used)
        if sync.waits or sync.publishes:
            kw["fc2"]["fgs"] = sync
        return kw

    def run_moe(self, ctx, out: torch.Tensor) -> None:
        call, self.call = self.call, None
        rows = max(ctx.all_rank_num_tokens) if ctx.all_rank_num_tokens else ctx.x.shape[0]
        if call is None and self.a2a is None and self.edges:
            call = _Call(self, rows)
        topk_w = ctx.token_final_scales
        if topk_w.dtype != torch.bfloat16:
            topk_w = topk_w.to(torch.bfloat16)
        ws = self.workspace(ctx.x.shape[0])
        run_expert_path(
            ctx.x,
            ctx.x_sf.view(torch.uint8),
            ctx.token_selected_experts,
            topk_w,
            ExpertWeights.from_backend(self.backend),
            out,
            ws=ws,
            stage_kwargs=self.stage_kwargs(call),
            tokens_per_rank=rows,
            act_bufs=(
                call.fc1_slots(ws.act.shape[0])
                if call is not None and call.fc1 is not None
                else None
            ),
        )
        if call is not None:
            for edge in ("fc1_fc2", "fc1_sf"):
                call.consumed(edge)
            if self.tail_fused:
                self.call = call

    def _combine_norm(self, call: _Call, final_hidden_states: torch.Tensor) -> torch.Tensor:
        """Push, then reduce the received rows and apply the latent RMSNorm in one kernel; ``run_tail`` takes the
        normed rows."""
        from .moe import tail

        norm = self.tail_norm
        inbox_kw = {}
        # The inbox changes what peers push, so every rank decides it on the rank-consistent row count; the
        # recv_norm edge only decides how the reduce waits for it.
        inbox = self.inbox and call.rows <= self.a2a.lamport_rows
        if inbox:
            inbox_kw = {"inbox": "lamport"}
        out = tail.combine_norm(
            self.a2a,
            final_hidden_states,
            norm.weight,
            float(norm.variance_epsilon),
            **inbox_kw,
        )
        _pending_tail.fused = out.data_ptr()
        return out

    def abandon(self) -> None:
        """The stock runner takes this forward: its producers never run, so its edges stay off (released
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
                and within_max_rows(runtime_max)
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
            call = _Call(self, runtime_max)
            out = a2a.dispatch(
                hidden_states,
                hidden_states_sf,
                token_selected_slots,
                token_final_scales,
                all_rank_num_tokens,
            )
            self.call = call
            return out

        def combine(final_hidden_states, **kwargs):
            call, self.call = self.call, None
            if call is not None and self.tail_fused and self.tail_norm is not None:
                return self._combine_norm(call, final_hidden_states)
            return m1_combine(final_hidden_states, **kwargs)

        comm.dispatch, comm.combine = dispatch, combine


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
                f"CuTe MoE path needs the TRTLLM-Gen MXFP4 x MXFP8 expert buffers [E, {shape[0]}, {shape[1]}] "
                f"{dtype}; got {tuple(t.shape)} {t.dtype} on {type(backend).__name__}"
            )


def install(routed_experts) -> bool:
    """Install the CuTe expert path on one MoE layer: the routed experts' TRTLLM-Gen W4A8 MXFP4 x MXFP8 ``run_moe``
    runs :func:`run_expert_path` (calls the path cannot take keep the stock runner: no combine payload buffer,
    unfinalized output, internal routing), dispatch / combine go through ``moe.comm.CuteA2A``, and the recipe's MoE
    edges (``fgs_edges``) run.

    Forwards with more than ``MAX_TOKENS_PER_RANK`` tokens on the fullest rank keep the stock path.
    """
    edges = fgs_edges()
    unknown = edges - set(EDGES) - set(V2_EDGES)
    if unknown:
        raise ValueError(
            f"MoE FGS edges: unknown {sorted(unknown)}; edges are {', '.join(EDGES + V2_EDGES)}"
        )
    if edges and not k3_edges.slots():
        raise ValueError(
            f"MoE FGS edges need edge slots ({k3_edges.ENV}=chain|chain_after|lamport)"
        )
    from .moe import comm as k3_comm

    backend = routed_experts.backend
    if not hasattr(getattr(backend, "activation", None), "gate_softcap"):
        raise ValueError(f"CuTe MoE path needs the SiTU experts, got {type(backend).__name__}")
    comm = getattr(routed_experts, "comm", None)
    # Only an NVLinkOneSided strategy with counted writes has the CFT workspace CuteA2A runs on; other strategies
    # (e.g. the allgather / reduce-scatter of a prefill model) keep their stock comm and the layer drops its comm edges.
    cft = comm is not None and getattr(comm, "can_use_cft_counted_writes", False)
    lamport_rows = _max_tokens_per_rank() if "recv_norm" in edges else 0
    a2a = k3_comm.install(comm, lamport_rows=lamport_rows) if cft else None
    if a2a is None and edges & _COMM_EDGES:
        edges = edges - _COMM_EDGES
    layer = _Layer(routed_experts, edges, a2a)
    # The forward's edge leases are taken at the CuTe dispatch, so it is wrapped for any edge, not only the comm ones.
    if a2a is not None:
        layer.install_comm(comm)
    _install_entry(routed_experts, layer)
    stock_run_moe = backend.run_moe
    checked = []
    ep = getattr(comm, "ep_size", 1)

    def run_moe(ctx, *, workspace=None):
        if layer.pending is not None:
            raise RuntimeError(
                "CuTe MoE path: the scheduler skipped the fused router / latent-down result"
            )
        plan = ctx.comm_plan
        out = None if plan is None else plan.moe_output
        if ctx.all_rank_num_tokens:
            tokens_per_rank = max(ctx.all_rank_num_tokens)
        else:
            tokens_per_rank = ctx.x.shape[0] // ep
        if (
            out is None
            or not ctx.do_finalize
            or ctx.token_selected_experts is None
            or ctx.x_sf is None
            or not within_max_rows(tokens_per_rank)
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


class _EntryBuffers:
    """Outputs of the fused entry kernels for up to ``max_rows`` tokens, shared by the layers (stream order)."""

    def __init__(self, device: torch.device, max_rows: int, kernels: Tuple[str, ...]):
        self.max_rows = max_rows
        self.scratch = {}
        for name in kernels:
            spec_fn = _resolve(f"{name}:workspace_spec")
            spec = spec_fn(max_rows) if spec_fn is not None else {}
            self.scratch[name] = {
                key: torch.zeros(shape, dtype=dtype, device=device)
                for key, (shape, dtype) in spec.items()
            }
        self.ids = torch.empty(max_rows, TOP_K, dtype=torch.int32, device=device)
        self.weights = torch.empty(max_rows, TOP_K, dtype=torch.bfloat16, device=device)
        self.x = torch.empty(max_rows, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
        self.x_sf = torch.empty(max_rows, HIDDEN // SF_VEC, dtype=torch.uint8, device=device)


_entry_buffers: Dict[torch.device, _EntryBuffers] = {}


def _entry_bufs(device: torch.device, kernels: Tuple[str, ...]) -> _EntryBuffers:
    bufs = _entry_buffers.get(device)
    if bufs is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("fused MoE entry: buffers must be allocated before graph capture")
        bufs = _entry_buffers[device] = _EntryBuffers(device, _max_tokens_per_rank(), kernels)
    return bufs


def _install_entry(routed_experts, layer: _Layer) -> None:
    """The routed experts take the fused entry kernels' results (router, latent down + MXFP8 quantize) through the
    backend's fused route + quantize hook, the scheduler's only per-forward way in ahead of the dispatch; ``entry``
    sets them per forward."""
    routed_experts._k3_moe_layer = layer
    kernels = ("router", "latent_down")
    comm = getattr(routed_experts, "comm", None)
    using_lb = getattr(routed_experts, "_using_load_balancer", lambda: False)()
    if (
        using_lb
        or getattr(routed_experts, "apply_router_weight_on_input", False)
        or (comm is not None and not comm.supports_post_quant_dispatch())
    ):
        raise ValueError(
            f"fused MoE entry ({', '.join(kernels)}) needs quantize-then-dispatch without EPLB or "
            "router weights on the input"
        )
    layer.entry_kernels = kernels
    backend = routed_experts.backend
    stock = backend.try_fused_route_quant

    def try_fused_route_quant(x, router_logits):
        pending, layer.pending = layer.pending, None
        if pending is None:
            return stock(x, router_logits)
        ids, weights, xq, x_sf = pending
        if ids is None:
            ids, weights = routed_experts.routing_method.apply(router_logits, None)
        if xq is None:
            xq, x_sf = backend.quantize_input(x)
        return ids, weights, xq, x_sf

    # A caller that forces the separated path (a routing override turning the hook off) routes and quantizes in two
    # calls; each takes its half of the pending result.
    routing = routed_experts.routing_method
    stock_apply, stock_quantize = routing.apply, backend.quantize_input

    def apply(router_logits, input_ids=None):
        pending = layer.pending
        if pending is None or pending[0] is None:
            return stock_apply(router_logits, input_ids)
        layer.pending = None if pending[2] is None else (None, None) + pending[2:]
        return pending[0], pending[1]

    def quantize_input(x, *args, **kwargs):
        pending = layer.pending
        if pending is None or pending[2] is None:
            return stock_quantize(x, *args, **kwargs)
        layer.pending = None if pending[0] is None else pending[:2] + (None, None)
        return pending[2], pending[3]

    backend.try_fused_route_quant = try_fused_route_quant
    routing.apply = apply
    backend.quantize_input = quantize_input


def entry(runtime, hidden_states: torch.Tensor, all_rank_num_tokens) -> Optional[tuple]:
    """The MoE block's entry: runs the fused router (the stock gate above ``ROUTER_MAX_ROWS`` tokens) and the fused
    latent down + MXFP8 quantize, and returns ``(routed_in, router_logits)`` for ``runtime.routed_experts``; None
    keeps the stock entry. The fused outputs reach the scheduler through the hook ``_install_entry`` put on the
    backend; ``routed_in`` and ``router_logits`` then only carry the token count (``hidden_states`` and the expert ids
    stand in for the latent rows and the logits).

    It engages only where the CuTe expert path does: at most ``MAX_TOKENS_PER_RANK`` tokens on the fullest rank,
    at least one token on every rank (no DP padding) and a single MoE chunk.
    """
    experts = runtime.routed_experts
    layer = getattr(experts, "_k3_moe_layer", None)
    if layer is None:
        return None
    layer.tail_norm = runtime.routed_expert_norm
    layer.balanced_ids = False
    if not layer.entry_kernels or torch.compiler.is_compiling():
        return None
    m = hidden_states.shape[0]
    counts = list(all_rank_num_tokens) if all_rank_num_tokens else [m]
    if (
        min(counts) == 0
        or not within_max_rows(max(counts))
        or experts.calculate_num_chunks([max(counts)] * len(counts)) != 1
    ):
        return None
    gate = runtime.gate
    stock_why = (
        f"above {_router_max_rows()} rows"
        if m > _router_max_rows()
        else f"{gate.weight.dtype} gate weight"
        if gate.weight.dtype != torch.float32
        else f"{gate.weight.shape[0]} experts, the kernel serves {_resolve('router:NUM_EXPERTS')}"
        if gate.weight.shape[0] != _resolve("router:NUM_EXPERTS")
        else ""
    )
    fused_router = not stock_why
    bufs = _entry_bufs(hidden_states.device, layer.entry_kernels)
    ids = weights = None
    latent_sync = _mix_sync(layer, "mix_latent", hidden_states)
    balanced = hot_ids = None
    if fused_router:
        ids, weights = bufs.ids[:m], bufs.weights[:m]
        balanced = getattr(experts, "_k3_balanced_routing", None)
        table = getattr(experts, "_k3_routing_table", None)
        if table is not None:
            balanced = None
            hot_ids = table(m, hidden_states.device)
    xq, x_sf = bufs.x[:m], bufs.x_sf[:m]

    def run_router():
        if not fused_router:
            return gate.compute_logits(hidden_states)
        _resolve("router:run")(
            hidden_states,
            gate.weight,
            gate.e_score_correction_bias,
            ids,
            weights,
            routed_scaling_factor=float(gate.routed_scaling_factor),
            enable_pdl=_pdl(),
            balanced=balanced,
            hot_ids=hot_ids,
        )
        return ids

    def run_latent():
        _resolve("latent_down:run")(
            hidden_states,
            runtime.routed_expert_down_proj.weight,
            xq,
            x_sf,
            fgs=latent_sync,
            enable_pdl=_pdl(),
            workspace=bufs.scratch["latent_down"] or None,
        )
        return hidden_states

    order = moe_streams.entry_order(m)
    if order == "branches" and (do_multi_stream() or torch.cuda.is_current_stream_capturing()):
        if layer.entry_events is None:
            layer.entry_events = tuple(torch.cuda.Event() for _ in range(3))
        router_logits, routed_in = moe_streams.run_entry_branches(
            hidden_states.device, layer.entry_events, run_router, run_latent
        )
    else:
        router_logits = run_router()
        routed_in = run_latent()
    layer.pending = (ids, weights, xq, x_sf)
    layer.balanced_ids = balanced is not None or hot_ids is not None
    return routed_in, router_logits


def _router_max_rows() -> int:
    return ROUTER_MAX_ROWS


def _mix_sync(layer: _Layer, edge: str, hidden_states: torch.Tensor) -> Optional[StageSync]:
    """The ``mix_latent`` consumer side: the MoE input itself, when it lives in the ``mix_out`` slot and the edge is
    on."""
    if (
        edge not in layer.edges
        or not k3_recipe.moe_on(edge, hidden_states.shape[0])
        or k3_edges.lease_containing(hidden_states) is None
    ):
        return None
    return StageSync({edge: EdgeSync(edge, "slot", hidden_states)}, {})


def mix_out(runtime, rows: int):
    """The slot the pre-MoE mix writes the MoE input into when ``mix_latent`` is on, or None for its plain buffer.
    The mix stores it as an FGS producer (no -0, dependents launched at entry); the slot is released once the MoE
    block has joined its streams (``mix_done``), which re-arms it on the reset stream."""
    layer = getattr(getattr(runtime, "routed_experts", None), "_k3_moe_layer", None)
    if layer is None or not any(k3_recipe.moe_on(e, rows) for e in layer.edges & _MIX_EDGES):
        return None
    lease = k3_edges.acquire(MIX_X, rows, window=(1, MIX_MAX_ROWS))
    return lease


def mix_done(hidden_states: torch.Tensor) -> None:
    """Release the ``mix_out`` slot ``hidden_states`` lives in, after the last of its readers was launched."""
    lease = k3_edges.lease_containing(hidden_states)
    if lease is not None:
        k3_edges.consumed(lease)


def run_tail(y: torch.Tensor, norm, up_proj, shared: torch.Tensor) -> torch.Tensor:
    """Post-combine latent tail: ``up_proj(norm(y)) + shared`` with the routed + shared add fused into the
    up-projection epilogue. ``y`` is the combine output [M, 3584]; it runs after the shared-expert join because
    ``shared`` is read in that epilogue. After the fused combine (``tail.combine_norm``) ``y`` already holds the
    normed rows and only the up-projection runs here."""
    from .moe import latent

    fused = getattr(_pending_tail, "fused", None)
    _pending_tail.fused = None
    if fused is not None and fused == y.data_ptr():
        out = torch.empty_like(shared)
        if y.shape[0] > TAIL_FGS_MAX_ROWS:
            return latent.up_proj(y, up_proj.weight, out, shared)
        latent.up_proj(y, up_proj.weight, out, shared, fused_tail=True, split3=True)
        return out
    m = y.shape[0]
    if not within_max_rows(m):
        # Stock norm module; above 32 rows latent.up_proj is the stock GEMM and add.
        return latent.up_proj(norm(y), up_proj.weight, torch.empty_like(shared), shared)
    norm_out = torch.empty_like(y)
    out = torch.empty_like(shared)
    return latent.run(
        y, norm.weight, float(norm.variance_epsilon), up_proj.weight, shared, out, norm_out
    )
