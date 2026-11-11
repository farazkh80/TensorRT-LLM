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
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch

from ..multi_stream_utils import do_multi_stream
from . import edges as k3_edges
from . import moe_streams
from . import recipe as k3_recipe

ENV = "TLLM_K3_MOE"
TAIL_ENV = "TLLM_K3_MOE_TAIL"
COMM_ENV = "TLLM_K3_COMM"
HIDDEN = 3584
INTER = 3072
TOP_K = 16
NUM_EXPERTS = 896
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
    "fc2_push": "moe.fc2_push",
}
# The switches both kernels of each unfused-path edge run under.
_EDGE_SWITCHES = {
    "disp_route": (ENV, COMM_ENV),
    "route_fc1": (ENV,),
    "fc1_fc2": (ENV,),
    "fc2_fin": (ENV,),
    "fin_push": (ENV, COMM_ENV),
    "comb_norm": (COMM_ENV, TAIL_ENV),
    "norm_up": (TAIL_ENV,),
}
_COMM_EDGES = frozenset(("disp_route", "fin_push", "comb_norm", "rn_up", "recv_norm", "recv_fc1"))
COMBINED = "moe.combined"
LATENT_NORM = "moe.latent_norm"
RN_READY = "moe.rn_ready"
ROUTE_READY = "moe.route_ready"
LATENT_READY = "moe.latent_ready"
# One word per layer the routed FC1 counts its CTAs into once each has written its share of the plan; an FC2 that
# overlaps FC1 reads the tile map (every expert's slice) only after it reaches FC1's grid size.
FC1_PLAN = "moe.fc1_plan"
# FC2 v2 adds once per 128-channel unit of a row to its fc2_push word.
FC2_PUSH_UNITS = HIDDEN // 128
ROUTER_MAX_ROWS_ENV = "TLLM_K3_MOE_ROUTER_MAX_ROWS"
# FC2 K stages of 256 columns, each fed by 4 FC1 units of 64 columns.
FC1_K_STAGES = INTER // 256
# Rows per rank the expert-path edges serve; the tail edges stop where the split-K up-projection consumer does.
FGS_MAX_ROWS = int(os.environ.get("TLLM_K3_MOE_FGS_MAX_ROWS", "64"))
TAIL_FGS_MAX_ROWS = 32
# Hazard tests only: the FGS producers (routing, FC1, FC2, finalize, combine) delay some published stores by this
# many ns.
_HAZARD_DELAY_NS = int(os.environ.get("TLLM_K3_MOE_HAZARD_DELAY_NS", "0"))
MAX_ROWS_ENV = "TLLM_K3_MOE_MAX_ROWS"
# Negative controls only: the consumers of these edges skip their waits.
_HAZARD_NOWAIT = frozenset(filter(None, os.environ.get("TLLM_K3_MOE_HAZARD_NOWAIT", "").split(",")))

ROUTER_ENV = "TLLM_K3_MOE_ROUTER"
LATENT_DOWN_ENV = "TLLM_K3_MOE_LATENT_DOWN"
DISPATCH_ENV = "TLLM_K3_MOE_DISPATCH"
INBOX_ENV = "TLLM_K3_COMM_INBOX"
FC1_ENV = "TLLM_K3_MOE_FC1"
# Tokens per rank up to which TLLM_K3_MOE_FC1=v2 runs (default 32); above it the unfused routing + FC1 kernels run (the
# fused routing scan costs more than the routing kernel there).
FC1_MAX_ROWS_ENV = "TLLM_K3_MOE_FC1_MAX_ROWS"
FIN_ENV = "TLLM_K3_MOE_FIN"
# fc2_push publisher: unset = the finalize tail publishes every row once the FC2 grid is complete; "unit" = FC2 adds
# per weight tile (a GPU-scope fence per unit).
FC2_PUSH_ENV = "TLLM_K3_MOE_FC2_PUSH"
UP_ENV = "TLLM_K3_MOE_UP"
SHARED_ENV = "TLLM_K3_MOE_SHARED"


@dataclass(frozen=True)
class Kernel:
    """A fused kernel selected by ``env=value``: its entry point ``module:attr`` in ``kimi_k3_fgs.moe`` and the
    switches it needs, as ``(env, accepted values)``. ``integrated`` is False until ``moe_chain`` calls it."""

    env: str
    value: str
    entry: str
    requires: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    integrated: bool = True


_MOE = ((ENV, ("cute",)),)
_COMM = ((COMM_ENV, ("cute",)),)
KERNELS = (
    Kernel(ROUTER_ENV, "cute", "router:run", _MOE),
    Kernel(LATENT_DOWN_ENV, "cute", "latent_down:run", _MOE),
    Kernel(LATENT_DOWN_ENV, "send", "latent_down:run", _MOE + ((DISPATCH_ENV, ("send",)),), False),
    Kernel(DISPATCH_ENV, "send", "comm:CuteA2A.prepare_dispatch", _MOE + _COMM),
    Kernel(INBOX_ENV, "dispatch", "comm:CuteA2A.recv_sync", ((DISPATCH_ENV, ("send",)),), False),
    Kernel(INBOX_ENV, "combine", "comm:CuteA2A.combine_push", ((TAIL_ENV, ("fused",)),)),
    Kernel(FC1_ENV, "v2", "fc1:run_v2", _MOE),
    Kernel(FIN_ENV, "fc2", "fc2:run_v2", _MOE),
    Kernel(FC2_PUSH_ENV, "unit", "fc2:run_v2", ((FIN_ENV, ("fc2",)),)),
    Kernel(TAIL_ENV, "fused", "tail:combine_norm", _COMM),
    Kernel(UP_ENV, "split3", "latent:up_proj", ((TAIL_ENV, ("fused",)),)),
)
_KERNEL_ENVS = frozenset(k.env for k in KERNELS)
# Values that select an unfused CuTe path rather than a fused kernel.
_UNFUSED_VALUES = {TAIL_ENV: ("cute",)}
_LIST_ENVS = frozenset((INBOX_ENV,))


@dataclass(frozen=True)
class EdgeSpec:
    """A fused-kernel FGS edge: producer and consumer entry points (``mix`` = the pre-MoE ``attn_res_mix``), the
    sync object kind (``counter``: int32 slab zero-armed per step; ``slot``: bf16 slot armed with -0; ``comm``: the
    all-to-all's receive counters or inbox), the default row window and the switches it needs."""

    producer: str
    consumer: str
    kind: str
    rows: Tuple[int, int]
    requires: Tuple[Tuple[str, Tuple[str, ...]], ...]


V2_EDGES = {
    "mix_router": EdgeSpec("mix", "router:run", "slot", (1, 64), ((ROUTER_ENV, ("cute",)),)),
    "mix_latent": EdgeSpec(
        "mix", "latent_down:run", "slot", (1, 64), ((LATENT_DOWN_ENV, ("cute", "send")),)
    ),
    "router_send": EdgeSpec(
        "router:run",
        "comm:CuteA2A.dispatch",
        "counter",
        (1, 64),
        ((ROUTER_ENV, ("cute",)), (DISPATCH_ENV, ("send",))),
    ),
    "latent_send": EdgeSpec(
        "latent_down:run",
        "comm:CuteA2A.dispatch",
        "counter",
        (1, 64),
        ((LATENT_DOWN_ENV, ("cute",)), (DISPATCH_ENV, ("send",))),
    ),
    "recv_fc1": EdgeSpec(
        "comm:CuteA2A.dispatch",
        "fc1:run_v2",
        "comm",
        (1, 64),
        ((DISPATCH_ENV, ("send",)), (FC1_ENV, ("v2",))),
    ),
    "fc2_push": EdgeSpec(
        "fc2:run_v2",
        "comm:CuteA2A.combine",
        "counter",
        (1, 32),
        ((FIN_ENV, ("fc2",)), (COMM_ENV, ("cute",))),
    ),
    "recv_norm": EdgeSpec(
        "comm:CuteA2A.combine_push",
        "tail:combine_norm",
        "comm",
        (1, 128),
        ((TAIL_ENV, ("fused",)), (INBOX_ENV, ("combine",))),
    ),
    "rn_up": EdgeSpec(
        "tail:combine_norm", "latent:up_proj", "counter", (1, 32), ((TAIL_ENV, ("fused",)),)
    ),
}
# Fused-kernel edges moe_chain leases and passes to their kernels; the others are refused until they are.
_INTEGRATED_V2_EDGES = frozenset(
    (
        "rn_up",
        "recv_norm",
        "mix_latent",
        "mix_router",
        "recv_fc1",
        "latent_send",
        "router_send",
        "fc2_push",
    )
)
# Edges whose producer is the pre-MoE mix: it writes the MoE input into a -0-armed slot (``mix_out``).
_MIX_EDGES = frozenset(("mix_router", "mix_latent"))
MIX_X = "moe.x"
MODEL_HIDDEN = 7168
MIX_MAX_ROWS = 64
# Unfused-path edges whose producer or consumer kernel a fused kernel replaces.
_REPLACED_BY = {
    "disp_route": (FC1_ENV, "v2"),
    "route_fc1": (FC1_ENV, "v2"),
    "fc2_fin": (FIN_ENV, "fc2"),
    "fin_push": (FIN_ENV, "fc2"),
    "comb_norm": (TAIL_ENV, "fused"),
    "norm_up": (TAIL_ENV, "fused"),
}
# Unfused-path edges a fused kernel's switch rules out although both kernels still run: the standalone finalize reads
# e2p before any acquire that covers the routed FC1's plan writes.
_REFUSED_WITH = {"fc2_fin": (FC1_ENV, "v2")}


@dataclass(frozen=True)
class EdgeSync:
    """One edge's sync object as a fused kernel sees it (see ``EdgeSpec`` for ``kind``). ``expected`` is the
    per-unit target when fixed here (0: the stage derives it). Consumer side: ``diag`` (``TLLM_K3_MOE_FGS_DIAG``)
    and ``hazard_nowait``; producer side: ``hazard_delay_ns``. A ``comm`` edge has no tensor; ``source`` is the
    dispatch's receive descriptor."""

    edge: str
    kind: str
    tensor: Optional[torch.Tensor]
    expected: int = 0
    diag: str = ""
    hazard_delay_ns: int = 0
    hazard_nowait: bool = False
    source: Optional[object] = None


@dataclass(frozen=True)
class StageSync:
    """The edges one call of a fused kernel consumes (``waits``) and produces (``publishes``); empty maps or no
    ``StageSync`` at all is the FUSED-OFF build."""

    waits: Mapping[str, EdgeSync]
    publishes: Mapping[str, EdgeSync]


def switch(env: str) -> str:
    """The value of a switch ("" when off); for a fused-kernel switch an unknown value is an error, never a stock
    run."""
    value = os.environ.get(env, "")
    if not value or env not in _KERNEL_ENVS:
        return value
    valid = {k.value for k in KERNELS if k.env == env} | set(_UNFUSED_VALUES.get(env, ()))
    values = value.split(",") if env in _LIST_ENVS else [value]
    unknown = [v for v in values if v not in valid]
    if unknown:
        raise ValueError(
            f"{env}={value!r}: unknown value(s) {unknown}; valid: {', '.join(sorted(valid))}"
        )
    return value


def switched(env: str, value: str) -> bool:
    current = switch(env)
    return value in current.split(",") if env in _LIST_ENVS else current == value


def _requirements_met(requires) -> bool:
    return all(switch(env) and any(switched(env, v) for v in values) for env, values in requires)


def active_kernels() -> List[Kernel]:
    return [k for k in KERNELS if switched(k.env, k.value)]


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


def _edge_caps(entry: str, kind: str) -> Tuple[str, ...]:
    """The edges a fused kernel declares it can consume (``kind="waits"``) or produce (``"publishes"``):
    ``FGS_WAITS`` / ``FGS_PUBLISHES`` of its module, a dict keyed by the entry's attribute path or one tuple for a
    module with a single entry point."""
    module, _, attr = entry.partition(":")
    try:
        mod = importlib.import_module(f"{__package__}.moe.{module}")
    except ImportError:
        return ()
    caps = getattr(mod, f"FGS_{kind.upper()}", ())
    return tuple(caps.get(attr, ()) if isinstance(caps, dict) else caps)


def _missing_kernels() -> List[str]:
    missing = []
    for k in active_kernels():
        what = f"{k.env}={k.value}"
        if not _requirements_met(k.requires):
            needs = ", ".join(f"{env}={'|'.join(v)}" for env, v in k.requires)
            missing.append(f"{what} needs {needs}")
        elif _resolve(k.entry) is None:
            missing.append(f"{what}: {k.entry} does not exist")
        elif not k.integrated:
            missing.append(f"{what}: {k.entry} is not called by moe_chain yet")
    return missing


def _missing_v2_sides(edges: frozenset) -> List[str]:
    """Why each requested fused-kernel edge cannot run, including unfused-path edges whose kernels are replaced."""
    missing = []
    for edge in sorted(edges):
        replaced = _REPLACED_BY.get(edge)
        if replaced is not None and switched(*replaced):
            missing.append(f"{edge}: {replaced[0]}={replaced[1]} replaces its kernel")
        refused = _REFUSED_WITH.get(edge)
        if refused is not None and switched(*refused):
            missing.append(f"{edge}: not with {refused[0]}={refused[1]}")
        if edge == "disp_route" and switched(DISPATCH_ENV, "send"):
            missing.append(f"disp_route: {DISPATCH_ENV}=send replaces its dispatch")
        if edge == "recv_fc1" and tail_fused() and not switched(INBOX_ENV, "combine"):
            missing.append(
                f"recv_fc1: {TAIL_ENV}=fused after a send-only dispatch needs {INBOX_ENV}=combine"
            )
        spec = V2_EDGES.get(edge)
        if spec is None:
            continue
        if edge not in _INTEGRATED_V2_EDGES:
            missing.append(f"{edge}: not leased by moe_chain yet")
            continue
        if not _requirements_met(spec.requires):
            needs = ", ".join(f"{env}={'|'.join(v)}" for env, v in spec.requires)
            missing.append(f"{edge}: needs {needs}")
            continue
        if spec.producer != "mix" and edge not in _edge_caps(spec.producer, "publishes"):
            missing.append(f"{edge}: {spec.producer} does not publish it")
        if edge not in _edge_caps(spec.consumer, "waits"):
            missing.append(f"{edge}: {spec.consumer} does not wait on it")
    if "fc1_fc2" in edges:
        if switched(FC1_ENV, "v2") and "fc1_fc2" not in _edge_caps("fc1:run_v2", "publishes"):
            missing.append("fc1_fc2: fc1:run_v2 does not publish it")
        if switched(FIN_ENV, "fc2") and "fc1_fc2" not in _edge_caps("fc2:run_v2", "waits"):
            missing.append("fc1_fc2: fc2:run_v2 does not wait on it")
    return missing


def _switch_on(env: str) -> bool:
    return tail_enabled() if env == TAIL_ENV else os.environ.get(env, "") == "cute"


def enabled() -> bool:
    return os.environ.get(ENV, "") == "cute"


def _max_tokens_per_rank() -> int:
    return int(os.environ.get(MAX_ROWS_ENV, "128"))


def within_max_rows(tokens_per_rank: int) -> bool:
    """Whether a forward with this many tokens on its fullest rank runs the CuTe path (expert path, all-to-all and
    tail); larger forwards, such as context steps, keep the stock kernels and never grow the workspace."""
    return tokens_per_rank <= _max_tokens_per_rank()


def run_shared(runtime, x: torch.Tensor) -> Optional[torch.Tensor]:
    """``TLLM_K3_MOE_SHARED=cute``: the shared expert as the CuTe gate/up + down pair on at most
    ``TLLM_K3_MOE_SHARED_CTAS`` CTAs up to ``TLLM_K3_MOE_SHARED_MAX_TOKENS`` tokens, or None for the stock module.
    The down GEMM still waits for the shared-down event ``moe_streams.arm_shared_down`` armed on ``down_proj``,
    which this path bypasses. The kernels share per-device scratch: one stream at a time per device."""
    value = os.environ.get(SHARED_ENV, "")
    if not value:
        return None
    if value != "cute":
        raise ValueError(f"{SHARED_ENV}={value!r}; valid: cute")
    from .moe import shared_expert

    mlp = runtime.shared_experts
    gate_up, down = mlp.gate_up_proj, mlp.down_proj
    m = x.shape[0]
    cap = shared_expert.config(m)
    why = (
        "above TLLM_K3_MOE_SHARED_MAX_TOKENS"
        if cap is None
        else "tensor-parallel shared expert"
        if gate_up.tp_size != 1 or down.tp_size != 1
        else f"{gate_up.weight.dtype} / {down.weight.dtype} weights"
        if gate_up.weight.dtype != torch.bfloat16 or down.weight.dtype != torch.bfloat16
        else ""
    )
    k3_edges.debug("moe.shared", m, f"stock ({why})" if why else f"cute ctas={cap}")
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
    return os.environ.get(TAIL_ENV, "") in ("cute", "fused")


def tail_fused() -> bool:
    return os.environ.get(TAIL_ENV, "") == "fused"


def _tile_bounds(num_experts: int) -> tuple:
    # The tile follows the mean rows per expert, R * TOP_K / num_experts: R <= 512 / 1024 at 896 experts.
    return 512 * num_experts // NUM_EXPERTS, 1024 * num_experts // NUM_EXPERTS


def _bucket_rows(rows: int, num_experts: int, max_rows: int) -> int:
    """The largest R up to ``max_rows`` that ``tile_n_for`` maps to the same tile_n as ``rows``."""
    if os.environ.get("TLLM_K3_MOE_TILE_N"):
        return max(rows, max_rows)
    r8, r16 = _tile_bounds(num_experts)
    return max(rows, min(r8 if rows <= r8 else r16 if rows <= r16 else max_rows, max_rows))


def tile_n_for(rows: int, num_experts: int = NUM_EXPERTS) -> int:
    """Token rows per FC tile (the N of the swap-AB MMA) for R received rows routed over ``num_experts``."""
    override = os.environ.get("TLLM_K3_MOE_TILE_N")
    if override:
        return int(override)
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
                f"{ENV}=cute: no workspace for {rows} rows (tile_n {tile_n}) before graph capture; "
                "run the batch eagerly first"
            )
        ws = _workspaces[key] = Workspace(device, max(rows, alloc_rows), num_local, tile_n)
    return ws


def _stage(name: str) -> Callable:
    if name == "fc1_v2":
        return _resolve("fc1:run_v2")
    if name == "fc2_v2":
        return _resolve("fc2:run_v2")
    return importlib.import_module(f"{__package__}.moe.{name}").run


def fc1_v2_on(tokens_per_rank: Optional[int] = None) -> bool:
    """``TLLM_K3_MOE_FC1=v2`` and, when known, the forward's tokens per rank within ``TLLM_K3_MOE_FC1_MAX_ROWS``."""
    if not switched(FC1_ENV, "v2"):
        return False
    return tokens_per_rank is None or tokens_per_rank <= int(os.environ.get(FC1_MAX_ROWS_ENV, "32"))


def expert_stages(tokens_per_rank: Optional[int] = None) -> tuple:
    """The expert-path stages the switches select: ``fc1_v2`` routes and runs FC1 (``TLLM_K3_MOE_FC1=v2``, up to
    ``TLLM_K3_MOE_FC1_MAX_ROWS`` tokens per rank), ``fc2_v2`` runs FC2 with the finalize fused
    (``TLLM_K3_MOE_FIN=fc2``)."""
    first = ("fc1_v2",) if fc1_v2_on(tokens_per_rank) else ("routing", "fc1")
    second = ("fc2_v2",) if switched(FIN_ENV, "fc2") else ("fc2", "finalize")
    return first + second


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
) -> Workspace:
    """Run routing -> FC1 -> FC2 -> finalize on the current stream; writes ``out`` [R, 3584] bf16.

    ``stages`` overrides any of ``STAGES`` (tests plug reference or candidate kernels in per stage); the rest come
    from ``kimi_k3_fgs.moe.<stage>.run``. Stage signatures:
      routing(topk_ids, local_expert_offset, num_local, plan)
      fc1(x, x_sf, weights, plan, act, act_sf)
      fc2(act, act_sf, weights, plan, y)
      finalize(y, topk_w, plan, out)
      fc1_v2(x, x_sf, topk_ids, weights, plan, act, act_sf)      routing fused, replaces routing + fc1
      fc2_v2(act, act_sf, weights, plan, topk_w, out)            finalize fused, replaces fc2 + finalize
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
    names = expert_stages(tokens_per_rank)
    run = {name: stages.get(name) or _stage(name) for name in names}
    if ws is None:
        ws = workspace(x.device, rows, weights.num_local, tile_n_for(rows))
    plan = ws.plan
    if "fc1_v2" in run:
        run["fc1_v2"](x, x_sf, topk_ids, weights, plan, ws.act, ws.act_sf, **kw.get("fc1", {}))
    else:
        run["routing"](
            topk_ids, weights.local_expert_offset, weights.num_local, plan, **kw.get("routing", {})
        )
        run["fc1"](x, x_sf, weights, plan, ws.act, ws.act_sf, **kw.get("fc1", {}))
    moe_streams.fc1_done()
    if "fc2_v2" in run:
        run["fc2_v2"](ws.act, ws.act_sf, weights, plan, topk_w, out, y=ws.y, **kw.get("fc2", {}))
    else:
        run["fc2"](ws.act, ws.act_sf, weights, plan, ws.y, **kw.get("fc2", {}))
        run["finalize"](ws.y, topk_w, plan, out, **kw.get("finalize", {}))
    return ws


def _pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def fgs_edges() -> frozenset:
    """The FGS edges ``TLLM_K3_MOE_FGS`` turns on: a comma list of ``EDGES`` and ``V2_EDGES``, or ``all`` (the
    unfused-path edges whose kernels the active switches keep, plus the fused-kernel edges moe_chain runs whose
    switches are on)."""
    recipe_edges = k3_recipe.moe_edge_names()
    if recipe_edges is not None:
        return recipe_edges
    value = os.environ.get(FGS_ENV, "")
    if value == "all":
        names = {
            e
            for e in EDGES
            if not (e in _REPLACED_BY and switched(*_REPLACED_BY[e]))
            and not (e in _REFUSED_WITH and switched(*_REFUSED_WITH[e]))
            and all(_switch_on(env) for env in _EDGE_SWITCHES[e])
        }
        names |= {
            e
            for e, spec in V2_EDGES.items()
            if e in _INTEGRATED_V2_EDGES and _requirements_met(spec.requires)
        }
        return frozenset(names)
    names = {v for v in value.split(",") if v}
    unknown = names - set(EDGES) - set(V2_EDGES)
    if unknown:
        raise ValueError(
            f"{FGS_ENV}: unknown edge(s) {sorted(unknown)}; edges are "
            f"{', '.join(EDGES + tuple(V2_EDGES))}"
        )
    return frozenset(names)


def _accepts(fn: Callable, *names: str) -> bool:
    params = inspect.signature(fn).parameters
    return all(n in params for n in names)


def _missing_sides(edges: frozenset, moe_on: bool, comm_on: bool, tail_on: bool) -> List[str]:
    """Every requested edge needs both its kernel sides and the stages it connects; an edge that cannot run is an
    error rather than a silent run without it."""
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
    return [
        f"{edge}: {what}"
        for edge in sorted(edges & need.keys())
        for ok, what in need[edge]
        if not ok
    ]


def _counter_words(edge: str, tiles: int, ep: int, rows: Optional[int] = None) -> int:
    """int32 words of ``edge``'s counter slab for a plan of ``tiles`` token tiles (the kernels size their checks by the
    workspace's plan, which can exceed the current forward's) and ``rows`` tokens per source rank."""
    rows = FGS_MAX_ROWS if rows is None else rows
    return {
        "route_fc1": 1,
        "fc1_fc2": tiles * FC1_K_STAGES,
        "fc2_fin": tiles,
        "fin_push": ep * rows,
        "fc2_push": ep * rows,
    }[edge]


# Default row windows (tokens per rank) of the counter edges; TLLM_K3_MOE_FGS_ROWS=edge:low-high,... overrides them.
# The per-row push pays up to 32 rows (above, the push waits on rows that finish together),
# and the FC2 -> finalize counters cost more than they hide at a few rows.
_EDGE_ROWS = {"fin_push": (1, 32), "fc2_fin": (9, 32)}
_EDGE_ROWS.update({edge: spec.rows for edge, spec in V2_EDGES.items()})


def configure_windows(windows: Mapping[str, Tuple[int, int]]) -> None:
    """Size the counter slabs and the mix slot for the recipe's MoE edge spans (``recipe.install``)."""
    global FGS_MAX_ROWS, MIX_MAX_ROWS
    for edge, (_, high) in windows.items():
        if edge in _COUNTERS:
            FGS_MAX_ROWS = max(FGS_MAX_ROWS, high)
        elif edge in _MIX_EDGES:
            MIX_MAX_ROWS = max(MIX_MAX_ROWS, high)


def _edge_rows(edge: str) -> tuple:
    if k3_recipe.owns("moe"):
        return k3_recipe.span((f"moe.{edge}",)) or (1, 0)
    windows = dict(_EDGE_ROWS)
    for item in filter(None, os.environ.get("TLLM_K3_MOE_FGS_ROWS", "").split(",")):
        name, _, span = item.partition(":")
        low, _, high = span.partition("-")
        windows[name] = (int(low), int(high or low))
    return windows.get(edge, (1, FGS_MAX_ROWS))


def _max_tiles(rows: int, num_local: int, num_experts: int) -> int:
    tile_n = tile_n_for(rows, num_experts)
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
            ws = layer.workspace(r)
            tiles = ws.plan.tile_idx_to_expert_idx.numel()
            for edge in counters:
                low, high = _edge_rows(edge)
                if not (low <= rows <= high and k3_recipe.moe_on(edge, rows)):
                    k3_edges.debug(f"moe.{edge}", rows, f"off: outside its rows {low}..{high}")
                    continue
                words = _counter_words(edge, tiles, layer.ep, rows)
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
                    diag = fgs_diag(edge)
                    k3_edges.debug(
                        f"moe.{edge}",
                        rows,
                        f"on, {words} words" + (f", diag={diag}" if diag else ""),
                    )
        if "comb_norm" in layer.edges and local_rows and k3_recipe.moe_on("comb_norm", local_rows):
            lease = k3_edges.acquire(COMBINED, local_rows, window=(1, TAIL_FGS_MAX_ROWS))
            if lease is not None:
                self.leases["comb_norm"] = lease
        if (
            "disp_route" in layer.edges
            and k3_edges.slots()
            and 1 <= rows <= FGS_MAX_ROWS
            and k3_recipe.moe_on("disp_route", rows)
        ):
            self.leases["disp_route"] = None
        if not fc1_v2_on(rows):
            if "recv_fc1" in layer.edges and switched(FC1_ENV, "v2"):
                k3_edges.debug("moe.recv_fc1", rows, f"off: above {FC1_MAX_ROWS_ENV}, unfused FC1")
            return
        low, high = _edge_rows("recv_fc1")
        if (
            "recv_fc1" in layer.edges
            and layer.send
            and low <= rows <= high
            and k3_recipe.moe_on("recv_fc1", rows)
        ):
            self.leases["recv_fc1"] = None
        if layer.fc1_plan and "fc1_fc2" in self.leases:
            lease = k3_edges.acquire_counters(FC1_PLAN, rows, 1, window=(1, FGS_MAX_ROWS))
            if lease is not None:
                self.leases["fc1_plan"] = lease
            else:
                k3_edges.consumed(self.leases.pop("fc1_fc2"))
                k3_edges.debug("moe.fc1_fc2", rows, f"off: {FC1_PLAN} {k3_edges.last_refusal()}")

    def on(self, edge: str) -> bool:
        return edge in self.leases

    def slab(self, edge: str) -> torch.Tensor:
        return self.leases[edge].tensor.view(-1)

    def consumed(self, edge: str) -> None:
        lease = self.leases.get(edge)
        if lease is not None:
            k3_edges.consumed(lease)


class _Layer:
    """The expert-path state of one MoE layer: its backend, its CuteA2A (``TLLM_K3_COMM=cute``), its FGS edges and the
    forward in flight between dispatch, run_moe and combine."""

    def __init__(self, routed_experts, edges: frozenset, a2a):
        self.backend = routed_experts.backend
        self.edges = edges
        self.a2a = a2a
        self.num_local = int(self.backend.expert_size_per_partition)
        self.num_experts = int(self.backend.num_slots)
        # Received rows R are ep * tokens on the fullest rank, at most the cap on the CuTe path; the stock
        # all-to-all delivers the same R, so the plan and the counter slabs are sized from its ep too.
        comm_ep = getattr(getattr(routed_experts, "comm", None), "ep_size", 1)
        self.ep = a2a.ep if a2a is not None else comm_ep
        self.max_rows = comm_ep * _max_tokens_per_rank()
        self.call: Optional[_Call] = None
        # (ids, weights, x, x_sf) of the forward ``entry`` ran, until the scheduler takes it; None: stock entry.
        self.pending: Optional[tuple] = None
        # The router_send slab of the forward in flight, released once its consumer (the dispatch) is launched.
        self.route_ready = None
        # The latent_send slab and its per-token target, released like route_ready.
        self.latent_ready = None
        # Benchmark-only hook, inert unless the layer-wise benchmark sets ``routed_experts._k3_balanced_routing``: the
        # fused router then writes the benchmark's Balanced ids into the pending buffers, which the benchmark keeps
        # (the early send already holds them).
        self.balanced_ids = False
        self.entry_kernels: Tuple[str, ...] = ()
        layer_idx = getattr(routed_experts, "layer_idx", None)
        # A workspace serves every R of its tile_n bucket, so a plan can be as large as the bucket's largest R.
        tiles = max(
            _max_tiles(
                _bucket_rows(self.ep * rows, self.num_experts, self.max_rows),
                self.num_local,
                self.num_experts,
            )
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
        if edges & _MIX_EDGES:
            k3_edges.declare(MIX_X, MODEL_HIDDEN, layer_idx, max_rows=MIX_MAX_ROWS, min_rows=1)
        if "router_send" in edges:
            k3_edges.declare_counters(
                ROUTE_READY, _router_max_rows(), layer_idx, max_rows=_router_max_rows(), min_rows=1
            )
        if "latent_send" in edges:
            high = _edge_rows("latent_send")[1]
            k3_edges.declare_counters(LATENT_READY, high, layer_idx, max_rows=high, min_rows=1)
        if "rn_up" in edges:
            k3_edges.declare_counters(
                RN_READY, TAIL_FGS_MAX_ROWS, layer_idx, max_rows=TAIL_FGS_MAX_ROWS, min_rows=1
            )
        self.fc1_plan = switched(FC1_ENV, "v2") and "fc1_fc2" in edges
        if self.fc1_plan:
            k3_edges.declare_counters(FC1_PLAN, 1, layer_idx, max_rows=FGS_MAX_ROWS, min_rows=1)
        self.tail_fused = tail_fused() and a2a is not None
        self.inbox = switched(INBOX_ENV, "combine") and self.tail_fused
        self.send = switched(DISPATCH_ENV, "send") and a2a is not None
        if self.send:
            self._send_fork = torch.cuda.Event()
            self._send_done = torch.cuda.Event()
        self.early = None
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

    def stage_kwargs(
        self, call: Optional[_Call], rows: int = 0, fc1_ctas: int = 0
    ) -> Dict[str, dict]:
        kw = {"routing": {"enable_pdl": _pdl()}, "fc1": {}, "fc2": {}, "finalize": {}}
        if fc1_v2_on(rows) or switched(FIN_ENV, "fc2"):
            return self._fused_stage_kwargs(kw, call, rows, fc1_ctas)
        if call is None:
            return kw

        # Upper-bound diagnostic: the consumers get always-complete counters instead of the producers' slabs.
        def consumer_slab(edge: str) -> torch.Tensor:
            return _full_counters(self.device) if fgs_diag(edge) == "oracle" else call.slab(edge)

        if call.on("disp_route"):
            kw["routing"]["source"] = call.source
            if fgs_diag("disp_route") == "oracle":
                kw["routing"]["hazard_nowait"] = True
        if call.on("route_fc1"):
            kw["routing"]["ready"] = call.slab("route_fc1")
            if _HAZARD_DELAY_NS:
                kw["routing"]["hazard_delay_ns"] = _HAZARD_DELAY_NS
            kw["fc1"]["plan_ready"] = consumer_slab("route_fc1")
        if call.on("fc1_fc2"):
            kw["fc1"]["fc1_done"] = call.slab("fc1_fc2")
            kw["fc2"]["fc1_done"] = consumer_slab("fc1_fc2")
            if _HAZARD_DELAY_NS:
                kw["fc1"]["hazard_delay_ns"] = _HAZARD_DELAY_NS
        if call.on("fc2_fin"):
            kw["fc2"]["fc2_done"] = call.slab("fc2_fin")
            kw["finalize"]["fc2_done"] = consumer_slab("fc2_fin")
            kw["finalize"]["fc2_expected"] = HIDDEN // 128
            if _HAZARD_DELAY_NS:
                kw["fc2"]["hazard_delay_ns"] = _HAZARD_DELAY_NS
        if call.on("fin_push"):
            kw["finalize"]["fin_done"] = call.slab("fin_push")
            if _HAZARD_DELAY_NS:
                kw["finalize"]["hazard_delay_ns"] = _HAZARD_DELAY_NS
        return kw

    def _edge_sync(self, call: _Call, edge: str, consumer: bool) -> EdgeSync:
        diag = fgs_diag(edge) if consumer else ""
        return EdgeSync(
            edge,
            "counter",
            _full_counters(self.device) if diag == "oracle" else call.slab(edge),
            diag=diag,
            hazard_delay_ns=0 if consumer else _HAZARD_DELAY_NS,
            hazard_nowait=consumer and edge in _HAZARD_NOWAIT,
        )

    def _fused_stage_kwargs(
        self, kw: Dict[str, dict], call: Optional[_Call], rows: int, fc1_ctas: int
    ) -> Dict[str, dict]:
        """Keyword arguments when a fused expert-path kernel runs: fused kernels take ``fgs=StageSync`` and
        ``enable_pdl``, unfused kernels beside them keep their own edge arguments. ``fc1_ctas``: the routed FC1's
        grid, the value its ``fc1_plan`` word reaches."""
        fc1_v2, fc2_v2 = fc1_v2_on(rows), switched(FIN_ENV, "fc2")
        fc1_fc2 = call is not None and call.on("fc1_fc2")
        stage_sync = {}
        for stage, fused in (("fc1", fc1_v2), ("fc2", fc2_v2)):
            if fused:
                kw[stage] = {"enable_pdl": _pdl()}
                stage_sync[stage] = StageSync({}, {})
        if fc1_v2 and call is not None:
            if call.on("recv_fc1"):
                stage_sync["fc1"].waits["recv_fc1"] = EdgeSync(
                    "recv_fc1",
                    "comm",
                    None,
                    diag=fgs_diag("recv_fc1"),
                    hazard_nowait="recv_fc1" in _HAZARD_NOWAIT,
                    source=call.source,
                )
            if call.on("fc1_plan"):
                stage_sync["fc1"].publishes["fc1_plan"] = self._edge_sync(call, "fc1_plan", False)
                if fc2_v2:
                    stage_sync["fc2"].waits["fc1_plan"] = EdgeSync(
                        "fc1_plan", "counter", call.slab("fc1_plan"), expected=fc1_ctas
                    )
                else:
                    kw["fc2"]["plan_pub"] = call.slab("fc1_plan")
                    kw["fc2"]["plan_pub_expected"] = fc1_ctas
        if fc2_v2 and call is not None and call.on("fc2_push"):
            stage_sync["fc2"].publishes["fc2_push"] = self._edge_sync(call, "fc2_push", False)
            mode = "per FC2 unit" if switched(FC2_PUSH_ENV, "unit") else "finalize tail"
            k3_edges.debug("moe.fc2_push", rows, f"published by the {mode}")
        if fc1_fc2:
            if fc1_v2:
                stage_sync["fc1"].publishes["fc1_fc2"] = self._edge_sync(call, "fc1_fc2", False)
            else:
                kw["fc1"]["fc1_done"] = call.slab("fc1_fc2")
                if _HAZARD_DELAY_NS:
                    kw["fc1"]["hazard_delay_ns"] = _HAZARD_DELAY_NS
            if fc2_v2:
                stage_sync["fc2"].waits["fc1_fc2"] = self._edge_sync(call, "fc1_fc2", True)
            else:
                diag = fgs_diag("fc1_fc2")
                kw["fc2"]["fc1_done"] = (
                    _full_counters(self.device) if diag == "oracle" else call.slab("fc1_fc2")
                )
        for stage, sync in stage_sync.items():
            if sync.waits or sync.publishes:
                kw[stage]["fgs"] = sync
            _log_stage(stage + "_v2", rows, sync)
        return kw

    def run_moe(self, ctx, out: torch.Tensor) -> None:
        call, self.call = self.call, None
        rows = max(ctx.all_rank_num_tokens) if ctx.all_rank_num_tokens else ctx.x.shape[0]
        if call is None and self.a2a is None and self.edges:
            call = _Call(self, rows, 0)
        topk_w = ctx.token_final_scales
        if topk_w.dtype != torch.bfloat16:
            if call is not None and call.on("recv_fc1"):
                raise RuntimeError(
                    "recv_fc1: the cast would read the received routing weights before they land"
                )
            topk_w = topk_w.to(torch.bfloat16)
        ws = self.workspace(ctx.x.shape[0])
        fc1_ctas = 0
        if call is not None and call.on("fc1_plan"):
            fc1_ctas = _resolve("fc1:num_ctas_for")(ctx.x, ws.plan)
        run_expert_path(
            ctx.x,
            ctx.x_sf.view(torch.uint8),
            ctx.token_selected_experts,
            topk_w,
            ExpertWeights.from_backend(self.backend),
            out,
            ws=ws,
            stage_kwargs=self.stage_kwargs(call, rows, fc1_ctas),
            tokens_per_rank=rows,
        )
        if call is not None:
            for edge in ("route_fc1", "fc1_fc2", "fc2_fin", "fc1_plan"):
                call.consumed(edge)
            if (
                call.on("fin_push")
                or call.on("fc2_push")
                or call.on("comb_norm")
                or self.tail_fused
            ):
                self.call = call

    def _dispatch_send(self, call: _Call, *args) -> tuple:
        """``TLLM_K3_MOE_DISPATCH=send``: the round's prepare kernel on the current stream, the dispatch on its own
        top-priority stream, joined back right after it. With ``recv_fc1`` the dispatch is send-only and the routed
        FC1 waits on the receive; the join still orders FC1 after the send kernel, which cannot run beside a spinning
        tile-32 FC1."""
        a2a = self.a2a
        a2a.prepare_dispatch()
        stream = _send_stream(self.device)
        self._send_fork.record()
        with torch.cuda.stream(stream):
            self._send_fork.wait()
            out = a2a.dispatch(
                *args, send_only=call.on("recv_fc1"), prepared=True, **self._producer_waits()
            )
            self._send_done.record()
        self._send_done.wait()
        waits = ",".join(
            e
            for e, on in (("latent_send", self.latent_ready), ("router_send", self.route_ready))
            if on
        )
        k3_edges.debug(
            "moe.dispatch",
            call.rows,
            (
                "send stream, send-only (FC1 waits on the receive), joined after the send"
                if call.on("recv_fc1")
                else "send stream, joined after the dispatch"
            )
            + (f", waits={waits}" if waits else ""),
        )
        return out

    def early_send(self, experts, counts, x, x_sf, ids, weights) -> bool:
        """With ``latent_send`` and ``router_send`` on, launch this forward's dispatch send before the entry kernels:
        its CTAs are resident first and put each token's row as the router's and the latent down's ready words
        complete. The dispatch wrapper later joins it instead of launching one; until then nothing on the current
        stream may wait for it (it waits on kernels launched after it)."""
        runtime_max = max(counts)
        requirement = getattr(experts.backend, "input_requirement", None)
        scales_dtype = getattr(requirement, "routing_scales_dtype", None)
        # Eager forwards first compile and load the entry kernels; loading a module while the send spins on them
        # can hang, so only graph-captured (multi-stream) forwards launch the send early.
        if (
            not self.send
            or not (do_multi_stream() or torch.cuda.is_current_stream_capturing())
            or self.route_ready is None
            or self.latent_ready is None
            or x is None
            or ids is None
            or scales_dtype not in (None, weights.dtype)
            or not within_max_rows(runtime_max)
            or not self.a2a.comm.use_cft_for_dispatch(runtime_max)
            or not self.a2a.comm.use_cft_for_combine(runtime_max)
        ):
            return False
        call = _Call(self, runtime_max, x.shape[0])
        stream = _send_stream(self.device)
        self._send_fork.record()
        with torch.cuda.stream(stream):
            self._send_fork.wait()
            self.a2a.prepare_dispatch()
            out = self.a2a.dispatch(
                x,
                x_sf,
                ids,
                weights,
                list(counts),
                send_only=call.on("recv_fc1"),
                prepared=True,
                **self._producer_waits(),
            )
            self._send_done.record()
        self.early = (call, (x, x_sf, ids, weights), list(counts), out)
        return True

    def _join_early_send(self, args, all_rank_num_tokens) -> tuple:
        call, launched, counts, out = self.early
        self.early = None
        same = list(all_rank_num_tokens) == counts and all(
            a is b
            or (
                a is not None
                and b is not None
                and a.data_ptr() == b.data_ptr()
                and a.shape == b.shape
                and a.dtype == b.dtype
            )
            for a, b in zip(args, launched)
        )
        if not same:
            raise RuntimeError(
                "the early dispatch send was launched with other inputs than the dispatch call"
            )
        self._send_done.wait()
        k3_edges.debug(
            "moe.dispatch",
            call.rows,
            "send stream launched before the entry kernels, waits=latent_send,router_send, joined at the dispatch",
        )
        return call, out

    def _producer_waits(self) -> dict:
        """The send's per-token waits on the entry kernels of this forward (``latent_send`` / ``router_send``)."""
        kw = {}
        if self.latent_ready is not None:
            kw.update(x_ready=self.latent_ready[0].tensor.view(-1), x_expected=self.latent_ready[1])
        if self.route_ready is not None:
            kw.update(route_ready=self.route_ready.tensor.view(-1), route_expected=1)
        return kw

    def _combine_norm(self, call: _Call, final_hidden_states: torch.Tensor) -> torch.Tensor:
        """``TLLM_K3_MOE_TAIL=fused``: push, then reduce the received rows and apply the latent RMSNorm in one
        kernel; ``run_tail`` takes the normed rows. With ``rn_up`` the kernel publishes a ready word per row that the
        latent up waits on instead of the grid dependency."""
        from .moe import tail

        local_rows = self.a2a.comm._dispatch_state["local_num_tokens"]
        ready_lease = None
        if (
            "rn_up" in self.edges
            and 1 <= local_rows <= TAIL_FGS_MAX_ROWS
            and k3_recipe.moe_on("rn_up", local_rows)
        ):
            ready_lease = k3_edges.acquire_counters(
                RN_READY, local_rows, local_rows, window=_edge_rows("rn_up")
            )
        k3_edges.debug(
            "moe.rn_up", local_rows, "off" if ready_lease is None else f"on, {local_rows} words"
        )
        fin_kw = _push_wait(call, final_hidden_states.numel() // HIDDEN)
        norm = self.tail_norm
        inbox_kw = {}
        # The inbox changes what peers push, so the recipe decides it on the rank-consistent row count.
        inbox = self.inbox and k3_recipe.moe_on("recv_norm", call.rows)
        if inbox:
            poll = "recv_norm" in self.edges and fgs_diag("recv_norm") != "ready"
            inbox_kw = {"inbox": "lamport", "lamport_wait": "poll" if poll else "done"}
            k3_edges.debug(
                "moe.recv_norm",
                local_rows,
                "on, inbox poll" if "recv_norm" in self.edges else "off, push-completion words",
            )
        out = tail.combine_norm(
            self.a2a,
            final_hidden_states,
            norm.weight,
            float(norm.variance_epsilon),
            ready=None if ready_lease is None else ready_lease.tensor.view(-1),
            hazard_delay_ns=_HAZARD_DELAY_NS if ready_lease is not None else 0,
            **inbox_kw,
            **fin_kw,
        )
        call.consumed("fin_push")
        call.consumed("fc2_push")
        k3_edges.debug(
            "moe.combine_norm",
            local_rows,
            f"fused reduce + norm, wire={'e4m3' if self.a2a.lpc else 'bf16'}, "
            f"{'Lamport inbox' if inbox else 'counted writes'}, pdl={int(_pdl())}",
        )
        _pending_tail.fused = (out.data_ptr(), ready_lease)
        return out

    def abandon(self) -> None:
        """The stock runner takes this forward: its producers never run, so its edges stay off (released
        unwritten; the step reset re-arms them)."""
        call, self.call = self.call, None
        if call is not None:
            if call.on("recv_fc1"):
                raise RuntimeError(
                    "recv_fc1: a send-only receive cannot go to the stock expert path"
                )
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
            if self.early is not None:
                call, out = self._join_early_send(
                    (hidden_states, hidden_states_sf, token_selected_slots, token_final_scales),
                    all_rank_num_tokens,
                )
                if call.on("recv_fc1"):
                    call.source = a2a.recv_source()
                self.call = call
                return out
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
            if self.send:
                out = self._dispatch_send(
                    call,
                    hidden_states,
                    hidden_states_sf,
                    token_selected_slots,
                    token_final_scales,
                    all_rank_num_tokens,
                )
            else:
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
            elif call.on("recv_fc1"):
                call.source = a2a.recv_source()
            self.call = call
            return out

        def combine(final_hidden_states, **kwargs):
            call, self.call = self.call, None
            if call is not None and self.tail_fused and self.tail_norm is not None:
                return self._combine_norm(call, final_hidden_states)
            if call is None or not (
                call.on("fin_push") or call.on("fc2_push") or call.on("comb_norm")
            ):
                return m1_combine(final_hidden_states, **kwargs)
            fin_kw = _push_wait(call, final_hidden_states.numel() // HIDDEN)
            out = call.leases["comb_norm"].tensor if call.on("comb_norm") else None
            if _HAZARD_DELAY_NS:
                fin_kw["hazard_delay_ns"] = _HAZARD_DELAY_NS
            y = a2a.combine(final_hidden_states, out=out, **fin_kw)
            call.consumed("fin_push")
            call.consumed("fc2_push")
            if out is not None:
                _pending_tail.lease = call.leases["comb_norm"]
            return y

        comm.dispatch, comm.combine = dispatch, combine


def _log_stage(stage: str, rows: int, sync: Optional[StageSync]) -> None:
    if sync is None or not (sync.waits or sync.publishes):
        path = "fused-off"
    else:
        diag = {e.diag for e in sync.waits.values()} - {""}
        path = f"waits={','.join(sorted(sync.waits))} publishes={','.join(sorted(sync.publishes))}"
        path += f" diag={','.join(sorted(diag))}" if diag else ""
    k3_edges.debug(f"moe.{stage}", rows, f"{path} pdl={int(_pdl())}")


def fgs_diag(edge: str) -> str:
    from .moe.sync import fgs_diag as diag

    return diag(edge)


_full: Dict[torch.device, torch.Tensor] = {}


def _full_counters(device: torch.device) -> torch.Tensor:
    """Counters that every wait finds complete (``TLLM_K3_MOE_FGS_DIAG=oracle`` only)."""
    if device not in _full:
        _full[device] = torch.full((1 << 16,), 1 << 20, dtype=torch.int32, device=device)
    return _full[device]


def _push_wait(call: _Call, rows: int) -> dict:
    """The combine push's per-row wait on the expert path's last kernel: finalize (``fin_push``) or FC2 v2
    (``fc2_push``)."""
    if call.on("fin_push"):
        edge, expected = "fin_push", finalize_chunks(rows)
    elif call.on("fc2_push"):
        edge, expected = "fc2_push", FC2_PUSH_UNITS
    else:
        return {}
    return {
        "fin_done": call.slab(edge),
        "fin_expected": 0 if fgs_diag(edge) == "oracle" else expected,
    }


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
    - ``TLLM_K3_MOE_FGS=<edges>|all``: the FGS edges; each needs both of its kernel sides and the stages it connects.
    - the fused-kernel switches of ``KERNELS``: each needs its entry point and the switches it builds on, and is
      refused while ``moe_chain`` does not call it yet.

    Forwards with more than ``TLLM_K3_MOE_MAX_ROWS`` (default 128) tokens on the fullest rank keep the stock path.
    """
    edges = fgs_edges()
    moe_on, comm_on, tail_on = enabled(), os.environ.get(COMM_ENV, "") == "cute", tail_enabled()
    missing = _missing_kernels()
    if missing:
        raise ValueError(f"fused MoE kernels: cannot run {'; '.join(missing)}")
    if not (moe_on or comm_on or edges):
        return False
    from .moe import comm as k3_comm

    missing = _missing_sides(edges, moe_on, comm_on, tail_on) + _missing_v2_sides(edges)
    if missing:
        raise ValueError(f"{FGS_ENV}: cannot run {'; '.join(missing)}")
    if edges and not k3_edges.slots():
        raise ValueError(f"{FGS_ENV} needs edge slots ({k3_edges.ENV}=chain|chain_after|lamport)")
    backend = routed_experts.backend
    if moe_on and not hasattr(getattr(backend, "activation", None), "gate_softcap"):
        raise ValueError(f"{ENV}=cute needs the SiTU experts, got {type(backend).__name__}")
    comm = getattr(routed_experts, "comm", None)
    # Only an NVLinkOneSided strategy with counted writes has the CFT workspace CuteA2A runs on; other strategies
    # (e.g. the allgather / reduce-scatter of a prefill model) keep their stock comm and the layer drops its comm edges.
    cft = comm is not None and getattr(comm, "can_use_cft_counted_writes", False)
    lamport_rows = _max_tokens_per_rank() if switched(INBOX_ENV, "combine") else 0
    a2a = k3_comm.install(comm, lamport_rows=lamport_rows) if comm_on and cft else None
    if a2a is not None:
        k3_edges.debug(
            "moe.comm",
            0,
            f"CuteA2A combine wire={'e4m3' if a2a.lpc else 'bf16'}"
            + (f", Lamport inbox {lamport_rows} rows" if lamport_rows else ""),
        )
    if a2a is None and edges & _COMM_EDGES:
        k3_edges.debug(
            "moe.install",
            0,
            f"{type(comm).__name__}: no CuTe all-to-all, {sorted(edges & _COMM_EDGES)} off",
        )
        edges = edges - _COMM_EDGES
    layer = _Layer(routed_experts, edges, a2a)
    # The forward's edge leases are taken at the CuTe dispatch, so it is wrapped for any edge, not only the comm ones.
    if a2a is not None and (edges or layer.tail_fused or layer.send):
        layer.install_comm(comm)
    if "rn_up" in edges and fgs_diag("rn_up") == "ready":
        raise ValueError("rn_up has no ready build; leave it out of TLLM_K3_MOE_FGS_DIAG_EDGES")
    if "recv_norm" in edges and fgs_diag("recv_norm") == "oracle":
        raise ValueError(
            "recv_norm has no oracle build; leave it out of TLLM_K3_MOE_FGS_DIAG_EDGES"
        )
    if not moe_on:
        return True
    _install_entry(routed_experts, layer)
    stock_run_moe = backend.run_moe
    checked = []
    ep = getattr(comm, "ep_size", 1)

    def run_moe(ctx, *, workspace=None):
        if layer.pending is not None:
            raise RuntimeError(
                f"{ENV}=cute: the scheduler skipped the fused router / latent-down result"
            )
        if layer.route_ready is not None:
            k3_edges.consumed(layer.route_ready)
            layer.route_ready = None
        if layer.latent_ready is not None:
            k3_edges.consumed(layer.latent_ready[0])
            layer.latent_ready = None
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


_send_streams: Dict[torch.device, torch.cuda.Stream] = {}


def _send_stream(device: torch.device) -> torch.cuda.Stream:
    if device not in _send_streams:
        high = torch.cuda.Stream.priority_range()[1]
        _send_streams[device] = torch.cuda.Stream(device=device, priority=high)
    return _send_streams[device]


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
    """``TLLM_K3_MOE_ROUTER`` / ``TLLM_K3_MOE_LATENT_DOWN``: the routed experts take the fused entry kernels'
    results through the backend's fused route + quantize hook, the scheduler's only per-forward way in ahead of
    the dispatch; ``entry`` sets them per forward."""
    routed_experts._k3_moe_layer = layer
    kernels = tuple(
        name
        for name, env in (("router", ROUTER_ENV), ("latent_down", LATENT_DOWN_ENV))
        if switched(env, "cute")
    )
    if not kernels:
        return
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
        k3_edges.debug("moe.entry", pending[0].shape[0], "fused ids through routing_method.apply")
        return pending[0], pending[1]

    def quantize_input(x, *args, **kwargs):
        pending = layer.pending
        if pending is None or pending[2] is None:
            return stock_quantize(x, *args, **kwargs)
        layer.pending = None if pending[0] is None else pending[:2] + (None, None)
        k3_edges.debug("moe.entry", pending[2].shape[0], "fused rows through quantize_input")
        return pending[2], pending[3]

    backend.try_fused_route_quant = try_fused_route_quant
    routing.apply = apply
    backend.quantize_input = quantize_input


def entry(runtime, hidden_states: torch.Tensor, all_rank_num_tokens) -> Optional[tuple]:
    """The MoE block's entry when a fused entry kernel is on: runs the fused router
    (``TLLM_K3_MOE_ROUTER=cute``) and / or the fused latent down + MXFP8 quantize
    (``TLLM_K3_MOE_LATENT_DOWN=cute``), the stock form of the other, and returns ``(routed_in, router_logits)``
    for ``runtime.routed_experts``; None keeps the stock entry. The fused outputs reach the scheduler through the
    hook ``_install_entry`` put on the backend; ``routed_in`` and ``router_logits`` then only carry the token count
    (``hidden_states`` and the expert ids stand in for the latent rows and the logits).

    It engages only where the CuTe expert path does: at most ``TLLM_K3_MOE_MAX_ROWS`` tokens on the fullest rank,
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
    fused_router = "router" in layer.entry_kernels
    if fused_router:
        stock_why = (
            f"above {ROUTER_MAX_ROWS_ENV}={_router_max_rows()}"
            if m > _router_max_rows()
            else f"{gate.weight.dtype} gate weight"
            if gate.weight.dtype != torch.float32
            else f"{gate.weight.shape[0]} experts, the kernel serves {_resolve('router:NUM_EXPERTS')}"
            if gate.weight.shape[0] != _resolve("router:NUM_EXPERTS")
            else ""
        )
        fused_router = not stock_why
        k3_edges.debug("moe.router", m, "fused" if fused_router else f"stock ({stock_why})")
    if not fused_router and "latent_down" not in layer.entry_kernels:
        return None
    bufs = _entry_bufs(hidden_states.device, layer.entry_kernels)
    ids = weights = xq = x_sf = None
    fused_latent = "latent_down" in layer.entry_kernels
    router_sync = _router_sync(layer, hidden_states) if fused_router else None
    latent_sync = _mix_sync(layer, "mix_latent", hidden_states)
    if fused_latent:
        latent_sync = _latent_send(layer, m, latent_sync)
    balanced = None
    if fused_router:
        ids, weights = bufs.ids[:m], bufs.weights[:m]
        balanced = getattr(experts, "_k3_balanced_routing", None)
        if balanced is not None:
            k3_edges.debug(
                "moe.router", m, f"benchmark Balanced ids (dp_rank, ep) = {tuple(balanced)}"
            )
    if fused_latent:
        xq, x_sf = bufs.x[:m], bufs.x_sf[:m]
    early = layer.early_send(experts, counts, xq, x_sf, ids, weights)

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
            fgs=router_sync,
            enable_pdl=_pdl(),
            balanced=balanced,
        )
        return ids

    def run_latent():
        if not fused_latent:
            return runtime._routed_projection(hidden_states, runtime.routed_expert_down_proj)
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
        order = "under"
        router_logits = run_router()
        routed_in = run_latent()
    k3_edges.debug("moe.entry_order", m, order + (", send launched first" if early else ""))
    if fused_router:
        _log_stage("router", max(counts), router_sync)
    if "latent_down" in layer.entry_kernels:
        _log_stage("latent_down", max(counts), latent_sync)
    layer.pending = (ids, weights, xq, x_sf)
    layer.balanced_ids = balanced is not None
    return routed_in, router_logits


def _router_max_rows() -> int:
    return int(os.environ.get(ROUTER_MAX_ROWS_ENV, "64"))


def _router_sync(layer: _Layer, hidden_states: torch.Tensor) -> Optional[StageSync]:
    """The fused router's edges: ``mix_router`` (consumer) and ``router_send`` (producer: one release add per token
    to ``moe.route_ready`` after its ids and weights)."""
    mix = _mix_sync(layer, "mix_router", hidden_states)
    waits = dict(mix.waits) if mix is not None else {}
    publishes = {}
    m = hidden_states.shape[0]
    if "router_send" in layer.edges and k3_recipe.moe_on("router_send", m):
        lease = k3_edges.acquire_counters(ROUTE_READY, m, m, window=_edge_rows("router_send"))
        if lease is not None:
            publishes["router_send"] = EdgeSync(
                "router_send",
                "counter",
                lease.tensor.view(-1),
                expected=1,
                hazard_delay_ns=_HAZARD_DELAY_NS,
            )
            layer.route_ready = lease
    if not waits and not publishes:
        return None
    return StageSync(waits, publishes)


def _latent_send(layer: _Layer, m: int, sync: Optional[StageSync]) -> Optional[StageSync]:
    """``latent_send`` (producer side): the fused latent down adds to ``moe.latent_ready[t]`` after token t's q and
    sf rows; the send dispatch reads them once the word reaches ``latent_down.ready_target``."""
    if "latent_send" not in layer.edges or not k3_recipe.moe_on("latent_send", m):
        return sync
    lease = k3_edges.acquire_counters(LATENT_READY, m, m, window=_edge_rows("latent_send"))
    if lease is None:
        k3_edges.debug("moe.latent_send", m, f"off: {k3_edges.last_refusal()}")
        return sync
    expected = _resolve("latent_down:ready_target")(m)
    layer.latent_ready = (lease, expected)
    publish = EdgeSync(
        "latent_send",
        "counter",
        lease.tensor.view(-1),
        expected=expected,
        hazard_delay_ns=_HAZARD_DELAY_NS,
    )
    return StageSync(dict(sync.waits) if sync is not None else {}, {"latent_send": publish})


def _mix_sync(layer: _Layer, edge: str, hidden_states: torch.Tensor) -> Optional[StageSync]:
    """The ``mix_router`` / ``mix_latent`` consumer side: the MoE input itself, when it lives in the ``mix_out``
    slot and the edge is on."""
    if (
        edge not in layer.edges
        or not k3_recipe.moe_on(edge, hidden_states.shape[0])
        or k3_edges.lease_containing(hidden_states) is None
    ):
        return None
    sync = EdgeSync(
        edge,
        "slot",
        hidden_states,
        diag=fgs_diag(edge),
        hazard_nowait=edge in _HAZARD_NOWAIT,
    )
    return StageSync({edge: sync}, {})


def mix_out(runtime, rows: int):
    """The slot the pre-MoE mix writes the MoE input into when a ``mix_*`` edge is on (``TLLM_K3_MOE_FGS``), or None
    for its plain buffer. The mix stores it as an FGS producer (no -0, dependents launched at entry); the slot is
    released once the MoE block has joined its streams (``mix_done``), which re-arms it on the reset stream."""
    layer = getattr(getattr(runtime, "routed_experts", None), "_k3_moe_layer", None)
    if layer is None or not any(k3_recipe.moe_on(e, rows) for e in layer.edges & _MIX_EDGES):
        return None
    lease = k3_edges.acquire(MIX_X, rows, window=(1, MIX_MAX_ROWS))
    k3_edges.debug(MIX_X, rows, "slot" if lease is not None else f"off: {k3_edges.last_refusal()}")
    return lease


def mix_done(hidden_states: torch.Tensor) -> None:
    """Release the ``mix_out`` slot ``hidden_states`` lives in, after the last of its readers was launched."""
    lease = k3_edges.lease_containing(hidden_states)
    if lease is not None:
        k3_edges.consumed(lease)


def run_tail(y: torch.Tensor, norm, up_proj, shared: torch.Tensor) -> torch.Tensor:
    """Post-combine latent tail (``TLLM_K3_MOE_TAIL=cute``): ``up_proj(norm(y)) + shared`` with the routed + shared
    add fused into the up-projection epilogue. ``y`` is the combine output [M, 3584]; it runs after the shared-expert
    join because ``shared`` is read in that epilogue. With ``comb_norm`` the norm polls the combine's slot; with
    ``norm_up`` it writes a slot the up-projection consumes as it lands. With ``TLLM_K3_MOE_TAIL=fused`` the combine
    already returned the normed rows (``tail.combine_norm``) and only the up-projection runs here, waiting on the
    per-row ready words with ``rn_up``."""
    from .moe import latent

    fused = getattr(_pending_tail, "fused", None)
    _pending_tail.fused = None
    if fused is not None and fused[0] == y.data_ptr():
        ready_lease = fused[1]
        out = torch.empty_like(shared)
        if y.shape[0] > TAIL_FGS_MAX_ROWS:
            return latent.up_proj(y, up_proj.weight, out, shared)
        ready = None
        if ready_lease is not None:
            ready = (
                _full_counters(y.device)
                if fgs_diag("rn_up") == "oracle"
                else ready_lease.tensor.view(-1)
            )
        split3 = switched(UP_ENV, "split3")
        k3_edges.debug(
            "moe.up",
            y.shape[0],
            "fused-tail split-K"
            + (" split3 (128x8 split 3 up to 8 rows), l2pf=1" if split3 else "")
            + (", waits rn_up" if ready is not None else ""),
        )
        latent.up_proj(y, up_proj.weight, out, shared, fused_tail=True, ready=ready, split3=split3)
        if ready_lease is not None:
            k3_edges.consumed(ready_lease)
        return out
    combined = getattr(_pending_tail, "lease", None)
    _pending_tail.lease = None
    m = y.shape[0]
    if not within_max_rows(m):
        # Stock norm module; above 32 rows latent.up_proj is the stock GEMM and add.
        return latent.up_proj(norm(y), up_proj.weight, torch.empty_like(shared), shared)
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
