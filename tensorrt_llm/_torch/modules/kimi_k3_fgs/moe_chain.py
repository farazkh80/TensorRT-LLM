# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode MoE expert path in CuTe DSL: routing -> FC1 -> FC2 -> finalize, the CuTe
all-to-all and the post-combine latent tail.

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
device for the largest R seen outside graph capture, so CUDA-graph replays see fixed addresses.
"""

import importlib
import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from ..multi_stream_utils import do_multi_stream
from . import moe_streams

HIDDEN = 3584
INTER = 3072
TOP_K = 16
NUM_EXPERTS = 896
SF_VEC = 32
TAIL_FGS_MAX_ROWS = 32
# Forwards with more tokens on the fullest rank, such as context steps, keep the stock kernels.
MAX_TOKENS_PER_RANK = 128
# The fused router serves up to this many tokens per rank; the stock gate GEMM runs above.
ROUTER_MAX_ROWS = 64


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
) -> Workspace:
    """Run routing -> FC1 -> FC2 + finalize on the current stream; writes ``out`` [R, 3584] bf16.

    ``stages`` overrides any of the stages (tests plug reference or candidate kernels in per stage); the rest come
    from ``kimi_k3_fgs.moe``. Stage signatures:
      routing(topk_ids, local_expert_offset, num_local, plan)
      fc1(x, x_sf, weights, plan, act, act_sf)
      fc2_v2(act, act_sf, weights, plan, topk_w, out)            FC2 with the finalize fused
    ``stage_kwargs`` adds keyword arguments per stage.
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
    act, act_sf = ws.act, ws.act_sf
    run["routing"](
        topk_ids, weights.local_expert_offset, weights.num_local, plan, **kw.get("routing", {})
    )
    run["fc1"](x, x_sf, weights, plan, act, act_sf, **kw.get("fc1", {}))
    run["fc2_v2"](act, act_sf, weights, plan, topk_w, out, y=ws.y, **kw.get("fc2", {}))
    return ws


def _pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


class _Call:
    """One forward of one MoE layer."""

    def __init__(self, layer: "_Layer", rows: int):
        self.rows = rows


class _Layer:
    """The expert-path state of one MoE layer: its backend, its CuteA2A and the
    forward in flight between dispatch, run_moe and combine."""

    def __init__(self, routed_experts, a2a):
        self.backend = routed_experts.backend
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
        self.tail_fused = a2a is not None
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

    def stage_kwargs(self) -> Dict[str, dict]:
        """Keyword arguments of the expert-path stages."""
        kw = {"routing": {"enable_pdl": _pdl()}, "fc1": {}, "fc2": {"enable_pdl": _pdl()}}
        return kw

    def run_moe(self, ctx, out: torch.Tensor) -> None:
        call, self.call = self.call, None
        rows = max(ctx.all_rank_num_tokens) if ctx.all_rank_num_tokens else ctx.x.shape[0]
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
            stage_kwargs=self.stage_kwargs(),
            tokens_per_rank=rows,
        )
        if call is not None:
            if self.tail_fused:
                self.call = call

    def _combine_norm(self, call: _Call, final_hidden_states: torch.Tensor) -> torch.Tensor:
        """Push, then reduce the received rows and apply the latent RMSNorm in one kernel; ``run_tail`` takes the
        normed rows."""
        from .moe import tail

        norm = self.tail_norm
        out = tail.combine_norm(
            self.a2a,
            final_hidden_states,
            norm.weight,
            float(norm.variance_epsilon),
        )
        _pending_tail.fused = out.data_ptr()
        return out

    def abandon(self) -> None:
        """The stock runner takes this forward."""
        self.call = None

    def install_comm(self, comm) -> None:
        """Wrap the CuteA2A dispatch / combine that ``moe.comm.install`` put on ``comm`` so a forward on the CuTe
        path ends in the fused combine + norm; anything else goes to those wrappers unchanged."""
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
    unfinalized output, internal routing) and dispatch / combine go through ``moe.comm.CuteA2A``.

    Forwards with more than ``MAX_TOKENS_PER_RANK`` tokens on the fullest rank keep the stock path.
    """
    from .moe import comm as k3_comm

    backend = routed_experts.backend
    if not hasattr(getattr(backend, "activation", None), "gate_softcap"):
        raise ValueError(f"CuTe MoE path needs the SiTU experts, got {type(backend).__name__}")
    comm = getattr(routed_experts, "comm", None)
    # Only an NVLinkOneSided strategy with counted writes has the CFT workspace CuteA2A runs on; other strategies
    # (e.g. the allgather / reduce-scatter of a prefill model) keep their stock comm.
    cft = comm is not None and getattr(comm, "can_use_cft_counted_writes", False)
    a2a = k3_comm.install(comm) if cft else None
    layer = _Layer(routed_experts, a2a)
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
