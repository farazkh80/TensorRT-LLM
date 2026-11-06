# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream placement of the Kimi K3 decode MoE layer (opt-in, default off; CUDA-graph multi-stream only).

``TLLM_K3_MOE_STREAMS=branches``: the router branch (router logits, then the whole routed path: top-k, quantize,
dispatch, experts, combine, tail) runs on one stream and the latent down-projection on another, both at the highest
stream priority, so neither queues behind the other and both win pending-CTA scheduling over the shared expert
(default priority). The latent branch is joined at the first routed-path call that reads the latent activations
(``try_fused_route_quant`` when the backend implements it, ``quantize_input`` or ``comm.dispatch``), so top-k never
waits for it.

``TLLM_K3_SHARED_DOWN_AT=combine_start``: the shared expert's down-projection waits until the routed path reaches
``comm.combine``, moving it from the expert window into the combine window. ``fc2_end``: it waits only for FC1
(``fc1_done``, called by the expert path between FC1 and FC2), so at the shared stream's lower priority its CTAs take
SMs as FC2's CTAs retire and it finishes before the combine, leaving the combine window to the latent up; without
that call in the forward it falls back to ``combine_start``. Priority only separates them under ``branches``. Use
with ``TLLM_K3_SHARED_AT``.

``TLLM_K3_MOE_ENTRY=under|branches|auto`` orders the fused entry kernels (``moe_chain.entry``): ``under`` (default)
launches the router right after the pre-MoE mix and the latent down after it on the same stream (with ``mix_router``
/ ``mix_latent`` both start under the mix); ``branches`` runs them on two streams one priority level below the dispatch
send stream, joined before the routed path; ``auto`` takes ``under`` up to ``ENTRY_UNDER_MAX_ROWS`` tokens and
``branches`` above.

The switches change only streams and event waits; kernels and their inputs are unchanged.
"""

import os
import threading
from typing import Dict, Optional, Tuple

import torch

from . import edges

STREAMS_ENV = "TLLM_K3_MOE_STREAMS"
SHARED_DOWN_ENV = "TLLM_K3_SHARED_DOWN_AT"
ENTRY_ENV = "TLLM_K3_MOE_ENTRY"
ENTRY_UNDER_MAX_ROWS = 16

_streams: Dict[int, Tuple[torch.cuda.Stream, torch.cuda.Stream]] = {}
_entry_streams: Dict[int, Tuple[torch.cuda.Stream, torch.cuda.Stream]] = {}
_fc2_end = threading.local()


def branches_enabled() -> bool:
    value = os.environ.get(STREAMS_ENV, "off").strip().lower() or "off"
    if value not in ("off", "branches"):
        raise ValueError(f"{STREAMS_ENV}={value!r}; valid: off, branches")
    return value == "branches"


def shared_down_at() -> Optional[str]:
    value = os.environ.get(SHARED_DOWN_ENV, "").strip().lower() or None
    if value not in (None, "combine_start", "fc2_end"):
        raise ValueError(f"{SHARED_DOWN_ENV}={value!r}; valid: combine_start, fc2_end")
    return value


def entry_order(rows: int) -> str:
    value = os.environ.get(ENTRY_ENV, "under").strip().lower() or "under"
    if value not in ("under", "branches", "auto"):
        raise ValueError(f"{ENTRY_ENV}={value!r}; valid: under, branches, auto")
    if value == "auto":
        return "under" if rows <= ENTRY_UNDER_MAX_ROWS else "branches"
    return value


def run_entry_branches(device: torch.device, events, router, latent):
    """Run ``router()`` and ``latent()`` on the two entry streams and join both on the current stream.

    The entry streams sit one level below the top priority so that a dispatch send launched ahead of them on the
    top-priority send stream gets its CTAs resident first while it waits on their per-token ready words."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _entry_streams:
        prio = torch.cuda.Stream.priority_range()[1] + 1
        _entry_streams[index] = (
            torch.cuda.Stream(device=index, priority=prio),
            torch.cuda.Stream(device=index, priority=prio),
        )
    fork, router_done, latent_done = events
    router_stream, latent_stream = _entry_streams[index]
    fork.record()
    with torch.cuda.stream(router_stream):
        fork.wait()
        router_out = router()
        router_done.record()
    with torch.cuda.stream(latent_stream):
        fork.wait()
        latent_out = latent()
        latent_done.record()
    router_done.wait()
    latent_done.wait()
    return router_out, latent_out


def branch_streams(device: torch.device) -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
    """(router stream, latent stream) of ``device``, created once at the highest priority."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _streams:
        high = torch.cuda.Stream.priority_range()[1]
        _streams[index] = (
            torch.cuda.Stream(device=index, priority=high),
            torch.cuda.Stream(device=index, priority=high),
        )
    return _streams[index]


def _wrap(owner, name: str, before) -> None:
    orig = getattr(owner, name)

    def wrapped(*args, **kwargs):
        before(name)
        return orig(*args, **kwargs)

    setattr(owner, name, wrapped)


def arm_latent_join(routed_experts, event: torch.cuda.Event, rows: int) -> None:
    """Make the routed path wait on ``event`` right before its first read of the latent activations."""
    if not getattr(routed_experts, "_k3_latent_join_installed", False):

        def join(site: str) -> None:
            pending = routed_experts._k3_latent_join
            if pending is not None:
                routed_experts._k3_latent_join = None
                pending[0].wait()
                edges.debug("moe.streams", pending[1], f"branches, latent joined at {site}")

        backend = routed_experts.backend
        fused = type(backend).try_fused_route_quant
        if not fused.__qualname__.startswith("MoEImplBase."):
            _wrap(backend, "try_fused_route_quant", join)
        _wrap(backend, "quantize_input", join)
        _wrap(routed_experts.comm, "dispatch", join)
        routed_experts._k3_latent_join_installed = True
    routed_experts._k3_latent_join = (event, rows)


def finish_latent_join(routed_experts, event: torch.cuda.Event) -> None:
    """Join the latent stream on the current stream if no routed-path call did (keeps the capture joined)."""
    if routed_experts._k3_latent_join is not None:
        routed_experts._k3_latent_join = None
        event.wait()


def arm_shared_down(comm, down_proj, event: torch.cuda.Event, rows: int, at: str) -> None:
    """Record ``event`` at ``fc1_done`` (``at="fc2_end"``) or when the routed path calls ``comm.combine``;
    ``down_proj`` waits on it."""
    if not getattr(comm, "_k3_shared_down_installed", False):

        def record(site: str) -> None:
            pending = getattr(comm, "_k3_shared_down", None)
            if pending is not None and pending[1] is None:
                pending[0].record()
                pending[1] = "combine_start"
                _fc2_end.pending = None

        _wrap(comm, "combine", record)
        comm._k3_shared_down_installed = True
    pending = [event, None]
    comm._k3_shared_down = pending
    _fc2_end.pending = pending if at == "fc2_end" else None
    if not getattr(down_proj, "_k3_shared_down_installed", False):

        def wait(site: str) -> None:
            wait_shared_down(down_proj)

        _wrap(down_proj, "forward", wait)
        down_proj._k3_shared_down_installed = True
    down_proj._k3_shared_down = (pending, rows)


def wait_shared_down(down_proj) -> None:
    """The current stream waits for the event ``arm_shared_down`` armed on ``down_proj`` (once); also for callers
    that run the down projection without ``down_proj.forward``."""
    armed = getattr(down_proj, "_k3_shared_down", None)
    if armed is not None:
        down_proj._k3_shared_down = None
        armed[0][0].wait()
        edges.debug("moe.shared_down", armed[1], f"after {armed[0][1]}")


def fc1_done() -> None:
    """Called on the expert path's stream between FC1 and FC2: releases an ``fc2_end`` shared down-projection."""
    pending = getattr(_fc2_end, "pending", None)
    if pending is not None:
        _fc2_end.pending = None
        pending[0].record()
        pending[1] = "fc1 done (fc2_end)"
