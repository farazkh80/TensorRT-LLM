# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream placement of the Kimi K3 decode MoE layer (CUDA-graph multi-stream only).

The shared expert's down-projection waits until the routed path reaches ``comm.combine``, moving it from the expert
window into the combine window.

``entry_order`` orders the fused entry kernels (``moe_chain.entry``): up to ``ENTRY_UNDER_MAX_ROWS`` tokens the router
launches right after the pre-MoE mix and the latent down after it on the same stream (with ``mix_latent`` the latent
down starts under the mix); above, they run on two streams one priority level below the top priority, joined before
the routed path.

Placement changes only streams and event waits; kernels and their inputs are unchanged.
"""

from typing import Dict, Tuple

import torch

ENTRY_UNDER_MAX_ROWS = 16

_entry_streams: Dict[int, Tuple[torch.cuda.Stream, torch.cuda.Stream]] = {}


def entry_order(rows: int) -> str:
    return "under" if rows <= ENTRY_UNDER_MAX_ROWS else "branches"


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


def _wrap(owner, name: str, before) -> None:
    orig = getattr(owner, name)

    def wrapped(*args, **kwargs):
        before(name)
        return orig(*args, **kwargs)

    setattr(owner, name, wrapped)


def arm_shared_down(comm, down_proj, event: torch.cuda.Event, rows: int, at: str) -> None:
    """Record ``event`` when the routed path calls ``comm.combine`` (``at``, the only placement); ``down_proj`` waits
    on it."""
    if not getattr(comm, "_k3_shared_down_installed", False):

        def record(site: str) -> None:
            pending = getattr(comm, "_k3_shared_down", None)
            if pending is not None and pending[1] is None:
                pending[0].record()
                pending[1] = "combine_start"

        _wrap(comm, "combine", record)
        comm._k3_shared_down_installed = True
    pending = [event, None]
    comm._k3_shared_down = pending
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
