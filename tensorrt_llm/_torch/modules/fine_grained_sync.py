# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework-level producer/consumer scheduling for fine-grained synchronization.

A module brackets its forward with :meth:`StreamManager.region` and each
kernel group with :meth:`StreamManager.stage`. Stages run on a small pool of
lane streams; every named producer -> consumer edge has an :class:`EdgePolicy`
that decides whether the consumer waits for the producer to complete.

``ORACLE`` edges do not wait for the producer to complete. The consumer waits
only until the producer starts (for a producer with nested stages, until its
last nested stage starts, which is the kernel group that writes the output),
then reads the output buffer as if every tile were already valid. This is the
best-case overlap a fine-grained-sync (data-as-flag) consumer could reach with
zero validity-check and buffer re-arm cost; waiting for the start keeps the
consumer ordered after the producer's own inputs. Numerics are only meaningful
when the buffer already holds the values the producer is about to write, e.g.
CUDA graph replays of a fixed input, and never for an edge whose consumer
updates the buffer in place. The producer must write a persistent buffer: a
caching-allocator block may be recycled from an earlier stage's temporary,
which rewrites it every replay before the producer runs, so the consumer would
read that temporary instead.

Configuration is read from the environment when the manager is first used:

- ``TLLM_FGS_MODE``: ``off`` (default), ``standalone`` or ``oracle``; the
  default policy of every edge.
- ``TLLM_FGS_EDGE_POLICIES``: comma-separated ``producer->consumer=policy``
  overrides.
- ``TLLM_FGS_NUM_LANES``: number of lane streams (default 2).
- ``TLLM_FGS_STAGES``: comma-separated stage names to schedule on lanes; any
  other stage runs inline on its calling stream (default: all stages).
- ``TLLM_FGS_ORACLE_FROM``: ``start`` (default) or ``none``; with ``none`` an
  ``ORACLE`` consumer does not wait for its producer at all.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterator, List, Mapping, Optional, Set, Tuple

import torch


class EdgePolicy(str, Enum):
    """How a consumer stage synchronizes with one of its producers."""

    STANDALONE = "standalone"
    ORACLE = "oracle"


@dataclass(frozen=True)
class FineGrainedSyncConfig:
    default_policy: EdgePolicy = EdgePolicy.STANDALONE
    edge_policies: Mapping[str, EdgePolicy] = field(default_factory=dict)
    num_lanes: int = 2
    stages: Optional[FrozenSet[str]] = None
    oracle_waits_for_start: bool = True

    def __post_init__(self) -> None:
        if self.num_lanes < 1:
            raise ValueError(f"num_lanes must be positive, got {self.num_lanes}")

    def policy(self, producer: str, consumer: str) -> EdgePolicy:
        return self.edge_policies.get(f"{producer}->{consumer}", self.default_policy)

    @classmethod
    def from_env(cls) -> Optional["FineGrainedSyncConfig"]:
        mode = os.environ.get("TLLM_FGS_MODE", "off").strip().lower()
        if mode in ("", "off"):
            return None
        edge_policies = {}
        for item in os.environ.get("TLLM_FGS_EDGE_POLICIES", "").split(","):
            if not item.strip():
                continue
            edge, policy = item.rsplit("=", 1)
            edge_policies[edge.strip()] = EdgePolicy(policy.strip().lower())
        stages = os.environ.get("TLLM_FGS_STAGES", "").strip()
        oracle_from = os.environ.get("TLLM_FGS_ORACLE_FROM", "start").strip().lower()
        if oracle_from not in ("start", "none"):
            raise ValueError(f"TLLM_FGS_ORACLE_FROM must be start or none, got {oracle_from!r}")
        return cls(
            default_policy=EdgePolicy(mode),
            edge_policies=edge_policies,
            num_lanes=int(os.environ.get("TLLM_FGS_NUM_LANES", "2")),
            stages=frozenset(s.strip() for s in stages.split(",")) if stages else None,
            oracle_waits_for_start=oracle_from == "start",
        )


class StreamManager:
    """Schedule named stages of one region across lane streams.

    Each stage forks its lane from the calling (origin) stream, and the region
    joins every lane that ran a stage back on exit, so it is valid inside CUDA
    graph capture. Work issued inside
    the region but outside any stage stays on the origin stream; every stage
    orders itself after the origin stream's work issued so far, so such work
    may produce stage inputs but must not consume stage outputs.

    A stage opened inside another stage treats the enclosing stage's stream as
    its origin, and the enclosing stage completes only after every stage nested
    in it, so an edge from or to the enclosing stage covers the nested work.
    Nested stages avoid the lanes of the stages enclosing them, so an ``ORACLE``
    edge between two nested stages needs at least two lanes beyond those.

    Tensors passed to :meth:`keep` stay referenced until the region joins, or
    for the manager's lifetime when the region is captured into a CUDA graph.
    Under ``ORACLE`` edges a consumer may still read a buffer after Python
    dropped it, and the caching allocator would otherwise hand that block to a
    later allocation on the producer's lane. A captured ``ORACLE`` consumer
    reads the buffer left by the previous replay, so no other allocation in
    the graph may reuse it.
    """

    def __init__(
        self, config: FineGrainedSyncConfig, device: Optional[torch.device] = None
    ) -> None:
        self.config = config
        self._lanes = [torch.cuda.Stream(device=device) for _ in range(config.num_lanes)]
        self._done_events: Dict[str, torch.cuda.Event] = {}
        self._start_events: Dict[str, torch.cuda.Event] = {}
        self._last_child: Dict[str, str] = {}
        self._stage_lane: Dict[str, int] = {}
        self._origin: Optional[torch.cuda.Stream] = None
        self._recorded: Set[str] = set()
        self._kept: List[torch.Tensor] = []
        self._graph_kept: List[torch.Tensor] = []
        self._open_stages: List[Tuple[int, List[str]]] = []
        self._lane_last_use = [0] * len(self._lanes)
        self._clock = 0

    @property
    def active(self) -> bool:
        return self._origin is not None

    @contextmanager
    def region(self) -> Iterator[None]:
        if self.active:
            raise RuntimeError("fine-grained sync regions cannot nest")
        origin = torch.cuda.current_stream()
        self._origin = origin
        self._lane_last_use = [0] * len(self._lanes)
        self._clock = 0
        try:
            yield
        finally:
            # Forking or joining an idle lane would carry stale capture
            # dependencies across regions, which multiplies graph edges and
            # stalls the replay.
            for index in {self._stage_lane[name] for name in self._recorded}:
                origin.wait_stream(self._lanes[index])
            if torch.cuda.is_current_stream_capturing():
                self._graph_kept.extend(self._kept)
            self._origin = None
            self._recorded.clear()
            self._kept.clear()
            self._open_stages.clear()

    @contextmanager
    def stage(self, name: str, after: Tuple[str, ...] = ()) -> Iterator[None]:
        if not self.active:
            raise RuntimeError(f"stage {name!r} requires an active region")
        enclosing = torch.cuda.current_stream()
        if self.config.stages is not None and name not in self.config.stages:
            for producer in after:
                if (
                    producer in self._recorded
                    and self.config.policy(producer, name) is not EdgePolicy.ORACLE
                ):
                    enclosing.wait_event(self._done_events[producer])
            yield
            return
        lane_index = self._pick_lane(name, after)
        lane = self._lanes[lane_index]
        lane.wait_stream(enclosing)
        for producer in after:
            if producer not in self._recorded:
                continue
            if self.config.policy(producer, name) is EdgePolicy.ORACLE:
                if self.config.oracle_waits_for_start:
                    source = producer
                    while source in self._last_child:
                        source = self._last_child[source]
                    lane.wait_event(self._start_events[source])
                continue
            if self._stage_lane[producer] != lane_index:
                lane.wait_event(self._done_events[producer])
        start = self._start_events.get(name)
        if start is None:
            start = self._start_events[name] = torch.cuda.Event()
        start.record(lane)
        self._open_stages.append((lane_index, []))
        with torch.cuda.stream(lane):
            yield
        children = self._open_stages.pop()[1]
        for child in children:
            if self._stage_lane[child] != lane_index:
                lane.wait_event(self._done_events[child])
        if children:
            self._last_child[name] = children[-1]
        else:
            self._last_child.pop(name, None)
        event = self._done_events.get(name)
        if event is None:
            event = self._done_events[name] = torch.cuda.Event()
        event.record(lane)
        self._stage_lane[name] = lane_index
        self._recorded.add(name)
        if self._open_stages:
            self._open_stages[-1][1].append(name)

    def _pick_lane(self, name: str, after: Tuple[str, ...]) -> int:
        # A consumer on its producer's lane is serialized by stream order, which
        # would silently turn an ORACLE edge into a STANDALONE one. A STANDALONE
        # consumer instead reuses its producer's lane to avoid a cross-lane wait.
        # An enclosing stage's lane is excluded for the same reason: the nested
        # stage waits for that lane, which may hold an earlier nested producer.
        producers = [p for p in after if p in self._recorded]
        excluded = {
            self._stage_lane[p]
            for p in producers
            if self.config.policy(p, name) is EdgePolicy.ORACLE
        }
        excluded.update(lane for lane, _ in self._open_stages)
        standalone_lanes = [
            self._stage_lane[p] for p in producers if self._stage_lane[p] not in excluded
        ]
        if standalone_lanes:
            lane_index = standalone_lanes[0]
        else:
            candidates = [i for i in range(len(self._lanes)) if i not in excluded]
            lane_index = min(
                candidates or range(len(self._lanes)), key=lambda i: self._lane_last_use[i]
            )
        self._clock += 1
        self._lane_last_use[lane_index] = self._clock
        return lane_index

    def keep(self, *tensors: Optional[torch.Tensor]) -> None:
        self._kept.extend(t for t in tensors if isinstance(t, torch.Tensor))


_local = threading.local()


def configure_fine_grained_sync(config: Optional[FineGrainedSyncConfig]) -> None:
    """Install (or with ``None`` disable) the calling thread's stream manager."""
    _local.manager = StreamManager(config) if config is not None else None


def get_stream_manager() -> Optional[StreamManager]:
    if not hasattr(_local, "manager"):
        configure_fine_grained_sync(FineGrainedSyncConfig.from_env())
    return _local.manager


def active_stream_manager() -> Optional[StreamManager]:
    """Return the stream manager only while one of its regions is open."""
    manager = get_stream_manager()
    return manager if manager is not None and manager.active else None


def maybe_stage(manager: Optional[StreamManager], name: str, after: Tuple[str, ...] = ()):
    return manager.stage(name, after) if manager is not None else nullcontext()
