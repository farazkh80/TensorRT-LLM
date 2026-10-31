# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework-level producer/consumer scheduling for fine-grained synchronization.

A module brackets its forward with :meth:`StreamManager.region` and each
kernel group with :meth:`StreamManager.stage`. Stages run on a small pool of
lane streams; every named producer -> consumer edge has an :class:`EdgePolicy`
that decides whether the consumer waits for the producer to complete.

``ORACLE`` edges do not wait for the producer to complete. The consumer waits
only until the producer starts (for a producer with nested stages, until its
last nested stage starts, which is the kernel group that writes the output)
and reaches every point it marks with :meth:`StreamManager.mark_output`,
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

- ``TLLM_FGS_MODE``: ``off`` (default), ``standalone``, ``oracle`` or ``lamport``; the
  default policy of every edge.
- ``TLLM_FGS_EDGE_POLICIES``: comma-separated ``producer->consumer=policy``
  overrides.
- ``TLLM_FGS_NUM_LANES``: number of lane streams (default 2).
- ``TLLM_FGS_STAGES``: comma-separated stage names to schedule on lanes; any
  other stage runs inline on its calling stream (default: all stages).
- ``TLLM_FGS_ORACLE_FROM``: ``start`` (default) or ``none``; with ``none`` an
  ``ORACLE`` consumer does not wait for its producer at all.
- ``TLLM_FGS_ORACLE_ORDER``: ``priority`` (default) or ``none``. A start event
  only says the producer's stream reached the stage, not that its kernel got
  SMs, so a wide consumer could otherwise take the GPU before its producer.
  With ``priority`` each stage runs on a lane whose priority falls with its
  depth along ``ORACLE`` edges, so when both are pending the block scheduler
  dispatches the producer first and the consumer fills the SMs it leaves.
- ``TLLM_FGS_RESET_BACKEND``: ``auto`` (default), ``fill``, ``cute``,
  ``template`` or ``memset``; how :class:`EdgeWorkspaces` re-arms a slot (see
  :class:`ResetBackend`).
- ``TLLM_FGS_RESET_CTAS``, ``TLLM_FGS_RESET_THREADS``, ``TLLM_FGS_RESET_UNROLL``:
  grid bound and shape of the ``cute`` reset (default 0 = enough CTAs to cover
  the buffer in one pass, 256, 4).

``LAMPORT`` edges are scheduled like ``ORACLE`` edges that always wait for the
producer's start, but the data itself carries readiness: the producer writes
into a slot of an :class:`EdgeWorkspace` that holds the exact sentinel
(``-0.0``, i.e. ``0x8000`` per BF16/FP16 lane and ``0x80000000`` per FP32 lane),
and the consumer kernel retries tiles that still hold it. Kernels never write
the sentinel; :class:`EdgeWorkspaces` owns the slot lifecycle:

1. Registering a workspace allocates every slot once (never a caching-allocator
   temporary of the forward) and fills it with the sentinel before any producer
   runs; the calling stream waits for that fill.
2. :meth:`EdgeWorkspaces.acquire` hands the producer the next slot in
   round-robin order and a per-slot generation. If the slot was re-armed
   earlier in the open scope, the calling stream first waits for that reset.
   Acquiring a slot whose previous lease is unreleased is an error: two slots
   only cover lifetimes that do not overlap the next-but-one use.
3. :meth:`EdgeWorkspaces.release` is called once per declared reader, with the
   streams that read the slot. The last release also joins every stream that
   wrote it (the acquiring stream and any :meth:`EdgeLease.mark_written`
   stream), so a producer's trailing or padded writes cannot land after the
   reset. The reset is enqueued right away on the single reset stream, after
   those joins, and records the slot's reset-ready event.
4. :meth:`EdgeWorkspaces.scope` forks the reset stream from the calling stream
   on entry and joins it back on exit. At scope exit every slot holds the
   sentinel again and no reset is in flight, so a scope captured into a CUDA
   graph is self-contained: replays launched on one stream are ordered by that
   stream, a replay never clears a slot an earlier replay still reads, and every
   replay re-arms what it used. Slots restart at 0 in every scope, so warm-up,
   capture and replay use the same slot sequence. Generations are host-side
   capture-time labels; replays do not advance them.

A reset covers only the rows the lease handed out, so a producer must write
inside its lease. Views of one slot (e.g. column ranges read by different
consumers) are aliases of one physical region; it is re-armed only after the
last of its declared readers releases it. Committed KV history, recurrent and
convolution state are not edge workspaces and are never reset here.

Stream topology: the calling (compute) stream, one reset stream per
:class:`EdgeWorkspaces`, and, only while stages are scheduled, the manager's
lane and side streams. All streams and events are created before capture.
"""

from __future__ import annotations

import os
import threading
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Set, Tuple, Union

import torch

# Distinct lane priorities for ORACLE chain depths 0..3; deeper stages share the lowest one.
_PRIORITY_LEVELS = 4


class EdgePolicy(str, Enum):
    """How a consumer stage synchronizes with one of its producers."""

    STANDALONE = "standalone"
    ORACLE = "oracle"
    LAMPORT = "lamport"


def _relaxed(policy: EdgePolicy) -> bool:
    return policy is EdgePolicy.ORACLE or policy is EdgePolicy.LAMPORT


class ResetBackend(str, Enum):
    """How a workspace slot is re-armed with the sentinel."""

    # Device-to-device copy from a shared sentinel template (reads + writes).
    TEMPLATE = "template"
    # PyTorch fill of an integer view; its grid grows with the buffer.
    FILL = "fill"
    # CuTe DSL fill with 128-bit stores and a bounded grid.
    CUTE = "cute"
    # cuMemsetD16Async / cuMemsetD32Async.
    MEMSET = "memset"
    # CUTE for resets of at least EdgeWorkspaces.cute_min_bytes, FILL below.
    AUTO = "auto"


# Element dtype -> (integer lane dtype, lane sentinel, 32-bit word of sentinels).
_SENTINELS: Dict[torch.dtype, Tuple[torch.dtype, int, int]] = {
    torch.bfloat16: (torch.int16, -0x8000, -0x7FFF8000),
    torch.float16: (torch.int16, -0x8000, -0x7FFF8000),
    torch.float32: (torch.int32, -0x80000000, -0x80000000),
}


@dataclass(frozen=True)
class FineGrainedSyncConfig:
    default_policy: EdgePolicy = EdgePolicy.STANDALONE
    edge_policies: Mapping[str, EdgePolicy] = field(default_factory=dict)
    num_lanes: int = 2
    stages: Optional[FrozenSet[str]] = None
    oracle_waits_for_start: bool = True
    oracle_ordered: bool = True
    reset_backend: ResetBackend = ResetBackend.AUTO
    reset_ctas: int = 0
    reset_threads: int = 256
    reset_unroll: int = 4

    def __post_init__(self) -> None:
        if self.num_lanes < 1:
            raise ValueError(f"num_lanes must be positive, got {self.num_lanes}")
        if self.reset_ctas < 0:
            raise ValueError(f"reset_ctas must be non-negative, got {self.reset_ctas}")

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
        order = os.environ.get("TLLM_FGS_ORACLE_ORDER", "priority").strip().lower()
        if order not in ("priority", "none"):
            raise ValueError(f"TLLM_FGS_ORACLE_ORDER must be priority or none, got {order!r}")
        return cls(
            default_policy=EdgePolicy(mode),
            edge_policies=edge_policies,
            num_lanes=int(os.environ.get("TLLM_FGS_NUM_LANES", "2")),
            stages=frozenset(s.strip() for s in stages.split(",")) if stages else None,
            oracle_waits_for_start=oracle_from == "start",
            oracle_ordered=order == "priority",
            reset_backend=ResetBackend(
                os.environ.get("TLLM_FGS_RESET_BACKEND", "auto").strip().lower()
            ),
            reset_ctas=int(os.environ.get("TLLM_FGS_RESET_CTAS", "0")),
            reset_threads=int(os.environ.get("TLLM_FGS_RESET_THREADS", "256")),
            reset_unroll=int(os.environ.get("TLLM_FGS_RESET_UNROLL", "4")),
        )

    @property
    def uses_lamport(self) -> bool:
        return self.default_policy is EdgePolicy.LAMPORT or any(
            p is EdgePolicy.LAMPORT for p in self.edge_policies.values()
        )


class EdgeLease:
    """One producer's use of one workspace slot, from acquire to its last release.

    Unpacks as ``(tensor, slot, gen)``.
    """

    def __init__(
        self,
        workspace: "EdgeWorkspace",
        slot: int,
        gen: int,
        tensor: torch.Tensor,
        readers: int,
        stream: torch.cuda.Stream,
    ) -> None:
        self.workspace = workspace
        self.slot = slot
        self.gen = gen
        self.tensor = tensor
        self.pending_readers = readers
        self.released = False
        self._acquire_stream = stream
        # stream handle -> index into the slot's preallocated join events.
        self._joins: Dict[int, int] = {}

    def __iter__(self) -> Iterator[object]:
        return iter((self.tensor, self.slot, self.gen))

    def view(self, alias: str) -> torch.Tensor:
        dim, start, length = self.workspace.aliases[alias]
        return self.tensor.narrow(dim, start, length)

    def mark_written(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Record that ``stream`` (default: current) finished writing this slot.

        Call it after a writer issued on a stream other than the acquiring one;
        the reset then waits for that point.
        """
        self._join(stream or torch.cuda.current_stream())

    def wait_ready(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Order ``stream`` after the reset that armed this slot, if one ran in this scope."""
        if self.released:
            raise RuntimeError(f"{self._label()} was released")
        if self.workspace._pending[self.slot]:
            (stream or torch.cuda.current_stream()).wait_event(self.workspace._ready[self.slot])

    def _label(self) -> str:
        return f"edge {self.workspace.name} slot {self.slot} gen {self.gen}"

    def _join(self, stream: torch.cuda.Stream) -> None:
        if self.released:
            raise RuntimeError(f"{self._label()} was released")
        events = self.workspace._joins[self.slot]
        index = self._joins.get(stream.cuda_stream)
        if index is None:
            if len(self._joins) == len(events):
                raise RuntimeError(
                    f"{self._label()}: more than {len(events)} streams touched the slot"
                )
            index = self._joins[stream.cuda_stream] = len(self._joins)
        events[index].record(stream)


class EdgeWorkspace:
    """Persistent sentinel-armed slots of one edge (one physical region per slot)."""

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        slots: int,
        backend: ResetBackend,
        reset_ctas: int,
        device: torch.device,
        max_streams: int,
    ) -> None:
        if dtype not in _SENTINELS:
            raise ValueError(f"edge {name}: no sentinel for {dtype}")
        if slots < 1:
            raise ValueError(f"edge {name}: slots must be positive, got {slots}")
        self.name = name
        self.shape = tuple(shape)
        self.dtype = dtype
        self.backend = backend
        self.reset_ctas = reset_ctas
        self.aliases: Dict[str, Tuple[int, int, int]] = {}
        self.slots = [torch.empty(self.shape, dtype=dtype, device=device) for _ in range(slots)]
        self._ready = [torch.cuda.Event() for _ in range(slots)]
        self._joins = [[torch.cuda.Event() for _ in range(max_streams)] for _ in range(slots)]
        self._gen = [0] * slots
        self._lease: List[Optional[EdgeLease]] = [None] * slots
        # True once the slot's reset was enqueued in the open scope.
        self._pending = [False] * slots
        self._cursor = 0

    @property
    def num_slots(self) -> int:
        return len(self.slots)

    @property
    def bytes_per_slot(self) -> int:
        return self.slots[0].numel() * self.slots[0].element_size()

    def define_alias(self, alias: str, dim: int, start: int, length: int) -> None:
        """Name a range of one dimension of every slot, e.g. the gate columns of a fused output."""
        dim = dim % len(self.shape)
        if dim == 0:
            raise ValueError(f"edge {self.name}: dim 0 is the leased row range, alias another dim")
        if start < 0 or length <= 0 or start + length > self.shape[dim]:
            raise ValueError(
                f"edge {self.name}: alias {alias} [{start}, {start + length}) is outside dim {dim}"
            )
        self.aliases[alias] = (dim, start, length)

    def generation(self, slot: int) -> int:
        return self._gen[slot]

    def in_use(self, slot: int) -> bool:
        return self._lease[slot] is not None


class EdgeWorkspaces:
    """Owns edge workspaces, their reset stream and the scope that brackets a forward.

    Not thread-safe; use one instance per thread (the stream manager is
    thread-local).
    """

    def __init__(
        self,
        backend: ResetBackend = ResetBackend.AUTO,
        reset_ctas: int = 0,
        reset_threads: int = 256,
        reset_unroll: int = 4,
        device: Optional[torch.device] = None,
        max_streams_per_slot: int = 8,
        cute_min_bytes: int = 4 << 20,
    ) -> None:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.backend = ResetBackend(backend)
        self.reset_ctas = reset_ctas
        self.reset_threads = reset_threads
        self.reset_unroll = reset_unroll
        self.max_streams_per_slot = max_streams_per_slot
        self.cute_min_bytes = cute_min_bytes
        self.reset_stream = torch.cuda.Stream(device=self.device)
        self._workspaces: Dict[str, EdgeWorkspace] = {}
        # Integer lane dtype -> uint8 template of the largest TEMPLATE slot.
        self._templates: Dict[torch.dtype, torch.Tensor] = {}
        # Outgrown templates stay alive (as int views): captured graphs may still copy from them.
        self._retired: List[torch.Tensor] = []
        self._origin: Optional[torch.cuda.Stream] = None

    @property
    def in_scope(self) -> bool:
        return self._origin is not None

    def __getitem__(self, name: str) -> EdgeWorkspace:
        return self._workspaces[name]

    def __contains__(self, name: str) -> bool:
        return name in self._workspaces

    def workspace(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.bfloat16,
        slots: int = 2,
        backend: Optional[ResetBackend] = None,
        reset_ctas: Optional[int] = None,
    ) -> EdgeWorkspace:
        """Register (or return the matching) workspace and arm all its slots.

        Dim 0 is the row dimension leases can truncate; size it for the
        largest token count the edge serves.
        """
        existing = self._workspaces.get(name)
        if existing is not None:
            if (existing.shape, existing.dtype, existing.num_slots) != (tuple(shape), dtype, slots):
                raise ValueError(
                    f"edge {name} is registered as {existing.shape} {existing.dtype} "
                    f"x{existing.num_slots}"
                )
            return existing
        if torch.cuda.is_current_stream_capturing() or self.in_scope:
            raise RuntimeError(f"register edge {name} before capture and outside a scope")
        ws = EdgeWorkspace(
            name,
            shape,
            dtype,
            slots,
            ResetBackend(backend or self.backend),
            self.reset_ctas if reset_ctas is None else reset_ctas,
            self.device,
            self.max_streams_per_slot,
        )
        if ws.backend is ResetBackend.TEMPLATE:
            self._ensure_template(_SENTINELS[dtype][0], ws.bytes_per_slot)
        self._workspaces[name] = ws
        self.preinitialize((ws,))
        return ws

    def preinitialize(self, workspaces: Optional[Iterable[EdgeWorkspace]] = None) -> None:
        """Fill every slot with the sentinel; the calling stream waits for it."""
        if torch.cuda.is_current_stream_capturing() or self.in_scope:
            raise RuntimeError("preinitialize before capture and outside a scope")
        workspaces = list(self._workspaces.values() if workspaces is None else workspaces)
        for ws in workspaces:
            busy = [slot for slot in range(ws.num_slots) if ws.in_use(slot)]
            if busy:
                raise RuntimeError(f"edge {ws.name}: slots {busy} are still leased")
        origin = torch.cuda.current_stream(self.device)
        self.reset_stream.wait_stream(origin)
        with torch.cuda.stream(self.reset_stream):
            for ws in workspaces:
                for slot, tensor in enumerate(ws.slots):
                    self._fill(ws, tensor)
                    # Creates each event before any capture can record it.
                    ws._ready[slot].record(self.reset_stream)
                    for event in ws._joins[slot]:
                        event.record(self.reset_stream)
        origin.wait_stream(self.reset_stream)

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Bracket one forward: fork the reset stream on entry, join it on exit.

        Nested scopes are no-ops; the outermost one owns the fork and join.
        """
        if self.in_scope:
            yield
            return
        origin = torch.cuda.current_stream(self.device)
        self.reset_stream.wait_stream(origin)
        self._origin = origin
        for ws in self._workspaces.values():
            ws._cursor = 0
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            leaked = self._drain_leases(origin)
            origin.wait_stream(self.reset_stream)
            self._origin = None
            for ws in self._workspaces.values():
                ws._pending = [False] * ws.num_slots
            if leaked and not failed:
                raise RuntimeError(f"scope ended with unreleased edge slots: {leaked}")

    def acquire(self, name: str, rows: Optional[int] = None, readers: int = 1) -> EdgeLease:
        """Hand the producer on the current stream the next armed slot of ``name``.

        ``rows`` truncates dim 0 (the reset then covers only those rows);
        ``readers`` is the number of :meth:`release` calls that free the slot.
        """
        if not self.in_scope:
            raise RuntimeError(f"acquire({name!r}) needs an open EdgeWorkspaces.scope()")
        if readers < 1:
            raise ValueError(f"edge {name}: readers must be positive, got {readers}")
        ws = self._workspaces[name]
        slot = ws._cursor
        held = ws._lease[slot]
        if held is not None:
            raise RuntimeError(
                f"edge {name}: slot {slot} is still held by gen {held.gen}; "
                f"its lifetime needs more than {ws.num_slots} slots"
            )
        ws._cursor = (slot + 1) % ws.num_slots
        ws._gen[slot] += 1
        stream = torch.cuda.current_stream(self.device)
        if ws._pending[slot]:
            stream.wait_event(ws._ready[slot])
        tensor = ws.slots[slot]
        if rows is not None:
            if not 0 < rows <= ws.shape[0]:
                raise ValueError(f"edge {name}: rows {rows} outside (0, {ws.shape[0]}]")
            tensor = tensor[:rows]
        lease = EdgeLease(ws, slot, ws._gen[slot], tensor, readers, stream)
        ws._lease[slot] = lease
        return lease

    def release(
        self,
        edge: Union[EdgeLease, str],
        slot: Optional[int] = None,
        gen: Optional[int] = None,
        streams: Optional[Iterable[torch.cuda.Stream]] = None,
    ) -> None:
        """Release one reader of a slot; the last release enqueues its reset.

        ``edge`` is the lease, or the edge name with ``slot`` and ``gen``.
        ``streams`` (default: the current stream) are the streams this reader
        ran on; each is joined before the reset.
        """
        lease = edge if isinstance(edge, EdgeLease) else self._lookup(edge, slot, gen)
        ws = lease.workspace
        if lease.released or ws._lease[lease.slot] is not lease:
            raise RuntimeError(f"edge {ws.name}: slot {lease.slot} gen {lease.gen} is not held")
        if not self.in_scope:
            raise RuntimeError(f"release of edge {ws.name} needs an open EdgeWorkspaces.scope()")
        for stream in streams if streams is not None else (torch.cuda.current_stream(self.device),):
            lease._join(stream)
        lease.pending_readers -= 1
        if lease.pending_readers == 0:
            self._recycle(lease)

    def ledger(self) -> List[Dict[str, object]]:
        """Device memory owned by the workspaces: one row per edge and per template."""
        rows: List[Dict[str, object]] = []
        for ws in self._workspaces.values():
            rows.append(
                {
                    "name": ws.name,
                    "shape": ws.shape,
                    "dtype": str(ws.dtype).replace("torch.", ""),
                    "slots": ws.num_slots,
                    "bytes_per_slot": ws.bytes_per_slot,
                    "bytes": ws.bytes_per_slot * ws.num_slots,
                    "backend": ws.backend.value,
                    "aliases": dict(ws.aliases),
                    "events": ws.num_slots * (1 + self.max_streams_per_slot),
                }
            )
        templates = [(lane, template, "") for lane, template in self._templates.items()]
        templates += [(template.dtype, template, " retired") for template in self._retired]
        for lane, template, suffix in templates:
            rows.append(
                {
                    "name": f"template[{str(lane).replace('torch.', '')}]{suffix}",
                    "bytes": template.numel() * template.element_size(),
                    "backend": ResetBackend.TEMPLATE.value,
                }
            )
        return rows

    def total_bytes(self) -> int:
        return sum(int(row["bytes"]) for row in self.ledger())

    def _lookup(self, name: str, slot: Optional[int], gen: Optional[int]) -> EdgeLease:
        ws = self._workspaces[name]
        if slot is None or gen is None:
            raise ValueError(f"release({name!r}) by name needs slot and gen")
        lease = ws._lease[slot]
        if lease is None or lease.gen != gen:
            held = None if lease is None else lease.gen
            raise RuntimeError(f"edge {name}: slot {slot} gen {gen} is not held (held gen: {held})")
        return lease

    def _recycle(self, lease: EdgeLease) -> None:
        ws = lease.workspace
        if lease._acquire_stream.cuda_stream not in lease._joins:
            lease._join(lease._acquire_stream)
        events = ws._joins[lease.slot]
        for index in lease._joins.values():
            self.reset_stream.wait_event(events[index])
        with torch.cuda.stream(self.reset_stream):
            self._fill(ws, lease.tensor)
        ws._ready[lease.slot].record(self.reset_stream)
        ws._pending[lease.slot] = True
        ws._lease[lease.slot] = None
        lease.released = True

    def _drain_leases(self, origin: torch.cuda.Stream) -> List[Tuple[str, int, int]]:
        leaked = []
        for ws in self._workspaces.values():
            for lease in ws._lease:
                if lease is None:
                    continue
                leaked.append((ws.name, lease.slot, lease.gen))
                lease._joins.clear()
                lease._join(origin)
                self._recycle(lease)
        return leaked

    def _ensure_template(self, lane: torch.dtype, nbytes: int) -> None:
        template = self._templates.get(lane)
        if template is not None:
            if template.numel() >= nbytes:
                return
            self._retired.append(template.view(lane))
        template = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        template.view(lane).fill_(-0x8000 if lane == torch.int16 else -0x80000000)
        self._templates[lane] = template

    def _fill(self, ws: EdgeWorkspace, tensor: torch.Tensor) -> None:
        lane, value, word = _SENTINELS[ws.dtype]
        backend = ws.backend
        if backend is ResetBackend.AUTO:
            nbytes = tensor.numel() * tensor.element_size()
            backend = ResetBackend.CUTE if nbytes >= self.cute_min_bytes else ResetBackend.FILL
        if backend is ResetBackend.FILL:
            tensor.view(lane).fill_(value)
        elif backend is ResetBackend.TEMPLATE:
            nbytes = tensor.numel() * tensor.element_size()
            tensor.view(-1).view(torch.uint8).copy_(self._templates[lane][:nbytes])
        elif backend is ResetBackend.CUTE:
            from .fine_grained_sync_fill import sentinel_fill_

            max_ctas = ws.reset_ctas or (1 << 30)
            sentinel_fill_(tensor, word, max_ctas, self.reset_threads, self.reset_unroll)
        elif backend is ResetBackend.MEMSET:
            _memset_sentinel(tensor, lane)
        else:
            raise ValueError(f"unsupported reset backend {backend}")


def _memset_sentinel(tensor: torch.Tensor, lane: torch.dtype) -> None:
    from cuda.bindings import driver

    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    pointer = driver.CUdeviceptr(tensor.data_ptr())
    if lane == torch.int16:
        result = driver.cuMemsetD16Async(pointer, 0x8000, tensor.numel(), stream)
    else:
        result = driver.cuMemsetD32Async(pointer, 0x80000000, tensor.numel(), stream)
    error = result[0] if isinstance(result, tuple) else result
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"sentinel memset failed: {error}")


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

    ``LAMPORT`` edges hand out slots of :attr:`edges` through :meth:`acquire`
    and :meth:`release`; a region opens an :meth:`EdgeWorkspaces.scope` unless
    one is already open, so open one around the whole forward to keep the
    reset join off every region's exit.
    """

    def __init__(
        self, config: FineGrainedSyncConfig, device: Optional[torch.device] = None
    ) -> None:
        self.config = config
        # num_lanes lanes per priority level; level 0 has the highest priority.
        self._levels = _PRIORITY_LEVELS if config.oracle_ordered else 1
        self._lane_level = [lvl for lvl in range(self._levels) for _ in range(config.num_lanes)]
        self._lanes = [
            torch.cuda.Stream(device=device, priority=lvl - (self._levels - 1))
            for lvl in self._lane_level
        ]
        self._side = [
            torch.cuda.Stream(device=device, priority=lvl - (self._levels - 1))
            for lvl in range(self._levels)
        ]
        self._depth: Dict[str, int] = {}
        self._done_events: Dict[str, torch.cuda.Event] = {}
        # Start event, then one per mark_output() of the stage's latest run.
        self._release_events: Dict[str, List[torch.cuda.Event]] = {}
        self._num_released: Dict[str, int] = {}
        self._last_child: Dict[str, str] = {}
        self._stage_lane: Dict[str, int] = {}
        self._origin: Optional[torch.cuda.Stream] = None
        self._recorded: Set[str] = set()
        self._kept: List[torch.Tensor] = []
        self._graph_kept: List[torch.Tensor] = []
        self._open_stages: List[Tuple[int, List[str]]] = []
        self._open_info: List[Tuple[str, int]] = []
        self._lane_last_use = [0] * len(self._lanes)
        self._clock = 0
        self._edges: Optional[EdgeWorkspaces] = None
        if config.uses_lamport:
            self._edges = self._make_edges(device)

    def _make_edges(self, device: Optional[torch.device]) -> EdgeWorkspaces:
        return EdgeWorkspaces(
            backend=self.config.reset_backend,
            reset_ctas=self.config.reset_ctas,
            reset_threads=self.config.reset_threads,
            reset_unroll=self.config.reset_unroll,
            device=device,
        )

    @property
    def edges(self) -> EdgeWorkspaces:
        if self._edges is None:
            self._edges = self._make_edges(None)
        return self._edges

    def acquire(self, name: str, rows: Optional[int] = None, readers: int = 1) -> EdgeLease:
        """:meth:`EdgeWorkspaces.acquire`, then hold relaxed consumers of the open stage until here.

        The mark orders those consumers after any wait for the slot's reset.
        """
        lease = self.edges.acquire(name, rows, readers)
        self.mark_output()
        return lease

    def release(
        self, lease: EdgeLease, streams: Optional[Iterable[torch.cuda.Stream]] = None
    ) -> None:
        self.edges.release(lease, streams=streams)

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
        edge_scope = ExitStack()
        if self._edges is not None:
            edge_scope.enter_context(self._edges.scope())
        try:
            yield
        finally:
            # Forking or joining an idle lane would carry stale capture
            # dependencies across regions, which multiplies graph edges and
            # stalls the replay.
            for index in {self._stage_lane[name] for name in self._recorded}:
                origin.wait_stream(self._lanes[index])
            edge_scope.close()
            if torch.cuda.is_current_stream_capturing():
                self._graph_kept.extend(self._kept)
            self._origin = None
            self._recorded.clear()
            self._kept.clear()
            self._open_stages.clear()
            self._open_info.clear()

    @contextmanager
    def stage(self, name: str, after: Tuple[str, ...] = ()) -> Iterator[None]:
        if not self.active:
            raise RuntimeError(f"stage {name!r} requires an active region")
        enclosing = torch.cuda.current_stream()
        if self.config.stages is not None and name not in self.config.stages:
            for producer in after:
                if producer not in self._recorded:
                    continue
                policy = self.config.policy(producer, name)
                if policy is EdgePolicy.LAMPORT:
                    self._wait_released(enclosing, producer)
                elif policy is not EdgePolicy.ORACLE:
                    enclosing.wait_event(self._done_events[producer])
            yield
            return
        depth = self._open_info[-1][1] if self._open_info else 0
        for producer in after:
            if producer in self._recorded:
                relaxed = _relaxed(self.config.policy(producer, name))
                depth = max(depth, self._depth[producer] + relaxed)
        lane_index = self._pick_lane(name, after, min(depth, self._levels - 1))
        lane = self._lanes[lane_index]
        lane.wait_stream(enclosing)
        for producer in after:
            if producer not in self._recorded:
                continue
            policy = self.config.policy(producer, name)
            if _relaxed(policy):
                if policy is EdgePolicy.LAMPORT or self.config.oracle_waits_for_start:
                    self._wait_released(lane, producer)
                continue
            if self._stage_lane[producer] != lane_index:
                lane.wait_event(self._done_events[producer])
        events = self._release_events.get(name)
        if events is None:
            events = self._release_events[name] = [torch.cuda.Event()]
        events[0].record(lane)
        self._num_released[name] = 1
        self._open_stages.append((lane_index, []))
        self._open_info.append((name, depth))
        with torch.cuda.stream(lane):
            yield
        children = self._open_stages.pop()[1]
        self._open_info.pop()
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
        self._depth[name] = depth
        self._recorded.add(name)
        if self._open_stages:
            self._open_stages[-1][1].append(name)

    def _wait_released(self, stream: torch.cuda.Stream, producer: str) -> None:
        source = producer
        while source in self._last_child:
            source = self._last_child[source]
        for event in self._release_events[source][: self._num_released[source]]:
            stream.wait_event(event)

    def _pick_lane(self, name: str, after: Tuple[str, ...], level: int) -> int:
        # A consumer on its producer's lane is serialized by stream order, which
        # would silently turn a relaxed edge into a STANDALONE one. A STANDALONE
        # consumer instead reuses its producer's lane to avoid a cross-lane wait.
        # An enclosing stage's lane is excluded for the same reason: the nested
        # stage waits for that lane, which may hold an earlier nested producer.
        producers = [p for p in after if p in self._recorded]
        excluded = {self._stage_lane[p] for p in producers if _relaxed(self.config.policy(p, name))}
        excluded.update(lane for lane, _ in self._open_stages)
        pool = [i for i in range(len(self._lanes)) if self._lane_level[i] == level]
        standalone_lanes = [
            self._stage_lane[p]
            for p in producers
            if self._stage_lane[p] in pool and self._stage_lane[p] not in excluded
        ]
        if standalone_lanes:
            lane_index = standalone_lanes[0]
        else:
            candidates = [i for i in pool if i not in excluded]
            lane_index = min(candidates or pool, key=lambda i: self._lane_last_use[i])
        self._clock += 1
        self._lane_last_use[lane_index] = self._clock
        return lane_index

    def mark_output(self) -> None:
        """Hold the open stage's ``ORACLE`` consumers until this point too.

        Call it right before an output kernel that is not the stage's first, on
        the stream that launches it (e.g. a norm after a residual mix, or a GEMM
        on a side stream); consumers wait for the stage's start and every mark.
        """
        if not self._open_info:
            return
        name = self._open_info[-1][0]
        events, count = self._release_events[name], self._num_released[name]
        if count == len(events):
            events.append(torch.cuda.Event())
        events[count].record(torch.cuda.current_stream())
        self._num_released[name] = count + 1

    def side_stream(self) -> Optional[torch.cuda.Stream]:
        """A stream at the open stage's priority for work forked off its lane; None outside a stage.

        Forked work on a default-priority stream would lose SMs to every
        ``ORACLE`` consumer and could run after them.
        """
        if not self._open_stages:
            return None
        return self._side[self._lane_level[self._open_stages[-1][0]]]

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
