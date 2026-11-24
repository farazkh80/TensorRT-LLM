# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sentinel-armed edge workspaces for fine-grained synchronization.

A producer writes into a slot of an :class:`EdgeWorkspace` that holds the exact sentinel (``-0.0``, i.e. ``0x8000``
per BF16/FP16 lane, ``0x80000000`` per FP32 lane and ``0x80`` per E4M3 byte; ``0xFF``, the NaN encoding, per UE8M0
scale byte held as uint8; zero per int32 counter word), and the consumer kernel retries tiles that still hold it.
Kernels never write the sentinel; :class:`EdgeWorkspaces` owns the slot lifecycle:

1. Registering a workspace allocates every slot once (never a caching-allocator temporary of the forward) and fills
   it with the sentinel before any producer runs; the calling stream waits for that fill.
2. :meth:`EdgeWorkspaces.acquire` hands the producer the next slot in round-robin order and a per-slot generation.
   If the slot was re-armed earlier in the open scope, the calling stream first waits for that reset. A slot is
   never reused within a scope before a flush, so a workspace needs as many slots as uses per step (e.g. one per
   layer).
3. :meth:`EdgeWorkspaces.release` is called once per declared reader, with the streams that read the slot. The last
   release marks the slot dirty; :meth:`EdgeWorkspaces.flush` waits for the joins of every dirty slot (the acquiring
   stream and every reader stream, so a producer's trailing or padded writes cannot land after the reset) and
   re-arms them all with one batched fill on the single reset stream (one launch per ``MAX_SPANS`` contiguous
   spans). The flush runs on its own once every slot of every workspace used in the scope is dirty, and otherwise
   at scope exit.
4. :meth:`EdgeWorkspaces.scope` forks the reset stream from the calling stream on entry and joins it back on exit.
   At scope exit every slot holds the sentinel again and no reset is in flight, so a scope captured into a CUDA
   graph is self-contained: replays launched on one stream are ordered by that stream, a replay never clears a slot
   an earlier replay still reads, and every replay re-arms what it used. Slots restart at 0 in every scope, so
   warm-up, capture and replay use the same slot sequence. Generations are host-side capture-time labels; replays do
   not advance them.

Leases of one workspace in a scope must share a row count: slot ``s`` of ``rows`` rows starts at ``s * rows`` rows,
so the dirty slots of a step form one contiguous span. A reset covers only the rows the lease handed out, so a
producer must write inside its lease. A ``bounded`` workspace narrows that further on the device: its leases carry a
one-word ``bound`` that a kernel of the use sets to the rows it used (e.g. from a routing plan), and the reset covers
only that many rows. Committed KV history, recurrent and convolution state are not edge workspaces and are never
reset here.

All streams and events are created before capture.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import torch


class ResetBackend(str, Enum):
    """How a workspace slot is re-armed with the sentinel."""

    # PyTorch fill of an integer view; its grid grows with the buffer.
    FILL = "fill"
    # CuTe DSL fill with 128-bit stores and a bounded grid.
    CUTE = "cute"
    # CUTE for resets of at least EdgeWorkspaces.cute_min_bytes, FILL below.
    AUTO = "auto"


# Element dtype -> (integer lane dtype, lane sentinel, 32-bit word of sentinels).
_SENTINELS: Dict[torch.dtype, Tuple[torch.dtype, int, int]] = {
    torch.bfloat16: (torch.int16, -0x8000, -0x7FFF8000),
    torch.float16: (torch.int16, -0x8000, -0x7FFF8000),
    torch.float32: (torch.int32, -0x80000000, -0x80000000),
    torch.int32: (torch.int32, 0, 0),
    torch.float8_e4m3fn: (torch.int8, -0x80, -0x7F7F7F80),
    # UE8M0 scale bytes: no -0, so the NaN encoding (never written for a finite scale).
    torch.uint8: (torch.uint8, 0xFF, -1),
}


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

    @property
    def bound(self) -> Optional[torch.Tensor]:
        """int32 [1]: rows of the lease the use wrote (a bounded workspace re-arms only those), else None."""
        bounds = self.workspace.bounds
        return None if bounds is None else bounds[self.slot : self.slot + 1]

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
    """Persistent sentinel-armed slots of one edge, carved from one allocation."""

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        slots: int,
        device: torch.device,
        max_streams: int,
        bounds: Optional[torch.Tensor] = None,
    ) -> None:
        if dtype not in _SENTINELS:
            raise ValueError(f"edge {name}: no sentinel for {dtype}")
        if slots < 1:
            raise ValueError(f"edge {name}: slots must be positive, got {slots}")
        self.name = name
        self.shape = tuple(shape)
        self.dtype = dtype
        self.backend = ResetBackend.AUTO
        numel = torch.Size(self.shape).numel()
        element_size = torch.empty((), dtype=dtype).element_size()
        # Slots start 256-byte aligned; the padding is part of the armed storage.
        stride = -(-numel * element_size // 256) * 256 // element_size
        self.storage = torch.empty(slots * stride, dtype=dtype, device=device)
        self.slots = [
            self.storage[i * stride : i * stride + numel].view(self.shape) for i in range(slots)
        ]
        self._row_numel = numel // self.shape[0]
        self.packed = self._row_numel * element_size % 16 == 0
        # int32 [slots]: rows each slot's last use wrote, set on the device; a bounded slot's reset covers only those.
        self.bounds = bounds
        self._scope_rows: Optional[int] = None
        # Released leases awaiting a flush.
        self._dirty: List[Optional[EdgeLease]] = [None] * slots
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

    def in_use(self, slot: int) -> bool:
        return self._lease[slot] is not None or self._dirty[slot] is not None

    @property
    def storage_bytes(self) -> int:
        return self.storage.numel() * self.storage.element_size()

    def _lease_tensor(self, slot: int, rows: int) -> torch.Tensor:
        if not self.packed:
            return self.slots[slot] if rows == self.shape[0] else self.slots[slot][:rows]
        numel = rows * self._row_numel
        return self.storage[slot * numel : (slot + 1) * numel].view(rows, *self.shape[1:])


class EdgeWorkspaces:
    """Owns edge workspaces, their reset stream and the scope that brackets a forward.

    Not thread-safe: callers must not use one instance from two threads at
    the same time (the Kimi K3 edges share one per device across executor
    threads that run forwards one after another).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        reset_threads: int = 256,
        reset_unroll: int = 4,
        max_streams_per_slot: int = 8,
        cute_min_bytes: int = 4 << 20,
    ) -> None:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.reset_threads = reset_threads
        self.reset_unroll = reset_unroll
        self.max_streams_per_slot = max_streams_per_slot
        self.cute_min_bytes = cute_min_bytes
        # Number of flushes that re-armed at least one slot.
        self.flushes = 0
        self._touched: Set[str] = set()
        self.reset_stream = torch.cuda.Stream(device=self.device)
        self._workspaces: Dict[str, EdgeWorkspace] = {}
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
        bounded: bool = False,
        bounds_from: Optional[str] = None,
    ) -> EdgeWorkspace:
        """Register (or return the matching) workspace and arm all its slots.

        Dim 0 is the row dimension leases can truncate; size it for the
        largest token count the edge serves. ``slots`` must cover every use of
        the edge in one scope. ``bounded`` gives each lease a ``bound`` word
        and re-arms only the rows it holds (rows of 16 bytes or a multiple).
        ``bounds_from`` names a registered bounded workspace with as many slots
        whose bound words this one shares: the two are leased together, so slot
        ``s`` of both holds the rows of one use.
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
        bounds = None
        if bounds_from is not None:
            owner = self._workspaces.get(bounds_from)
            if owner is None or owner.bounds is None or owner.num_slots != slots:
                raise ValueError(
                    f"edge {name}: {bounds_from} is not a registered bounded edge of {slots} slots"
                )
            bounds = owner.bounds
        elif bounded:
            bounds = torch.full((slots,), shape[0], dtype=torch.int32, device=self.device)
        ws = EdgeWorkspace(
            name,
            shape,
            dtype,
            slots,
            self.device,
            self.max_streams_per_slot,
            bounds,
        )
        if bounds is not None and ws._row_numel * ws.storage.element_size() % 16:
            raise ValueError(
                f"edge {name}: a bounded workspace needs rows of a multiple of 16 bytes"
            )
        self._workspaces[name] = ws
        self.preinitialize((ws,))
        if bounds is not None:
            from .fine_grained_sync_fill import compile_sentinel_fill_bounded

            compile_sentinel_fill_bounded(self.reset_threads)
        else:
            from .fine_grained_sync_fill import compile_sentinel_fill_spans

            # The first flush may run inside a graph capture; compile before it.
            compile_sentinel_fill_spans(self.reset_threads, self.reset_unroll)
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
                self._fill(ws, ws.storage)
                for slot in range(ws.num_slots):
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
        self._touched.clear()
        for ws in self._workspaces.values():
            ws._cursor = 0
            ws._scope_rows = None
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            leaked = self._drain_leases(origin)
            self.flush()
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
        if ws._dirty[slot] is not None:
            raise RuntimeError(
                f"edge {name}: slot {slot} was already used in this scope and not flushed; "
                f"the edge needs a slot per use ({ws.num_slots} registered)"
            )
        rows = ws.shape[0] if rows is None else rows
        if not 0 < rows <= ws.shape[0]:
            raise ValueError(f"edge {name}: rows {rows} outside (0, {ws.shape[0]}]")
        if ws.packed:
            if ws._scope_rows is None:
                ws._scope_rows = rows
            elif ws._scope_rows != rows:
                raise ValueError(
                    f"edge {name}: {rows} rows after {ws._scope_rows} in one scope; packed slots "
                    "need one row count per scope"
                )
        ws._cursor = (slot + 1) % ws.num_slots
        ws._gen[slot] += 1
        stream = torch.cuda.current_stream(self.device)
        if ws._pending[slot]:
            stream.wait_event(ws._ready[slot])
        tensor = ws._lease_tensor(slot, rows)
        self._touched.add(name)
        lease = EdgeLease(ws, slot, ws._gen[slot], tensor, readers, stream)
        ws._lease[slot] = lease
        return lease

    def release(
        self, lease: EdgeLease, streams: Optional[Iterable[torch.cuda.Stream]] = None
    ) -> None:
        """Release one reader of a slot; the last release marks it dirty for the next flush.

        ``streams`` (default: the current stream) are the streams this reader
        ran on; each is joined before the reset.
        """
        ws = lease.workspace
        if lease.released or ws._lease[lease.slot] is not lease:
            raise RuntimeError(f"edge {ws.name}: slot {lease.slot} gen {lease.gen} is not held")
        if not self.in_scope:
            raise RuntimeError(f"release of edge {ws.name} needs an open EdgeWorkspaces.scope()")
        for stream in streams if streams is not None else (torch.cuda.current_stream(self.device),):
            lease._join(stream)
        lease.pending_readers -= 1
        if lease.pending_readers == 0:
            self._retire(lease)
            if self._all_dirty():
                self.flush()

    def flush(self) -> None:
        """Re-arm every slot released since the last flush with one batched fill.

        Inside a scope only; the reset stream first waits for every join of
        those slots. A no-op when nothing is dirty.
        """
        if not self.in_scope:
            raise RuntimeError("flush needs an open EdgeWorkspaces.scope()")
        dirty = [
            (ws, lease)
            for ws in self._workspaces.values()
            for lease in ws._dirty
            if lease is not None
        ]
        if not dirty:
            return
        for ws, lease in dirty:
            for index in lease._joins.values():
                self.reset_stream.wait_event(ws._joins[lease.slot][index])
        with torch.cuda.stream(self.reset_stream):
            self._fill_batch(dirty)
        for ws, lease in dirty:
            ws._ready[lease.slot].record(self.reset_stream)
            ws._pending[lease.slot] = True
            ws._dirty[lease.slot] = None
        self.flushes += 1

    def ledger(self) -> List[Dict[str, object]]:
        """Device memory owned by the workspaces: one row per edge."""
        return [
            {
                "name": ws.name,
                "shape": ws.shape,
                "dtype": str(ws.dtype).replace("torch.", ""),
                "slots": ws.num_slots,
                "bytes_per_slot": ws.bytes_per_slot,
                "bytes": ws.storage_bytes,
                "packed": ws.packed,
                "bounded": ws.bounds is not None,
                "events": ws.num_slots * (1 + self.max_streams_per_slot),
            }
            for ws in self._workspaces.values()
        ]

    def _retire(self, lease: EdgeLease) -> None:
        ws = lease.workspace
        if lease._acquire_stream.cuda_stream not in lease._joins:
            lease._join(lease._acquire_stream)
        ws._lease[lease.slot] = None
        ws._dirty[lease.slot] = lease
        lease.released = True

    def _all_dirty(self) -> bool:
        stepped = [self._workspaces[name] for name in self._touched]
        return bool(stepped) and all(lease is not None for ws in stepped for lease in ws._dirty)

    def _drain_leases(self, origin: torch.cuda.Stream) -> List[Tuple[str, int, int]]:
        leaked = []
        for ws in self._workspaces.values():
            for lease in ws._lease:
                if lease is None:
                    continue
                leaked.append((ws.name, lease.slot, lease.gen))
                lease._joins.clear()
                lease._join(origin)
                self._retire(lease)
        return leaked

    def _fill_batch(self, dirty: List[Tuple[EdgeWorkspace, EdgeLease]]) -> None:
        # Adjacent dirty slots of one workspace merge into one span (packed slots of one scope are adjacent), then
        # CUTE spans share launches.
        spans: List[Tuple[EdgeWorkspace, torch.Tensor]] = []
        for ws in {id(ws): ws for ws, _ in dirty}.values():
            if ws.bounds is not None:
                self._fill_bounded(ws, [lease for other, lease in dirty if other is ws])
                continue
            base = ws.storage.data_ptr()
            element_size = ws.storage.element_size()
            ranges = sorted(
                (
                    (lease.tensor.data_ptr() - base) // element_size,
                    (lease.tensor.data_ptr() - base) // element_size + lease.tensor.numel(),
                )
                for other, lease in dirty
                if other is ws
            )
            merged = [list(ranges[0])]
            for start, end in ranges[1:]:
                if start == merged[-1][1]:
                    merged[-1][1] = end
                else:
                    merged.append([start, end])
            spans += [(ws, ws.storage[start:end]) for start, end in merged]
        auto_bytes = sum(
            t.numel() * t.element_size() for ws, t in spans if ws.backend is ResetBackend.AUTO
        )
        cute_spans = []
        for ws, tensor in spans:
            backend = ws.backend
            if backend is ResetBackend.AUTO:
                cute = auto_bytes >= self.cute_min_bytes
                backend = ResetBackend.CUTE if cute else ResetBackend.FILL
            if backend is ResetBackend.CUTE:
                cute_spans.append((tensor, _SENTINELS[ws.dtype][2]))
            else:
                self._fill(ws, tensor, backend)
        if cute_spans:
            from .fine_grained_sync_fill import sentinel_fill_spans_

            sentinel_fill_spans_(cute_spans, 1 << 30, self.reset_threads, self.reset_unroll)

    def _fill_bounded(self, ws: EdgeWorkspace, leases: List[EdgeLease]) -> None:
        # One launch per run of consecutive slots at a fixed spacing (packed slots of one scope are).
        from .fine_grained_sync_fill import sentinel_fill_bounded_

        row_bytes = ws._row_numel * ws.storage.element_size()
        word = _SENTINELS[ws.dtype][2]
        leases = sorted(leases, key=lambda lease: lease.slot)
        runs: List[List[EdgeLease]] = []
        for lease in leases:
            run = runs[-1] if runs else None
            if (
                run is not None
                and lease.slot == run[-1].slot + 1
                and lease.tensor.numel() == run[0].tensor.numel()
                and (
                    len(run) == 1
                    or lease.tensor.data_ptr() - run[-1].tensor.data_ptr()
                    == run[1].tensor.data_ptr() - run[0].tensor.data_ptr()
                )
            ):
                run.append(lease)
            else:
                runs.append([lease])
        base = ws.storage.data_ptr()
        element_size = ws.storage.element_size()
        for run in runs:
            cap_bytes = run[0].tensor.numel() * element_size
            slot_bytes = (
                run[1].tensor.data_ptr() - run[0].tensor.data_ptr() if len(run) > 1 else cap_bytes
            )
            start = (run[0].tensor.data_ptr() - base) // element_size
            end = start + ((len(run) - 1) * slot_bytes + cap_bytes) // element_size
            sentinel_fill_bounded_(
                ws.storage[start:end],
                ws.bounds[run[0].slot : run[0].slot + len(run)],
                slot_bytes,
                cap_bytes,
                row_bytes,
                word,
                1 << 30,
                self.reset_threads,
            )

    def _fill(
        self, ws: EdgeWorkspace, tensor: torch.Tensor, backend: Optional[ResetBackend] = None
    ) -> None:
        lane, value, word = _SENTINELS[ws.dtype]
        backend = backend or ws.backend
        if backend is ResetBackend.AUTO:
            nbytes = tensor.numel() * tensor.element_size()
            backend = ResetBackend.CUTE if nbytes >= self.cute_min_bytes else ResetBackend.FILL
        if backend is ResetBackend.FILL:
            tensor.view(lane).fill_(value)
        else:
            from .fine_grained_sync_fill import sentinel_fill_

            sentinel_fill_(tensor, word, 1 << 30, self.reset_threads, self.reset_unroll)
