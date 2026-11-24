# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fine-grained-sync edges of the Kimi K3 decode layers.

The mode (:func:`mode`) is ``chain_after`` when the installed ``FineGrainedSyncConfig`` lists an edge, else ``off``
(the same kernels, no edge). ``chain_after`` runs each decode KDA layer as one PDL chain of slot edges
(``kda_chain``): pre-attention mix -> x_norm -> qkvg GEMM, then [f_a | b] and f_b GEMMs -> KdaDecodeT1 -> o_proj ->
pre-MoE mix, and each decode MLA layer as one chain (``mla_chain``), each inside the recipe's row window.

``kimi_k3_cute_gemm.linear`` / ``bmm`` treat any operand view of a live slot the same way: an input slot selects the
report_valid consumer build of the GEMM, an output slot its producer build (.pzo, trigger at entry). A slot acquired
with ``readers=n`` is released by the n-th consuming launch.

Modules ``declare`` their edges at construction, once per layer. The model forward runs under ``model_scope``, which
registers the declared edges on first use (outside a CUDA graph capture) and brackets the whole forward with one
``EdgeWorkspaces.scope``; the reset stream joins once at the end. An edge has one slot per declaring layer and the
step's slots are re-armed by one batched fill after the last release. Outside a scope, or for a batch outside the
row window, ``acquire`` returns None and the producer writes its usual buffer.
"""

import threading
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple

import torch

from ....logger import logger
from . import recipe

ENV = "TLLM_K3_FGS"
# KDA chain row window; ``configure_window`` sets the recipe's kda.* span.
MIN_ROWS = 8
MAX_ROWS = 64
# Row window of the o_proj-input edges (kda.core_o, mla.gated) used outside a chain.
LAMPORT_MIN_ROWS = 16
LAMPORT_MAX_ROWS = 16

# Edge name -> (row width, layers that declared it).
_declared: Dict[str, Tuple[int, Set[int]]] = {}
# Edge name -> slot rows, when a declaring module's row window ends above MAX_ROWS.
_rows: Dict[str, int] = {}
# Edge name -> (min, max) rows its declaring modules acquire it for.
_windows: Dict[str, Tuple[int, int]] = {}
# Counter edge name -> int32 words per slot (int32 slabs armed to zero instead of bf16 slots).
_counter_words: Dict[str, int] = {}
# Byte edge name -> (slot rows, element dtype, bounded, bound-sharing edge or None); see ``declare_bytes``.
_byte_slots: Dict[str, Tuple[int, torch.dtype, bool, Optional[str]]] = {}
_local = threading.local()
# One slot set per device: the executor runs forwards from several threads (memory profiling, warm-up, the step
# loop), never concurrently, and a set per thread would keep a full copy of every edge's slots for each of them.
_device_workspaces: Dict[int, object] = {}
_device_workspaces_lock = threading.Lock()


def configure_window(window: Optional[Tuple[int, int]]) -> None:
    """Set the decode KDA chain's row window to the recipe's kda.* span (None: the chain never runs)."""
    global MIN_ROWS, MAX_ROWS
    MIN_ROWS, MAX_ROWS = window if window is not None else (1 << 30, MAX_ROWS)


def mode() -> str:
    return recipe.mode()


def slots() -> bool:
    """Whether producers get edge slots."""
    return mode() == "chain_after"


def chain() -> bool:
    return mode() == "chain_after"


def producer_early(rows: Optional[int] = None) -> bool:
    """Whether o_proj-input producers use their fine-grained-sync producer build: only for ``rows`` inside the chain
    window (outside it a layer runs the stock kernels, no early trigger)."""
    if chain() and rows is not None:
        return MIN_ROWS <= rows <= MAX_ROWS
    return chain()


def lamport_window() -> Tuple[int, int]:
    """Row window of the o_proj-input edges used outside a chain."""
    return max(MIN_ROWS, LAMPORT_MIN_ROWS), min(MAX_ROWS, LAMPORT_MAX_ROWS)


def declare(
    name: str,
    width: int,
    layer_idx: int,
    max_rows: Optional[int] = None,
    min_rows: Optional[int] = None,
) -> None:
    """Declare a [rows, width] bf16 edge used once per step by ``layer_idx`` for ``min_rows`` to ``max_rows`` rows.

    Registered by the next ``model_scope`` outside capture, with ``max(MAX_ROWS, max_rows)`` slot rows.
    """
    if slots():
        _declared.setdefault(name, (width, set()))[1].add(layer_idx)
        if max_rows is not None and max_rows > _rows.get(name, MAX_ROWS):
            _rows[name] = max_rows
        low, high = _windows.get(name, (MIN_ROWS, MAX_ROWS))
        _windows[name] = (
            min(low, MIN_ROWS if min_rows is None else min_rows),
            max(high, MAX_ROWS if max_rows is None else max_rows),
        )


def declare_counters(
    name: str,
    words: int,
    layer_idx: int,
    max_rows: Optional[int] = None,
    min_rows: Optional[int] = None,
) -> None:
    """Declare an int32 counter slab of ``words`` words used once per step by ``layer_idx``.

    Slots are armed to zero and re-armed like the bf16 edges. Producers add to the words
    after their stores, consumers wait for an expected count; the slab is not a GEMM operand. ``words`` is the most
    any forward uses: ``acquire_counters`` truncates to the words of the current forward.
    """
    if slots():
        _counter_words[name] = max(words, _counter_words.get(name, 0))
        declare(name, 1, layer_idx, max_rows=max_rows, min_rows=min_rows)


def declare_bytes(
    name: str,
    width: int,
    dtype: torch.dtype,
    slot_rows: int,
    layer_idx: int,
    max_rows: int,
    min_rows: int = 1,
    policy: Optional[str] = None,
    bounded: bool = False,
    bounds_from: Optional[str] = None,
) -> None:
    """Declare a [slot_rows, width] edge of 1-byte ``dtype`` (float8_e4m3fn data armed with -0, uint8 UE8M0 scales
    armed with 0xFF) used once per step by ``layer_idx`` for ``min_rows`` to ``max_rows`` tokens per rank.

    ``slot_rows`` counts rows of the edge (e.g. permuted rows), not tokens. ``bounded`` re-arms only the rows a kernel
    of the use writes into ``lease.bound``; ``bounds_from`` (an edge declared before this one, acquired together with
    it) shares that edge's bound words. Not a GEMM operand: ``acquire_bytes`` hands the slots out."""
    if slots():
        _byte_slots[name] = (
            max(slot_rows, _byte_slots.get(name, (0,))[0]),
            dtype,
            bounded,
            bounds_from,
        )
        declare(name, width, layer_idx, max_rows=max_rows, min_rows=min_rows)


def _workspaces():
    device = torch.cuda.current_device()
    workspaces = _device_workspaces.get(device)
    if workspaces is None:
        with _device_workspaces_lock:
            workspaces = _device_workspaces.get(device)
            if workspaces is None:
                from ..fine_grained_sync import EdgeWorkspaces

                workspaces = _device_workspaces[device] = EdgeWorkspaces()
    return workspaces


def _state():
    if not hasattr(_local, "live"):
        _local.live = {}
        _local.readers = {}
        _local.step = {}
    return _workspaces(), _local.live


@contextmanager
def model_scope(
    layers: Optional[Iterable[int]] = None, rows: Optional[int] = None
) -> Iterator[None]:
    """Bracket one model forward; a no-op unless the mode has slots and an edge was declared.

    ``layers`` are the layer indices the forward runs (default: every declaring layer); an edge is registered only
    for the declaring layers among them. A forward of ``rows`` rows that no edge window covers runs without the
    reset-stream fork / join.
    """
    if not slots() or not _declared:
        yield
        return
    workspaces, _ = _state()
    if workspaces.in_scope:
        yield
        return
    run = None if layers is None else set(layers)
    users = {
        name: declared if run is None else declared & run
        for name, (_, declared) in _declared.items()
    }
    missing = [name for name in _declared if users[name] and name not in workspaces]
    if missing and not torch.cuda.is_current_stream_capturing():
        for name in missing:
            width = _declared[name][0]
            slot_rows, dtype, bounded, bounds_from = _byte_slots.get(name, (0, None, False, None))
            num_slots = len(users[name])
            if name in _counter_words:
                workspaces.workspace(name, (_counter_words[name], 1), torch.int32, slots=num_slots)
                continue
            if name in _byte_slots:
                workspaces.workspace(
                    name,
                    (slot_rows, width),
                    dtype,
                    slots=num_slots,
                    bounded=bounded,
                    bounds_from=bounds_from,
                )
                continue
            workspaces.workspace(
                name, (_rows.get(name, MAX_ROWS), width), torch.bfloat16, slots=num_slots
            )
        logger.info(f"{ENV}={mode()}: edge slots {workspaces.ledger()}")
    elif missing:
        logger.warning_once(
            f"{ENV}=lamport: edges {missing} are first used inside a CUDA graph capture and"
            " cannot be registered there; their producers keep their own buffers (run one"
            " eager forward first)",
            key="k3_fgs_unregistered",
        )
    if rows is not None and not any(
        low <= rows <= high for name, (low, high) in _windows.items() if users[name]
    ):
        yield
        return
    try:
        with workspaces.scope():
            yield
    finally:
        _local.live.clear()
        _local.readers.clear()
        _local.step.clear()


def reset_stream() -> torch.cuda.Stream:
    """The reset stream of this device's edge workspaces."""
    return _state()[0].reset_stream


def step_cache() -> dict:
    """Scratch dict of the open ``model_scope`` (cleared when it closes): work done once per forward."""
    _state()
    return _local.step


def acquire(name: str, rows: int, readers: int = 1, window: Optional[Tuple[int, int]] = None):
    """The next armed slot of ``name`` truncated to ``rows``, or None to keep the producer's own buffer.

    ``window`` is the (min, max) row range the edge serves (default ``MIN_ROWS``, ``MAX_ROWS``). The slot is released
    after ``readers`` consuming launches (``consumed``).
    """
    low, high = (MIN_ROWS, MAX_ROWS) if window is None else window
    if not slots():
        return None
    if not low <= rows <= high:
        return None
    if not recipe.slot_allowed(name, rows):
        return None
    if not hasattr(_local, "live"):
        return None
    from .. import kimi_k3_cute_gemm

    if not kimi_k3_cute_gemm.fgs_dsl():
        return None
    workspaces, live = _state()
    if not workspaces.in_scope:
        return None
    if name not in workspaces:
        return None
    lease = workspaces.acquire(name, rows=rows)
    live[lease.tensor.data_ptr()] = lease
    _local.readers[lease.tensor.data_ptr()] = readers
    return lease


def acquire_counters(name: str, rows: int, words: int, readers: int = 1, window=None):
    """The next zero-armed counter slot of ``name`` truncated to ``words`` int32 words (``lease.tensor`` is
    [words, 1]), or None (same refusals as ``acquire`` except the GEMM ones)."""
    low, high = (MIN_ROWS, MAX_ROWS) if window is None else window
    if not slots():
        return None
    if not low <= rows <= high:
        return None
    if not recipe.slot_allowed(name, rows):
        return None
    if not hasattr(_local, "live"):
        return None
    workspaces, live = _state()
    if not workspaces.in_scope:
        return None
    if name not in workspaces or name not in _counter_words:
        return None
    lease = workspaces.acquire(name, rows=words)
    live[lease.tensor.data_ptr()] = lease
    _local.readers[lease.tensor.data_ptr()] = readers
    return lease


def acquire_bytes(name: str, rows: int, slot_rows: int, readers: int = 1, window=None):
    """The next armed slot of byte edge ``name`` truncated to ``slot_rows`` rows, for a forward of ``rows`` tokens per
    rank, or None (same refusals as ``acquire_counters``)."""
    low, high = (MIN_ROWS, MAX_ROWS) if window is None else window
    if not slots():
        return None
    if not low <= rows <= high:
        return None
    if not recipe.slot_allowed(name, rows):
        return None
    if not hasattr(_local, "live"):
        return None
    workspaces, live = _state()
    if not workspaces.in_scope:
        return None
    if name not in workspaces or name not in _byte_slots:
        return None
    if slot_rows > workspaces[name].shape[0]:
        return None
    lease = workspaces.acquire(name, rows=slot_rows)
    live[lease.tensor.data_ptr()] = lease
    _local.readers[lease.tensor.data_ptr()] = readers
    return lease


def lease_containing(t: torch.Tensor):
    """The live lease whose slot holds ``t``'s first element (``t`` may be any view of it), else None."""
    live = getattr(_local, "live", None)
    if not live:
        return None
    ptr = t.data_ptr()
    for base, lease in live.items():
        if base <= ptr < base + lease.tensor.numel() * lease.tensor.element_size():
            return lease
    return None


def release(lease) -> None:
    """Release the slot after its last reader was launched on the current stream."""
    workspaces, live = _state()
    live.pop(lease.tensor.data_ptr(), None)
    _local.readers.pop(lease.tensor.data_ptr(), None)
    workspaces.release(lease)


def consumed(lease) -> None:
    """One reader of ``lease`` was launched on the current stream; the last one releases the slot."""
    ptr = lease.tensor.data_ptr()
    left = _local.readers.get(ptr, 1) - 1
    if left > 0:
        _local.readers[ptr] = left
    else:
        release(lease)


def ledger() -> Optional[list]:
    workspaces = (
        _device_workspaces.get(torch.cuda.current_device()) if torch.cuda.is_available() else None
    )
    return workspaces.ledger() if workspaces is not None else None
