# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fine-grained-sync edges on the Kimi K3 decode o_proj inputs (``TLLM_K3_FGS``).

``off`` (default) keeps today's kernels. ``early`` builds the producers of the
o_proj inputs (KDA decode core, MLA output gate) in their fine-grained-sync
producer mode, which releases PDL dependents at kernel entry; the o_proj
consumer still waits for producer completion. ``lamport`` additionally hands
those producers a sentinel-armed slot of an :class:`EdgeWorkspaces` edge when the
decode batch is in ``[MIN_ROWS, MAX_ROWS]``; ``kimi_k3_cute_gemm.linear`` recognizes
the slot, runs the report_valid consumer on it and releases it, which re-arms the
slot on the workspaces' reset stream.

``kimi_k3_cute_gemm.linear`` / ``bmm`` treat any operand view of a live slot the
same way: an input slot selects the report_valid consumer build of the GEMM, an
output slot its producer build (.pzo, trigger at entry). A slot acquired with
``readers=n`` is released by the n-th consuming launch.

``chain`` keeps the ``lamport`` edges and runs the whole decode KDA layer as one
PDL chain of slot edges (``kda_chain``): pre-attention mix -> x_norm -> [f_a | b]
and qkvg GEMMs -> f_b GEMM -> KdaDecodeT1 -> o_proj -> pre-MoE mix. The MoE input
stays a plain buffer. ``chain_after`` launches [f_a | b] and f_b behind qkvg instead of ahead of it.
Both chain modes also run each decode MLA layer as one chain (``mla_chain``), with its own row window.

Modules ``declare`` their edges at construction, once per layer. The model
forward runs under ``model_scope``, which registers the declared edges on first
use (outside a CUDA graph capture) and brackets the whole forward with one
``EdgeWorkspaces.scope``; the reset stream joins once at the end. With
``TLLM_FGS_RESET_POLICY=release`` (default) an edge has two slots that alternate
across layers and each release re-arms its slot; with ``step`` an edge has one
slot per declaring layer and the step's slots are re-armed by one batched fill
after the last release. Outside a scope, or for a batch outside the row window,
``acquire`` returns None and the producer writes its usual buffer.
"""

import os
import threading
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple

import torch

from ....logger import logger

ENV = "TLLM_K3_FGS"
MODES = ("off", "early", "lamport", "chain", "chain_after")
# Below 16 rows the report_valid consumer is slower than the early-trigger path with a waiting consumer.
MIN_ROWS = int(os.environ.get("TLLM_K3_FGS_MIN_ROWS", "16"))
MAX_ROWS = int(os.environ.get("TLLM_K3_FGS_MAX_ROWS", "16"))
# Above 16 rows the o_proj consumer of a ``lamport`` edge (outside a chain) runs 32-row split-K tiles, slower than the
# table's GEMM at 64 rows.
LAMPORT_MAX_ROWS = int(os.environ.get("TLLM_K3_FGS_LAMPORT_MAX_ROWS", "16"))

# Edge name -> (row width, layers that declared it).
_declared: Dict[str, Tuple[int, Set[int]]] = {}
# Edge name -> slot rows, when a declaring module's row window ends above MAX_ROWS.
_rows: Dict[str, int] = {}
_local = threading.local()
# TLLM_K3_FGS_DEBUG=1: ``debug`` logs which path each kernel took, once per (site, rows, path).
DEBUG = os.environ.get("TLLM_K3_FGS_DEBUG", "0") == "1"
_debug_seen: Set[Tuple[str, int, str]] = set()
# Why the last acquire returned None (process-wide, so a missing scope in another thread shows too).
_refusal = [""]


def mode() -> str:
    value = os.environ.get(ENV, "off").strip().lower() or "off"
    if value not in MODES:
        raise ValueError(f"{ENV}={value!r}; valid: {', '.join(MODES)}")
    return value


def debug(site: str, rows: int, path: str) -> None:
    """Log the path ``site`` took for ``rows`` rows (once each) when TLLM_K3_FGS_DEBUG=1, in every mode."""
    if DEBUG and (site, rows, path) not in _debug_seen:
        _debug_seen.add((site, rows, path))
        logger.info(f"TLLM_K3_FGS_DEBUG {ENV}={mode()} {site} rows={rows}: {path}")


def slots() -> bool:
    """Whether producers get edge slots (``lamport``, and the ``chain`` modes, which add the decode KDA chain edges)."""
    return mode() in ("lamport", "chain", "chain_after")


def chain() -> bool:
    return mode() in ("chain", "chain_after")


def producer_early(rows: Optional[int] = None) -> bool:
    """Whether o_proj-input producers use their fine-grained-sync producer build.

    In the chain modes only for ``rows`` inside the chain window: outside it a layer runs the stock kernels (no early
    trigger).
    """
    value = mode()
    if value in ("chain", "chain_after") and rows is not None:
        return MIN_ROWS <= rows <= MAX_ROWS
    return value != "off"


def lamport_window() -> Tuple[int, int]:
    """Row window of the ``lamport`` o_proj-input edges used outside a chain."""
    return MIN_ROWS, min(MAX_ROWS, LAMPORT_MAX_ROWS)


def declare(name: str, width: int, layer_idx: int, max_rows: Optional[int] = None) -> None:
    """Declare a [rows, width] bf16 edge used once per step by ``layer_idx``.

    Registered by the next ``model_scope`` outside capture, with ``max(MAX_ROWS, max_rows)`` slot rows.
    """
    if slots():
        _declared.setdefault(name, (width, set()))[1].add(layer_idx)
        if max_rows is not None and max_rows > _rows.get(name, MAX_ROWS):
            _rows[name] = max_rows


def _state():
    if not hasattr(_local, "workspaces"):
        from ..fine_grained_sync import EdgeWorkspaces

        _local.workspaces = EdgeWorkspaces.from_env()
        _local.live = {}
        _local.readers = {}
        _local.step = {}
    return _local.workspaces, _local.live


@contextmanager
def model_scope(layers: Optional[Iterable[int]] = None) -> Iterator[None]:
    """Bracket one model forward; a no-op unless the mode has slots and an edge was declared.

    ``layers`` are the layer indices the forward runs (default: every declaring layer); an edge is registered only
    for the declaring layers among them.
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
        from ..fine_grained_sync import ResetPolicy

        per_step = workspaces.reset_policy is ResetPolicy.STEP
        for name in missing:
            width = _declared[name][0]
            num_slots = len(users[name]) if per_step else 2
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
    debug("model_scope", 0, f"reset-stream fork / join around the forward, {len(workspaces.ledger())} edges")
    try:
        with workspaces.scope():
            yield
    finally:
        _local.live.clear()
        _local.readers.clear()
        _local.step.clear()


def step_cache() -> dict:
    """Scratch dict of the open ``model_scope`` (cleared when it closes): work done once per forward."""
    _state()
    return _local.step


def acquire(name: str, rows: int, readers: int = 1, window: Optional[Tuple[int, int]] = None):
    """The next armed slot of ``name`` truncated to ``rows``, or None to keep the producer's own buffer.

    ``window`` is the (min, max) row range the edge serves (default ``MIN_ROWS``, ``MAX_ROWS``). The slot is released
    after ``readers`` consuming launches (``consumed``). Slots are handed out only while ``kimi_k3_cute_gemm`` is on,
    which consumes every GEMM input slot.
    """
    low, high = (MIN_ROWS, MAX_ROWS) if window is None else window
    if not slots():
        return _refuse(f"{ENV}={mode()} has no slots")
    if not low <= rows <= high:
        return _refuse(f"{rows} rows outside [{low}, {high}]")
    if not hasattr(_local, "workspaces"):
        return _refuse(f"no model_scope opened in thread {threading.current_thread().name}")
    from .. import kimi_k3_cute_gemm

    if not kimi_k3_cute_gemm.enabled():
        return _refuse("TLLM_K3_CUTE_GEMM off")
    if not kimi_k3_cute_gemm.fgs_dsl():
        return _refuse("CuTe DSL without the v1 mbarriers (needs 4.9)")
    workspaces, live = _state()
    if not workspaces.in_scope:
        return _refuse("outside a model_scope")
    if name not in workspaces:
        return _refuse(f"edge {name} not registered")
    lease = workspaces.acquire(name, rows=rows)
    live[lease.tensor.data_ptr()] = lease
    _local.readers[lease.tensor.data_ptr()] = readers
    return lease


def _refuse(reason: str) -> None:
    _refusal[0] = reason
    return None


def last_refusal() -> str:
    """Why the last ``acquire`` returned None."""
    return _refusal[0]


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
    return _local.workspaces.ledger() if hasattr(_local, "workspaces") else None
