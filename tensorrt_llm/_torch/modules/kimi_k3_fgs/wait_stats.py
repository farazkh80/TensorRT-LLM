# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device counters for the FGS wait sites (``TLLM_K3_FGS_WAIT_STATS=1``).

A kernel that takes a counter address adds to it when it reaches the counted site: a TMA ``report_valid`` load that
landed invalid and was re-issued, a slow path, a software poll. A real run then shows which wait sites ran and how
often. Each counter is one int64 in a per-device buffer; with the stats off every address is 0 and kernels skip the
add, so timed runs keep the stats off.
"""

import atexit
import os
from typing import Dict, Optional

import torch

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

ENV = "TLLM_K3_FGS_WAIT_STATS"
MAX_COUNTERS = 1024

_index: Dict[str, int] = {}
_buffers: Dict[int, torch.Tensor] = {}


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def addr(name: str, device: Optional[torch.device] = None) -> int:
    """Address of counter ``name`` on ``device`` (default: the current device), or 0 when the stats are off."""
    if not enabled():
        return 0
    index = _index.setdefault(name, len(_index))
    if index >= MAX_COUNTERS:
        raise RuntimeError(f"{ENV}: more than {MAX_COUNTERS} counters")
    dev = torch.cuda.current_device() if device is None else torch.device(device).index
    buf = _buffers.get(dev)
    if buf is None:
        buf = _buffers[dev] = torch.zeros(MAX_COUNTERS, dtype=torch.int64, device=f"cuda:{dev}")
    return buf.data_ptr() + 8 * index


def snapshot() -> Dict[str, int]:
    """Counter values summed over this process's devices (synchronizes)."""
    totals = {name: 0 for name in _index}
    for buf in _buffers.values():
        values = buf.cpu().tolist()
        for name, index in _index.items():
            totals[name] += values[index]
    return totals


def dump() -> None:
    if not _index:
        return
    rank = os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    for name, value in sorted(snapshot().items()):
        print(f"[wait-stats] rank {rank} {name} {value}", flush=True)


if enabled():
    atexit.register(dump)


@dsl_user_op
def add(counter: cutlass.Int64, value: cutlass.Int32, *, loc=None, ip=None) -> None:
    """``counter += value`` (relaxed, device scope); the caller skips it when ``counter`` is 0."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(counter).ir_value(loc=loc, ip=ip),
            cutlass.Int64(value).ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.add.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
