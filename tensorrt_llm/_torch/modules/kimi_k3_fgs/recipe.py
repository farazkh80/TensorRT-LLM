# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model's fine-grained-sync recipe (``FineGrainedSyncConfig``): which edges run at each decode batch size.

:func:`install` runs once, before the model declares its edges. The recipe owns an edge family (kda.*, mla.*, moe.*)
unless one of the family's environment variables is set (``FAMILY_ENVS``), in which case the family runs exactly as
the variables say. In an owned family an edge runs only at the rows of its windows, and an edge the recipe does not
list is off. The mode, the reset policy and the kernel switches become defaults of their environment variables, so a
variable that is set wins.

Edge names: the KDA chain edges (``KDA_EDGES``); ``mla.chain``, every mla.* slot (the decode MLA layer runs as a
chain only with all of them); ``moe.<edge>`` for the MoE edges of ``moe_chain``, where ``moe.recv_norm`` also selects
the Lamport combine inbox at its rows.
"""

import os
from typing import Dict, Optional, Tuple

KDA_EDGES = ("kda.x_norm", "kda.qkvg", "kda.bfa", "kda.g", "kda.core_o", "kda.attn_o")
MLA_CHAIN = "mla.chain"
FAMILY_ENVS = {
    "kda": ("TLLM_K3_FGS_MIN_ROWS", "TLLM_K3_FGS_MAX_ROWS", "TLLM_K3_FGS_DROP_EDGES"),
    "mla": ("TLLM_K3_FGS_MLA_MIN_ROWS", "TLLM_K3_FGS_MLA_MAX_ROWS", "TLLM_K3_FGS_DROP_EDGES"),
    "moe": (
        "TLLM_K3_MOE_FGS",
        "TLLM_K3_MOE_FGS_ROWS",
        "TLLM_K3_MOE_FGS_MAX_ROWS",
        "TLLM_K3_COMM_INBOX",
    ),
}
# The fused-MoE kernel switches the MoE edges need; they are read when the MoE layers are built or run.
SWITCH_PREFIXES = ("TLLM_K3_MOE", "TLLM_K3_COMM", "TLLM_K3_SHARED")
MODE_ENV = "TLLM_K3_FGS"
RESET_ENV = "TLLM_FGS_RESET_POLICY"
INBOX_ENV = "TLLM_K3_COMM_INBOX"

_windows: Dict[str, Tuple[Tuple[int, int], ...]] = {}
_owned: frozenset = frozenset()


def moe_edges() -> Tuple[str, ...]:
    from . import moe_chain

    return tuple(f"moe.{e}" for e in moe_chain.EDGES + tuple(moe_chain.V2_EDGES))


def install(config) -> None:
    """Apply ``config`` (a ``FineGrainedSyncConfig``, or None for the environment-only behaviour)."""
    global _windows, _owned
    _windows, _owned = {}, frozenset()
    if config is None:
        return
    bad = sorted(n for n in config.kernel_switches if not n.startswith(SWITCH_PREFIXES))
    if bad:
        raise ValueError(
            f"fine_grained_sync_config: kernel_switches {bad} are not fused-MoE switches "
            f"({', '.join(p + '*' for p in SWITCH_PREFIXES)}); set them in the environment"
        )
    for name, value in config.kernel_switches.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault(MODE_ENV, "off" if config.edges == {} else config.mode)
    os.environ.setdefault(RESET_ENV, config.reset_policy)
    if config.edges is None:
        return
    known = set(KDA_EDGES) | {MLA_CHAIN} | set(moe_edges())
    unknown = sorted(set(config.edges) - known)
    if unknown:
        raise ValueError(
            f"fine_grained_sync_config: unknown edge(s) {unknown}; edges are {', '.join(sorted(known))}"
        )
    _windows = {name: config.windows(name) for name in config.edges}
    _owned = frozenset(
        family for family, names in FAMILY_ENVS.items() if not any(n in os.environ for n in names)
    )
    if "moe" in _owned and "moe.recv_norm" in _windows:
        os.environ.setdefault(INBOX_ENV, "combine")

    from . import edges, mla_chain, moe_chain

    if "kda" in _owned:
        edges.configure_window(span(KDA_EDGES))
    if "mla" in _owned:
        mla_chain.configure_window(span((MLA_CHAIN,)))
    if "moe" in _owned:
        moe_chain.configure_windows(
            {n[len("moe.") :]: span((n,)) for n in _windows if n.startswith("moe.")}
        )


def owns(family: str) -> bool:
    return family in _owned


def span(names) -> Optional[Tuple[int, int]]:
    """The smallest window holding every window of ``names``, or None when none of them runs."""
    spans = [w for name in names for w in _windows.get(name, ())]
    if not spans:
        return None
    return min(low for low, _ in spans), max(high for _, high in spans)


def on(edge: str, rows: int) -> bool:
    return any(low <= rows <= high for low, high in _windows.get(edge, ()))


def slot_allowed(slot: str, rows: int) -> bool:
    """Whether the recipe lets the kda / mla edge holding ``slot`` run at ``rows`` (True outside owned families;
    MoE slots are gated by edge name in ``moe_chain``)."""
    family = slot.partition(".")[0]
    if family == "kda" and slot in KDA_EDGES and owns("kda"):
        return on(slot, rows)
    if family == "mla" and owns("mla"):
        return on(MLA_CHAIN, rows)
    return True


def moe_on(edge: str, rows: int) -> bool:
    """Whether the recipe runs MoE edge ``edge`` at ``rows`` (always True when the recipe does not own moe.*)."""
    return not owns("moe") or on(f"moe.{edge}", rows)


def moe_edge_names() -> Optional[frozenset]:
    """The MoE edges the recipe lists (without the ``moe.`` prefix), or None when it does not own moe.*."""
    if not owns("moe"):
        return None
    return frozenset(n[len("moe.") :] for n in _windows if n.startswith("moe."))
