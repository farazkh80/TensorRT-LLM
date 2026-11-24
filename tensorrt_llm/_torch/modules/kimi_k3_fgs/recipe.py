# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model's fine-grained-sync recipe (``FineGrainedSyncConfig``): which edges run at each decode batch size.

:func:`install` runs once, before the model declares its edges. An edge runs only at the rows of its windows, and an
edge the recipe does not list is off; a config without ``edges`` runs ``DEFAULT_EDGES``, an empty map none (FGS off on
the same kernels).

Edge names: the KDA chain edges (``KDA_EDGES``); ``mla.chain``, every mla.* slot (the decode MLA layer runs as a
chain only with all of them) except ``mla.attn_o``, the chain's o_proj -> pre-MoE mix edge, which when listed runs only
at its own rows (unlisted: wherever the chain runs); ``moe.<edge>`` for the MoE edges of ``moe_chain``, where
``moe.recv_norm`` also selects the Lamport combine inbox at its rows.
"""

from typing import Dict, Optional, Tuple

KDA_EDGES = ("kda.x_norm", "kda.qkvg", "kda.bfa", "kda.g", "kda.core_o", "kda.attn_o")
MLA_CHAIN = "mla.chain"
MLA_ATTN_O = "mla.attn_o"
# Edge windows of the decode recipe, run when the config lists no edges.
DEFAULT_EDGES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "kda.x_norm": ((8, 64),),
    "kda.qkvg": ((8, 64),),
    "kda.bfa": ((8, 64),),
    "kda.g": ((8, 64),),
    "kda.core_o": ((8, 64),),
    "mla.chain": ((1, 64),),
    "mla.attn_o": ((1, 16),),
    "moe.recv_norm": ((1, 128),),
    "moe.fc1_fc2": ((1, 128),),
    "moe.mix_latent": ((1, 64),),
}
_FAMILIES = frozenset(("kda", "mla", "moe"))

_windows: Dict[str, Tuple[Tuple[int, int], ...]] = {}
_owned: frozenset = frozenset()


def moe_edges() -> Tuple[str, ...]:
    from . import moe_chain

    return tuple(f"moe.{e}" for e in moe_chain.EDGES + tuple(moe_chain.V2_EDGES))


def mode() -> str:
    """``chain_after`` when the installed recipe runs an edge, else ``off``."""
    return "chain_after" if _windows else "off"


def install(config) -> None:
    """Apply ``config`` (a ``FineGrainedSyncConfig``, or None for FGS off)."""
    global _windows, _owned
    _windows, _owned = {}, frozenset()
    if config is None:
        return
    known = set(KDA_EDGES) | {MLA_CHAIN, MLA_ATTN_O} | set(moe_edges())
    names = DEFAULT_EDGES if config.edges is None else config.edges
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(
            f"fine_grained_sync_config: unknown edge(s) {unknown}; edges are {', '.join(sorted(known))}"
        )
    _windows = (
        dict(DEFAULT_EDGES) if config.edges is None else {n: config.windows(n) for n in names}
    )
    _owned = _FAMILIES

    from . import edges, mla_chain, moe_chain

    edges.configure_window(span(KDA_EDGES))
    mla_chain.configure_window(span((MLA_CHAIN,)))
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
    """Whether the recipe lets the kda / mla edge holding ``slot`` run at ``rows`` (MoE slots are gated by edge name
    in ``moe_chain``)."""
    family = slot.partition(".")[0]
    if family == "kda" and slot in KDA_EDGES and owns("kda"):
        return on(slot, rows)
    if family == "mla" and owns("mla"):
        if slot == MLA_ATTN_O and MLA_ATTN_O in _windows:
            return on(MLA_CHAIN, rows) and on(MLA_ATTN_O, rows)
        return on(MLA_CHAIN, rows)
    return True


def moe_on(edge: str, rows: int) -> bool:
    """Whether the recipe runs MoE edge ``edge`` at ``rows``."""
    return not owns("moe") or on(f"moe.{edge}", rows)


def moe_edge_names() -> Optional[frozenset]:
    """The MoE edges the recipe lists (without the ``moe.`` prefix), or None when no recipe is installed."""
    if not owns("moe"):
        return None
    return frozenset(n[len("moe.") :] for n in _windows if n.startswith("moe."))
