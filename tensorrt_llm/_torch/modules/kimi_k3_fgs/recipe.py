# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model's fine-grained-sync recipe (``FineGrainedSyncConfig``): which edges run at each decode batch size.

:func:`install` runs once, before the model declares its edges. An edge runs only at the rows of its windows, and an
edge the recipe does not list is off; a config without ``edges`` runs ``DEFAULT_EDGES``, an empty map none (FGS off on
the same kernels).
"""

from typing import Dict, Optional, Tuple

# Edge windows of the decode recipe, run when the config lists no edges.
DEFAULT_EDGES: Dict[str, Tuple[Tuple[int, int], ...]] = {}
_FAMILIES = frozenset()

_windows: Dict[str, Tuple[Tuple[int, int], ...]] = {}
_owned: frozenset = frozenset()


def mode() -> str:
    """``chain_after`` when the installed recipe runs an edge, else ``off``."""
    return "chain_after" if _windows else "off"


def install(config) -> None:
    """Apply ``config`` (a ``FineGrainedSyncConfig``, or None for FGS off)."""
    global _windows, _owned
    _windows, _owned = {}, frozenset()
    if config is None:
        return
    known = set()
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
    """Whether the recipe lets the edge holding ``slot`` run at ``rows``."""
    return True
