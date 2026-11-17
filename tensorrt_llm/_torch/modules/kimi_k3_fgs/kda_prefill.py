# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill as one PDL chain of fine-grained-sync edges with chain-owned slots and producer re-arm
(``TLLM_K3_FGS_PREFILL=1``), or the same kernels chained by standard PDL (``TLLM_K3_FGS_PREFILL=pdl``, the FGS-off
twin). Decode is untouched: the mixer calls :func:`forward` only for a prefill-only batch.

``TLLM_K3_FGS_PREFILL_SYNC=counter`` (default) links pk2 and pk8 with per-row-block counters instead of Lamport
slots: int32 words, one per 128-row block of the producer's output, in ``[pairs, blocks]`` slabs. The qkvg GEMM's
epilogue publishes each stored q / k / v tile (``cp.async.bulk.wait_group 0``, ``fence.proxy.async.global``,
``red.release.gpu.add``); a conv warp acquires the words of the source blocks it reads (halo included) before its
first load. The GEMM's gate tiles (N index >= 144) count into a second word row that each gated-norm CTA acquires
before its first ``z`` load: the norm is not the GEMM's PDL dependent, so nothing else implies the gate columns are
complete. Each gated-norm CTA publishes after a CTA barrier; o_proj's TMA warp acquires the word of its M tile,
then ``fence.proxy.async.global``, then issues the tile's TMA loads. Coverage is every block: publishing costs one
thread per tile / CTA, so no last-wave rule is needed. Re-arm: the next layer's qkvg GEMM arm warp zeroes the
previous pair's slabs (kilobytes). The data buffers are plain (no sentinels, no slot memory), so the window can
cover every token count. ``=lamport`` keeps the sentinel slots below.

  pre-attention mix -> qkvg GEMM --pk2 p_qkvg--> KdaConv | bfa, f_b GEMMs | BT16 --pk7 p_o--> gated norm
  --pk8 p_core--> o_proj --pk9 p_attn_o--> pre-MoE mix (consumer only, through ``edges.register_external``)

Slots are bf16 ``[pairs, MAX_ROWS, width]`` buffers per edge, allocated on the first eager prefill and armed with
-0.0 once; the edge workspaces never touch them. KDA layers take pairs in registration order (ordinal mod 2; an odd
layer count gives the last layer a third pair, so the next forward's first layer never writes the pair it arms).
Re-arm is producer side, inside the next layer's o_proj GEMM: its arm warp stores -0.0 over every slot of layer L
(p_qkvg, p_o, p_core, p_attn_o). o_proj is the chain's one GEMM with DRAM headroom (K = 12288, 1-2 TB/s); the qkvg
GEMM streams its 704 MB weight per N pass and pays the arm traffic in full (A-02), the norm has no idle warp. Every
reader of layer L's slots has completed before the MoE's plain launches of layer L, which complete before layer L+1
starts, and the arm warp runs after its grid wait; the next writer of layer L's pair is layer L+2's qkvg GEMM, after
o_proj(L+1) in stream order. A layer that
refuses the chain while a pair is dirty fills it from the host (:func:`rearm_skipped`), so stale data is never read
as valid.

``TLLM_K3_FGS_PREFILL_EDGES`` (default ``pk2,pk8``; ``pk9`` loses its mix-consumer cost to a tail that is
not there with raster M (A-03); ``pk7`` needs BT16's decomposition route, i.e. one request
at 96 heads, and is diagnostic), ``_MIN_ROWS`` / ``_MAX_ROWS`` (1024 / 262144 tokens per rank with the counter link, 4096 with Lamport slots,
which are sized to ``_MAX_ROWS``; on one GPU the counter link gains 0.6-1.7 % of the layer up to 5460 tokens and is
noise-level above, the Lamport slots gain 1.5 % at 1365 and lose 0.5-0.9 % at 8192 to their 0.8-1 GB re-arm), ``_RASTER`` (producer GEMM tile
order; ``n`` publishes row blocks in order but costs the
weight-streaming qkvg GEMM 3-21 %, A-02), ``_NORM_CTAS`` (gated norm persistent grid of the FGS roles; 0 = the
default grid, one CTA per row block), ``_DELAY_NS`` (producer GEMMs spin at entry; hazard tests),
``_MAX_SPINS`` (consumer rounds before the grid-wait fallback; small values force the slow path),
``_QKVG_CTAS`` (cap the qkvg GEMM's persistent grid; with a few CTAs its gate tiles outlive the conv → norm chain,
the hazard test of the norm's acquire), ``_NORM_GATE=0`` (drop the norm's acquire: that test's negative control),
``_CONV=bulk|cpasync`` (the conv's source loads: one ``cp.async.bulk`` per ring slot onto an mbarrier, after the
counter acquire and a proxy fence, or the per-lane ``cp.async`` ring; the twin uses the same build).
"""

import atexit
import collections
import functools
import os
from typing import Dict, List, Optional, Tuple

import torch

from ....logger import logger
from .. import kimi_k3_cute_gemm
from . import edges

ENV = "TLLM_K3_FGS_PREFILL"
SYNC = os.environ.get("TLLM_K3_FGS_PREFILL_SYNC", "counter")
if SYNC not in ("counter", "lamport"):
    raise ValueError(f"TLLM_K3_FGS_PREFILL_SYNC={SYNC!r}; valid: counter, lamport")
MIN_ROWS = int(os.environ.get("TLLM_K3_FGS_PREFILL_MIN_ROWS", "1024"))
# Lamport slots cost 123 KB per token of window; the counter link has no slot memory.
MAX_ROWS = int(
    os.environ.get("TLLM_K3_FGS_PREFILL_MAX_ROWS", "262144" if SYNC == "counter" else "4096")
)
RASTER = os.environ.get("TLLM_K3_FGS_PREFILL_RASTER", "m")
NORM_CTAS = int(os.environ.get("TLLM_K3_FGS_PREFILL_NORM_CTAS", "0"))
DELAY_NS = int(os.environ.get("TLLM_K3_FGS_PREFILL_DELAY_NS", "0"))
MAX_SPINS = int(os.environ.get("TLLM_K3_FGS_PREFILL_MAX_SPINS", str(1 << 16)))
QKVG_CTAS = int(os.environ.get("TLLM_K3_FGS_PREFILL_QKVG_CTAS", "0")) or None
NORM_GATE = os.environ.get("TLLM_K3_FGS_PREFILL_NORM_GATE", "1") != "0"
CONV = os.environ.get("TLLM_K3_FGS_PREFILL_CONV", "bulk")
if CONV not in ("bulk", "cpasync"):
    raise ValueError(f"TLLM_K3_FGS_PREFILL_CONV={CONV!r}; valid: bulk, cpasync")
# Counter slab words per pair: a word per 64 output rows covers every rubin tactic's CTA tile.
COUNTER_WORDS = -(-(-(-MAX_ROWS // 64)) // 4) * 4

P_QKVG, P_O, P_CORE, P_ATTN_O = "kda.p_qkvg", "kda.p_o", "kda.p_core", "kda.p_attn_o"
_EDGE_IDS = {"pk2": P_QKVG, "pk7": P_O, "pk8": P_CORE, "pk9": P_ATTN_O}
NEG0 = -0x8000  # bf16 -0.0 as int16


def mode() -> str:
    value = os.environ.get(ENV, "0").strip().lower() or "0"
    if value not in ("0", "1", "pdl"):
        raise ValueError(f"{ENV}={value!r}; valid: 0, 1, pdl")
    return value


def active() -> bool:
    return mode() != "0"


def fgs() -> bool:
    return mode() == "1"


@functools.lru_cache(maxsize=None)
def edges_on() -> frozenset:
    value = os.environ.get("TLLM_K3_FGS_PREFILL_EDGES", "pk2,pk8")
    ids = frozenset(v.strip().lower() for v in value.split(",") if v.strip())
    # pm* ids belong to the MLA prefill chain (``mla_prefill``), which shares the variable.
    unknown = ids - set(_EDGE_IDS) - frozenset(i for i in ids if i.startswith("pm") or i == "gproj")
    if unknown:
        raise ValueError(
            f"TLLM_K3_FGS_PREFILL_EDGES: unknown {sorted(unknown)}; valid: {', '.join(_EDGE_IDS)}"
        )
    return frozenset(_EDGE_IDS[i] for i in ids if i in _EDGE_IDS)


def on(edge: str) -> bool:
    return fgs() and edge in edges_on()


# KDA layer indices in construction (= forward) order.
_layers: List[int] = []


def register(layer_idx: int) -> None:
    if layer_idx not in _layers:
        _layers.append(layer_idx)


def _pair(ordinal: int) -> int:
    count = len(_layers)
    return 2 if count % 2 and ordinal == count - 1 else ordinal % 2


class _Slots:
    """One device's slot pairs: ``buf[edge][pair, :rows, :]``; ``dirty[edge][pair]`` = rows written by the last
    use, 0 once re-armed."""

    def __init__(self, device: torch.device, widths: Dict[str, int], pairs: int):
        self.buf: Dict[str, torch.Tensor] = {}
        self.dirty: Dict[str, List[int]] = {}
        for edge, width in widths.items():
            if edge in edges_on():
                buf = torch.empty((pairs, MAX_ROWS, width), dtype=torch.bfloat16, device=device)
                buf.view(torch.int16).fill_(NEG0)
                self.buf[edge] = buf
                self.dirty[edge] = [0] * pairs
        total = sum(b.numel() * 2 for b in self.buf.values())
        shown = {e: w for e, w in widths.items() if e in self.buf}
        logger.info(
            f"{ENV}=1: KDA prefill slots {pairs} pairs x {MAX_ROWS} rows, {shown} -> {total / 2**30:.2f} GiB"
        )

    def slot(self, edge: str, pair: int, rows: int) -> Optional[torch.Tensor]:
        buf = self.buf.get(edge)
        return None if buf is None else buf[pair, :rows]

    def arm_region(self, edge: str, pair: int) -> Optional[torch.Tensor]:
        """The dirty rows of ``edge``'s ``pair`` as a flat region for a GEMM arm warp (marks them clean)."""
        rows = self.dirty.get(edge, [0] * 3)[pair] if edge in self.buf else 0
        if rows == 0:
            return None
        self.dirty[edge][pair] = 0
        return self.buf[edge][pair, :rows]

    def fill_dirty(self) -> None:
        for edge, buf in self.buf.items():
            for pair, rows in enumerate(self.dirty[edge]):
                if rows:
                    buf[pair, :rows].view(torch.int16).fill_(NEG0)
                    self.dirty[edge][pair] = 0


_slots: Dict[torch.device, _Slots] = {}


def _slots_of(device: torch.device, widths: Dict[str, int]) -> _Slots:
    slots = _slots.get(device)
    if slots is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{ENV}=1: slots are allocated by an eager prefill, before any capture"
            )
        slots = _slots[device] = _Slots(device, widths, 2 + len(_layers) % 2)
    return slots


def rearm_skipped(device: torch.device) -> None:
    """A layer did not run the chain: fill every dirty slot from the host (the next use's GEMM would have)."""
    slots = _slots.get(device)
    if slots is not None:
        slots.fill_dirty()
    counters = _counters.get(device)
    if counters is not None:
        counters.fill_dirty()


class _Counters:
    """One device's counter slabs ``slab[edge][pair, :words]`` (int32, zero = not published); ``dirty[edge][pair]`` =
    words the last use touched, 0 once re-armed."""

    def __init__(self, device: torch.device, pairs: int):
        self.slab: Dict[str, torch.Tensor] = {}
        self.dirty: Dict[str, List[int]] = {}
        for edge in (P_QKVG, P_CORE):
            if edge in edges_on():
                # p_qkvg: row 0 counts the q / k / v tiles (the conv's gate), row 1 the gate tiles (the norm's).
                shape = (pairs, 2, COUNTER_WORDS) if edge == P_QKVG else (pairs, COUNTER_WORDS)
                self.slab[edge] = torch.zeros(shape, dtype=torch.int32, device=device)
                self.dirty[edge] = [0] * pairs
        logger.info(
            f"{ENV}=1 counter link: {sorted(self.slab)} x {pairs} pairs x {COUNTER_WORDS} words"
        )

    def words(self, edge: str, pair: int, blocks: int) -> Optional[torch.Tensor]:
        slab = self.slab.get(edge)
        if slab is None:
            return None
        if blocks > COUNTER_WORDS:
            raise RuntimeError(
                f"{edge}: {blocks} row blocks above the {COUNTER_WORDS}-word counter slab"
            )
        self.dirty[edge][pair] = max(self.dirty[edge][pair], -(-blocks // 4) * 4)
        return slab[pair]

    def arm_regions(self, pair: int) -> Tuple:
        regions = []
        for edge, slab in self.slab.items():
            words = self.dirty[edge][pair]
            if words:
                self.dirty[edge][pair] = 0
                if slab.dim() == 3:
                    regions.append((slab[pair, 0, :words], 0))
                    regions.append((slab[pair, 1, :words], 0))
                else:
                    regions.append((slab[pair, :words], 0))
        return tuple(regions)

    def fill_dirty(self) -> None:
        for edge, slab in self.slab.items():
            for pair, words in enumerate(self.dirty[edge]):
                if words:
                    slab[pair][..., :words].zero_()
                    self.dirty[edge][pair] = 0


_counters: Dict[torch.device, _Counters] = {}
# Wait statistics of the counter-link consumers: conv (gate rounds, grid-wait fallbacks), norm (acquire rounds,
# CTAs that waited), o_proj (retry slots: [1] gate rounds, [2] fallbacks); zeroed after each report.
_stats: Dict[torch.device, Dict[str, torch.Tensor]] = {}


def _stats_of(device: torch.device) -> Dict[str, torch.Tensor]:
    stats = _stats.get(device)
    if stats is None:
        from .dense_gemm_fgs import RETRY_COUNTER_SLOTS

        stats = _stats[device] = {
            "conv": torch.zeros(2, dtype=torch.int32, device=device),
            "norm": torch.zeros(2, dtype=torch.int32, device=device),
            "o_proj": torch.zeros(RETRY_COUNTER_SLOTS, dtype=torch.int32, device=device),
        }
    return stats


def stats_report(reset: bool = True) -> Dict[str, list]:
    """The consumers' wait counters since the last report (synchronizes the device)."""
    report = {}
    for device, stats in _stats.items():
        for name, t in stats.items():
            report[f"{name}@{device.index}"] = t.tolist()
            if reset:
                t.zero_()
    return report


def _counters_of(device: torch.device) -> _Counters:
    counters = _counters.get(device)
    if counters is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{ENV}=1: counter slabs are allocated by an eager prefill, before any capture"
            )
        counters = _counters[device] = _Counters(device, 2 + len(_layers) % 2)
    return counters


outcomes: collections.Counter = collections.Counter()
LOG_EVERY = 20
# The wait report synchronizes the device (``tolist``), so it is rarer than the outcome log.
STATS_EVERY = 200


def _note(layer_idx: int, refusal: Optional[str]) -> None:
    if not outcomes:
        atexit.register(
            lambda: logger.info(f"{ENV}={mode()} KDA prefill layer-forwards: {dict(outcomes)}")
        )
    outcome = mode() if refusal is None else f"off: {refusal}"
    if outcome not in outcomes:
        logger.info(f"{ENV}={mode()} KDA prefill chain, first {outcome!r} at KDA layer {layer_idx}")
    outcomes[outcome] += 1
    if sum(outcomes.values()) % LOG_EVERY == 0:
        logger.info(f"{ENV}={mode()} KDA prefill layer-forwards: {dict(outcomes)}")
        if edges.DEBUG and _stats and sum(outcomes.values()) % STATS_EVERY == 0:
            logger.info(
                f"{ENV}={mode()} KDA prefill waits (conv rounds/fallbacks, norm rounds/ctas, "
                f"o_proj -/rounds/fallbacks): {stats_report()}"
            )


def _kernels():
    try:
        from ...attention.kernels.blackwell.kimi_k3_kda import (
            fused_prefill_conv,
            gated_rmsnorm,
            kda_chunked_bt16,
        )
    except ImportError:  # release-image layout
        from ..kimi_kda.cute_kda import fused_prefill_conv, gated_rmsnorm, kda_chunked_bt16
    return fused_prefill_conv, gated_rmsnorm, kda_chunked_bt16


@functools.lru_cache(maxsize=16)
def _compile_bt16(device: torch.device, lower_bound: float, mode_: Optional[str], producer: bool):
    import cutlass

    bt16 = _kernels()[2]
    with torch.cuda.device(device):
        return bt16.compile(
            dtype=cutlass.BFloat16,
            state_dtype=cutlass.Float32,
            gate_dtype=cutlass.BFloat16,
            safe_gate=True,
            gate_lower_bound=lower_bound,
            has_state_in=True,
            has_state_out=True,
            has_state_indices=True,
            mode=mode_,
            fgs_producer=producer,
        )


def _refusal(mixer, rows: int, num_prefills: int, ssm_pool, layer_cache) -> Optional[str]:
    from ..fine_grained_sync import active_stream_manager

    sms = torch.cuda.get_device_properties(ssm_pool.device).multi_processor_count
    for refused, reason in (
        (not kimi_k3_cute_gemm.enabled(), "TLLM_K3_CUTE_GEMM off"),
        (not MIN_ROWS <= rows <= MAX_ROWS, f"{rows} rows outside [{MIN_ROWS}, {MAX_ROWS}]"),
        (active_stream_manager() is not None, "stream manager active"),
        (
            mixer._qkvg_proj_weight is None or mixer._bfa_proj_weight is None,
            "no merged projection weights",
        ),
        (not mixer.use_full_rank_gate, "low-rank output gate"),
        (mixer._o_allreduce is not None, "o_proj TP all-reduce"),
        (
            getattr(mixer.o_proj, "weight", None) is None
            or mixer.o_proj.weight.dtype != torch.bfloat16
            or mixer.f_b_proj.weight.dtype != torch.bfloat16,
            "quantized o_proj / f_b_proj",
        ),
        (mixer._has_kda_replay_caches(layer_cache), "replay caches"),
        (
            not mixer._fuse_prefill_conv or mixer._dispatch.prefill_kernel_path != "optimized",
            "stock prefill path",
        ),
        (
            mixer._packed_conv_weight is None or mixer._A_log_f32 is None,
            "prefill constants not built",
        ),
        (ssm_pool.dtype != torch.float32, "BF16 state (Small-BH route)"),
        (mixer.layer_idx not in _layers, "layer not registered"),
        (
            on(P_O) and 2 * num_prefills * mixer.num_heads > sms,
            "pk7 needs BT16's decomposition route",
        ),
    ):
        if refused:
            return reason
    return None


def forward(
    mixer,
    x2d: torch.Tensor,
    cu_seqlens: torch.Tensor,
    mamba_metadata,
    num_prefills: int,
    conv_pool: torch.Tensor,
    ssm_pool: torch.Tensor,
    layer_cache,
) -> Optional[torch.Tensor]:
    """The prefill core ``[T, H, 128]`` of one KDA layer through the chain (or its PDL twin), or None when the chain
    does not apply (nothing launched; the caller runs its own path). :func:`o_proj` must follow."""
    rows = x2d.shape[0]
    refusal = _refusal(mixer, rows, num_prefills, ssm_pool, layer_cache)
    _note(mixer.layer_idx, refusal)
    if refusal is not None:
        if fgs():
            rearm_skipped(x2d.device)
        return None
    if fgs() and SYNC == "counter":
        return _forward_counter(
            mixer, x2d, cu_seqlens, mamba_metadata, num_prefills, conv_pool, ssm_pool, layer_cache
        )
    conv_mod, norm_mod, _ = _kernels()
    d, H, hd = mixer.proj_size, mixer.num_heads, mixer.head_dim
    T = rows
    device = x2d.device
    slot_indices = mamba_metadata.state_indices[:num_prefills]
    has_init = mamba_metadata.has_initial_states[:num_prefills]
    starts = mamba_metadata.query_start_loc[: num_prefills + 1]
    live = fgs()
    slots = None
    pair = prev = 0
    if live:
        widths = {P_QKVG: 4 * d, P_O: d, P_CORE: d, P_ATTN_O: mixer.hidden_size}
        slots = _slots_of(device, widths)
        ordinal = _layers.index(mixer.layer_idx)
        pair, prev = _pair(ordinal), _pair((ordinal - 1) % len(_layers))

    from ..kimi_kda.kimi_kda_mixer import _reset_state_rows

    # Ahead of the GEMM, so the conv is the GEMM's direct PDL dependent.
    _reset_state_rows(ssm_pool, slot_indices, has_init)
    qkvg_slot = slots.slot(P_QKVG, pair, T) if live else None
    qkvg = kimi_k3_cute_gemm.linear_fgs(
        x2d,
        mixer._qkvg_proj_weight,
        qkvg_slot if qkvg_slot is not None else x2d.new_empty(T, 4 * d),
        produce=qkvg_slot is not None,
        raster_along=RASTER if qkvg_slot is not None else "m",
        start_delay_ns=DELAY_NS if qkvg_slot is not None else 0,
    )
    if qkvg_slot is not None:
        slots.dirty[P_QKVG][pair] = T
    q, k, v = (x2d.new_empty(1, T, H, hd) for _ in range(3))
    conv_mod.fused_prefill_conv(
        qkvg,
        mixer._packed_conv_weight,
        conv_pool,
        starts,
        slot_indices,
        has_init,
        getattr(mamba_metadata, "kda_max_seqlen", T),
        q,
        k,
        v,
        fgs_consumer=qkvg_slot is not None,
        fgs_max_spins=MAX_SPINS,
        pdl=qkvg_slot is None,
    )
    bfa = kimi_k3_cute_gemm.linear(x2d, mixer._bfa_proj_weight)
    g = kimi_k3_cute_gemm.linear(bfa[:, :hd], mixer.f_b_proj.weight)
    beta = bfa[:, hd : hd + H].unsqueeze(0)

    o_slot = slots.slot(P_O, pair, T) if live else None
    o = _bt16(
        mixer, q, k, v, g.view(1, T, H, hd), beta, ssm_pool, slot_indices, cu_seqlens, mamba_metadata,
        None if o_slot is None else o_slot.view(1, T, H, hd),
    )  # fmt: skip
    if o_slot is not None:
        slots.dirty[P_O][pair] = T
    mixer._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)

    core_slot = slots.slot(P_CORE, pair, T) if live else None
    core = norm_mod.kda_gated_rmsnorm(
        o.view(T * H, hd),
        qkvg[:, 3 * d :].view(T, H, hd),
        mixer.o_norm.weight,
        mixer.o_norm.eps,
        out=None if core_slot is None else core_slot.view(T * H, hd),
        use_pdl=True,
        fgs_consumer=o_slot is not None,
        fgs_max_spins=MAX_SPINS,
        fgs_ctas=NORM_CTAS if (o_slot is not None or core_slot is not None) else 0,
        fgs_producer=core_slot is not None,
    )
    if core_slot is not None:
        slots.dirty[P_CORE][pair] = T
    mixer._k3_prefill_pending = (pair, prev, T, core_slot is not None, None)
    edges.debug(
        "kda.prefill_chain",
        T,
        f"{mode()} pair {pair} edges {sorted(e for e in edges_on() if live)} norm_ctas {NORM_CTAS or 'default'}",
    )
    return core.view(T, H, hd)


def _forward_counter(
    mixer, x2d, cu_seqlens, mamba_metadata, num_prefills, conv_pool, ssm_pool, layer_cache
):
    conv_mod, norm_mod, _ = _kernels()
    d, H, hd = mixer.proj_size, mixer.num_heads, mixer.head_dim
    T = x2d.shape[0]
    device = x2d.device
    slot_indices = mamba_metadata.state_indices[:num_prefills]
    has_init = mamba_metadata.has_initial_states[:num_prefills]
    starts = mamba_metadata.query_start_loc[: num_prefills + 1]
    counters = _counters_of(device)
    stats = _stats_of(device)
    ordinal = _layers.index(mixer.layer_idx)
    pair, prev = _pair(ordinal), _pair((ordinal - 1) % len(_layers))
    cta_m, cta_n = kimi_k3_cute_gemm.rubin_tile(4 * d, mixer.hidden_size, T)
    blocks = -(-T // cta_m)
    qkv_tiles = 3 * d // cta_n
    gate_tiles = 4 * d // cta_n - qkv_tiles
    pk2 = counters.words(P_QKVG, pair, blocks) if P_QKVG in edges_on() else None

    from ..kimi_kda.kimi_kda_mixer import _reset_state_rows

    _reset_state_rows(ssm_pool, slot_indices, has_init)
    qkvg = kimi_k3_cute_gemm.linear_fgs(
        x2d,
        mixer._qkvg_proj_weight,
        x2d.new_empty(T, 4 * d),
        early_trigger=pk2 is not None,
        publish=None if pk2 is None else (pk2, qkv_tiles),
        arm=counters.arm_regions(prev),
        start_delay_ns=DELAY_NS if pk2 is not None else 0,
        max_ctas=QKVG_CTAS,
    )
    q, k, v = (x2d.new_empty(1, T, H, hd) for _ in range(3))
    conv_mod.fused_prefill_conv(
        qkvg,
        mixer._packed_conv_weight,
        conv_pool,
        starts,
        slot_indices,
        has_init,
        getattr(mamba_metadata, "kda_max_seqlen", T),
        q,
        k,
        v,
        fgs_max_spins=MAX_SPINS,
        pdl=pk2 is None,
        fgs_counters=None if pk2 is None else pk2[0],
        fgs_counter_rows=cta_m if pk2 is not None else 0,
        fgs_counter_target=qkv_tiles if pk2 is not None else 0,
        bulk_src=CONV == "bulk",
        fgs_stats=stats["conv"] if pk2 is not None else None,
    )
    bfa = kimi_k3_cute_gemm.linear(x2d, mixer._bfa_proj_weight)
    g = kimi_k3_cute_gemm.linear(bfa[:, :hd], mixer.f_b_proj.weight)
    beta = bfa[:, hd : hd + H].unsqueeze(0)
    o = _bt16(
        mixer,
        q,
        k,
        v,
        g.view(1, T, H, hd),
        beta,
        ssm_pool,
        slot_indices,
        cu_seqlens,
        mamba_metadata,
        None,
    )
    mixer._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)

    o_cta_m, _ = kimi_k3_cute_gemm.rubin_tile(mixer.hidden_size, d, T)
    pk8 = counters.words(P_CORE, pair, -(-T // o_cta_m)) if P_CORE in edges_on() else None
    norm_rows_per_cta = norm_mod.rows_per_cta(T * H, device.index)
    core = norm_mod.kda_gated_rmsnorm(
        o.view(T * H, hd),
        qkvg[:, 3 * d :].view(T, H, hd),
        mixer.o_norm.weight,
        mixer.o_norm.eps,
        use_pdl=True,
        fgs_ctas=NORM_CTAS if pk8 is not None else 0,
        fgs_counters=pk8,
        fgs_counter_rows=o_cta_m * H if pk8 is not None else 0,
        fgs_gate=(pk2[1], cta_m * H, gate_tiles) if pk2 is not None and NORM_GATE else None,
        fgs_stats=stats["norm"] if pk2 is not None and NORM_GATE else None,
    )
    gate = None if pk8 is None else (pk8, H, norm_rows_per_cta, MAX_SPINS)
    mixer._k3_prefill_pending = (pair, prev, T, False, gate)
    edges.debug(
        "kda.prefill_chain",
        T,
        f"{mode()} counter link pair {pair} edges {sorted(e for e in edges_on() if e in (P_QKVG, P_CORE))} "
        f"blocks {blocks} x {cta_m} rows, qkv tiles {qkv_tiles}, norm rows/cta {norm_rows_per_cta}",
    )
    return core.view(T, H, hd)


def _bt16(mixer, q, k, v, g, beta, ssm_pool, slot_indices, cu_seqlens, mamba_metadata, out):
    """The BT16 chain on raw q / k / v as ``trtllm::kda_prefill`` runs it (FP32 state), into ``out`` when given
    (a p_o slot: the decomposition kernel's producer build)."""
    device = q.device
    n, heads = slot_indices.numel(), q.shape[2]
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    decomp = 2 * n * heads <= sms
    beta = beta if decomp and beta.stride(1) * 2 % 16 == 0 else beta.contiguous()
    if beta.data_ptr() % 16:
        beta = beta.clone(memory_format=torch.contiguous_format)
    producer = out is not None
    if producer and not decomp:
        raise RuntimeError("pk7 needs BT16's decomposition route")
    compiled = _compile_bt16(
        device, mixer.gate_lower_bound, "decomp" if producer else None, producer
    )
    output = out if out is not None else torch.empty_like(q)
    nonempty = min(q.shape[1], n)
    capacity = nonempty + (q.shape[1] - nonempty) // 16
    size = compiled.workspace_size_from_total_chunks(n, heads, capacity, device)
    workspace = torch.empty(size, dtype=torch.uint8, device=device) if size else None
    seq_order = getattr(mamba_metadata, "kda_seq_order", None)
    cu_chunks = getattr(mamba_metadata, "kda_cu_chunks", None)
    generate = seq_order is None or cu_chunks is None
    if generate:
        seq_order = torch.empty(n, dtype=torch.int32, device=device)
        cu_chunks = torch.empty(n + 1, dtype=torch.int32, device=device)
    compiled(
        q,
        k,
        v,
        g,
        mixer._A_log_f32,
        mixer._dt_bias_f32.reshape(heads, 128),
        beta,
        cu_seqlens,
        ssm_pool,
        output,
        ssm_pool,
        workspace,
        torch.cuda.current_stream(device).cuda_stream,
        scale=mixer.head_k_dim**-0.5,
        seq_order=seq_order,
        planned_cu_chunks=cu_chunks,
        planned_total_chunks=capacity,
        generate_planned_metadata=generate,
        state_indices=slot_indices,
    )
    return output


def pending(mixer) -> bool:
    return getattr(mixer, "_k3_prefill_pending", None) is not None


def o_proj(mixer, core2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``core2d @ weight^T`` closing the chain :func:`forward` opened: consumes p_core, produces p_attn_o (registered
    for the pre-MoE mix), arms every slot of the previous layer."""
    pair, prev, T, core_in_slot, gate = mixer._k3_prefill_pending
    mixer._k3_prefill_pending = None
    if core2d.shape[0] != T:
        raise RuntimeError(
            f"KDA prefill chain: o_proj got {core2d.shape[0]} rows for a {T}-row layer"
        )
    if fgs() and SYNC == "counter":
        return kimi_k3_cute_gemm.linear_fgs(
            core2d, weight, core2d.new_empty(T, weight.shape[0]), early_trigger=False, gate=gate,
            retry_counter=_stats_of(core2d.device)["o_proj"] if gate is not None else None,
        )  # fmt: skip
    slots = _slots.get(core2d.device) if fgs() else None
    out_slot = slots.slot(P_ATTN_O, pair, T) if slots is not None else None
    arm = ()
    if slots is not None:
        arm = tuple(
            r
            for r in (slots.arm_region(e, prev) for e in (P_QKVG, P_O, P_CORE, P_ATTN_O))
            if r is not None
        )
    out = kimi_k3_cute_gemm.linear_fgs(
        core2d,
        weight,
        out_slot if out_slot is not None else core2d.new_empty(T, weight.shape[0]),
        consume=core_in_slot,
        produce=out_slot is not None,
        raster_along=RASTER if out_slot is not None else "m",
        arm=arm,
        start_delay_ns=DELAY_NS if out_slot is not None else 0,
    )
    if out_slot is not None:
        slots.dirty[P_ATTN_O][pair] = T
        edges.register_external(out, P_ATTN_O)
    return out
