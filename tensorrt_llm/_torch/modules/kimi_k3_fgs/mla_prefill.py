# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA prefill as one PDL chain of fine-grained-sync edges (``TLLM_K3_FGS_PREFILL=1``), or the same kernels chained by
standard PDL (``TLLM_K3_FGS_PREFILL=pdl``, the FGS-off twin). Decode is untouched: the Kimi K3 MLA module calls
:func:`forward` only for a context-only batch.

  fused q_a | kv_a GEMM -> q_a norm -> kv_a norm -> q_b GEMM -> k_nope BMM -> latent append -> v GEMM
  --pm7 (v slot)--> context FMHA --pm9 (o slot)--> output gate --pm11 (counters)--> o_proj
                                 \\--gproj--> g_proj (no grid wait, fills the FMHA tail) --> output gate

Every consumer loads with TMA. pm7 and pm9 are Lamport edges checked by the hardware (TMA ``report_valid`` on the
bf16 -0.0 sentinel: the FMHA on V, the TMA output gate on o); pm11 is a counter link (word per o_proj M tile; the
gate CTAs ``red.release`` after their stores, o_proj's TMA warp acquires, ``fence.proxy.async.global``, then loads).

Chain invariant (``griddepcontrol.wait`` is not transitive: it waits for the immediate predecessor's completion
only, and a predecessor's trigger says nothing about its own stores): a kernel that skips its grid wait reads its
predecessor's output only through a check, and may read an older kernel's output plainly only if every kernel from
that producer up to and including the predecessor waits at entry (the predecessor's trigger comes after its wait, so
the producer is complete and flushed). Hence the FMHA (no wait) reads Q and K plainly: q_b, the BMM and the append
wait at entry and the v GEMM waits before it triggers; they run their pzo builds because the FMHA still checks every
operand against -0.0. The gate (no wait) checks both o and g. g_proj (no wait with ``gproj``) reads x_norm, complete
since the fused GEMM. ``gproj`` needs ``pm9`` (the gate must check what a g_proj that did not wait may still be
writing, and the FMHA output).

Slots: v, o and g of each pair, bf16 ``[pairs, MAX_ROWS, heads * v_dim]`` armed with -0.0, plus the pm11 counter slab;
re-armed producer-side by the next MLA layer's o_proj arm warp (its DRAM headroom, as the KDA chain; rows used only).
Readers of pair(L) (FMHA L, gate L, o_proj L's counter gate) complete before layer L's pre-MoE mix, which waits at
entry on o_proj L, whose counter gate covered every gate CTA, whose report_valid covered every FMHA tile. A layer that
refuses the chain while a pair is dirty fills it from the host (:func:`rearm_skipped`).

Env (read at import): ``TLLM_K3_FGS_PREFILL`` (shared with ``kda_prefill``), ``TLLM_K3_FGS_PREFILL_EDGES`` (ids
``pm7,pm9,pm11,gproj`` of this chain, default ``pm9,pm11,gproj``; KDA ids are ignored here),
``TLLM_K3_FGS_PREFILL_MIN_ROWS``, ``TLLM_K3_FGS_MLA_PREFILL_MAX_ROWS`` (slot rows, 72 KB per row per pair),
``TLLM_K3_FGS_PREFILL_DELAY_NS`` (the v GEMM spins at entry: hazard tests), ``TLLM_K3_FGS_PREFILL_MAX_SPINS`` (gate /
o_proj rounds before the grid-wait fallback).
"""

import atexit
import collections
import functools
import math
import os
from typing import Dict, List, NamedTuple, Optional

import torch

from ....logger import logger
from .. import kimi_k3_cute_gemm
from . import edges

ENV = "TLLM_K3_FGS_PREFILL"
MIN_ROWS = int(os.environ.get("TLLM_K3_FGS_PREFILL_MIN_ROWS", "1024"))
# A-11: the chain gains 3 % of the layer at 1365 tokens and is at or below noise from 5460 up, where the slot re-arm
# (72 KB per token per layer) and the TMA gate's standing cost take over.
MAX_ROWS = int(os.environ.get("TLLM_K3_FGS_MLA_PREFILL_MAX_ROWS", "4096"))
DELAY_NS = int(os.environ.get("TLLM_K3_FGS_PREFILL_DELAY_NS", "0"))
MAX_SPINS = int(os.environ.get("TLLM_K3_FGS_PREFILL_MAX_SPINS", str(1 << 16)))
COUNTER_WORDS = -(-(-(-MAX_ROWS // 64)) // 4) * 4

PM7, PM9, PM11, GPROJ = "pm7", "pm9", "pm11", "gproj"
_EDGE_IDS = (PM7, PM9, PM11, GPROJ)
# pm7 (v GEMM -> FMHA) is opt-in: the FMHA consumer build costs 7 % of the FMHA and a K = 512 GEMM has no tail (A-11).
_DEFAULT_EDGES = frozenset((PM9, PM11, GPROJ))
NEG0 = -0x8000


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
    value = os.environ.get("TLLM_K3_FGS_PREFILL_EDGES")
    if value is None:
        return _DEFAULT_EDGES
    ids = frozenset(v.strip().lower() for v in value.split(",") if v.strip()) & frozenset(_EDGE_IDS)
    if GPROJ in ids and PM9 not in ids:
        raise ValueError("TLLM_K3_FGS_PREFILL_EDGES: gproj needs pm9 (the gate must check the FMHA output)")
    return ids


def on(edge: str) -> bool:
    return fgs() and edge in edges_on()


class Weights(NamedTuple):
    fused_a: torch.Tensor  # [q_lora + kv_lora + rope, hidden]
    q_a_norm: torch.Tensor
    q_a_eps: float
    kv_a_norm: torch.Tensor
    kv_a_eps: float
    q_b: torch.Tensor  # [heads * (nope + rope), q_lora]
    w_k: torch.Tensor  # [heads, nope, kv_lora]
    w_v: torch.Tensor  # [heads, v, kv_lora]
    g_proj: torch.Tensor  # [heads * v, hidden]
    o_proj: torch.Tensor  # [hidden, heads * v]
    softmax_scale: float


def weights_of(mla) -> Optional[Weights]:
    """The chain's view of a Kimi K3 ``MLA`` module, or None when its context path is not the chain's kernel
    sequence (quantized or biased projections, no fused q_a | kv_a projection, no output gate, rotary variants)."""
    if (
        not getattr(mla, "fuse_qkv_a_proj", False)
        or not getattr(mla, "use_output_gate", False)
        or getattr(mla, "is_lite", False)
        or getattr(mla, "sparse_attn_hooks", None) is not None
        or getattr(mla, "apply_rotary_emb", True)
        or getattr(mla, "llama_4_scaling", False)
    ):
        return None
    linears = (mla.kv_a_proj_with_mqa, mla.q_b_proj, mla.kv_b_proj, mla.g_proj, mla.o_proj)
    if any(
        getattr(m, "weight", None) is None or m.bias is not None or m.weight.dtype != torch.bfloat16
        for m in linears
    ):
        return None
    w_k, w_v = mla._get_kvbproj_head_weights()
    return Weights(
        mla.kv_a_proj_with_mqa.weight,
        mla.q_a_layernorm.weight,
        mla.q_a_layernorm.variance_epsilon,
        mla.kv_a_layernorm.weight,
        mla.kv_a_layernorm.variance_epsilon,
        mla.q_b_proj.weight,
        w_k,
        w_v,
        mla.g_proj.weight,
        mla.o_proj.weight,
        1.0 / (math.sqrt(mla.qk_nope_head_dim + mla.qk_rope_head_dim) * mla.mha.q_scaling),
    )


# MLA layer indices in construction (= forward) order.
_layers: List[int] = []


def register(layer_idx: int) -> None:
    if layer_idx not in _layers:
        _layers.append(layer_idx)


def _pair(ordinal: int) -> int:
    count = len(_layers)
    return 2 if count % 2 and ordinal == count - 1 else ordinal % 2


class _Slots:
    """One device's v / o slot pairs (``buf[name][pair, :rows]``, -0.0 armed) and pm11 counter slabs
    (``cnt[pair, :words]``, zero = unpublished); ``dirty`` = rows / words the last use touched, 0 once re-armed."""

    def __init__(self, device: torch.device, width: int, pairs: int):
        self.buf: Dict[str, torch.Tensor] = {}
        self.dirty: Dict[str, List[int]] = {}
        for name, edge in (("v", PM7), ("o", PM9), ("g", PM9)):
            if edge in edges_on():
                buf = torch.empty((pairs, MAX_ROWS, width), dtype=torch.bfloat16, device=device)
                buf.view(torch.int16).fill_(NEG0)
                self.buf[name] = buf
                self.dirty[name] = [0] * pairs
        self.cnt = None
        if PM11 in edges_on():
            self.cnt = torch.zeros((pairs, COUNTER_WORDS), dtype=torch.int32, device=device)
            self.dirty["cnt"] = [0] * pairs
        total = sum(b.numel() * 2 for b in self.buf.values())
        logger.info(
            f"{ENV}=1: MLA prefill slots {sorted(self.buf)} x {pairs} pairs x {MAX_ROWS} rows -> "
            f"{total / 2**30:.2f} GiB, counters {self.cnt is not None}"
        )

    def slot(self, name: str, pair: int, rows: int) -> Optional[torch.Tensor]:
        buf = self.buf.get(name)
        if buf is None:
            return None
        self.dirty[name][pair] = rows
        return buf[pair, :rows]

    def words(self, pair: int, blocks: int) -> Optional[torch.Tensor]:
        if self.cnt is None:
            return None
        if blocks > COUNTER_WORDS:
            raise RuntimeError(f"pm11: {blocks} row blocks above the {COUNTER_WORDS}-word counter slab")
        self.dirty["cnt"][pair] = max(self.dirty["cnt"][pair], -(-blocks // 4) * 4)
        return self.cnt[pair]

    def arm_regions(self, pair: int) -> tuple:
        regions = []
        for name, buf in self.buf.items():
            rows = self.dirty[name][pair]
            if rows:
                self.dirty[name][pair] = 0
                regions.append(buf[pair, :rows])
        if self.cnt is not None and self.dirty["cnt"][pair]:
            words = self.dirty["cnt"][pair]
            self.dirty["cnt"][pair] = 0
            regions.append((self.cnt[pair, :words], 0))
        return tuple(regions)

    def fill_dirty(self) -> None:
        for name, buf in self.buf.items():
            for pair, rows in enumerate(self.dirty[name]):
                if rows:
                    buf[pair, :rows].view(torch.int16).fill_(NEG0)
                    self.dirty[name][pair] = 0
        if self.cnt is not None:
            for pair, words in enumerate(self.dirty["cnt"]):
                if words:
                    self.cnt[pair, :words].zero_()
                    self.dirty["cnt"][pair] = 0


_slots: Dict[torch.device, _Slots] = {}
# Wait statistics: gate (TMA re-issues, CTAs that re-issued), o_proj (retry slots: [1] gate rounds, [2] grid-wait
# fallbacks); zeroed after each report.
_stats: Dict[torch.device, Dict[str, torch.Tensor]] = {}


def _stats_of(device: torch.device) -> Dict[str, torch.Tensor]:
    stats = _stats.get(device)
    if stats is None:
        from .dense_gemm_fgs import RETRY_COUNTER_SLOTS

        stats = _stats[device] = {
            "gate": torch.zeros(4, dtype=torch.int32, device=device),
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


def _slots_of(device: torch.device, width: int) -> _Slots:
    slots = _slots.get(device)
    if slots is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{ENV}=1: MLA prefill slots are allocated by an eager prefill, before any capture")
        slots = _slots[device] = _Slots(device, width, 2 + len(_layers) % 2)
    return slots


def rearm_skipped(device: torch.device) -> None:
    """A layer did not run the chain: fill every dirty slot from the host (the next use's o_proj would have)."""
    slots = _slots.get(device)
    if slots is not None:
        slots.fill_dirty()


outcomes: collections.Counter = collections.Counter()
LOG_EVERY = 20
# The wait report synchronizes the device (``tolist``), so it is rarer than the outcome log.
STATS_EVERY = 200


def _note(layer_idx: int, refusal: Optional[str]) -> None:
    if not outcomes:
        atexit.register(lambda: logger.info(f"{ENV}={mode()} MLA prefill layer-forwards: {dict(outcomes)}"))
    outcome = mode() if refusal is None else f"off: {refusal}"
    if outcome not in outcomes:
        logger.info(f"{ENV}={mode()} MLA prefill chain, first {outcome!r} at MLA layer {layer_idx}")
    outcomes[outcome] += 1
    if sum(outcomes.values()) % LOG_EVERY == 0:
        logger.info(f"{ENV}={mode()} MLA prefill layer-forwards: {dict(outcomes)}")
        if edges.DEBUG and _stats and sum(outcomes.values()) % STATS_EVERY == 0:
            logger.info(f"{ENV}={mode()} MLA prefill waits (gate re-issues/ctas, o_proj -/rounds/fallbacks): "
                        f"{stats_report()}")


def _kernels():
    from ...attention.kernels.rubin.kimi_k3_mla import (
        mla_context_append,
        mla_context_fmha,
        mla_output_gate_tma,
        mla_rmsnorm,
    )

    return (
        mla_rmsnorm.mla_rmsnorm,
        mla_context_append.mla_context_append,
        mla_context_fmha.mla_context_fmha,
        mla_output_gate_tma.mla_output_gate_tma,
    )


def refusal(layer_idx: int, rows: int, weights: Optional[Weights]) -> Optional[str]:
    for refused, reason in (
        (not kimi_k3_cute_gemm.enabled(), "TLLM_K3_CUTE_GEMM off"),
        (not MIN_ROWS <= rows <= MAX_ROWS, f"{rows} rows outside [{MIN_ROWS}, {MAX_ROWS}]"),
        (weights is None, "not the chain's kernel sequence"),
        (fgs() and not kimi_k3_cute_gemm.fgs_dsl(), "report_valid needs CuTe DSL 4.9"),
        (layer_idx not in _layers, "layer not registered"),
        (fgs() and len(_layers) < 2, "slot pairs need two MLA layers"),
    ):
        if refused:
            return reason
    return None


def forward(
    owner,
    layer_idx: int,
    weights: Weights,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    kv_pool: torch.Tensor,
    page_rows: torch.Tensor,
    kv_lens: torch.Tensor,
    fgs_stats: Optional[torch.Tensor] = None,
    trace: Optional[dict] = None,
) -> Optional[torch.Tensor]:
    """The context attention output ``[T, heads * v]`` of one MLA layer (x_norm in, through the FMHA) by the chain
    or its PDL twin, or None when the chain does not apply (nothing launched). :func:`output_gate` and
    :func:`o_proj` must follow; ``owner`` carries the pending state between them. ``trace`` (tests) receives the
    intermediates by name."""
    rows = x.shape[0]
    reason = refusal(layer_idx, rows, weights)
    _note(layer_idx, reason)
    if reason is not None:
        if fgs():
            rearm_skipped(x.device)
        return None
    rmsnorm, append, fmha, _ = _kernels()
    w = weights
    T = rows
    heads, nope, kv_lora = w.w_k.shape
    v_dim = w.w_v.shape[1]
    q_lora = w.q_a_norm.numel()
    rope = w.fused_a.shape[0] - q_lora - kv_lora
    live = fgs()
    slots = _slots_of(x.device, heads * v_dim) if live else None
    ordinal = _layers.index(layer_idx)
    pair, prev = _pair(ordinal), _pair((ordinal - 1) % len(_layers))

    qkv_a = kimi_k3_cute_gemm.linear(x, w.fused_a)
    q_a_n = rmsnorm(qkv_a[:, :q_lora], w.q_a_norm, x.new_empty(T, q_lora), w.q_a_eps, enable_pdl=True)
    kv_c = rmsnorm(
        qkv_a[:, q_lora : q_lora + kv_lora], w.kv_a_norm, x.new_empty(T, kv_lora), w.kv_a_eps, enable_pdl=True
    )
    k_pe = qkv_a[:, q_lora + kv_lora :]
    # pzo on q, k and v: the FMHA consumer checks every operand against -0.0, Q and K included.
    q = kimi_k3_cute_gemm.linear_fgs(
        q_a_n, w.q_b, x.new_empty(T, heads * (nope + rope)), produce=live, early_trigger=False
    )
    k = x.new_empty(T, heads, nope + rope)
    kimi_k3_cute_gemm.bmm_fgs(
        kv_c.unsqueeze(0).expand(heads, T, kv_lora), w.w_k, k[..., :nope].transpose(0, 1), produce=live,
        early_trigger=False,
    )  # fmt: skip
    append(kv_c, k_pe, k, kv_pool, page_rows, kv_lens, cu_seqlens, max_seqlen, fgs_producer=live, pdl=not live)
    v_slot = slots.slot("v", pair, T) if live else None
    v = v_slot if v_slot is not None else x.new_empty(T, heads * v_dim)
    kimi_k3_cute_gemm.linear_fgs(
        kv_c, w.w_v.reshape(heads * v_dim, kv_lora), v, produce=live, early_trigger=v_slot is not None,
        start_delay_ns=DELAY_NS if live else 0,
    )  # fmt: skip
    o_slot = slots.slot("o", pair, T) if live else None
    o = o_slot if o_slot is not None else x.new_empty(T, heads * v_dim)
    fmha(
        q.view(T, heads, nope + rope), k, v.view(T, heads, v_dim), o.view(T, heads, v_dim), cu_seqlens, max_seqlen,
        w.softmax_scale, fgs_consumer=v_slot is not None, fgs_producer=o_slot is not None,
        pdl=v_slot is None and o_slot is None, fgs_stats=fgs_stats if v_slot is not None else None,
    )  # fmt: skip
    owner._k3_prefill_pending = (pair, prev, T, o_slot is not None)
    if trace is not None:
        trace.update(qkv_a=qkv_a, q_a_n=q_a_n, kv_c=kv_c, q=q, k=k, v=v, o=o)
    edges.debug(
        "mla.prefill_chain",
        T,
        f"{mode()} pair {pair} edges {sorted(e for e in edges_on() if live)}",
    )
    return o


def pending(owner) -> bool:
    return getattr(owner, "_k3_prefill_pending", None) is not None


def output_gate(
    owner,
    weights: Weights,
    x: torch.Tensor,
    o: torch.Tensor,
    fgs_stats: Optional[torch.Tensor] = None,
    trace: Optional[dict] = None,
) -> torch.Tensor:
    """``o * sigmoid(g_proj(x))`` for the layer :func:`forward` opened: g_proj fills the FMHA tail (``gproj``), the
    TMA gate checks o (``pm9``) and publishes the pm11 counters."""
    pair, prev, T, o_in_slot = owner._k3_prefill_pending
    gate = _kernels()[3]
    live = fgs()
    g_slot = _slots_of(o.device, o.shape[1]).slot("g", pair, T) if o_in_slot else None
    g = kimi_k3_cute_gemm.linear_fgs(
        x, weights.g_proj, g_slot if g_slot is not None else o.new_empty(o.shape), produce=o_in_slot,
        early_trigger=False, wait=not on(GPROJ),
    )  # fmt: skip
    words = None
    cta_m = 0
    if on(PM11):
        cta_m, _ = kimi_k3_cute_gemm.rubin_tile(weights.o_proj.shape[0], o.shape[1], T)
        words = _slots_of(o.device, o.shape[1]).words(pair, -(-T // cta_m))
    gated = gate(
        o, g, pdl=not o_in_slot, fgs_consumer=o_in_slot, fgs_max_spins=MAX_SPINS,
        fgs_stats=(fgs_stats if fgs_stats is not None else _stats_of(o.device)["gate"]) if o_in_slot else None,
        fgs_counters=words, fgs_counter_rows=cta_m,
    )  # fmt: skip
    link = None if words is None else (words, o.shape[1], 32 * 256, MAX_SPINS)
    owner._k3_prefill_pending = (pair, prev, T, live, link)
    if trace is not None:
        trace.update(g=g, gated=gated)
    return gated


def o_proj(owner, gated: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``gated @ weight^T`` closing the chain: consumes the pm11 counters and arms the previous pair's slots."""
    pair, prev, T, live, link = owner._k3_prefill_pending
    owner._k3_prefill_pending = None
    if gated.shape[0] != T:
        raise RuntimeError(f"MLA prefill chain: o_proj got {gated.shape[0]} rows for a {T}-row layer")
    arm = _slots_of(gated.device, gated.shape[1]).arm_regions(prev) if live else ()
    return kimi_k3_cute_gemm.linear_fgs(
        gated, weight, gated.new_empty(T, weight.shape[0]), early_trigger=False, gate=link, arm=arm,
        retry_counter=_stats_of(gated.device)["o_proj"] if link is not None else None,
    )  # fmt: skip


def forward_from_module(mla, hidden_states: torch.Tensor, attn_metadata) -> Optional[torch.Tensor]:
    """Run :func:`forward` for a Kimi K3 ``MLA`` module on a context-only batch it serves with the CuTe context
    path (``_k3_ctx_skip_concat``), or return None."""
    meta = attn_metadata
    if not (
        active()
        and mla._k3_ctx_skip_concat(meta)
        and getattr(meta, "num_ctx_cached_tokens", 0) == 0
        and not mla.mapping.has_cp_helix()
    ):
        return None
    weights = weights_of(mla)
    rows = meta.num_ctx_tokens
    if rows > hidden_states.shape[0]:
        return None
    batch = meta.num_contexts
    pool_idx = int(meta.host_kv_cache_pool_mapping[mla.mha.get_local_layer_idx(meta), 0])
    cu_seqlens = meta.mla_prepare_ctx_cu_seqlens()
    return forward(
        mla,
        mla.layer_idx,
        weights,
        hidden_states[:rows],
        cu_seqlens,
        int(meta.prompt_lens_cpu_runtime[:batch].max()),
        meta.kv_cache_manager.get_buffers(mla.mha.layer_idx),
        meta.kv_cache_block_offsets[pool_idx, :batch, 0, :],
        meta.kv_lens_cuda_runtime[:batch].to(torch.int32),
    )
