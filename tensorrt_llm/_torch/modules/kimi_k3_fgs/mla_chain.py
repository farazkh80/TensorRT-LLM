# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode MLA layer as one PDL chain of fine-grained-sync edges (``TLLM_K3_FGS=chain``).

Every kernel runs on the main stream with its original grid; each edge tensor is an :mod:`edges` slot, so each
kernel is a report_valid / Lamport consumer of its inputs and an early-triggering, -0.0-free producer of its output:

  pre-attention mix --x_norm (2 readers)--> fused q_a | kv_a GEMM --qkv_a (3)--> q_a norm --q_a_norm--> q_b GEMM
                                        \\-> g_proj GEMM --g--> output gate         kv_a norm --kv_c--> latent append
  q_b --q (2)--> W_UK BMM --fused_q[..., :512]--> split_kv + reduction --attn_latent--> W_UV BMM --attn_out--> gate
       \\-> latent append --fused_q[..., 512:], this step's cache rows--> split_kv
  output gate --gated--> o_proj --attn_o--> pre-MoE mix (consumer only: the MoE input is a plain buffer)

:func:`begin` launches the layer's step inputs on the edge workspaces' reset stream, beside the pre-attention mix
and the head-phase GEMMs: the decode page table, once per forward the split-KV plan, and the -0.0 sentinel in this
step's cache rows and in the partial LSE entries the reduction reads. The reset stream forks from the main stream
where the mix is launched, behind the previous layer's MoE and so behind the previous MLA layer's reduction, the last
reader of the split-KV workspace and plan that every MLA layer shares. :func:`run` joins it right before the latent
append, the first writer of this step's cache rows; split_kv and the reduction, which read the page table, plan and
sentinels, follow on the main stream. The chain covers decode batches of ``TLLM_K3_FGS_MLA_MIN_ROWS`` to
``TLLM_K3_FGS_MLA_MAX_ROWS`` rows; any edge without a slot leaves the whole layer on the stock path, except attn_o:
outside the recipe's ``mla.attn_o`` rows o_proj writes a plain buffer and the pre-MoE mix waits for it to complete
(a mix polling the slot stays resident beside a multi-wave o_proj and stretches it).

The chain's reduction reduces several head rows per CTA from 8 decode rows up (``TLLM_K3_FGS_MLA_RED_ROWS``, default
``auto``): fewer CTAs let W_UV, its PDL dependent, launch sooner; the output is bit-identical. With
``TLLM_K3_FGS_DEBUG=1`` the path log names the rows per CTA (and the skip-empty split clusters when on) per row count.

Consumer wait of the chain's side kernels (``TLLM_K3_FGS_MLA_CHECK``): ``sw`` (default) polls the slots with
``ld.global.cg`` / ``ld.relaxed.gpu`` loads, ``tma`` loads them by TMA with ``report_valid`` and re-issues invalid
loads; per-site overrides as ``tma,gate=sw`` (sites ``norm``, ``append``, ``rows``, ``lse``, ``gate``; ``qpace`` =
the split-KV Q retry warp's pacing: ``tma`` waits with ``mbarrier.try_wait`` instead of ``nanosleep``). With
``lse=tma`` the partial LSE entries are padded to 16 B. ``TLLM_K3_FGS_MLA_STORE=tma`` makes the side kernels store
through TMA. Links off (``edges.links_off``): the side kernels and the split-KV / reduction wait for their producer
grid, load without ``report_valid`` and the per-step cache-row / LSE fills are skipped; ``edges.links_trigger`` places
their PDL trigger. ``TLLM_K3_FGS_MLA_GPROJ_STREAM=aux`` runs g_proj on the MLA module's aux stream, forked after the
fused q_a | kv_a GEMM and joined before the gate. ``TLLM_K3_FGS_MLA_AUX`` (links off only; ``kvnorm``, ``append``)
runs the kv_a norm beside the q_a norm and the latent append beside W_UK on that stream, as the stock path does.
``TLLM_K3_FGS_WAIT_STATS=1`` counts the re-issues (``mla.<site>.retry``) and the software waits (``mla.<site>.sw_wait``,
``mla.q.sw_sleep``) of every site.
"""

import functools
import math
import os
from typing import Dict, NamedTuple, Optional, Tuple

import torch

from .. import kimi_k3_cute_gemm
from . import edges, recipe, wait_stats

X_NORM = "mla.x_norm"
QKV_A = "mla.qkv_a"
GATE = "mla.g"
Q_A_NORM = "mla.q_a_norm"
KV_C = "mla.kv_c"
Q = "mla.q"
FUSED_Q = "mla.fused_q"
ATTN_LATENT = "mla.attn_latent"
ATTN_OUT = "mla.attn_out"
GATED = "mla.gated"
ATTN_O = "mla.attn_o"

# (edge, readers) in the launch order of the producers.
_EDGES = (
    (X_NORM, 2),
    (QKV_A, 3),
    (GATE, 1),
    (Q_A_NORM, 1),
    (KV_C, 1),
    (Q, 2),
    (FUSED_Q, 1),
    (ATTN_LATENT, 1),
    (ATTN_OUT, 1),
    (GATED, 1),
    (ATTN_O, 1),
)

# At 1 row the table's g_proj / o_proj are direct GEMMs; the chain runs their direct FGS builds
# (``kimi_k3_cute_gemm``), which match the stock path bit for bit.
# Up to this many rows g_proj launches after the reduction instead of right after the fused q_a | kv_a GEMM: at 1 row
# it is a multi-wave direct GEMM whose waves would hold back every later launch of the attention path.
LATE_GPROJ_ROWS = int(os.environ.get("TLLM_K3_FGS_MLA_LATE_GPROJ_ROWS", "1"))
MIN_ROWS = int(os.environ.get("TLLM_K3_FGS_MLA_MIN_ROWS", "1"))
MAX_ROWS = int(os.environ.get("TLLM_K3_FGS_MLA_MAX_ROWS", "32"))
# Split-KV clusters without K tiles exit at entry (``KimiK3MlaDecodeFgs(fgs_skip_empty=True)``).
SKIP_EMPTY_SPLITS = os.environ.get("TLLM_K3_FGS_MLA_SKIP_EMPTY", "0") == "1"
# Split-KV reduction: head rows per CTA (1, 2, 4, or ``auto``: 1 below 8 decode rows, 2 below 16, else 4) and the
# launch's preferred smem carveout in percent (unset: the driver's default for its 1 KB per CTA).
RED_ROWS = os.environ.get("TLLM_K3_FGS_MLA_RED_ROWS", "auto")
RED_CARVEOUT = int(os.environ.get("TLLM_K3_FGS_MLA_RED_CARVEOUT", "-1"))

CHECK_SITES = ("norm", "append", "rows", "lse", "gate", "qpace")
CHECK_ENV = "TLLM_K3_FGS_MLA_CHECK"
STORE_ENV = "TLLM_K3_FGS_MLA_STORE"
GPROJ_STREAM_ENV = "TLLM_K3_FGS_MLA_GPROJ_STREAM"
AUX_ENV = "TLLM_K3_FGS_MLA_AUX"


def _parse_check(value: str) -> Dict[str, str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    default = "sw"
    if items and "=" not in items[0]:
        default = items.pop(0)
    if default not in ("sw", "tma"):
        raise ValueError(f"{CHECK_ENV}={value!r}: the default must be sw or tma")
    check = {site: default for site in CHECK_SITES}
    for item in items:
        site, _, mode = item.partition("=")
        if site not in CHECK_SITES or mode not in ("sw", "tma"):
            raise ValueError(f"{CHECK_ENV}: bad item {item!r} (sites {CHECK_SITES}, modes sw / tma)")
        check[site] = mode
    return check


CHECK = _parse_check(os.environ.get(CHECK_ENV, "sw"))
STORE = os.environ.get(STORE_ENV, "thread")
GPROJ_STREAM = os.environ.get(GPROJ_STREAM_ENV, "main")
AUX = frozenset(filter(None, os.environ.get(AUX_ENV, "").split(",")))
if STORE not in ("thread", "tma"):
    raise ValueError(f"{STORE_ENV}={STORE!r}; valid: thread, tma")
if GPROJ_STREAM not in ("main", "aux"):
    raise ValueError(f"{GPROJ_STREAM_ENV}={GPROJ_STREAM!r}; valid: main, aux")
if AUX - {"kvnorm", "append"}:
    raise ValueError(f"{AUX_ENV}={','.join(sorted(AUX))!r}; valid items: kvnorm, append")


def _red_rows(batch: int) -> int:
    if RED_ROWS == "auto":
        return 1 if batch < 8 else 2 if batch < 16 else 4
    return int(RED_ROWS)


def _trigger() -> str:
    return "end" if edges.links_off() and edges.links_trigger() == "stock" else "entry"


def _aux_on(item: str) -> bool:
    return item in AUX and edges.links_off()


# In-kernel sentinel re-arm (comma list): ``kv_c`` = the latent append re-arms compressed_kv after reading it (its last
# reader); ``gated`` = the output gate arms the slot of the edge's next use in the step. The framework fill still
# runs for these slots (redundant); TLLM_K3_FGS_MLA_REARM_DELAY_NS delays the re-arm stores (hazard tests).
REARM = frozenset(filter(None, os.environ.get("TLLM_K3_FGS_MLA_REARM", "").split(",")))
REARM_DELAY_NS = int(os.environ.get("TLLM_K3_FGS_MLA_REARM_DELAY_NS", "0"))
# Split-plan buffer capacity (requests); the stock planner serves up to 1024.
_PLAN_CAPACITY = 1024


class Weights(NamedTuple):
    fused_a: torch.Tensor  # [q_lora + kv_lora + rope, hidden]
    q_a_norm: torch.Tensor
    q_a_eps: float
    kv_a_norm: torch.Tensor
    kv_a_eps: float
    q_b: torch.Tensor  # [heads * (nope + rope), q_lora]
    k_b: torch.Tensor  # k_b_proj_trans [heads, kv_lora, nope]
    v_b: torch.Tensor  # v_b_proj [heads, v, kv_lora]
    g_proj: torch.Tensor  # [heads * v, hidden]
    o_proj: torch.Tensor  # [hidden, heads * v]
    softmax_scale: float


def configure_window(window: Optional[Tuple[int, int]]) -> None:
    """Set the chain's row window to the recipe's mla.chain span (None: the chain never runs)."""
    global MIN_ROWS, MAX_ROWS
    MIN_ROWS, MAX_ROWS = window if window is not None else (1 << 30, 0)


def declare(
    layer_idx: int,
    hidden: int,
    q_lora: int,
    kv_lora: int,
    rope: int,
    heads: int,
    nope: int,
    v_dim: int,
) -> None:
    """Declare the chain edges of one MLA layer."""
    if not edges.chain() or MIN_ROWS > MAX_ROWS:
        return
    widths = {
        X_NORM: hidden,
        QKV_A: q_lora + kv_lora + rope,
        GATE: heads * v_dim,
        Q_A_NORM: q_lora,
        KV_C: kv_lora,
        Q: heads * (nope + rope),
        FUSED_Q: heads * (kv_lora + rope),
        ATTN_LATENT: heads * kv_lora,
        ATTN_OUT: heads * v_dim,
        GATED: heads * v_dim,
        ATTN_O: hidden,
    }
    for name, _ in _EDGES:
        edges.declare(name, widths[name], layer_idx, max_rows=MAX_ROWS, min_rows=MIN_ROWS)


def weights_of(mla) -> Optional[Weights]:
    """The chain's view of a Kimi K3 ``MLA`` module, or None when its decode is not the chain's kernel sequence
    (quantized or biased projections, no fused q_a | kv_a projection, no output gate, sparse / rotary / scaled
    variants)."""
    if (
        not getattr(mla, "fuse_qkv_a_proj", False)
        or not getattr(mla, "use_output_gate", False)
        or getattr(mla, "is_lite", False)
        or getattr(mla, "sparse_attn_hooks", None) is not None
        or getattr(mla, "apply_rotary_emb", True)
        or getattr(mla, "llama_4_scaling", False)
    ):
        return None
    linears = (mla.kv_a_proj_with_mqa, mla.q_b_proj, mla.g_proj, mla.o_proj)
    if any(m.bias is not None or m.weight.dtype != torch.bfloat16 for m in linears):
        return None
    k_b, v_b = getattr(mla, "k_b_proj_trans", None), getattr(mla, "v_b_proj", None)
    if k_b is None or v_b is None or k_b.dtype != torch.bfloat16 or v_b.dtype != torch.bfloat16:
        return None
    return Weights(
        mla.kv_a_proj_with_mqa.weight,
        mla.q_a_layernorm.weight,
        mla.q_a_layernorm.variance_epsilon,
        mla.kv_a_layernorm.weight,
        mla.kv_a_layernorm.variance_epsilon,
        mla.q_b_proj.weight,
        k_b,
        v_b,
        mla.g_proj.weight,
        mla.o_proj.weight,
        1.0 / (math.sqrt(mla.qk_nope_head_dim + mla.qk_rope_head_dim) * mla.mqa.q_scaling),
    )


class Chain:
    """One MLA layer's slots and decode-step inputs, from :func:`begin` to :func:`run`."""

    def __init__(
        self,
        leases: Dict[str, object],
        decode: "_Decode",
        weights: Weights,
        prepared: torch.cuda.Event,
        aux_stream: Optional[torch.cuda.Stream],
    ):
        self.leases = leases
        self.decode = decode
        self.weights = weights
        self.prepared = prepared
        self.aux_stream = aux_stream

    def slot(self, name: str) -> torch.Tensor:
        return self.leases[name].tensor


def begin(
    owner,
    kv_pool: torch.Tensor,
    page_rows: torch.Tensor,
    kv_lens: torch.Tensor,
    max_requests: int,
    weights: Weights,
) -> Optional[Chain]:
    """Acquire every slot of one MLA layer's decode step and launch its step inputs on the reset stream, or return
    None (nothing held, nothing launched).

    ``kv_pool`` is the layer's ``get_buffers`` view, ``page_rows`` its ``[B, max_blocks]`` block offsets and
    ``kv_lens`` the int32 KV lengths including this step's token; ``owner`` keeps the layer's persistent buffers
    (and its ``aux_stream`` serves the g_proj / kv norm / append side launches when selected).
    Call it right before the pre-attention mix, which then writes ``chain.slot(X_NORM)``; no PDL launch that
    triggers early may come in between.
    """
    if not edges.chain():
        return None
    rows = page_rows.shape[0]
    if not MIN_ROWS <= rows <= MAX_ROWS or not recipe.slot_allowed(X_NORM, rows):
        return None
    decode = _Decode.build(owner, kv_pool, page_rows, kv_lens, max_requests, weights)
    if decode is None:
        return None
    leases = {}
    for name, readers in _EDGES:
        if name == ATTN_O and not recipe.slot_allowed(ATTN_O, rows):
            edges.debug(ATTN_O, rows, "off: o_proj writes a plain buffer")
            continue
        lease = edges.acquire(name, rows, readers=readers, window=(MIN_ROWS, MAX_ROWS))
        if lease is None:
            for held in leases.values():
                edges.release(held)
            return None
        leases[name] = lease
    side = edges.reset_stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        decode.prepare()
    prepared = torch.cuda.Event()
    prepared.record(side)
    aux = getattr(owner, "aux_stream", None)
    if aux is None and (GPROJ_STREAM == "aux" or (AUX and edges.links_off())):
        edges.debug("mla.aux", rows, "no aux stream on the MLA module: side launches stay on the main stream")
    return Chain(leases, decode, weights, prepared, aux)


def run(chain: Chain) -> torch.Tensor:
    """The MLA layer from x_norm to o_proj on the current stream; returns the ``ATTN_O`` slot (the decoder layer's
    pre-MoE mix consumes it)."""
    k = _kernels()
    w = chain.weights
    leases = chain.leases
    x = chain.slot(X_NORM)
    rows = x.shape[0]
    heads, kv_lora, nope = w.k_b.shape
    v_dim = w.v_b.shape[1]
    q_lora = w.q_a_norm.numel()
    cur = torch.cuda.current_stream()
    aux = chain.aux_stream
    counters = k.counters(x.device)
    links_off = edges.links_off()
    trigger = _trigger()
    side_kwargs = dict(fgs_consumer=True, fgs_producer=True, fgs_store=STORE, fgs_wait_grid=links_off,
                       fgs_trigger=trigger)  # fmt: skip

    qkv_a = chain.slot(QKV_A)
    kimi_k3_cute_gemm.linear(x, w.fused_a, qkv_a)
    g = chain.slot(GATE)
    late_gproj = rows <= LATE_GPROJ_ROWS
    g_done = None
    if not late_gproj:
        if GPROJ_STREAM == "aux" and aux is not None:
            # Fork after the fused q_a | kv_a GEMM, join before the gate.
            aux.wait_stream(cur)
            with torch.cuda.stream(aux):
                kimi_k3_cute_gemm.linear(x, w.g_proj, g)
            g_done = torch.cuda.Event()
            g_done.record(aux)
            edges.debug("mla.g_proj", rows, "aux stream beside the query path")
        else:
            kimi_k3_cute_gemm.linear(x, w.g_proj, g)

    q_a_n = chain.slot(Q_A_NORM)
    kv_c = chain.slot(KV_C)
    norm_kwargs = dict(side_kwargs, fgs_check=CHECK["norm"], fgs_counters=counters["norm"])
    kv_done = None
    if _aux_on("kvnorm") and aux is not None:
        aux.wait_stream(cur)
        with torch.cuda.stream(aux):
            k.rmsnorm(qkv_a[:, q_lora : q_lora + kv_lora], w.kv_a_norm, kv_c, w.kv_a_eps, **norm_kwargs)
            edges.consumed(leases[QKV_A])
        kv_done = torch.cuda.Event()
        kv_done.record(aux)
        edges.debug("mla.kv_a_norm", rows, "aux stream beside the q_a norm")
    k.rmsnorm(qkv_a[:, :q_lora], w.q_a_norm, q_a_n, w.q_a_eps, **norm_kwargs)
    edges.consumed(leases[QKV_A])
    if kv_done is None:
        k.rmsnorm(qkv_a[:, q_lora : q_lora + kv_lora], w.kv_a_norm, kv_c, w.kv_a_eps, **norm_kwargs)
        edges.consumed(leases[QKV_A])

    q = chain.slot(Q).view(rows, heads, -1)
    kimi_k3_cute_gemm.linear(q_a_n, w.q_b, q.view(rows, -1))
    fused_q = chain.slot(FUSED_Q).view(rows, heads, -1)
    d = chain.decode
    append_kwargs = dict(side_kwargs, fgs_check=CHECK["append"], fgs_counters=counters["append"])

    def append():
        if kv_done is not None:
            torch.cuda.current_stream().wait_event(kv_done)
        torch.cuda.current_stream().wait_event(chain.prepared)
        k.latent_append(
            q[..., nope:],
            kv_c,
            qkv_a[:, q_lora + kv_lora :],
            fused_q,
            d.kv_pool,
            d.page_rows,
            d.kv_lens,
            **append_kwargs,
        )
        for name in (Q, KV_C, QKV_A):
            edges.consumed(leases[name])

    append_done = None
    if _aux_on("append") and aux is not None:
        # Beside the W_UK BMM, as the stock path runs it.
        aux.wait_stream(cur)
        with torch.cuda.stream(aux):
            append()
        append_done = torch.cuda.Event()
        append_done.record(aux)
        edges.debug("mla.append", rows, "aux stream beside W_UK")
    kimi_k3_cute_gemm.bmm(
        q[..., :nope].transpose(0, 1), w.k_b, fused_q[..., :kv_lora].transpose(0, 1)
    )
    if append_done is None:
        append()
    else:
        cur.wait_event(append_done)

    latent = chain.slot(ATTN_LATENT)
    d.run(fused_q, latent, k.decode_counters(x.device))
    edges.consumed(leases[FUSED_Q])
    if late_gproj:
        kimi_k3_cute_gemm.linear(x, w.g_proj, g)
    attn_out = chain.slot(ATTN_OUT)
    kimi_k3_cute_gemm.bmm(
        latent.view(rows, heads, kv_lora).transpose(0, 1),
        w.v_b,
        attn_out.view(rows, heads, v_dim).transpose(0, 1),
    )

    if g_done is not None:
        cur.wait_event(g_done)
    gated = chain.slot(GATED)
    k.gate(attn_out, g, out=gated, fgs_check=CHECK["gate"], fgs_counters=counters["gate"], **side_kwargs)
    edges.consumed(leases[ATTN_OUT])
    edges.consumed(leases[GATE])
    o = chain.slot(ATTN_O) if ATTN_O in leases else gated.new_empty(rows, w.o_proj.shape[0])
    kimi_k3_cute_gemm.linear(gated, w.o_proj, o)
    return o


def _kernels():
    from ...attention.kernels.rubin.kimi_k3_mla import (
        mla_latent_append,
        mla_output_gate,
        mla_rmsnorm,
    )

    latent_append = mla_latent_append.mla_latent_append
    gate = mla_output_gate.mla_output_gate
    if "kv_c" in REARM:
        latent_append = functools.partial(
            latent_append, fgs_rearm_ckv=True, fgs_rearm_delay_ns=REARM_DELAY_NS
        )
    if "gated" in REARM:
        gate = functools.partial(_gate_arming_next_use, gate)
    return _Kernels(mla_rmsnorm.mla_rmsnorm, latent_append, gate)


def _gate_arming_next_use(gate, o, g, out=None, **kwargs):
    lease = edges.lease_containing(out)
    ws = None if lease is None else lease.workspace
    arm = None
    if ws is not None and lease.slot + 1 < ws.num_slots:
        # The slot the next acquire of this edge in the step hands out (the cursor walks the slots in order).
        arm = ws._lease_tensor(lease.slot + 1, lease.tensor.shape[0])
    return gate(o, g, out=out, fgs_arm=arm, fgs_arm_delay_ns=REARM_DELAY_NS, **kwargs)


class _Kernels(NamedTuple):
    rmsnorm: object
    latent_append: object
    gate: object

    @staticmethod
    def counters(device) -> Dict[str, Dict[str, int]]:
        return _side_counters(torch.device(device))

    @staticmethod
    def decode_counters(device) -> Optional[torch.Tensor]:
        return _decode_counters(torch.device(device))


@functools.lru_cache(maxsize=None)
def _side_counters(device: torch.device) -> Dict[str, Dict[str, int]]:
    """``wait_stats`` counter addresses of the side kernels (0 when the stats are off)."""
    return {
        site: {
            "retry": wait_stats.addr(f"mla.{site}.retry", device),
            "sw": wait_stats.addr(f"mla.{site}.sw_wait", device),
        }
        for site in ("norm", "append", "gate")
    }


@functools.lru_cache(maxsize=None)
def _decode_counters(device: torch.device) -> Optional[torch.Tensor]:
    """The split-KV / reduction counter addresses (``mla_decode_fgs.CNT_*``), or None when the stats are off."""
    if not wait_stats.enabled():
        return None
    from ...attention.kernels.rubin.kimi_k3_mla import mla_decode_fgs as mdf

    names = {
        mdf.CNT_Q: "mla.q.retry",
        mdf.CNT_ROWS: "mla.rows.retry",
        mdf.CNT_LSE: "mla.lse.retry",
        mdf.CNT_Q_SLEEP: "mla.q.sw_sleep",
        mdf.CNT_ROWS_SW: "mla.rows.sw_wait",
        mdf.CNT_LSE_SW: "mla.lse.sw_wait",
    }
    values = [wait_stats.addr(names[i], device) if i in names else 0 for i in range(mdf.CNT_LEN)]
    return torch.tensor(values, dtype=torch.int64, device=device)


_SHARED: Dict[tuple, torch.Tensor] = {}


def _shared(name: str, numel: int, dtype: torch.dtype, device, zero: bool = False) -> torch.Tensor:
    """A persistent buffer of at least ``numel`` elements shared by every MLA layer (one per name and device)."""
    key = (name, dtype, torch.device(device))
    buf = _SHARED.get(key)
    if buf is None or buf.numel() < numel:
        buf = (torch.zeros if zero else torch.empty)(numel, dtype=dtype, device=device)
        _SHARED[key] = buf
    return buf[:numel]


def _owned(owner, key: tuple, make) -> torch.Tensor:
    bufs = owner.__dict__.setdefault("_k3_mla_chain_bufs", {})
    if key not in bufs:
        bufs[key] = make()
    return bufs[key]


class _Decode:
    """The split-KV decode of one layer as the stock ``cute_dsl_mla_decode_fp16_blackwell`` op runs it (runner
    default tactic, workspace layout, per-request split plan), built with the fine-grained-sync roles."""

    _compiled: Dict[tuple, object] = {}

    @classmethod
    def build(
        cls, owner, kv_pool, page_rows, kv_lens, max_requests, weights
    ) -> Optional["_Decode"]:
        heads, kv_lora, _ = weights.k_b.shape
        rope = weights.fused_a.shape[0] - weights.q_a_norm.numel() - kv_lora
        if (
            kv_pool.dim() != 5
            or kv_pool.shape[1] != 1
            or kv_pool.shape[3] != 1
            or kv_pool.shape[-1] < kv_lora + rope
            or kv_pool.dtype != torch.bfloat16
            or page_rows.dtype != torch.int32
            or kv_lens.dtype != torch.int32
            or kv_lens.shape[0] != page_rows.shape[0]
            or page_rows.shape[0] > _PLAN_CAPACITY
        ):
            return None
        return cls(
            owner,
            kv_pool,
            page_rows,
            kv_lens,
            max_requests,
            heads,
            kv_lora,
            rope,
            weights.softmax_scale,
        )

    def __init__(
        self, owner, kv_pool, page_rows, kv_lens, max_requests, heads, kv_lora, rope, softmax_scale
    ):
        import cutlass

        from ...attention.kernels.rubin.kimi_k3_mla import mla_decode_fgs as mdf
        from ...custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner as Runner

        self.owner, self.kv_pool, self.page_rows, self.kv_lens = owner, kv_pool, page_rows, kv_lens
        self.heads, self.kv_lora, self.softmax_scale = heads, kv_lora, softmax_scale
        self.page_size = kv_pool.shape[2]
        batch = page_rows.shape[0]
        self.batch = batch
        self.c_latent, self.c_rope = _decode_pool_views(kv_pool, kv_lora, rope)
        self.lse_pad = mdf.LSE_PAD if CHECK["lse"] == "tma" else 1

        runner = Runner(cutlass.BFloat16, heads, 1, self.page_size, max_requests)
        self.runner = runner
        max_batch = max(batch, max_requests)
        lse_off, lse_size, split_off, split_size, total = runner.get_workspace_layout(
            heads, 1, kv_lora, max_batch, cutlass.Float32
        )
        if self.lse_pad > 1:
            # Each partial LSE entry takes lse_pad words instead of one.
            split_size = -(-(split_size * (kv_lora + self.lse_pad) // (kv_lora + 1)) // 256) * 256
            total = split_off + split_size
        ws = _shared("mla_decode_workspace", total, torch.uint8, kv_pool.device, zero=True)
        lse = ws[lse_off : lse_off + lse_size].view(torch.float32).view(max_batch, 1, heads)
        self.lse = lse[:batch].permute(2, 1, 0)
        self.split_ws = ws[split_off : split_off + split_size]
        split_kv = runner.default_tactic(batch)[2]
        q_tiles = runner.get_num_q_tiles(heads, 1)
        split_rows = split_size // (runner._MMA_TILE_M * (kv_lora + self.lse_pad) * 4)
        fitting = split_rows // (batch * q_tiles)
        self.use_workspace = split_kv > 1 and fitting >= 1
        self.split_kv = max(1, min(split_kv, fitting))
        if self.lse_pad > 1:
            self.split_kv = min(self.split_kv, mdf.LSE_TMA_SPLITS)
        self.q_tiles = q_tiles
        # Persistent buffers are allocated here, on the main stream; prepare() runs on the reset stream.
        max_blocks = page_rows.shape[1]
        self.page_table_buf = _owned(
            owner,
            ("page_table", max_blocks, batch, page_rows.device),
            lambda: torch.empty_strided(
                (max_blocks, batch), (1, max_blocks), dtype=torch.int32, device=page_rows.device
            ),
        )
        self.plan_buf = (
            _shared("mla_split_plan", _PLAN_CAPACITY, torch.int32, kv_pool.device)[:batch]
            if self.use_workspace
            else None
        )
        self.page_table = None
        self.plan = None

    def prepare(self) -> None:
        """Launch the layer's step inputs: page table, split plan (once per forward), current-row and LSE sentinels
        (the sentinels only with the links on)."""
        from ...attention.kernels.rubin.kimi_k3_mla import mla_decode_fgs
        from ...attention.kernels.rubin.kimi_k3_mla.mla_latent_append import mla_decode_page_table

        self.page_table = mla_decode_page_table(self.kv_pool, self.page_rows, self.page_table_buf)
        if self.use_workspace:
            self.plan = _split_plan(self.runner, self.kv_lens, self.plan_buf, self.heads)
        if edges.links_off():
            return
        mla_decode_fgs.fill_current_rows_sentinel(self.c_latent, self.page_table, self.kv_lens, 1)
        if self.use_workspace:
            mla_decode_fgs.fill_lse_sentinel(
                self.split_ws, self.heads, self.q_tiles, self.kv_lens, self.split_kv, self.plan,
                lse_pad=self.lse_pad,
            )  # fmt: skip

    def run(self, fused_q: torch.Tensor, out: torch.Tensor, counters: Optional[torch.Tensor]) -> None:
        """Split-KV kernel and reduction from the ``fused_q`` slot [B, H, 576] into the ``out`` slot [B, H * 512];
        ``counters`` is the ``wait_stats`` address tensor of ``_decode_counters`` (or None)."""
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        from ...attention.kernels.rubin.kimi_k3_mla.mla_decode_fgs import KimiK3MlaDecodeFgs

        batch, heads, kv_lora = self.batch, self.heads, self.kv_lora
        q4 = fused_q.view(batch, 1, heads, -1)
        q_latent = q4[..., :kv_lora].permute(2, 3, 1, 0)
        q_rope = q4[..., kv_lora:].permute(2, 3, 1, 0)
        o = out.view(batch, 1, heads, kv_lora).permute(2, 3, 1, 0)
        split_ws = self.split_ws if self.use_workspace else None
        plan = self.plan if self.use_workspace else None
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        red_rows = _red_rows(batch) if plan is not None else 1
        links_off = edges.links_off()
        if plan is not None:
            edges.debug("mla.reduction", batch, f"{red_rows} head rows per CTA")
        if SKIP_EMPTY_SPLITS:
            edges.debug("mla.split_kv", batch, "split clusters without K tiles exit at entry")
        edges.debug(
            "mla.decode", batch,
            f"rows check {CHECK['rows']}, lse check {CHECK['lse']}, q pace {CHECK['qpace']}, store {STORE}, "
            f"links {'off' if links_off else 'on'}",
        )  # fmt: skip
        key = (
            self.page_size,
            heads,
            self.use_workspace,
            plan is not None,
            red_rows,
            counters is not None,
            fused_q.device,
        )
        fn = self._compiled.get(key)
        if fn is None:
            import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops as ops

            var_split = plan is not None
            clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
            mla = KimiK3MlaDecodeFgs(
                cutlass.Float32, cutlass.Float32, (128, 128), (128, 256), clusters, self.page_size, 0.0, False,
                True, var_split, num_heads=heads, seq_len_q=1,
                reducer_max_splits=ops.MLA_DECODE_MAX_SPLITS if var_split else 32,
                fgs_consumer=True, fgs_producer=True, fgs_reduction=True, fgs_skip_empty=SKIP_EMPTY_SPLITS,
                fgs_red_rows=red_rows,
                fgs_red_carveout=RED_CARVEOUT if RED_CARVEOUT >= 0 else None,
                fgs_rows_check=CHECK["rows"], fgs_lse_check=CHECK["lse"] if var_split else "sw",
                fgs_q_pace="try" if CHECK["qpace"] == "tma" else "sleep", fgs_store=STORE,
                fgs_wait_grid=links_off, fgs_trigger=_trigger(),
            )  # fmt: skip

            def ct(t, leading, align=16):
                return from_dlpack(t, assumed_align=align).mark_layout_dynamic(leading_dim=leading)

            o_ct = (
                from_dlpack(o, assumed_align=16)
                .mark_layout_dynamic(leading_dim=1)
                .mark_compact_shape_dynamic(mode=1, stride_order=(3, 2, 0, 1), divisibility=8)
            )
            fn = cute.compile(
                mla.run_fgs,
                ct(q_latent, 1), ct(q_rope, 1), ct(self.c_latent, 1), ct(self.c_rope, 1), ct(self.page_table, 0),
                o_ct, ct(self.lse, 0),
                None if split_ws is None else from_dlpack(split_ws, assumed_align=32).mark_layout_dynamic(),
                self.split_kv, from_dlpack(self.kv_lens, assumed_align=16).mark_layout_dynamic(), None, None,
                None if plan is None else from_dlpack(plan, assumed_align=4).mark_layout_dynamic(),
                cutlass.Float32(self.softmax_scale), cutlass.Float32(1.0), None,
                None if counters is None else from_dlpack(counters, assumed_align=8).mark_layout_dynamic(),
                stream,
                options="--opt-level 2",
            )  # fmt: skip
            self._compiled[key] = fn
        page_table = self.page_table
        if page_table.shape[0] == 1 and page_table.shape[1] == 1:
            # A (1, 1) table has no extent > 1 to deduce the leading dimension from at call time.
            page_table = from_dlpack(page_table, assumed_align=16).mark_layout_dynamic(
                leading_dim=0
            )
        fn(q_latent, q_rope, self.c_latent, self.c_rope, page_table, o, self.lse, split_ws, self.split_kv,
           self.kv_lens, None, None, plan, self.softmax_scale, 1.0, None, counters, stream)  # fmt: skip


def _decode_pool_views(kv_pool: torch.Tensor, kv_lora: int, rope: int):
    """``CuteDslMlaFmha._run_mla_decode``'s ([page_size, kv_lora, pages], [page_size, rope, pages]) cache views: a
    per-layer view of an interleaved pool becomes the packed view of all layers' pages (the page table carries the
    layer's slot)."""
    packed = 1
    for size in kv_pool.shape[1:]:
        packed *= size
    stride = kv_pool.stride(0)
    layers = stride // packed if packed else 1
    if layers > 1 and stride == layers * packed:
        shape = (kv_pool.shape[0] * layers, *kv_pool.shape[1:])
        kv_pool = kv_pool.as_strided(shape, (packed, *kv_pool.stride()[1:]), 0)
    pages = kv_pool[:, 0, :, 0, : kv_lora + rope]
    return pages[..., :kv_lora].permute(1, 2, 0), pages[..., kv_lora:].permute(1, 2, 0)


def _split_plan(runner, kv_lens: torch.Tensor, buf: torch.Tensor, heads: int) -> torch.Tensor:
    """This forward's per-request split counts (``CuteDslMlaFmha._split_kv_plan``) in ``buf``, launched by the first
    MLA layer that needs them."""
    batch = buf.shape[0]
    step = edges.step_cache()
    key = ("mla_split_plan", kv_lens.data_ptr(), batch)
    plan = step.get(key)
    if plan is None:
        from ...attention.mla import k3_cute_mla_enabled

        if k3_cute_mla_enabled("plan"):
            from ...attention.kernels.rubin.kimi_k3_mla.mla_split_kv_plan import (
                launch_mla_split_kv_plan as launch,
            )
        else:
            from ...attention.backends.fmha.cute_dsl_mla import launch_mla_split_kv_plan as launch
        launch(
            kv_lens,
            buf,
            num_requests=batch,
            rows_per_request=runner.get_num_q_tiles(heads, 1),
            wave_clusters=max(runner._get_max_active_blocks() // runner._CLUSTER_SHAPE_MNK[0], 1),
        )
        plan = step[key] = buf
    return plan
