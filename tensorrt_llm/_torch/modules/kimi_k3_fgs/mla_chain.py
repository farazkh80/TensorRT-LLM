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
``TLLM_K3_FGS_MLA_MAX_ROWS`` rows; any edge without a slot leaves the whole layer on the stock path.

The chain's reduction reduces several head rows per CTA from 8 decode rows up (``TLLM_K3_FGS_MLA_RED_ROWS``, default
``auto``): fewer CTAs let W_UV, its PDL dependent, launch sooner; the output is bit-identical. With
``TLLM_K3_FGS_DEBUG=1`` the path log names the rows per CTA (and the skip-empty split clusters when on) per row count.
"""

import functools
import math
import os
from typing import Dict, NamedTuple, Optional

import torch

from .. import kimi_k3_cute_gemm
from . import edges

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
MIN_ROWS = int(os.environ.get("TLLM_K3_FGS_MLA_MIN_ROWS", "1"))
MAX_ROWS = int(os.environ.get("TLLM_K3_FGS_MLA_MAX_ROWS", "32"))
# Split-KV clusters without K tiles exit at entry (``KimiK3MlaDecodeFgs(fgs_skip_empty=True)``).
SKIP_EMPTY_SPLITS = os.environ.get("TLLM_K3_FGS_MLA_SKIP_EMPTY", "0") == "1"
# Split-KV reduction: head rows per CTA (1, 2, 4, or ``auto``: 1 below 8 decode rows, 2 below 16, else 4) and the
# launch's preferred smem carveout in percent (unset: the driver's default for its 1 KB per CTA).
RED_ROWS = os.environ.get("TLLM_K3_FGS_MLA_RED_ROWS", "auto")
RED_CARVEOUT = int(os.environ.get("TLLM_K3_FGS_MLA_RED_CARVEOUT", "-1"))


def _red_rows(batch: int) -> int:
    if RED_ROWS == "auto":
        return 1 if batch < 8 else 2 if batch < 16 else 4
    return int(RED_ROWS)


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
    if not edges.chain():
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
    ):
        self.leases = leases
        self.decode = decode
        self.weights = weights
        self.prepared = prepared

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
    ``kv_lens`` the int32 KV lengths including this step's token; ``owner`` keeps the layer's persistent buffers.
    Call it right before the pre-attention mix, which then writes ``chain.slot(X_NORM)``; no PDL launch that
    triggers early may come in between.
    """
    if not edges.chain():
        return None
    rows = page_rows.shape[0]
    if not MIN_ROWS <= rows <= MAX_ROWS:
        return None
    decode = _Decode.build(owner, kv_pool, page_rows, kv_lens, max_requests, weights)
    if decode is None:
        return None
    leases = {}
    for name, readers in _EDGES:
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
    return Chain(leases, decode, weights, prepared)


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

    qkv_a = chain.slot(QKV_A)
    kimi_k3_cute_gemm.linear(x, w.fused_a, qkv_a)
    g = chain.slot(GATE)
    kimi_k3_cute_gemm.linear(x, w.g_proj, g)

    q_a_n = chain.slot(Q_A_NORM)
    k.rmsnorm(qkv_a[:, :q_lora], w.q_a_norm, q_a_n, w.q_a_eps, fgs_consumer=True, fgs_producer=True)
    edges.consumed(leases[QKV_A])
    kv_c = chain.slot(KV_C)
    k.rmsnorm(
        qkv_a[:, q_lora : q_lora + kv_lora],
        w.kv_a_norm,
        kv_c,
        w.kv_a_eps,
        fgs_consumer=True,
        fgs_producer=True,
    )
    edges.consumed(leases[QKV_A])

    q = chain.slot(Q).view(rows, heads, -1)
    kimi_k3_cute_gemm.linear(q_a_n, w.q_b, q.view(rows, -1))
    fused_q = chain.slot(FUSED_Q).view(rows, heads, -1)
    kimi_k3_cute_gemm.bmm(
        q[..., :nope].transpose(0, 1), w.k_b, fused_q[..., :kv_lora].transpose(0, 1)
    )
    d = chain.decode
    torch.cuda.current_stream().wait_event(chain.prepared)
    k.latent_append(
        q[..., nope:],
        kv_c,
        qkv_a[:, q_lora + kv_lora :],
        fused_q,
        d.kv_pool,
        d.page_rows,
        d.kv_lens,
        fgs_consumer=True,
        fgs_producer=True,
    )
    for name in (Q, KV_C, QKV_A):
        edges.consumed(leases[name])

    latent = chain.slot(ATTN_LATENT)
    d.run(fused_q, latent)
    edges.consumed(leases[FUSED_Q])
    attn_out = chain.slot(ATTN_OUT)
    kimi_k3_cute_gemm.bmm(
        latent.view(rows, heads, kv_lora).transpose(0, 1),
        w.v_b,
        attn_out.view(rows, heads, v_dim).transpose(0, 1),
    )

    gated = chain.slot(GATED)
    k.gate(attn_out, g, out=gated, fgs_consumer=True, fgs_producer=True)
    edges.consumed(leases[ATTN_OUT])
    edges.consumed(leases[GATE])
    o = chain.slot(ATTN_O)
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


_SHARED: Dict[tuple, torch.Tensor] = {}


def _shared(name: str, numel: int, dtype: torch.dtype, device) -> torch.Tensor:
    """A persistent buffer of at least ``numel`` elements shared by every MLA layer (one per name and device)."""
    key = (name, dtype, torch.device(device))
    buf = _SHARED.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=dtype, device=device)
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

        from ...custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner as Runner

        self.owner, self.kv_pool, self.page_rows, self.kv_lens = owner, kv_pool, page_rows, kv_lens
        self.heads, self.kv_lora, self.softmax_scale = heads, kv_lora, softmax_scale
        self.page_size = kv_pool.shape[2]
        batch = page_rows.shape[0]
        self.batch = batch
        self.c_latent, self.c_rope = _decode_pool_views(kv_pool, kv_lora, rope)

        runner = Runner(cutlass.BFloat16, heads, 1, self.page_size, max_requests)
        self.runner = runner
        max_batch = max(batch, max_requests)
        lse_off, lse_size, split_off, split_size, total = runner.get_workspace_layout(
            heads, 1, kv_lora, max_batch, cutlass.Float32
        )
        ws = _shared("mla_decode_workspace", total, torch.uint8, kv_pool.device)
        lse = ws[lse_off : lse_off + lse_size].view(torch.float32).view(max_batch, 1, heads)
        self.lse = lse[:batch].permute(2, 1, 0)
        self.split_ws = ws[split_off : split_off + split_size]
        split_kv = runner.default_tactic(batch)[2]
        q_tiles = runner.get_num_q_tiles(heads, 1)
        split_rows = split_size // (runner._MMA_TILE_M * (kv_lora + 1) * 4)
        fitting = split_rows // (batch * q_tiles)
        self.use_workspace = split_kv > 1 and fitting >= 1
        self.split_kv = max(1, min(split_kv, fitting))
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
        """Launch the layer's step inputs: page table, split plan (once per forward), current-row and LSE sentinels."""
        from ...attention.kernels.rubin.kimi_k3_mla import mla_decode_fgs
        from ...attention.kernels.rubin.kimi_k3_mla.mla_latent_append import mla_decode_page_table

        self.page_table = mla_decode_page_table(self.kv_pool, self.page_rows, self.page_table_buf)
        if self.use_workspace:
            self.plan = _split_plan(self.runner, self.kv_lens, self.plan_buf, self.heads)
        mla_decode_fgs.fill_current_rows_sentinel(self.c_latent, self.page_table, self.kv_lens, 1)
        if self.use_workspace:
            mla_decode_fgs.fill_lse_sentinel(
                self.split_ws, self.heads, self.q_tiles, self.kv_lens, self.split_kv, self.plan
            )

    def run(self, fused_q: torch.Tensor, out: torch.Tensor) -> None:
        """Split-KV kernel and reduction from the ``fused_q`` slot [B, H, 576] into the ``out`` slot [B, H * 512]."""
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
        if plan is not None:
            edges.debug("mla.reduction", batch, f"{red_rows} head rows per CTA")
        if SKIP_EMPTY_SPLITS:
            edges.debug("mla.split_kv", batch, "split clusters without K tiles exit at entry")
        key = (
            self.page_size,
            heads,
            self.use_workspace,
            plan is not None,
            red_rows,
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
                cutlass.Float32(self.softmax_scale), cutlass.Float32(1.0), None, stream,
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
           self.kv_lens, None, None, plan, self.softmax_scale, 1.0, None, stream)  # fmt: skip


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
