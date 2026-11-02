# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-level fine-grained-sync oracle for one Kimi K3 KDA prefill layer.

Times one ``KimiKDALinearAttention`` prefill forward (projections through
``o_proj``) under CUDA graph replay in these schedules:

- ``off``: the stock single-stream schedule.
- ``standalone``: stages on lane streams, every edge waits for its producer.
  Only independent branches overlap.
- ``oracle``: every producer -> consumer edge is ``ORACLE``; consumers launch
  without waiting and read the previous replay's (identical) buffers. This is
  the best-case overlap a fine-grained-sync consumer could reach with free validity
  checks and free buffer re-arming.
- ``oracle-safe``: ``oracle`` except the two edges around the convolution,
  which rewrites the projection buffer in place; its output can be checked
  against ``off``.
- ``fgs-poll-ref``: the gated norm replaced by a Triton kernel; every edge waits.
- ``fgs-poll``: that kernel polls the chunk output for a sentinel instead of
  waiting, launched once every chunk-producer CTA is resident.
- ``fgs-poll-unordered``: as ``fgs-poll`` without the launch ordering; with a
  consumer CTA per SM it can starve the persistent producer and hang
  (``FGS_POLL_CONSUMER_CTAS`` caps the grid).

Example::

    python tests/microbenchmarks/kda_fine_grained_sync_oracle.py --seq-lens 1024,4096 --lanes 2,4
"""

import argparse
import os
import statistics
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fine_grained_sync import (
    EdgePolicy,
    FineGrainedSyncConfig,
    configure_fine_grained_sync,
)
from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention

HIDDEN_SIZE = 7168
HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
# causal_conv1d rewrites the channel-major projection buffer in place, so a
# reader of that buffer that does not wait sees a mix of raw and convolved values.
IN_PLACE_EDGES = ("kda.qkvg_proj->kda.conv1d", "kda.conv1d->kda.chunk")


# 0xFFFF is a NaN bit pattern that the chunk kernel's finite bf16 output never takes.
FGS_SENTINEL = -1
FGS_POLL_EDGE = "kda.chunk->kda.output_gate"


@triton.jit
def _fgs_poll_rms_norm_gated_kernel(
    o_ptr,
    z_ptr,
    w_ptr,
    y_ptr,
    n_rows,
    heads,
    z_token_stride,
    eps,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    POLL: tl.constexpr,
):
    cols = tl.arange(0, D)[None, :]
    w = tl.load(w_ptr + cols).to(tl.float32)
    for block in range(tl.program_id(0), tl.cdiv(n_rows, ROWS), tl.num_programs(0)):
        rows = block * ROWS + tl.arange(0, ROWS)[:, None]
        mask = (rows < n_rows) & (cols < D)
        bits = tl.load(o_ptr + rows * D + cols, mask=mask, other=0, volatile=True)
        if POLL:
            while tl.sum(tl.sum(((bits == -1) & mask).to(tl.int32), axis=1), axis=0) > 0:
                bits = tl.load(o_ptr + rows * D + cols, mask=mask, other=0, volatile=True)
        x = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1)[:, None] / D + eps)
        z_offs = (rows // heads) * z_token_stride + (rows % heads) * D + cols
        z = tl.load(z_ptr + z_offs, mask=mask, other=0).to(tl.float32)
        y = x * rstd * w * tl.sigmoid(z)
        tl.store(y_ptr + rows * D + cols, y.to(tl.bfloat16), mask=mask)


def install_polling_output_gate(attention, poll: bool, num_ctas: int) -> dict:
    """Replace the gated norm with a consumer that polls the chunk output for the sentinel.

    With ``poll`` the consumer spins on each row until the producer has overwritten
    every sentinel. The caller re-arms ``state["buffer"]`` once the whole forward
    (and so the producer) has finished; re-arming right after the consumer would let
    the same replay's producer overwrite the sentinel. While ``dry`` is set (eager
    warmup) the consumer reads a zero copy so it can never spin.
    """
    state = {"dry": False, "buffer": None}

    def output_gate(x, o, onorm_g):
        rows = o.reshape(-1, attention.head_dim)
        z = onorm_g.reshape(-1, attention.num_heads, attention.head_dim)
        y = torch.empty_like(rows)
        bits = rows.view(torch.int16)
        if state["dry"]:
            bits = torch.zeros_like(bits)
        row_block = 8
        grid = min(num_ctas, triton.cdiv(rows.shape[0], row_block))
        _fgs_poll_rms_norm_gated_kernel[(grid,)](
            bits,
            z,
            attention.o_norm.weight,
            y,
            rows.shape[0],
            attention.num_heads,
            z.stride(0),
            attention.o_norm.eps,
            D=attention.head_dim,
            ROWS=row_block,
            POLL=poll,
        )
        if poll and not state["dry"]:
            state["buffer"] = bits
        return y

    attention._output_gate = output_gate
    return state


def _kernel_node_name(node) -> str:
    from cuda.bindings import driver

    err, params = driver.cuGraphKernelNodeGetParams(node)
    if err != driver.CUresult.CUDA_SUCCESS:
        return ""
    if int(params.func):
        err, name = driver.cuFuncGetName(params.func)
    else:
        err, name = driver.cuKernelGetName(params.kern)
    return name.decode() if err == driver.CUresult.CUDA_SUCCESS else ""


def order_consumer_after_producer_launch(graph, producer: str, consumer: str) -> None:
    """Add a launch-completion edge so the consumer launches once every producer CTA is resident.

    A polling consumer launched earlier can fill the SMs a persistent producer still
    needs and spin forever; this is the ordering fine-grained-sync early launch relies on.
    """
    from cuda.bindings import driver

    raw = driver.CUgraph(graph.raw_cuda_graph())
    err, _, count = driver.cuGraphGetNodes(raw, 0)
    err, nodes, count = driver.cuGraphGetNodes(raw, count)
    kernels = []
    for node in nodes[:count]:
        err, kind = driver.cuGraphNodeGetType(node)
        if kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            kernels.append((node, _kernel_node_name(node)))
    sources = [node for node, name in kernels if producer in name]
    targets = [node for node, name in kernels if consumer in name]
    assert len(sources) == 1 and len(targets) == 1, (
        f"expected one {producer} and one {consumer} node, got {len(sources)}, {len(targets)}"
    )
    edge = driver.CUgraphEdgeData()
    edge.from_port = driver.CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER
    edge.type = driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
    (err,) = driver.cuGraphAddDependencies(raw, sources, targets, [edge], 1)
    assert err == driver.CUresult.CUDA_SUCCESS, f"cuGraphAddDependencies failed: {err}"


class _LayerCache:
    def __init__(self, conv: torch.Tensor, temporal: torch.Tensor) -> None:
        self.conv = conv
        self.temporal = temporal
        self.kda_k_cache = None
        self.kda_g_cache = None
        self.has_kda_replay_caches = False


class _KvCacheManager:
    def __init__(self, layer_cache: _LayerCache) -> None:
        self._layer_cache = layer_cache

    def mamba_layer_cache(self, _layer_idx: int) -> _LayerCache:
        return self._layer_cache


def make_attention(num_heads: int) -> KimiKDALinearAttention:
    cfg = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        rms_norm_eps=1e-5,
        linear_attn_config={
            "num_heads": num_heads,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": CONV_KERNEL_SIZE,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    attention = KimiKDALinearAttention(cfg, layer_idx=0).to("cuda")
    with torch.no_grad():
        for name, param in attention.named_parameters():
            if name.endswith("weight") and param.ndim == 2:
                param.normal_(std=0.02)
        attention.dt_bias.zero_()
    attention.finalize_decode_weights()
    print(f"KDA prefill kernel path: {attention._dispatch.prefill_kernel_path}")
    return attention


def make_metadata(seq_lens: list[int], num_heads: int) -> SimpleNamespace:
    device = "cuda"
    batch = len(seq_lens)
    num_tokens = sum(seq_lens)
    starts = [0]
    for length in seq_lens:
        starts.append(starts[-1] + length)
    cu_seqlens = torch.tensor(starts, dtype=torch.long, device=device)
    order = sorted(range(batch), key=lambda i: (-seq_lens[i], i))
    cu_chunks = [0]
    for length in seq_lens:
        cu_chunks.append(cu_chunks[-1] + (length + 15) // 16)
    mamba = SimpleNamespace(
        state_indices=torch.arange(batch, dtype=torch.int32, device=device),
        query_start_loc_long=cu_seqlens,
        query_start_loc=cu_seqlens.to(torch.int32),
        has_initial_states=torch.zeros(batch, dtype=torch.bool, device=device),
        use_initial_states=False,
        kda_max_seqlen=max(seq_lens),
        kda_seq_order=torch.tensor(order, dtype=torch.int32, device=device),
        kda_cu_chunks=torch.tensor(cu_chunks, dtype=torch.int32, device=device),
        _arange_buffer=torch.arange(batch + 1, dtype=torch.int32, device=device),
    )
    projection_size = num_heads * HEAD_DIM
    layer_cache = _LayerCache(
        conv=torch.zeros(
            batch, 3 * projection_size, CONV_KERNEL_SIZE - 1, dtype=torch.bfloat16, device=device
        ),
        temporal=torch.zeros(
            batch, num_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
        ),
    )
    return SimpleNamespace(
        mamba_metadata=mamba,
        num_contexts=batch,
        num_ctx_tokens=num_tokens,
        num_tokens=num_tokens,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        kv_cache_manager=_KvCacheManager(layer_cache),
    )


def mode_config(mode: str, lanes: int):
    if mode == "off":
        return None
    if mode == "standalone":
        return FineGrainedSyncConfig(EdgePolicy.STANDALONE, {}, lanes)
    if mode == "oracle":
        return FineGrainedSyncConfig(EdgePolicy.ORACLE, {}, lanes)
    if mode in ("fgs-poll", "fgs-poll-unordered", "fgs-poll-ref"):
        policy = EdgePolicy.STANDALONE if mode == "fgs-poll-ref" else EdgePolicy.ORACLE
        return FineGrainedSyncConfig(EdgePolicy.STANDALONE, {FGS_POLL_EDGE: policy}, lanes)
    if mode == "oracle-safe":
        return FineGrainedSyncConfig(
            EdgePolicy.ORACLE, {edge: EdgePolicy.STANDALONE for edge in IN_PLACE_EDGES}, lanes
        )
    raise ValueError(f"unknown mode {mode!r}")


class _Eager:
    """Graph-like wrapper that re-runs the forward eagerly under one config."""

    def __init__(self, attention, hidden, metadata, config) -> None:
        self._args = (attention, hidden, metadata, config)

    @torch.no_grad()
    def replay(self) -> torch.Tensor:
        attention, hidden, metadata, config = self._args
        configure_fine_grained_sync(config)
        try:
            return attention(hidden, metadata)
        finally:
            configure_fine_grained_sync(None)


@torch.no_grad()
def capture(attention, hidden, metadata, config, warmup: int, use_graph: bool = True, mode=None):
    poll_state = None
    if mode in ("fgs-poll", "fgs-poll-unordered", "fgs-poll-ref"):
        poll_state = install_polling_output_gate(
            attention,
            poll=mode != "fgs-poll-ref",
            num_ctas=int(
                os.environ.get(
                    "FGS_POLL_CONSUMER_CTAS",
                    torch.cuda.get_device_properties(0).multi_processor_count,
                )
            ),
        )
        poll_state["dry"] = True
    else:
        attention.__dict__.pop("_output_gate", None)
    if not use_graph:
        eager = _Eager(attention, hidden, metadata, config)
        for _ in range(warmup):
            output = eager.replay()
        torch.cuda.synchronize()
        return eager, output
    configure_fine_grained_sync(config)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            attention(hidden, metadata)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    if poll_state is not None:
        poll_state["dry"] = False
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        output = attention(hidden, metadata)
        if poll_state is not None and poll_state["buffer"] is not None:
            poll_state["buffer"].fill_(FGS_SENTINEL)
    configure_fine_grained_sync(None)
    if mode == "fgs-poll":
        # k4_persistent is the chunk-stage kernel that writes the polled output.
        order_consumer_after_producer_launch(graph, "k4_persistent", "fgs_poll_rms_norm_gated")
    graph.instantiate()
    # Two replays: oracle consumers read the previous replay's buffers.
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    return graph, output


def time_replays(graph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000.0 / replays


def compare(actual: torch.Tensor, expected: torch.Tensor) -> str:
    a, e = actual.float().flatten(), expected.float().flatten()
    if not torch.isfinite(a).all():
        return "non-finite"
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    return f"cos={cosine:.6f} max_abs={(a - e).abs().max().item():.3e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq-lens", default="1024,2048,4096,8192")
    parser.add_argument("--batch", type=int, default=1, help="sequences of each length")
    parser.add_argument("--num-heads", type=int, default=96, help="96 = DEP (full), 12 = TEP8")
    parser.add_argument("--lanes", default="2,4")
    parser.add_argument("--modes", default="off,standalone,oracle,oracle-safe")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--no-graph", action="store_true", help="time eager forwards instead of graph replays"
    )
    parser.add_argument(
        "--profile-mode",
        default=None,
        help="'mode:lanes:seq_len' -- capture once and replay inside cudaProfilerStart/Stop",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    attention = make_attention(args.num_heads)

    if args.profile_mode:
        mode, lanes, seq_len = args.profile_mode.split(":")
        seq_lens = [int(seq_len)] * args.batch
        hidden = torch.randn(sum(seq_lens), HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        metadata = make_metadata(seq_lens, args.num_heads)
        graph, _ = capture(
            attention,
            hidden,
            metadata,
            mode_config(mode, int(lanes)),
            args.warmup,
            not args.no_graph,
            mode,
        )
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    schedules = [("off", 0)] + [
        (mode, lanes)
        for mode in args.modes.split(",")
        if mode != "off"
        for lanes in (int(x) for x in args.lanes.split(","))
    ]
    print(
        f"KDA prefill, hidden={HIDDEN_SIZE}, heads={args.num_heads}x{HEAD_DIM}, batch={args.batch}"
    )
    print(f"{'ISL':>6} {'mode':>12} {'lanes':>5} {'us':>9} {'vs off':>8}  check vs off")
    for seq_len in (int(x) for x in args.seq_lens.split(",")):
        seq_lens = [seq_len] * args.batch
        hidden = torch.randn(sum(seq_lens), HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        metadata = make_metadata(seq_lens, args.num_heads)
        captured = [
            capture(
                attention,
                hidden,
                metadata,
                mode_config(mode, lanes),
                args.warmup,
                not args.no_graph,
                mode,
            )
            for mode, lanes in schedules
        ]
        # Alternate schedules within every sample so clock and thermal drift
        # affect all of them alike.
        times = [[] for _ in schedules]
        for _ in range(args.samples):
            for index, (graph, _) in enumerate(captured):
                times[index].append(time_replays(graph, args.replays))
        medians = [statistics.median(t) for t in times]
        reference = captured[0][1]
        poll_reference = next(
            (out for (m, _), (_, out) in zip(schedules, captured) if m == "fgs-poll-ref"), None
        )
        for (mode, lanes), (_, output), latency in zip(schedules, captured, medians):
            delta_pct = 100.0 * (medians[0] / latency - 1.0)
            check = "" if mode == "off" else compare(output, reference)
            if (
                mode.startswith("fgs-poll")
                and mode != "fgs-poll-ref"
                and poll_reference is not None
            ):
                check += f" | vs fgs-poll-ref {compare(output, poll_reference)}"
            print(
                f"{seq_len:>6} {mode:>12} {lanes or '-':>5} {latency:>9.1f} "
                f"{delta_pct:>+7.1f}%  {check}"
            )
        del captured


if __name__ == "__main__":
    main()
