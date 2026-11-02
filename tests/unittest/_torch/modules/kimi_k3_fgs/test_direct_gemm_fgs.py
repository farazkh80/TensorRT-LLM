# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FGS builds of the direct (register-prefetch) BF16 GEMM at the Kimi K3 ``fi_direct`` decode rows on SM107.

Ready inputs: every build equals FlashInfer's ``run_direct_dense`` bitwise (producer builds up to -0.0). Hazards: the
activation starts as the -0.0 sentinel and a writer kernel on another stream publishes it block by block, late and
out of order; the consumer must reload (counted) and still match bitwise. Live pair: a producer build writes a slot
that a consumer build reads, PDL-chained on one stream.
"""

import dataclasses
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

if os.environ.get("FGS_STANDALONE") == "1":
    import direct_gemm_fgs as fgs
else:
    from tensorrt_llm._torch.modules.kimi_k3_fgs import direct_gemm_fgs as fgs

NEG0 = -32768

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the producer roles use cvt.rn.pzo (SM107)",
)

# name -> (M, N, K, tactic): the kimi_k3_cute_gemm fi_direct rows (bfa, o_proj, g_proj, merged qkvg).
CASES = {
    "bfa1": (1, 224, 7168, (128, 1, 1)),
    "bfa2": (2, 224, 7168, (128, 1, 2)),
    "bfa4": (4, 224, 7168, (128, 1, 4)),
    "oproj1": (1, 7168, 12288, (256, 2, 1)),
    "gproj1": (1, 12288, 7168, (128, 2, 1)),
    "qkvg1": (1, 49152, 7168, (128, 1, 1)),
    "qkvg2": (2, 49152, 7168, (64, 4, 2)),
}
DEFAULT = dict(check_valid=False, pdl_wait_activation=True)
# Every element checked unless the name says elem0 (element 0 of each 16 B chunk, the TMA report_valid rule).
CONSUMERS = {
    "rows": dict(check_valid=True, check_mode="rows"),
    "row": dict(check_valid=True, check_mode="row"),
    "heads": dict(check_valid=True, check_mode="heads"),
    "heads_before_b": dict(check_valid=True, check_mode="heads_before_b"),
    "rows_wait": dict(check_valid=True, check_mode="rows", pdl_wait_activation=True),
    "rows_uncapped": dict(check_valid=True, check_mode="rows", min_ctas_per_sm=1),
    "rows_elem0": dict(check_valid=True, check_mode="rows", check_elements=False),
    "heads_elem0": dict(check_valid=True, check_mode="heads", check_elements=False),
    "row_relaxed_elem0": dict(
        check_valid=True, check_mode="row", relaxed_first=True, check_elements=False
    ),
}
PRODUCERS = {
    "pzo": dict(DEFAULT, pzo=True),
    "pzo_early": dict(DEFAULT, pzo=True, early_trigger=True),
    "group8": dict(DEFAULT, pzo=True, early_trigger=True, group8=True),
    "head_last": dict(DEFAULT, pzo=True, early_trigger=True, head_last=True),
    "consumer_pzo": dict(check_valid=True, pzo=True, early_trigger=True),
    "consumer_group8": dict(check_valid=True, pzo=True, early_trigger=True, group8=True),
}
# Reload rounds after which a wait gives up and is counted (retry_counter[1]) instead of hanging the test.
SPIN_LIMIT = 200_000


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def inputs(name: str, seed: int = 0):
    m, n, k, tactic = CASES[name]
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
    x[x == 0] = 1.0
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    return x, w, fgs.DirectTactic(*tactic)


def flashinfer(x, w, tactic, out=None):
    from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic, run_direct_dense

    out = (
        torch.empty(x.shape[0], w.shape[0], device="cuda", dtype=torch.bfloat16)
        if out is None
        else out
    )
    run_direct_dense(x, w.t(), out, True, DirectTactic(*dataclasses.astuple(tactic)))
    return out


def run(x, w, tactic, flags, counter=None, buf=None, fill=float("nan")):
    """Output buffer with one spare row past M (must stay NaN)."""
    m = x.shape[0]
    if buf is None:
        buf = torch.full((m + 1, w.shape[0]), fill, device="cuda", dtype=torch.bfloat16)
        buf[m] = float("nan")
    fgs.run_direct_dense_fgs(x, w.t(), buf[:m], True, tactic, retry_counter=counter, **flags)
    return buf


@pytest.mark.parametrize("name", list(CASES))
def test_default_build_matches_flashinfer(name: str) -> None:
    x, w, tactic = inputs(name)
    ref = flashinfer(x, w, tactic)
    got = run(x, w, tactic, DEFAULT)
    torch.cuda.synchronize()
    f = ref.float()
    assert bool(torch.isfinite(f).all()) and f.abs().max().item() > 0
    assert torch.equal(bits(got[:-1]), bits(ref))
    assert bool(torch.isnan(got[-1].float()).all())


@pytest.mark.parametrize("name,variant", [(n, v) for n in CASES for v in CONSUMERS])
def test_ready_consumer_matches_flashinfer(name: str, variant: str) -> None:
    x, w, tactic = inputs(name, seed=1)
    ref = flashinfer(x, w, tactic)
    counter = torch.zeros(fgs.RETRY_COUNTER_SLOTS, dtype=torch.int32, device="cuda")
    got = run(x, w, tactic, CONSUMERS[variant], counter)
    torch.cuda.synchronize()
    assert torch.equal(bits(got[:-1]), bits(ref)), f"{name}/{variant} differs from FlashInfer"
    assert counter.tolist() == [0, 0]


def _negative_zero_inputs(name: str):
    """Rows 0 .. M/2 hold one min-denormal activation; times -0.5 it rounds to bf16 -0.0 without .pzo."""
    x, w, tactic = inputs(name, seed=2)
    m = x.shape[0]
    tiny = torch.tensor(2.0**-133, dtype=torch.bfloat16, device="cuda")
    x[: max(1, m // 2)] = 0.0
    x[: max(1, m // 2), 0] = tiny
    w[:16, 0] = -0.5
    return x, w, tactic


@pytest.mark.parametrize("armed", [False, True])
@pytest.mark.parametrize("name,variant", [(n, v) for n in CASES for v in PRODUCERS])
def test_producer_build_changes_only_negative_zero(name: str, variant: str, armed: bool) -> None:
    """``armed``: the output starts as -0.0, so head_last chunk heads wait for the rest of their chunk."""
    x, w, tactic = _negative_zero_inputs(name)
    ref = flashinfer(x, w, tactic)
    counter = torch.zeros(fgs.RETRY_COUNTER_SLOTS, dtype=torch.int32, device="cuda")
    got = run(
        x,
        w,
        tactic,
        dict(PRODUCERS[variant], spin_limit=SPIN_LIMIT),
        counter,
        fill=-0.0 if armed else float("nan"),
    )
    torch.cuda.synchronize()
    assert counter[1].item() == 0, f"{counter[1].item()} chunk-head waits gave up"
    assert bool((bits(ref) == NEG0).any()), "the inputs no longer produce -0.0 in the default build"
    canon = bits(ref).clone()
    canon[canon == NEG0] = 0
    assert torch.equal(bits(got[:-1]), canon)
    assert bool(torch.isnan(got[-1].float()).all())


@dsl_user_op
def _ld_global_v4(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ld_relaxed_u32(addr: Int64, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _st_global_v4(addr: Int64, v, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [addr.ir_value(loc=loc, ip=ip)] + [x.ir_value(loc=loc, ip=ip) for x in v],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _st_global_v2(addr: Int64, w0, w1, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [addr.ir_value(loc=loc, ip=ip), w0.ir_value(loc=loc, ip=ip), w1.ir_value(loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# The writer publishes anyway this long after it started if the flag never comes.
FLAG_TIMEOUT_NS = 1_000_000_000


class ColumnBlockWriter:
    """Test producer: copies ``src`` -> ``dst`` ([M, K], K contiguous) one block of ``block_cols`` columns (all
    rows) per CTA, 16 B per store. Block p is published ``delay + ranks[p] * step`` after ``flag`` turns non-zero
    (set on the consumer's stream right before the consumer). It never writes sentinels."""

    def __init__(self, block_cols: int, threads: int = 128, split: bool = False):
        """``split``: each chunk is published as two 8 B stores, elements 0-3 first and 4-7 ``split_ns`` later."""
        assert block_cols % 8 == 0
        self.block_cols, self.threads, self.split = block_cols, threads, split

    @cute.jit
    def __call__(
        self,
        src,
        dst,
        ranks,
        flag,
        delay_ns: Int64,
        step_ns: Int64,
        split_ns: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(src, dst, ranks, flag, delay_ns, step_ns, split_ns).launch(
            grid=(cute.size(ranks), 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, src, dst, ranks, flag, delay_ns: Int64, step_ns: Int64, split_ns: Int64):
        p, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        give_up = cute.arch.globaltimer() + Int64(FLAG_TIMEOUT_NS)
        while (_ld_relaxed_u32(flag.iterator.toint()) == 0) & (cute.arch.globaltimer() < give_up):
            cute.arch.nanosleep(sleep_time=200)
        target = cute.arch.globaltimer() + delay_ns + Int64(ranks[p]) * step_ns
        while cute.arch.globaltimer() < target:
            cute.arch.nanosleep(sleep_time=200)
        chunks = self.block_cols // 8
        for half in cutlass.range_constexpr(2 if self.split else 1):
            if cutlass.const_expr(half == 1):
                cute.arch.fence_acq_rel_gpu()
                target = cute.arch.globaltimer() + split_ns
                while cute.arch.globaltimer() < target:
                    cute.arch.nanosleep(sleep_time=200)
            for c in cutlass.range(tid, src.shape[0] * chunks, self.threads, unroll=1):
                r = c // chunks
                col = p * self.block_cols + (c - r * chunks) * 8
                src_off = Int64(r) * Int64(src.stride[0]) + Int64(col)
                dst_off = Int64(r) * Int64(dst.stride[0]) + Int64(col)
                v = _ld_global_v4(src.iterator.toint() + src_off * 2)
                if cutlass.const_expr(self.split):
                    _st_global_v2(
                        dst.iterator.toint() + (dst_off + 4 * half) * 2,
                        v[2 * half],
                        v[2 * half + 1],
                    )
                else:
                    _st_global_v4(dst.iterator.toint() + dst_off * 2, v)
        cute.arch.fence_acq_rel_gpu()


@cute.kernel
def _gpu_sleep_kernel(ns: Int64):
    end = cute.arch.globaltimer() + ns
    while cute.arch.globaltimer() < end:
        cute.arch.nanosleep(sleep_time=1000)


@cute.jit
def _gpu_sleep_launch(ns: Int64, stream: cuda.CUstream):
    _gpu_sleep_kernel(ns).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


_SLEEP = []


def gpu_sleep(ns: int) -> None:
    """Hold the current stream for ``ns`` on the GPU."""
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if not _SLEEP:
        _SLEEP.append(cute.compile(_gpu_sleep_launch, Int64(0), stream))
    _SLEEP[0](Int64(ns), stream)


_WRITERS = {}
BLOCK_COLS = 128


def late_write(src, dst, ranks, delay_us: float, step_us: float, flag=None, split_us: float = 0.0):
    if flag is None:
        flag = torch.ones(1, dtype=torch.int32, device="cuda")
    args = [from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1) for t in (src, dst)]
    ranks_t = from_dlpack(ranks, assumed_align=4).mark_layout_dynamic()
    flag_t = from_dlpack(flag, assumed_align=4)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    split = split_us > 0
    if split not in _WRITERS:
        _WRITERS[split] = cute.compile(
            ColumnBlockWriter(BLOCK_COLS, split=split),
            *args,
            ranks_t,
            flag_t,
            Int64(0),
            Int64(0),
            Int64(0),
            stream,
        )
    _WRITERS[split](
        *args,
        ranks_t,
        flag_t,
        Int64(int(delay_us * 1e3)),
        Int64(int(step_us * 1e3)),
        Int64(int(split_us * 1e3)),
        stream,
    )


def ranks_for(pattern: str, n: int, seed: int = 0) -> torch.Tensor:
    if pattern == "forward":
        r = torch.arange(n)
    elif pattern == "reverse":
        r = torch.arange(n - 1, -1, -1)
    elif pattern == "random":
        r = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    elif pattern == "one_late":
        r = torch.zeros(n, dtype=torch.int64)
        r[n // 2] = 50
    else:
        raise ValueError(pattern)
    return r.to(torch.int32).cuda()


def run_hazard(
    name: str,
    pattern: str,
    variant: str,
    delay_us: float = 300.0,
    step_us: float = 2.0,
    split_us: float = 0.0,
):
    """Writer on its own stream first, consumer second; returns (output, FlashInfer reference, reloads, waits given
    up, published).

    The consumer's stream sleeps on the GPU before it sets the writer's flag, so the consumer is already queued
    when the writer's clock starts, however slow the host is."""
    x, w, tactic = inputs(name, seed=3)
    ref = flashinfer(x, w, tactic)
    ranks = ranks_for(pattern, x.shape[1] // BLOCK_COLS)
    warm = x.clone()
    late_write(x, warm, ranks, 0.0, 0.0, split_us=split_us)
    gpu_sleep(0)
    run(warm, w, tactic, dict(CONSUMERS[variant], spin_limit=SPIN_LIMIT))
    torch.cuda.synchronize()

    slot = torch.full_like(x, -0.0)
    counter = torch.zeros(fgs.RETRY_COUNTER_SLOTS, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    got = torch.full(
        (x.shape[0] + 1, w.shape[0]), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    sp, sc = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(sp):
        late_write(x, slot, ranks, delay_us, step_us, flag, split_us=split_us)
    with torch.cuda.stream(sc):
        gpu_sleep(HOST_SLACK_NS)
        flag.fill_(1)
        run(slot, w, tactic, dict(CONSUMERS[variant], spin_limit=SPIN_LIMIT), counter, buf=got)
    torch.cuda.synchronize()
    reloads, gave_up = counter.tolist()
    return got[:-1], ref, reloads, gave_up, bool(torch.equal(bits(slot), bits(x)))


def stale_blocks(name: str, got: torch.Tensor) -> str:
    """Which single writer block, read as the sentinel, reproduces ``got`` (a sentinel times w adds +-0 to the FMA
    chain, so it equals FlashInfer with that block zeroed)."""
    x, w, tactic = inputs(name, seed=3)
    hits = []
    for b in range(x.shape[1] // BLOCK_COLS):
        xs = x.clone()
        xs[:, b * BLOCK_COLS : (b + 1) * BLOCK_COLS] = 0.0
        if torch.equal(bits(flashinfer(xs, w, tactic)), bits(got)):
            hits.append(b)
    return f"stale block(s) {hits}" if hits else "no single stale block reproduces it"


def check_hazard(name: str, pattern: str, variant: str, split_us: float = 0.0) -> None:
    got, ref, reloads, gave_up, published = run_hazard(name, pattern, variant, split_us=split_us)
    assert published, f"{name}/{pattern}: the writer did not publish the activation"
    assert gave_up == 0, (
        f"{name}/{pattern}/{variant}: {gave_up} warps gave up waiting ({reloads} reloads)"
    )
    diff = int((bits(got) != bits(ref)).sum())
    if diff:
        why = stale_blocks(name, got) if CASES[name][0] == 1 else "M > 1, not analysed"
        raise AssertionError(
            f"{name}/{pattern}/{variant}: {diff} output elements differ ({reloads} reloads; {why})"
        )
    assert reloads > 0, (
        f"{name}/{pattern}/{variant}: the writer finished before the consumer read (no reload)"
    )
    print(f"{name}/{pattern}/{variant}: {reloads} reload rounds")


HAZARD_CASES = ["bfa1", "bfa4", "oproj1", "gproj1", "qkvg1", "qkvg2"]
HOST_SLACK_NS = 2_000_000


@pytest.mark.parametrize("pattern", ["forward", "reverse", "random", "one_late"])
@pytest.mark.parametrize("name", HAZARD_CASES)
def test_late_out_of_order_producer(name: str, pattern: str) -> None:
    check_hazard(
        name, pattern, fgs.default_check_mode(fgs.DirectTactic(*CASES[name][3]), CASES[name][2])
    )


@pytest.mark.parametrize("variant", ["row", "heads", "heads_before_b", "rows_elem0", "heads_elem0"])
@pytest.mark.parametrize("name", ["bfa1", "bfa2", "qkvg1"])
def test_late_producer_other_check_modes(name: str, variant: str) -> None:
    for pattern in ("reverse", "random"):
        check_hazard(name, pattern, variant)


@pytest.mark.parametrize("variant", ["rows", "row", "heads", "heads_before_b"])
@pytest.mark.parametrize("name", ["bfa1", "bfa4", "oproj1", "qkvg1", "qkvg2"])
def test_partially_written_chunks(name: str, variant: str) -> None:
    """Chunks published as two 8 B halves 50 us apart (element 0 first): an element-checking consumer waits for the
    second half."""
    check_hazard(name, "random", variant, split_us=50.0)


@pytest.mark.parametrize("store", ["pzo", "group8", "head_last"])
@pytest.mark.parametrize(
    "producer,consumer", [("oproj1", "qkvg1"), ("oproj1", "gproj1"), ("oproj1", "bfa1")]
)
def test_live_pair_into_consumer(producer: str, consumer: str, store: str) -> None:
    """o_proj-shaped producer build (``store``, .pzo, entry trigger) writes a -0.0 slot that an element-checking
    consumer build reads, PDL-chained on one stream, three steps with new producer inputs and the slot re-armed in
    between (the model's step reuse): every step equals FlashInfer on the canonicalized intermediate, so no L1 line
    of an earlier step's slot contents is consumed. ``pzo`` is the direct-to-direct pair (2 B stores)."""
    _, w1, t1 = inputs(producer)
    _, w2, t2 = inputs(consumer, seed=5)
    assert w1.shape[0] == w2.shape[1]
    counter = torch.zeros(fgs.RETRY_COUNTER_SLOTS, dtype=torch.int32, device="cuda")
    slot = torch.empty(CASES[producer][0], w1.shape[0], device="cuda", dtype=torch.bfloat16)
    out = torch.empty(slot.shape[0], w2.shape[0], device="cuda", dtype=torch.bfloat16)
    mode = fgs.default_check_mode(t2, w2.shape[1])
    for step in range(4):  # step 0 compiles both builds
        x = inputs(producer, seed=10 + step)[0]
        mid_ref = flashinfer(x, w1, t1)
        bits(mid_ref)[bits(mid_ref) == NEG0] = 0
        ref = flashinfer(mid_ref, w2, t2)
        bits(slot).fill_(NEG0)
        counter.zero_()
        fgs.run_direct_dense_fgs(
            x,
            w1.t(),
            slot,
            True,
            t1,
            retry_counter=counter,
            spin_limit=SPIN_LIMIT,
            **PRODUCERS["pzo_early" if store == "pzo" else store],
        )
        fgs.run_direct_dense_fgs(
            slot,
            w2.t(),
            out,
            True,
            t2,
            retry_counter=counter,
            check_mode=mode,
            spin_limit=SPIN_LIMIT,
        )
        torch.cuda.synchronize()
        assert counter[1].item() == 0, (
            f"{producer}->{consumer} step {step}: {counter[1].item()} waits gave up"
        )
        assert torch.equal(bits(slot), bits(mid_ref)), (
            f"{producer}->{consumer} step {step}: producer output"
        )
        assert torch.equal(bits(out), bits(ref)), (
            f"{producer}->{consumer} step {step}: {counter[0].item()} reloads"
        )
        print(f"{producer}->{consumer} step {step}: {counter[0].item()} reload rounds")
