# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests of the fine-grained-sync edge workspaces (sentinel slots, resets, graphs).

Runs under pytest, or standalone with a PASS/FAIL summary:
``FGS_OVL_DIR=<dir with fine_grained_sync.py> python3 test_fine_grained_sync_edges.py``
loads the module from that directory instead of the installed package.
"""

import importlib
import os
import sys
import traceback
import types

import torch

try:
    import pytest
except ImportError:
    pytest = None

# About 50 ms on a ~2 GHz SM clock; long enough that an unordered reset always wins the race.
SLOW_CYCLES = 100_000_000
_fgs = None


def fgs():
    global _fgs
    if _fgs is None:
        ovl = os.environ.get("FGS_OVL_DIR")
        if ovl:
            package = types.ModuleType("fgs_ovl")
            package.__path__ = [ovl]
            sys.modules["fgs_ovl"] = package
            _fgs = importlib.import_module("fgs_ovl.fine_grained_sync")
        else:
            _fgs = importlib.import_module("tensorrt_llm._torch.modules.fine_grained_sync")
    return _fgs


def _backends():
    return ["auto", "fill", "cute", "template", "memset"]


def _parametrize(name, values):
    if pytest is None:
        return lambda fn: fn
    return pytest.mark.parametrize(name, values)


if pytest is not None:
    pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _lane(dtype):
    return (
        (torch.int16, -0x8000)
        if dtype in (torch.bfloat16, torch.float16)
        else (torch.int32, -(2**31))
    )


def non_sentinel(t: torch.Tensor) -> torch.Tensor:
    lane, value = _lane(t.dtype)
    return (t.reshape(-1).view(lane) != value).sum()


def count_non_sentinel(t: torch.Tensor) -> int:
    return int(non_sentinel(t).item())


def junk_(t: torch.Tensor) -> torch.Tensor:
    lane, _ = _lane(t.dtype)
    t.view(lane).fill_(0x1234)
    return t


def _edges(backend, **kwargs):
    return fgs().EdgeWorkspaces(backend=fgs().ResetBackend(backend), **kwargs)


def test_cute_fill_standalone():
    fill = importlib.import_module(fgs().__name__.rsplit(".", 1)[0] + ".fine_grained_sync_fill")
    word = -0x7FFF8000
    guard = 4096
    for vecs in (1, 3, 1024, 1025, 4096 * 4 + 7, (1 << 20) + 5):
        for ctas in (1, 4, 212):
            buf = torch.full((2 * guard + vecs * 8 + 8,), 0x1234, dtype=torch.int16, device="cuda")
            target = buf[guard : guard + vecs * 8].view(torch.bfloat16)
            fill.sentinel_fill_(target, word, max_ctas=ctas)
            torch.cuda.synchronize()
            assert count_non_sentinel(target) == 0, (vecs, ctas)
            assert bool((buf[:guard] == 0x1234).all()) and bool(
                (buf[guard + vecs * 8 :] == 0x1234).all()
            ), (
                "fill wrote outside its tensor",
                vecs,
                ctas,
            )
    odd = torch.full((15,), 0x1234, dtype=torch.int16, device="cuda").view(torch.bfloat16)
    fill.sentinel_fill_(odd, word)
    assert count_non_sentinel(odd) == 0, "byte tail"


SHAPES = [
    ((16, 224), torch.bfloat16),
    ((16, 49152), torch.bfloat16),
    ((16, 96, 128), torch.float16),
    ((7, 9), torch.bfloat16),
    ((7, 9), torch.float32),
    ((512, 4096), torch.bfloat16),
]


@_parametrize("backend", _backends())
def test_preinit_bitexact(backend):
    edges = _edges(backend)
    for index, (shape, dtype) in enumerate(SHAPES):
        edges.workspace(f"e{index}", shape, dtype)
    torch.cuda.synchronize()
    for index, _ in enumerate(SHAPES):
        for slot in edges[f"e{index}"].slots:
            assert count_non_sentinel(slot) == 0, (backend, SHAPES[index])
            junk_(slot)
    edges.preinitialize()
    torch.cuda.synchronize()
    for index, _ in enumerate(SHAPES):
        for slot in edges[f"e{index}"].slots:
            assert count_non_sentinel(slot) == 0, (backend, SHAPES[index], "after junk")


@_parametrize("backend", _backends())
def test_reset_bitexact_and_row_bound(backend):
    edges = _edges(backend)
    ws = edges.workspace("rows", (64, 7168), torch.bfloat16)
    odd = edges.workspace("odd", (7, 9), torch.bfloat16)
    f32 = edges.workspace("f32", (40, 12), torch.float32)
    for slot in range(2):
        junk_(ws.slots[slot][16:])
    with edges.scope():
        for _ in range(2):
            lease = edges.acquire("rows", rows=16)
            assert lease.tensor.shape == (16, 7168)
            lease.tensor.normal_()
            edges.release(lease)
        for name in ("odd", "f32"):
            for _ in range(2):
                lease = edges.acquire(name)
                lease.tensor.normal_()
                edges.release(lease)
    torch.cuda.synchronize()
    for slot in range(2):
        assert count_non_sentinel(ws.slots[slot][:16]) == 0, backend
        assert count_non_sentinel(ws.slots[slot][16:]) == 48 * 7168, "reset wrote past its rows"
        assert count_non_sentinel(odd.slots[slot]) == 0, backend
        assert count_non_sentinel(f32.slots[slot]) == 0, backend


def _slow_reader_run(backend, reader_joined):
    edges = _edges(backend)
    n = 4 << 20
    edges.workspace("e", (n,), torch.bfloat16)
    src = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    src.view(torch.int16)[src.view(torch.int16) == -0x8000] = 0
    out = torch.empty_like(src)
    reader = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    with edges.scope():
        lease = edges.acquire("e")
        lease.tensor.copy_(src)
        reader.wait_stream(main)
        with torch.cuda.stream(reader):
            torch.cuda._sleep(SLOW_CYCLES)
            out.copy_(lease.tensor)
        edges.release(lease, streams=[reader] if reader_joined else None)
    torch.cuda.synchronize()
    clobbered = count_non_sentinel(src) - count_non_sentinel(out)
    return clobbered, torch.equal(out.view(torch.int16), src.view(torch.int16)), edges


@_parametrize("backend", ["fill", "cute"])
def test_release_waits_for_slow_reader(backend):
    clobbered, intact, edges = _slow_reader_run(backend, reader_joined=True)
    assert intact and clobbered == 0, "reset clobbered a slot its last reader was still reading"
    assert count_non_sentinel(edges["e"].slots[0]) == 0
    # Control: releasing on the producer's stream only must visibly clobber the reader.
    clobbered, intact, _ = _slow_reader_run(backend, reader_joined=False)
    assert not intact and clobbered > 0, "control did not clobber; the test cannot detect a race"


def test_fan_out_readers():
    edges = _edges("fill")
    n = 1 << 20
    edges.workspace("e", (n,), torch.bfloat16)
    src = torch.ones(n, device="cuda", dtype=torch.bfloat16)
    outs = [torch.empty_like(src) for _ in range(2)]
    readers = [torch.cuda.Stream() for _ in range(2)]
    main = torch.cuda.current_stream()
    with edges.scope():
        lease = edges.acquire("e", readers=2)
        lease.tensor.copy_(src)
        for index, (stream, out) in enumerate(zip(readers, outs)):
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                torch.cuda._sleep(SLOW_CYCLES * (2 - index))
                out.copy_(lease.tensor)
        edges.release(lease, streams=[readers[1]])
        assert not lease.released
        edges.release(lease, streams=[readers[0]])
        assert lease.released
    torch.cuda.synchronize()
    for out in outs:
        assert torch.equal(out, src)


def test_alias_views():
    edges = _edges("fill")
    ws = edges.workspace("qkvg", (16, 49152), torch.bfloat16)
    ws.define_alias("qkv", 1, 0, 36864)
    ws.define_alias("gate", 1, 36864, 12288)
    try:
        ws.define_alias("bad", 1, 40000, 12288)
        raise AssertionError("an alias past the slot was accepted")
    except ValueError:
        pass
    readers = [torch.cuda.Stream() for _ in range(2)]
    sums = [torch.empty((), device="cuda") for _ in range(2)]
    main = torch.cuda.current_stream()
    with edges.scope():
        lease = edges.acquire("qkvg", rows=8, readers=2)
        assert lease.view("qkv").shape == (8, 36864) and lease.view("gate").shape == (8, 12288)
        lease.view("qkv").fill_(1.0)
        lease.view("gate").fill_(2.0)
        for stream, alias, total in zip(readers, ("qkv", "gate"), sums):
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                torch.cuda._sleep(SLOW_CYCLES // 10)
                total.copy_(lease.view(alias).float().sum())
            edges.release(lease, streams=[stream])
    torch.cuda.synchronize()
    assert sums[0].item() == 8 * 36864 and sums[1].item() == 2 * 8 * 12288
    assert count_non_sentinel(ws.slots[0]) == 0


def _late_writer_run(mark_written):
    edges = _edges("fill")
    edges.workspace("e", (64, 1024), torch.bfloat16)
    writer = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    head = torch.empty(16, 1024, device="cuda", dtype=torch.bfloat16)
    with edges.scope():
        lease = edges.acquire("e")
        lease.tensor[:16].fill_(1.0)
        writer.wait_stream(main)
        with torch.cuda.stream(writer):
            torch.cuda._sleep(SLOW_CYCLES)
            lease.tensor[16:].fill_(2.0)
            if mark_written:
                lease.mark_written()
        head.copy_(lease.tensor[:16])
        edges.release(lease)
    torch.cuda.synchronize()
    return count_non_sentinel(edges["e"].slots[0])


def test_reset_joins_all_writers():
    assert _late_writer_run(mark_written=True) == 0, "a late writer landed after the reset"
    assert _late_writer_run(mark_written=False) > 0, "control: unjoined late writer went unseen"


def test_generations_and_slot_wrap():
    edges = _edges("fill")
    edges.workspace("e", (16, 1024), torch.bfloat16)
    seen = []
    with edges.scope():
        for _ in range(5):
            tensor, slot, gen = edges.acquire("e")
            tensor.fill_(1.0)
            seen.append((slot, gen))
            edges.release("e", slot, gen)
    assert seen == [(0, 1), (1, 1), (0, 2), (1, 2), (0, 3)], seen
    with edges.scope():
        lease = edges.acquire("e")
        assert (lease.slot, lease.gen) == (0, 4), "slots restart at 0 in every scope"
        try:
            edges.release("e", 0, 3)
            raise AssertionError("release of a stale generation was accepted")
        except RuntimeError:
            pass
        edges.release(lease)
        try:
            edges.release(lease)
            raise AssertionError("double release was accepted")
        except RuntimeError:
            pass
    try:
        with edges.scope():
            edges.acquire("e")
            edges.acquire("e")
            edges.acquire("e")
        raise AssertionError("a third live lease on two slots was accepted")
    except RuntimeError:
        pass
    torch.cuda.synchronize()
    assert not edges["e"].in_use(0) and not edges["e"].in_use(1), "failed scope leaked leases"
    for slot in edges["e"].slots:
        assert count_non_sentinel(slot) == 0, "failed scope did not re-arm its slots"
    try:
        with edges.scope():
            edges.acquire("e")
        raise AssertionError("a scope ending with a live lease did not raise")
    except RuntimeError:
        pass
    torch.cuda.synchronize()
    assert count_non_sentinel(edges["e"].slots[0]) == 0


def _stale_run(skip_wait):
    edges = _edges("template")
    ws = edges.workspace("e", (64, 1 << 20), torch.bfloat16)
    reader = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    violations = torch.zeros((), dtype=torch.int64, device="cuda")
    with edges.scope():
        for step in range(6):
            if skip_wait:
                ws._pending = [False] * ws.num_slots
            lease = edges.acquire("e")
            violations += non_sentinel(lease.tensor)
            lease.tensor.fill_(float(step + 1))
            reader.wait_stream(main)
            with torch.cuda.stream(reader):
                if step == 0:
                    torch.cuda._sleep(SLOW_CYCLES)
                lease.tensor.sum()
            edges.release(lease, streams=[reader])
    torch.cuda.synchronize()
    return int(violations.item())


def test_producer_waits_for_reset():
    assert _stale_run(skip_wait=False) == 0, "a producer saw the previous generation's data"
    assert _stale_run(skip_wait=True) > 0, "control: skipping the reset wait went unseen"


def _layered_forward(edges, x_in, out, violations, side, layers=5, slow=True):
    main = torch.cuda.current_stream()
    with edges.scope():
        cur, prev = x_in, None
        for layer in range(layers):
            la = edges.acquire("a")
            violations += non_sentinel(la.tensor)
            torch.add(cur, float(layer + 1), out=la.tensor)
            if prev is not None:
                edges.release(prev)
            lb = edges.acquire("b")
            violations += non_sentinel(lb.tensor)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                if slow:
                    torch.cuda._sleep(SLOW_CYCLES // 50)
                torch.mul(la.tensor, 0.5, out=lb.tensor)
                lb.mark_written()
            edges.release(la, streams=[side])
            main.wait_stream(side)
            cur, prev = lb.tensor, lb
        out.copy_(cur)
        edges.release(prev)


def _layered_reference(x, layers=5):
    for layer in range(layers):
        x = torch.mul(torch.add(x, float(layer + 1)), 0.5)
    return x


@_parametrize("backend", _backends())
def test_graph_capture_and_replays(backend):
    edges = _edges(backend)
    edges.workspace("a", (16, 4096), torch.bfloat16)
    edges.workspace("b", (16, 4096), torch.bfloat16)
    x_in = torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x_in)
    violations = torch.zeros((), dtype=torch.int64, device="cuda")
    side = torch.cuda.Stream()

    for _ in range(2):
        _layered_forward(edges, x_in, out, violations, side)
    torch.cuda.synchronize()
    assert torch.equal(out, _layered_reference(x_in)), "eager forward mismatch"

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _layered_forward(edges, x_in, out, violations, side)
    torch.cuda.synchronize()
    violations.zero_()

    for _ in range(3):
        x_in.copy_(torch.randn_like(x_in))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, _layered_reference(x_in)), "replay mismatch"
        for name in ("a", "b"):
            for slot in edges[name].slots:
                assert count_non_sentinel(slot) == 0, "replay ended with an unarmed slot"

    inputs = [torch.randn_like(x_in) for _ in range(10)]
    results = []
    for x in inputs:
        x_in.copy_(x)
        graph.replay()
        results.append(out.clone())
    torch.cuda.synchronize()
    for x, result in zip(inputs, results):
        assert torch.equal(result, _layered_reference(x)), "back-to-back replay mismatch"
    assert int(violations.item()) == 0, "a producer saw an unarmed slot during replay"


def test_memory_ledger():
    edges = _edges("template")
    specs = {
        "mla.dec.fused_q": ((16, 96, 576), torch.bfloat16),
        "kda.dec.qkvg": ((16, 49152), torch.bfloat16),
        "kda.dec.core_o": ((16, 96, 128), torch.bfloat16),
        "mla.dec.gated": ((16, 12288), torch.bfloat16),
        "mla.dec.partial": ((16, 512), torch.float32),
    }
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for name, (shape, dtype) in specs.items():
        edges.workspace(name, shape, dtype)
    torch.cuda.synchronize()
    delta = torch.cuda.memory_allocated() - before

    def rounded(nbytes):
        return -(-nbytes // 512) * 512

    expected = 0
    largest = {torch.int16: 0, torch.int32: 0}
    for shape, dtype in specs.values():
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        expected += 2 * rounded(nbytes)
        lane = _lane(dtype)[0]
        largest[lane] = max(largest[lane], nbytes)
    expected += sum(rounded(n) for n in largest.values())
    assert delta == expected, (delta, expected)
    rows = {row["name"]: row for row in edges.ledger()}
    assert rows["kda.dec.qkvg"]["bytes_per_slot"] == 16 * 49152 * 2
    assert rows["kda.dec.qkvg"]["bytes"] == 2 * 16 * 49152 * 2
    assert rows["template[int16]"]["bytes"] == 16 * 96 * 576 * 2
    assert rows["template[int32]"]["bytes"] == 16 * 512 * 4
    assert edges.total_bytes() == sum(int(row["bytes"]) for row in rows.values())
    # A larger TEMPLATE edge registered later grows the template; the old one stays alive.
    edges.workspace("grow", (16, 96, 1024), torch.bfloat16)
    rows = {row["name"]: row for row in edges.ledger()}
    assert rows["template[int16]"]["bytes"] == 16 * 96 * 1024 * 2
    assert rows["template[int16] retired"]["bytes"] == 16 * 96 * 576 * 2
    print("    ledger:")
    for row in rows.values():
        print(f"      {row['name']:<18} {row.get('slots', '-')!s:>5} {int(row['bytes']):>10} B")


def test_stream_manager_lamport_region():
    mod = fgs()
    assert mod.FineGrainedSyncConfig.from_env() is None or os.environ.get("TLLM_FGS_MODE")
    config = mod.FineGrainedSyncConfig(default_policy=mod.EdgePolicy.LAMPORT)
    manager = mod.StreamManager(config)
    manager.edges.workspace("e", (16, 1024), torch.bfloat16)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)

    def region():
        with manager.region():
            assert manager.edges.in_scope
            with manager.stage("p"):
                lease = manager.acquire("e")
                lease.tensor.copy_(x)
            assert manager._num_released["p"] == 2, "acquire did not mark the producer stage"
            with manager.stage("c", after=("p",)):
                consumer = torch.cuda.current_stream()
            # A real LAMPORT consumer polls the sentinel; this test only checks ordering
            # plumbing, so it reads after the producer completes.
            consumer.wait_event(manager._done_events["p"])
            with torch.cuda.stream(consumer):
                out.copy_(lease.tensor)
            manager.release(lease, streams=[consumer])
        assert not manager.edges.in_scope

    region()
    torch.cuda.synchronize()
    assert torch.equal(out, x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        region()
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, x)
        assert count_non_sentinel(manager.edges["e"].slots[0]) == 0


def test_standalone_unchanged():
    mod = fgs()
    manager = mod.StreamManager(mod.FineGrainedSyncConfig(default_policy=mod.EdgePolicy.STANDALONE))
    assert manager._edges is None, "non-LAMPORT managers must not create a reset stream"
    x = torch.randn(1024, device="cuda")
    with manager.region():
        with manager.stage("p"):
            y = x * 2
        with manager.stage("c", after=("p",)):
            z = y + 1
        manager.keep(y, z)
    torch.cuda.synchronize()
    assert torch.equal(z, x * 2 + 1)


TESTS = [
    ("cute_fill_standalone", test_cute_fill_standalone, ()),
    *[(f"preinit_bitexact[{b}]", test_preinit_bitexact, (b,)) for b in _backends()],
    *[
        (f"reset_bitexact_row_bound[{b}]", test_reset_bitexact_and_row_bound, (b,))
        for b in _backends()
    ],
    *[
        (f"release_waits_slow_reader[{b}]", test_release_waits_for_slow_reader, (b,))
        for b in ("fill", "cute")
    ],
    ("fan_out_readers", test_fan_out_readers, ()),
    ("alias_views", test_alias_views, ()),
    ("reset_joins_all_writers", test_reset_joins_all_writers, ()),
    ("generations_slot_wrap", test_generations_and_slot_wrap, ()),
    ("producer_waits_for_reset", test_producer_waits_for_reset, ()),
    *[(f"graph_capture_replays[{b}]", test_graph_capture_and_replays, (b,)) for b in _backends()],
    ("memory_ledger", test_memory_ledger, ()),
    ("stream_manager_lamport_region", test_stream_manager_lamport_region, ()),
    ("standalone_unchanged", test_standalone_unchanged, ()),
]


def main():
    only = sys.argv[1:]
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}, module {fgs().__file__}")
    results = []
    for name, fn, args in TESTS:
        if only and not any(key in name for key in only):
            continue
        try:
            fn(*args)
            results.append((name, "PASS", ""))
        except Exception as error:  # noqa: BLE001 - report every failure and keep going
            results.append((name, "FAIL", f"{type(error).__name__}: {error}"))
            traceback.print_exc()
        torch.cuda.synchronize()
        print(f"{results[-1][1]}  {name}  {results[-1][2]}", flush=True)
    failed = [r for r in results if r[1] == "FAIL"]
    print(f"\nSUMMARY: {len(results) - len(failed)}/{len(results)} PASS")
    for name, _, why in failed:
        print(f"  FAIL {name}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    if "FGS_OVL_DIR" not in os.environ:
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovl")
        if os.path.isdir(local):
            os.environ["FGS_OVL_DIR"] = local
    sys.exit(main())
