"""Minimal K2.6 throughput bench — bypasses tokenizer like minimal_generate.py.

trtllm-bench needs a working HF AutoTokenizer; K2.6's tokenization_kimi.py
chokes on AutoTokenizer (fast tokenizer can't be built; slow fallback hangs).
For the kernel-level A/B we don't actually need correct text, we just need to
measure TTFT, ITL, and aggregate throughput. So feed pre-baked token-ID
prompts of the requested ISL.

CLI mirrors the subset of trtllm-bench throughput we care about:

  python minimal_bench.py \
      --model_dir /scratch/hf-cache-patched/k2.6-bf16kv \
      --tp_size 4 \
      --extra_llm_api_options /path/to/bench-config.yml \
      --isl 1024 --osl 1024 \
      --num_requests 16 --concurrency 1

Outputs an "Aggregate" block + "Per User" block compatible with the
verify_backend_swap.sh-style summary grep ("Per User Output", "Token
Throughput", "Time To First", "Inter Token").
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

from tensorrt_llm import LLM, SamplingParams


def _build_llm_kwargs_from_yaml(yml_path: str | None) -> dict:
    """Read an extra_llm_api_options YAML and return a kwargs dict."""
    if not yml_path:
        return {}
    with open(yml_path) as f:
        cfg = yaml.safe_load(f) or {}
    # Some keys in the YAML have different names from LLM kwargs; pass them
    # through verbatim — LLM(**...) raises if anything is unrecognized, so
    # the YAML is the source of truth.
    return cfg


def _make_prompt(token_ids: list[int], isl: int) -> list[int]:
    """Pad/truncate a base sequence to the requested input length."""
    if isl <= len(token_ids):
        return token_ids[:isl]
    times = (isl + len(token_ids) - 1) // len(token_ids)
    return (token_ids * times)[:isl]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--tp_size", type=int, default=4)
    p.add_argument("--extra_llm_api_options", required=True)
    p.add_argument("--isl", type=int, default=1024)
    p.add_argument("--osl", type=int, default=1024)
    p.add_argument("--num_requests", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    NEVER_HIT_END_ID = 99999999
    BASE_TOKENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    prompt_ids = _make_prompt(BASE_TOKENS, args.isl)

    yml_kwargs = _build_llm_kwargs_from_yaml(args.extra_llm_api_options)
    # CLI args win over the YAML for: model dir, tp size, trust_remote_code,
    # and skip_tokenizer_init (which the YAML must not override — we *need*
    # to skip the tokenizer to bypass K2.6's broken HF AutoTokenizer).
    for k in ("model", "tensor_parallel_size", "trust_remote_code",
              "skip_tokenizer_init"):
        yml_kwargs.pop(k, None)

    llm_kwargs = dict(
        model=args.model_dir,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        **yml_kwargs,
    )

    print(f"[bench] starting at {time.strftime('%H:%M:%S')}", flush=True)
    print(
        f"[bench] model={args.model_dir} tp={args.tp_size} isl={args.isl} "
        f"osl={args.osl} num_req={args.num_requests} conc={args.concurrency}",
        flush=True,
    )
    print(f"[bench] llm_kwargs (from YAML overlay):", flush=True)
    for k, v in yml_kwargs.items():
        print(f"           {k} = {v}", flush=True)

    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[bench] LLM init done in {time.time()-t0:.1f}s", flush=True)

    sp = SamplingParams(
        max_tokens=args.osl,
        end_id=NEVER_HIT_END_ID,
        temperature=1.0,
    )

    # ---- Warmup ----
    if args.warmup > 0:
        print(f"[bench] running {args.warmup} warmup request(s)...", flush=True)
        wt0 = time.time()
        warm_outs = llm.generate([prompt_ids] * args.warmup, sp)
        wt = time.time() - wt0
        warm_tokens = sum(len(o.outputs[0].token_ids) for o in warm_outs)
        print(
            f"[bench]   warmup: {warm_tokens} tokens in {wt:.2f}s "
            f"({warm_tokens/max(wt,1e-6):.1f} tok/s aggregate)",
            flush=True,
        )

    # ---- Real timed run ----
    print(
        f"[bench] timed run: {args.num_requests} requests, "
        f"concurrency={args.concurrency}",
        flush=True,
    )

    def _run_one(_: int):
        t = time.time()
        outs = llm.generate([prompt_ids], sp)
        elapsed = time.time() - t
        out_tokens = len(outs[0].outputs[0].token_ids) if outs else 0
        return elapsed, out_tokens

    run_t0 = time.time()
    per_req: list[tuple[float, int]] = []
    if args.concurrency <= 1:
        for i in range(args.num_requests):
            per_req.append(_run_one(i))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for r in ex.map(_run_one, range(args.num_requests)):
                per_req.append(r)
    run_elapsed = time.time() - run_t0

    # ---- Metrics ----
    total_out_tokens = sum(t for _, t in per_req)
    total_in_tokens = args.isl * args.num_requests
    agg_tok_per_s = total_out_tokens / max(run_elapsed, 1e-6)
    per_user_tok_per_s = [
        t / max(e, 1e-6) for e, t in per_req if e > 0 and t > 0
    ]
    per_user_med = (
        statistics.median(per_user_tok_per_s) if per_user_tok_per_s else 0.0
    )

    # Per-token average latency on the timed run: total decode time / total
    # decode tokens. This is the closest proxy to ITL we have without
    # streaming.
    avg_itl_ms = (
        run_elapsed * 1000.0 / max(total_out_tokens, 1)
        if total_out_tokens else 0.0
    )

    print()
    print("=== Aggregate ===")
    print(f"  Total requests: {args.num_requests}")
    print(f"  Total wall time (timed run): {run_elapsed:.2f}s")
    print(f"  Input tokens: {total_in_tokens}")
    print(f"  Output tokens: {total_out_tokens}")
    print(f"  Token Throughput: {agg_tok_per_s:.1f} tok/s")
    print(f"  Inter Token avg: {avg_itl_ms:.2f} ms/tok")
    print()
    print("=== Per User ===")
    print(f"  Per User Output Throughput median: {per_user_med:.2f} tok/s")
    if per_user_tok_per_s:
        print(
            f"  Per User Output Throughput min/max: "
            f"{min(per_user_tok_per_s):.2f} / "
            f"{max(per_user_tok_per_s):.2f} tok/s"
        )
    # TTFT: not available without streaming. Report -1 to signal so.
    print(f"  Time To First Token: -1 ms (not measured; non-streaming)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
