# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generate mixed traffic against a trtllm-serve endpoint for demo purposes."""

import argparse
import concurrent.futures
import json
import random
import time
import urllib.request
import urllib.error


PROMPTS = [
    "Explain quantum computing in simple terms",
    "Write a haiku about GPU memory",
    "What is the meaning of life?",
    "Count from 1 to 20",
    "Translate 'hello world' to 5 languages",
    "Write Python code to sort a list",
    "Summarize the history of NVIDIA in 3 sentences",
    "What are the benefits of tensor parallelism?",
    "Explain KV cache in LLM inference",
    "Write a short poem about parallel computing",
]

BAD_REQUESTS = [
    # Missing model field
    {"messages": [{"role": "user", "content": "hello"}]},
    # Empty messages
    {"model": "test", "messages": []},
    # Invalid role
    {"model": "test", "messages": [{"role": "cat", "content": "meow"}]},
    # Huge max_tokens
    {"model": "test", "messages": [{"role": "user", "content": "hi"}],
     "max_tokens": 999999999},
    # Garbage JSON
    None,  # will send raw garbage
]


def send_good_request(endpoint: str, model: str, prompt: str,
                      max_tokens: int = 100) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": random.uniform(0.1, 1.0),
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"})
    try:
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=120)
        elapsed = time.time() - t0
        body = json.loads(resp.read())
        tokens = body.get("usage", {}).get("completion_tokens", 0)
        return {"status": "ok", "tokens": tokens, "time": round(elapsed, 2)}
    except Exception as e:
        return {"status": "error", "error": str(e)[:100]}


def send_bad_request(endpoint: str, bad: dict) -> dict:
    if bad is None:
        data = b"{{{{not json at all!!!}}"
    else:
        data = json.dumps(bad).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return {"status": "unexpected_ok", "code": resp.status}
    except urllib.error.HTTPError as e:
        return {"status": "rejected", "code": e.code,
                "reason": e.reason[:60]}
    except Exception as e:
        return {"status": "error", "error": str(e)[:80]}


def get_model(endpoint: str) -> str:
    try:
        resp = urllib.request.urlopen(f"{endpoint}/v1/models", timeout=5)
        data = json.loads(resp.read())
        return data["data"][0]["id"]
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Generate mixed traffic for trtllm-ops demo")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--good", type=int, default=50,
                        help="Number of good requests")
    parser.add_argument("--bad", type=int, default=5,
                        help="Number of bad requests to inject")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--continuous", action="store_true",
                        help="Keep sending forever")
    args = parser.parse_args()

    model = get_model(args.endpoint)
    print(f"Target: {args.endpoint}")
    print(f"Model:  {model}")
    print(f"Plan:   {args.good} good + {args.bad} bad requests, "
          f"concurrency={args.concurrency}")
    print()

    total_ok = 0
    total_err = 0
    total_tokens = 0
    t_start = time.time()

    round_num = 0
    while True:
        round_num += 1
        # Build task list: mix good and bad
        tasks = []
        for _ in range(args.good):
            prompt = random.choice(PROMPTS)
            tasks.append(("good", prompt))
        for _ in range(args.bad):
            bad = random.choice(BAD_REQUESTS)
            tasks.append(("bad", bad))
        random.shuffle(tasks)

        print(f"--- Round {round_num}: sending {len(tasks)} requests ---")

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency) as pool:
            futs = {}
            for kind, data in tasks:
                if kind == "good":
                    f = pool.submit(send_good_request, args.endpoint,
                                    model, data, args.max_tokens)
                else:
                    f = pool.submit(send_bad_request, args.endpoint, data)
                futs[f] = kind

            for f in concurrent.futures.as_completed(futs):
                kind = futs[f]
                result = f.result()
                if kind == "good" and result["status"] == "ok":
                    total_ok += 1
                    total_tokens += result.get("tokens", 0)
                    print(f"  \033[32m✓\033[0m {result['tokens']}tok "
                          f"{result['time']}s", end="  ")
                elif kind == "bad":
                    total_err += 1
                    print(f"  \033[33m✗ BAD→{result.get('code','?')}\033[0m",
                          end="  ")
                else:
                    total_err += 1
                    print(f"  \033[31m✗ {result.get('error','?')[:40]}\033[0m",
                          end="  ")

        elapsed = time.time() - t_start
        tps = total_tokens / max(elapsed, 0.01)
        print(f"\n\n  Total: {total_ok} ok, {total_err} err, "
              f"{total_tokens} tokens, {tps:.0f} tok/s avg\n")

        if not args.continuous:
            break

    print(f"\nDone. {total_ok + total_err} requests in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
