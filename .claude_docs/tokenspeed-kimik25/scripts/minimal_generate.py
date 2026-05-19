"""Minimal Kimi K2.6 generate test — skips tokenizer entirely.

Used by Phase 3 (verify_kernel_swap) when quickstart_advanced.py's tokenizer
init silently fails on the NVFP4 K2-Thinking snapshot. We only need 32
generated tokens for nsys to capture kernel symbols; don't care what
they decode to.

Reads three env vars:
  MODEL_DIR  — path to checkpoint snapshot
  TP_SIZE    — tensor parallel size (default 4)
  MAX_TOKENS — tokens to generate (default 32)

Run with TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1 TLLM_TOKENSPEED_MLA={0|1} to
exercise the TokenSpeed swap.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tensorrt_llm import LLM, SamplingParams


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--tp_size", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=32)
    p.add_argument("--attention_backend", default="TRTLLM")
    p.add_argument("--moe_backend", default="AUTO")
    # NB: kv_cache_dtype default "auto" means the model config drives the choice
    # (K2.6 NVFP4 defaults to fp8 KV which causes trtllm_gen.is_supported() to
    # reject the BF16-Q/FP8-KV/BF16-O combo — see Phase 3 diagnostic finding).
    # Override to "bfloat16" to force the Q/KV/O dtypes to all be bf16, which
    # lets trtllm_gen.is_supported() pass and exercise the spike's swap.
    # Valid values per TorchLlmArgs.kv_cache_config.dtype:
    #   "auto" | "fp8" | "nvfp4" | a torch.dtype string ("bfloat16", "float16", ...)
    p.add_argument("--kv_cache_dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "fp8", "nvfp4"])
    args = p.parse_args()

    # [EOS] in the Kimi tokenizer is token id 163585 (per tokenizer_config.json
    # added_tokens_decoder). Set a value we'll never hit so generation runs
    # all 32 tokens.
    NEVER_HIT_END_ID = 99999999

    # Pre-encoded dummy prompt: a short non-trivial sequence of token ids
    # that's within the vocab. The Kimi vocab is ~163600 tokens; using
    # low ids avoids special tokens. Exactly which prompt we use doesn't
    # matter — we just need the engine to process it and run decode.
    DUMMY_PROMPT_IDS = [128, 256, 512, 1024, 2048, 4096, 8192]

    print(f"[minimal] starting at {time.strftime('%H:%M:%S')}", flush=True)
    print(
        f"[minimal] TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION="
        f"{os.environ.get('TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION')} "
        f"TLLM_TOKENSPEED_MLA={os.environ.get('TLLM_TOKENSPEED_MLA')}",
        flush=True,
    )

    llm_kwargs = {
        "model": args.model_dir,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": True,
        "skip_tokenizer_init": True,    # don't load the broken Kimi tokenizer
        "attn_backend": args.attention_backend,
        # moe_backend intentionally left as engine default — we care about MLA decode.
    }
    if args.kv_cache_dtype != "auto":
        # Pass via kv_cache_config dict (matches the YAML bench configs' shape).
        llm_kwargs["kv_cache_config"] = {"dtype": args.kv_cache_dtype}

    t0 = time.time()
    llm = LLM(**llm_kwargs)
    t_init = time.time() - t0
    print(f"[minimal] LLM init done in {t_init:.1f}s", flush=True)

    sp = SamplingParams(
        max_tokens=args.max_tokens,
        end_id=NEVER_HIT_END_ID,
        temperature=1.0,
    )

    t0 = time.time()
    outputs = llm.generate([DUMMY_PROMPT_IDS], sp)
    t_gen = time.time() - t0
    out_ids = outputs[0].outputs[0].token_ids if outputs else []
    print(f"[minimal] generated {len(out_ids)} tokens in {t_gen:.2f}s "
          f"({len(out_ids) / max(t_gen, 1e-6):.1f} tok/s)", flush=True)
    print(f"[minimal] first 8 token ids: {list(out_ids[:8])}", flush=True)
    print(f"[minimal] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
