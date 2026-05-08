"""Token-parity check for hybrid-luke vs hybrid-old.

Identical to ``parity_check.py`` but parametrizes the MoE backend so we can
A/B test the lukealonso b12x decode kernel against the flashinfer b12x decode
kernel. Both runs use ``TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD=64``
so prefill is on the CUTLASS path in both arms; only decode differs.

Usage:
    # Existing hybrid — flashinfer-b12x decode kernel
    TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD=64 \\
      python3 parity_check_b12x_luke.py --model_dir <path> \\
        --moe-backend FLASHINFER --out /tmp/parity_old.txt

    # New hybrid — lukealonso-b12x decode kernel
    TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD=64 \\
      python3 parity_check_b12x_luke.py --model_dir <path> \\
        --moe-backend B12X_LUKE --out /tmp/parity_new.txt

    diff /tmp/parity_old.txt /tmp/parity_new.txt
"""

import argparse
import os
import sys

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

# Same prompt as parity_check.py so we can cross-compare with HYBRID_RESULTS.md
# parity OFF/ON arms if needed. Designed to tokenize to >= 100 tokens.
LONG_PROMPT = (
    "The history of computing began in the early 19th century with Charles "
    "Babbage's design of the analytical engine, which introduced the concept "
    "of programmable computation. Through the 20th century, electromechanical "
    "relays gave way to vacuum tubes, then to transistors, and ultimately to "
    "integrated circuits. The exponential growth of transistor density, "
    "described by Moore's law, drove relentless improvements in performance "
    "and energy efficiency for several decades. Today, modern GPU "
    "architectures continue this legacy by combining thousands of parallel "
    "execution units with specialized matrix-multiply tensor cores. List the "
    "five most important milestones in the history of GPU architecture: "
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_tokens", type=int, default=32)
    ap.add_argument("--moe-backend", default="B12X_LUKE",
                    choices=["FLASHINFER", "B12X_LUKE"])
    ap.add_argument("--out", default="/tmp/parity_tokens_b12x_luke.txt")
    args = ap.parse_args()

    threshold = os.environ.get("TRTLLM_FLASHINFER_PREFILL_VIA_CUTLASS_THRESHOLD",
                               "<unset>")
    print(f"[parity] threshold env = {threshold}")
    print(f"[parity] moe_backend = {args.moe_backend}")
    print(f"[parity] prompt = {LONG_PROMPT[:80]!r}...")

    llm = LLM(
        model=args.model_dir,
        backend="pytorch",
        max_batch_size=1,
        max_num_tokens=4096,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, free_gpu_memory_fraction=0.6),
        moe_config=MoeConfig(backend=args.moe_backend),
        enable_chunked_prefill=True,
    )

    sp = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        seed=42,
    )

    out = llm.generate([LONG_PROMPT], sampling_params=sp)[0]
    out_ids = list(out.outputs[0].token_ids)
    in_ids = list(out.prompt_token_ids)
    print(f"[parity] prompt token count = {len(in_ids)}")
    print(f"[parity] output_ids ({len(out_ids)}): {out_ids}")
    text = out.outputs[0].text
    print(f"[parity] output_text: {text!r}")

    with open(args.out, "w") as f:
        f.write(f"moe_backend={args.moe_backend}\n")
        f.write(f"prompt_len={len(in_ids)}\n")
        f.write("output_ids=" + " ".join(str(t) for t in out_ids) + "\n")
        f.write(f"output_text={text!r}\n")
    print(f"[parity] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
