#!/usr/bin/env bash
# Patch K2.5 NVFP4 -> BF16 KV by stripping `quantization.kv_cache_quant_algo`
# from hf_quant_config.json. Mirrors the K2.6 BF16-KV patch.
#
# Pre-req: hf download nvidia/Kimi-K2.5-NVFP4 --cache-dir /scratch/hf-cache
# has completed.
#
# Output: /scratch/hf-cache-patched/k2.5-bf16kv/ — symlink tree to the
# original snapshot, with hf_quant_config.json rewritten.

set -euo pipefail

SRC="${SRC:-/scratch/hf-cache/models--nvidia--Kimi-K2.5-NVFP4/snapshots}"
DST="${DST:-/scratch/hf-cache-patched/k2.5-bf16kv}"

# Resolve the snapshot dir (there should be exactly one).
SNAP=$(ls -d $SRC/*/ 2>/dev/null | head -1)
if [[ -z "$SNAP" ]]; then
    echo "ERROR: no snapshot under $SRC" >&2
    exit 1
fi
SNAP=${SNAP%/}
echo "[patch] source snapshot: $SNAP"

mkdir -p "$DST"
echo "[patch] destination: $DST"

# Symlink everything from the source snapshot into DST.
for f in "$SNAP"/*; do
    name=$(basename "$f")
    ln -sf "$f" "$DST/$name"
done

# Override the symlink for hf_quant_config.json with a real file that
# has kv_cache_quant_algo removed.
rm -f "$DST/hf_quant_config.json"
python3 - "$SNAP/hf_quant_config.json" "$DST/hf_quant_config.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = json.load(f)
removed = cfg.get("quantization", {}).pop("kv_cache_quant_algo", None)
with open(dst, "w") as f:
    json.dump(cfg, f, indent=4)
print(f"[patch] removed quantization.kv_cache_quant_algo (was: {removed!r})")
PY

echo "[patch] verifying patched config..."
python3 - "$DST/hf_quant_config.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
assert cfg.get("quantization", {}).get("kv_cache_quant_algo") is None, \
    "kv_cache_quant_algo still present!"
print("[patch] OK: kv_cache_quant_algo not in patched hf_quant_config.json")
PY

echo "[patch] $DST ready (BF16 KV)"
ls -la "$DST" | head -10
