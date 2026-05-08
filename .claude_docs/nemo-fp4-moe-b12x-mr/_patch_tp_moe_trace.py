#!/usr/bin/env python3
"""Insert trace prints into b12x.integration.tp_moe._launch_compact_static.

Idempotent (skips if already patched). Verifies all 5 needles match
before writing back, so partial application is impossible.
"""
import sys

PATH = '/usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py'

with open(PATH) as f:
    src = f.read()

print(f'[patch] file length: {len(src)} bytes')
if '[trace-luke]' in src:
    print('[patch] already patched — skipping')
    sys.exit(0)

needles_inserts = [
    # A: before use_micro_direct = ...
    (
        '    use_micro_direct = quant_mode in {"nvfp4", "w4a16"} and micro_cls.is_supported(',
        (
            '    if getattr(_launch_compact_static, "_probe", 0) < 6:\n'
            '        _launch_compact_static._probe = getattr(_launch_compact_static, "_probe", 0) + 1\n'
            '        print(f"[trace-luke] _launch_compact_static: m={m} k={k} n={n} num_topk={num_topk} E={weight_E} si={share_input_across_experts} ses={share_expert_scales} quant_mode={quant_mode}", flush=True)\n'
            '    use_micro_direct = quant_mode in {"nvfp4", "w4a16"} and micro_cls.is_supported('
        ),
        'before',  # replace mode: needle replaced by insert
    ),
    # B: after use_w4a16_micro_compact = False
    (
        '    use_w4a16_micro_compact = False\n',
        (
            '    use_w4a16_micro_compact = False\n'
            '    if getattr(_launch_compact_static, "_probe", 0) <= 6:\n'
            '        print(f"[trace-luke]   use_micro_direct={use_micro_direct}", flush=True)\n'
        ),
        'after',
    ),
    # C: before "if _compiled_direct_micro_accepts_block_dim("
    (
        '        if _compiled_direct_micro_accepts_block_dim(\n            compiled,\n            _DIRECT_MICRO_BLOCK_DIM,\n        ):\n',
        (
            '        _trace_gate = _compiled_direct_micro_accepts_block_dim(compiled, _DIRECT_MICRO_BLOCK_DIM)\n'
            '        if getattr(_launch_compact_static, "_probe", 0) <= 6:\n'
            '            print(f"[trace-luke]   _compiled_direct_micro_accepts_block_dim={_trace_gate} (BLOCK_DIM={_DIRECT_MICRO_BLOCK_DIM})", flush=True)\n'
            '        if _compiled_direct_micro_accepts_block_dim(\n            compiled,\n            _DIRECT_MICRO_BLOCK_DIM,\n        ):\n'
        ),
        'before',
    ),
    # D: tag MICRO LAUNCHED before its return
    (
        '            )\n            return\n\n    if use_w4a16_micro_compact:\n',
        (
            '            )\n'
            '            if getattr(_launch_compact_static, "_probe", 0) <= 6:\n'
            '                print("[trace-luke]   ** MICRO LAUNCHED **", flush=True)\n'
            '            return\n\n    if use_w4a16_micro_compact:\n'
        ),
        'before',
    ),
]
# Skip the "FALLING THROUGH TO STATIC" probe — `if quant_mode == "w4a16":` has
# 6 occurrences across the file and only one is inside _launch_compact_static.
# Patches A-D are enough to answer the question:
# - A prints on entry,
# - B prints use_micro_direct,
# - C prints the runtime block-dim gate,
# - D prints "MICRO LAUNCHED" iff the kernel actually fired.
# Absence of D output ==> the dispatch fell through to MoEStaticKernel.

for i, (needle, insert, mode) in enumerate(needles_inserts):
    count = src.count(needle)
    print(f'[patch] needle #{i+1}: {count} occurrences ({mode})')
    if count == 0:
        print(f'  FAILED: needle not found')
        print(f'  needle = {needle!r}')
        sys.exit(1)
    if mode == 'before-once' and count > 1:
        # only replace first occurrence
        src = src.replace(needle, insert, 1)
    else:
        src = src.replace(needle, insert)

print(f'[patch] new file length: {len(src)} bytes')
with open(PATH, 'w') as f:
    f.write(src)
print('[patch] done')
