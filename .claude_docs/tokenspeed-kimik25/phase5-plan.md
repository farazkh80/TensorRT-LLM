# Phase 5 plan — K2.5 NVFP4 + EAGLE-3 (MLA) A/B (TRTLLM vs TOKENSPEED_MLA)

**Goal:** Test whether TokenSpeed's MLA-decode kernel produces any perf
benefit on Kimi K2.5 NVFP4 in the **MTP / EAGLE-3 regime** (`max_draft_len=3`
→ `q_len_per_req=4`) — the regime where TokenSpeed's headline 2× claim
lives. The Phase 4 grid covered only `q_len_per_req=1` and found TS −3 to
−7% on K2.6. Phase 5 closes the gap to the headline regime.

**Hardware:** B300 SXM6 (sm_103a), single node, TP=4.
**TRT-LLM:** `tokenspeed-kimik25-eval-public` rebased on
`upstream/main@f278c4f170` + PR #9677 (Eagle: MLA Based Eagle) +
PR #14291 (FMHA JIT fix). Both already in our tree.

## Inputs

| Asset | Source | Size | Status |
|---|---|---|---|
| Target: `nvidia/Kimi-K2.5-NVFP4` | HF | 591 GB (119 safetensors) | needs download |
| Draft: `nvidia/Kimi-K2.5-Thinking-Eagle3` | HF | 3.68 GB (1 safetensors) | needs download |
| Reference recipe | `NVIDIA/srt-slurm` PR #24 → `recipes/kimi2.5/.../ISL1K_OSL1K/MTP/ctx1dep4_gen5tep4_batch2_allconc_eplb0_mtp3.yaml` | — | recipe is GB200/disagg; we adapt the **decode side** to single-node B300 |
| Disk | `/scratch` | 2.4 TB free | OK (need ~600 GB) |

## Decisions (user-confirmed)

- **KV dtype:** BF16 (patched K2.5 — strip `kv_cache_quant_algo`). Trades
  prod fidelity for a clean TS A/B (TS kernel requires `kv.dtype == q.dtype`).
- **Bench harness:** `trtllm-bench` with K2.5 EAGLE3 config YAML (the
  K2.5 tokenizer should work per the recipe's `trust_remote_code: true`).
- **Grid scope:** Minimal smoke first — 1 config × 2 arms = 2 runs.
  Iterate only if the smoke shows something interesting.

## Execution steps

### Step 1 — Download K2.5 + EAGLE3 to `/scratch/hf-cache/` (~20–60 min)

```bash
docker exec tokenspeed-spike-k26 bash -lc '
  cd /scratch
  HF_HOME=/scratch/hf-cache huggingface-cli download nvidia/Kimi-K2.5-NVFP4 \
    --cache-dir /scratch/hf-cache --quiet &
  PID_TARGET=$!
  HF_HOME=/scratch/hf-cache huggingface-cli download nvidia/Kimi-K2.5-Thinking-Eagle3 \
    --cache-dir /scratch/hf-cache --quiet &
  PID_DRAFT=$!
  wait $PID_TARGET $PID_DRAFT
'
```

Run in background, monitor via `du -sh /scratch/hf-cache/models--nvidia--Kimi-K2.5-*` until target hits ~590 GB.

**Verification gates:**
- Snapshot exists: `/scratch/hf-cache/models--nvidia--Kimi-K2.5-NVFP4/snapshots/<sha>/config.json`
- All 119 K2.5 safetensors present (none truncated; check sizes against `siblings` from HF API).
- EAGLE3 `model.safetensors` is 3.68 GB.

**Risks:**
- HF rate limits / slow mirror → may take >1 hour. Acceptable.
- Disk pressure → if K2.6's 556 GB + K2.5's 591 GB > 1.15 TB approaches the 2.4 TB free, fall back to deleting `models--moonshotai--Kimi-K2-Thinking` (we use the NVIDIA NVFP4 quant for K2.6, not the Moonshot original).

### Step 2 — Patch K2.5 NVFP4 → BF16 KV (~30 s)

Mirror the K2.6 BF16-KV patch:

```bash
SRC=/scratch/hf-cache/models--nvidia--Kimi-K2.5-NVFP4/snapshots/<sha>
DST=/scratch/hf-cache-patched/k2.5-bf16kv
mkdir -p $DST
# Symlink everything except hf_quant_config.json…
for f in $SRC/*; do
  ln -s "$f" "$DST/$(basename $f)"
done
# …and rewrite hf_quant_config.json without kv_cache_quant_algo
rm "$DST/hf_quant_config.json"
python3 -c "
import json
c = json.load(open('$SRC/hf_quant_config.json'))
c.pop('kv_cache_quant_algo', None)
json.dump(c, open('$DST/hf_quant_config.json','w'), indent=2)
"
```

**Verification gates:**
- `cat $DST/hf_quant_config.json` shows no `kv_cache_quant_algo` key.
- `ls $DST` shows all original config + safetensor files (as symlinks).

**Risks:**
- K2.5's `hf_quant_config.json` schema differs from K2.6 → if the pop is
  a no-op or `kv_cache_quant_algo` lives in a nested key, inspect schema
  and adjust.

### Step 3 — Build K2.5 + EAGLE3 trtllm-bench config (5 min)

New file: `.claude_docs/tokenspeed-kimik25/bench-k25-mtp3.yml`. Adapt the
decode side of `ctx1dep4_gen5tep4_batch2_allconc_eplb0_mtp3.yaml`:

```yaml
# K2.5 NVFP4 + EAGLE-3 (max_draft_len=3), TP=4, single-node B300
# Adapted from srt-slurm PR #24 recipe (decode side, no disagg).

tensor_parallel_size: 4
moe_expert_parallel_size: 4
pipeline_parallel_size: 1
enable_attention_dp: false
enable_lm_head_tp_in_adp: false
trust_remote_code: true
max_batch_size: 2
max_num_tokens: 8
max_seq_len: 2088
print_iter_log: true
stream_interval: 100

cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4]

moe_config:
  backend: TRTLLM
  use_low_precision_moe_combine: true

kv_cache_config:
  dtype: auto                # BF16 from the patched snapshot
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.85

nvfp4_gemm_config:
  allowed_backends: [cutlass, cublaslt, cutedsl, cuda_core]

speculative_config:
  decoding_type: Eagle
  max_draft_len: 3
  speculative_model_dir: /scratch/hf-cache/models--nvidia--Kimi-K2.5-Thinking-Eagle3/snapshots/<sha>

attn_backend: TRTLLM         # ts sidecar will sed this to TOKENSPEED_MLA
```

**Verification gates:**
- All `speculative_config` keys parse via `EagleDecodingConfig` schema.
- The EAGLE3 snapshot dir contains `config.json` + `model.safetensors`.

**Risks:**
- `KimiK25ForConditionalGeneration` (multimodal) loading via PyTorch backend — TRT-LLM should pick up only the text decoder. **If it fails: debug the loading path; do NOT fall back to K2.6.** Per user direction (2026-05-23), Phase 5 uses K2.5 for both arms. Fallback paths if K2.5 won't load are: (a) inspect the `KimiK25ForConditionalGeneration` registration in `tensorrt_llm/_torch/models/`, add support if missing; (b) load via the underlying `DeepseekV3ForCausalLM` arch by overriding the architecture string in a config patch. Output quality / acceptance-rate / TS divergence are out of scope for Phase 5 — we are getting the kernel-level perf number first; output sanity check is a separate later phase.
- K2.5 HF AutoTokenizer broken (like K2.6) → fall back to `minimal_bench.py`
  with `skip_tokenizer_init=True` (this fallback IS allowed — it doesn't change which model we run).

### Step 4 — Smoke A/B (~20–30 min)

For each backend in `[TRTLLM, TOKENSPEED_MLA]`:

```bash
trtllm-bench \
  --model /scratch/hf-cache-patched/k2.5-bf16kv \
  throughput \
  --backend pytorch \
  --tp_size 4 \
  --dataset synth_1k1k_32.txt \
  --concurrency 2 \
  --num_requests 32 \
  --extra_llm_api_options bench-k25-mtp3-{base,ts}.yml \
  > /scratch/runs/k2.6-spike/phase5-k25-mtp3/{base,ts}.log
```

**Per-arm capture:**
- Init time (engine load + EAGLE draft load + warmup compile).
- Aggregate throughput (tok/s).
- Per-user throughput.
- ITL (inter-token latency).
- **Acceptance rate** — EAGLE-specific metric; trtllm-bench reports this
  in its summary when `speculative_config` is set.
- Wall time / num requests.

**Verification gates:**
- Both arms produce `Token Throughput`, `Inter Token avg`, `Per User
  Output Throughput median` lines.
- TS arm successfully dispatches to `TokenSpeedMLAAttention._run` (check
  for the gate log we added) — not silently falling back to thop.
- No NVRTC errors (PR #14291 fix should hold).

**Risks:**
- TS arm `_is_supported` check rejects `q_len_per_req=4`: the dispatch
  gate in `tokenspeed_mla_attention.py` doesn't restrict q_len — should
  pass. If it rejects, falls back to TrtllmAttention; the bench still
  runs but the A/B is meaningless. Need to grep server log for the
  fallback message.
- TS kernel parity divergence (max abs 0.33 / max rel ~1166× on
  bs8/q_len=4 per the DSV3-Lite spike). **Output check deferred** —
  Phase 5 captures perf-only. Output sanity / acceptance-rate parity
  evaluation is a follow-on phase.
- Acceptance-rate may differ between arms if the kernel produces
  different logits (TS divergence). For a perf-only comparison this
  contaminates the throughput number (TS may "accept" more or fewer
  tokens per draft step than TRTLLM). Mitigation: report tok/s alongside
  **forward-pass time** when available, and call out the contamination
  in the writeup. Final verdict on the TS kernel quality waits for the
  output check.

### Step 5 — Report (~10 min)

Write `.claude_docs/tokenspeed-kimik25/phase5-k25-eagle3.md` with:

- Headline table (TRTLLM vs TS for aggregate / per-user / ITL / acceptance / wall)
- Caveats: BF16 KV not prod, multi-modal target wrapper, TS divergence
- Verdict: does TS show its headline 2× in this regime, or does the
  Phase 4 conclusion ("no TS benefit") extend to MTP=3 too?
- Recommended next steps (small grid if smoke is promising; else close
  the TokenSpeed evaluation)

## Estimated total wall time

| Step | Time |
|---|---|
| 1. Download K2.5 + EAGLE3 | 20–60 min |
| 2. Patch BF16 KV | 30 s |
| 3. Build bench config | 5 min |
| 4. Smoke A/B (2 runs) | 20–30 min |
| 5. Report | 10 min |
| **Total** | **~1 to 2 hours** (mostly download-bound) |

## Outputs

- `/scratch/hf-cache/models--nvidia--Kimi-K2.5-NVFP4/`
- `/scratch/hf-cache/models--nvidia--Kimi-K2.5-Thinking-Eagle3/`
- `/scratch/hf-cache-patched/k2.5-bf16kv/` (BF16 KV symlink tree)
- `.claude_docs/tokenspeed-kimik25/bench-k25-mtp3.yml`
- `/scratch/runs/k2.6-spike/phase5-k25-mtp3/{base,ts}.log`
- `.claude_docs/tokenspeed-kimik25/phase5-k25-eagle3.md` (final writeup)

## Open questions before execution

1. Does the bench need `synth_1k1k_32.txt` pre-generated? `trtllm-bench
   prepare-dataset` can build it on the fly — first invocation will
   spend ~30 s on that.
2. Should the EAGLE3 draft model be loaded from a symlinked path under
   `/scratch/hf-cache-patched/` or directly from `/scratch/hf-cache/`?
   No KV patch needed on the draft; can point at the original.
3. Failure of the BF16-patched approach (K2.5 schema differs): inspect
   the K2.5 `hf_quant_config.json` schema and re-do the patch in the
   right place. Do not fall back to FP8 KV for the A/B (TS arm needs
   `kv.dtype == q.dtype`). A standalone FP8 KV TRTLLM smoke is OK as a
   side-check that the recipe runs at all, but the A/B requires BF16.
4. **K2.5 for both arms throughout Phase 5** (user direction 2026-05-23).
   No K2.6 fallback. Output quality / acceptance / TS-divergence checks
   are deferred to a later phase.
