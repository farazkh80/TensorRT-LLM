# Kimi K2.6 TokenSpeed MLA — Run Book

End-to-end runbook for evaluating the TokenSpeed MLA decode kernel against
the FlashInfer / trtllm-gen baseline on Kimi K2.6 (= K2-Thinking architecture)
NVFP4. This is the hybrid spike path Rajeev signed Faraz onto per the email
thread (2026-05-13): use the spike's env-var swap to get a measurement
quickly; if the JIRA-claimed wins reproduce (`~9%` bs=1, `~11%` @ 100 TPS/user),
invest in landing `TokenSpeedMLAAttention(TrtllmAttention)` per the design
draft. If they don't, the spike's measurement is itself the result.

Background and design context in `understanding.md` (this directory).

## Host context

- 8× B300 SXM6 sm_103, 275 GB HBM each (verified 2026-05-19).
- Scratch: `/home/scratch.fkhoubsirat_coreai`, 3 TB, fresh.
- TensorRT-LLM source: `/home/farazkh_scratch/parallel/TensorRT-LLM` on
  branch `tokenspeed-kimik25-eval-public`.
- TokenSpeed source clone (reference only): `/home/farazkh_scratch/tokenspeed`.
- Container: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14` (already pulled).

## Directory layout (this folder)

```
.claude_docs/tokenspeed-kimik25/
├── understanding.md              JIRA + email + spike summary + risk register
├── bench-*.yml                   Eleven experiment configs (the grid; see §7
│                                 of understanding.md)
├── code/
│   ├── tokenspeed_mla.py         Wrapper (drop-in for FlashInfer MLA decode)
│   └── test_tokenspeed_mla.py    Parametrized parity test vs FlashInfer
├── scripts/
│   ├── apply_patches.py          Inline patches to rc14 container
│   ├── setup_k26_container.sh    Phase 2: container bring-up
│   ├── verify_kernel_swap.sh     Phase 3: nsys go/no-go gate
│   └── run_bench.sh              Phase 4: A/B runner (one config at a time)
└── runbook.md                    This file
```

## Phases

### Phase 0 — workspace prep (done)

```bash
mkdir -p /home/scratch.fkhoubsirat_coreai/{hf-cache,runs/k2.6-spike,logs}
```

### Phase 1 — checkpoint download (in progress)

```bash
# Background-launched; resumable via snapshot_download cache layout.
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -u \
    /home/scratch.fkhoubsirat_coreai/runs/k2.6-spike/download_k26.py \
    > /home/scratch.fkhoubsirat_coreai/runs/k2.6-spike/download.log 2>&1 &

# Repo:   nvidia/Kimi-K2-Thinking-NVFP4
# Size:   ~553 GB, 133 files
# Target: /home/scratch.fkhoubsirat_coreai/hf-cache/
# Resolved path (snapshot):
#   /home/scratch.fkhoubsirat_coreai/hf-cache/models--nvidia--Kimi-K2-Thinking-NVFP4/snapshots/<sha>/
```

Verify completion: `tail -3 .../download.log` shows `done in NN.N min`.

### Phase 2 — container bring-up (~3 min)

```bash
cd /home/farazkh_scratch/parallel/TensorRT-LLM/.claude_docs/tokenspeed-kimik25/scripts
./setup_k26_container.sh
```

This:
- Removes any existing `tokenspeed-spike-k26` container.
- Starts a new one against rc14 with the scratch mount, host source, and
  TokenSpeed clone bind-mounted.
- `pip install tokenspeed-mla` inside the container.
- Copies `code/tokenspeed_mla.py` into the container's site-packages.
- Runs `apply_patches.py` to add the `TOKENSPEED_MLA` registry entry and
  the `TLLM_TOKENSPEED_MLA=1` env-var swap inside `run_mla_generation`.
- Smoke-tests the import + selector.

Idempotent — safe to re-run after edits.

### Phase 3 — kernel-swap verify (~5 min) ★ THE GATE — **RESULT: STOP**

```bash
./verify_kernel_swap.sh
```

Runs `minimal_generate.py` twice under `nsys profile`, once baseline
(`TLLM_TOKENSPEED_MLA=0`) and once variant (`TLLM_TOKENSPEED_MLA=1`), both
with `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1`. Diffs the kernel name sets
between the two traces. (Switched from `quickstart_advanced.py` after the
NVIDIA Kimi-K2-Thinking-NVFP4 snapshot's custom tokenizer silently hung
at `AutoTokenizer.from_pretrained` even with `--trust_remote_code`.
`minimal_generate.py` uses `skip_tokenizer_init=True` and a dummy
token-id prompt — we only need 32 generated tokens for kernel-symbol
capture.)

Three outcomes:

| Outcome | Verdict | Action |
|---|---|---|
| **Kernel sets differ; variant has TokenSpeed symbols** | Swap fires. | Proceed to Phase 4. |
| **Kernel sets identical** | Swap silent — repeats spike step 6's DSV3-Lite finding. K2.6 also bypasses `run_mla_generation`. | Stop. Spike patches are dead code; jump to `TokenSpeedMLAAttention` backend class. |
| **Crash / OOM / shape bug** | K2.6-specific. | Debug. |

**Result on K2.6 NVFP4 / B300 sm_103 / TP4 (2026-05-19):** kernel sets
**identical** (9 unique kernels per trace, 238,090 `cudaLaunchKernel`
calls each, generation latency 22.7 vs 23.6 tok/s = noise). The env-var
swap inside `run_mla_generation` is dead code on K2.6 default config,
same as DSV3-Lite step 6. Full evidence in `understanding.md` §8.

**Phases 4 and 5 are skipped** — the bench grid would produce noise on
the same dead-code path. The only viable next step is the backend-class
implementation, paused for leadership sign-off (see §6 below).

### Phase 4 — bench grid (SKIPPED — Phase 3 result blocks)

The bench grid configs (`bench-*.yml` in this directory) remain valid as
the experimental plan for the backend-class follow-up. They are NOT to be
run against the current spike-patches setup because Phase 3 showed both
base and variant traverse the same dead-code path — the resulting numbers
would all be noise.

When the `TokenSpeedMLAAttention` backend class lands, re-enable Phase 4
by re-running these configs with `--attention_backend TOKENSPEED_MLA` on
the variant side instead of the env-var swap. `scripts/run_bench.sh` will
need a small update to support that toggle.

### Phase 5 — decide (Phase 3 already decided the next step)

The hybrid criterion's three branches were:

1. **Wins reproduce ≥ JIRA numbers**: commit spike code, file backend class
   implementation, push for sign-off.
2. **Wins marginal (<5%)**: document & close.
3. **Wins zero / negative**: re-check, or escalate to backend class.

**Result: branch 3.** Phase 3 ran and the kernel sets are identical
between base and variant — same finding as DSV3-Lite step 6. The spike's
env-var swap doesn't intercept K2.6's MLA decode path. The next step is
the `TokenSpeedMLAAttention(TrtllmAttention)` backend class implementation
per `../tokenspeed-mla-dsv3-lite/design-tokenspeed-attn-backend.md`.

**Paused for leadership sign-off** before starting the ~1-day
implementation — see `understanding.md` §8 for the sign-off targets
(Rajeev / Sharan / June / Albert / Julien) and what each needs to confirm.

## Known risks (numbered to align with `understanding.md` §7 register)

| # | Risk | First-touch |
|---|---|---|
| 1 | Checkpoint MTP layers (need ≥3 native for MTP=3 configs) | Phase 4 step 5 |
| 2 | Spec-decode parity divergence (max abs 0.33 / max rel ~1166×) | Phase 4 step 5 |
| 3 | KV cache pressure × draft chain at long context | Phase 4 step 8 |
| 4 | **rc14 `TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1` × `trtllm-bench` shape bug** | Phase 4 step 1 |
| 5 | Prefill cost dominates → synth dataset under-reports TS | Phase 4 step 8 |
| 6 | KV reuse validity (verify reuse stats > 0 for multi-turn) | Phase 4 step 9 |
| 7 | P–D balance (disagg only) | Phase 4 disagg |
| 8 | KV transfer cost (disagg only) | Phase 4 disagg |
| 9 | Multi-process disagg launch infra | Phase 4 disagg |
| 10 | Host RAM for CPU KV offload (64 GiB × 8 ranks) | Phase 4 step 11 |
| 11 | PCIe / NVLink-C2C bandwidth for offload churn | Phase 4 step 11 |
| 12 | TS × host cache thrashing | Phase 4 step 11 |

Risk #4 (rc14 trtllm-bench bug) is the most likely to bite early: spike
step 7 showed both base and ts failing with shape `-1023` at ISL=1024.
If that recurs on K2.6, two workarounds:
- Pull rc15+ container, reverify `apply_patches.py` anchors, rerun.
- Stay on rc14, fall back to per-iter nsys measurement on
  `quickstart_advanced.py` like spike step 6 v3 — kernel-time A/B only,
  no TPOT/throughput numbers but enough for the hybrid go/no-go signal.

## TS env-var convention

For every command in Phases 3+, the swap is controlled by **two** env vars
held in lockstep:

```
TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1   # force FlashInfer-Python MLA path
TLLM_TOKENSPEED_MLA={0|1}              # 0=baseline, 1=variant
```

`TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION` defaults to 0 in rc14, which routes
MLA decode through the C++ thop direct cubin path — bypassing the spike's
Python-level swap entirely (spike step 6 finding). Both vars are required
for any swap to fire.

For the disagg config (`bench-60k1k_disagg_p4d4_conc1_mtp3.yml`), the TS
env vars go ONLY on the D-side worker process — never on P. P only does
prefill; the MLA decode kernel never fires there.

## References

- `understanding.md` — JIRA / email / spike summary + risk register.
- `code/tokenspeed_mla.py` — wrapper source, deployed inside the container.
- `scripts/apply_patches.py` — what the patches actually change.
- `../tokenspeed-mla-dsv3-lite/` — original spike's nsys traces (1.2 GB);
  step / summary docs were deleted after the K2.5 branch was cut but are
  also captured in `understanding.md`.
- Slack TRTLLM-12510 thread.
- Email thread "TokenSpeed MLA Kernel" (2026-05-09 → 2026-05-13).
