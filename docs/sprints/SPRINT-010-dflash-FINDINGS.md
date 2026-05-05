# Sprint 010-dflash Findings

**Date**: 2026-05-05
**Status**: COMPLETE — patch ships safely opt-in (default 0 = no change);
hypothesis on prose median lift did NOT hold for thinking-off bench.

## Headline

**p_min draft truncation does not unlock the prose acceptance gap on
the thinking-off bench.** It's safe (default `DRAFT_P_MIN=0` = current
behavior), but adjusting it doesn't move the median in the direction we
hoped:

| Config | Quicksort× | Pythagorean× | DC× | Hamlet× | SQL× | Median | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 N=4 (Sprint 008) | **2.22** | 1.15 | 0.60 | 0.53 | 1.02 | 1.02 | 0.53 |
| **p_min=0.3 N=4** | 2.21 | 1.15 | 0.61 | 0.34 | 1.01 | 1.01 | 0.34 |
| **p_min=0.5 N=4** | 2.22 | 1.15 | 0.61 | 0.53 | 1.02 | 1.02 | 0.53 |
| **p_min=0.75 N=4** | 2.18 | 1.15 | 0.61 | **0.34** | 1.01 | 1.01 | 0.34 |
| BF16 N=2 (default) | 1.41 | 1.23 | 1.03 | 0.91 | 1.21 | **1.21** | **0.91** |

p_min=0.5 is essentially indistinguishable from BF16 N=4 baseline. p_min
≥0.75 actively hurts Hamlet (0.53 → 0.34, acceptance 16% → 0%).

## What we did

1. **Surveyed `spiritbuun/buun-llama-cpp` fork** (their public llama.cpp
   tree). Key DFlash-relevant commits: `cab1fb5` (p_min + adaptive draft
   length), `683c5ac` (upstream PR #22506 for autoregressive p_min),
   `a45cdda` (GPU cross-attention ring buffer), `ff0444e` (multi-slot
   batched draft). Cherry-picking the full chain would require ~10
   prior infrastructure commits.
2. **Implemented a minimal p_min patch** in our fork's
   `common_speculative_state_dflash::draft()` (5 lines, fork commit
   `5833a843c`). Skips check on i==1 so we always commit one draft.
3. **Wired `DRAFT_P_MIN` env** through `docker/entrypoint.sh` and
   `docker-compose.yml` → `--draft-p-min` CLI flag (already in
   upstream's arg.cpp).
4. **Smoke test (thinking-ON Hamlet)**: 3-of-3 trials at p_min=0.5 N=4
   showed **51.8 tps vs 37 tps baseline (+40%)**, **31.4% acceptance vs
   15.6% baseline (~doubled)**. Looked like a clear win.
5. **Full bench (thinking-OFF, all 5 prompts)**: results above.
   Hypothesis did NOT replicate on the no-think regime.

## Why the hypothesis didn't hold

The smoke test win on thinking-ON was illusory in the sense it didn't
generalize. Mechanism:

- p_min checks the **draft model's top-1 probability** for each
  position the diffusion produced. If `top-1 < p_min`, stop drafting.
- This catches **hesitantly wrong** drafts (low confidence + wrong).
- It does NOT catch **confidently wrong** drafts (high confidence +
  wrong) — those are the failure mode on entropic prose.

The DFlash draft is a small (1.7B) model. On thinking-OFF prose
prompts (Hamlet, DC trip), the draft confidently picks plausible-
looking tokens that the larger target rejects. The draft's top-1
probability stays high; p_min never fires; no truncation.

On thinking-ON prompts the `<think>...</think>` content has different
entropy patterns — more boilerplate reasoning steps where confident
predictions actually match the target's. There p_min's truncation
helps (the smoke test data). But thinking-off is what most operators
care about (~1.8× throughput uplift from disabling thinking, per
spiritbuun's README).

## Hypothesis verdicts

| Hypothesis | Verdict |
|---|---|
| p_min truncates entropic-tail drafts → cheaper verify batch | ✅ Mechanically works (verified via smoke test thinking-ON) |
| Higher p_min → better Hamlet acceptance | ❌ Refuted (0.53 → 0.34 at p_min=0.75) |
| Sprint 005 ≥1.3× hard gate clears | ❌ Median stays at 1.02 (N=4) / 1.21 (N=2 default) |
| Optimal N moves up with p_min (parallelism + only-keep-confident) | ❌ Refuted at N=4; not worth testing N=8/16 |

## Status

- **Default unchanged**: `DRAFT_P_MIN=0` (current behavior, no
  truncation). BF16 N=2 stays the docker-compose default with median
  1.21×.
- **Patch ships opt-in**: operators with thinking-on workloads can
  set `DRAFT_P_MIN=0.5` to potentially gain on prose (the smoke test
  showed +40% on thinking-on Hamlet). Document this regime in README.
- **F-030 closed**: the patch is in fork; we know what it does and
  doesn't do.

## What this tells us about the DFlash track

After Sprints 008/009/010, the picture is clear:

1. **Pipeline is solved** (Sprint 008: VRAM-shadow ckpt, ~1% wallclock
   save tax).
2. **Quantization is at its ceiling** (Sprint 009: Q8 ≈ +5% over BF16).
3. **Per-position confidence truncation doesn't help thinking-off**
   (Sprint 010: this finding).

The remaining gap to ≥1.3× median is **draft model quality on prose**.
Solutions left:
- A different/smaller/domain-tuned **draft model** (distillation).
- A different **draft architecture** that's structurally cheaper per
  draft (EAGLE3 — but the user explicitly rejected this in Sprint 009
  closure as abandoning the diffusion thesis).
- **Cherry-pick spiritbuun's full DFlash improvements** (~10-20
  commits): adaptive n_draft based on tracked acceptance rate, GPU
  argmax + ring buffer, multi-slot batched draft. Adaptive n_draft is
  the closest to a fundamental fix because it adapts the speculative
  budget per-prompt rather than per-position.

## Sprint 011 recommendation

**Cherry-pick spiritbuun's DFlash adaptive draft length + GPU
ring-buffer improvements.** This combines:
- Their `cab1fb5` adaptive n_draft (track acceptance rate, reduce N
  after 3 consecutive low-acceptance rounds, recover gradually) —
  this DOES address confidently-wrong drafts because it monitors
  acceptance, not confidence.
- Their `a45cdda` GPU cross-attention ring buffer (cleaner than our
  F-024 fix; might subsume it).
- Possibly their `ff0444e` multi-slot batched draft (orthogonal but
  addresses our single-slot limitation).

Estimated 3-5 days fork integration work. Higher leverage than p_min
for the same code real estate.

## Definition of Done

- ✅ Sprint 010 patch landed in fork (commit `5833a843c`).
- ✅ `DRAFT_P_MIN` env wired in entrypoint.sh + docker-compose.yml.
- ✅ Smoke test passes (3-of-3 Hamlet thinking-ON, +40% tps).
- ✅ Full sweep at p_min ∈ {0.3, 0.5, 0.75} × N=4 completed.
- ✅ Comparison table written.
- ✅ Findings doc with hypothesis verdicts and Sprint 011 rec.
- ⏭ Default unchanged (Sprint 010 ships as opt-in env knob).

## Reproduction

```bash
# Smoke test (thinking-ON wins)
LLAMA_SPEC_VRAM_CKPT=1 DRAFT_N_MAX=4 DRAFT_P_MIN=0.5 make run-qwen
# (curl with default chat template → enable_thinking=true)

# Bench (thinking-OFF doesn't move)
docker compose --profile qwen down
PROFILE=qwen NO_THINK=1 LLAMA_SPEC_VRAM_CKPT=1 \
  ./scripts/run_sprint008_experiment.sh \
    ../SPRINT-010-dflash-experiments/p_min-N4-0.5 \
    DRAFT_N_MAX=4 LLAMA_SPEC_VRAM_CKPT=1 DRAFT_P_MIN=0.5
```

Artifacts: `docs/sprints/SPRINT-010-dflash-experiments/p_min-N4-{0.3,0.5,0.75}/`
