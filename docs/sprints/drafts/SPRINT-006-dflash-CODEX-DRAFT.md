# Sprint 006-dflash: DFlash Performance Investigation

> **Track suffix**: `-dflash`. This draft assumes the investigation takes the
> `006` slot and the existing `docs/sprints/SPRINT-006-dflash.md` EAGLE3 stub
> is renumbered during merge if the team agrees.

**Status**: draft
**Created**: 2026-04-29
**Sprint type**: investigation sprint
**Depends on**: `docs/sprints/SPRINT-005-dflash.md`,
`docs/sprints/SPRINT-005-FOLLOWUPS-dflash.md`
**Target hardware**: RTX 5090 (32 GB), 123 GB system RAM
**Estimated effort**: 1.5 weeks single-engineer
**Branches**:
- Repo: `sprint/006-dflash`
- Fork: `feature/sprint-004-rebase-dflash`

## Overview

Sprint 005 proved that the current DFlash stack is correct enough to run but not
good enough to justify itself. The hard failure is not subtle: median DFlashx is
below 1.0 on both the 27B and 35B-A3B targets, and the post-v8 F-014 fix chain
made the failure mode safer but slower. The next sprint should not try to ship
another speculative mode or broaden runtime surface area. It should answer, with
numbers, which of three mechanisms is dominant: the F-018 skip heuristic, wasted
16-token draft blocks on early rejection, or checkpoint/restore overhead on
hybrid targets.

This draft keeps scope deliberately narrow. It selects five experiments from the
intent's E1-E11 list: `E3`, `E5`, `E1`, `E2`, and `E4`. That set is the highest
signal per unit effort because it first measures the cost centers, then isolates
the known regression, then tests the two most plausible remediation levers. It
does not spend this sprint on EAGLE3 productization, custom draft training, or
external reference work unless the selected runs end inconclusive early.

Project conventions to preserve:
- The dflash track stays on suffixed sprint docs and its own branch chain.
- Bench comparisons keep sampling fixed at `temp=0`, `top_k=1`, `seed=42`,
  `tokens=256`.
- Fork-side changes stay small, instrumented, and explainable. The
  `llama-cpp-rq/AGENTS.md` policy means any upstreamable fork patch still needs a
  human-owned cleanup and disclosure pass later; this sprint is for local
  investigation, not upstream PR preparation.

## Use Cases

1. **Root-cause the missing uplift**: an engineer can point to measured evidence
   for whether throughput is mostly being lost to rollback overhead, low draft
   alignment, or the skip-round safety heuristic.
2. **Choose the next sprint rationally**: instead of continuing with stale
   EAGLE3 assumptions, the team can decide whether Sprint 007 should optimize
   DFlash, pivot to a different speculative mode, or stop investing in this path.
3. **Turn follow-ups into reproducible experiments**: F-016 and F-018 stop being
   narrative conclusions in sprint notes and become repeatable measurements with
   artifacts in versioned docs.
4. **Avoid overfitting to quicksort**: the sprint must explain why quicksort can
   win while the median still loses, so future work is not justified off a single
   friendly prompt.

## Architecture

The architecture of this sprint is an instrumentation loop, not a product
surface expansion.

```text
fork pin (86272e841 baseline)
  -> add timing + rejection-position instrumentation
  -> expose corrected speculative metrics to server results
  -> run controlled benchmark matrix from turbo repo
  -> store per-experiment JSON + markdown summaries
  -> aggregate findings into one decision document
  -> choose Sprint 007 branch
```

The critical control points remain the same as Sprint 005:
- `common/speculative.cpp` owns draft generation, accepted-prefix accounting,
  block-size behavior, and rollback-aware speculative state.
- `tools/server/server-context.cpp` owns checkpoint restore, accepted-prefix
  replay, and the `spec_skip_next_round` throttle.
- `tools/server/server-task.h` / `.cpp` are the narrowest place to surface
  experiment metrics back to the harness.
- `scripts/bench_speculative.py` is the canonical runner and should become the
  single place that consumes real acceptance and timing-breakdown fields.

Hybrid-target constraint stays load-bearing: `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`
means full restore plus replay is the only safe rollback path. This sprint is
therefore not trying to eliminate restore semantics; it is trying to measure when
that cost dominates and when policy choices are making it worse than necessary.

## Implementation

### Global run protocol

Every experiment phase follows the same discipline:
- Start from the same fork pin or a single isolated patch on top of it.
- Run a fast canary first on `qwen` thinking-off with `P1 quicksort` and one
  entropic prompt (`P4 Hamlet` by default).
- Promote to the full 5-prompt x 3-trial matrix only if the canary completes
  without crashes or hangs.
- Record `tok/s`, `draft_n_generated`, `draft_n_acc_tokens`, rejection position,
  checkpoint save time, checkpoint restore time, accepted-prefix replay time, and
  round count.
- Re-run the best candidate from the sprint on `qwen` thinking-on and a single
  confirmation cell on `qwen36` to check that the conclusion is not unique to one
  profile.

### Phase 1: E3 checkpoint cost profile

**Hypothesis**: checkpoint save, checkpoint restore, and accepted-prefix replay
consume a large enough fraction of speculative wall time on entropic prompts that
even a good heuristic cannot recover the lost throughput alone.

**Method**:
- Add timing spans around checkpoint save, restore, and replay in
  `tools/server/server-context.cpp`.
- Surface those spans through `result_timings`.
- Run the canary pair, then the full 5-prompt matrix on `qwen` thinking-off.
- Compare speculative timing breakdown against `qwen-target-only` wall time for
  the same prompts.

**Expected signal**:
- Quicksort shows restore/replay as a small minority of wall time.
- Entropic prompts show restore plus replay as a large and repeatable fraction of
  the speculative path.

**Decision rule**:
- If restore plus replay is `>=25%` of speculative wall time on at least two
  entropic prompts, checkpoint cost becomes a first-class Sprint 007 branch.
- If it stays `<15%` almost everywhere, treat checkpoint cost as secondary and
  focus on draft policy.

### Phase 2: E5 rejection-position profile

**Hypothesis**: poor prompts are rejecting very early inside each 16-token draft
block, so most draft work is wasted before the target can benefit.

**Method**:
- Instrument accepted-prefix length and first rejection position per draft round
  in `common/speculative.cpp`.
- Persist one compact histogram per prompt in experiment artifacts.
- Run the full `qwen` thinking-off matrix using the Phase 1 instrumentation
  build.

**Expected signal**:
- Quicksort clusters toward late-block acceptance.
- Entropic prompts cluster at rejection positions `0-2` or accepted prefixes of
  `0-2`.

**Decision rule**:
- If `>=70%` of rejected rounds on at least two entropic prompts fail by token
  `2`, smaller or adaptive block size moves to the top of the remediation list.
- If rejection positions are broadly distributed, block size is not the primary
  lever.

### Phase 3: E1 skip-flag-off canary

**Hypothesis**: the v8 `spec_skip_next_round` policy explains a large share of the
post-fix slowdown, but disabling it outright reintroduces the deterministic
partial-acceptance loop on hostile prompts.

**Method**:
- Add an experiment toggle to disable `spec_skip_next_round`.
- Run only the canary pair first, with watchdog timeout and a counter for
  repeated same-position partial acceptances.
- Promote to a broader run only if the entropic canary does not loop.

**Expected signal**:
- Quicksort throughput improves materially.
- The entropic prompt either hangs, times out, or shows repeated identical
  partial positions.

**Decision rule**:
- If quicksort recovers `>=25%` tok/s versus the v8 baseline and the entropic
  prompt shows loop behavior, F-018 is confirmed as a real regression but also a
  necessary safety mechanism.
- If throughput barely changes, stop treating the skip flag as the dominant cost.

### Phase 4: E2 `DRAFT_N_MAX` sweep

**Hypothesis**: `16` is too large for the current draft quality, and a smaller
block reduces wasted work enough to improve median throughput even if nominal
acceptance falls.

**Method**:
- Sweep `DRAFT_N_MAX` across `4`, `8`, and `16`.
- Keep all other settings fixed and use the corrected real-acceptance metrics.
- Run the full matrix on `qwen` thinking-off.

**Expected signal**:
- Quicksort prefers `8` or `16`.
- Entropic prompts improve meaningfully at `4` or `8` because failed blocks are
  cheaper to verify and replay.

**Decision rule**:
- If `8` beats `16` by `>=15%` median tok/s on the 5-prompt set, a smaller or
  adaptive block size becomes the default Sprint 007 plan.
- If all smaller settings are flat or worse, block size tuning is not enough.

### Phase 5: E4 adaptive skip heuristic

**Hypothesis**: skipping only after repeated partial acceptances at the same
position preserves the F-014 correctness fix while recovering throughput lost to
the current always-skip policy.

**Method**:
- Replace the unconditional skip policy with a bounded heuristic such as "skip
  only after `K` consecutive partials at the same rejection position", starting
  with `K=2` and `K=3`.
- Reuse the rejection-position counters from Phase 2.
- Run canary, then the full `qwen` thinking-off matrix, then a 50-request soak on
  the winning setting.
- If the result is promising, run one confirmation sweep on `qwen` thinking-on
  and one confirmation cell on `qwen36`.

**Expected signal**:
- No return of infinite restore loops.
- Better tok/s than the v8 baseline, especially on prompts that currently fall to
  one-token cycles after a single partial.

**Decision rule**:
- If the heuristic improves median `qwen` tok/s by `>=20%` versus the v8
  baseline with zero hangs in the soak run, Sprint 007 should productize and
  republish around this fix.
- If it is safe but gains `<10%`, treat it as a local patch rather than the next
  sprint's center of gravity.

## Sprint 007 Recommendation Decision Tree

```text
Start with Sprint 006 findings
|
+-- Adaptive skip heuristic clears safety + >=20% median gain?
|   |
|   +-- Yes -> Sprint 007 = productize F-018 fix
|   |          - land adaptive heuristic cleanly
|   |          - wire corrected F-016 metrics into published bench docs
|   |          - rerun canonical L4 and decide if DFlash remains default-worthy
|   |
|   +-- No
|
+-- Rejection profile shows early failures (0-2 tokens) and DRAFT_N_MAX 8 beats 16?
|   |
|   +-- Yes -> Sprint 007 = dynamic block-size / rejection-aware DFlash
|   |          - shrink blocks on hostile prompts
|   |          - keep quicksort-friendly larger blocks where they still win
|   |
|   +-- No
|
+-- Checkpoint restore + replay is >=25% of wall time on entropic prompts?
|   |
|   +-- Yes -> Sprint 007 = rollback-cost reduction
|   |          - optimize checkpoint cadence and replay policy
|   |          - investigate whether full-state restore makes DFlash structurally weak
|   |
|   +-- No
|
+-- None of the above recover useful throughput?
|   |
|   +-- Draft/target mismatch still looks structural -> Sprint 007 = EAGLE3 or
|   |   alternative draft-path investigation
|   |
|   +-- Broadly negative result -> Sprint 007 = declare current DFlash path
|       non-competitive for this stack and stop spending sprint-sized effort on it
```

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `../llama-cpp-turboquant/common/speculative.cpp` | Modify | Add rejection-position logging, experiment toggles, block-size and adaptive-skip behavior |
| `../llama-cpp-turboquant/tools/server/server-context.cpp` | Modify | Time checkpoint save/restore/replay and host the skip-policy experiments |
| `../llama-cpp-turboquant/tools/server/server-task.h` | Modify | Extend timing/result schema for real speculative diagnostics |
| `../llama-cpp-turboquant/tools/server/server-task.cpp` | Modify | Serialize experiment metrics to API-visible result payloads |
| `scripts/bench_speculative.py` | Modify | Consume corrected metrics, run canary/full matrices, emit per-experiment summaries |
| `docs/sprints/SPRINT-006-dflash-experiments/` | Create | Store JSON outputs and one markdown summary per experiment run |
| `docs/sprints/SPRINT-006-dflash-FINDINGS.md` | Create | Aggregate hypothesis verdicts and the Sprint 007 recommendation |

## Definition of Done

- All five selected experiments (`E3`, `E5`, `E1`, `E2`, `E4`) have a written
  run record with code SHA, command line, prompt set, and numerical outcome.
- `scripts/bench_speculative.py` consumes the real acceptance counters from
  F-016 rather than the misleading post-truncation metric.
- The sprint produces one findings document that labels each top hypothesis
  `confirmed`, `refuted`, or `inconclusive`.
- The team can state how much of the observed slowdown comes from each of:
  skip policy, early block rejection, and checkpoint restore/replay cost.
- Sprint 007 has a written recommendation branch, not just a preference.
- At least one remediation candidate is either promoted with evidence or ruled
  out with evidence.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Investigation scope balloons into feature work | High | High | Fix the sprint to five experiments and require a decision after Phase 5 before adding E6-E11 |
| Instrumentation perturbs the very performance being measured | Medium | High | Keep timers and counters minimal, validate one baseline rerun with instrumentation compiled in but all toggles off |
| Local-only fork commits create reproducibility drift | High | Medium | Either push the current fork chain before Phase 1 or freeze a documented local Docker override and record it in every artifact |
| Prompt-specific instability, especially P3, muddies conclusions | Medium | Medium | Use canary prompts for iteration and treat P3 as confirmatory rather than blocking if transport noise returns |
| Adaptive heuristic fixes one profile but not the other | Medium | Medium | Keep `qwen` as the main gate and use `qwen36` as confirmation so the sprint does not overclaim universality |
| The fork's local experiments are not upstream-ready under `llama-cpp-rq` policy | High | Low | Treat all fork patches here as local diagnostics; any upstream attempt requires later human-authored reshaping and disclosure |

## Security

- Debug instrumentation must not dump full prompt or completion text into
  committed artifacts; store counts, timings, rejection positions, and prompt IDs
  only.
- Experiment toggles must remain opt-in and local. No debug mode should become
  the default server path.
- Timing and metric schema changes should stay bounded to speculative diagnostics
  so they do not accidentally expose unrelated server internals through the API.

## Dependencies

- The current local fork chain through `86272e841`, plus either a pushed remote
  ref or a documented local build override.
- `scripts/bench_speculative.py` updated to consume `draft_n_generated` and
  `draft_n_acc_tokens`.
- Stable access to the canonical 5-prompt benchmark set and the existing
  `qwen`, `qwen-target-only`, and `qwen36` profiles.
- GPU time on the RTX 5090 for repeated 3-trial sweeps and soak runs.
- Continued adherence to the dflash-track branch and document naming
  conventions.

## Open Questions

1. Should this investigation formally take the `006` slot, with the current
   EAGLE3 stub renumbered, or should the merged sprint become `007` instead?
2. Is a single `qwen36` confirmation cell enough for this sprint, or does the
   team want a full MoE confirmation sweep on the winning variant?
3. Should the `DRAFT_N_MAX` sweep stay at `4/8/16`, or is there enough value in
   testing `2` or `12` to justify the extra matrix cost?
4. What is the minimum outcome that counts as a successful remediation signal for
   Sprint 007 planning: any positive median gain, `>=1.0x`, or something closer
   to the original `>=1.3x` ambition?
5. If the five selected experiments still leave the cause ambiguous, should the
   first extension be `E6` offline draft-vs-target alignment or `E7` EAGLE3 as an
   alternative baseline?
