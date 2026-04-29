# Sprint 006-dflash: DFlash Performance Investigation

> **Track suffix**: `-dflash`. The dflash track does not merge to `main`
> (parallel vLLM-substrate Sprint 004/005 numbering conflict on `main`);
> the chain stays self-contained on `sprint/004-dflash` →
> `sprint/005-dflash` → `sprint/006-dflash` → ...

**Status**: Planning (2026-04-29)
**Sprint type**: Investigation + targeted code experiments
**Created**: 2026-04-29
**Depends on**: SPRINT-005-dflash (Phase 1 numbers), SPRINT-005-FOLLOWUPS-dflash.md (F-018, F-016, F-014 perf regression)
**Target hardware**: RTX 5090 (32 GB), 123 GB system RAM
**Estimated effort**: 1.5 weeks single-engineer
**Branches**:
- Repo: `sprint/006-dflash` (off `sprint/005-dflash`)
- Fork: `feature/sprint-004-rebase-dflash` at `86272e841` (LOCAL — see "Fork push" below)

---

## Overview

Sprint 005 proved that the current DFlash stack is correct enough to run
but not good enough to justify itself. The hard failure isn't subtle:
median DFlash× is below 1.0 on both the 27B and 35B-A3B targets in both
thinking-on and thinking-off regimes, and the post-v8 F-014 fix chain
made the failure mode safer but slower (51 tok/s on qwen quicksort
think-off, vs the 125 tok/s pre-v8 number that came from the buggy
loop). The next sprint should not ship another speculative variant or
broaden runtime surface area. It should answer, with numbers, which of
three mechanisms is dominant:

1. The F-018 `spec_skip_next_round` heuristic always firing on every
   partial acceptance.
2. Wasted 16-token draft blocks on early rejection.
3. Checkpoint save / restore / replay overhead on hybrid targets.

Five experiments in order: measure cost centers (Phase 1: E3), profile
the rejection pattern (Phase 2: E5), isolate the known regression
(Phase 3: E1 skip-off canary), test block-size remediation (Phase 4:
E2 DRAFT_N_MAX sweep), and test the policy fix (Phase 5: E4 adaptive
skip heuristic). Each phase has a written decision rule that names
what counts as a confirmed / refuted / inconclusive outcome.

**This sprint does not commit to shipping any fix.** The output is a
findings document and a Sprint 007 recommendation. If a remediation
candidate emerges with strong evidence, Sprint 007 productionizes it.

### GPU sequencing

The user's risk-embedder is training on the RTX 5090. Sprint 006 is
structured so **Phase 0 runs without GPU** — instrumentation,
metric-plumbing, harness updates, baseline doc review. Phases 1–5
require GPU and wait for training to release the device.

### Project conventions to preserve

- The dflash track stays on suffixed sprint docs and its own branch chain.
- Bench comparisons keep sampling fixed at `temp=0`, `top_k=1`,
  `seed=42`, `tokens=256` (Sprint 005 conventions).
- Fork-side changes stay small, instrumented, and explainable. Any
  upstreamable fork patch needs a separate human-owned cleanup pass
  later; this sprint is for local investigation, not upstream PR
  preparation.

### Fork push

Fork commits `5f58c0d81..86272e841` on
`feature/sprint-004-rebase-dflash` (the F-014 v3..v8 chain + F-016
metric) are **local only** in `/home/ravi/repos/llama-cpp-turboquant/`.
1Password's interactive signing requirement blocks pushes from
non-interactive shells. Sprint 006 does **not** block on the push —
the repo retains `docker/Dockerfile.local` as a build-from-local-checkout
override (used during Sprint 005 F-014 iterations). When the user
pushes from their interactive shell, Sprint 006 (or 007) should bump
the canonical `docker/Dockerfile` pin to `86272e841` and retire the
override.

---

## Use Cases

1. **Root-cause the missing uplift**: an engineer can point to measured
   evidence for whether throughput is mostly being lost to rollback
   overhead, low draft alignment, or the skip-round safety heuristic.
2. **Choose the next sprint rationally**: instead of continuing on
   stale assumptions (the EAGLE3 stub, originally Sprint 006, was
   conditional on Sprint 005's gate passing — it didn't), the team
   can decide whether Sprint 007 should optimize DFlash, pivot to
   another speculative path, or stop investing here.
3. **Turn follow-ups into reproducible experiments**: F-016 and F-018
   stop being narrative conclusions in sprint notes and become
   repeatable measurements with versioned artifacts.
4. **Avoid overfitting to quicksort**: the sprint must explain why
   quicksort can win while the median still loses, so future work
   isn't justified off a single friendly prompt.

---

## Architecture

```text
fork pin (86272e841 baseline, LOCAL)
  ↓
Phase 0: instrumentation + plumbing + harness updates (NO GPU)
  ↓
Phase 1 (E3): timing instrumentation around checkpoint save/restore/replay
  ↓
Phase 2 (E5): rejection-position histogram instrumentation
  ↓
Phase 3 (E1): skip-flag-off canary (does the loop come back?)
  ↓
Phase 4 (E2): DRAFT_N_MAX sweep at 4 / 8 / 16
  ↓
Phase 5 (E4): adaptive skip-flag heuristic (skip after K consecutive)
  ↓
Aggregate findings → SPRINT-006-dflash-FINDINGS.md
  ↓
Sprint 007 recommendation (decision tree below)
```

The critical control points stay where Sprint 005 left them:

- **fork** `common/speculative.cpp` — DFlash impl (begin/rollback/draft/accept),
  block-decode loop, accumulated_ctx growth.
- **fork** `tools/server/server-context.cpp` — slot loop, partial-acceptance
  restore, checkpoint save/restore (~150 MiB GPU memcpy/round),
  `spec_skip_next_round` flag.
- **fork** `tools/server/server-task.{h,cpp}` — narrowest place to surface
  experiment metrics back to the harness.
- **repo** `scripts/bench_speculative.py` — canonical runner; should
  become the single place that consumes real acceptance + timing-breakdown
  fields.

Hybrid-target constraint stays load-bearing: hybrid Qwen3.6 contexts
require `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`, so full-state restore +
replay is the only safe rollback. This sprint isn't trying to eliminate
restore; it's trying to measure when its cost dominates and when policy
is making it worse than necessary.

---

## Implementation

### Global run protocol

Every experiment phase follows the same discipline:

- Start from the same fork pin or a single isolated patch on top of it.
- Run a fast canary first on `qwen` thinking-off with `P1 quicksort`
  and one entropic prompt (`P4 Hamlet` by default).
- Promote to the full 5-prompt × 3-trial matrix only if the canary
  completes without crashes or hangs.
- Record `tok/s`, `draft_n_generated`, `draft_n_acc_tokens`, rejection
  position (Phase 2+), checkpoint save/restore/replay times (Phase 1+),
  round count.
- Re-run the best candidate on `qwen` thinking-on and a single
  confirmation cell on `qwen36` to check that the conclusion isn't
  unique to one profile.
- 3 trials per cell; report median; flag any cell where stddev/median
  > 5% as suspicious.

### Phase 0 — Instrumentation + harness (NO GPU)

**Goal**: Prepare all code changes, metric plumbing, and harness
consumption updates so Phases 1–5 are pure bench runs. This phase
runs while the GPU is busy with risk-embedder training.

**Tasks**:

- [ ] **Bench harness consumes F-016 metric**: update
  `scripts/bench_speculative.py::parse_draft_stats` to read
  `timings.draft_n_generated` / `timings.draft_n_acc_tokens` from
  the response (added in fork commit `a2a6168ea`). Fall back to
  the post-truncation `draft_n` / `draft_n_accepted` fields on
  older fork pins. Switch the harness's `acceptance_rate` field
  to the real one when available.
- [ ] **Republish Sprint 005 numbers** in
  `docs/BENCHMARK-REPORT.md` § Sprint 005, with corrected
  acceptance rate. The original `100%` figure was a metric bug;
  the table should show real values (1–25% range).
- [ ] **Instrumentation patches authored** (not yet built):
  - Fork `common/speculative.cpp`: rejection-position counters
    on accepted-prefix length per draft round (Phase 2 needs this).
  - Fork `tools/server/server-context.cpp`: timing spans around
    checkpoint save (`server_get_checkpoint`), restore
    (`llama_state_seq_set_data_ext`), and accepted-prefix replay
    (Phase 1 needs this).
  - Fork `tools/server/server-task.{h,cpp}`: extend
    `result_timings` schema with the new diagnostic fields.
  - Adaptive skip-flag scaffolding on `server_slot`: counter for
    consecutive partials at same `pos_max`, threshold-driven flag
    (Phase 5 needs this; landed gated behind a runtime env so
    Phase 3 can disable the original flag without removing it).
- [ ] **Bench harness emits per-experiment summary MD**: new
  flag `--experiment-name N` that writes results to
  `docs/sprints/SPRINT-006-dflash-experiments/EN/results.json` +
  `summary.md`. One-line addition.
- [ ] **No-instrumentation baseline rerun** (deferred to Phase 1
  start, requires GPU): documented as a precondition.
- [ ] **Phase 0 review**: walk Sprint 005's bench code + fork's
  speculative.cpp + server-context.cpp once more with the
  hypothesis list in hand. Note any code-level observations that
  could refine the experiment design before GPU spending starts.

**Phase gate**: All Phase 1–5 instrumentation patches authored as
local commits or pending diffs on the fork. Bench harness can
parse new fields. Output dir scaffolded.

### Phase 1 — E3 checkpoint cost profile (GPU)

**Hypothesis**: checkpoint save, checkpoint restore, and
accepted-prefix replay consume a large enough fraction of speculative
wall time on entropic prompts that even a good heuristic cannot
recover the lost throughput alone.

**Method**:

- Build the fork with Phase 0's timing-span patches.
- Confirm a no-instrumentation baseline first: rebuild without
  the timing patches, rerun canary, confirm tok/s within 5% of
  Sprint 005's post-v8 numbers (rules out the
  "instrumentation-perturbed" risk).
- Rebuild with timing patches, run canary, then full 5-prompt
  matrix on `qwen` thinking-off.
- Compare speculative timing breakdown to `qwen-target-only` wall
  time for the same prompts.

**Expected signal**:

- Quicksort shows restore/replay as a small minority of wall time.
- Entropic prompts show restore + replay as a large and repeatable
  fraction of the speculative path.

**Decision rule**:

- If restore + replay ≥ 25% of speculative wall time on at least
  two entropic prompts → checkpoint cost becomes a first-class
  Sprint 007 branch.
- If it stays < 15% almost everywhere → checkpoint cost is
  secondary; focus on draft policy.

**Artifact**: `experiments/E3_checkpoint_cost/results.json` +
`summary.md`.

### Phase 2 — E5 rejection-position profile (GPU)

**Hypothesis**: poor prompts reject very early inside each 16-token
draft block, so most draft work is wasted before the target can
benefit.

**Method**:

- Build with Phase 0's rejection-position counter.
- Persist one compact histogram per prompt (rejection position
  distribution; accepted-prefix length distribution).
- Run the full `qwen` thinking-off matrix using the Phase 1
  build (timing + position counters both compiled in).

**Expected signal**:

- Quicksort clusters toward late-block acceptance.
- Entropic prompts cluster at rejection positions 0–2 or
  accepted-prefix lengths 0–2.

**Decision rule**:

- If ≥ 70% of rejected rounds on at least two entropic prompts
  fail by token 2 → smaller / adaptive block size goes to top
  of remediation list.
- If rejection positions are broadly distributed → block size
  is not the primary lever.

**Artifact**: `experiments/E5_rejection_profile/results.json` +
`summary.md` (with histogram tables).

### Phase 3 — E1 skip-flag-off canary (GPU)

**Hypothesis**: the v8 `spec_skip_next_round` policy explains a
large share of the post-fix slowdown, but disabling it outright
reintroduces the deterministic partial-acceptance loop on hostile
prompts.

**Method**:

- Phase 0's adaptive-skip scaffolding includes a
  `LLAMA_SPEC_DISABLE_SKIP=1` env that bypasses the flag entirely.
- Run only the canary pair first, with watchdog timeout (30s
  max per request) and a counter for repeated same-position
  partial acceptances (added in Phase 0).
- Promote to a broader run only if the entropic canary doesn't
  loop.

**Expected signal**:

- Quicksort throughput improves materially.
- The entropic prompt either hangs, times out, or shows repeated
  identical partial positions.

**Decision rule**:

- If quicksort recovers ≥ 25% tok/s vs the v8 baseline AND the
  entropic prompt shows loop behavior → F-018 confirmed as a
  real regression but also a necessary safety mechanism. Move
  to Phase 5 (adaptive heuristic).
- If throughput barely changes → stop treating the skip flag as
  the dominant cost; weight other phases more.

**Artifact**: `experiments/E1_skip_off/results.json` + `summary.md`.

### Phase 4 — E2 DRAFT_N_MAX sweep (GPU)

**Hypothesis**: 16 is too large for the current draft quality, and
a smaller block reduces wasted work enough to improve median
throughput even if nominal acceptance falls.

**Method**:

- Sweep `DRAFT_N_MAX` across 4, 8, 16 (default).
- All other settings fixed; use corrected real-acceptance metrics.
- Run full matrix on `qwen` thinking-off.

**Expected signal**:

- Quicksort prefers 8 or 16.
- Entropic prompts improve at 4 or 8 because failed blocks are
  cheaper to verify and replay.

**Decision rule**:

- If 8 beats 16 by ≥ 15% median tok/s on the 5-prompt set →
  smaller / adaptive block size becomes default Sprint 007 plan.
- If all smaller settings flat or worse → block size tuning
  isn't enough.

**Artifact**: `experiments/E2_draft_n_max/results.json` + `summary.md`.

### Phase 5 — E4 adaptive skip heuristic (GPU)

**Hypothesis**: skipping only after repeated partial acceptances at
the same position preserves the F-014 correctness fix while
recovering throughput lost to the current always-skip policy.

**Method**:

- Replace unconditional skip with bounded heuristic: "skip only
  after K consecutive partials at the same rejection position",
  starting K=2 and K=3.
- Reuse Phase 2 rejection-position counters.
- Run canary, then full `qwen` thinking-off matrix, then a
  50-request soak on the winning K to confirm no hang
  re-emergence.
- If promising, run one confirmation sweep on `qwen` thinking-on
  and one confirmation cell on `qwen36`.

**Expected signal**:

- Zero infinite restore loops in soak.
- Better tok/s than v8 baseline, especially on prompts that
  currently fall to one-token cycles after a single partial.

**Decision rule**:

- If heuristic improves median `qwen` tok/s by ≥ 20% vs v8
  baseline with zero hangs in soak → Sprint 007 productizes
  this as the F-018 fix.
- If safe but gains < 10% → treat as local patch, not Sprint
  007's center of gravity.

**Artifact**: `experiments/E4_adaptive_skip/results.json` +
`summary.md`. Best K value documented.

### Phase 6 — Findings + Sprint 007 recommendation

**Goal**: aggregate everything into a single decision document.

**Tasks**:

- [ ] Write `docs/sprints/SPRINT-006-dflash-FINDINGS.md`:
  hypothesis-by-hypothesis verdict (confirmed / refuted /
  inconclusive) with supporting data. Include a numerical
  attribution: "X% of v8's perf loss came from the skip flag,
  Y% from checkpoint cost, Z% unexplained".
- [ ] Sprint 007 recommendation following the decision tree
  below.
- [ ] Update `SPRINT-005-FOLLOWUPS-dflash.md` F-018 status with
  the chosen remediation path.
- [ ] If user has pushed fork commits in the meantime: bump
  `docker/Dockerfile` ROTORQUANT_COMMIT to `86272e841`, retire
  `docker/Dockerfile.local`.

---

## Sprint 007 Recommendation Decision Tree

```text
Start with Sprint 006 findings
│
├── Adaptive skip heuristic (Phase 5) clears safety + ≥20% median gain?
│   │
│   ├── Yes → Sprint 007 = productize F-018 fix
│   │         - land adaptive heuristic cleanly on fork
│   │         - wire corrected F-016 metrics into published bench docs
│   │         - rerun canonical L4 and decide if DFlash remains default-worthy
│   │
│   └── No
│
├── Rejection profile (Phase 2) shows early failures (0–2 tokens) AND
│   DRAFT_N_MAX 8 beats 16 (Phase 4)?
│   │
│   ├── Yes → Sprint 007 = dynamic block-size / rejection-aware DFlash
│   │         - shrink blocks on hostile prompts
│   │         - keep quicksort-friendly larger blocks where they win
│   │
│   └── No
│
├── Checkpoint restore + replay (Phase 1) ≥ 25% of wall time on entropic prompts?
│   │
│   ├── Yes → Sprint 007 = rollback-cost reduction
│   │         - optimize checkpoint cadence and replay policy
│   │         - investigate whether full-state restore makes DFlash
│   │           structurally weak on hybrid Qwen3.6
│   │
│   └── No
│
├── None of the above recover useful throughput?
│   │
│   ├── Draft/target mismatch still looks structural →
│   │   Sprint 007 = EAGLE3 (existing SPRINT-007-dflash.md stub) or
│   │   alternative draft-path investigation
│   │
│   └── Broadly negative result → Sprint 007 = declare current DFlash
│       path non-competitive for this stack and stop sprint-sized
│       investment here (operators continue using qwen-target-only)
```

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| **Fork: `/home/ravi/repos/llama-cpp-turboquant/`** | | |
| `common/speculative.cpp` | Modify (Phase 0) | Add rejection-position counters; experiment toggles for block size + skip behavior |
| `tools/server/server-context.cpp` | Modify (Phase 0) | Time checkpoint save/restore/replay; adaptive-skip scaffolding gated by env |
| `tools/server/server-task.h` | Modify (Phase 0) | Extend `result_timings` schema with new diagnostic fields |
| `tools/server/server-task.cpp` | Modify (Phase 0) | Serialize experiment metrics to API-visible result payloads |
| **Repo: `/home/ravi/repos/turbo/`** | | |
| `scripts/bench_speculative.py` | Modify (Phase 0) | Consume `draft_n_generated` + `draft_n_acc_tokens`; emit per-experiment summaries; `--experiment-name` flag |
| `docs/sprints/SPRINT-006-dflash-experiments/E1_skip_off/` | Create | Phase 3 results + summary |
| `docs/sprints/SPRINT-006-dflash-experiments/E2_draft_n_max/` | Create | Phase 4 results + summary |
| `docs/sprints/SPRINT-006-dflash-experiments/E3_checkpoint_cost/` | Create | Phase 1 results + summary |
| `docs/sprints/SPRINT-006-dflash-experiments/E4_adaptive_skip/` | Create | Phase 5 results + summary |
| `docs/sprints/SPRINT-006-dflash-experiments/E5_rejection_profile/` | Create | Phase 2 results + summary |
| `docs/sprints/SPRINT-006-dflash-FINDINGS.md` | Create | Aggregate hypothesis verdicts + Sprint 007 recommendation |
| `docs/BENCHMARK-REPORT.md` | Modify (Phase 0) | Republish Sprint 005 §10 with corrected F-016 acceptance rates |
| `docs/sprints/SPRINT-005-FOLLOWUPS-dflash.md` | Modify (Phase 6) | Update F-018 status with remediation path |

---

## Definition of Done

### Hard gates (sprint fails if any miss)

1. **F-018 root-caused**. Phase 3 (E1) tells us whether the skip flag is
   the dominant throttle, with a numerical fraction.
2. **Per-prompt rejection profile collected** for at least 2 prompts
   (quicksort + one entropic). Phase 2 (E5).
3. **At least 4 of 5 experiments executed** with documented numerical
   outcome and a written verdict.
4. **Findings document written** —
   `SPRINT-006-dflash-FINDINGS.md` — labels each top hypothesis
   confirmed / refuted / inconclusive with supporting data.
5. **Sprint 007 recommendation** is written per the decision tree, with
   the chosen branch.
6. **Bench harness consumes the F-016 metric**. The "100%" misleading
   `acceptance_rate` is gone from the harness output for fork pins
   that include `a2a6168ea`.

### Soft gates

- **Sprint 005 numbers republished** in BENCHMARK-REPORT with
  corrected acceptance rates.
- **Fork commits pushed** (user action; Sprint 006 doesn't block on it).
  Once pushed, `docker/Dockerfile` pin updated and
  `docker/Dockerfile.local` retired.
- **Soak test passes** (50 sequential requests on the Phase 5 winning
  configuration with no hangs).
- **`qwen36` confirmation cell** completed for the Phase 5 winner.

### Code hygiene

- All git operations: `git add -u` or explicit file lists.
- Commit messages: imperative subject + Co-Authored-By trailer.
- Sprint branch is `sprint/006-dflash` (off `sprint/005-dflash`); no
  merge to `main` (dflash-track convention).

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Investigation scope balloons into feature work | High | High | Fix sprint to 5 experiments; require explicit go/no-go after Phase 5 before considering E6–E11 |
| Instrumentation perturbs the very performance being measured | Medium | High | No-instrumentation baseline rerun is a Phase 1 prerequisite; keep timers/counters minimal |
| Local-only fork commits create reproducibility drift | High | Medium | Sprint uses `docker/Dockerfile.local` override; document version pin in every artifact JSON |
| Prompt-specific instability (e.g., P3 transport noise from F-014 era) muddies conclusions | Medium | Medium | Use canary prompts (P1 + P4) for iteration; treat P3 as confirmatory, not blocking |
| Adaptive heuristic fixes one profile but not the other | Medium | Medium | `qwen` is gate; `qwen36` is single confirmation cell — don't overclaim universality |
| GPU contention with risk-embedder training | Certain | Medium (sequencing only) | Phase 0 is GPU-free; Phases 1–5 wait for training to finish; no concurrent runs |
| Fork patches not upstream-ready under llama.cpp policy | High | Low | All Sprint 006 fork patches are local diagnostic only; upstream effort is its own future sprint |

---

## Security

- Debug instrumentation must NOT dump full prompt or completion text
  into committed artifacts. Store counts, timings, rejection
  positions, prompt indices only — no user-content leakage into
  versioned summaries.
- Experiment toggles (env vars `LLAMA_SPEC_DISABLE_SKIP`,
  `LLAMA_SPEC_ADAPTIVE_SKIP_K`, etc.) must remain opt-in and local.
  No debug mode becomes the default server path.
- Timing + metric schema changes stay bounded to speculative
  diagnostics; don't accidentally expose unrelated server internals
  through the API.

---

## Dependencies

### Prior work in this repo

- Sprint 005-dflash (Phase 1 numbers, F-014 fix chain, F-016 metric).
- `scripts/bench_speculative.py` `--no-think` flag (added 2026-04-28).
- `docker/Dockerfile.local` (build-from-local-checkout override from
  Sprint 005's F-014 iterations).

### Upstream

- Fork at `feature/sprint-004-rebase-dflash` `86272e841` or later.
  Commits `5f58c0d81..86272e841` are LOCAL ONLY pending interactive push.

### External artifacts

- `Qwen3.6-27B-DFlash-bf16.gguf`, `Qwen3.6-35B-A3B-DFlash-bf16.gguf`
  in `llm-models` volume (Sprint 004 F-001).
- No new GGUFs needed for the 5-experiment cut. (E8 target-quant sweep
  would need Q5_K_M / Q8_0 27B; deferred — see DEFERRED doc.)

### Hardware

- RTX 5090 (32 GB), 123 GB system RAM.
- **GPU constraint**: risk-embedder training is currently using the
  GPU. Phase 0 runs concurrently. Phases 1–5 wait.

---

## Open Questions

1. Is a single `qwen36` confirmation cell enough for the Phase 5
   winner, or does the team want a full MoE confirmation sweep?
2. Should the `DRAFT_N_MAX` sweep stay at 4 / 8 / 16, or is there
   value in testing 2 or 12 to justify extra matrix cost?
3. What is the minimum outcome that counts as a successful
   remediation signal for Sprint 007 planning: any positive median
   gain, ≥1.0×, or something closer to the original ≥1.3× ambition?
4. If the 5 experiments leave the cause ambiguous, should the first
   extension be E6 offline draft-vs-target alignment or E7 EAGLE3 as
   alternative baseline? (Both are in DEFERRED.)
5. Phase 0's "Republish Sprint 005 numbers" — should that go in a
   new BENCHMARK-REPORT subsection (e.g., § Sprint 005 Revised) or
   replace the existing tables? Replacement is cleaner but loses
   provenance of the original "100%" figure.
