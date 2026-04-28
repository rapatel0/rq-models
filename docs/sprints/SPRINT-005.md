# Sprint 005: L4 Benchmark Publish + Experimentation

**Status**: Planning
**Created**: 2026-04-27
**Depends on**: Sprint 004 (rebase + DFlash cherry-pick + source-converted drafts)
**Target hardware**: RTX 5090 (32 GB), 123 GB system RAM
**Estimated effort**: 1.5 weeks single-engineer
**Branches**:
- Repo: `rapatel0/rq-models` `sprint/005-l4-publish` (off `sprint/004-dflash` once that merges to `main`, otherwise off `sprint/004-dflash`)
- Fork: no fork-side changes expected unless Phase 4 (forced-rejection) requires the `LLAMA_SPEC_FORCE_REJECT_AT` env hook (F-002).

---

## Overview

Sprint 004 shipped the DFlash stack — rebased fork, cherry-picked draft graph,
source-converted GGUFs, Docker profiles, validation harness. The headline
gates (L1, L2, L3, L4 from Sprint 004's Definition of Done) have *runnable*
scripts but no published numbers — `BENCHMARK-REPORT.md` §10 still reads `TBD`.

Sprint 005 is the measurement sprint. Three jobs, in order:

1. **Run the canonical L4 5-prompt benchmark** with thinking-on (the deployed
   regime), populate the §10 tables on `qwen` (MoE+DFlash default) and
   `qwen-target-only` (the throughput escape hatch). The ≥1.3× median gate
   is structurally on the dense `qwen36-27b-dflash` (PREVIEW=1) profile per
   the original sprint plan, but Sprint 005 publishes for *all three* deployed
   profiles to give operators an honest picture.

2. **Run the experiment sweep** the team flagged at Sprint 004 close: target
   Q5_K_M vs Q4_K_XL, draft KV cache type (planar3 / iso3 / Q8_0 / f16),
   `--draft-max` (8 / 16 / 24), and the F-010 multi-slot characterization
   (`N_PARALLEL=2,4` against single-slot speculative). Output is a delta
   table per knob, not a tuning recommendation — operators pick their tradeoff.

3. **Close the forced-rejection correctness gate** (Sprint 004 hard gate #5):
   add the `LLAMA_SPEC_FORCE_REJECT_AT=N` debug env in fork's
   `common/speculative.cpp` (F-002), author the C++ unit test that asserts
   post-restore recurrent-state bytes equal the target-only trajectory at
   position N+1 (F-003 subtests F-H), bump the fork pin, flip the xfail
   decorator on `tests/test_speculative.py::test_force_reject_preserves_output`.

Out of scope: EAGLE3 (Sprint 006), multi-slot batched-draft optimization
(Sprint 007), non-greedy sampler validation (D-003 / future), streaming
(D-006).

---

## Use Cases

1. **Operator picks DFlash with calibrated expectations**: A user looks at
   `BENCHMARK-REPORT.md` §10 for `qwen` (MoE+DFlash) and sees the per-prompt
   tok/s table with real numbers, headline median ratio, snapshot wallclock,
   acceptance rate. They know whether DFlash helps their workload.

2. **Operator picks a knob with a measurement justification**: "Should I
   bump my draft to f16 KV cache?" → look at the draft-KV sweep delta table.
   "Will I lose throughput at 4 concurrent users?" → look at the multi-slot
   row.

3. **Reviewer trusts the validation gate**: Sprint 004 hard gate #5
   (forced-rejection correctness) has a test that fires; the C++ test passes;
   the pytest `xfail` flips to `xpass-strict`. The "we couldn't actually
   exercise the rollback path" footnote goes away.

4. **PR reviewer can reproduce numbers**: `make bench-dflash-all` runs the
   three legs sequentially against a host with warmed `llm-models`, writes
   the same JSON + summary table the sprint published, ±10%.

---

## Architecture

### Decision tree for each measurement

```
                     ┌─────────────────────────────┐
                     │ Run canonical L4 sweep?     │
                     │  - qwen (MoE+DFlash)        │
                     │  - qwen-target-only         │
                     │  - qwen36-27b-dflash (PREVIEW)│
                     └──────────────┬──────────────┘
                                    │
                  ┌─────────────────┼──────────────────┐
                  ▼                 ▼                  ▼
        target-only leg       autoregressive      dflash leg
       (no draft, baseline)   (draft, no DFlash)  (draft + --dflash)
                  │                 │                  │
                  └─────────┬───────┴──────────────────┘
                            ▼
                  bench_speculative.py --finalize
                            │
                            ▼
              SPRINT-005-L4-results.json
              SPRINT-005-L4-summary.md
                            │
                            ▼
              BENCHMARK-REPORT.md §10 cells filled
```

### Phase ordering

```
Phase 0:   Sprint setup                    (~5%,  0.5 days)
Phase 0.5: 27B correctness probe (blocking)(~3%,  0.5 days)
Phase 1:   Canonical L4 sweep              (~18%, 2 days)
Phase 2:   Experiment sweep                (~30%, 3 days)
Phase 3:   Forced-rejection (F-002 + F-003)(~25%, 2.5 days)
Phase 4:   BENCHMARK-REPORT publish + docs (~15%, 1.5 days)
Phase 5:   Sprint outcome + DoD            (~4%,  0.5 days)
```

---

## Implementation

### Phase 0: Sprint setup (~5%)

**Goal**: Branch off, image rebuilt, draft GGUFs in volume, smoke check
that the L4 harness can hit a running profile.

**Tasks**:
- [ ] Create `sprint/005-l4-publish` branch off the post-Sprint-004 base.
- [ ] Confirm `rotorquant:latest` is the rebuilt image at fork pin
      `1c9b77fdd` (or later if fork advances). `make build` if not.
- [ ] Confirm `Qwen3.6-27B-DFlash-bf16.gguf` and
      `Qwen3.6-35B-A3B-DFlash-bf16.gguf` are in `llm-models` volume.
      `make convert-drafts` if not.
- [ ] Smoke `make run-qwen-bg && curl /health && curl /v1/chat/completions`
      with one prompt at `--temp 0 --top-k 1`. Assert response coherent.
      `make stop`.

**Phase gate**: `qwen` profile boots, `/health` returns 200, one greedy
completion succeeds.

### Phase 0.5: 27B correctness probe (blocking)

**Goal**: Decisively answer "is the dense 27B-DFlash output correct, or
is the 37% acceptance hiding a verify-path bug?" before any benchmarking.

**Why blocking**: PR #22105 was tuned primarily on MoE targets
(gpt-oss-20B reference, Qwen3.6-35B-A3B test set). The dense 27B is a
hybrid model with a different layer ratio (48 of 64 `linear_attention`
layers vs 30 of 40 on the 35B-A3B); the rollback path exercises the
recurrent-state checkpoint heavily on every rejection. Sprint 004 smoke
saw 37% acceptance on the 27B vs 100% on the 35B-A3B on similar probes.
Most likely explanation is draft training distribution drift (z-lab
marks 27B as preview). Possible-but-unlikely explanation: a verify-path
bug specific to the dense 27B that surfaces under thinking-on prompts.

**Why probing acceptance ≠ probing correctness**: speculative decoding
is *designed* such that acceptance rate doesn't determine output
correctness — rejected drafts fall back to sampling from the target's
logits at the rejection point. So even at 0% acceptance, output should
match target-only token-for-token. Only iff the verify path is bug-free.

**Tasks**:
- [ ] Bring up `qwen36-27b` (target-only) profile in background. Capture
      a 256-token greedy completion on the quicksort prompt at
      `--temp 0 --top-k 1 --seed 42`. Save token sequence as
      `docs/sprints/SPRINT-005-27b-target-only.tokens.json`.
- [ ] `make stop`. Bring up `qwen36-27b-dflash` (PREVIEW=1) on the same
      port. Same prompt + sampling params. Save as
      `SPRINT-005-27b-dflash.tokens.json`.
- [ ] Diff the two sequences byte-for-byte. Expected: 256/256 match.

**Phase gate (the answer)**:
- **Pass (256/256)**: 27B output is correct; 37% acceptance is a perf
  observation, not a correctness bug. PREVIEW gate stays for "drafts
  iterating", not for "broken". Continue to Phase 1.
- **Fail (any divergence)**: STOP. 27B-DFlash disabled in
  `make run-qwen36-27b-dflash` (entrypoint refuses `qwen3.6-27b*` +
  dflash regardless of PREVIEW=1) until rollback path debugged. Open
  F-011 in `SPRINT-005-FOLLOWUPS.md` documenting the divergence
  position, expected vs actual tokens at the divergence, and a
  hypothesis. Sprint 005 descopes to 35B-only L4 publish.

**Files**:
- `scripts/validate_dflash.py` — already exists; the `--reference none`
  mode does the L2 greedy equivalence already. May need an
  `--save-tokens <path>` flag to dump the per-position token IDs (Phase
  1 will reuse this for per-prompt correctness checks anyway).
- `docs/sprints/SPRINT-005-27b-correctness-probe.md` — short MD with
  result + hypothesis + next action.

**Estimated effort**: ~30 minutes once rotorquant rebuild is warm. Two
compose-up/stop cycles, two completions, one diff.

### Phase 1: Canonical L4 sweep (~20%)

**Goal**: Per-prompt tok/s + acceptance rate across the 5-prompt set on
all three deployed profiles, written to JSON + the BENCHMARK-REPORT
summary table.

**Files**:
- `Makefile` — add `bench-dflash-all` orchestrator (F-007 from Sprint 004
  follow-ups). Runs the three legs sequentially with health-poll between.
- `scripts/bench_speculative.py` — already exists. May need a profile-arg
  for the orchestrator to know which compose profile to bring up between
  legs.
- `docs/sprints/SPRINT-005-L4-results.json` (NEW — generated)
- `docs/sprints/SPRINT-005-L4-summary.md` (NEW — generated)

**Tasks**:
- [ ] Add `bench-dflash-all` Make target. Orchestrates:
      1. `make run-qwen-target-only-bg` → `bench-dflash-leg LEG=target-only`
         → `make stop`
      2. `make run-qwen-bg SPECULATIVE_MODE=autoregressive` →
         `bench-dflash-leg LEG=autoregressive` → `make stop`
      3. `make run-qwen-bg` → `bench-dflash-leg LEG=dflash` → `make stop`
      4. `bench-dflash` (finalize)
- [ ] Run for `qwen` (MoE+DFlash). Three legs, 5 prompts, 3 trials each.
      Thinking-on. Greedy. Save JSON. Generate summary MD.
- [ ] Run for `qwen36-27b-dflash` (PREVIEW=1). Same shape. Distinct output
      JSON path so they don't overwrite each other:
      `SPRINT-005-L4-27b.json`, `SPRINT-005-L4-35b.json`.
- [ ] Capture acceptance rate per leg. The harness today reads
      `timings.predicted_per_second` for tok/s; extend to also pull
      `timings.draft_n` / `timings.draft_n_accepted` if llama-server exposes
      them, otherwise parse from the per-leg log via a regex sidecar.

**Phase gate**: All three profiles produce 5×3=15 timed completions per
leg with no `urllib.error.HTTPError`. Median DFlash× ratio captured.

### Phase 2: Experiment sweep (~30%)

**Goal**: Per-knob delta tables. Each knob varied at 2-4 levels against a
single fixed prompt (the quicksort headline) to keep the matrix tractable.

**Sweep matrix (fixed: thinking-on, greedy, ctx=2048, single-slot,
default `qwen` config)**:

| Knob | Levels | Free-form notes |
|---|---|---|
| Target weight quant | Q4_K_XL (default), Q5_K_M, Q8_0 | costs VRAM; 27B Q8_0 is ~28 GB |
| Draft KV cache type | planar3, iso3, Q8_0, f16 | draft KV is small (<1 GB) so all four are headroom-safe |
| `--draft-max` | 8, 16 (default), 24 | smaller blocks waste less on rejection |
| `N_PARALLEL` (multi-slot characterization, F-010) | 1, 2, 4 | expect sub-linear; quantify the cost |

**Tasks**:
- [ ] Add `scripts/sweep_dflash.py` (NEW). Driven by a YAML or inline dict
      of `{knob_name: [levels]}`. For each level: bring up the right profile
      (or set EXTRA_ARGS / env override on the existing one), run one prompt
      × 3 trials, record tok/s + acceptance rate, tear down. Idempotent;
      resumable.
- [ ] Run target-quant sweep on the 27B-dflash profile. Drop Q5_K_M and
      Q8_0 GGUFs into the `llm-models` volume first (one-time download from
      unsloth or the equivalent repo).
- [ ] Run draft-KV sweep on `qwen36-27b-dflash`. Draft KV cache type is
      already `DRAFT_KV_CACHE_TYPE` env in entrypoint.
- [ ] Run `--draft-max` sweep. `DRAFT_N_MAX` env exists.
- [ ] Run multi-slot N_PARALLEL=1/2/4 on `qwen36-27b-dflash`. Per-slot
      decode tok/s + aggregate throughput. Compare to N_PARALLEL=1
      single-slot.
- [ ] Emit `docs/sprints/SPRINT-005-experiments.json` (NEW) with per-knob
      delta tables.

**Phase gate**: Each knob produces a delta table with at least 3 levels
measured. No knob shows internal contradiction (e.g., KV variant flipping
between 0.8× and 1.5× with no apparent cause — investigate before
accepting).

### Phase 3: Forced-rejection correctness (~25%)

**Goal**: Close Sprint 004 hard gate #5 — exercise the speculative
checkpoint+restore path deterministically, assert post-restore state
matches target-only trajectory.

**Files (fork)**:
- `common/speculative.cpp` — add `LLAMA_SPEC_FORCE_REJECT_AT=N` env
  handling. When set, force the verify path to reject draft tokens at
  position N (0-indexed) regardless of acceptance probability.
- `tests/test-checkpoint-hybrid-state.cpp` — NEW. Subtests F-H:
  - F: After force-reject at N=4, recurrent state at position N+1 equals
    target-only trajectory at position N+1 (bytes-equal assert).
  - G: Same for full-attention K (planar3 / iso3 layouts).
  - H: 16-token full-block force-reject (N=15) — entire snapshot consumed.
- `tests/CMakeLists.txt` — register the new CTest target.

**Files (repo)**:
- `tests/test_speculative.py` — flip
  `test_force_reject_preserves_output` from `@pytest.mark.xfail` to
  `@pytest.mark.xfail(strict=True)`. Once F-002 lands, it should xpass.
- `docker/Dockerfile` — bump ROTORQUANT_COMMIT to the new fork pin.

**Tasks**:
- [ ] Author `LLAMA_SPEC_FORCE_REJECT_AT` env in fork. ~30 LOC.
- [ ] Author `tests/test-checkpoint-hybrid-state.cpp` subtests F-H.
- [ ] Bump fork pin in `docker/Dockerfile`. Rebuild image. Run pytest
      against a live server + force-reject env. Confirm xpass.
- [ ] Update `SPRINT-004-FOLLOWUPS.md` F-002 → closed; F-003 → partially
      mitigated (subtests A/B/D/E from F-003 still open — see Sprint 008).

**Phase gate**: pytest `test_force_reject_preserves_output` passes
(xpass-strict). C++ subtests F-H all pass under `ctest`.

### Phase 4: BENCHMARK-REPORT publish + docs (~15%)

**Goal**: §10 TBD cells filled. Operators can read tok/s + acceptance
+ ratio numbers without consulting the sprint authors.

**Files**:
- `docs/BENCHMARK-REPORT.md` — replace TBD cells in §10's L4 / parity /
  snapshot-cost / acceptance-rate subsections with real numbers.
- `README.md` — update "Speculative Decoding" section's reproduction
  instructions to point at `make bench-dflash-all`.
- `docs/sprints/SPRINT-004.md` — flip the relevant DoD checkboxes from
  `[ ]` / `[partial]` to `[x]`. Status: complete.
- `docs/sprints/SPRINT-005-FOLLOWUPS.md` (NEW) — execution-discovered
  follow-ups for Sprint 006+.

**Tasks**:
- [ ] Embed `SPRINT-005-L4-summary.md` content into BENCHMARK-REPORT.md
      §10 (or anchor-link if too large). Per-profile sub-tables.
- [ ] Embed `SPRINT-005-experiments.json` rendered as MD tables.
- [ ] Update README's "Speculative Decoding" section: replace "median TBD"
      with the headline ratio. Add a one-line summary of the experiment
      sweep findings.
- [ ] Open Sprint-006 ticket items in `SPRINT-005-FOLLOWUPS.md` for
      anything discovered (e.g., if multi-slot characterization shows >50%
      sub-linear at N=4, escalate to Sprint 007 priority).

**Phase gate**: A PR reviewer reading only BENCHMARK-REPORT §10 can
answer "should I use DFlash?" without running anything.

### Phase 5: Sprint outcome + DoD (~5%)

**Goal**: Sprint marked complete, follow-ups doc finalized, branch
prepared for merge.

**Tasks**:
- [ ] Update SPRINT-005.md status → complete (or complete-with-followups).
- [ ] Final commit on `sprint/005-l4-publish`; tag.
- [ ] User-approved merge into `main`.

---

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `Makefile` | Modify | Add `bench-dflash-all` orchestrator |
| `scripts/sweep_dflash.py` | Create | Per-knob experiment driver |
| `scripts/bench_speculative.py` | Modify (small) | Acceptance-rate capture |
| `tests/test_speculative.py` | Modify | Flip xfail decorator post-F-002 |
| `docker/Dockerfile` | Modify | Bump ROTORQUANT_COMMIT post-Phase-3 |
| Fork `common/speculative.cpp` | Modify | LLAMA_SPEC_FORCE_REJECT_AT env |
| Fork `tests/test-checkpoint-hybrid-state.cpp` | Create | Subtests F-H |
| `docs/BENCHMARK-REPORT.md` | Modify | Fill §10 TBD cells |
| `README.md` | Modify | Update repro instructions |
| `docs/sprints/SPRINT-005-L4-results.json` | Generate | Per-profile L4 results |
| `docs/sprints/SPRINT-005-L4-summary.md` | Generate | Markdown summary |
| `docs/sprints/SPRINT-005-experiments.json` | Generate | Per-knob delta tables |
| `docs/sprints/SPRINT-005-FOLLOWUPS.md` | Create | Execution-discovered followups |

---

## Definition of Done

### Hard gates (sprint fails if any miss)

1. **27B correctness probe pass** (Phase 0.5): 256/256 token match
   between target-only and target+DFlash on the dense 27B at greedy
   sampling. If fail → 27B is not measurable until the verify-path bug
   is fixed; sprint descopes to 35B-only.
2. **L4 canonical sweep numbers exist** for `qwen` (MoE+DFlash) and
   `qwen36-27b-dflash` (PREVIEW, only if Hard Gate #1 passes), both with
   thinking-on, written to `SPRINT-005-L4-results.json`.
3. **L4 ≥1.3× median gate** evaluated on `qwen36-27b-dflash`. Pass/fail
   recorded; if fail, root-caused (e.g., "thinking-on cuts acceptance to X%
   vs PR's no-think Y%; expected with current draft training data").
4. **Forced-rejection correctness** (Sprint 004 DoD #5): C++ subtests F-H
   pass; pytest `test_force_reject_preserves_output` xpasses-strict.
5. **BENCHMARK-REPORT.md §10 has zero TBD cells** in the L4 / parity /
   snapshot-cost / acceptance-rate subsections.
6. **Reproducibility**: `make bench-dflash-all` on a fresh-clone host with
   warmed `llm-models` reproduces the headline ratio within ±10%.

### Soft gates

- **Experiment sweep all knobs measured** at the levels in Phase 2.
- **Multi-slot N_PARALLEL=4 characterization**: per-slot tok/s ≥ 0.5×
  single-slot. <0.5× → escalate to Sprint 007 priority.
- **L2 greedy equivalence** (Sprint 004 DoD #6): now actually runnable —
  `validate_dflash.py --reference none` should produce 256/256 token
  match on at least 3 of the 5 prompts.
- **L3 z-lab pytorch parity** (Sprint 004 DoD #7): runnable; ≥64-token
  match on 3 of 5 prompts; acceptance ±5pp. SHA-pin captured (F-006 close).

### Code hygiene

- All git operations: `git add -u` or explicit file lists.
- Commit messages: imperative subject + Co-Authored-By trailer.
- No `.env` / `HF_TOKEN` / etc committed.
- Sprint branch is `sprint/005-l4-publish`; merge to `main` only after
  user approval.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| L4 ≥1.3× median fails on 27B-DFlash with thinking-on | Medium | Medium | Document the gap honestly. Sprint 004 chose thinking-on as the validation regime; PR's 60–80pp acceptance loss with thinking-on is a known deployment cost. May need to reframe the gate as "no regression vs target-only" rather than "speedup ≥1.3×" if measurements show that |
| `LLAMA_SPEC_FORCE_REJECT_AT` env doesn't fit cleanly post-cherry-pick (PR #22105 rewrote `common/speculative.cpp`) | Medium | Medium | Phase 3 starts with a 2-hour spike to identify hook points before committing to the implementation; if hook is non-trivial (>100 LOC), descope to a runtime-flag instead of an env |
| Multi-slot characterization reveals >50% sub-linear at N=4 | Medium | Low | Already a known sub-linear case (F-010); this becomes the trigger to prioritize Sprint 007 over Sprint 006 |
| z-lab pytorch reference doesn't run on RTX 5090 sm_120 | Low | Medium | Run reference on a separate machine, store outputs as JSON fixtures (Sprint 004 R10 mitigation) |
| GPU contention with concurrent training jobs causes intermittent OOM | High | Medium | All bench runs in <15-min windows; coordinate around training schedule. `bench_speculative.py` reports per-run free VRAM; aborts <8 GB headroom |

---

## Dependencies

### Prior work in this repo
- Sprint 004 (rebase + DFlash + source-converted drafts + harness scripts)

### Upstream
- Fork at `feature/sprint-004-rebase-dflash` `1c9b77fdd` or later.

### External artifacts
- `Qwen3.6-27B-DFlash-bf16.gguf`, `Qwen3.6-35B-A3B-DFlash-bf16.gguf` in
  `llm-models` volume (produced by `make convert-drafts`).
- For target-quant sweep: `Qwen3.6-27B-Q5_K_M.gguf` and `-Q8_0.gguf` from
  `unsloth/Qwen3.6-27B-GGUF` (one-time download; ~50 GB combined).

### Hardware
- RTX 5090 (32 GB), 123 GB system RAM, NVIDIA driver 570+.

---

## Open Questions

1. **Should we also run Sprint 005 against the 35B target-quant sweep?**
   The 35B-A3B is MoE so target weight quant matters less (3B active
   params). Tentatively no — focus on 27B for the headline ratio gate.

2. **Is the autoregressive leg worth running given the 5-prompt set is
   tight?** It's a comparison point, not a gate. Skip-able if time-pressed.

3. **Do we publish per-prompt acceptance rates or only per-leg medians?**
   Per-prompt is more informative (shows which prompts DFlash helps most)
   but doubles the table size. Tentative: per-prompt in JSON,
   median-of-medians in MD summary.
