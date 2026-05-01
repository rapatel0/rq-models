# Sprint 007-dflash: Wire VRAM-shadow checkpoint into speculative path

> **Track suffix**: `-dflash`. Does not merge to `main`.

**Status**: Planning (2026-05-01)
**Sprint type**: Implementation + measurement
**Created**: 2026-05-01
**Depends on**: Sprint 006-dflash findings (top recommendation: VRAM-shadow wire-up)
**Estimated effort**: ~5 days single-engineer
**Branches**:
- Repo: `sprint/007-dflash` (off `sprint/006-dflash`)
- Fork: `feature/sprint-004-rebase-dflash` at `4ef60a057` (LOCAL ONLY)

## Overview

Sprint 006 surfaced — and codex peer review confirmed — that the
fork's `vram_seq_checkpoint` (Sprint 003 commit `9b191cd87`) is built
specifically for the speculative `PARTIAL_ONLY` snapshot path but was
never wired into the server-side speculative checkpoint flow. The
server still calls `llama_state_seq_get_data_ext` /
`set_data_ext` (host PCIe pageable) at three sites in
`tools/server/server-context.cpp`. Sprint 006 measured the resulting
ckpt save+restore overhead at ~50% of speculative wallclock.

Sprint 007 wires the existing infrastructure in. The codex review also
flagged two methodology bugs from Sprint 006 (F-022 cumulative
counters, F-023 timer conflation) that should be fixed in the same
sprint so post-wire-up re-runs of E2 / E5 produce trustworthy data.

## Use Cases

1. **Operators see speculative paying off**: with VRAM-shadow saves
   at HBM bandwidth (D→D `cudaMemcpyDeviceToDevice`), the
   ~38%-of-wallclock save tax should drop to <1%. Combined with
   block-size choice (TBD post-wire-up), median DFlash× may exceed
   1.0× target-only on more prompts.
2. **Sprint 008 has clean data to design adaptive policies on**:
   F-022's cumulative-counter fix gives true per-request acceptance.
   F-023's timer split lets future E3 reads attribute "save we'd
   recover by VRAM swap" vs "sync wait that's a different problem".
3. **The DRAFT_N_MAX default is chosen on real numbers**: Sprint 006
   recommended N=4 based on host-path data; codex flagged that with
   cheap saves the optimal N may move up. Sprint 007 measures and
   decides.

## Architecture

```text
fork pin (4ef60a057 baseline)
  ↓
Phase 1: per-slot vram_seq_checkpoint construct + 3 call-site swaps
  ↓
Phase 2: split E3 timer (sync-wait / snapshot-copy / serialize)
  ↓
Phase 3: F-022 fix — reset cumulative counters in common_speculative_begin()
  ↓
Phase 4: build + smoke validate (3 sequential prompts; no crashes;
        ckpt save% measurably lower; per-request acceptance correct)
  ↓
Phase 5: re-run E2 (DRAFT_N_MAX 4/8/16) and E5 (rejection profile)
        on the wired-up build
  ↓
Phase 6: choose DRAFT_N_MAX default; write Sprint 007 findings;
        recommendation for Sprint 008 (EAGLE3 productionization
        if DFlash now ships, or further optimization if still
        marginal)
```

Constraints (all match the qwen profile):
- `vram_seq_checkpoint` requires `GGML_USE_CUDA` ✓ (rotorquant
  build is CUDA 13.1)
- requires hybrid recurrent memory ✓ (Qwen3.6 has it)
- requires single-sequence ✓ (qwen profile is N_PARALLEL=1)
- requires `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` context ✓ (the slot's
  speculative path runs this mode for hybrid)

The host path stays as a fallback for non-CUDA / non-hybrid / multi-seq
contexts.

## Implementation

### Phase 1 — Wire `vram_seq_checkpoint` into speculative path

**Files (fork)**:
- `tools/server/server-context.cpp`
  - `server_slot`: add `std::unique_ptr<vram_seq_checkpoint> vram_ckpt`
  - `server_slot::reset()` or constructor: try construct
    `vram_seq_checkpoint` if `GGML_USE_CUDA && hybrid && n_parallel == 1`
  - Speculative save site (~line 416): if `vram_ckpt && vram_ckpt->is_valid()`,
    call `vram_ckpt->save()`; else fall back to
    `server_get_checkpoint(...)` (the host path).
  - Speculative restore site (~line 3136): if VRAM shadow active and
    last save used it, call `vram_ckpt->restore()`; else fall back to
    `llama_state_seq_set_data_ext`.
  - Track which path each save used so restore picks the right one.
  - `server_get_checkpoint` itself stays untouched for prompt-cache
    use at line 56 / 172 (those serve different purposes).

**Decision rule**:
- VRAM path used: ckpt save+restore overhead < 5% of speculative
  wallclock on qwen think-off Hamlet.
- VRAM path NOT used (fallback only): mark phase failed; investigate
  constraint mismatch.

### Phase 2 — Split E3 timer (per F-023)

**Files (fork)**:
- `tools/server/server-context.cpp`
  - Replace single `t_ckpt_save_us` accumulator with three:
    `t_ckpt_sync_us` (cudaSynchronize wait), `t_ckpt_copy_us`
    (the actual `vram_ckpt->save()` or `llama_state_seq_get_data_ext`
    invocation), `t_ckpt_serialize_us` (host-side state-blob construction
    on the fallback path; 0 on the VRAM path).
- `tools/server/server-task.h` / `.cpp`: extend `result_timings`
  with the three split fields. Keep the old `spec_t_ckpt_save_us`
  for back-compat (sum of the three).
- `scripts/bench_speculative.py`: parse the three new fields
  alongside the existing combined one.

### Phase 3 — Fix F-022 cumulative counter bug

**Files (fork)**:
- `common/speculative.cpp` `common_speculative_begin()` at line
  ~1452: reset `impl->n_gen_tokens = 0; impl->n_acc_tokens = 0;
  impl->n_gen_drafts = 0; impl->n_acc_drafts = 0;
  impl->n_call_begin = 0; impl->n_call_draft = 0;
  impl->n_call_accept = 0;` for each impl before calling
  `impl->begin(prompt)`.

This makes per-request `draft_n_generated` and `draft_n_acc_tokens`
actually per-request rather than cumulative-across-the-slot's-life.

### Phase 4 — Build + smoke

**Tasks**:
- [ ] `make build` from `docker/Dockerfile.local` (build-from-local-fork-
      checkout). User must push fork commits OR rebuild via the
      override; Sprint 007 doesn't block on push.
- [ ] Boot `qwen` profile, send 3 sequential P1 quicksort requests.
- [ ] Verify: no crashes, response includes new split-timer fields,
      `draft_n_generated` for trial 0 ≈ trial 1 ≈ trial 2 (proves
      F-022 fix works).
- [ ] Verify: `spec_t_ckpt_copy_us / spec_t_ckpt_save_us` ratio
      drops dramatically vs Sprint 006 baseline (proves VRAM path
      is firing).

**Phase gate**: 3 sequential requests succeed; per-request
acceptance numbers are independent (not cumulative); ckpt save
copy time per request is < 1ms (was 35ms on host path).

### Phase 5 — Re-run E2 and E5

**Tasks**:
- [ ] `DRAFT_N_MAX={4,8,16}` × full 5-prompt × 3-trial bench using
      `scripts/run_sprint006_experiment.sh` (rename / extend for
      007, but the structure is the same). Save artifacts under
      `docs/sprints/SPRINT-007-dflash-experiments/E2-rerun/`.
- [ ] Histogram capture as in Sprint 006's E5 — should now reflect
      true per-request rejection patterns.
- [ ] Compare: did the optimal N shift relative to Sprint 006? Did
      any prompt cross the ≥1.0× / ≥1.3× thresholds vs target-only?

**Phase gate**: full sweep completes without crashes; data is
populated per-N; comparison table written.

### Phase 6 — Choose default + write findings

**Tasks**:
- [ ] Pick the new `DRAFT_N_MAX` default (could be 4, 8, 16, or
      adaptive). Update `docker/entrypoint.sh` default.
- [ ] If gates are now met on more prompts: update
      `docs/BENCHMARK-REPORT.md` § Sprint 005 → § Sprint 007 with
      the new headline numbers.
- [ ] Update README's Speculative Decoding section.
- [ ] Write `docs/sprints/SPRINT-007-dflash-FINDINGS.md` with
      hypothesis verdicts (did VRAM-shadow give the expected
      speedup? Did the optimal N change?).
- [ ] Sprint 008 recommendation: EAGLE3 productionization
      (existing 008 stub) vs further optimization (e.g., adaptive
      block size, draft distillation).

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `tools/server/server-context.cpp` | Modify (fork) | Per-slot vram_seq_checkpoint + 3 call-site swaps + split timer |
| `tools/server/server-task.h` / `.cpp` | Modify (fork) | result_timings split fields |
| `common/speculative.cpp` | Modify (fork) | F-022 reset cumulative counters in begin() |
| `scripts/bench_speculative.py` | Modify (repo) | Parse new split-timer fields |
| `scripts/run_sprint006_experiment.sh` | Reuse / extend | Driver for E2 re-run |
| `docs/sprints/SPRINT-007-dflash-experiments/E2-rerun/` | Create | Re-run artifacts |
| `docs/sprints/SPRINT-007-dflash-FINDINGS.md` | Create | Outcomes + Sprint 008 recommendation |
| `docker/entrypoint.sh` | Modify (repo) | New DRAFT_N_MAX default if changed |
| `docs/BENCHMARK-REPORT.md` | Modify (repo) | Republish numbers if gates now met |
| `README.md` | Modify (repo) | Update Speculative Decoding guidance |

---

## Definition of Done

### Hard gates

1. `vram_seq_checkpoint` instances are constructed and used by the
   speculative path on the qwen profile. Verified via the new
   split-timer fields (copy time per save < 1 ms).
2. F-022 fixed. `draft_n_generated` and `draft_n_acc_tokens` are
   per-request after the fix, verified by a 3-trial smoke test on a
   single prompt.
3. F-023 split-timer in place. `result_timings` exposes
   `spec_t_ckpt_sync_us`, `spec_t_ckpt_copy_us`,
   `spec_t_ckpt_serialize_us` separately.
4. Full E2 sweep at N={4,8,16} re-run on the wired-up build.
   Comparison table vs Sprint 006 written.
5. SPRINT-007-FINDINGS.md exists with hypothesis verdicts and
   Sprint 008 recommendation.

### Soft gates

- DRAFT_N_MAX default updated in `docker/entrypoint.sh` if Phase 5
  data warrants.
- BENCHMARK-REPORT republished if any gate (≥1.0× or ≥1.3×) now
  passes on >2 prompts.
- README's DFlash guidance reflects current numbers.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `vram_seq_checkpoint` constraints mismatch a real edge case | Medium | High | Preserve host path as fallback; gate VRAM path with `is_valid()` check |
| Build breaks under combined edits | Medium | Medium | Single rebuild after all 3 phases of edits, validate via incremental syntax check |
| Re-run E2 gives different ranking than Sprint 006 | Likely (that's the whole point) | Low (information win) | This sprint is designed to discover this |
| F-022 fix changes other code's expectation of cumulative counters | Low | Medium | Grep for callers of `n_gen_tokens` / `n_acc_tokens` outside `common_speculative_*` before flipping |

---

## Open questions

1. Should the VRAM-path detection be runtime auto-detect or env-gated
   for bring-up? (Recommend: env-gated `LLAMA_SPEC_VRAM_CKPT=1` for
   first build, flip to auto-on once verified.)
2. If post-wire-up E2 says optimal N is still 4, do we ship that as
   default immediately or wait for 008 to add adaptive sizing?
3. EAGLE3 productionization (existing Sprint 008 stub) — promote to
   Sprint 009 if VRAM wire-up unlocks DFlash, or keep at 008?
