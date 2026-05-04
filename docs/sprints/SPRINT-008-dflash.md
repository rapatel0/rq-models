# Sprint 008-dflash: VRAM-shadow ckpt cells[] snapshot + clean re-bench

> **Track suffix**: `-dflash`. Does not merge to `main`.

**Status**: Planning (2026-05-01)
**Sprint type**: Implementation + measurement
**Created**: 2026-05-01
**Depends on**: Sprint 007-dflash partial close (F-022 fix shipped on host
path; VRAM scaffolding in code env-gated off; F-024 root-cause documented)
**Estimated effort**: ~2 days single-engineer
**Branches**:
- Repo: `sprint/008-dflash` (off `sprint/007-dflash`)
- Fork: `feature/sprint-004-rebase-dflash` continues from `43c8c1dfe`

## Overview

Sprint 007 wired `vram_seq_checkpoint` into the speculative path,
exposed it via `LLAMA_SPEC_VRAM_CKPT=1`, then discovered the path
fails on round 2 because the existing implementation copies recurrent
tensor bytes only and does not snapshot the recurrent memory's
`cells[]` metadata (per-cell `pos`, `seq_id`, `tail`, `src` plus
container-level `head`, `used`). The host path's
`llama_state_seq_*_data_ext` serializes both via `state_write_meta` /
`state_read_meta`. Codex peer review (gpt-5.3-codex high) pinned this
as the root cause.

Sprint 008 closes that gap. Once VRAM ckpt actually works, the ~38%
wallclock save tax measured in Sprint 006 E3 disappears, which (per
Sprint 006 findings) was the dominant overhead. With the save cheap,
we re-run E2 (DRAFT_N_MAX sweep) on the corrected metric and pick a
new default.

## Use Cases

1. **DFlash speculative finally pays off on the median 5-prompt set.**
   Sprint 005's gate (≥1.3× target-only) failed at 0.80× on qwen.
   Codex hypothesis: with cheap saves + N=4 (Sprint 006 E2's
   recommendation), the median crosses ≥1.0× and code-class prompts
   may exceed 2×.
2. **Operators can flip on speculative without thinking about the cost.**
   Today the docker default has `DRAFT_N_MAX=16` because we never
   measured below — but with 38% save tax that's the *only* knob we
   could move. Post-Sprint 008 we measure properly and ship a default
   tuned for the real bottleneck.
3. **F-022 fix gets exercised under realistic conditions.** Sprint 007
   verified F-022 works on a single-shot 200-token request. Sprint 008
   re-bench (3-trial × 5-prompt) is the first real test that
   per-request `draft_n_generated` numbers stay clean across many
   sequential requests.

## Architecture

```text
fork pin (43c8c1dfe baseline)
  ↓
Phase 1: extend vram_seq_checkpoint with cells[] snapshot/restore
  ↓
Phase 2: smoke test (3 sequential P1 quicksort prompts; verify
        round 2/3 succeed; verify save copy time < 1ms; verify
        F-022 counters per-request)
  ↓
Phase 3: full re-bench at DRAFT_N_MAX={4,8,16} × 5-prompt × 3-trial
        on the working VRAM path. Capture E5 rejection histograms
        for each N.
  ↓
Phase 4: choose DRAFT_N_MAX default. Update docker/entrypoint.sh.
        Republish BENCHMARK-REPORT.md if any prompt clears the
        Sprint 005 ≥1.0× / ≥1.3× gates.
  ↓
Phase 5: write SPRINT-008-dflash-FINDINGS.md; pick next sprint
        (EAGLE3 productionization moves to 009 if Sprint 008 ships;
        further DFlash optimization if still under target).
```

Constraints (unchanged from Sprint 007):
- requires `GGML_USE_CUDA` ✓
- requires hybrid recurrent memory ✓
- requires single-sequence ✓ (qwen profile is `N_PARALLEL=1`)
- requires `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` context ✓

The host path stays as a runtime fallback for non-CUDA / non-hybrid /
multi-seq contexts. Once Sprint 008 phase gates pass, the env-toggle
default flips: `LLAMA_SPEC_VRAM_CKPT=1` becomes the docker-compose
default, with `=0` available as the escape hatch.

## Implementation

### Phase 1 — Extend `vram_seq_checkpoint` with cells[] snapshot

**Files (fork)**:
- `src/llama-vram-checkpoint.h`:
  - Add private fields: `std::vector<llama_memory_recurrent::kv_cell>
    cells_snapshot`, `uint32_t head_snapshot`, `uint32_t used_snapshot`.
  - May need `friend class llama_memory_recurrent` or a `state_snapshot()` /
    `state_restore_snapshot()` method on `llama_memory_recurrent` —
    pick whichever is least invasive.
- `src/llama-vram-checkpoint.cpp`:
  - In `save()`: after the cudaMemcpyAsync loops, copy `mem_recr->cells`
    into `cells_snapshot`, save `head` / `used`. Then sync.
  - In `restore()`: assign `mem_recr->cells = cells_snapshot;
    mem_recr->head = head_snapshot; mem_recr->used = used_snapshot;`
    *before* the cudaMemcpyAsync (or order doesn't matter since meta
    is host-side; pick what reads cleanly).

**Tasks**:
- [ ] Add the fields + ctor zero-init.
- [ ] Wire the host-side cells[] vector copy in save().
- [ ] Wire the host-side cells[] vector restore in restore().
- [ ] Verify size: `cells.size() = mem_recr->size`. ~120 cells per
      single-seq qwen3.6 setup × ~96 bytes each = ~12 KB (negligible
      vs 149 MiB tensor data).

**Phase gate**: clean build, no test failures.

### Phase 2 — Smoke test

**Tasks**:
- [ ] `LLAMA_SPEC_VRAM_CKPT=1 docker compose --profile qwen up -d`
- [ ] Send 3 sequential quicksort requests. Verify all return HTTP 200,
      coherent completions.
- [ ] Verify `spec_t_ckpt_save_us / spec_n_ckpt_save` ratio drops
      from ~35 ms (Sprint 006) to <1 ms (target: HBM-bound,
      ~149 MiB / 3 TB/s ≈ 50 µs).
- [ ] Verify per-request `draft_n_generated` is independent across
      the 3 trials (F-022 sanity check at sequential request scale).

**Phase gate**: 3-of-3 sequential requests succeed; ckpt copy time
per save < 1 ms; per-request acceptance numbers are independent
(not cumulative).

### Phase 3 — Re-run E2 sweep + capture E5

**Tasks**:
- [ ] `DRAFT_N_MAX={4,8,16}` × full 5-prompt × 3-trial bench using
      `scripts/run_sprint006_experiment.sh` (rename / extend for
      Sprint 008, but the structure is the same). Save artifacts
      under `docs/sprints/SPRINT-008-dflash-experiments/E2-rerun/`.
- [ ] Capture E5 rejection-position histograms per N.
- [ ] Compare side-by-side with Sprint 006: did the optimal N shift?
      Did any prompt cross ≥1.0× or ≥1.3× target-only?

**Phase gate**: full sweep completes without crashes; data populated
per-N; comparison table written.

### Phase 4 — Choose DRAFT_N_MAX default + republish

**Tasks**:
- [ ] Pick the new `DRAFT_N_MAX` default (4 / 8 / 16 / adaptive).
      Update `docker/entrypoint.sh`.
- [ ] If gates are now met on more prompts: update
      `docs/BENCHMARK-REPORT.md` with new headline numbers and
      Sprint 008 callout.
- [ ] Update `README.md` Speculative Decoding section if guidance
      changes.
- [ ] Flip `LLAMA_SPEC_VRAM_CKPT` default to `1` in
      `docker-compose.yml` (the `=0` escape hatch stays).

**Phase gate**: README + BENCHMARK-REPORT consistent with
Sprint 008 numbers; default flag flipped.

### Phase 5 — Findings + recommend next sprint

**Tasks**:
- [ ] Write `docs/sprints/SPRINT-008-dflash-FINDINGS.md` with
      hypothesis verdicts (did VRAM-shadow give the expected
      speedup? Did the optimal N change? Did F-022's per-request
      counter hold up under multi-request load?).
- [ ] Write `docs/sprints/SPRINT-008-dflash-FOLLOWUPS.md` for
      execution-discovered items (if any).
- [ ] Sprint 009 recommendation: EAGLE3 productionization (the
      previously deleted Sprint 008-EAGLE3 plan can be revived as
      Sprint 009-eagle3 if useful) vs further DFlash optimization
      (e.g., adaptive block size, draft distillation).

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/llama-vram-checkpoint.h` | Modify (fork) | Add cells_snapshot fields |
| `src/llama-vram-checkpoint.cpp` | Modify (fork) | Snapshot/restore cells[] alongside D->D copy |
| `src/llama-memory-recurrent.h` | Modify (fork) | Friend class or state_snapshot() method |
| `tools/server/server-context.cpp` | No change | Wire-up already in place from Sprint 007 |
| `docker-compose.yml` | Modify (repo) | Flip LLAMA_SPEC_VRAM_CKPT default to 1 |
| `docker/entrypoint.sh` | Modify (repo) | Update DRAFT_N_MAX default per Phase 4 |
| `docs/sprints/SPRINT-008-dflash-experiments/` | Create | E2 re-run + E5 histograms |
| `docs/sprints/SPRINT-008-dflash-FINDINGS.md` | Create | Outcomes + Sprint 009 recommendation |
| `docs/BENCHMARK-REPORT.md` | Modify (repo) | Republish numbers if gates pass |
| `README.md` | Modify (repo) | Update DFlash guidance |

---

## Definition of Done

### Hard gates

1. `vram_seq_checkpoint::save()` and `restore()` snapshot/restore the
   recurrent memory's `cells[]` metadata in addition to tensor bytes.
   Verified by: 3-of-3 sequential P1 quicksort requests pass on
   `LLAMA_SPEC_VRAM_CKPT=1` (was 1-of-N in Sprint 007).
2. Per-save copy time < 1 ms (vs ~35 ms host path measured in
   Sprint 006).
3. Full E2 sweep at N={4,8,16} re-run on the working VRAM build.
   Comparison table vs Sprint 006 written.
4. SPRINT-008-FINDINGS.md exists with hypothesis verdicts and
   Sprint 009 recommendation.

### Soft gates

- `DRAFT_N_MAX` default updated in `docker/entrypoint.sh` if Phase 3
  data warrants.
- BENCHMARK-REPORT republished if ≥1.0× or ≥1.3× target-only gate
  passes on >2 prompts.
- `LLAMA_SPEC_VRAM_CKPT` default flipped to 1 in docker-compose.yml.
- README's DFlash guidance reflects current numbers.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `cells[]` snapshot triggers a different bug we don't yet see | Medium | Medium | Phase 2 smoke test catches it before Phase 3 invests in full sweep |
| Friend-class or method exposure forces an awkward API change | Low | Low | Worst case: just `#include` the recurrent header into vram_checkpoint.cpp |
| Re-run E2 still shows median below ≥1.0× even with VRAM cheap | Possible | Low (information win) | Sprint 008 outcome: file as "DFlash architecture limit, recommend EAGLE3 or distillation" |
| `cells[]` vector deep-copy itself is too slow to matter | Low | Low | ~12 KB host-side vector copy is negligible; if it shows up in profile, switch to memcpy on raw pointers |

---

## Open questions

1. Should the snapshot live inside `vram_seq_checkpoint` (current
   plan) or as a separate `mem_recr->state_snapshot()` method?
   Current plan keeps everything in one class for grep-ability.
2. Once `LLAMA_SPEC_VRAM_CKPT=1` is the default, should we remove the
   host-path code entirely, or keep it as the non-CUDA / non-hybrid
   fallback? Recommend keep — it's the only path on multi-seq /
   non-hybrid setups.
3. Does Sprint 008 close the DFlash track (median gate met) or does it
   surface a deeper issue (drafts can't predict prose)? Sprint 008's
   Phase 5 makes this call based on numbers.
