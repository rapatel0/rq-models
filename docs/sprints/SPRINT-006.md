# Sprint 006: EAGLE3 Productionization

**Status**: Planning (draft)
**Created**: 2026-04-27
**Depends on**: Sprint 005 (L4 numbers published, forced-rejection gate closed)
**Target hardware**: RTX 5090 (32 GB), 123 GB system RAM
**Estimated effort**: 1 week single-engineer
**Branches**:
- Repo: `rapatel0/rq-models` `sprint/006-eagle3` (off `main` post-Sprint-005)
- Fork: minimal — graph already in tree from Phase 3 cherry-pick

---

## Overview

The Sprint 004 cherry-pick of PR #22105 squash-merged the entire shared
tree of #22105 (DFlash) and #18039 (EAGLE3). The full EAGLE3 model graph
(`src/models/eagle3.cpp`, +186 LOC) is already in fork; the runtime
`--eagle3` flag is wired through `common/arg.cpp`; the convert path
supports EAGLE3 drafts. **Sprint 006 is pure repo-side**: profile,
entrypoint dispatch, draft conversion, validation harness, benchmark
publish.

EAGLE3 is autoregressive 1-token-per-step (vs DFlash's 16-token block).
Different best-fit prompt profile — generally narrower acceptance
distribution, more consistent single-stream throughput. Useful as a
fallback when DFlash drafts aren't available for a target architecture
or when prompt regime is hostile to block-diffusion (e.g., highly
repetitive or formulaic outputs).

Out of scope: any fork-side EAGLE3 changes beyond bug fixes; multi-slot
batched-draft (Sprint 007); cross-architecture EAGLE3 portability (e.g.,
Llama-3 EAGLE3 drafts).

---

## Use Cases

1. **Operator picks between DFlash and EAGLE3** based on workload. The
   sprint outputs comparable tok/s + acceptance numbers on the same
   5-prompt set under the same conditions.

2. **EAGLE3 as DFlash fallback**: When a target has only EAGLE3 drafts
   available (broader community ecosystem), the operator can opt in via
   `make run-qwen-eagle3` rather than waiting for DFlash drafts to drop.

3. **EAGLE3 + 27B as preview path**: If z-lab's 27B-DFlash drafts iterate
   slowly and EAGLE3-flavored 27B drafts are more stable, Sprint 006
   gives operators a more-stable alternative.

---

## Architecture

```
                    Existing in fork (no fork change):
        ┌────────────────────────────────────────────────────────┐
        │  src/models/eagle3.cpp           full graph (cherry-pick)│
        │  common/arg.cpp                  --eagle3 runtime flag   │
        │  common/speculative.cpp          common_speculative_init │
        │                                  _eagle3 dispatch        │
        │  convert_hf_to_gguf.py           Eagle3Model class       │
        │  examples/speculative-simple/    --eagle3 in CLI         │
        └────────────────────────────────────────────────────────┘
                                │
                                ▼
                  Sprint 006 deliverables (repo-side):
        ┌────────────────────────────────────────────────────────┐
        │  docker/entrypoint.sh             SPECULATIVE_MODE=eagle3│
        │                                   path; --eagle3 flag    │
        │  docker-compose.yml               qwen-eagle3,           │
        │                                   qwen36-27b-eagle3      │
        │                                   profiles               │
        │  Makefile                         run-* + bench-eagle3   │
        │  scripts/convert_eagle3_drafts.sh Mirror of dflash       │
        │                                   converter for EAGLE3   │
        │                                   draft repos            │
        │  scripts/bench_speculative.py     +eagle3 leg            │
        │  scripts/validate_dflash.py       (rename →              │
        │                                   validate_speculative.py│
        │                                   or add --mode eagle3)  │
        │  docs/BENCHMARK-REPORT.md §10.x   EAGLE3 sub-section     │
        └────────────────────────────────────────────────────────┘
```

### Phase ordering

```
Phase 0: EAGLE3 draft sourcing             (~15%, 1 day)
Phase 1: convert_eagle3_drafts.sh          (~15%, 1 day)
Phase 2: entrypoint dispatch + profile     (~20%, 1.5 days)
Phase 3: validation harness extension      (~20%, 1.5 days)
Phase 4: benchmark + publish               (~20%, 1.5 days)
Phase 5: docs + sprint outcome             (~10%, 0.5 days)
```

---

## Implementation

### Phase 0: EAGLE3 draft sourcing

**Goal**: Identify the right EAGLE3 draft safetensors for our two targets.

**Tasks**:
- [ ] Survey HF for `eagle3` + `qwen3.6` tags. Likely candidates: `z-lab`,
      `lmsys`, community converters. Pin candidate repos + SHAs in this
      sprint doc before starting Phase 1.
- [ ] If no Qwen3.6-targeted EAGLE3 draft exists: descope to Qwen3.5-27B
      EAGLE3 (drafts more available for older targets) and document the
      gap as a Sprint 007 followup.
- [ ] Note license + access status for each candidate (gated/open).

**Phase gate**: A pinned safetensors path for at least one Qwen3 target.

### Phase 1: `convert_eagle3_drafts.sh`

**Goal**: Mirror of `convert_dflash_drafts.sh` but for EAGLE3 schema.

**Files**:
- `scripts/convert_eagle3_drafts.sh` — NEW. Same idempotent pattern as
  the dflash converter: download draft safetensors, download target
  tokenizer files, run `convert_hf_to_gguf.py`, publish into
  `llm-models` volume.
- `Makefile` — `convert-eagle3-drafts` target.

**Tasks**:
- [ ] Clone `convert_dflash_drafts.sh` shape, swap pinned repos.
- [ ] Verify `convert_hf_to_gguf.py` EAGLE3 path uses
      `--target-model-dir` the same way DFlash does (Sprint 004 finding).
- [ ] First conversion: produce `Qwen3.6-XXB-EAGLE3-bf16.gguf` and
      publish.
- [ ] Smoke load via `llama-cli --model <draft> -ngl 0 -n 0` to verify
      tensor names.

**Phase gate**: At least one EAGLE3 draft GGUF produced and loads in
`llama-speculative-simple --eagle3`.

### Phase 2: entrypoint dispatch + compose profile

**Files**:
- `docker/entrypoint.sh` — extend `SPECULATIVE_MODE` validator from
  `target-only|autoregressive|dflash` → `target-only|autoregressive|dflash|eagle3`.
  Command builder adds `--eagle3` flag when `SPECULATIVE_MODE=eagle3`.
- `docker-compose.yml` — `qwen-eagle3` (35B) and/or `qwen36-27b-eagle3`
  (27B) profiles. Both gated by `PREVIEW=1` until Sprint 006 publishes
  L4 numbers — can drop the gate at Sprint outcome if numbers clear bar.
- `Makefile` — `run-qwen-eagle3[-bg]`, `run-qwen36-27b-eagle3[-bg]`,
  aggregate stop/logs/clean updated.

**Tasks**:
- [ ] Extend `SPECULATIVE_MODE` validation case statement.
- [ ] Add `--eagle3` to the `CMD` builder when mode is eagle3.
- [ ] New compose services. Single-slot default, planar3 KV (mirror of
      dflash profiles).
- [ ] Update `MODELS` registry with `qwen3.6-XXB-eagle3` entries pointing
      at `local/...` (same pattern as dflash).
- [ ] Smoke `make run-qwen36-27b-eagle3`. `/health`, one greedy
      completion. `make stop`.

**Phase gate**: At least one EAGLE3 profile boots and serves a coherent
greedy completion.

### Phase 3: validation harness extension

**Files**:
- `scripts/bench_speculative.py` — extend `LEGS = [...]` to include
  `eagle3`. Update finalize() rendering to handle 4 legs.
- `scripts/validate_dflash.py` → consider rename to
  `validate_speculative.py` and add `--mode {dflash,eagle3}`. Or just
  duplicate as `validate_eagle3.py` if rename is too disruptive.
- `tests/test_speculative.py` — add EAGLE3-flavored tests parallel to
  the DFlash ones.

**Tasks**:
- [ ] Decide rename vs new file (recommend new file `validate_eagle3.py`
      to keep diff small; can refactor later).
- [ ] Run `validate_eagle3.py --reference none` on the 5-prompt set to
      check L2 greedy equivalence.
- [ ] Run `validate_eagle3.py --reference zlab` if a pytorch reference
      exists for EAGLE3 (likely yes from z-lab).

**Phase gate**: L2 (greedy match 256/256 on 3+ prompts) passes for
EAGLE3.

### Phase 4: benchmark + publish

**Files**:
- `Makefile` — `bench-eagle3-leg LEG=eagle3`, `bench-eagle3-all`.
- `docs/sprints/SPRINT-006-L4-results.json`, `-summary.md` (NEW).
- `docs/BENCHMARK-REPORT.md` — `§10.x EAGLE3` subsection.

**Tasks**:
- [ ] Run the 5-prompt × 3-trial benchmark on `qwen36-27b-eagle3`.
      Compare to Sprint 005's DFlash and target-only numbers (same
      prompts, same seed, same temp, same trials).
- [ ] Publish a 4-way comparison table (target-only / autoregressive /
      DFlash / EAGLE3) on the 27B.
- [ ] If 35B EAGLE3 draft exists: same exercise on `qwen-eagle3`.

**Phase gate**: 4-way comparison table written to BENCHMARK-REPORT.md
§10. No EAGLE3 cell shows >5% regression vs target-only on any prompt
(soft gate; correctness-only).

### Phase 5: docs + sprint outcome

**Tasks**:
- [ ] README "Speculative Decoding" section: add EAGLE3 row to the
      profile table; one-line guidance on when to pick EAGLE3 vs DFlash.
- [ ] `SPRINT-004-FOLLOWUPS.md` F-008 → closed.
- [ ] `SPRINT-006-FOLLOWUPS.md` (NEW) for execution-discovered items.
- [ ] Sprint marked complete.

---

## Definition of Done

### Hard gates

1. **EAGLE3 boots** on at least one Qwen3 target via the new compose
   profile.
2. **L2 greedy equivalence**: 256/256 token match on at least 3 of 5
   prompts.
3. **No regression vs target-only**: per-prompt EAGLE3 tok/s ≥ 0.95×
   target-only on the 5-prompt set.
4. **4-way comparison table** published in BENCHMARK-REPORT.md §10.

### Soft gates

- **EAGLE3 ≥1.0× target-only median**: not gated, but anything below is
  worth a footnote.
- **EAGLE3 vs DFlash crossover identified**: prompt regimes where each
  wins, with one-line guidance in README.

### Code hygiene

- Standard Sprint code hygiene from Sprint 004 carries forward.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No Qwen3.6 EAGLE3 draft exists in the wild | Medium | High | Phase 0 explicitly checks first; descope to Qwen3.5-27B EAGLE3 if needed |
| EAGLE3 conversion path in fork's `convert_hf_to_gguf.py` has Qwen3.6 tokenizer issue (same chkhsh problem as Sprint 004) | Low | Low | The `qwen35`/`qwen36` chkhsh mapping is already in fork (commit `1c9b77fdd`); EAGLE3 would inherit |
| EAGLE3 acceptance characteristically degrades with thinking-on like DFlash does | Medium | Medium | Document; report both thinking-on and no-think (the latter as opt-in only) |
| 4-way comparison runs out of GPU time | Low | Low | Per-leg compose-up/stop is the bottleneck, ~5 min × 4 legs = ~20 min |

---

## Dependencies

- Sprint 005 must be complete (forced-rejection gate closed, L4 numbers
  published) so the 4-way comparison has a stable DFlash baseline.
- At least one EAGLE3 draft safetensors source for Qwen3.x.

---

## Open Questions

1. **EAGLE3 draft source pin**: Phase 0 task. Sprint can't start without.

2. **Is EAGLE3 single-slot only like DFlash?** Need to verify the
   server-context.cpp slot ↔ speculative integration handles EAGLE3 the
   same as DFlash. Tentatively yes (same `common_speculative_init`
   dispatch).

3. **Should EAGLE3 profile be PREVIEW-gated?** Tentatively yes for first
   ship, drop the gate at Sprint outcome if numbers clear the soft
   gates.
