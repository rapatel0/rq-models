# Sprint 009-dflash: DFlash imatrix calibration + Q4 default

> **Track suffix**: `-dflash`. Does not merge to `main`.

**Status**: Planning (2026-05-04)
**Sprint type**: Tooling + measurement
**Created**: 2026-05-04
**Depends on**: Sprint 008-dflash close (BF16 N=2 default at median 1.21×;
F-027 imatrix gap documented; Q4 plain regresses Hamlet to 0.57×)
**Estimated effort**: ~2-3 days single-engineer

**Branches**:
- Repo: `sprint/009-dflash` (off `sprint/008-dflash`)
- Fork: `feature/sprint-004-rebase-dflash` continues from `526097eed`

---

## Overview

Sprint 008 established that:
1. **VRAM-shadow ckpt + N=2** delivers median 1.21× and worst-case
   0.91× — the speculative pipeline is no longer the bottleneck.
2. **Draft generation overhead** is now the ceiling. The 1.7B BF16
   draft costs ~7-10 ms per round; quantizing it to Q4_K_M would cut
   that to ~2-3 ms.
3. **Plain Q4_K_M regresses prose catastrophically** (Hamlet 0.57× vs
   BF16 0.91×, with 2/3 trials at 0% draft acceptance) — exactly the
   class of failure imatrix calibration prevents.

Sprint 009 closes F-027: build a calibration path that produces a
proper imatrix for the DFlash draft model, requantize to Q4_K_M with
that imatrix, re-bench, and (if quality holds) flip the default.

The hard part isn't the quantize step — it's that `llama-imatrix`
asserts on `model.target_tok_embd != nullptr` (`src/models/dflash.cpp:39`),
which means the DFlash decoder can't run without a target context. We
need to either (a) modify llama-imatrix to accept a target model and
run the DFlash speculation flow during calibration, or (b) build a
side tool that collects activation statistics through the speculative
server during a real workload run.

## Use Cases

1. **Operators get faster speculative without quality risk**: post-Sprint
   009 the docker-compose default could be Q4 imatrix-calibrated (~3×
   faster draft gen, preserved acceptance). Median should land 1.4-1.5×
   based on the math from Sprint 008 (Q4 gen cost drops by ~5 ms × 100
   rounds = 500 ms saved per 200-token output).
2. **The DFlash track has a clean answer** to "should we go to EAGLE3 or
   distillation next?" — we'll know whether quantizing the existing draft
   is enough or whether the architecture (or draft size) is the real
   ceiling.
3. **Future quantization experiments** become repeatable. Today's Q4 path
   is one-shot manual; Sprint 009 produces a `convert_dflash_drafts.sh`
   extension that handles imatrix capture + Q4 publish in one command.

## Architecture

```text
fork pin (526097eed baseline)
  ↓
Phase 1: Choose calibration approach (modify llama-imatrix vs server-side
        collector vs ask z-lab for theirs). Decide based on time/risk.
  ↓
Phase 2: Implement chosen approach. Capture imatrix from a calibration
        run on the DFlash speculative pipeline.
  ↓
Phase 3: Quantize Qwen3.6-27B-DFlash-bf16.gguf → Q4_K_M with imatrix.
        Verify quantize completes without shape mismatch errors.
  ↓
Phase 4: Smoke test (3 sequential prompts) — verify Q4-imatrix doesn't
        replicate the BF16-Q4-plain Hamlet 0% acceptance failure.
  ↓
Phase 5: Re-bench at N={2,4} on Q4-imatrix. Compare to BF16 N=2 and
        Q4-plain N=2.
  ↓
Phase 6: If acceptance held: flip docker-compose default to Q4-imatrix,
        republish BENCHMARK-REPORT. If not: document, keep BF16 default,
        ship Q4-imatrix as opt-in.
  ↓
Phase 7: Findings + Sprint 010 recommendation (distillation? EAGLE3?
        close DFlash track?).
```

## Implementation

### Phase 1 — Pick the calibration approach

Three candidates, in increasing engineering cost:

**Option A (cheapest): Ask z-lab for their training-time imatrix.**
The DFlash drafts on z-lab's HF org were almost certainly imatrix-aware
when trained — z-lab uses unsloth-style calibration. If they publish
the imatrix file (or have it on their internal storage), we can use it
directly. Cost: 1 GitHub issue / email. Risk: they may not respond /
may not have it preserved.

**Option B: Modify `llama-imatrix` to load a target alongside.**
Add `--target-model PATH` flag. When DFlash draft is loaded, also load
the target, hook them up the way speculative-simple does, and run the
calibration corpus through the *speculative* path. Activation stats
get collected on the draft. Cost: ~1 day of fork work + bench.
Risk: medium — touches the imatrix tool's batch loop.

**Option C: Server-side imatrix collector.**
Add an `IMATRIX_OUTPUT=path` env to llama-server that, when set,
collects activation stats during real speculative runs and writes
imatrix on shutdown. Cost: ~2 days + integration. Risk: higher
(touches request hot path).

**Recommended order**: try A first (1 day), fall back to B if no
response within 24h, fall back to C only if B turns out to be too
invasive.

**Tasks**:
- [ ] File HF discussion / issue on `z-lab/Qwen3.6-27B-DFlash` asking
      if they have imatrix.
- [ ] If yes: download, skip to Phase 3.
- [ ] If no within 24h: scope Option B.

### Phase 2 — Implement calibration (Option B path)

If we land on Option B:

**Files (fork)**:
- `tools/imatrix/imatrix.cpp`:
  - Add `--target-model` CLI arg.
  - When draft loads as DFlash, also init a target context and
    `llama_set_dflash_*` setup analogous to `common/speculative.cpp`'s
    DFlash path.
  - Run calibration corpus through the speculative pipeline (target
    decode → draft encode → draft decode), collect stats per draft
    forward.
- `tools/imatrix/CMakeLists.txt`: ensure imatrix links libllama-common
  (it probably already does).

**Tasks**:
- [ ] Add the flag plumbing.
- [ ] Wire target model alongside the draft.
- [ ] Run `llama-imatrix --target-model <Qwen3.6-27B-Q4_K_XL.gguf>
      --model <DFlash-bf16.gguf> --imatrix-output <path>
      --file <calib.txt>` end-to-end.
- [ ] Validate output imatrix has entries for all DFlash decoder
      tensors (tensor names match those in the BF16 GGUF).

**Phase gate**: imatrix file produced; tensor names match the DFlash
draft's; `llama-quantize --imatrix <new.imatrix> ...` runs without
shape mismatch errors.

### Phase 3 — Quantize with imatrix

**Tasks**:
- [ ] `llama-quantize --imatrix <DFlash.imatrix>
      Qwen3.6-27B-DFlash-bf16.gguf
      Qwen3.6-27B-DFlash-Q4_K_M-imatrix.gguf Q4_K_M`
- [ ] Register a new model key `qwen3.6-27b-dflash-q4-imatrix` in
      `docker/entrypoint.sh` MODELS table.
- [ ] Update `scripts/convert_dflash_drafts.sh` to optionally capture
      imatrix + publish a Q4 variant alongside BF16. Idempotent.

### Phase 4 — Smoke test

**Tasks**:
- [ ] Boot `LLAMA_SPEC_VRAM_CKPT=1 DRAFT_N_MAX=2
      DRAFT_MODEL_NAME=qwen3.6-27b-dflash-q4-imatrix
      docker compose --profile qwen up -d`.
- [ ] 3 sequential Hamlet prompts (the prompt that broke Q4-plain).
      Verify all return ≥60% draft acceptance and tps ≥ BF16 baseline.

**Phase gate**: 3-of-3 Hamlet trials succeed with draft acceptance
≥60% and tps within ±10% of BF16 baseline. (If acceptance is much
higher than BF16, suspicious — likely an artifact.)

### Phase 5 — Bench Q4-imatrix at N={2,4}

**Tasks**:
- [ ] `PROFILE=qwen NO_THINK=1 LLAMA_SPEC_VRAM_CKPT=1
      DRAFT=qwen3.6-27b-dflash-q4-imatrix
      ./scripts/run_sprint008_experiment.sh
      E2-rerun-N2-q4-imatrix
      DRAFT_N_MAX=2
      DRAFT_MODEL_NAME=qwen3.6-27b-dflash-q4-imatrix`
- [ ] Same for N=4.
- [ ] Side-by-side table vs BF16 N=2 / Q4-plain N=2 / N=4.

**Phase gate**: Q4-imatrix N=2 worst-case ≥0.85× (vs BF16 0.91× — a
modest regression is acceptable as a quality/cost trade-off, but
collapsing to 0.57× like Q4-plain isn't).

### Phase 6 — Default flip + republish

**Tasks (only if Phase 5 gates pass)**:
- [ ] Flip `docker-compose.yml` `DRAFT_MODEL_NAME` default to
      `qwen3.6-27b-dflash-q4-imatrix`.
- [ ] Update README + BENCHMARK-REPORT with Sprint 009 numbers.

### Phase 7 — Findings + Sprint 010 rec

**Tasks**:
- [ ] `docs/sprints/SPRINT-009-dflash-FINDINGS.md` with:
  - Did Option A / B / C fire?
  - imatrix-aware Q4 numbers vs BF16 / Q4-plain
  - Whether ≥1.3× hard gate finally clears
- [ ] If gate clears: DFlash track WIN. Sprint 010 = "ship-and-monitor"
- [ ] If gate misses: distillation or EAGLE3 next. Pick based on
      remaining gap pattern.

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `tools/imatrix/imatrix.cpp` | Modify (fork) | `--target-model` flag for DFlash imatrix calibration (Option B) |
| `scripts/convert_dflash_drafts.sh` | Modify (repo) | Add optional imatrix capture + Q4 publish |
| `docker/entrypoint.sh` | Modify (repo) | Register `qwen3.6-27b-dflash-q4-imatrix` model key |
| `docker-compose.yml` | Modify (repo) | Flip default if Phase 5 gates pass |
| `docs/sprints/SPRINT-009-dflash-experiments/` | Create | Q4-imatrix bench artifacts |
| `docs/sprints/SPRINT-009-dflash-FINDINGS.md` | Create | Outcomes + Sprint 010 rec |
| `docs/BENCHMARK-REPORT.md` | Modify (repo) | Republish if gate clears |
| `README.md` | Modify (repo) | Update DFlash guidance |

---

## Definition of Done

### Hard gates

1. A working DFlash imatrix exists, produced via Option A / B / C —
   either from z-lab or from our calibration tooling.
2. `llama-quantize --imatrix <DFlash.imatrix>` runs without shape
   mismatch.
3. Q4_K_M-imatrix DFlash draft passes 3-of-3 Hamlet smoke test with
   ≥60% draft acceptance.
4. SPRINT-009-FINDINGS.md exists with Q4-imatrix vs BF16 vs Q4-plain
   side-by-side table.

### Soft gates

- Q4-imatrix N=2 worst-case ≥0.85×.
- Q4-imatrix N=2 median ≥1.3× — would clear the Sprint 005 hard gate.
- Default flipped to Q4-imatrix.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| z-lab doesn't have imatrix preserved | High | Low (Option B fallback) | Already plan B in Phase 1 |
| `llama-imatrix` modification too invasive | Medium | Medium | If it's >1.5 days of fork work, descope to Option C or close Sprint 009 with imatrix gap unfilled |
| Q4-imatrix still regresses on Hamlet | Possible | Medium | Worst case: keep BF16 default, ship Q4-imatrix as opt-in for code-heavy. Same as Sprint 008 Q4-plain status. |
| Calibration corpus matters more than tooling | Possible | Low | Use bartowski-style calib corpus (already downloaded in /tmp/imatrix-calibration.txt); document the choice |

---

## Open questions

1. Does z-lab publish imatrix? **Phase 1 decides**.
2. Is the DFlash draft architecture stable enough that adding imatrix
   support to `llama-imatrix` is worthwhile, or will the architecture
   churn under EAGLE3 productionization (Sprint 010)? Recommend
   investing the day if the speedup ceiling matters short-term.
3. Should the Q4-imatrix bench also test N=3 / N=8 / N=16, or just
   stick with N=2 / N=4? Recommend N=2 + N=4 only — Sprint 008 already
   showed monotonic decline past N=4.
