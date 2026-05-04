# Sprint 008-dflash Follow-ups

Items discovered during Sprint 008 execution that need future work.

## F-027: Imatrix calibration for DFlash drafts

**What**: `llama-imatrix` cannot run DFlash drafts standalone — the
DFlash decoder graph asserts `model.target_tok_embd != nullptr`
(`/src/src/models/dflash.cpp:39`), which is only set during speculative
decoding when the target context is provided. So we can't compute an
imatrix for the DFlash draft via the standard tooling.

**Why**: Discovered when attempting Sprint 008 Phase 4b (Q4
quantization). Bartowski's pre-computed Qwen3.6-27B imatrix
(`bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-imatrix.gguf`) is
not compatible — `llama-quantize` rejects it because the DFlash
decoder's `attn_output` tensor has shape (4096, 5120) but the Qwen3.6
target's is (6144, 5120). The DFlash decoder is z-lab's custom ~1.7B
architecture, not a slice of Qwen3.6.

Without imatrix calibration, plain Q4_K_M of the DFlash draft has a
catastrophic worst-case regression: Hamlet drops from 0.91× (BF16) to
**0.57× (Q4 plain)** — and 2 of 3 trials hit 0% draft acceptance.
Median improves slightly (1.21 → 1.28) but the failure mode is
unacceptable for a default.

**Severity**: **Important**. Without imatrix, Q4 quantization is opt-in
for code-heavy workloads only. With imatrix, Q4 might unlock a stable
1.3-1.5× median (with the 3.4× smaller draft → ~3× faster draft gen).

**Fix paths**:
1. Modify `llama-imatrix` to accept a target model and run DFlash
   draft inside the speculative-decoding flow during calibration.
   Requires changes in fork.
2. Ask z-lab to publish their training-time imatrix. Easiest if they
   do; we can't construct it ourselves without the training data.
3. Build a custom imatrix collector inside `llama-server` —
   instrument the speculative path to record activation statistics
   during real workload runs. Most ergonomic but largest engineering
   surface.
4. Skip Q4 entirely; live with BF16 draft cost. Current default.

**Suggested sprint**: Whenever DFlash track returns to active work, OR
if z-lab publishes an imatrix.

**Files**:
- `tools/imatrix/imatrix.cpp` (would need DFlash awareness for option 1)
- `src/models/dflash.cpp` (relax the target_tok_embd assert? unsafe)
- z-lab/Qwen3.6-27B-DFlash HF repo (option 2: ask upstream)

## F-028: docker-compose DRAFT_MODEL_NAME hardcoded for qwen profiles

**What**: `docker-compose.yml` line 78 was `DRAFT_MODEL_NAME: qwen3.6-27b-dflash`
hardcoded; same at line 125 for qwen36. Env-level overrides
(`DRAFT_MODEL_NAME=foo docker compose up`) were silently dropped
because the compose value won.

**Why**: Discovered during Sprint 008 Phase 4b Q4 bench attempt. First
run produced numbers identical to BF16 even with the env override, and
only the run.log header noted the override variable. The bench loaded
BF16 anyway.

**Severity**: Important (silently corrupts experiment results).
Fixed in Sprint 008 commit (line 78/125 changed to
`${DRAFT_MODEL_NAME:-qwen3.6-27b-dflash}`).

**Files** (already fixed in Sprint 008):
- `docker-compose.yml`

## F-029: bench harness can't finalize when AR leg can't run

**What**: `scripts/bench_speculative.py --finalize` errors with
"missing legs: ['target-only', 'autoregressive']" when only the
dflash leg has been written. The autoregressive leg can't load
DFlash GGUFs (they have DFlash architecture metadata; AR mode loads
them as plain causal LMs which fails). So when testing a DFlash-only
draft, we can't get the standard summary table.

**Why**: Discovered when running Q4 bench. Worked around by writing
the summary manually for the Q4 N=2 result.

**Severity**: Nice-to-have. Doesn't block measurement, just makes the
artifact format inconsistent.

**Suggested fix**: Add `--legs dflash` flag to bench finalize so it
emits a single-leg summary if AR/target-only weren't run. Or: have
the experiment driver `run_sprint008_experiment.sh` accept a
`LEGS=dflash` env to skip the failing leg.

**Files**:
- `scripts/bench_speculative.py` (finalize logic)
- `scripts/run_sprint008_experiment.sh` (driver)

---

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| F-027: DFlash imatrix calibration gap | Important | When DFlash track reactivates, or if z-lab ships one | `tools/imatrix/imatrix.cpp`, `src/models/dflash.cpp`, z-lab HF repo |
| F-028: hardcoded DRAFT_MODEL_NAME silent override (FIXED) | Important | Done in Sprint 008 | `docker-compose.yml` |
| F-029: bench finalize requires all 3 legs | Nice-to-have | Whenever DFlash track resumes | `scripts/bench_speculative.py`, `scripts/run_sprint008_experiment.sh` |
