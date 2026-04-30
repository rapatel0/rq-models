# E4 — adaptive skip heuristic (K consecutive partials)

**Date**: 2026-04-30
**Profile**: qwen + thinking-off
**Env override**: `LLAMA_SPEC_ADAPTIVE_SKIP_K={2,3}`
**Fork pin**: `4ef60a057` (v8 + Sprint 006 instrumentation + F-018 v1 early-return fix)

## Method

Replace the always-skip-on-partial behavior with skip-after-K-
consecutive-partials-at-same-pos_max. K=1 is the v8 default. Test K=2
and K=3.

Required a fork-side fix first (`F-018 v1` commit `4ef60a057`):
DFlash and EAGLE3 draft() now early-return on `n_new < 1` instead of
asserting. Without this, K>=2 crashes immediately because the FIRST
partial doesn't fire the skip flag, the rollback advances
dflash_n_past = prompt.size(), and the second draft attempt sees
n_new = 0 and aborts the server.

## Results

| Prompt | Baseline (K=1) tps | K=2 tps | K=3 tps |
|--------|---------------:|--------:|--------:|
| P1 quicksort | 56.2 | 53.2 | 53.5 |
| P2 Pythagoras | 14.3 | 14.1 | 14.2 |
| P3 DC trip | 10.9 | 10.9 | 10.9 |
| P4 Hamlet | 11.0 | 10.9 | 11.0 |
| P5 SQL | 15.1 | 15.0 | 15.0 |

## Verdict — **REFUTED for temp=0 sampling**

Per the Phase 5 decision rule:

> "If heuristic improves median qwen tok/s by ≥20% vs v8 baseline
> with zero hangs in soak → Sprint 007 productizes this as the
> F-018 fix."

K=2 and K=3 produce **near-identical tps** to K=1 across every prompt
(deltas <5%, sometimes negative). The hypothesis that adaptive
skip would recover lost throughput is **refuted at temp=0**.

## Why it fails — deterministic-partial dynamic

With deterministic sampling (temp=0, top_k=1, seed=42), the verify
pass against the same target KV state produces the **same partial
acceptance** every time. K=2 means we *try a second round* before
falling back to single-token decode; that second round produces the
same partial, wasting an extra ~38% of the round on a checkpoint
save (per E3 numbers) for zero forward progress.

For K>=2 to be a net win, the speculative path would need to
*sometimes* succeed on the retry. That requires either:
- Non-deterministic sampling (temp>0, top_k>1), where the second
  round might sample differently
- A different draft state (e.g., shorter block, different prefix
  warm-up)
- Non-deterministic verify (not currently the case)

For Sprint 005's regime (temp=0, top_k=1), **K is the wrong knob**.

## Companion finding from #save / #restore counters

K=2 and K=3 show **identical save+restore counts** to baseline
(K=1). That confirms the heuristic isn't actually firing
differently in practice — every partial round still results in the
same checkpoint save (it happens before verify, regardless of skip
policy) and the same restore (we still partial-accept and
roll back). The skip flag only changes whether the NEXT round is
single-token or another speculative attempt; it doesn't reduce
checkpoint cost.

## Implications for Sprint 007

The adaptive-K direction is **not a viable F-018 remediation**. The
Sprint 007 recommendation should not be "ship adaptive heuristic"
based on this data.

Other directions that remain plausible:
- **E2 (block-size reduction)**: shrinking DRAFT_N_MAX may reduce
  the per-round cost of failed drafts (fewer mask tokens, smaller
  decoder forward) AND reduce the checkpoint save overhead
  (smaller state to copy). Highest priority among remaining
  options.
- **Save-cadence reduction** (out of original sprint scope, but
  surfaced by E3): defer/amortize the GPU→host KV copy. Could
  cut up to 38% of speculative wallclock.
- **Distill smaller draft on operator-class data**: the 1.7B
  z-lab draft just doesn't predict prose well; a draft trained
  on representative content might raise per-prompt acceptance to
  the point where partials are rare and the skip flag rarely
  fires.
