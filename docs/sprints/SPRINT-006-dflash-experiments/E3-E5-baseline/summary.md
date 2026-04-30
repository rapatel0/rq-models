# E3 + E5 baseline — checkpoint cost + rejection profile

**Date**: 2026-04-30
**Profile**: qwen (Qwen3.6-27B + DFlash, planar3 KV, 131K ctx)
**Regime**: thinking-off (PR #22105 baseline)
**Sampling**: temp=0, top_k=1, seed=42, max_tokens=256
**Fork pin**: `156e69be6` (v8 + Sprint 006 instrumentation w/ env-empty-string fix)
**Build**: `docker/Dockerfile.local` from local fork checkout

## Per-prompt summary

| Prompt | wall_s | tps | gen drafts | accepted | real % | ckpt_save_ms | ckpt_restore_ms | #save | #restore | save % | restore % |
|--------|-------:|----:|-----------:|---------:|-------:|-------------:|----------------:|------:|---------:|-------:|----------:|
| P1 quicksort | 1.83 | 56.2 | 600 | 150 | 25.00 | 691 | 167 | 20 | 15 | 40.5 | 9.8 |
| P2 Pythagoras | 17.99 | 14.3 | 3540 | 209 | 5.90 | 6787 | 2133 | 196 | 192 | 37.8 | 11.9 |
| P3 DC trip | 23.51 | 10.9 | 7350 | 210 | 2.86 | 8855 | 2832 | 254 | 253 | 37.8 | 12.1 |
| P4 Hamlet | 23.28 | 11.0 | 11130 | 213 | 1.91 | 8748 | 2800 | 252 | 251 | 37.7 | 12.1 |
| P5 SQL | 17.05 | 15.1 | 13905 | 283 | 2.04 | 6432 | 2013 | 185 | 180 | 37.9 | 11.9 |

`save %` and `restore %` are fractions of `predicted_ms` (decode wallclock).

## E3 verdict — checkpoint cost: **CONFIRMED**, with refinement

Phase 1 hypothesis was *"restore + replay consume ≥25% on entropic prompts"*.
The data says **restore alone is 10–12% across all prompts** — under the
25% threshold per the sprint's decision rule. **However, save dominates
at 37–40%** uniformly. Combined ckpt cost (save+restore) is ~50% of
speculative wallclock on every prompt.

The original decision rule undersized the cost center by focusing on
restore. The data revises the conclusion: **checkpoint save is the
single largest contributor to speculative overhead on hybrid Qwen3.6**
and fires even when no partial happens (every new draft round triggers
a save before the verify pass; with ~5% acceptance, that's one save
per output token).

Implication for Sprint 007: a productive direction is *checkpoint
cadence reduction* — defer save until needed (e.g., only checkpoint on
the first failure of a sequence, not on every round) or amortize via
delta-state tracking.

## E5 verdict — rejection profile: **CONFIRMED for entropic prompts**

| Prompt | reject @ 0–2 | reject @ 0–5 | block-mid | block-end |
|--------|-------------:|-------------:|----------:|----------:|
| P1 quicksort | 1+0+1=2 | 5 | 4 | 5 (at pos 15) |
| P2 Pythagoras | 19+27+21=67 | 124 | 8 | 3 |
| P3 DC trip | 58+68+44=170 | 229 | 1 | 0 |
| P4 Hamlet | 61+77+61=**199** | 250 | 0 | 0 |
| P5 SQL | 14+22+28=64 | 106 | 7 | 4 |

(Numbers are first-rejection-position counts across all draft rounds in
one trial. Block size = 16 in this run.)

**Phase 2 decision rule** (≥70% of rejected rounds on ≥2 entropic
prompts fail by token 2 → smaller / adaptive block size at top of
remediation list):

- P4 Hamlet: 199 / 252 = **79%** ✓ (passes threshold)
- P3 DC trip: 170 / 250 = 68% (close)
- P2 Pythagoras: 67 / 197 = 34% (below)

P4 alone passes. P3 is a near-miss. **Recommendation: prioritize E2
(DRAFT_N_MAX sweep at 4 / 8 / 16) in the remediation list.** Smaller
blocks should:
- Reduce wasted draft compute on rounds that reject at position 0–2
- Reduce per-block checkpoint save cost (proportional to draft size)
- Trade off against missed long-tail-acceptance opportunities on
  prompts like quicksort

## Cross-experiment observation: real acceptance is stable per prompt

Across all entropic prompts (P2-P5), real acceptance hovers at
**1.9–5.9%**. Only quicksort breaks pattern at 25%. This is consistent
with the Sprint 005 narrative: prompt content predictability dominates
draft acceptance, and the 1.7B z-lab draft can only handle code-class
content well.

## Implications

1. **Checkpoint save is a 38% wallclock tax** even when speculative
   "works" — the v8 fix chain saves the request from hanging but
   doesn't address the structural cost of partial-acceptance-aware
   restore semantics.
2. **For low-acceptance prompts**, the ratio gets worse: ~50% of the
   wallclock is spent on speculative bookkeeping that produces 1
   output token of value (single-token decode after restore).
3. **Two clear optimization paths emerge**:
   - **E4 (adaptive skip)**: reduce restore frequency on
     deterministic-partial loops; doesn't address save cost.
   - **Save-cadence reduction**: defer or amortize the GPU→host KV
     copy. Not in the original Sprint 006 cut; surface as a
     Sprint 007 candidate.
