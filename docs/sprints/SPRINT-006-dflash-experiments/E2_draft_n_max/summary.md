# E2 — DRAFT_N_MAX block-size sweep

**Date**: 2026-04-30
**Profile**: qwen + thinking-off
**Env override**: `DRAFT_N_MAX={4, 8, 16}`
**Fork pin**: `4ef60a057`

## Cross-N comparison

| Prompt | N=4 tps | N=8 tps | N=16 (baseline) | Best |
|--------|--------:|--------:|---------------:|:----:|
| P1 quicksort | 62.2 | 62.7 | 56.2 | **N=8** (62.7) |
| P2 Pythagoras | 36.3 | 24.7 | 14.3 | **N=4** (36.3) |
| P3 DC trip | 19.9 | 10.8 | 10.9 | **N=4** (19.9) |
| P4 Hamlet | 17.2 | 10.2 | 11.0 | **N=4** (17.2) |
| P5 SQL | 32.5 | 27.8 | 15.1 | **N=4** (32.5) |
| **Median** | **32.5** | 24.7 | 14.3 | **N=4** |
| **Median DFlash×** vs target~70 | **0.46×** | 0.35× | 0.20× | **N=4** |

## Real acceptance rate (F-016 metric)

| Prompt | N=4 % | N=8 % | N=16 % |
|--------|------:|------:|-------:|
| P1 quicksort | **96.0** | 61.1 | 25.0 |
| P2 Pythagoras | 63.6 | 24.5 | 5.9 |
| P3 DC trip | 40.0 | 9.1 | 2.9 |
| P4 Hamlet | 30.6 | 5.5 | 1.9 |
| P5 SQL | 33.9 | 7.8 | 2.0 |

Acceptance scales **inversely** with block size — by a lot. At N=4,
even Hamlet (the hardest entropic prompt) hits 31% real acceptance,
2× the N=16 quicksort number.

## Verdict — **CONFIRMED, smaller block dominates**

Per Phase 4 decision rule:

> "If 8 beats 16 by ≥15% median tok/s on the 5-prompt set →
> smaller / adaptive block size becomes default Sprint 007 plan."

- N=8 vs N=16 median tps: 24.7 vs 14.3 = **+73%** ✓ passes
- N=4 vs N=16 median tps: 32.5 vs 14.3 = **+127%** ✓ much stronger
- N=4 vs N=8 median tps: 32.5 vs 24.7 = **+32%** N=4 wins

**N=4 is the clear winner.** Sprint 007 productionization candidate:
either ship N=4 as the new default DRAFT_N_MAX (operator override
preserved) OR add adaptive block sizing that shrinks on entropic
content.

## Ckpt cost still dominates at smaller N

| N | save% (Hamlet) | save_ms | #save |
|---|---------------:|--------:|------:|
| 4 | 41.6 | 6170 | 173 |
| 8 | 35.4 | 8863 | 252 |
| 16 (baseline) | 37.7 | 8748 | 252 |

Save cost stays ~35–42% of wallclock across all N values. Reducing
N reduces per-save cost slightly (smaller KV slice to copy) but
also increases #save (more rounds). Net: ckpt save % is fairly
N-invariant.

This means **even with N=4, Sprint 007 still has a save-cost
optimization opportunity worth pursuing** — it's a ~40% wallclock
tax that's orthogonal to block size.

## Speculative still doesn't beat target-only

Even with N=4, the **best** speculative result is qwen quicksort at
62.2 tok/s vs target-only 70.3 tok/s = **0.89×**. **No prompt at any
N value beat target-only on this stack.**

The fundamental cost ratio: target's per-token cost (~14ms) is
already cheap on the 5090 for Q4_K_XL Qwen3.6-27B. The
speculative path's per-round cost (target verify + ckpt save + draft
+ ckpt restore on partial) adds a fixed overhead per N tokens. Even
at high acceptance, that overhead ≥ target's cost.

For this stack to benefit from speculative decoding, the target
needs to be slower (e.g., heavier quant, larger model) or the
speculative path's checkpoint cost needs to drop dramatically.

## Implications

1. **Ship N=4 default for Sprint 007**: 2.3× median throughput
   improvement vs current N=16, virtually no risk.
2. **Adaptive N**: stretch goal — shrink to 4 when entropic
   patterns detected, grow back to 16 on code-class prompts where
   long blocks pay off.
3. **Save-cost reduction**: ~40% wallclock tax even at N=4. The
   right Sprint 008 candidate.
4. **DFlash × target-only on this stack**: still <1.0 even at
   optimal N. DFlash itself isn't broken, but the cost ratio doesn't
   favor it. Sprint 007 should publish honest "DFlash gives <1×
   speedup on Qwen3.6 + 5090, use target-only" guidance for
   operators.

## Recommendation for Sprint 007 decision tree

This experiment hits the **block-size branch** of the Phase 6
recommendation tree:

> "Rejection profile (Phase 2) shows early failures (0–2 tokens) AND
> DRAFT_N_MAX 8 beats 16 (Phase 4)? Yes → Sprint 007 = dynamic
> block-size / rejection-aware DFlash"

Both conditions met. Sprint 007 productizes adaptive (or fixed
N=4) block size as the F-018 fix.
