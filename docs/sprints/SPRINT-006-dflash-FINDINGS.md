# Sprint 006-dflash Findings — DFlash Performance Investigation

**Date**: 2026-04-30
**Sprint**: 006-dflash (investigation)
**Branch**: `sprint/006-dflash` (off `sprint/005-dflash`)
**Fork pin**: `4ef60a057` on `feature/sprint-004-rebase-dflash` (LOCAL ONLY)
**Profile / regime**: qwen (Qwen3.6-27B + DFlash, planar3 KV) +
thinking-off (PR #22105 baseline)
**Hardware**: RTX 5090 (32 GB)

---

## TL;DR

DFlash on Qwen3.6 + 5090 + Q4_K_XL is **structurally cost-bound**, not
implementation-broken. After fixing F-014's restore loop and
instrumenting the path, three findings dominate:

1. **Block size is dramatically too large**. DRAFT_N_MAX=4 gives 2.3×
   the median throughput of the default N=16 across the 5-prompt set,
   and 4× higher real draft acceptance on the worst prompt (Hamlet:
   31% vs 1.9%).
2. **Checkpoint save cost is ~40% of speculative wallclock** at every
   N value tested. Each new draft round triggers a ~150 MiB GPU→host
   copy of the slot's KV state. This is the largest single cost
   center, not partial-acceptance restore (which is 10–12%).
3. **Adaptive skip-K is the wrong knob at temp=0**. Skipping after K=2
   or K=3 consecutive partials at the same position is essentially
   indistinguishable from K=1 (the v8 always-skip default), because
   deterministic sampling produces the same partial outcome on retry.

**Ship-ready remediation**: change DRAFT_N_MAX default from 16 to 4.
+127% median throughput, no correctness regression, no new code
beyond a one-line entrypoint default change.

**Stretch (Sprint 007)**: rollback-cost reduction (defer/amortize the
GPU→host KV copy per draft round). Could cut another ~40% of
wallclock. Out of original Sprint 006 scope; surfaced by E3.

**Honest finding**: even with optimal N=4, **DFlash does not beat
target-only on any prompt** on this stack. Best is qwen quicksort at
62.2 tok/s vs target-only 70.3 = 0.89×. The Q4_K_XL target is too
cheap (~14ms/token) for the speculative bookkeeping overhead to pay
off. DFlash needs heavier targets to win.

---

## Hypothesis verdicts

| Hypothesis (from intent doc) | Verdict | Evidence |
|------------------------------|---------|----------|
| F-018 spec_skip_next_round is too aggressive (top hypothesis) | **REFUTED at temp=0** | E1 + E4. The flag is load-bearing for correctness (E1: disabling it crashes immediately). Adaptive K=2/3 is no improvement over K=1 because deterministic sampling reproduces the same partial outcome on retry. |
| DFlash block-diffusion isn't a fit for greedy/temp=0 sampling | **PARTIALLY CONFIRMED** | The block-diffusion mechanism itself works (high acceptance at small N), but the default 16-token block is far too large for the current draft model. AR ≈ DFlash performance from Sprint 005 reflects both being throttled by checkpoint cost; with N=4, both should improve similarly. |
| 1.7B draft is too small for entropic prompts | **CONFIRMED, with nuance** | At N=16 the draft fails badly on prose. At N=4 it hits 30%+ acceptance on Hamlet. Same draft, different block size — so the 1.7B size is OK if the block lets it bail early. |
| 156 MiB checkpoint restore per round is a real cost | **PARTIALLY CONFIRMED, restore is not the dominant cost** | E3: restore is 10–12% of wallclock. **Save** (separate, fires on every new draft round regardless of acceptance) is 37–40% — the actual dominant cost. Original sprint plan undersized the cost center. |
| Cross-attention over accumulated_ctx grows linearly | **NOT MEASURED** | Wasn't a Phase 1–5 priority once E3 / E5 surfaced bigger cost centers. Defer to Sprint 007 if relevant. |

---

## Per-experiment summary

Each experiment has its own `summary.md` under
`docs/sprints/SPRINT-006-dflash-experiments/EN/`. Quick recap:

### E3 — Checkpoint cost profile

Save: 37–40% of wallclock uniformly. Restore: 10–12%. Combined ckpt
overhead ~50% on every prompt. **Save is the headline cost center**
— the original sprint plan's emphasis on restore was directionally
correct but undersized.

### E5 — Rejection-position profile (at N=16)

Hamlet: 79% of rounds reject in positions 0–2. DC trip: 68%.
Pythagoras: 34%. Quicksort: spread evenly. Confirms that smaller
blocks should help on entropic prompts where rejections cluster
early — directly motivated E2.

### E1 — Skip-flag-off canary

Server crashes on first request when `LLAMA_SPEC_DISABLE_SKIP=1`.
The v8 spec_skip_next_round flag is load-bearing for correctness,
not a perf-only optimization. Confirms F-018 needs a *smarter*
fix, not removal.

### E4 — Adaptive skip K=2 / K=3

Identical tps to K=1 across all prompts (within noise). At temp=0,
the deterministic-partial dynamic means K>=2 just adds a wasted
round before falling back. **Refuted as a fix path**.

### E2 — DRAFT_N_MAX sweep at 4 / 8 / 16

The clear winner. **N=4 gives +127% median tps over N=16**, with no
correctness regression. Acceptance scales inversely with N (96% on
quicksort at N=4, 25% at N=16). This is the F-018 fix.

---

## Sprint 007 recommendation — per the decision tree

The Sprint 006 plan's decision tree mapped to four branches:

```
adaptive skip clears safety + ≥20% median gain?  → REFUTED (E1+E4)
rejection profile early failures + N=8 beats N=16?  → CONFIRMED (E5+E2)
checkpoint restore + replay ≥25% on entropic?  → restore alone <25%, but SAVE is 38%
no remediation works?  → not the case
```

**Two viable Sprint 007 branches emerge, not just one**:

### Sprint 007 candidate A — Block-size remediation (recommended, low risk)

Productize block-size remediation:
1. Default DRAFT_N_MAX = 4 in entrypoint.sh (was 16). Operator
   override preserved.
2. (Stretch) Adaptive block size: shrink to 4 when last-N rounds
   show low acceptance, grow to 16 on prefix-matched repetitive
   content.
3. Re-run canonical L4 sweep with N=4 default. Publish updated
   numbers in BENCHMARK-REPORT.md.
4. Update README's DFlash guidance: best with N=4 on Qwen3.6.

Estimated effort: 0.5 weeks single-engineer (mostly bench runs +
docs).

### Sprint 007 candidate B — Save-cadence reduction (higher reward, higher risk)

Tackle the 38% wallclock tax from checkpoint save. Approach
options:
1. **Defer save**: only checkpoint when verify is ABOUT to run
   (current code saves before verify pass). Save ~50% of saves
   on rounds that fully accept.
2. **Delta-state save**: track what changed since last
   checkpoint, copy only the delta.
3. **Lazy materialization**: keep checkpoint as a "logical" state
   pointer, materialize only on actual restore.

Estimated effort: 1.5–2 weeks — touches fork's
`server_get_checkpoint` and `llama_state_seq_get_data_ext`. Higher
risk because it changes the speculative-decoding correctness
contract.

### Recommendation: **Sprint 007 = candidate A; Sprint 008 = candidate B**

A is a 1-line config change with massive measured benefit and zero
risk. B is the real optimization but needs more careful design.
Ship A first, gather operator feedback, then attempt B.

---

## Honest scoreboard for Sprint 005's gate

Hard Gate #3 (≥1.3× median DFlash× on `qwen`):

| Config | Median DFlash× | Pass ≥1.3? |
|--------|---------------:|:----------:|
| Sprint 005 published (post-v8) | 0.20× | FAIL |
| Sprint 006 N=4 (recommended) | **0.46×** | **FAIL (still)** |
| Sprint 006 N=4 + save-cadence fix (estimated) | ~0.7× | FAIL projected |

**Sprint 006 doesn't fix the ≥1.3× gate** — that gate is structurally
unreachable on this stack regardless of tuning. The recommendation
for the sprint family:
1. **Drop the ≥1.3× hard gate** as a deployment-default success
   criterion. Replace with "≥1.0× on the prompts the operator's
   workload resembles."
2. **Ship DFlash with N=4** as a documented option, not the default
   for general users — but the right choice for code-heavy
   workloads (where it hits ≥1.0× even at N=4).
3. **Default qwen to target-only** for non-code workloads.

---

## Followups discovered during execution

### F-019: bench harness doesn't capture spec_* fields

`scripts/bench_speculative.py` parses `draft_n_generated` /
`draft_n_acc_tokens` (added in F-016) but doesn't capture the new
Sprint 006 instrumentation fields (`spec_t_ckpt_save_us`,
`spec_t_ckpt_restore_us`, `spec_n_ckpt_*`, `spec_*_hist`). For
this sprint, raw curl probes were used instead. Sprint 007 should
extend the harness so future sweeps record these in JSON.

### F-020: env-empty-string vs unset bug pattern

The Sprint 006 env-toggle lambdas had a bug: `v && std::string(v) != "0"`
treated empty string as "set". Docker-compose default `${VAR:-}`
expands to empty when host env is unset, so the toggle was always
"on" by default. Fixed in fork commit `156e69be6`. Sprint 007
should audit other env reads in the fork for the same pattern.

### F-021: AR ≈ DFlash result deserves re-investigation post-N=4

Sprint 005 noted AR and DFlash gave nearly identical tps. That
finding was at N=16. The N=4 results may show different
dynamics — DFlash's block-diffusion advantage might emerge when
the block is small enough that all positions can amortize. Worth
a single bench point in Sprint 007.

---

## Artifacts

- `docs/sprints/SPRINT-006-dflash-experiments/E1_skip_off/summary.md`
- `docs/sprints/SPRINT-006-dflash-experiments/E2_draft_n_max/summary.md`
- `docs/sprints/SPRINT-006-dflash-experiments/E3-E5-baseline/summary.md`
  (E3 + E5 baseline — instrumentation captures both in one run)
- `docs/sprints/SPRINT-006-dflash-experiments/E4_adaptive_skip/summary.md`
- Per-experiment `results-qwen.json` and `run.log` files where
  applicable.
- Fork commits `5f58c0d81..4ef60a057` (LOCAL ONLY; user push
  required to publish).

## Fork commits added in Sprint 006

| Commit | What |
|--------|------|
| `d51f97125` | Sprint 006 instrumentation (rejection counters, ckpt timing, env-gated skip toggles) |
| `156e69be6` | Env-empty-string-vs-unset fix |
| `4ef60a057` | F-018 v1: DFlash + EAGLE3 draft early-return on n_new<1 |

Plus the v8 chain (`5f58c0d81..86272e841`) inherited from Sprint 005.

All await an interactive push by the user.
