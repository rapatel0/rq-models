# Sprint 006-dflash Intent — DFlash Performance Investigation

> **Track suffix**: `-dflash`. The dflash track does not merge to `main`
> (parallel vLLM-substrate Sprint 004/005 numbering conflict on `main`);
> the chain stays self-contained on `sprint/004-dflash` →
> `sprint/005-dflash` → `sprint/006-dflash` → ...

**Created**: 2026-04-29
**Sprint type**: Investigation + targeted code experiments (not a feature
delivery sprint)
**Naming caveat**: there is an existing `docs/sprints/SPRINT-006-dflash.md`
stub from Sprint 005's planning era describing **EAGLE3
productionization**. That stub is now stale because Sprint 005's gate
failed and the priority pivoted from "ship more speculative variants" to
"figure out why the speculative we have isn't winning". The interview
will resolve naming: rename the EAGLE3 stub → `SPRINT-007-dflash.md` (or
similar) and use 006 for this investigation sprint, vs. keeping 006 for
EAGLE3 and using 007 for investigation.

---

## Seed prompt

> Okay now we need an investigation sprint. Please dig into the details
> and figure out why we are not seeing performance uplift. Lets also
> brainstorm some code experiments to run

---

## Orientation summary

**Current state (2026-04-29)**:

- Sprint 005-dflash closed **complete-with-followups**. Hard Gate #3
  (≥1.3× median DFlash×) **FAILED** on both `qwen` (27B+DFlash) and
  `qwen36` (35B-A3B+DFlash MoE) across both thinking-on and thinking-off
  regimes. Quicksort thinking-off on qwen hit 1.78× — confirming the
  implementation works on its target regime — but median across the
  5-prompt set ranged 0.52×–0.80×.
- F-011 + F-014 + F-016 fixes shipped during Sprint 005 (correctness +
  crash + metric). F-014 took an 8-iteration debugging chain
  (40856a1d2..86272e841 on the fork) and ended up introducing a
  performance regression via the `spec_skip_next_round` flag (the v5
  workaround for an infinite restore loop). That perf regression is
  filed as **F-018** in `SPRINT-005-FOLLOWUPS-dflash.md`.
- The fork commits `5f58c0d81..86272e841` are **local only** in
  `/home/ravi/repos/llama-cpp-turboquant/`, awaiting an interactive
  push (1Password agent can't sign from non-interactive shells). Sprint
  006 work that depends on the fork pin will block until that push.

**Recent work themes**:

- DFlash + EAGLE3 cherry-pick + Sprint 004 base — landed.
- L4 benchmark publishing across 4 regimes — landed
  (`docs/BENCHMARK-REPORT.md` §Sprint 005).
- 8-iteration F-014 fix chain — landed locally on fork.
- Bench harness `--no-think` flag for both-regime publish — landed.

**Key modules likely involved**:

- Fork (`/home/ravi/repos/llama-cpp-turboquant/`):
  - `common/speculative.cpp` — DFlash impl (`begin`, `rollback`,
    `draft`, `accept`); accumulated_ctx growth; block-decode loop.
  - `tools/server/server-context.cpp` — slot loop, partial-acceptance
    restore, `spec_skip_next_round` flag, checkpoint save/restore
    (~150 MiB GPU memcpy per round).
  - `tools/server/server-task.h` / `.cpp` — `result_timings` schema
    (`draft_n_generated`, `draft_n_acc_tokens` fields added in F-016).
- Repo (`/home/ravi/repos/turbo/`):
  - `scripts/bench_speculative.py` — bench harness; `--no-think` flag;
    metric parsing.
  - `scripts/sweep_dflash.py` — Phase 2 sweep driver (largely unused;
    F-012 Q5/Q8 GGUFs missing).
  - `docs/sprints/SPRINT-005-FOLLOWUPS-dflash.md` — F-011..F-018.

**Constraints / patterns to respect**:

- dflash-track sprint docs use `-dflash` suffix on filenames.
- Fork commits go on `feature/sprint-004-rebase-dflash`.
- Repo branch chain: `sprint/004-dflash` → `sprint/005-dflash` →
  `sprint/006-dflash`. No merging to `main` for the dflash track.
- Hybrid Qwen3.6 target context requires
  `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` — partial-state seq_rm not
  available; checkpoint+restore is the only safe rollback.
- Per-prompt comparison must hold sampling fixed (`temp=0`, `top_k=1`,
  `seed=42`, `tokens=256`) — Sprint 005 conventions.

**Vision document**: **None exists** for the dflash track at
`docs/sprints/VISION.md`. There is a `SPRINT-ROADMAP-dflash.md` (from
Sprint 005 planning era) but it's a roadmap doc, not a tracked vision.
Per the sprint-plan skill: when no vision exists but prior sprints do,
note absence and proceed. Sprint 006 may surface a vision-establishment
recommendation but won't block on it.

**Deferred items from prior sprints**:

- D-001 (Sprint 004): EAGLE3 productionization — was tentatively
  targeted at 005 or 006. Currently stubbed in
  `SPRINT-006-dflash.md` (the existing file). Investigation sprint
  pre-empts EAGLE3; recommend deferring D-001 to Sprint 007 unless
  EAGLE3 turns out to be relevant to the investigation hypothesis.

**Follow-up items actionable now**:

- **F-018** (Important): adaptive `spec_skip_next_round` heuristic —
  the v5 fix is too aggressive (always skips on partial), throttling
  good prompts to 1 token/cycle. Pre-v8 perf was 125 tok/s on qwen
  quicksort think-off; post-v8 it's 51. Direct cost of the fix.
- **F-012** (Important): Q5/Q8 27B GGUFs for target-quant sweep — needed
  if any experiment requires varying target weight quant.
- **F-016** (Important): bench harness should consume the new
  `draft_n_generated` / `draft_n_acc_tokens` fields landed in fork
  commit `a2a6168ea`. Currently the bench's `acceptance_rate` field
  uses the misleading post-truncation metric; we should switch and
  republish.
- **F-015** (Important): redesign `tests/test_speculative.py::TestForceReject`
  so it actually exercises `LLAMA_SPEC_FORCE_REJECT_AT`. Out of scope
  for this investigation sprint unless we use force-reject for
  experiments.
- **F-011, F-013, F-014 (correctness), F-017**: RESOLVED — no action.

---

## What we know (the evidence)

Cross-regime decode tok/s and DFlash× ratios from Sprint 005's bench
runs (pre-v8 fix-chain numbers; the fix chain landed late in 005 and
post-v8 numbers showed even lower throughput due to F-018):

| Profile | Regime | Quicksort× | Median× | Gate ≥1.3× |
|---------|--------|-----------:|--------:|:----------:|
| qwen   (27B+DFlash)         | thinking-on  | 1.10× | 0.80× | FAIL |
| qwen   (27B+DFlash)         | thinking-off | **1.78×** | 0.67× | FAIL |
| qwen36 (35B-A3B+DFlash MoE) | thinking-on  | 0.83× | 0.52× | FAIL |
| qwen36 (35B-A3B+DFlash MoE) | thinking-off | 1.15× | 0.58× | FAIL |

Per-prompt detail (qwen think-off, pre-v8): P1=1.78×, P2=0.89×,
P3=transport-error-resolved, P4=0.46×, P5=0.28×.

Post-v8 validation (5 prompts back-to-back, qwen think-off):

| Prompt | tps | comp_tok | raw drafts | accepted | real % |
|--------|----:|---------:|-----------:|---------:|-------:|
| P1 quicksort | 51.3 | 96 | 300 | 75 | 25.0% |
| P2 Pythagoras | 13.7 | 256 | 3240 | 134 | 4.1% |
| P3 DC trip | 10.5 | 256 | 7050 | 135 | 1.9% |
| P4 Hamlet | 10.6 | 256 | 10830 | 138 | 1.3% |
| P5 SQL | 14.5 | 256 | 13605 | 208 | 1.5% |

(Real acceptance per F-016's new `draft_n_generated` /
`draft_n_acc_tokens` metric — the bench's old "acceptance_rate" field
read 100% throughout.)

---

## Hypotheses (ranked, to be tested by experiments)

1. **F-018 — `spec_skip_next_round` is too aggressive** (highest
   priority; cheapest to test). Fires on every partial regardless of
   pattern; should fire only after K consecutive partials at the same
   position. Pre-v8 numbers suggest this is the dominant throttle for
   prompts that don't actually loop.
2. **DFlash block-diffusion isn't a fit for greedy/temp=0 sampling.**
   Block-diffusion's strength is parallel speculation; greedy requires
   exact-token match per position. Comparing AR vs DFlash columns: they
   produce nearly identical tok/s across all 4 cells, suggesting the
   block-diffusion dimension contributes ~0% lift over autoregressive.
3. **1.7B draft is too small for entropic prompts.** Acceptance scales
   with output predictability — quicksort (code, repetitive) wins;
   Pythagoras / Hamlet / SQL (entropic) lose. PR #22105's published
   1.5–2× was likely measured on more code-class prompts.
4. **156 MiB checkpoint restore per draft round is a real cost.** PCIe
   + GPU work for state_seq_get/set_data_ext. For prompts hitting
   partial often, this is non-trivial.
5. **Cross-attention over `accumulated_ctx` grows linearly.** By token
   256, the draft cross-attends over 280-position embeddings every
   block — superlinear effective cost.

---

## Success criteria

This is an **investigation sprint**, not a delivery sprint. Success is
information, not a 1.3× speedup.

**Hard gates (sprint fails if any miss)**:

1. **F-018 root-caused**. We know whether `spec_skip_next_round` fired
   on every partial vs only persistent-partial loops. Numerical answer
   to "what fraction of v8's perf loss was the skip flag vs other
   factors".
2. **Per-prompt rejection profile collected**. For at least 2
   representative prompts (quicksort, one entropic), we have logged
   which positions in each block get rejected. Tells us whether
   smaller block_size would help.
3. **At least 4 of the experiments E1–E11 (proposed below) executed**
   with documented numerical outcome.
4. **Decision document written** — for each hypothesis above, a
   verdict: confirmed / refuted / inconclusive, with the data
   supporting it.
5. **Sprint 007 (or next) recommendation**: based on findings, what
   should the dflash track do next? Concrete options: (a) ship F-018
   adaptive heuristic, (b) try EAGLE3 as alternative, (c) train a
   distilled draft, (d) declare DFlash inappropriate for current
   stack.

**Soft gates**:

- Push the fork commits (`5f58c0d81..86272e841`) to
  `rapatel0/llama-cpp-turboquant` so Sprint 006 doesn't have to keep
  using the local-checkout Dockerfile override.
- Bench harness updated to read `draft_n_generated` /
  `draft_n_acc_tokens` (F-016 follow-through). Re-publish the
  Sprint 005 numbers with the corrected metric.
- F-014 v8 perf regression has at least a draft fix on the fork.

---

## Verification strategy

For each experiment:

- **Baseline**: full 5-prompt × 3-trial bench at temp=0/top_k=1/seed=42,
  tokens=256, on the running fork pin (`86272e841`).
- **Variant**: change one variable (block size, skip flag, target
  quant, etc.). Re-bench.
- **Comparison**: per-prompt tok/s + real acceptance rate (via
  F-016 metric) before/after.
- **Statistical sanity**: 3 trials per cell; report median; flag any
  cell where stddev/median > 5% as suspicious.

Investigation deliverables:

- `docs/sprints/SPRINT-006-dflash-experiments/` directory with one
  subdir per experiment, each containing the JSON results + an MD
  summary.
- `docs/sprints/SPRINT-006-dflash-FINDINGS.md` aggregating per-experiment
  outcomes and hypothesis verdicts.

---

## Uncertainty assessment

| Dimension | Level | Reason |
|-----------|-------|--------|
| Correctness | **Low** | We're not changing semantics — measuring + small targeted tweaks (skip-flag heuristic, block size). Fork's correctness already validated by Sprint 005's F-011 + F-014 chain. |
| Scope | **Medium** | Investigation can sprawl. Need to interview to fix scope at ~5–8 experiments and 1.5 weeks single-engineer. |
| Architecture | **Low** | No new components. Reading + light instrumentation + parameter sweeps in the existing fork. |

Overall: **Medium uncertainty**, primarily on scope. The interview
should set scope tightly.

---

## Open questions for the interview

1. **Naming**: rename the existing EAGLE3 stub at
   `SPRINT-006-dflash.md` to free 006 for this investigation? Or
   number this 007? (The existing 006 stub has no execution; renaming
   is cheap.)
2. **Scope budget**: 1 week, 1.5 weeks, or 2 weeks single-engineer?
3. **Top experiments**: of E1–E11 below, which 4–6 to commit to in
   Phase order? (Tier 1 cheap-and-high-signal recommended: E1 disable
   skip-flag, E2 lower DRAFT_N_MAX, E3 profile checkpoint cost, E4
   adaptive skip heuristic.)
4. **Push gating**: should Sprint 006 block on the user pushing the
   local fork commits, or can we keep using the Dockerfile.local
   override workaround?
5. **EAGLE3 as experiment**: is testing EAGLE3 (E5 below) in scope, or
   does it pre-empt a future Sprint 007 EAGLE3 productionization?
6. **Distilled-draft experiment** (E11): is training a custom draft in
   scope, or out-of-scope for this investigation?

---

## Proposed experiments (E1–E11)

(Detailed in the seed-prompt response; consensus drafts will refine
ordering and add any missing ones.)

- **E1**: Disable `spec_skip_next_round` flag, measure F-014 hang
  re-emergence vs perf recovery. (Tier 1)
- **E2**: Lower `DRAFT_N_MAX` from 16 → 4 / 8. (Tier 1)
- **E3**: Profile checkpoint save/restore cost. (Tier 1)
- **E4**: Adaptive skip-flag (only after K consecutive partials at same
  position). (Tier 1, the F-018 fix candidate)
- **E5**: Token-by-token rejection profile (instrument `common_sampler_sample_and_accept_n`). (Tier 2)
- **E6**: Draft-vs-target offline alignment (run both standalone, diff
  positions). (Tier 2)
- **E7**: EAGLE3 draft as alternative. (Tier 2)
- **E8**: Smaller target quant (Q3_K_M / Q2_K). (Tier 3)
- **E9**: Multi-slot N_PARALLEL=2 amortization. (Tier 3)
- **E10**: Distilled / smaller custom draft. (Tier 4)
- **E11**: Z-lab reference acceptance rate comparison (external
  diagnostic). (Tier 4)
