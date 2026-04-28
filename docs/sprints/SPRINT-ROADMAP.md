# Sprint Roadmap (post-004)

**Created**: 2026-04-27
**Last updated**: 2026-04-27

This roadmap captures the next 6–9 months of speculative-decoding +
RotorQuant work after Sprint 004. Sprints 005 and 006 are concretely
planned in their own docs; Sprints 007–010 are sketched here at "what /
why / when" level. Re-plan in detail as each sprint approaches.

The throughline: **Sprint 004 shipped DFlash; the next sprints publish
proof, broaden coverage, optimize, then explore.** Each sprint either
closes a Sprint 004 hard gate, opens a new capability, or improves
operator experience.

---

## Sprint 005 — L4 Benchmark Publish + Experimentation

**Status**: planned · see `SPRINT-005.md`

**Goal**: Fill BENCHMARK-REPORT.md §10's TBD cells with real numbers;
close Sprint 004 hard gate #5 (forced-rejection).

**Headline deliverable**: `BENCHMARK-REPORT.md §10` reads as a complete
report — operators can answer "should I use DFlash" without consulting
sprint authors.

---

## Sprint 006 — EAGLE3 Productionization

**Status**: planned · see `SPRINT-006.md`

**Goal**: EAGLE3 speculative decoding shipped behind opt-in profile,
4-way benchmark vs target-only / autoregressive / DFlash on the same
5-prompt set.

**Why before 007**: Pure repo-side; the model graph is already in fork
from Phase 3 cherry-pick. Lower risk + smaller scope than Sprint 007's
fork-side rewrite. Closes F-008.

---

## Sprint 007 — Multi-slot Batched-Draft Inference

**Status**: sketched

**What**: Eliminate the per-slot draft serialization
(`tools/server/server-context.cpp:339, TAG_SERVER_SPEC_REWORK`). Each
server slot currently has its own `common_speculative` context but all
slots funnel into one shared draft `llama_context` with
`params_dft.n_parallel = 1`. Multi-user throughput scales sub-linearly
because draft inference becomes a serialization point. Sprint 007 batches
all slots' draft inferences into one draft pass.

**Why**: Multi-user serving on the MoE 35B + DFlash combo currently
loses much of DFlash's per-user speedup at >2 concurrent users. F-010
characterized the cost in Sprint 005 Phase 2; this sprint addresses it.

**When**: After Sprint 005 quantifies the sub-linearity. If Sprint 005
shows ≤30% sub-linear at N=4, deprioritize. If ≥50% sub-linear,
escalate.

**How**:
1. Decide between forking-picking upstream PR #18961 (which is in flight)
   vs implementing against our cherry-picked tree. Upstream rebase debt
   is the deciding factor — if #18961 is stable, fork-pick; else
   reimplement.
2. Convert the per-slot draft loop into a batched draft pass: all
   active slots' current `(prefix, last_sampled)` pairs assembled into a
   single batch, draft generates `N_max` tokens per slot in one forward
   pass.
3. Per-slot verify path stays as-is (it's already independent).
4. Test: greedy match at N_PARALLEL=1 vs N_PARALLEL=4; per-slot decode
   tok/s ≥0.85× single-slot at N=4.

**Key risks**:
- Upstream PR #18961 rebases mid-sprint (Sprint 004 R11 reprise).
- Draft KV-cache layout assumes single-slot today; batched draft may
  require draft KV refactor.
- Verify-path slot identification needs to track batch indices through
  the verify graph.

**Hard gates**:
- N_PARALLEL=1 baseline byte-equal vs Sprint 005's results.
- Per-slot tok/s ≥ 0.85× single-slot at N=4.
- Per-slot acceptance rate within ±2pp of single-slot.
- 4-slot greedy match: 256/256 tokens per slot vs single-slot reference.

**Estimated effort**: 2–3 weeks. Higher variance — if upstream PR
lands cleanly we fork-pick (1 week); if we reimplement (3 weeks).

---

## Sprint 008 — Long-Context + Streaming + Stochastic Sampling

**Status**: sketched (likely 3 separate subsprints; bundled here as one
"production maturity" sprint)

**What**: Three Sprint 004 deferred items (D-003, D-006, soft long-context
gate) that round out production-readiness without adding new speculative
mechanisms.

### 8.1: Long-context smoke test

- 32K-prompt greedy decode under DFlash on `qwen` and
  `qwen36-27b-dflash`. Assert no OOM, no checkpoint corruption, no
  acceptance-rate cliff.
- Promote Sprint 004's soft gate to hard gate post-validation.

### 8.2: Streaming SSE under speculative

- `stream: true` on `/v1/chat/completions` with DFlash. Per-block emit
  boundaries are the subtle interaction: when DFlash drafts 16 tokens
  and 12 are accepted, what does SSE see?
- Validate ordering, partial-token UTF-8 handling, end-of-stream signals.

### 8.3: Non-greedy sampler validation

- D-003 from Sprint 004 deferred. Greedy was the only validated regime.
- Distribution-level metrics (KL on logits, chi-squared on token
  frequencies across many seeds) since token sequences are non-deterministic
  under sampling.
- Hard gate: KL < 0.01 between target-only and target+DFlash on the
  5-prompt set, 100-seed average.

**Why bundled**: Each is small; bundling avoids per-sprint overhead.

**Estimated effort**: 2 weeks total.

---

## Sprint 009 — Multi-target Hot-Swap + Cross-Architecture EAGLE3

**Status**: speculative

**What**: Two operator-experience improvements that aren't gated on
upstream:

1. **Multi-target hot-swap**: One container hosting multiple targets
   (e.g. Qwen3.6-27B and Qwen3.6-35B-A3B simultaneously) with per-request
   target selection. Reduces operational overhead for users running
   multiple models. Requires llama-server multi-model support (upstream
   PR territory) or a thin proxy in front of N llama-servers.

2. **Cross-architecture EAGLE3**: EAGLE3 drafts trained on Qwen3 might
   work as drafts for related Qwen3 variants (different size, different
   tune). If acceptance is non-zero, this expands the reachable
   target/draft pairs without new draft training.

**Why later**: Both depend on Sprint 008's stochastic sampling
validation (multi-target inevitably gets temp>0 traffic).

---

## Sprint 010 — PARO / Sparse Attention Speculation

**Status**: research-tier

**What**: z-lab also publishes PARO models — Probabilistic AdaPtive
Sparse Attention drafts. Different speculative mechanism than DFlash
(block-diffusion) or EAGLE3 (autoregressive single-token); orthogonal
to both.

**Why explore**: Sparse attention may give a third operating point on
the speedup-vs-acceptance frontier. Particularly relevant if DFlash
and EAGLE3 both saturate around 1.3–1.5× and we want to push further.

**Pre-requisite**: Sprints 005-007 done; we need stable DFlash + EAGLE3
baselines before adding a third axis to the comparison.

**Risk**: PARO may need fork-side cherry-picks comparable in size to
PR #22105. Could become a 6+ week sprint.

---

## Cross-cutting concerns

These are not standalone sprints; they're priorities to fold into each
sprint's scope.

### Upstream rebase cadence

The fork is at `feature/sprint-004-rebase-dflash` `1c9b77fdd`. Upstream
master moves; periodic rebase keeps the delta manageable. **Recommended
cadence**: rebase during the Phase-0 setup of every sprint that touches
the fork. Don't let the gap exceed 2 sprints.

### Performance regression tracking

Sprint 005 publishes baseline numbers. Subsequent sprints should
re-run the canonical L4 5-prompt benchmark in their Phase 0 to
catch regressions before they ship. `make bench-dflash-all` is the
one-shot.

### Documentation gardening

Each sprint adds to BENCHMARK-REPORT.md §10. After Sprint 008, §10 is
likely big enough to warrant subdivision into §10.1 (architecture),
§10.2 (methodology), §10.3 (DFlash), §10.4 (EAGLE3), §10.5 (long-context),
etc. Plan a doc-gardening pass at end of Sprint 008.

### z-lab draft refresh tracking

z-lab iterates on draft training. Operators need to know when to
re-run `make convert-drafts`. Add a quarterly check (or HF API watch)
for new draft SHAs; document in `SPRINT-004-FOLLOWUPS.md` F-001 and
keep the entrypoint's `MODELS_HASH` map current.

---

## Decision log

### 2026-04-27 — Sprint 005 ahead of Sprint 006

**Decision**: Run measurement (005) before EAGLE3 productionization (006).

**Why**: Sprint 005 closes Sprint 004's open hard gates and produces
the baseline DFlash numbers that Sprint 006's 4-way comparison
table needs. Sprint 006 without Sprint 005 leaves you comparing
EAGLE3 to a TBD baseline.

### 2026-04-27 — Sprint 007 conditional on Sprint 005's multi-slot finding

**Decision**: Sprint 007 priority is set by Sprint 005 Phase 2's
N_PARALLEL=4 characterization. ≥50% sub-linear → escalate; ≤30%
sub-linear → deprioritize.

**Why**: Don't spend 2-3 weeks rewriting the draft batching path if
the per-slot serialization isn't the bottleneck for our deployed
configs.

### 2026-04-27 — Sprint 008 bundles three deferred items

**Decision**: Long-context + streaming + non-greedy sampling shipped
together as Sprint 008.

**Why**: Each individually is too small to be a sprint; together they
constitute a "production maturity" delivery. Operators need all three
before deploying speculative decoding to anything beyond a single
greedy single-slot use case.
