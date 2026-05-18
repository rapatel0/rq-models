# Sprint Roadmap (post-004) — DFlash track

**Track suffix**: `-dflash`
**Created**: 2026-04-27
**Last updated**: 2026-04-27

> **Naming context**: this repo currently runs *two parallel sprint
> tracks* with overlapping numbering: a vLLM-substrate planar3-port
> track (no suffix, lives on `main`) and a llama.cpp DFlash
> speculative-decoding track (`-dflash` suffix, lives on
> `sprint/004-dflash`). Until the planning skill convention is sorted,
> all docs for this track carry the suffix. See
> `SPRINT-004-dflash.md` for this track's Sprint 004 sources of
> truth and `SPRINT-004.md` for the vLLM track's.

This roadmap captures the next 6–9 months of speculative-decoding +
RotorQuant work after Sprint 004-dflash. Sprints 005-dflash and
006-dflash are concretely planned in their own docs; Sprints
007-mtp and 008-dflash-010-dflash are sketched here at "what / why / when" level.
Re-plan in detail as each sprint approaches.

The throughline: **Sprint 004 shipped DFlash; the next sprints publish
proof, broaden coverage, optimize, then explore.** Each sprint either
closes a Sprint 004 hard gate, opens a new capability, or improves
operator experience.

---

## Sprint 005 — L4 Benchmark Publish + Experimentation

**Status**: planned · see `SPRINT-005-dflash.md`

**Goal**: Fill BENCHMARK-REPORT.md §10's TBD cells with real numbers;
close Sprint 004 hard gate #5 (forced-rejection).

**Headline deliverable**: `BENCHMARK-REPORT.md §10` reads as a complete
report — operators can answer "should I use DFlash" without consulting
sprint authors.

---

## Sprint 006 — EAGLE3 Productionization

**Status**: planned · see `SPRINT-006-dflash.md`

**Goal**: EAGLE3 speculative decoding shipped behind opt-in profile,
4-way benchmark vs target-only / autoregressive / DFlash on the same
5-prompt set.

**Why before 007**: Pure repo-side; the model graph is already in fork
from Phase 3 cherry-pick. Lower risk + smaller scope than Sprint 007's
fork-side rewrite. Closes F-008.

---

## Sprint 007 — True Multislot MTP Draft Inference

**Status**: planned · see `SPRINT-007-mtp.md`

**What**: Diagnose and, if bounded, fix the upstream llama.cpp
`draft-mtp` multi-slot bottleneck for Qwen3.6-27B MTP on the homelab
RTX 4090. Production B1 stays default while an explicit preview path
tests `N_PARALLEL=2/4`.

**Why**: The current 4090 matrix shows MTP-off dense 27B scales well
(`39.7 -> 124.5 t/s` aggregate from `np=1 -> 4`), but MTP-on barely
scales (`68.1 -> 77.2 t/s`). The bottleneck is MTP plus multi-slot
speculative scheduling, not dense 27B batching in general.

**When**: Now. The MTP matrix is sufficient to escalate this from
sketch to planned sprint.

**How**:
1. Preserve the current A/B matrix in benchmark artifacts before code
   changes.
2. Add gated instrumentation around MTP draft, target verify, and
   accept/rollback phases.
3. Spike the actual `draft-mtp` bottleneck before refactoring, because
   upstream already tracks some per-sequence state.
4. Patch the bounded bottleneck behind `PREVIEW=1` and `MTP_MULTISLOT=1`.
5. Test target-only greedy equivalence at `N_PARALLEL=1/2/4`; require
   `B4 >= 1.4 * B1` and `B4 >= 0.70 * A4` before preview promotion.

**Key risks**:
- The current upstream MTP code already appears partially batched, so
  the real bottleneck may sit deeper than the obvious loop.
- Cross-slot hidden-state, sampler, or rollback corruption can produce
  plausible but wrong text.
- 24 GB VRAM headroom is tight at production context.
- Instrumentation can perturb timings if not gated carefully.

**Hard gates**:
- B1 production behavior stays unchanged by default.
- Target-only greedy match: 256/256 generated tokens per slot at
  `N_PARALLEL=1/2/4`.
- Per-slot MTP acceptance at `N_PARALLEL=2/4` remains within +/- 5pp
  of B1.
- Preview promotion requires `B4 >= 1.4 * B1` and `B4 >= 0.70 * A4`.

**Estimated effort**: 2-3 weeks. Higher variance: the sprint can end
with instrumentation plus a blocked implementation report if the
upstream MTP scheduling change exceeds the spike bounds.

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
