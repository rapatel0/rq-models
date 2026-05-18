# Sprint 007 Codex Critique

> Critiques of `SPRINT-007-CLAUDE-DRAFT.md` and `SPRINT-007-GEMINI-DRAFT.md`

---

## Critique of SPRINT-007-CLAUDE-DRAFT.md

### Strengths

1. **Grounded in the actual repo and intent**: The draft ties itself to the
   current `entrypoint.sh` guard, the reported A/B matrix, the pinned upstream
   `b9196` patch flow, and the requirement that B1 remain default. That makes
   it executable rather than aspirational.

2. **Best phase ordering of the two drafts**: Phase 1 spike + Phase 2
   instrumentation are load-bearing deliverables, and the plan explicitly allows
   a useful "instrumentation-only" outcome if Phase 3 blows up. That is the
   right shape for risky upstream C++ work.

3. **Correctness strategy is strong**: Byte-exact checks, rejection-heavy
   prompts, long-prompt coverage, per-slot acceptance preservation, and B1/A4
   regression gates all align well with the intent's "correctness before
   throughput" priority.

4. **Deployment safety is explicit**: The draft keeps B1 as the supported
   default, requires opt-in gating for multislot MTP, and separates sprint
   completion from promotion of a user-facing profile.

5. **Risk handling is concrete**: VRAM pressure, homelab contention, upstream
   drift, missing benchmark artifacts, and scope creep are all named with
   specific mitigations instead of generic caution language.

### Weaknesses

1. **The document churn is too high for the sprint size**: `SPRINT-007-spike.md`,
   followups, baseline/result markdown + JSON artifacts, benchmark report,
   README, roadmap updates, and the final sprint doc create a lot of overhead
   before the hardest technical work is proven viable.

2. **Instrumentation scope may be heavier than necessary**: Adding new
   `timings.spec.*` fields, structured server traces, trace capture plumbing,
   and harness summarization is useful, but the draft may be underestimating the
   cost of changing observability surfaces in addition to the core batching fix.

3. **The gate model is slightly muddled**: In the Definition of Done, throughput
   is listed under "hard gates" but is also described as optional for sprint
   closure if the profile remains preview-only. That should be split more
   cleanly into "sprint complete" vs "promotion to experimental profile."

4. **The canonical correctness oracle is a little blurry**: The draft checks B1
   byte-equality against saved B1 output, but the stronger invariant from the
   intent is equality to target-only greedy output. B1 equivalence is useful,
   but it should be secondary to A1 target-only equivalence.

5. **Phase 3 may still be under-scoped**: The draft correctly suspects that the
   change could ripple through rollback and state bookkeeping, but it still
   leans toward a mostly localized `common/speculative.cpp` + server-callsite
   change. That may prove optimistic.

### Risk Analysis Gaps

- **Request lifecycle handling** is under-modeled. The draft does not explicitly
  discuss client disconnects, cancelled generations, or a slot completing while
  others are still mid-draft/verify.

- **Mixed request behavior** is mostly absent. The correctness plan is greedy
  and homogeneous, but real concurrent slots may differ in `n_predict`, stop
  strings, or sampling parameters even if the draft path itself is global.

- **Observability compatibility** is not called out as a risk. If `timings`
  payloads or server logs change shape, downstream scripts and dashboards may
  need coordinated updates.

- **Sparse occupancy** is not treated as a first-class risk. `parallel=4` with
  only 2 active slots can expose different batching bugs than a fully saturated
  4-slot run.

### Missing Edge Cases

- Slot churn during a run: one slot completes or is cancelled while others
  continue.
- Uneven prompt lengths at `np=4`, especially one long prompt plus several short
  prompts.
- Sparse occupancy: `parallel=4` with only 2 active requests.
- Mixed stop conditions and different `n_predict` values per slot.
- Forced reject/rollback cases affecting multiple slots in the same draft step,
  not just naturally rejection-heavy prompts.
- Streaming response ordering and latency once batched draft results are
  scattered back to per-slot outputs.

### DoD Completeness: 8.5/10

Strong overall. The main improvements are:
- separate "sprint completed" gates from "profile promotable" gates;
- add an explicit slot-churn / cancellation correctness case;
- require peak VRAM reporting at B2/B4, not just abort thresholds;
- state clearly that target-only greedy output is the primary oracle.

---

## Critique of SPRINT-007-GEMINI-DRAFT.md

### Strengths

1. **Concise and readable**: The draft gets to the problem quickly and stays
   easy to scan. That helps when the audience already understands the MTP
   problem and mainly needs the implementation target.

2. **The top-line objective is correct**: It keeps B1 as default, targets true
   multislot MTP rather than the hybrid router, and uses the right headline
   comparison against A4.

3. **The implementation surface is directionally right**: `common/speculative.cpp`,
   `server-context.cpp`, patch regeneration, deployment guards, and benchmark
   scripts are the correct broad areas to touch.

4. **The main high-impact risks are at least acknowledged**: acceptance
   collapse, VRAM pressure, upstream drift, and the possibility that the sprint
   ends at instrumentation are all present.

### Weaknesses

1. **Phase 0 is not operationally correct as written**: It asks for B2/B4
   baseline collection even though the repo currently hard-blocks MTP with
   `N_PARALLEL > 1`. That failure mode is central to the sprint and should be
   explicit in the baseline plan.

2. **The implementation plan is too vague for risky upstream C++ work**:
   "optimize the loop" and "parallel execution where possible" do not identify
   the actual likely state-model change or where rollback complexity lives.

3. **There is no real descope structure**: If instrumentation reveals a deeper
   refactor, the draft does not define a strong fallback deliverable beyond
   general reporting.

4. **Correctness coverage is thin**: Three prompts and a generic deterministic
   diff harness are not enough for speculative decoding, where state-sharing
   bugs often only appear under rejection-heavy or uneven-slot conditions.

5. **Deployment gating is underspecified relative to the repo**: The draft
   mentions an `--experimental-multislot-mtp` flag, but the actual safety
   surfaces here are `entrypoint.sh`, compose, Helm values, and fail-closed
   defaults.

6. **One use case overclaims**: "4x the request volume on the same VRAM
   footprint" is not supported by the stated throughput target or by the known
   hidden-state / concurrency overheads.

### Risk Analysis Gaps

- **Rollback correctness** is not treated as its own risk. Different slots may
  accept different counts of draft tokens in the same step, which is exactly
  where hidden-state and rollback bugs tend to hide.

- **Request lifecycle risk** is missing: disconnects, slot churn, partial
  occupancy, and admission timing do not appear in the plan.

- **Patch reproducibility risk** is missing: there is no explicit clean
  apply-from-scratch gate for the regenerated upstream patch.

- **Observability is underspecified**: the draft says to analyze logs but does
  not define the trace fields or artifact needed to prove the serialization
  point.

- **Environment noise risk** is missing: all meaningful gates depend on one
  shared 4090 node, but homelab contention and run-to-run cleanliness are not
  called out.

- **Single-slot regression risk** is underplayed: the draft never makes B1
  byte-exact preservation at `np=1` a first-class gate after the refactor.

### Missing Edge Cases

- Rejection-heavy prompts and forced reject/rollback scenarios.
- Long-prompt / high-context cases.
- Uneven prompt lengths across slots.
- `np=1` regression after the multislot refactor.
- `np=2` as its own checkpoint before jumping to `np=4`.
- Sparse occupancy: `parallel=4` with only 2 active requests.
- One slot finishing early while another continues.
- Peak VRAM and headroom measurement under concurrent load.

### DoD Completeness: 5.5/10

The draft has the right headline metrics but is incomplete for execution.
Missing:
- an instrumentation artifact with concrete fields and a decision-useful report;
- an explicit instrumentation-only / descope success path;
- patch apply/build reproducibility gates;
- richer correctness cases beyond simple greedy equivalence;
- explicit fail-closed deployment behavior in compose/Helm/entrypoint;
- clear peak-memory and sparse-occupancy checks.

---

## Summary

| Criterion | Claude Draft | Gemini Draft |
|---|:---:|:---:|
| Grounded in repo + intent | ✅ | ⚠️ |
| Descope-friendly phase ordering | ✅ | ❌ |
| Implementation specificity | ✅ | ⚠️ |
| Correctness coverage | ✅ | ❌ |
| Deployment safety | ✅ | ⚠️ |
| Risk depth | ✅ | ⚠️ |
| Documentation overhead | ⚠️ too high | ✅ concise |
| DoD clarity | ⚠️ gate-model cleanup needed | ⚠️ incomplete |

**Recommendation**: Use the Claude draft as the base. It is materially stronger
on correctness, observability, gating, and fallback planning. Pull in Gemini's
brevity and simpler top-line framing, but trim Claude's document churn and make
the distinction between "sprint succeeded" and "experimental profile is
promotable" explicit. Add slot-churn, sparse-occupancy, and peak-VRAM checks to
whichever draft is merged.
