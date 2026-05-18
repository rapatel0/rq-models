# Sprint 007 Merge Notes

**Sprint**: `SPRINT-007-mtp`
**Date**: 2026-05-18
**Seed**: `$sprint-plan changes for 2`

## Inputs

- Intent: `docs/sprints/drafts/SPRINT-007-INTENT.md`
- Claude draft: `docs/sprints/drafts/SPRINT-007-CLAUDE-DRAFT.md`
- Codex draft: `docs/sprints/drafts/SPRINT-007-CODEX-DRAFT.md`
- Gemini draft: `docs/sprints/drafts/SPRINT-007-GEMINI-DRAFT.md`
- Claude critique: `docs/sprints/drafts/SPRINT-007-CLAUDE-CRITIQUE.md`
- Codex critique: `docs/sprints/drafts/SPRINT-007-CODEX-CRITIQUE.md`
- Gemini critique: `docs/sprints/drafts/SPRINT-007-GEMINI-CRITIQUE.md`

## Weather Report

Weather report fetched successfully. Sprint Planning row recommended
`consensus(opus-4.7, gpt-5.4)` with max / extra-high effort and noted that
Gemini can improve consensus. All three draft and critique legs were attempted.
Gemini's pinned model name was stale, so Gemini ran on the CLI default after
the first failure.

## Draft Strengths

### Claude

- Strongest phase ordering: spike, instrumentation, descope path, then patch.
- Most concrete about fail-closed production behavior and B1 preservation.
- Best edge-case list for speculative correctness.
- Strongest reminder that patch changes must be regenerated from a fresh
  upstream checkout.

### Codex

- Clearest distinction between sprint completion and profile promotion.
- Strong security/isolation framing for cross-slot state.
- Strong baseline preservation and reproducibility gates.
- Good operator-facing file list and deployment-surface coverage.

### Gemini

- Concise top-line framing.
- Useful critique additions: VRAM profiling, asynchronous slot depths, and
  instrumentation overhead.
- Good recommendation to merge Claude's phase ordering with Codex's promotion
  gates.

## Accepted Critiques

- Use Claude as the structural base, but trim document churn.
- Split gates into "sprint completion" and "preview promotion."
- Make target-only greedy output the primary correctness oracle.
- Add slot churn, cancellation, sparse occupancy, uneven prompt lengths, mixed
  `n_predict`, and peak VRAM checks.
- Add a LOC/scope-bound spike before refactoring.
- Keep instrumentation gated so it does not distort normal benchmark numbers.
- Keep production B1 as default regardless of sprint outcome.
- Preserve the user-reported matrix before code changes.

## Rejected Or Deferred Critiques

- Rejected making the hybrid router part of core scope. It is captured in
  `SPRINT-007-mtp-DEFERRED.md`.
- Deferred generalizing the batched-draft interface to DFlash/EAGLE3.
- Deferred stochastic sampling and streaming SSE validation.
- Deferred upstream PR work until the RotorQuant stack proves the patch.

## Interview Handling

No blocking interview was conducted. The plan proceeds with these assumptions:

- Scope is upstream `draft-mtp` only.
- `np=4` is the promotion target because A4 is the known aggregate ceiling.
- `np=2` is a required checkpoint but not sufficient to replace the `np=4`
  promotion target unless the user later approves that tradeoff.
- Production B1 remains default even if preview promotion gates pass.
- Homelab 4090 access is available for final correctness and throughput
  matrices.

The remaining choices are preserved as Open Questions in the final sprint doc.

## Final Artifacts

- Final sprint: `docs/sprints/SPRINT-007-mtp.md`
- Deferred items: `docs/sprints/SPRINT-007-mtp-DEFERRED.md`
