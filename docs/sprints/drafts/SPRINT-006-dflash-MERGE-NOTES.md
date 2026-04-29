# Sprint 006-dflash Merge Notes

**Date**: 2026-04-29
**Workflow**: sprint-plan skill, partially executed.

## What ran

- **Phase 0** (weather report): pulled the Sprint Planning row.
  Consensus operator on 2026-04-29 is `consensus(opus-4.7, gpt-5.4)`
  with max / extra-high effort, with Gemini as a third ideator when
  available.
- **Phase 1** (orient): inventoried VISION (none), prior deferred
  (SPRINT-004-DEFERRED-dflash.md), prior follow-ups (SPRINT-004 +
  SPRINT-005 dflash followups), existing 006 stub (EAGLE3
  productionization, never executed).
- **Phase 2** (intent): wrote
  `drafts/SPRINT-006-dflash-INTENT.md`.
- **Phase 3** (parallel drafts):
  - Codex (gpt-5.4 high) ✅ — strong draft, picked 5 experiments
    (E3, E5, E1, E2, E4) with hypothesis / method / decision rule.
    16k. Foundation of the merge.
  - Claude (opus-4.7 max) ❌ — failed: model `opus-4.7` not
    available via this CLI/account. No retry attempted; user said
    "focus on planning".
  - Gemini (gemini-3.1-pro-preview-customtools) ❌ — failed: 404
    on the model name. No retry.
- **Phase 4** (cross-critique): **skipped**. Only one draft to
  compare; cross-critique is meaningful with ≥2.
- **Phase 5** (interview): single AskUserQuestion call with three
  questions (naming, scope, GPU sequencing). User picked all
  recommendations.
- **Phase 6** (merge): this document, plus SPRINT-006-dflash.md and
  SPRINT-006-DEFERRED-dflash.md.

## Codex draft strengths (kept)

- Tight 5-experiment selection with explicit decision rule per
  experiment.
- Sprint 007 decision tree based on possible findings.
- Architecture diagram framing investigation as instrumentation
  loop, not feature surface.
- Discipline section: every experiment uses the same
  baseline/canary/full-matrix pattern.

## Additions from intent + interview

- **Phase 0 GPU-free pre-flight**: instrumentation, plumbing,
  harness updates, doc review. Sprint can start now even though
  the user's risk-embedder is training on the only GPU.
- **GPU-blocking annotation**: Phases 1–5 explicitly marked. The
  build / runtime work in those phases waits for training to
  finish.
- **F-016 metric consumption**: bench harness updated to read the
  new `draft_n_generated` / `draft_n_acc_tokens` fields landed in
  fork commit `a2a6168ea`. Republish Sprint 005 numbers with the
  corrected metric as part of Phase 0.
- **Naming change applied**: existing `SPRINT-006-dflash.md` (EAGLE3
  stub) renamed to `SPRINT-007-dflash.md`. The renaming preserves
  EAGLE3 as a future option but unblocks the 006 slot for
  investigation (which has to come first now that Sprint 005's gate
  failed).
- **Push gating clarified**: not blocking. The repo's
  `docker/Dockerfile.local` override (left over from Sprint 005's
  F-014 fix iterations) lets Sprint 006 build from local fork
  checkout. The user pushes when convenient; we update the pin
  later.
- **Fork-side commits are local-only**: documented in Sprint
  doc. Reproducibility risk noted.

## Critiques accepted (self-critique since no peer drafts)

- Codex's risk row "Investigation scope balloons" is real — added
  an explicit go/no-go after Phase 5 before considering E6–E11.
- Codex's "Instrumentation perturbs the very performance being
  measured" risk is real — added a no-instrumentation baseline
  rerun as a prerequisite for the cost-profile experiment (E3).
- Codex's "Adaptive heuristic fixes one profile but not the other"
  is real — kept `qwen` as the gate, `qwen36` as a single
  confirmation cell.

## Critiques rejected

- None. The codex draft was tight enough that no significant
  rewrites were needed.

## Interview refinements applied

1. Naming: 006 = investigation, 007 = EAGLE3 stub renamed.
2. Scope: 5-experiment cut (codex's recommendation) accepted.
3. GPU: Phase 0 instrumentation + harness work runs without GPU;
   Phase 1–5 wait for training to finish.

## Open items resolved by deferral

- **EAGLE3 productionization** (D-001 from Sprint 004): now
  documented in `SPRINT-007-dflash.md` (renamed). Stays deferred,
  triggered by Sprint 006's findings if investigation says
  "DFlash structurally weak, try EAGLE3".
- **F-012 (Q5/Q8 GGUFs)**: deferred — not in Sprint 006 scope
  unless a target-quant sweep is added later (which would be E8,
  not in the cut).
- **F-015 (pytest force-reject redesign)**: deferred — not on
  the investigation critical path. Will become Sprint 007+ work.
- **E6 (offline alignment), E7 (EAGLE3), E8 (target quant), E9
  (multi-slot), E10 (custom draft), E11 (z-lab reference)**:
  deferred. Each only fires if Phase 5 leaves the cause
  ambiguous. Captured in `SPRINT-006-DEFERRED-dflash.md`.

## Open questions for user-during-execution

The codex draft's "Open Questions" section (5 questions) carries
over to the final sprint. Sprint executor decides during run
whether to escalate any to the user.
