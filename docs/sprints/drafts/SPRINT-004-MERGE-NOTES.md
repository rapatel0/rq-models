# Sprint 004 Merge Notes

**Date**: 2026-04-26
**Inputs**: SPRINT-004-INTENT.md (revised v2 with Critical Architecture Finding),
SPRINT-004-CLAUDE-DRAFT.md (v2 hybrid-aware), SPRINT-004-CODEX-DRAFT.md (v2),
SPRINT-004-CLAUDE-CRITIQUE.md, SPRINT-004-CODEX-CRITIQUE.md, interview answers.

Two-way consensus (opus-4.7 + gpt-5.4); Gemini was unavailable due to quota.
v1 drafts (`*-v1.md`) preceded the hybrid-architecture finding and are
preserved for history but not used for this merge.

---

## Architectural finding mid-planning

Mid-Phase-3, the user revealed that `llama_memory_recurrent::seq_rm()` returns
`false` for any partial removal that includes the final position — by design,
because recurrent state is not decomposable by token position. Verification
from `config.json` for both production targets confirmed:

- **Qwen3.6-27B "dense"**: 64 layers, **48 linear_attention** + 16 full_attention
- **Qwen3.6-35B-A3B MoE**: 40 layers, **30 linear_attention** + 10 full_attention

Both production targets are **75% recurrent-state hybrid**. v1 drafts assumed
pure attention; both were re-run with the corrected brief. The hybrid finding
forced a structural change to the sprint:

- **Dropped**: ~80 LOC of `kv_cache_quantized_seq_rm()` block-aware partial-
  truncation work (Claude v1) — superseded by upstream speculative checkpointing
  (PRs #19493 + #22227, already in master).
- **Dropped**: Standalone autoregressive `qwen36-spec` profile *as a `seq_rm`
  correctness vehicle* (Codex v1) — doesn't validate the right thing because
  hybrid layers go through checkpointing, not `seq_rm`.
- **New**: speculative checkpointing × RotorQuant deferred-K interaction surface.
  Snapshot must capture both deferred f16 staging and quantized planar/iso K,
  AND the recurrent state of linear_attention layers.

---

## Strengths kept from each draft

### From Claude v2

- **"Two invariants" framing** (§Architecture): real backend buffer, not
  dequantized view; verify forbidden during deferred staging. Better mental
  model than Codex's prose.
- **Three-stage delivery diagram** (§Sprint scope diagram).
- **`Hybrid speculative checkpointing × RotorQuant deferred-K`** section —
  the strongest single section in either draft. Survived merge with minor
  additions for recurrent-state coverage.
- **Sprint scope diagram** clarity around what's in vs out.
- **Realistic 1.5–2.0× framing** in Overview — kept; widened per interview Q2.
- **Phase 3 fork-level C++ test** — preserved, but renamed and extended
  (Claude critique §2.1).

### From Codex v2

- **Phase ordering with explicit dependency chain**: rebase → checkpoint
  validation → DFlash cherry-pick → Docker profiles → benchmarks. Cleaner.
- **Pinned commit lists** with explicit SHAs for cherry-picks (rebase target,
  EAGLE3 base stack, DFlash delta).
- **`SPECULATIVE_MODE`, `DRAFT_MODEL_NAME`, etc. env var design** for
  `entrypoint.sh` — concrete and avoids forking entrypoint into 3 copies.
- **`--enable EXPERIMENTAL` flag pattern** for MoE profile gating.
- **"Do not resurrect snapshot-and-replay"** — shows Codex actually read
  upstream commit history.

---

## Critiques accepted

### Both critiques flagged (highest priority)

| # | Issue | Resolution |
|---|-------|-----------|
| C1 | Recurrent state of linear_attention layers untested in proposed checkpoint test | **Accepted (hard DoD gate)** — interview Q1. Test renamed `tests/test-checkpoint-hybrid-state.cpp`; covers (a) deferred K, (b) all 4 quantized K layouts, (c) SSM/recurrent state on at least one linear_attention layer, (d) cross-layer mixed batch |
| C2 | Speedup gate scope: single quicksort prompt is too narrow given 2.5× spread on PR data | **Accepted (median gate)** — interview Q2. Median across 3-5 prompts ≥ 1.3×; quicksort headline ≥ 1.5× |
| C3 | z-lab pytorch reference scripted but not gated | **Accepted (hard DoD gate)** — interview Q4. ≥64 token match + acceptance rate ±5pp |
| C4 | Forced-rejection coverage too weak; greedy equivalence can mask silent recurrent corruption | **Accepted (both: fault injection + curated prompts)** — interview Q6. Adds `LLAMA_SPEC_FORCE_REJECT_AT=N` debug env var in `common/speculative.cpp` |

### Claude critique acceptances (reviewing Codex v2)

| # | Issue | Resolution |
|---|-------|-----------|
| Cl-1 (§2.3) | Snapshot cost gated only on downstream speedup | **Accepted** — interview Q5. Numeric ceiling at Phase 2 (≤5 ms snapshot+restore at 65K planar3, sized during the spike) |
| Cl-2 (§2.5) | Speedup gate spans prompts inconsistently | Resolved by C2 above |
| Cl-3 (§2.7) | No risk row for recurrent-state corruption | **Accepted** — added as primary risk |
| Cl-4 (§2.8) | Pseudocode elides hybrid mechanics | **Accepted** — annotate recurrent path in architecture diagram |
| Cl-5 (§2.9) | `entrypoint.sh` refactor scope is one bullet | **Accepted** — split into 3 sub-tasks with "preserve all 8 existing profile launches" sub-gate |
| Cl-6 (§4.1) | Convert-during-checkpoint TOCTOU edge case | **Accepted** — added to Phase 2 test matrix |
| Cl-7 (§4.2) | Snapshot device residency unspecified | **Accepted** — Phase 1 source-spike (interview Q7) covers this |
| Cl-8 (§4.4) | Forced-rejection mechanism unspecified | Resolved by C4 |
| Cl-9 (R-G7) | Acceptance-rate degradation from quantized K | **Accepted** — added as risk + measured in Phase 5 bench |

### Codex critique acceptances (reviewing Claude v2)

| # | Issue | Resolution |
|---|-------|-----------|
| Cx-1 (W#1) | Cost model needs split: full_attention KV / recurrent state / V cache / target vs draft | **Accepted** — added to Phase 2 task list and BENCHMARK-REPORT §10 schema |
| Cx-2 (W#3) | Phase 3 buffer-mutation tests don't exercise checkpoint→verify→partial reject→restore→replay | **Accepted** — Phase 3 test matrix expanded |
| Cx-3 (W#4) | Phase 3 only names planar3/iso3 subtests for 4-mode support | **Accepted** — interview Q3 (all 4 KV types bit-exact) |
| Cx-4 (W#5) | Scope contradiction: standalone autoregressive out, but Phase 6 still benchmarks `target+autoregressive draft` | **Accepted** — kept as comparison baseline only, explicitly labeled "non-hybrid baseline reference" |
| Cx-5 (W#6) | Speedup gate inconsistency between Phase 6 / DoD / Risks | Resolved by C2 |
| Cx-6 (W#7) | DoD validates 8 existing profiles but not new DFlash profiles | **Accepted** — added DoD line for new profile health checks |
| Cx-7 (W#8) | Use Cases makes premature latency claims | **Accepted** — softened until Phase 5 measurements land |
| Cx-8 (Risk gaps) | No risk for happy-path L2 passing without checkpoint restore ever firing | **Accepted** — added; mitigated by fault injection (C4) |
| Cx-9 (Risk gaps) | Rejection timing around deferred-K boundary | **Accepted** — added to Phase 3 test matrix |
| Cx-10 (Edge cases) | Long-context VRAM peak under target+draft+checkpoint | **Accepted** — Phase 2 measures explicitly at 8K/16K/32K/65K/131K/262K |

---

## Critiques rejected (with reasoning)

### From Claude critique (reviewing Codex v2)

- **§2.6 "z-lab silently dropped"** — partially merged: z-lab kept but hardened
  to a DoD gate per interview Q4, not dropped entirely.
- **§4.2 "Delta vs full snapshot path"** — partially merged: instead of
  expanding scope to add COW/delta support, Phase 1 source-spike (interview
  Q7) determines which model upstream uses. If full-copy and unacceptable,
  the abort condition fires; we don't take on COW work in this sprint.

### From Codex critique (reviewing Claude v2)

- **§2 "long-context snapshot model is wrong"** — not rejected exactly, but
  reframed: the *risk* is correct; the *measurement* fix is the same as the
  state-accounting work we're adding (Cx-1). One change, not two.
- **§5 "single consistent answer on whether <1.5× is sprint failure"** —
  resolved by interview Q2 making the median ≥1.3× the hard gate; quicksort
  ≥1.5× is reported as a soft headline, not a hard gate. Below median 1.3×
  is sprint failure.

---

## Interview refinements applied

| Q | Decision | Where applied in final sprint |
|---|----------|-------------------------------|
| Q1: Recurrent state coverage | Hard DoD gate | DoD line, Phase 2 task, test file rename to `test-checkpoint-hybrid-state.cpp` |
| Q2: Speedup gate scope | Median across 3-5 prompts ≥ 1.3×; quicksort headline ≥1.5× | DoD lines, Phase 5 bench design, Risks |
| Q3: KV coverage | All 4 KV types bit-exact | DoD line, Phase 2 test matrix |
| Q4: z-lab gate | Hard DoD gate (≥64 token match + accept ±5pp) | DoD line, Phase 5 task, Dependencies |
| Q5: Snapshot cost gate | Numeric ceiling at Phase 2 before DFlash | DoD line, Phase 2 task, Risks |
| Q6: Forced rejection | Both fault injection + curated prompts | Phase 3 test, Phase 5 e2e test, fork-level patch in `common/speculative.cpp` |
| Q7: Spike depth | Source read + 1-paragraph summary | Phase 1 task, BENCHMARK-REPORT §10 |

---

## Final sprint shape

- **6 phases**, ~3-4 weeks of single-engineer effort:
  - Phase 1: Rebase + checkpoint architecture spike (~25%)
  - Phase 2: Checkpoint × deferred-K validation (all 4 KV types, recurrent state, snapshot cost ceiling, fault injection patch) (~25%)
  - Phase 3: Cherry-pick DFlash from PR #22105 (~15%)
  - Phase 4: Docker profiles + entrypoint refactor (~10%)
  - Phase 5: Validation harness + z-lab + benchmarks (~15%)
  - Phase 6: Documentation + ship gates (~10%)

- **8 hard DoD gates** (vs Codex v2's 9 mostly-vacuous + Claude v2's 11
  mixed). All concrete, all measurable, all on the hybrid surface.

- **Stack-cut deferred items**: EAGLE3 model graph, multi-slot speculative,
  non-greedy validation, MoE deep-dive, COW upstream contribution.
