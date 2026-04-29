# Sprint 006-dflash Deferred Items

Items proposed during Sprint 006 planning (in the intent doc, the
Codex draft, or the interview) but explicitly excluded from the
final 5-experiment cut. Each item names what, why deferred, target
sprint, prerequisites, and affected files.

---

## D-006-001: E6 — Offline draft-vs-target token alignment

**What**: Run the DFlash draft model alone on each of the 5 prompts
at temp=0/seed=42; run the target model alone with the same params;
diff position-by-position. Measure the upper bound on achievable
acceptance rate independent of speculative-decoding bookkeeping.

**Why deferred**: Sprint 006 has 5 experiments already. E6 is a
*diagnostic upper bound* — useful for interpretation but doesn't
directly produce a remediation candidate. Defer to a "phase 6
extension" if the 5-experiment cut leaves the cause ambiguous.

**Target sprint**: Sprint 006 phase-6 extension if cause ambiguous;
otherwise Sprint 007 if the recommendation calls for it.

**Prerequisites**: standalone draft model runnable via `llama-cli`;
no fork changes.

**Files**: new bench script under `scripts/` (e.g.,
`align_draft_target.py`); results under
`docs/sprints/SPRINT-006-dflash-experiments/E6_alignment/`.

---

## D-006-002: E7 — EAGLE3 as alternative speculative path

**What**: Convert / load Qwen3.6-EAGLE3 draft (if available from
z-lab); benchmark on the same 5-prompt set. Compare real acceptance
rate and tok/s to DFlash. EAGLE3 is autoregressive 1-token-per-step
vs DFlash's 16-token block — different acceptance dynamics.

**Why deferred**: bigger effort (model conversion + new profile
plumbing), and only worth running if Phase 1–5 say DFlash is
structurally weak. The existing `SPRINT-007-dflash.md` (renamed
EAGLE3 productionization stub) already captures the broader
EAGLE3 productization scope.

**Target sprint**: Sprint 007 if Phase 6 decision tree says
"draft/target mismatch structural → EAGLE3 baseline".

**Prerequisites**: Sprint 006 findings document; existing fork
already has EAGLE3 graph from Sprint 004 cherry-pick.

**Files**: `docker-compose.yml` (new EAGLE3 profile);
`docker/entrypoint.sh`; potential new convert step.

---

## D-006-003: E8 — Smaller target quant (Q3_K_M / Q2_K)

**What**: Convert qwen3.6-27b target to Q3_K_M or Q2_K. Re-bench.
Slower target compute → relative speedup from draft increases. Tests
hypothesis 4 (cost ratio target/draft).

**Why deferred**: requires ~50 GB of GGUF download + conversion
work. F-012 in SPRINT-005-FOLLOWUPS-dflash.md already names the
prerequisite. Worth pursuing if Phase 1–5 suggest target compute
isn't the bottleneck — but expensive to set up for a single signal.

**Target sprint**: Sprint 007 or later, contingent on Sprint 006
findings making target quant relevant.

**Prerequisites**: F-012 (Q5/Q8/Q3 GGUFs in llm-models volume).

**Files**: `docker-compose.yml` (variant target profiles);
`docs/sprints/SPRINT-006-dflash-experiments/E8_target_quant/`.

---

## D-006-004: E9 — Multi-slot N_PARALLEL=2 amortization

**What**: Test whether multi-slot batched verify (2 concurrent
requests) amortizes the draft cost across requests, recovering
throughput.

**Why deferred**: F-010 in SPRINT-005-FOLLOWUPS-dflash.md is a known
sub-linear case for DFlash (single shared draft context serializes).
Sprint 005 plan already noted N_PARALLEL=4 expected sub-linear; this
sprint isn't the right place for that work. Defer to the sprint that
does multi-slot batched-draft optimization (originally Sprint 007 in
the roadmap, now Sprint 008+ given the renaming).

**Target sprint**: Sprint 008+ (multi-slot batched-draft).

**Prerequisites**: per-slot speculative state isolation (currently a
known limitation).

**Files**: fork's `common/speculative.cpp` slot-handling;
`docker-compose.yml` N_PARALLEL config.

---

## D-006-005: E10 — Distilled / smaller custom draft

**What**: Train a tiny (~200M-500M) draft on representative operator
prompts, see if cost/acceptance tradeoff improves vs the 1.7B z-lab
draft.

**Why deferred**: model-training work, way out of investigation
scope. Sprint 006 is measurement + small remediation, not custom
training. Only worth it if Phase 6 decision tree says "draft model
quality is the bottleneck and existing alternatives don't help".

**Target sprint**: future sprint, conditional on Sprint 006 findings.

**Prerequisites**: training corpus; distillation harness; storage
for the new draft GGUF.

**Files**: out of repo for training; affects
`docker-compose.yml` profile and `Makefile` if a custom draft ships.

---

## D-006-006: E11 — Z-lab reference acceptance rate comparison

**What**: External diagnostic. Find z-lab's published acceptance
rates on prose-class prompts and compare to ours. If their published
number is also low (e.g., ~5%), our 1–25% is in line and the
investigation should focus on remediation rather than re-diagnosing.

**Why deferred**: external reference work, not a code change. Sprint
006 can pull this in informally during Phase 6 findings if useful;
not a separate experiment phase.

**Target sprint**: Phase 6 of Sprint 006 (informal); not a separate
sprint.

**Prerequisites**: access to z-lab's published bench results.

**Files**: cited in `SPRINT-006-dflash-FINDINGS.md` if used.

---

## D-006-007: F-015 — Pytest force-reject test redesign

**What**: Redesign `tests/test_speculative.py::TestForceReject` so
it actually exercises `LLAMA_SPEC_FORCE_REJECT_AT`. Requires a
server fixture that brings up two distinct containers (one with env
unset, one with env=8) on different ports. Test compares outputs
across the two.

**Why deferred**: not on the investigation critical path. The C++
ctest at `tests/test-checkpoint-hybrid-state.cpp` (fork commit
`afec36229`) is the real validator and doesn't need this redesign.
Pytest fix is a code-hygiene item.

**Target sprint**: Sprint 007 or later code-hygiene sprint.

**Prerequisites**: none (independent of Sprint 006 outcome).

**Files**: `tests/test_speculative.py`; possibly new
`tests/conftest.py`.

---

## D-006-008: F-012 — Q5/Q8 27B GGUFs in llm-models volume

**What**: One-time download of `Qwen3.6-27B-Q5_K_M.gguf` and
`Qwen3.6-27B-Q8_0.gguf` into the `llm-models` Docker volume from
unsloth or equivalent.

**Why deferred**: Sprint 006's 5-experiment cut doesn't include a
target-quant sweep (E8 covers that, also deferred). Only needed if
Sprint 007 or later plans pick up E8.

**Target sprint**: triggered by E8 (D-006-003) above.

**Prerequisites**: HF token with access (if gated); ~50 GB free in
volume.

**Files**: `llm-models` Docker volume contents; no source code
change.

---

## D-006-009: Fork upstream PR preparation

**What**: Re-shape Sprint 005 + 006 fork patches (F-011, F-014,
F-016) for upstream submission to llama.cpp. Per
`llama-cpp-rq/AGENTS.md` policy, fork-side patches need a separate
human-owned cleanup and disclosure pass before upstream.

**Why deferred**: completely separate concern from Sprint 006's
investigation focus. Whether to even pursue upstream depends on
Sprint 006's findings — if the F-014 fix turns out to be a
workaround for a deeper bug, the upstream patch shape will look
different.

**Target sprint**: post-Sprint-006, if findings warrant.

**Prerequisites**: Sprint 006 findings; fork commits pushed publicly;
human-owned PR shaping pass.

**Files**: separate upstream fork; not in this repo.

---

## Summary table

| Item | Target Sprint | Blocker / Trigger |
|------|---------------|-------------------|
| D-006-001: E6 offline alignment | 006 phase-6 extension or 007 | Cause ambiguous after 5-experiment cut |
| D-006-002: E7 EAGLE3 baseline | 007 | "Draft/target mismatch structural" decision branch |
| D-006-003: E8 target quant | 007+ | F-012 GGUFs downloaded; cost-ratio hypothesis prioritized |
| D-006-004: E9 multi-slot | 008+ | Per-slot speculative state isolation |
| D-006-005: E10 distilled draft | future | "Draft quality bottleneck" decision branch + training resources |
| D-006-006: E11 z-lab reference | 006 phase-6 (informal) | None — pull in if findings benefit |
| D-006-007: F-015 pytest redesign | 007+ code hygiene | Independent |
| D-006-008: F-012 Q5/Q8 GGUFs | Triggered by E8 | One-time download |
| D-006-009: Upstream PR prep | post-006 | Sprint 006 findings + human PR shaping |
