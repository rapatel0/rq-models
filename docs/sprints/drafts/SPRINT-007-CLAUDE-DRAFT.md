# Sprint 007-mtp: True Multislot MTP Draft Inference

> **Track suffix**: `-mtp` — this sprint is MTP-specific, even though it
> shares server-side `common_speculative` plumbing with the older DFlash and
> EAGLE3 sprint track. See `SPRINT-ROADMAP-dflash.md` for the broader
> speculative-decoding roadmap.

**Status**: Planning (draft)
**Created**: 2026-05-18
**Depends on**: nothing in the local tree at HEAD `53ae72f`; the homelab
agent MTP matrix referenced in the intent must be imported in Phase 0.
**Target hardware**: RTX 4090 (24 GB) `gpu-02-4090rtx` homelab node;
local laptop dev box for build/syntax checks only.
**Estimated effort**: 2–3 weeks single-engineer, high variance — the
sprint can legitimately end at instrumentation if upstream `draft-mtp`
needs a deeper scheduling rewrite than the sprint can afford.

**Branches**:
- Repo: `rapatel0/rq-models` `sprint/007-mtp-multislot` off `main` at
  `53ae72f`.
- Upstream patch: a fresh checkout of `ggml-org/llama.cpp` at
  `LLAMA_CPP_REF=b9196`, RotorQuant patch reapplied; multi-slot MTP
  changes layered on top and regenerated into
  `docker/patches/llama-b9196-rotorquant.patch.gz`. Do not edit the
  compressed patch by hand.

---

## Overview

Production today runs profile **B1**: `MODEL_NAME=qwen3.6-27b-mtp`,
`N_PARALLEL=1`, `ctx=196608`, `--spec-type draft-mtp`,
`--spec-draft-n-max 4`, `--spec-draft-p-min 0.75`, `--ubatch-size 32`,
`q4_0` KV cache. The repo intentionally refuses to start an MTP profile
with `N_PARALLEL > 1` — `docker/entrypoint.sh:179-181` hard-exits, and
`k8s/values.yaml:75` documents the limit. The homelab agent matrix that
seeded this sprint shows the cost: MTP-off scales nearly linearly on
the 4090 (`np=1/2/4 = 39.7 / 71.5 / 124.5 t/s`), but MTP-on barely
scales at all (`np=1/2/4 = 68.1 / 73.1 / 77.2 t/s`). The bottleneck is
upstream `draft-mtp` scheduling at the server-context layer, not the
target model's general batching capacity.

Two paths exist for getting more multi-user throughput out of the
deployed config:

1. **Hybrid router** (Sprint 007-INTENT's "path 1", explicitly out of
   scope here): keep MTP-on for `N=1` traffic and route to an MTP-off
   pool for `N>1` traffic. Cheap; ships next week; doesn't fix the
   underlying scheduling issue.
2. **True multislot MTP** (this sprint, path 2): change upstream
   `draft-mtp` so a single draft pass batches all active slots. If the
   matrix's MTP-off `np=4` aggregate of 124.5 t/s reflects the target
   model's available batching headroom, true multislot MTP should clear
   `124.5 * 0.7 = 87 t/s` aggregate at `np=4` while keeping per-slot
   acceptance ≥ B1.

The sprint commits to path 2. The deliverable is **either** a measured
multislot MTP that clears the gates and ships as an experimental Helm
profile, **or** the instrumentation, regenerated patch, and benchmark
artifacts that prove which scheduling point is the bottleneck and feed
the next sprint. **Production B1 stays default for the duration of
this sprint regardless of outcome.**

Out of scope:
- Hybrid router (path 1); a load-aware MTP-disable fallback may ship as
  a Phase 5 safety net but the primary goal is true multislot.
- DFlash / EAGLE3 multislot (already covered by their own sprints,
  separate plumbing).
- Generalizing the batched-draft interface for non-MTP speculative
  modes (open question — see §Open Questions).
- Multi-GPU MTP (single 4090 only).
- New draft training; we use the existing Qwen3.6-27B-MTP-GGUF as-is.

---

## Use Cases

1. **Operator at the homelab 4090 picks B4 over B1 for multi-user
   work**. Today the operator picks A4 (MTP-off) when they want
   multi-user throughput and pays a 1.7× per-user decode latency vs B1.
   After this sprint, an operator can opt into B4 (MTP-on, `np=4`,
   experimental flag) and get aggregate throughput within 70% of A4
   while keeping per-user decode latency competitive with B1.

2. **Operator stays on B1 by default and never notices the sprint**.
   The Helm chart, the compose default profile, and `make
   run-qwen36-27b-mtp` all stay pinned to `nParallel=1`. The
   experimental multislot profile is opt-in via `PREVIEW=1` + an
   explicit `N_PARALLEL=4` env. No regression to existing deploys.

3. **Reviewer can read the bottleneck**. Whether the sprint ships
   multislot or descopes to instrumentation, the artifact in
   `docs/benchmarks/results/mtp-multislot-007.md` answers: "for
   `draft-mtp` on Qwen3.6-27B on a 4090, which line of
   `common_speculative.cpp` / `server-context.cpp` is the
   serialization point that breaks multi-slot scaling?"

4. **Future sprint can reuse the multi-slot draft batch interface for
   DFlash and EAGLE3**. The current MTP path, the DFlash path, and the
   EAGLE3 path all funnel into a shared draft `llama_context` with
   `params_dft.n_parallel = 1`. A batched-slot interface generalizes —
   but only if this sprint's implementation is plumbed at the right
   layer. (Tracked as Open Question 1; the sprint defers the
   generalization unless it falls out for free.)

---

## Architecture

### What's serialized today (paths inside the pinned `b9196` patch)

Per the intent's reading of the upstream code:

```
tools/server/server-context.cpp
    └── one ctx_dft per server (not per slot)
    └── common_speculative_init(ctx_dft, n_parallel)
         creates n_parallel slots inside common_speculative
    └── per slot, per generation step:
         params_dft.n_parallel = 1          ← serialization point
         common_speculative_draft(ctx_dft, ...)
         verify per-slot; accept/rollback per-slot
```

```
common/speculative.cpp
    └── common_speculative_state_draft_mtp
         per-sequence pending_h, verify_h, samplers
         shared MTP draft llama_batch
    └── draft() builds the draft batch from "active slots"
         but the draft forward itself runs at np=1
```

The user-reported matrix proves the target can batch (np=4 MTP-off is
3.1× np=1). The MTP-on multi-slot result (np=4 is 1.13× np=1) shows
that nearly all the batching headroom is consumed by serialization in
the draft+verify path, not by the target's prefill/decode kernel.

### What "true multislot MTP" means

A draft pass that runs forward exactly once for all active slots'
draft tokens combined, with per-slot sequence-ID tagging so the verify
step can dispatch each slot's draft tokens back to its own
`pending_h` / `verify_h` state. Concretely:

```
for each generation step:
    active = slots with pending requests
    batch = []
    for slot in active:
        for k in 0..N_max-1:
            batch.append((slot.last_sampled_h, slot.seq_id, slot.pos))
    one llama_decode(ctx_dft, batch)        ← batched draft inference
    split outputs by seq_id back into per-slot draft token streams
    per slot: target verify + accept/rollback (already independent)
```

This is the same shape the upstream PR #18961 sketches for DFlash
multi-slot; the MTP variant is structurally similar but operates on
the `state_draft_mtp` hidden-state carry rather than a draft-only
model.

### Phase ordering

```
Phase 0:  Setup + matrix import          (~5%,  1 day)
Phase 1:  Static analysis + spike        (~10%, 1.5 days)
Phase 2:  Instrumentation (no semantics) (~15%, 2 days)
Phase 3:  Multislot draft batching       (~25%, 3.5 days)
Phase 4:  Patch regeneration + build     (~10%, 1.5 days)
Phase 5:  Repo gating (experimental)     (~10%, 1.5 days)
Phase 6:  Correctness harness            (~10%, 1.5 days)
Phase 7:  Benchmark matrix + decision    (~10%, 1.5 days)
Phase 8:  Sprint outcome + docs          (~5%,  1 day)
```

Phases 1, 2, 6 are valuable on their own even if Phase 3 doesn't
converge; the sprint is structured so an early descope at Phase 3 still
delivers usable artifacts.

---

## Implementation

### Phase 0: Setup + matrix import (~5%)

**Goal**: branch created, homelab matrix preserved as a checked-in
artifact, image rebuilt from current `main`, B1 baseline reproduced
within 5% on the homelab 4090.

**Files**:
- `docs/benchmarks/results/mtp-multislot-007-baseline.md` (NEW) —
  human-readable summary of the imported matrix plus a fresh local
  reproduction.
- `docs/benchmarks/results/mtp-multislot-007-baseline.json` (NEW) —
  machine-readable, same shape as `bench_n_parallel.py`'s output so
  later phases can diff cleanly.

**Tasks**:
- [ ] Cut `sprint/007-mtp-multislot` off `main` at `53ae72f`.
- [ ] Determine whether the homelab agent commits referenced in the
      intent (`5a8d51e`, `16ec054`, `e930e56`, `dfcbd8f`, `f70afac`)
      are pushed to any remote. If yes, fetch them and preserve any
      benchmark artifacts. If no, treat the matrix in
      `SPRINT-007-INTENT.md` as the canonical artifact and reproduce
      it locally as Phase 0's first task.
- [ ] On `gpu-02-4090rtx`: rebuild image off `main` (no patch changes
      yet). Run `scripts/bench_n_parallel.py` for `np ∈ {1, 2, 4}` on
      both the `qwen36-27b-mtp` profile (B-series, currently fails at
      `np>1`) and `qwen36-27b` (A-series). The MTP `np=2/4` runs will
      fail to start by design; record the failure as confirmation of
      the entrypoint guard. Capture the A-series numbers; check they
      reproduce 39.7 / 71.5 / 124.5 t/s within 10%.
- [ ] Capture the B1 run (`np=1`) and confirm aggregate matches
      reported 68.1 t/s within 5%. If not, halt; the divergence is
      itself a finding and the sprint either fixes it first or
      documents it before continuing.
- [ ] Write `mtp-multislot-007-baseline.md` with both matrices side by
      side and a one-paragraph reading of the bottleneck.

**Phase gate**: artifact committed, B1 reproduced, A4 reproduced. If
A4 doesn't reproduce, root-cause before continuing — drift in the
target-only path invalidates every downstream comparison.

### Phase 1: Static analysis + spike (~10%)

**Goal**: identify the exact upstream lines that serialize draft
inference and decide whether the implementation lives in
`common/speculative.cpp` only, in `tools/server/server-context.cpp`
only, or in both.

**Files (read-only this phase)**:
- Fresh checkout of `ggml-org/llama.cpp@b9196`, RotorQuant patch
  applied for reference.
- `common/speculative.cpp` — focus on
  `common_speculative_state_draft_mtp`, `common_speculative_init`, the
  body of `common_speculative_draft`, and any
  `for (auto & seq : ...)` loop that drives draft forwards.
- `tools/server/server-context.cpp` — focus on `TAG_SERVER_SPEC_REWORK`
  region (per the existing roadmap comment), the per-slot loop that
  collects draft params, and the `common_speculative_draft` call site.

**Tasks**:
- [ ] Read both files end-to-end. Build a callgraph from server slot
      → `common_speculative_draft` → `llama_decode` on `ctx_dft`.
      Mark every place a `for slot in slots` loop touches the draft
      path.
- [ ] Confirm or refute the intent's hypothesis that
      `params_dft.n_parallel = 1` is the immediate serialization
      point. Two real possibilities:
      a) The init line is `n_parallel=1` and the draft `llama_context`
         can simply be reinitialized at a higher `n_parallel`; the
         per-slot loop is the only structural change.
      b) The draft batch is currently single-sequence by construction
         (e.g., the `state_draft_mtp` is hardcoded for one
         `seq_id`); changing `n_parallel` is necessary but not
         sufficient — `pending_h` / `verify_h` need per-seq plumbing.
- [ ] Look for upstream PR #18961 references in the source. If the PR
      is already partially in-tree (annotated as pending), use that as
      a reference design and adopt its naming conventions to minimize
      future rebase pain.
- [ ] Write `docs/sprints/SPRINT-007-spike.md` with a 2-page summary:
      "the change is N lines centered in functions X/Y, and the risk
      is Z."

**Phase gate**: spike doc explicitly classifies the change as either
"≤200 LOC, contained" (continue to Phase 2 confidently) or ">200 LOC,
needs a refactor" (continue but flag the descope decision earlier).

### Phase 2: Instrumentation (~15%)

**Goal**: ship a patch that adds per-slot, per-draft-step timing and
batch-size tracing **without changing semantics**, so any reader can
see exactly where the serialization is being paid. This phase ships
even if Phase 3 doesn't.

**Files**:
- Upstream `common/speculative.cpp` — add timing scaffolding around the
  `llama_decode` call in `common_speculative_draft`. Use
  `ggml_time_us()` to keep dependencies tight.
- Upstream `tools/server/server-context.cpp` — record per-slot draft
  start/end, draft batch tokens, draft tokens emitted, draft tokens
  accepted, slot wait time before draft start.
- `scripts/mtp_probe.py` — extend to consume the new fields (likely
  surfaced via `timings.draft_*` in the completion response, falling
  back to per-server log scrape if upstream's JSON path is too rigid).
- `scripts/bench_n_parallel.py` — extend aggregate output to include
  per-slot acceptance and draft batch size distribution.

**Tasks**:
- [ ] Add a `LLAMA_SPEC_TRACE` env (default off) that gates the
      tracing output. Off ⇒ zero-cost; on ⇒ log a structured JSON line
      per slot per draft step.
- [ ] Surface the four headline numbers via existing
      `timings.*`: `draft_batch_n`, `draft_seqs_n`, `draft_us`,
      `verify_us`. If upstream's `timings` struct doesn't have a clean
      place, emit them under `timings.spec.*`.
- [ ] Extend `mtp_probe.py` with `--trace-out <path>` that captures the
      JSON trace lines from the server log into a file the bench
      harness can summarize.
- [ ] Re-run the baseline matrix with `LLAMA_SPEC_TRACE=1`. Append
      results to `mtp-multislot-007-baseline.md` with a histogram of
      `draft_batch_n` per `N_PARALLEL` level. The expected reading:
      MTP-on at `np=4` shows `draft_batch_n ≈ 1` even though four
      slots are active — that's the serialization, visible.

**Phase gate**: instrumentation artifact shows a `draft_batch_n` per-step
distribution. The histogram either confirms the serialization
hypothesis or surfaces a different bottleneck (e.g., verify-side
sequential pacing, KV-eviction storm). Either way, Phase 3's
implementation target is now grounded in measured data.

### Phase 3: Multislot draft batching (~25%)

**Goal**: change the draft path so a single `llama_decode` on `ctx_dft`
processes all active slots' draft tokens in one batch. Semantics
preserved: each slot's draft stream and accept/reject decisions are
identical to single-slot behavior.

**Files (upstream)**:
- `common/speculative.cpp` — most of the change lives here. The
  `state_draft_mtp` struct grows from per-context to per-sequence; the
  `common_speculative_draft` body builds a multi-seq draft batch
  instead of looping per slot.
- `tools/server/server-context.cpp` — collect per-slot draft params
  into a vector once, call `common_speculative_draft` once with the
  vector, then dispatch per-slot acceptances.
- `common/speculative.h` — interface evolution. Keep the single-slot
  call signature as a thin wrapper that calls the new multi-slot path
  with a vector of length 1. **No fork-internal callers of the old
  signature are broken.**

**Files (repo)**:
- `docker/patches/llama-b9196-rotorquant.patch.gz` — regenerated from
  the fresh upstream checkout via the existing mise task
  (`mise run llama-cpp-rebase`, per commit `53ae72f`).

**Tasks**:
- [ ] Apply the existing RotorQuant patch to a fresh `b9196` checkout
      in a worktree.
- [ ] Implement multi-slot `state_draft_mtp` (per-seq `pending_h`,
      `verify_h`, samplers; previous per-context fields become
      per-`seq_id` indexed).
- [ ] Rewrite `common_speculative_draft` to assemble a batched
      `llama_batch` across slots and issue one `llama_decode`.
- [ ] Update `tools/server/server-context.cpp` to call the new path
      once per generation step rather than per slot.
- [ ] Keep the single-slot path bit-exact: when `n_active_slots == 1`,
      the produced draft tokens, acceptance decisions, and rollbacks
      must match B1 byte-for-byte against Phase 0's saved B1 outputs.
- [ ] Static check: `llama-server --help` still advertises
      `draft-mtp` and accepts `--spec-draft-n-max` / `--spec-draft-p-min`.
- [ ] Local laptop build: image builds, `llama-server` boots with a
      tiny model + `--parallel 2 --spec-type draft-mtp
      --spec-draft-n-max 2 --no-warmup`, two concurrent toy requests
      return coherent text. (No throughput claims at this scale.)

**Phase gate**: build clean, B1 byte-exact preserved on a single-slot
toy run, two concurrent toy requests both produce coherent text with
nonzero accepted drafts. **No throughput gate yet** — that's Phase 7.

If this phase reveals a >2× larger LOC change than Phase 1 estimated
(e.g., `state_draft_mtp` needs to become a multi-sequence object that
ripples into the rollback path), call the descope decision here:
ship Phases 0–2 + instrumentation, mark Phase 3 as "next-sprint
descope", document the structural blockers, do not flip Phase 5's
experimental gate.

### Phase 4: Patch regeneration + build (~10%)

**Goal**: the multi-slot changes live in a compressed patch that
applies cleanly to a fresh upstream `b9196` checkout. Docker build
reproduces from scratch.

**Files**:
- `docker/patches/llama-b9196-rotorquant.patch.gz` (regenerated).
- `docker/Dockerfile` — no changes expected; the
  `ARG ROTORQUANT_PATCH=llama-b9196-rotorquant.patch.gz` indirection
  already handles it.
- `mise.toml` (if present) — confirm the rebase task captures the
  multislot delta; otherwise extend.

**Tasks**:
- [ ] From the dev worktree, produce a clean diff against
      `b9196 + RotorQuant base patch`. Verify the diff applies in
      reverse cleanly (no merge artifacts).
- [ ] `gzip` the diff into the patches dir.
- [ ] `docker build` from a clean clone (or `git clean -fdx /src`-equivalent
      in the build stage). Confirm reproducibility.
- [ ] `make build && make run-qwen36-27b-mtp-bg` boots; tear down
      cleanly with `make stop`.

**Phase gate**: clean rebuild from a fresh checkout produces a working
image; no `git apply` warnings.

### Phase 5: Repo gating — experimental profile (~10%)

**Goal**: the multi-slot MTP profile is reachable via opt-in only;
existing B1 deployments behave identically.

**Files**:
- `docker/entrypoint.sh` — relax the `N_PARALLEL != 1` exit when
  `MTP_MULTISLOT=1` is set, with a loud warning banner. Otherwise the
  old guard stays.
- `docker-compose.yml` — add a new profile `qwen36-27b-mtp-multislot`
  with `N_PARALLEL=4`, `MTP_MULTISLOT=1`, and `PREVIEW=1` gate (mirror
  the DFlash PREVIEW pattern).
- `k8s/values.yaml` — new `mtpMultislot: false` flag plus comment
  explaining the experimental status. The deployed default stays
  `nParallel: 1`.
- `k8s/templates/deployment.yaml` — surface `MTP_MULTISLOT` env when
  `.Values.mtpMultislot` is true and `.Values.nParallel > 1`.
- `k8s/README.md` — documentation paragraph and a warning that B1
  remains the supported default until this sprint's gates close.
- `Makefile` — new `run-qwen36-27b-mtp-multislot[-bg]` target plus
  appropriate `stop` / `logs` profile inclusion.

**Tasks**:
- [ ] Wire the env gate through entrypoint. Keep the guard's failure
      mode obvious: without `MTP_MULTISLOT=1`, an MTP profile with
      `N_PARALLEL>1` still exits with the original error.
- [ ] Smoke `make run-qwen36-27b-mtp-bg` (B1) — unchanged behavior,
      stays on `np=1`.
- [ ] Smoke `MTP_MULTISLOT=1 N_PARALLEL=2
      make run-qwen36-27b-mtp-multislot-bg` — boots; `/health`
      returns 200; two concurrent toy requests both return coherent
      text.

**Phase gate**: existing B1 paths unchanged; new B2/B4 path opt-in
reachable; loud warning printed on stdout when B2/B4 is enabled.

### Phase 6: Correctness harness (~10%)

**Goal**: a deterministic byte-equal check across multi-slot
configurations. This is the gate the intent prioritizes over
throughput, and a failure here halts the sprint regardless of how good
the throughput numbers look.

**Files**:
- `scripts/mtp_correctness.py` (NEW) — multi-slot byte-equal harness.
  Mirrors `scripts/validate_dflash.py`'s shape but extended for
  concurrent slots.
- `tests/test_mtp_multislot.py` (NEW) — pytest wrapper that boots the
  multislot profile, runs the harness, and asserts pass.
- `scripts/mtp_probe.py` — `--save-tokens <path>` flag if not already
  present.

**Test plan**:
1. A1 reference: target-only `qwen36-27b` greedy, `--temp 0
   --top-k 1 --seed 42`, three distinct prompts, 256 tokens per
   prompt. Save tokens.
2. B1 reference: same prompts, MTP-on single-slot. Save tokens. Byte
   match A1.
3. B2/B4 test: same prompts, MTP-on multislot, each prompt routed to a
   different slot. Each slot's token sequence must match the
   corresponding A1/B1 reference byte-for-byte.
4. Rejection-heavy mixed: one slot gets a "predictable" prompt
   (formatted code), one gets a "rejection-heavy" prompt (free-form
   reasoning). Run with `LLAMA_SPEC_FORCE_REJECT_AT` if Sprint 005's
   force-reject env is in tree; else rely on natural rejection. Both
   slots' outputs match their references.
5. Long-prompt: one slot gets a 4k-token prefix. Single-slot
   equivalence preserved.

**Tasks**:
- [ ] Write the harness. Reuse `bench_n_parallel.py`'s concurrent
      request infrastructure for the fan-out.
- [ ] Run against B1 first to validate the harness itself.
- [ ] Run against B2, then B4. Any divergence is a Phase 3 bug — drop
      back into the patch and root-cause before continuing.

**Phase gate**: all five test cases pass with 256/256 token match per
slot. Without this gate, **the sprint must not flip the experimental
profile and must not publish a "ready to use" recommendation**.

### Phase 7: Benchmark matrix + decision (~10%)

**Goal**: the full A/B matrix, head-to-head, with enough granularity
to decide ship-vs-instrumentation-only.

**Files**:
- `docs/benchmarks/results/mtp-multislot-007.md` — headline report.
- `docs/benchmarks/results/mtp-multislot-007.json` — machine artifact.

**Matrix**:

| Profile | MTP | N_PARALLEL | Expected from intent | Gate |
|---|---|---|---|---|
| A1 | off | 1 | 39.7 t/s | regression ≤5% vs intent matrix |
| A2 | off | 2 | 71.5 t/s | regression ≤5% |
| A4 | off | 4 | 124.5 t/s | regression ≤5% |
| B1 | on | 1 | 68.1 t/s | regression ≤5% vs B1 production |
| B2 | on | 2 | ≥ 1.0× B1 per-slot, aggregate ≥ B1 | soft |
| B4 | on | 4 | ≥ 1.4× B1 aggregate AND ≥ 0.7× A4 aggregate | **hard** |

Per the intent: if B4 doesn't clear both gates, the experimental
profile does not ship — only the patch, the instrumentation, and the
report ship, and B1 stays default.

**Tasks**:
- [ ] Run the matrix on `gpu-02-4090rtx`, three trials per cell, same
      prompt set across cells.
- [ ] Capture per-slot acceptance ratio at B2 and B4. Per the intent,
      acceptance must stay within ±5pp of B1's acceptance on the same
      prompt set.
- [ ] Emit the JSON artifact (same shape as `bench_n_parallel.py`
      output, extended with `acceptance` and `draft_batch_n`
      histogram).
- [ ] Write the headline report with: matrix table, acceptance table,
      ship/no-ship decision, observed bottleneck (informed by Phase 2
      trace data).

**Phase gate**: report committed; decision recorded; if "ship," the
PREVIEW gate on the compose profile can be flipped at Phase 8; if "no
ship," PREVIEW stays on and the experimental Helm flag stays opt-in
behind a stronger warning.

### Phase 8: Sprint outcome + docs (~5%)

**Files**:
- `docs/sprints/SPRINT-007-mtp.md` — the final merged sprint doc, status:
  complete (with or without followups).
- `docs/sprints/SPRINT-007-FOLLOWUPS.md` — execution-discovered items.
- `README.md` — add a "Speculative Decoding · MTP multi-slot" note
  whether the gate passed or not; honesty.
- `docs/BENCHMARK-REPORT.md` — append §11 (or wherever the next
  section sits) with the headline B4-vs-B1-vs-A4 numbers.
- `docs/sprints/SPRINT-ROADMAP-dflash.md` — Sprint 007 status flipped
  from "sketched" to "complete" or "partial".

**Tasks**:
- [ ] Update sprint status.
- [ ] Capture any followups, especially: generalizing the batched
      interface for DFlash/EAGLE3, the load-aware fallback as a
      future safety net, any upstream patch contribution opportunity.
- [ ] User-approved merge into `main`. **B1 remains the helm
      default. The compose default profile remains B1.** That is
      non-negotiable per the intent's "Constraints" section, even if
      every gate passed.

---

## Files Summary

| File | Action | Phase | Purpose |
|---|---|---|---|
| `docs/sprints/SPRINT-007-mtp.md` | Create | 8 | Final merged sprint doc |
| `docs/sprints/SPRINT-007-FOLLOWUPS.md` | Create | 8 | Discovered followups |
| `docs/sprints/SPRINT-007-spike.md` | Create | 1 | Spike summary; load-bearing for Phase 3 scope decision |
| `docs/benchmarks/results/mtp-multislot-007-baseline.md` | Create | 0 | Imported + reproduced matrix |
| `docs/benchmarks/results/mtp-multislot-007-baseline.json` | Create | 0 | Machine artifact for the baseline |
| `docs/benchmarks/results/mtp-multislot-007.md` | Create | 7 | Final headline matrix + decision |
| `docs/benchmarks/results/mtp-multislot-007.json` | Create | 7 | Machine artifact for the final matrix |
| `docker/patches/llama-b9196-rotorquant.patch.gz` | Modify (regenerate) | 4 | Carries the multislot MTP changes |
| `docker/entrypoint.sh` | Modify | 5 | Relax `N_PARALLEL` guard behind `MTP_MULTISLOT=1` |
| `docker-compose.yml` | Modify | 5 | New `qwen36-27b-mtp-multislot` profile, PREVIEW-gated |
| `Makefile` | Modify | 5 | `run-qwen36-27b-mtp-multislot[-bg]`, stop/logs profile inclusion |
| `k8s/values.yaml` | Modify | 5 | `mtpMultislot: false` flag + documentation |
| `k8s/templates/deployment.yaml` | Modify | 5 | Surface `MTP_MULTISLOT` env when configured |
| `k8s/README.md` | Modify | 5 | Document experimental status; B1 stays supported default |
| `scripts/mtp_probe.py` | Modify | 2, 6 | Trace consumption; `--save-tokens` |
| `scripts/bench_n_parallel.py` | Modify | 2, 7 | Per-slot acceptance + draft batch histogram |
| `scripts/mtp_correctness.py` | Create | 6 | Multi-slot byte-equal harness |
| `tests/test_mtp_multislot.py` | Create | 6 | pytest wrapper around the harness |
| `docs/BENCHMARK-REPORT.md` | Modify | 8 | New §11 with the multislot MTP results |
| `README.md` | Modify | 8 | Speculative-decoding MTP multi-slot status |
| `docs/sprints/SPRINT-ROADMAP-dflash.md` | Modify | 8 | Sprint 007 status update |

---

## Definition of Done

### Hard gates (sprint fails if any miss)

1. **Phase 0 baseline reproduced**: B1 within 5% of the intent's
   reported 68.1 t/s; A4 within 5% of 124.5 t/s on the homelab 4090.
   If not, root-cause before continuing.
2. **Phase 6 correctness — single-slot bit-exact**: B1 token sequence
   byte-equal vs B1 production output on three distinct prompts, 256
   tokens each, greedy seed=42.
3. **Phase 6 correctness — multi-slot bit-exact**: each slot at
   `np=2` and `np=4` produces a token sequence byte-equal to its
   single-slot reference on the same three prompts.
4. **Phase 6 correctness — rejection-heavy and long-prompt cases
   pass**: byte-equal preserved across all five test cases.
5. **Acceptance preservation**: per-slot MTP acceptance at `np=2` and
   `np=4` stays within ±5pp of B1's acceptance on the same prompt
   set, OR the report explicitly documents the acceptance collapse
   and the sprint does not ship the experimental profile.
6. **Throughput gate (only required if shipping experimental
   profile)**: B4 aggregate ≥ 1.4× B1 aggregate AND B4 aggregate ≥
   0.7× A4 aggregate. If either fails, the sprint **may still close
   successfully** but the experimental profile must remain
   PREVIEW-gated with documented expected throughput.
7. **B1 regression gate**: B1 single-user decode within 5% of the
   intent's 68.1 t/s after the patch is applied.
8. **A4 regression gate**: MTP-off A4 aggregate within 5% of 124.5 t/s
   after the patch is applied. (Multislot MTP must not regress the
   MTP-off path.)
9. **Deployment gate**: Helm default `nParallel=1`; compose default
   profile is B1; `make run-qwen36-27b-mtp` boots B1; opt-in path
   requires explicit `MTP_MULTISLOT=1` (compose) or `mtpMultislot:
   true` (Helm).
10. **Patch hygiene**: regenerated `llama-b9196-rotorquant.patch.gz`
    applies cleanly to a fresh upstream `b9196` checkout; `docker
    build` reproduces from scratch.

### Soft gates

- **Trace artifact** explains the residual bottleneck for any
  multislot profile that misses the throughput gate.
- **Instrumentation surfaced via `timings.spec.*`** so future
  benchmarks don't need ad-hoc log scraping.
- **Spike doc (`SPRINT-007-spike.md`) bounds the change size** before
  Phase 3 begins; if reality wildly exceeds the bound, that's logged
  as a lesson learned.

### Code hygiene

- All git operations: `git add -u` or explicit file lists (per user's
  global instructions).
- Commit messages: imperative subject + Co-Authored-By trailer.
- No fork edits in `/tmp` — all upstream changes go through the
  worktree → regenerate patch flow.
- No `.env` / `HF_TOKEN` committed.
- Sprint branch `sprint/007-mtp-multislot`; merge to `main` only after
  explicit user approval.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Upstream `draft-mtp` needs a deeper rewrite than Phase 1's spike estimates (the `state_draft_mtp` per-context → per-sequence promotion ripples into rollback / KV bookkeeping) | Medium-high | High | Phase 1 budgets a dedicated 1.5-day spike; Phase 3 has an explicit "stop and descope" exit if reality > 2× the bound. Even on descope, Phases 0–2 ship usable instrumentation. |
| Acceptance collapses under multi-slot (e.g., shared sampler state across sequences, hidden-state interference) | Medium | High | Phase 6's per-slot acceptance vs single-slot reference is a hard gate; sub-5pp is the ship bar, anything worse is a clear "instrumentation-only" outcome. |
| Multi-slot draft path drifts B1 single-slot output (subtle ordering bug in the batched path that's invisible at `np=1`) | Medium | High | Phase 3's first acceptance test is byte-exact preservation at `np=1`; this happens before any throughput measurement. |
| KV/VRAM budget on the 4090 (B1 is 21.4 / 23 GB with ~1.2 GB free per the intent) blows up under multi-slot KV growth | Medium | High | Phase 5 explicitly does *not* change `ctx_size` for the multislot profile; the per-slot context already gets divided by `n_parallel`. Bench harness aborts a run if VRAM headroom drops below 0.5 GB (mirror Sprint 005's pattern). |
| `b9196` moves (upstream tags can be force-pushed; pin in `Dockerfile` is `b9196` not a commit SHA) | Low | Medium | Phase 4 captures the exact upstream commit SHA in `Dockerfile` if the tag is mutable, and the regenerated patch is tied to that SHA. |
| Hybrid router (path 1) is shipped concurrently by someone else, invalidating part of the sprint's premise | Low | Medium | Sprint is documented as "true multislot" specifically; even if hybrid ships, the path-2 result is independently valuable as a foundation for DFlash/EAGLE3 multislot work. Communicate with the team in Phase 0. |
| Homelab 4090 contention with training jobs causes intermittent OOM / timing noise | High | Low | All bench runs scheduled in <15-min windows, coordinated. `bench_n_parallel.py` already health-snapshots before and after each lane; extend to abort if `/health` shows running slots from another tenant. |
| The reported matrix can't be reproduced in Phase 0 (homelab agent commits not present, environment drift) | Medium | Medium | Treat the intent's matrix as the canonical artifact. Reproduce A1/A4 and B1 with the current image. If A4 doesn't reproduce, the bottleneck reading changes and the sprint may need to refit gates before Phase 3. |
| Generalizing to DFlash/EAGLE3 (open question 1) creeps into the scope | Low-medium | Medium | Sprint explicitly defers generalization; only adopt it if the Phase 3 implementation naturally accommodates DFlash without extra work. Otherwise leave hooks and document in followups. |

---

## Security

- **No new network-facing code**. The trace env (`LLAMA_SPEC_TRACE`)
  logs to stderr and does not open new sockets.
- **No new secrets**. `MTP_MULTISLOT` is a boolean env; `HF_TOKEN` is
  the only credential the entrypoint reads and is unchanged.
- **Experimental profile is opt-in**: relaxing the `N_PARALLEL` guard
  in `entrypoint.sh` is gated on `MTP_MULTISLOT=1` and prints a loud
  warning banner. A misconfigured deploy that forgets to set the env
  continues to fail closed at the existing guard.
- **Upstream patch hygiene**: regenerated patch is reviewed via
  `git diff --stat` and a side-by-side comparison against the pre-Phase-3
  patch in the PR description; no out-of-band binary blobs.
- **Trace output**: trace lines include `seq_id`, draft batch sizes,
  and timings — no prompt text, no completion text, no model weights.
  Safe to ship in container logs.
- **Build reproducibility**: `docker build` reproduces from a fresh
  upstream fetch + patch apply; no implicit toolchain or fetched-blob
  drift.
- **Default-secure deployment**: Helm chart default
  `mtpMultislot: false`, `nParallel: 1`. Production B1 unchanged.

---

## Dependencies

### Prior work in this repo
- Commit `53ae72f` baseline — mise task for llama.cpp rebase workflow.
- Commit `67aee3c` — RotorQuant build rebased onto llama.cpp `b9196`.
- Commit `4fa2a2a`, `3586ced`, `5661c3e` — Qwen MTP draft defaults and
  speed profile already on `main`.

### Upstream
- `ggml-org/llama.cpp@b9196` with RotorQuant patch applied. Multislot
  MTP changes layer on top.
- Upstream PR #18961 (DFlash multi-slot, in flight) referenced as a
  shape reference if visible at sprint start; not required to be
  in-tree.

### External artifacts
- `Qwen3.6-27B-MTP-GGUF` (existing on the `llm-models` volume) —
  unchanged.
- The homelab agent matrix (`np=1/2/4` MTP-on and MTP-off) referenced
  in `SPRINT-007-INTENT.md` — imported in Phase 0.

### Hardware
- `gpu-02-4090rtx` homelab node — RTX 4090 24 GB, the only required
  hardware (per the intent's constraint that scope is single-4090 MTP).
- Laptop dev box — sufficient for build / syntax checks / tiny-model
  smoke. Not used for any throughput claim.

### People
- User as approver of: experimental-profile flip (Phase 5), final
  merge to `main` (Phase 8).
- Homelab 4090 schedule — coordinate around any concurrent training
  work.

---

## Open Questions

1. **Scope of the batched-draft interface — MTP only, or also DFlash
   and EAGLE3?** The intent flags this as Open Question 1. Tentative:
   MTP only this sprint; leave the call-shape compatible so DFlash /
   EAGLE3 can adopt the same path in Sprint 008+. Cost of locking in
   too tight: future sprints re-pay the interface cost. Cost of
   over-generalizing now: drift from upstream PR #18961, more
   rebase pain.

2. **Is `np=2` enough for production, or is the hard target `np=4`?**
   The intent's A4 number (124.5 t/s aggregate) is the
   most-likely-aspirational ceiling; B2 may already meet realistic
   workloads. Tentative: gate on B4 because the A-series data shows
   it's where the gap is largest and the most visible to operators.
   B2 numbers are reported but not gated.

3. **Per-user vs aggregate trade-off**: if multislot MTP improves
   aggregate but reduces per-user decode below B1, does it still
   ship? Intent suggests no (the gates require "beat B1 per-user AND
   aggregate"). Tentative: report per-user; require per-user ≥ 0.95×
   B1 as a soft gate, not hard, because workloads vary. Operator-side
   choice via PREVIEW gating handles the ambiguity.

4. **Load-aware fallback as a safety net**: should this sprint also
   ship a runtime path that auto-disables MTP at `np > 1`, even though
   the primary goal is true multislot? Tentative: only if Phase 3
   descopes; otherwise the fallback is its own follow-up. The
   experimental gate is enough operator protection while multislot
   MTP is preview.

5. **Iterative CUDA / C++ bench access on the 4090**: can the
   implementation agent use `gpu-02-4090rtx` for iterative
   `llama-server` profiling during Phase 3, or is benching gated to
   Phase 7's matrix run? Tentative: iterative is permitted with
   tiny-context smoke (≤4k ctx, ≤256 tokens) any time the homelab
   isn't running training; full-context matrix runs only in a
   scheduled window.

6. **Upstream contribution**: if multislot MTP works, should the
   delta be contributed upstream (independent of #18961) or kept as a
   RotorQuant patch? Tentative: keep as a patch through Sprint 007;
   evaluate upstreaming in the Sprint 007 outcome doc once the patch
   is stable and benchmarked.

7. **Sprint suffix decision**: resolved to `-mtp` so the sprint artifact
   matches the MTP-specific implementation scope.
