# Sprint 007-mtp: True Multislot MTP Draft Inference

> **Track suffix**: `-mtp`. This sprint is MTP-specific. It still works in the
> same llama.cpp `common_speculative` / server scheduling area as the earlier
> DFlash and EAGLE3 plans, but the artifact names should make the MTP scope
> explicit.

**Status**: In Progress
**Created**: 2026-05-18
**Depends on**: current `main` at `53ae72f`, upstream llama.cpp `b9196`
(`7ba22c6a0918b5db16029c2a120bf04a56e78b78`), and the existing RotorQuant
patch-regeneration workflow.
**Target hardware**: homelab RTX 4090 node `gpu-02-4090rtx`, 24 GB VRAM.
**Production baseline to preserve**: `B1` = Qwen3.6-27B MTP, `draft-mtp`,
`n_parallel=1`, `ctx=196608`, `draft_n_max=4`, `draft_p_min=0.75`,
`ubatch=32`, `q4_0` KV.
**Estimated effort**: 2-3 weeks, high variance. The sprint may legitimately
end with instrumentation and a blocked implementation report if upstream
`draft-mtp` needs a deeper refactor than the spike can bound.

**Branches**:
- Repo: `rapatel0/rq-models` `sprint/007-mtp-multislot` off `main`.
- Upstream patch workspace: fresh `ggml-org/llama.cpp` checkout at `b9196`,
  RotorQuant patch applied, MTP multislot changes layered on top, then
  regenerated into `docker/patches/llama-b9196-rotorquant.patch.gz`.

---

## Overview

The deployed MTP path is fast for one stream and poor for concurrent streams.
The user-reported matrix is the trigger for this sprint:

| Mode | `np=1` | `np=2` | `np=4` | `np=1 -> np=4` |
|---|---:|---:|---:|---:|
| A: MTP off aggregate | 39.7 t/s | 71.5 t/s | 124.5 t/s | 3.14x |
| B: MTP on aggregate | 68.1 t/s | 73.1 t/s | 77.2 t/s | 1.13x |

This proves the dense 27B Q4 model on the 4090 can benefit from server slots.
The bottleneck is specific to MTP plus the multi-slot speculative loop. The
current repo correctly protects production by refusing `N_PARALLEL>1` for
`*-mtp` models, but that guard also prevents real multi-user MTP throughput.

Sprint 007 implements or conclusively diagnoses true multislot MTP for upstream
llama.cpp `draft-mtp`. It does not ship the short-term hybrid router as the main
solution. Production stays on `B1` unless the new path clears correctness,
acceptance, throughput, and regression gates.

The main output is one of two acceptable outcomes:

1. A preview-only multislot MTP profile that passes correctness and performance
   gates on the 4090.
2. A committed instrumentation and failure report proving where the upstream
   `draft-mtp` path serializes or loses acceptance, with production still pinned
   to `B1`.

Out of scope:

- Hybrid load-aware routing as the primary deliverable.
- DFlash and EAGLE3 multislot generalization.
- Multi-GPU or tensor-parallel MTP.
- New draft model training or changes to the Unsloth GGUF.
- Making multislot MTP the production default in this sprint.

---

## Use Cases

1. **Multi-client 4090 serving with MTP still enabled**

   An operator can test `np=2` and `np=4` MTP on one 4090 without falling back
   to MTP-off A4. If the gates pass, the profile gives aggregate throughput
   closer to A4 while retaining MTP's single-stream advantage.

2. **Safe solo-agent production**

   Hermes-style single-user loops continue using `B1`. Compose, Helm, and
   `make run-qwen36-27b-mtp-speed` keep `n_parallel=1` by default.

3. **Actionable bottleneck evidence**

   If multislot MTP does not clear the gates, the sprint still answers whether
   the failure is draft context decode, target verify, accept/rollback, sampler
   state, VRAM pressure, or slot scheduling.

4. **Reusable speculative scheduling knowledge**

   The sprint may inform later DFlash/EAGLE3 work, but it does not generalize
   the interface unless that falls out as a low-risk side effect.

---

## Architecture

Current relevant surfaces:

- `docker/entrypoint.sh` detects `*-mtp`, resolves `draft-mtp`, appends
  speculative flags, and exits if `N_PARALLEL != 1`.
- `k8s/values.yaml` documents `nParallel=1` as required for MTP.
- Upstream `tools/server/server-context.cpp` creates one target context, one
  MTP draft context, initializes `common_speculative_init(..., n_parallel)`,
  collects per-slot draft params, calls `common_speculative_draft`, then
  verifies and accepts per slot.
- Upstream `common/speculative.cpp` holds MTP per-sequence state
  (`pending_h`, `verify_h`, samplers, draft lengths) and uses a shared
  `llama_batch` for MTP draft work.

The important observation from the code review is that the current MTP code
already has some per-sequence state and appears to assemble active slots into
draft batches. Therefore this sprint must not assume the fix is simply "batch
the loop." Phase 1 must identify the exact serialization or acceptance-loss
point before rewriting.

Candidate causes to test:

- The MTP draft context is initialized or scheduled in a way that prevents real
  multi-sequence execution despite `n_parallel>1`.
- The draft loop batches rows, but CPU-side sampling, hidden-state copies, or
  accept/rollback dominates at `np=2/4`.
- Variable draft depths across slots cause underfilled draft batches.
- Target verification works, but per-slot partial acceptance or rollback
  destroys acceptance and erases throughput gains.
- The benchmark path is correct, but instrumentation shows the theoretical MTP
  overhead is too high for `np=4` on this model/card.

### Control Surface

The implementation must be fail-closed:

- Existing MTP profiles keep rejecting `N_PARALLEL>1` unless both
  `PREVIEW=1` and `MTP_MULTISLOT=1` are set.
- Preview compose and Helm examples may set `N_PARALLEL=2` or `4`, but chart
  defaults remain `nParallel: 1`.
- Docs must distinguish "preview multislot MTP" from "recommended production
  B1."

### Correctness Oracle

The primary oracle is target-only greedy output, not B1 output. B1 equivalence
is still useful for regression detection, but target-only output is the
speculative decoding contract.

---

## Implementation

### Phase 0: Baseline Preservation

**Goal**: commit the current matrix before changing code.

Tasks:

- [x] Check whether the reported homelab commits are available upstream. If so,
      pull or cherry-pick the benchmark artifact commit.
- [x] If the commits are not available, encode the reported matrix from the
      operator notes and clearly label provenance.
- [x] Create `docs/benchmarks/qwen-mtp-4090-baseline-2026-05-18.md`.
- [x] Create `docs/sprints/artifacts/SPRINT-007-MTP-MATRIX.json`.
- [x] Record the exact production B1 config, VRAM, acceptance, and
      `A1/A2/A4` vs `B1/B2/B4` numbers.

Phase gate:

- The baseline matrix is committed and B1 remains the active production config.

### Phase 1: Spike and Instrumentation

**Goal**: locate the bottleneck before refactoring.

Tasks:

- [x] Inspect fresh upstream `b9196` plus the RotorQuant patch. Name the exact
      functions and data structures that must change.
- [x] Add gated instrumentation around:
      `common_speculative_draft`, MTP draft `llama_decode`, target verify,
      accept/rollback, hidden-state copy, sampler sample/accept, and batch
      construction.
- [x] Keep instrumentation behind a fast runtime gate such as
      `LLAMA_SPEC_TRACE=1` so normal benchmark timings are not distorted.
- [x] Emit a JSON artifact with at least:
      `active_slots`, `draft_batch_n`, `draft_tokens_requested`,
      `draft_tokens_generated`, `draft_tokens_accepted`, `draft_ms`,
      `verify_ms`, `accept_ms`, `rollback_count`, `slot_id`,
      `prompt_tokens`, `predicted_tokens`, `n_parallel`, and peak VRAM.
- [x] Bound the implementation. If the spike requires touching llama memory
      allocator internals, sampler core, or more than roughly 500 lines outside
      `common/speculative.cpp` and `tools/server/server-context.cpp`, stop and
      write a blocked report before attempting a production patch.

Phase gate:

- A trace report identifies whether the bottleneck is draft decode, target
  verify, accept/rollback, sampler/copy overhead, or acceptance collapse.
  Phase 1 instrumentation is landed; this gate still requires a homelab trace
  run on `gpu-02-4090rtx`.

### Phase 2: Multislot Correctness Harness

**Goal**: make bad speculative state impossible to miss.

Tasks:

- [ ] Extend `scripts/mtp_probe.py` or add a small companion script for
      deterministic concurrent completions.
- [ ] Capture token IDs for target-only greedy references and MTP outputs.
- [ ] Support `np=1`, `np=2`, `np=4`, and sparse occupancy
      (`parallel=4` with only two active requests).
- [ ] Include prompt classes:
      normal prose, code/rejection-heavy, long prompt, uneven prompt lengths,
      and mixed `n_predict` values.
- [ ] Add a slot-churn case: one slot stops or cancels while another continues.
- [ ] If feasible, add a forced-reject debug hook for MTP analogous to the
      DFlash forced-rejection gate. If too large, document it as a follow-up
      and use natural rejection-heavy prompts in this sprint.

Phase gate:

- The harness can fail on token mismatch and report the first divergent slot,
  position, expected token, actual token, acceptance count, and rollback count.

### Phase 3: Upstream `draft-mtp` Multislot Patch

**Goal**: remove the real bottleneck found in Phase 1 while preserving B1.

Tasks:

- [ ] Modify the fresh upstream patch workspace, not the compressed patch by
      hand.
- [ ] Ensure the MTP draft context, batch construction, and per-sequence state
      support active slots without collapsing to single-slot execution.
- [ ] Preserve per-slot hidden-state carryover, sampler state, draft results,
      and rollback bookkeeping keyed by sequence ID, not transient batch row.
- [ ] Handle variable draft depths: slots may request fewer remaining tokens,
      reject earlier, or finish while other slots continue.
- [ ] Keep target verification and accept/rollback independent per slot unless
      instrumentation proves a safe batched verify change is required.
- [ ] Keep `np=1` behavior byte-for-byte compatible with the current B1 path.

Phase gate:

- Local tiny-context smoke starts with `--parallel 2`, `draft-mtp`, nonzero
  accepted drafts, and no token mismatch on a short deterministic prompt.

### Phase 4: Patch Regeneration and Local Validation

**Goal**: make the change reproducible in the repo build.

Tasks:

- [ ] Regenerate `docker/patches/llama-b9196-rotorquant.patch.gz`.
- [ ] Verify the patch applies cleanly to a fresh upstream `b9196` checkout.
- [ ] Build `llama-server` locally where possible.
- [ ] Verify `llama-server --help` still advertises `draft-mtp` and all
      RotorQuant KV cache types.
- [ ] Run `bash -n docker/entrypoint.sh`.
- [ ] Run `python3 -m py_compile scripts/mtp_probe.py`.
- [ ] Run compose config and Helm template checks for B1 and preview profiles.

Phase gate:

- A clean checkout can reproduce the patched server and packaging checks pass.

### Phase 5: Repo Gating and Preview Profiles

**Goal**: expose multislot only as an explicit experiment.

Tasks:

- [ ] Add `MTP_MULTISLOT=1` handling to `docker/entrypoint.sh`.
- [ ] Keep current MTP guard unless `PREVIEW=1` and `MTP_MULTISLOT=1` are both
      set.
- [ ] Add compose and Makefile preview paths for `np=2` and `np=4`.
- [ ] Add Helm values for preview multislot, defaulting off.
- [ ] Update `README.md` and `k8s/README.md` with B1 default and preview
      instructions.

Phase gate:

- Default B1 launch remains unchanged; preview launch requires explicit opt-in.

### Phase 6: Homelab Correctness Matrix

**Goal**: prove correctness on the actual 4090 target before measuring speed.

Tasks:

- [ ] Run target-only greedy references for the prompt set.
- [ ] Run MTP `np=1`, `np=2`, `np=4`, and sparse occupancy.
- [ ] Record token diffs, acceptance, rollback count, peak VRAM, and trace
      summary for each slot.
- [ ] Stop on first correctness failure and write a failure report.

Phase gate:

- 256 generated tokens per slot match target-only greedy output on at least
  three prompts at `np=1`, `np=2`, and `np=4`.

### Phase 7: Homelab Throughput Matrix

**Goal**: decide whether multislot MTP is promotable.

Tasks:

- [ ] Re-run `A1/A2/A4` MTP-off controls.
- [ ] Re-run `B1/B2/B4` MTP-on candidates.
- [ ] Use distinct prompts per concurrent request, `n_predict=1024`, no client
      serialization, and clean GPU state between runs.
- [ ] Capture aggregate t/s, per-slot t/s, acceptance, rollback, trace timing,
      and peak VRAM.
- [ ] Compare against the promotion gates below.

Phase gate:

- A written promotion decision exists in
  `docs/benchmarks/qwen-mtp-multislot-4090-2026-05-xx.md`.

### Phase 8: Outcome and Documentation

**Goal**: close the sprint with an honest operator recommendation.

Tasks:

- [ ] If gates pass, document preview multislot MTP as promotion-ready but keep
      B1 as default pending explicit user approval.
- [ ] If gates fail, document the bottleneck and keep B1 as the only
      recommended MTP production profile.
- [ ] Create `SPRINT-007-mtp-FOLLOWUPS.md` only for execution-discovered
      issues, not for generic future ideas already captured in the deferred doc.
- [ ] Update `docs/benchmarks/` and the README MTP section.

---

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `docker/patches/llama-b9196-rotorquant.patch.gz` | Modify | Carry upstream `draft-mtp` instrumentation and multislot fix |
| `docker/entrypoint.sh` | Modify | Fail-closed preview gate for MTP `N_PARALLEL>1` |
| `docker-compose.yml` | Modify | Add explicit preview surfaces for MTP `np=2/4` |
| `Makefile` | Modify | Add preview run and benchmark helpers |
| `k8s/values.yaml` | Modify | Keep B1 default; add preview multislot values |
| `k8s/templates/deployment.yaml` | Modify | Wire preview environment controls |
| `k8s/README.md` | Modify | Document homelab 4090 preview deployment and gates |
| `README.md` | Modify | Explain B1 default vs preview multislot MTP |
| `scripts/mtp_probe.py` | Modify | Add concurrent correctness, acceptance, and JSON output |
| `scripts/bench_n_parallel.py` | Modify | Generate A/B `np=1/2/4` matrices |
| `docs/benchmarks/qwen-mtp-4090-baseline-2026-05-18.md` | Create | Preserve current matrix |
| `docs/benchmarks/qwen-mtp-multislot-4090-2026-05-xx.md` | Create | Final benchmark and promotion decision |
| `docs/sprints/artifacts/SPRINT-007-MTP-MATRIX.json` | Create | Machine-readable baseline |
| `docs/sprints/artifacts/SPRINT-007-MTP-MULTISLOT.json` | Create | Machine-readable final results |

---

## Definition of Done

### Sprint Completion Gates

- [ ] Baseline matrix is committed before implementation starts.
- [ ] Instrumentation can identify per-phase MTP costs without affecting normal
      runs when disabled.
- [ ] Patch applies cleanly to fresh upstream `b9196` and builds `llama-server`.
- [ ] B1 default launch behavior is unchanged.
- [ ] `np=1` patched MTP matches target-only greedy output and current B1
      throughput within 5%.
- [ ] Correctness harness covers `np=1/2/4`, sparse occupancy, uneven prompt
      lengths, long prompt, rejection-heavy prompt, mixed `n_predict`, and slot
      churn or cancellation.
- [ ] If implementation is blocked, a committed report names the exact blocker,
      traces, and follow-up work; production remains B1.

### Preview Promotion Gates

These gates are required before recommending the preview profile for real
multi-client traffic:

- [ ] `np=2` and `np=4` token outputs match target-only greedy references for
      256 tokens per slot on at least three prompts.
- [ ] Per-slot MTP acceptance at `np=2/4` stays within `+/- 5` percentage
      points of B1 on the same prompt set.
- [ ] `B4` aggregate throughput is at least `1.4 * B1`.
- [ ] `B4` aggregate throughput is at least `0.70 * A4`.
- [ ] `A4` MTP-off control remains within 5% of the preserved 124.5 t/s
      baseline, or the report explains environmental drift.
- [ ] Peak VRAM is recorded for B1/B2/B4 and B4 leaves enough headroom to avoid
      OOM under normal homelab load.
- [ ] Compose and Helm require explicit preview flags for multislot MTP.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Cross-slot hidden-state or rollback corruption | High | High | Token-level target-only oracle, forced/natural rejection cases, per-slot traces |
| Code appears batched already, so the actual bottleneck is deeper | High | High | Phase 1 spike before refactor; instrumentation-only outcome allowed |
| Acceptance collapses at `np=4` | Medium | High | Acceptance gate; compare draft depths and rollback counts |
| 4090 VRAM headroom disappears | Medium | High | Hold KV budget constant, record peak VRAM, fail before production docs change |
| Slot churn/cancellation corrupts batch-row mapping | Medium | High | Explicit slot-churn correctness case |
| Mixed prompt lengths or sparse occupancy expose scheduler bugs | Medium | Medium | Include uneven prompts and `parallel=4` with two active requests |
| Instrumentation perturbs timings | Medium | Medium | Gate behind `LLAMA_SPEC_TRACE`; benchmark with tracing off and summary counters on |
| Upstream `b9196` tag moves or rebase changes MTP internals | Low | Medium | Pin commit SHA in docs and validate patch against fresh checkout |
| Preview flag accidentally becomes production default | Low | High | Fail-closed entrypoint, Helm defaults off, docs callout |

---

## Security

- Multislot batching must preserve cross-request isolation. Hidden states,
  sampled tokens, sampler state, and rollback state must never be addressed by
  transient batch row alone.
- Benchmark artifacts must use public or synthetic prompts only.
- Preview controls are an operations boundary: no default chart or compose path
  should enable multislot MTP implicitly.
- No authentication or public API change is planned.

---

## Dependencies

- Upstream llama.cpp `b9196` and commit
  `7ba22c6a0918b5db16029c2a120bf04a56e78b78`.
- Existing RotorQuant patch workflow in `docker/Dockerfile` and
  `docker/patches/`.
- Homelab access to `gpu-02-4090rtx` for correctness and throughput matrices.
- Existing MTP GGUF from `unsloth/Qwen3.6-27B-MTP-GGUF`.
- Existing `scripts/mtp_probe.py` and `scripts/bench_n_parallel.py`.
- Profile-first guidance in `docs/INFERENCE_LESSONS.md` and
  `docs/THROUGHPUT_CONFIGURATION_MODEL.md`.

---

## Open Questions

1. If `np=2` clears all gates but `np=4` does not, should `np=2` be considered
   useful enough for preview deployment?
2. Should the first preview preserve full `ctx=196608` total context, or use the
   lower committed `131072` compose default until VRAM headroom is proven?
3. If instrumentation proves the draft path cannot scale enough, should the
   next sprint be the hybrid router fallback or a deeper upstream MTP refactor?
4. If the patch works, should it be upstreamed to llama.cpp immediately or kept
   in the RotorQuant patch until another rebase cycle confirms stability?
