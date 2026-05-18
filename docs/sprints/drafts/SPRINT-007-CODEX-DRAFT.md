# Sprint 007-mtp: True Multislot MTP for Upstream `draft-mtp`

**Status**: Planning draft
**Created**: 2026-05-18
**Depends on**: current `main` at upstream llama.cpp `b9196`, the existing RotorQuant patch flow, and preservation of the reported 4090 MTP matrix before implementation starts
**Target hardware**: `gpu-02-4090rtx`, 24 GB VRAM
**Production baseline to preserve**: `B1` = Qwen3.6-27B MTP, `draft-mtp`, `n_parallel=1`, `ctx=196608`, `draft_n_max=4`, `draft_p_min=0.75`, `ubatch=32`, `q4_0` KV

## Overview

Sprint 007 targets the real bottleneck in the current Qwen3.6-27B MTP path:
upstream `draft-mtp` is functional for one slot, but aggregate throughput barely
improves when the server is asked to batch multiple concurrent users. The user
reported matrix is decisive: MTP-off scales from `39.7 -> 71.5 -> 124.5 tok/s`
at `np=1/2/4`, while MTP-on only moves from `68.1 -> 73.1 -> 77.2 tok/s`. The
model and GPU can batch; the draft path is what does not scale.

This sprint therefore does not build a routing workaround that disables MTP
above one slot. It implements true multislot speculative decode for the pinned
upstream llama.cpp `draft-mtp` path, measures whether batching actually holds
up on the 4090, and keeps all production surfaces pinned to `B1` until the
experimental path clears hard correctness and throughput gates.

The sprint is profile-first. Phase 0 preserves the current matrix in committed
artifacts and adds the instrumentation needed to explain where draft batching is
stalling. If the fork-side change produces correctness but not enough scaling,
the sprint still ships the instrumentation, benchmark artifacts, and safe
preview path, but `B1` remains the default.

Out of scope: DFlash or EAGLE3 generalization, multi-GPU work, speculative
mode routing as the primary solution, and any production default flip before
the multislot MTP gates pass.

## Use Cases

1. **One-GPU multi-user serving with MTP still enabled**: an operator running
   the 27B dense MTP profile on a 24 GB 4090 can serve `np=2` or `np=4`
   concurrent users without collapsing to near-single-slot aggregate throughput.

2. **Safe production posture while optimizing**: compose and Helm keep the
   known-good `B1` profile as the default, while an explicit preview flag or
   preview profile unlocks multislot MTP for validation and operator testing.

3. **Correctness before speed**: reviewers can prove that multislot MTP emits
   the same greedy output as target-only decode across multiple simultaneous
   sessions, including reject-heavy prompts and long prompts.

4. **Actionable benchmark evidence**: operators get an A/B matrix for
   `A1/A2/A4` (MTP off) and `B1/B2/B4` (MTP on), plus per-slot acceptance and
   per-phase timings, so a production promotion decision is based on evidence
   rather than startup banners.

## Architecture

The implementation keeps the current high-level serving shape:

- one target model context for verification and final token commitment
- one draft model context for MTP drafting
- one `common_speculative` state object that tracks per-sequence hidden state,
  pending tokens, verify state, sampler state, and accept/rollback bookkeeping

The change is in how active slots are assembled for draft work and how their
results are mapped back:

```text
Current path
slot 0 draft step -> shared ctx_dft
slot 1 draft step -> shared ctx_dft
slot 2 draft step -> shared ctx_dft
slot 3 draft step -> shared ctx_dft
then per-slot verify/accept

Sprint 007 target path
collect active slots -> build one batched draft pass on ctx_dft
slot->batch-row map -> draft up to N tokens per active slot
scatter drafted candidates back into per-slot speculative state
then keep per-slot verify/accept and rollback logic independent
```

### Architectural invariants

- The pinned upstream `draft-mtp` path remains the only speculative mode in
  scope. No fork-specific `--spec-type mtp` revival except existing runtime
  compatibility detection.
- `B1` behavior is unchanged when the experimental multislot flag is absent.
  The current entrypoint guard remains in force by default.
- The first multislot experiments keep the total KV budget constant relative to
  `B1`; they do not silently buy throughput by inflating VRAM or shrinking the
  verification problem.
- Slot isolation is preserved. Draft buffers, hidden-state carryover, samplers,
  and rollback state must stay keyed by slot/sequence, not by transient batch
  row order.

### Planned control surface

1. `docker/entrypoint.sh` continues to reject `N_PARALLEL>1` for `*-mtp`
   unless a new explicit preview flag is set.
2. Compose and Helm expose that preview flag separately from the production
   `B1` profile so operators opt in on purpose.
3. `scripts/mtp_probe.py` and `scripts/bench_n_parallel.py` become the
   correctness and performance harnesses for the new path.
4. The upstream patch remains the source of truth. Changes happen against a
   fresh `b9196` checkout and are re-generated into
   `docker/patches/llama-b9196-rotorquant.patch.gz`.

## Implementation

### Phase 0: Preserve the baseline and add visibility

**Goal**: do not start optimizing from an uncommitted anecdote.

**Files**:
- `docs/benchmarks/qwen-mtp-4090-baseline-2026-05-xx.md`
- `docs/sprints/artifacts/SPRINT-007-MTP-MATRIX.json`
- `scripts/bench_n_parallel.py`
- `scripts/mtp_probe.py`

**Tasks**:
- [ ] Preserve the reported `A1/A2/A4` and `B1/B2/B4` matrix in committed
      artifacts before touching the patch. If the missing homelab commits are
      available, import their raw output; otherwise encode the matrix from the
      operator notes and clearly label provenance.
- [ ] Extend `scripts/bench_n_parallel.py` so it can emit the exact matrix
      shape needed for this sprint: aggregate tok/s, per-slot tok/s, wallclock,
      health snapshots, and repeated runs for `np=1/2/4`.
- [ ] Extend `scripts/mtp_probe.py` beyond a single request. It should support
      concurrent slot probes, deterministic prompt sets, per-slot acceptance,
      and JSON output for later comparison.
- [ ] Add timing and counter hooks around draft, verify, and accept/rollback
      phases so the sprint can distinguish "draft path is serialized" from
      "draft path is batched but rollback is dominating."

**Phase gate**: the baseline artifact is committed, reproducible enough to
benchmark against, and the harness can report draft acceptance per slot.

### Phase 1: Guard correctness before changing scheduling

**Goal**: make multislot bugs easy to catch and hard to rationalize away.

**Files**:
- `docker/patches/llama-b9196-rotorquant.patch.gz`
- `scripts/mtp_probe.py`
- `docs/benchmarks/qwen-mtp-multislot-correctness-2026-05-xx.md`

**Tasks**:
- [ ] Add deterministic multislot validation to the patch or harness so each
      slot can be compared against a target-only reference at the token level.
- [ ] Build a prompt set with at least three prompt classes:
      normal prose, reject-heavy/code-style output, and one long prompt that
      exercises larger KV and more rollback surface.
- [ ] Verify greedy equivalence at `np=1`, `np=2`, and `np=4` for 256 decode
      tokens per slot. A failed diff blocks the batching rewrite from rolling
      into repo-side preview profiles.
- [ ] Capture per-slot accept/reject counters and rollback counts for every
      validation run so correctness regressions can be tied back to state
      transitions rather than only final text mismatches.

**Phase gate**: multislot validation can fail loudly on token divergence, and
the current `B1` path still passes the single-slot reference checks.

### Phase 2: Patch upstream `draft-mtp` for true multislot batching

**Goal**: replace per-slot draft serialization with one batched draft pass over
all active slots.

**Files**:
- `docker/patches/llama-b9196-rotorquant.patch.gz`

**Patch areas inside the upstream tree**:
- `common/speculative.cpp`
- `tools/server/server-context.cpp`
- any nearby helper or state definitions required to carry batch-row metadata

**Tasks**:
- [ ] Audit the current `common_speculative_state_draft_mtp` flow and document
      where active slots are still effectively serialized despite the shared
      draft batch object.
- [ ] Introduce an explicit slot-to-batch-row map for the draft step. The map
      must survive accept, partial accept, and reject paths without crossing
      sampler or hidden-state ownership between slots.
- [ ] Rework the draft loop so each speculative step assembles all active slots
      into one `ctx_dft` pass up to `draft_n_max`, then scatters results back
      into per-slot speculative state for verification.
- [ ] Preserve the existing single-slot path semantics. `np=1` must remain
      byte-for-byte identical to the current `B1` logic apart from added
      counters and traces.
- [ ] Keep the patch rebase-friendly: no repo-local edits against `/tmp`, no
      silent drift from the pinned upstream checkout, and no expansion of scope
      into unrelated speculative modes.

**Phase gate**: fresh upstream `b9196` accepts the regenerated patch, builds
`llama-server`, and the patched server can produce accepted drafts at `np=2`.

### Phase 3: Expose a safe preview path in rq-models

**Goal**: let operators test multislot MTP without changing the default.

**Files**:
- `docker/entrypoint.sh`
- `docker-compose.yml`
- `Makefile`
- `k8s/values.yaml`
- `k8s/templates/deployment.yaml`
- `k8s/README.md`
- `README.md`

**Tasks**:
- [ ] Add a dedicated explicit flag for multislot MTP preview, for example
      `MTP_MULTISLOT_EXPERIMENT=1`. Without it, `*-mtp` profiles still reject
      `N_PARALLEL>1`.
- [ ] Keep the existing `qwen36-27b-mtp-speed` profile as the production
      default. Add a preview compose profile or documented preview override for
      `np=2/4` testing instead of mutating the default profile in place.
- [ ] Add the same preview control to Helm values and deployment templates.
      Default values remain `nParallel=1` and preview disabled.
- [ ] Document the constant-total-KV-budget rule for first-pass experiments so
      reviewers know whether a gain came from better batching or from a hidden
      memory trade.
- [ ] Update the operator docs to make the promotion policy explicit: preview
      path is for validation only until Sprint 007 hard gates clear.

**Phase gate**: compose and Helm can launch an experimental multislot MTP
server, while the out-of-the-box production config remains `B1`.

### Phase 4: Benchmark, decide, and publish

**Goal**: decide whether multislot MTP deserves promotion, not merely whether
it boots.

**Files**:
- `docs/benchmarks/qwen-mtp-multislot-4090-2026-05-xx.md`
- `docs/sprints/artifacts/SPRINT-007-MTP-MULTISLOT.json`
- `README.md`
- `k8s/README.md`

**Tasks**:
- [ ] Run the canonical `A1/A2/A4` and `B1/B2/B4` matrix on the homelab 4090
      with the same prompt set and seed policy used in Phase 0.
- [ ] Record aggregate tok/s, per-slot tok/s, acceptance, rollback counts,
      batch sizes reaching `ctx_dft`, and any VRAM headroom changes.
- [ ] Compare `B4` against both promotion thresholds:
      `B4 >= 1.4 * B1` and `B4 >= 0.70 * A4`.
- [ ] If the hard gates pass, update docs to describe the preview path as
      promotion-ready. If they do not pass, publish the failure analysis and
      keep production docs pointed at `B1`.

**Phase gate**: a documented promotion decision exists, backed by committed
artifacts and reproducible commands.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `docker/patches/llama-b9196-rotorquant.patch.gz` | Modify | Carry the upstream `draft-mtp` multislot batching and any required counters |
| `docker/entrypoint.sh` | Modify | Keep the `B1` guard by default; allow multislot MTP only behind an explicit preview flag |
| `docker-compose.yml` | Modify | Add preview launch surface for `np=2/4` without changing the default MTP speed profile |
| `Makefile` | Modify | Add or update preview run targets and benchmark helpers |
| `k8s/values.yaml` | Modify | Keep production defaults pinned to `B1`; add preview controls for multislot validation |
| `k8s/templates/deployment.yaml` | Modify | Wire preview controls into the container environment |
| `k8s/README.md` | Modify | Document safe preview deployment and promotion criteria |
| `README.md` | Modify | State that MTP multislot is experimental until gates pass; preserve `B1` as the recommended path |
| `scripts/mtp_probe.py` | Modify | Add concurrent-slot correctness and acceptance checks |
| `scripts/bench_n_parallel.py` | Modify | Generate `A1/A2/A4` and `B1/B2/B4` benchmark artifacts |
| `docs/benchmarks/qwen-mtp-4090-baseline-2026-05-xx.md` | Create | Human-readable preserved baseline matrix |
| `docs/benchmarks/qwen-mtp-multislot-correctness-2026-05-xx.md` | Create | Token-level multislot correctness report |
| `docs/benchmarks/qwen-mtp-multislot-4090-2026-05-xx.md` | Create | Final benchmark analysis and promotion recommendation |
| `docs/sprints/artifacts/SPRINT-007-MTP-MATRIX.json` | Create | Machine-readable baseline matrix |
| `docs/sprints/artifacts/SPRINT-007-MTP-MULTISLOT.json` | Create | Machine-readable multislot results with timings and counters |

## Definition of Done

- [ ] The current 4090 `A1/A2/A4` and `B1/B2/B4` matrix is preserved in
      committed benchmark artifacts before optimization starts.
- [ ] The upstream patch applies cleanly to a fresh llama.cpp `b9196` checkout
      and builds `llama-server`.
- [ ] `np=1` multislot-capable code remains behaviorally identical to the
      current `B1` path for greedy decode.
- [ ] Greedy deterministic correctness passes for 256 generated tokens per slot
      on at least three prompts at `np=1`, `np=2`, and `np=4`.
- [ ] Per-slot draft acceptance at `np=2` and `np=4` stays within
      `+/- 5` percentage points of `B1`, or the sprint publishes a failure
      report and keeps `B1` as the default.
- [ ] `B4` aggregate throughput beats `1.4 * B1` and retains at least
      `70%` of `A4`, or the sprint publishes a failure report and keeps `B1`
      as the default.
- [ ] `B1` single-user throughput remains within `5%` of the baseline captured
      in Phase 0.
- [ ] Compose and Helm require an explicit preview control before they allow
      multislot MTP.
- [ ] README and Kubernetes docs clearly distinguish preview multislot MTP from
      the production default.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cross-slot rollback or hidden-state corruption produces plausible but wrong text | High | High | Block promotion on token-level greedy equivalence at `np=1/2/4`; capture rollback counters and slot-level diffs |
| Draft batching improves code structure but not throughput | High | High | Instrument per-phase timings first; ship artifacts and keep `B1` default if `B4` misses the gate |
| 24 GB VRAM headroom disappears at `np=4` with production-like context | Medium | High | Hold total KV budget constant in preview runs, record VRAM headroom, and avoid context inflation during benchmarking |
| The needed fix is deeper than `server-context.cpp` plus `common/speculative.cpp` | Medium | High | Keep scope limited to instrumentation plus the smallest viable batching rewrite; do not promise production rollout up front |
| Upstream speculative code moves during or right after the sprint | Medium | Medium | Patch against the pinned `b9196` base first; revisit rebase only after the sprint proves the approach |
| Preview controls drift into the default path by accident | Low | High | Require an explicit preview flag in entrypoint, compose, and Helm; keep `nParallel=1` defaults unchanged |

## Security

- The main security concern is cross-request state isolation. Multislot draft
  batching must not leak hidden states, sampler history, or drafted tokens
  between concurrent sessions when batch-row order changes.
- Preview gating is a security and operations boundary. Default deployments
  must not enable the experimental path implicitly through environment
  inheritance or chart defaults.
- Benchmark artifacts should use public prompts and synthetic workloads only.
  Do not capture production prompts or completions in committed artifacts.
- No new public API surface is required. This sprint changes server internals,
  deployment gating, and observability, not authentication or request schema.

## Dependencies

- Upstream llama.cpp stable tag `b9196` and the existing RotorQuant patch
  regeneration workflow in `docker/patches/`.
- Existing rq-models MTP serving surface in `docker/entrypoint.sh`,
  `docker-compose.yml`, and the Helm chart.
- `scripts/mtp_probe.py` and `scripts/bench_n_parallel.py` as the base
  validation and benchmarking harnesses.
- Homelab access to `gpu-02-4090rtx` for the real `A/B` matrix; local laptop
  validation is only enough for build and tiny-context smoke tests.
- The repo's profile-first conventions in `docs/INFERENCE_LESSONS.md` and
  `docs/THROUGHPUT_CONFIGURATION_MODEL.md`.

## Open Questions

1. Should Sprint 007 target only upstream `draft-mtp`, or should the batching
   abstraction be designed for later reuse by DFlash and EAGLE3 even if this
   sprint only wires MTP?
2. Is `np=2` enough to count as a production-worthy result if `np=4` remains
   below the throughput gate, or is `np=4` the real target because `A4` is the
   proven non-MTP batching ceiling on this card?
3. Should the preview path preserve the full `196608` total context budget from
   the deployed `B1` profile, or should the first experiments use the committed
   `131072` compose/Kubernetes defaults to reduce VRAM risk and simplify A/B?
4. Is a dedicated `MTP_MULTISLOT_EXPERIMENT` flag preferable to a generic
   `PREVIEW=1` convention that does not yet exist in repo code?
5. If multislot correctness is solid but acceptance collapses at `np=4`, is the
   right follow-up another llama.cpp scheduling sprint or an operator-level
   routing fallback in a later sprint?
