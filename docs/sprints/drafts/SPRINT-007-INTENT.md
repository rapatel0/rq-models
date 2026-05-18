# Sprint 007 Intent: Multislot MTP Draft Inference

**Seed prompt**: `$sprint-plan changes for 2`

**Interpretation**: plan the second path from the prior discussion: a true
multislot MTP implementation, not the short-term hybrid router that switches
between MTP-on single-slot and MTP-off multi-slot deployments.

**Track**: MTP-specific speculative-decoding track. Existing roadmap convention
used the `-dflash` suffix for speculative work, but this sprint should use
`SPRINT-007-mtp.md` so the artifact name matches the work.

## Orientation Summary

- Current `main` is clean at `53ae72f` and does not contain the reported
  homelab agent commits (`5a8d51e`, `16ec054`, `e930e56`, `dfcbd8f`,
  `f70afac`). Sprint Phase 0 must either pull those commits if pushed or
  preserve the reported MTP matrix in a benchmark artifact before touching
  implementation.
- The decisive matrix from the user shows the target model batches well when
  MTP is off: `np=1/2/4 = 39.7/71.5/124.5 t/s`, while MTP-on barely scales:
  `np=1/2/4 = 68.1/73.1/77.2 t/s`. This proves dense 27B Q4 on the 4090 is
  not generally unable to batch; the bottleneck is MTP + multi-slot scheduling.
- Repo config intentionally blocks MTP multi-slot today:
  `docker/entrypoint.sh` forces `N_PARALLEL=1` for `*-mtp` models and
  `k8s/values.yaml` documents that MTP currently requires `nParallel=1`.
- Upstream pinned llama.cpp is `b9196` with RotorQuant patching layered on top.
  Relevant upstream files are `common/speculative.cpp` and
  `tools/server/server-context.cpp`, especially `common_speculative_state_draft_mtp`,
  `common_speculative_draft`, slot draft parameter collection, and accept/rollback.
- Existing roadmap already sketches Sprint 007 as multi-slot batched-draft
  inference for DFlash. This sprint should reuse the structure but specialize
  gates and implementation around upstream `draft-mtp` for Qwen3.6 27B MTP on
  the RTX 4090.

## Relevant Codebase Areas

- `docker/Dockerfile`: pins upstream `ggml-org/llama.cpp` via `LLAMA_CPP_REF`
  and applies `docker/patches/llama-b9196-rotorquant.patch.gz`.
- `docker/entrypoint.sh`: detects `*-mtp`, resolves `draft-mtp`, appends
  `--spec-type`, `--spec-draft-n-max`, `--spec-draft-p-min`, `--no-warmup`,
  and currently exits when `N_PARALLEL != 1`.
- `docker-compose.yml`, `Makefile`, `k8s/values.yaml`,
  `k8s/templates/deployment.yaml`, `k8s/README.md`: deployment/config surface
  that must keep production B1 as the default until multi-slot MTP clears gates.
- `scripts/mtp_probe.py`: single-request MTP acceptance probe; likely needs
  multi-slot acceptance, aggregate throughput, per-slot throughput, and
  correctness extensions.
- `scripts/bench_n_parallel.py`: existing parallel benchmark harness to reuse
  or supersede for A/B matrix collection.
- Upstream patch contents from `docker/patches/llama-b9196-rotorquant.patch.gz`:
  implementation changes must be made against a fresh upstream checkout and
  regenerated into the compressed patch, not manually edited only in `/tmp`.
- Upstream `common/speculative.cpp`:
  `common_speculative_state_draft_mtp` holds per-sequence `pending_h`,
  `verify_h`, samplers, and a shared MTP draft `llama_batch`; `draft()`
  batches active slots for each draft step but still shows poor scaling in
  production.
- Upstream `tools/server/server-context.cpp`:
  server creates one `ctx_dft`, initializes `common_speculative_init(...,
  n_parallel)`, collects per-slot draft params, calls
  `common_speculative_draft`, then verifies/accepts per slot.

## Constraints

- Default production behavior must remain B1 until the sprint proves a better
  profile: `MTP on`, `n_parallel=1`, `ctx=196608`, `draft-mtp`, draft max 4,
  `p_min=0.75`, `ubatch=32`, `q4_0` KV.
- Any implementation must preserve upstream rebasing discipline: keep official
  `draft-mtp`; do not revive the old fork-specific `--spec-type mtp` except
  for runtime compatibility.
- RTX 4090 target has tight VRAM: current B1 is ~21.4/23 GB with ~1.2 GB free.
  Multi-slot work must explicitly budget KV and avoid accidental context
  inflation.
- Correctness matters more than throughput: speculative decode must match
  target-only greedy output under forced accept/reject patterns and across
  multiple slots.
- This is a llama.cpp/server patch plus repo packaging task. The Docker patch
  must apply cleanly to the pinned upstream tag and build `llama-server`.

## Success Criteria

- `N_PARALLEL=2` and `N_PARALLEL=4` are allowed for MTP profiles only behind an
  explicit experimental flag until gates pass.
- Greedy deterministic correctness: MTP-on multi-slot matches MTP-off
  target-only for 256 generated tokens per slot on at least three distinct
  prompts at `np=1`, `np=2`, and `np=4`.
- Per-slot MTP acceptance at `np=2/4` stays within ±5 percentage points of B1
  on the same prompt set, or the sprint documents the acceptance collapse and
  stops before production rollout.
- Throughput gate: `np=4` MTP-on aggregate must beat B1 by at least 1.4x and
  must retain at least 70% of A4 MTP-off aggregate. If not met, ship
  instrumentation and keep B1 default.
- Regression gate: B1 single-user decode must remain within 5% of the current
  baseline, and MTP-off A4 must remain within 5% of 124.5 t/s aggregate on the
  same homelab 4090.
- Deployment gate: Helm and compose expose a safe experimental profile and keep
  production values pinned to B1 unless the user explicitly flips profiles.

## Verification Strategy

- Phase 0: reproduce or import the reported matrix into
  `docs/benchmarks/results/` plus an explanatory Markdown report.
- Static checks: patch applies to fresh upstream `b9196`; `llama-server --help`
  advertises `draft-mtp` and RotorQuant KV types; shell/Python syntax checks.
- Local smoke if model is cached: start `llama-server` with small context,
  `--parallel 2`, `draft-mtp`, draft max 2/4, and verify nonzero accepted
  drafts on two concurrent requests.
- Homelab matrix on `gpu-02-4090rtx`: A1/A2/A4 MTP-off controls and B1/B2/B4
  MTP-on variants, same prompt set, `n_predict=1024`, distinct prompts per slot.
- Correctness harness: deterministic token diff for concurrent slots; include
  rejection-heavy prompts and at least one long prompt.
- Metrics: record draft time, target verify time, accept/rollback time, per-slot
  accepted/total drafts, batch sizes passed to `ctx_dft`, and aggregate/per-slot
  decode t/s.

## Uncertainty Assessment

- Correctness: High. MTP uses target hidden-state carryover, draft context
  state, and accept/reject rollback; subtle multi-slot bugs can still produce
  plausible text.
- Scope: High. The current data proves a bottleneck but not the exact cause;
  the sprint may end at instrumentation if upstream `draft-mtp` needs a deeper
  refactor.
- Architecture: Medium-high. The current MTP implementation already tracks
  per-sequence state, so this may be a scheduling/measurement fix; however
  improving actual scaling may require modifying common speculative internals.

## Open Questions

1. Should this sprint optimize only upstream `draft-mtp`, or should it generalize
   the batched-draft interface for DFlash/EAGLE3 as well?
2. Is `np=2` enough for production, or is the hard target `np=4` because A4
   already shows the desirable aggregate gain?
3. Are we willing to accept a profile that improves aggregate throughput but
   reduces per-user decode below B1, or must multi-slot MTP beat both B1
   per-user and A4 aggregate?
4. Should the sprint first implement a load-aware fallback that disables MTP
   when active slots > 1, as an operator safety net, even though the requested
   primary goal is true multislot MTP?
5. Can the implementation agent use the homelab 4090 for iterative C++/CUDA
   benchmarking, or should local laptop validation remain limited to build and
   tiny-context smoke tests?

## Prior Deferred / Follow-Up Items

- `docs/sprints/SPRINT-ROADMAP-dflash.md` Sprint 007 is directly actionable:
  multi-slot batched-draft inference was sketched as the next optimization once
  multi-slot sub-linearity was characterized.
- `docs/sprints/SPRINT-005-dflash.md` F-010 characterization is now effectively
  satisfied by the user-provided MTP matrix, though the artifact is not present
  in this checkout. Preserve that result before implementation.
- Older deferred items from Sprints 001-003 are mostly RotorQuant, Web UI,
  SpectralQuant, CI, or multi-GPU work and should stay out of this sprint.

## Vision Context

No `docs/sprints/VISION.md` exists. Planning proceeds from the existing
speculative-decoding roadmap and the current MTP production benchmark.
