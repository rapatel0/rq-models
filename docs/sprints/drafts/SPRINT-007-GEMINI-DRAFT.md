# Sprint 007-mtp: Multislot MTP Draft Inference

**Date**: 2026-05-18
**Hardware**: NVIDIA RTX 4090 (24 GB)
**Model targets**: Qwen3.6-27B (dense, Q4_0, MTP draft-max=4)
**Baseline**: B1 (MTP on, n_parallel=1) vs A4 (MTP off, n_parallel=4)
**Status**: draft (GEMINI-DRAFT)

---

## Overview

Sprint 007 implements true multislot MTP (Multi-Token Prediction) support in the `rq-models` stack by optimizing the upstream `llama.cpp` `draft-mtp` implementation. While existing deployments support MTP for single-slot inference, the current production environment forces `N_PARALLEL=1` for MTP models due to scaling bottlenecks (B1 @ 68.1 t/s vs B4 @ 77.2 t/s, while MTP-off A4 reaches 124.5 t/s).

This sprint moves beyond the short-term hybrid router and targets the core multi-slot scheduling and batched-draft efficiency within `llama-server`. The goal is to unlock `N_PARALLEL=2` and `N_PARALLEL=4` for MTP-enabled profiles, providing a high-throughput path for the dense 27B model on the RTX 4090 while maintaining the speculative decoding gains of MTP.

Production default remains B1 (single-slot MTP) until aggregate throughput and correctness gates are cleared at `np=4`.

## Use Cases

1. **High-Throughput Speculative Serving**: Users serving multiple concurrent requests on an RTX 4090 gain the benefits of MTP's lower per-token latency combined with batched throughput.
2. **Resource-Efficient Scaling**: Moving from `np=1` to `np=4` with MTP allows the platform to handle 4x the request volume on the same VRAM footprint, provided the acceptance and compute scaling bottlenecks are resolved.
3. **Draft-Token Optimization**: Improvements to the batched-draft interface in `llama-server` lay the groundwork for future DFlash and EAGLE3 speculative decoding integrations.

## Architecture

### Substrate layout

Implementation involves patching the pinned `llama.cpp` version (`b9196`) within the `docker/` build context.

```
rq-models/
├── docker/
│   ├── Dockerfile                           ← Patches llama.cpp b9196
│   ├── entrypoint.sh                        ← Runtime flag resolution for MTP + N_PARALLEL
│   └── patches/
│       └── llama-b9196-rotorquant.patch.gz  ← Updated to include multislot MTP fixes
├── scripts/
│   ├── mtp_probe.py                         ← EXTEND: multi-slot acceptance + throughput
│   └── bench_n_parallel.py                  ← REUSE: for A/B matrix collection
```

### Technical Approach

1. **Batch Draft Optimization**: Audit and refactor `common/speculative.cpp` to ensure the `ctx_dft` (draft context) effectively batches active slots during each draft step. Current sub-linearity suggests the draft-token generation may be serializing or redundantly re-processing state.
2. **MTP Hidden State Carryover**: Verify the integrity of `pending_h` and `verify_h` in `common_speculative_state_draft_mtp` across concurrent slots.
3. **Server-Side Scheduling**: Optimize `tools/server/server-context.cpp` to ensure that `common_speculative_draft` and `common_speculative_verify` interactions with the server's slot management do not introduce unnecessary sync points.

## Implementation

### Phase 0: Reproduce & Benchmark (Target: 1-2 days)

1. **Artifact Preservation**: Run `scripts/bench_n_parallel.py` on the homelab 4090 (`gpu-02`) to capture the current A1/A2/A4 (MTP-off) and B1/B2/B4 (MTP-on) baseline matrix.
2. **Result Export**: Document findings in `docs/benchmarks/results/SPRINT-007-BASELINE.md`.
3. **mtp_probe Extension**: Update `scripts/mtp_probe.py` to support concurrent requests and per-slot acceptance metrics.

### Phase 1: Instrumentation & Diagnosis (Target: 2-3 days)

1. **Timer Logging**: Inject high-resolution timers into `common/speculative.cpp` to measure:
    - Draft time (per step and total).
    - Target verification time.
    - Slot-to-slot sync/blocking overhead.
2. **Bottleneck Identification**: Analyze logs to determine if the 77.2 t/s cap at `np=4` is due to draft-compute serialization, KV-cache contention, or CPU-side scheduling delays in the server.

### Phase 2: Implementation (Target: 1 week)

1. **Patch Development**: Apply fixes to a fresh `llama.cpp` checkout.
    - Ensure `llama_batch` in the draft context is correctly aggregated for all active sequences.
    - Optimize the loop in `common_speculative_draft` for parallel execution where possible.
    - Resolve any per-slot state leaks or redundant context switching.
2. **Patch Regeneration**: Update `docker/patches/llama-b9196-rotorquant.patch.gz` with the new logic.
3. **Deployment Flags**: Update `docker/entrypoint.sh` and `k8s/values.yaml` to allow `N_PARALLEL > 1` for MTP profiles behind the `--experimental-multislot-mtp` flag.

### Phase 3: Correctness & Performance Gates (Target: 3 days)

1. **Correctness Gate**: Run the deterministic token diff harness. Concurrent slots must match target-only greedy output for 256+ tokens.
2. **Regression Gate**: Verify B1 (single-user) stays within 5% of current baseline.
3. **Throughput Gate**: `np=4` MTP-on must beat B1 by ≥ 1.4x (target > 95 t/s) and retain ≥ 70% of A4 MTP-off aggregate.

## Files Summary

- `docs/sprints/drafts/SPRINT-007-GEMINI-DRAFT.md`: This document.
- `docker/patches/llama-b9196-rotorquant.patch.gz`: Updated with MTP multislot fixes.
- `docker/entrypoint.sh`: Updated for `N_PARALLEL` logic.
- `scripts/mtp_probe.py`: Extended for multi-slot metrics.
- `docs/benchmarks/results/SPRINT-007-RESULTS.md`: Final performance matrix.

## Definition of Done

- [ ] Current performance matrix (A1-A4, B1-B4) reproduced and documented in `docs/benchmarks/`.
- [ ] `llama-server` built with updated patch successfully handles `N_PARALLEL=4` with `draft-mtp`.
- [ ] **Correctness**: MTP-on multi-slot matches MTP-off target-only output across 3 distinct prompts (greedy).
- [ ] **Throughput**: `np=4` MTP-on aggregate throughput ≥ 95 t/s on RTX 4090.
- [ ] **Acceptance**: Per-slot MTP acceptance at `np=4` within ±5% of B1 baseline.
- [ ] `docker/entrypoint.sh` allows multi-slot MTP only when experimental flag is set.
- [ ] B1 remains the default production profile in `k8s/values.yaml`.

## Risks

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Acceptance Collapse** | High | Multi-slot bugs might degrade acceptance rates. Monitor via `mtp_probe.py`. |
| **VRAM Overflow** | High | Multi-slot MTP increases KV-cache and hidden-state overhead. Maintain strict budget (< 23GB on 4090). |
| **Upstream Drift** | Medium | Patching `b9196` may conflict with other rotorquant changes. Use a clean rebase workflow. |
| **Sub-linear Compute Scaling** | Medium | If the bottleneck is purely the draft-model's serial nature, the sprint may only deliver instrumentation. |

## Security

- No changes to authentication or external API surface.
- Patches are applied to the pinned upstream commit to maintain software supply chain integrity.

## Dependencies

- **Upstream**: `llama.cpp` tag `b9196`.
- **Hardware**: Access to `gpu-02` (RTX 4090) for matrix collection.
- **Reference**: Existing RotorQuant patch for KV-cache compatibility.

## Open Questions

1. Does the current sub-linearity originate in the server's request handling or the `common/speculative.cpp` implementation?
2. Can we achieve >1.4x scaling without significant changes to how `ctx_dft` handles multiple sequences?
3. Should we implement a fallback to single-slot MTP if VRAM headroom falls below a certain threshold when `N_PARALLEL > 1`?
