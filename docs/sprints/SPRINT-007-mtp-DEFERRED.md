# Sprint 007-mtp Deferred Items

Items discussed in the intent, drafts, or critiques but intentionally excluded
from the final Sprint 007-mtp scope.

---

## D-001: Hybrid Load-Aware Router

**What**: Route solo traffic to MTP-on B1 and concurrent traffic to an MTP-off
multi-slot pool such as A4. This could be a thin proxy, two Kubernetes
deployments behind a load-aware service, or a server-side policy that disables
speculation when active slots exceed one.

**Why deferred**: The user specifically selected path 2: true multislot MTP.
The router is a practical fallback, but making it the main sprint would avoid
the requested upstream scheduling work.

**Target sprint**: Sprint 008 or immediately after Sprint 007 if true multislot
MTP misses promotion gates.

**Prerequisites**: Sprint 007 instrumentation and final benchmark report.

**Files**: `k8s/`, `docker-compose.yml`, possible new proxy/router script.

---

## D-002: Generalized Batched-Draft Interface for DFlash and EAGLE3

**What**: Turn the MTP-specific multislot work into a generic batched draft
interface shared by `draft-mtp`, DFlash, and EAGLE3 speculative modes.

**Why deferred**: Generalization increases scope and risk. Sprint 007 should
prove the MTP path first and only reuse abstractions if they fall out naturally.

**Target sprint**: Future speculative-decoding sprint.

**Prerequisites**: Sprint 007 MTP implementation or instrumentation report;
clear evidence that DFlash/EAGLE3 have the same scheduler bottleneck.

**Files**: upstream `common/speculative.cpp`, `tools/server/server-context.cpp`,
future DFlash/EAGLE3 fork files.

---

## D-003: Streaming SSE Validation Under Multislot MTP

**What**: Validate `stream: true` behavior when MTP drafts multiple tokens and
different slots accept different draft lengths.

**Why deferred**: Sprint 007 focuses on greedy correctness and aggregate decode
throughput. Streaming ordering and partial-token boundary behavior deserve their
own focused validation after non-streaming correctness passes.

**Target sprint**: Future production-hardening sprint.

**Prerequisites**: Multislot MTP correctness gates pass for non-streaming
requests.

**Files**: server streaming path in upstream llama.cpp, `scripts/mtp_probe.py`,
server API tests.

---

## D-004: Non-Greedy / Stochastic Sampling Validation

**What**: Validate multislot MTP under temperature, top-p, top-k, stop strings,
and differing per-slot sampling parameters.

**Why deferred**: The primary correctness oracle is greedy target-only token
matching. Stochastic validation needs distribution-level tests and is too broad
for the first multislot MTP sprint.

**Target sprint**: Future validation sprint after greedy correctness is stable.

**Prerequisites**: Sprint 007 greedy correctness and preview gating.

**Files**: `scripts/mtp_probe.py`, new stochastic validation harness, server
tests.

---

## D-005: Upstream llama.cpp PR

**What**: Prepare the MTP multislot patch for upstream submission to
`ggml-org/llama.cpp`.

**Why deferred**: The patch must first prove correctness and performance in the
RotorQuant stack. Upstreaming before the 4090 matrix passes would add process
overhead while the design is still volatile.

**Target sprint**: After Sprint 007 if the patch passes promotion gates.

**Prerequisites**: Clean patch, correctness report, benchmark report, and
minimal RotorQuant-specific coupling.

**Files**: upstream `common/speculative.cpp`, `tools/server/server-context.cpp`,
possibly upstream tests.

---

## D-006: Automatic Production Default Flip

**What**: Make multislot MTP the default production profile for multi-client
traffic.

**Why deferred**: Sprint 007 may produce a preview profile, but production
promotion requires explicit operator approval after the benchmark report.

**Target sprint**: After Sprint 007 gates pass and the user approves rollout.

**Prerequisites**: Preview promotion gates pass; homelab deployment validated
under real traffic.

**Files**: `docker/entrypoint.sh`, `docker-compose.yml`, `k8s/values.yaml`,
`k8s/README.md`, `README.md`.

---

## D-007: Multi-GPU / Tensor-Parallel MTP

**What**: Extend MTP multislot serving to multi-GPU deployments.

**Why deferred**: Current target is one 24 GB RTX 4090. Multi-GPU changes the
memory, scheduling, and communication model and is not needed for the immediate
homelab bottleneck.

**Target sprint**: Future, only if single-GPU MTP is stable.

**Prerequisites**: Sprint 007 single-GPU correctness and throughput stable.

**Files**: `docker/entrypoint.sh`, `docker-compose.yml`, `k8s/`, upstream
llama.cpp split/tensor-parallel paths.

---

## Summary Table

| Item | Target Sprint | Blocker |
|---|---|---|
| Hybrid load-aware router | Sprint 008 or fallback sprint | Sprint 007 benchmark result |
| General batched-draft interface | Future | MTP path proven or clearly diagnosed |
| Streaming SSE validation | Future hardening | Non-streaming correctness passes |
| Stochastic sampling validation | Future validation | Greedy correctness stable |
| Upstream llama.cpp PR | After Sprint 007 | Patch passes correctness/perf gates |
| Production default flip | After Sprint 007 + approval | Preview promotion gates pass |
| Multi-GPU / tensor-parallel MTP | Future | Single-GPU MTP stable |
