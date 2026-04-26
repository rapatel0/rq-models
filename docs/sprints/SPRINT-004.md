# Sprint 004: vLLM Substrate Bring-up + planar3 Port

**Date**: 2026-04-25
**Hardware**: NVIDIA RTX 5090 (32 GB), with Qwen3.6-27B Q4_K_M as primary target
**Model targets**: Qwen3.6-27B (primary, dense, planar3 default), Qwen3.5-27B (parity check), Qwen3.6-35B-A3B (future MoE in Sprint 005+)
**Current best baseline (llama.cpp substrate)**: planar3 PPL = 8.20 (wikitext-2, ctx=4096)
**Status**: planned (supersedes Sprint 003 SpectralQuant — see [SPRINT-003.md](SPRINT-003.md) close note)

---

## Overview

rq-models currently runs on a llama.cpp fork (`johndpope/llama-cpp-turboquant`,
branch `feature/planarquant-kv-cache`, commit `20efe75`) which has the
RotorQuant iso3/planar3/planar4/iso4 KV cache kernels integrated. This
substrate is production-stable but limited compared to vLLM in three
dimensions that matter for rq-models's roadmap:

1. **Multi-tenant throughput**: vLLM's continuous batching is materially
   stronger than llama.cpp's parallel mode for serving multiple concurrent
   requests on a single GPU.
2. **Speculative decoding**: vLLM has a maintained plug-in architecture
   (Eagle-3, DFlash via Speculators, n-gram lookup). llama.cpp's specdec
   support lags. When dflash matures (~6+ months out) we want to plug
   into a maintained substrate, not chase porting work.
3. **Multi-GPU**: vLLM tensor + pipeline parallelism is years ahead of
   llama.cpp's split-mode.

Sprint 004 brings up a vLLM-substrate path for rq-models with **one quant
variant (planar3, current dense default) ported end-to-end**, validated
against the llama.cpp baseline at PPL parity. This is a **bring-up
sprint**, not a feature-completeness sprint. Other variants (iso3, iso4,
planar4) and MoE models follow in Sprints 005+.

The win condition is a working serving path: `docker compose --profile
qwen36-27b-vllm up` serves Qwen3.6-27B with planar3 KV at PPL within
Δppl ≤ 0.05% of the llama.cpp substrate, with end-to-end aggregate
throughput at N=4 matching or exceeding the llama.cpp path on the same
hardware.

The **substrate fork** is the durable artifact. Once it exists and
serves one model with one quant variant, future sprints add variants
and models without re-doing fork-setup work.

## Use Cases

1. **Multi-tenant production serving**: rq-models users serving multiple
   concurrent chat sessions on a single RTX 5090 / data-center GPU
   benefit from vLLM continuous batching's higher aggregate throughput.
2. **Future speculative decoding research**: when dflash / Eagle-3 /
   lookup decoding stabilize, plugging them into the rq-models stack
   requires the engine to have a maintained specdec interface. vLLM has
   it; llama.cpp lags.
3. **Multi-GPU deployment**: 4× V100, 2× A100, etc. via vLLM tensor
   parallelism. The k8s + Helm chart work on `main` (commit `58c1ee8`)
   already targets multi-GPU; this sprint gives it an engine that
   actually uses the GPUs in parallel.
4. **Alternate substrate for benchmarking**: cross-substrate PPL +
   throughput comparison validates that RotorQuant is the differentiator,
   not a property of one engine's idiosyncrasies. This matters for
   defensibility of the research contribution.
5. **Foundation for non-GGUF model support**: vLLM loads HF transformers
   format directly. Future models that don't have GGUF builds (or have
   them late) can still run on rq-models via the vLLM path.

## Architecture

### Substrate layout

```
rapatel0/rq-vllm   (new)         ← fork of upstream vllm-project/vllm
├── vllm/                         ← upstream tree, periodically rebased
├── vllm/model_executor/layers/quantization/
│   └── rotorquant.py             ← NEW: RotorQuantConfig (mirror GPTQConfig pattern)
├── vllm/model_executor/layers/quantization/kernels/
│   └── rotorquant/               ← NEW: ported KV-write/read CUDA kernels
│       ├── planar3.cu            ← Sprint 004 scope: this only
│       ├── (iso3.cu)             ← Sprint 005
│       ├── (iso4.cu)             ← Sprint 005
│       └── (planar4.cu)          ← Sprint 005
└── tests/quantization/test_rotorquant.py   ← parity tests against llama.cpp
```

### Plug-in points

vLLM's quantization architecture (per [vllm.model_executor.layers.quantization](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization))
exposes three customization surfaces:

1. **`QuantizationConfig`** subclass (e.g., `GPTQConfig`, `AWQConfig`) —
   describes how a model identifies as RotorQuant-quantized and which
   layers to transform.
2. **`QuantizeMethodBase`** subclass — applied per-layer; rewrites
   `forward` to use the custom kernels.
3. **KV cache layer registration** — vLLM's KV cache manager already
   handles paged blocks. We add a custom dtype enum entry and the
   pack/unpack kernels for our 3 bpe block layout.

`mitkox/vllm-turboquant`'s patches (4 commits, `c6b2ee9`, `cee479e`,
`5fc73a3`, `7a8a095`) demonstrate the integration shape for a different
quant scheme — read them as **reference for where the plug-in points
are**, not as the merge base. Our fork bases on upstream vLLM directly.

### KV layout reuse

The current rq-models RotorQuant planar3 kernels (in the
`johndpope/llama-cpp-turboquant` fork at commit `20efe75`) implement the
3 bpe planar rotation + Lloyd-Max codebook. The math is identical
between substrates; only the kernel-launch boilerplate, paged-block
addressing, and host↔device API change. The CUDA `__device__` functions
that do the rotation + quantization should port near-1:1.

### vLLM version pinning

Pin to the **latest stable vLLM tag** at sprint kickoff (current as of
2026-04-25 is `v0.7.x`; check for newer at sprint start). vLLM's
internal API churn is significant; pinning protects the port from
upstream breakage during the sprint. Rebasing onto a newer tag is its
own follow-up sprint.

## Implementation

### Phase 0 — Fork setup + bring-up of unmodified vLLM (target: 2-3 days)

1. Fork `vllm-project/vllm` to `rapatel0/rq-vllm`. Pin to latest stable
   tag.
2. Add Docker build target alongside existing `docker/Dockerfile`'s
   llama.cpp path: `docker/Dockerfile.vllm` builds vLLM with our fork as
   source.
3. Bring up Qwen3.6-27B serving on the unmodified fork via `docker
   compose --profile qwen36-27b-vllm-baseline up` to confirm
   environment, drivers, model loading work end-to-end before any
   RotorQuant integration. Sanity: a `/v1/chat/completions` returns
   sensible tokens.
4. Run `scripts/eval_perplexity.py` against the unmodified vLLM serving
   (with f16 KV) on Qwen3.6-27B; record baseline. This is the
   "no-quantization vLLM" reference number.

### Phase 1 — RotorQuantConfig scaffolding (target: 2-3 days)

1. Add `vllm/model_executor/layers/quantization/rotorquant.py`. Define
   `RotorQuantConfig` and `RotorQuantQuantizeMethod`. Mirror GPTQ config
   loading from model HF metadata (`quantization_config.quant_method =
   "rotorquant"`).
2. Register in `vllm/model_executor/layers/quantization/__init__.py`'s
   `QUANTIZATION_METHODS` dict.
3. Wire a no-op kernel path: `RotorQuantQuantizeMethod` initially does
   passthrough fp16 — i.e., pretends to be quantized but actually runs
   f16 KV. Goal: verify the registration plumbing without touching
   kernels yet.
4. Smoke test: load Qwen3.6-27B with `--quantization rotorquant`, get
   identical output to `--quantization none` (f16 KV).

### Phase 2 — planar3 kernel port (target: 1 week)

1. Port the planar3 KV-write kernel from `johndpope/llama-cpp-turboquant`
   commit `20efe75` (file path TBD — locate `planar3` quant-write CUDA
   sources in that fork) to `vllm/model_executor/layers/quantization/kernels/rotorquant/planar3.cu`.
2. Port the planar3 KV-read kernel similarly.
3. Wire into the no-op `RotorQuantQuantizeMethod` from Phase 1; replace
   passthrough with real planar3 quantization.
4. Bypass any vLLM mechanisms incompatible with 3-bpe block packing
   (potentially: paged-block size assumptions). Document each bypass.
5. Smoke test: load Qwen3.6-27B with `--quantization rotorquant
   --rotorquant-mode planar3`, generate tokens, verify they're sensible
   (not random noise — sanity check that quantization isn't catastrophic).

### Phase 3 — PPL validation gate (target: 2-3 days)

1. Run `scripts/eval_perplexity.py` against the planar3 vLLM serving on:
   - Qwen3.6-27B (primary)
   - Qwen3.5-27B (parity check)
2. Compare against the llama.cpp planar3 baselines:
   - llama.cpp planar3 PPL on Qwen3.6-27B: TODO (capture as part of this
     sprint if not already in `docs/BENCHMARK-REPORT.md`)
   - llama.cpp planar3 PPL on Qwen3.5-27B: 8.20
3. **Hard gate**: Δppl ≤ 0.05% (i.e., absolute Δ ≤ 0.0041 PPL units at
   8.20). If Δppl exceeds this threshold, kernel port has a correctness
   bug — debug before Phase 4.
4. Save results to `docs/SPRINT-004-PPL-COMPARISON.md`.

### Phase 4 — Throughput benchmark (target: 2-3 days)

1. Run `scripts/bench_n_parallel.py` against both substrates at N ∈ {1,
   2, 4, 8} using matched workload.
2. Record:
   - Aggregate tok/s
   - Per-session tok/s
   - TTFT (time to first token)
   - End-to-end latency at p50, p95
3. **Soft gate**: vLLM aggregate throughput at N=4 should match or
   exceed llama.cpp at N=4 on the same hardware. If significantly
   slower, document why (likely PagedAttention block-size mismatch with
   3-bpe layout — design follow-up for Sprint 005).
4. Save results to `docs/SPRINT-004-THROUGHPUT-COMPARISON.md`.

### Phase 5 — Deployment + closeout (target: 1-2 days)

1. Add `docker/Dockerfile.vllm` and corresponding entry in
   `docker-compose.yml`: `qwen36-27b-vllm` profile.
2. Update `docker/test.sh` to smoke-test the new profile.
3. Update `Makefile` with `make run-qwen36-27b-vllm` target.
4. Update `README.md` with the substrate-choice section: when to use
   llama.cpp path vs vLLM path.
5. Write `SPRINT-004-FOLLOWUPS.md` for any deferred items found during
   execution.

## Files Summary

**New repository**:
- `rapatel0/rq-vllm` (fork of upstream vllm-project/vllm at latest stable
  tag)

**New files in `rapatel0/rq-vllm`**:
- `vllm/model_executor/layers/quantization/rotorquant.py`
- `vllm/model_executor/layers/quantization/kernels/rotorquant/planar3.cu`
- `tests/quantization/test_rotorquant.py`
- `docs/ROTORQUANT.md` (in-fork documentation)

**Files in `rapatel0/rq-models`**:
- `docs/sprints/SPRINT-004.md` (this doc)
- `docs/sprints/SPRINT-004-DEFERRED.md` (companion)
- `docs/SPRINT-004-PPL-COMPARISON.md` (Phase 3 output)
- `docs/SPRINT-004-THROUGHPUT-COMPARISON.md` (Phase 4 output)
- `docker/Dockerfile.vllm` (new)
- `docker-compose.yml` (add `qwen36-27b-vllm` profile)
- `Makefile` (add `run-qwen36-27b-vllm` target)
- `README.md` (substrate-choice section)
- `docs/sprints/SPRINT-004-FOLLOWUPS.md` (Phase 5 output, if any items
  emerge)

## Definition of Done

- [ ] `rapatel0/rq-vllm` exists, forked from upstream vLLM at a pinned
      stable tag, recorded in this doc and the in-fork README.
- [ ] Unmodified vLLM fork serves Qwen3.6-27B successfully (Phase 0
      smoke).
- [ ] `RotorQuantConfig` + `RotorQuantQuantizeMethod` plumbing registered
      in vLLM (Phase 1).
- [ ] planar3 KV-write + KV-read kernels ported (Phase 2).
- [ ] `--quantization rotorquant --rotorquant-mode planar3` produces
      sensible tokens for Qwen3.6-27B and Qwen3.5-27B (Phase 2 smoke).
- [ ] **PPL parity gate**: Δppl ≤ 0.05% vs llama.cpp planar3 baseline on
      Qwen3.6-27B and Qwen3.5-27B (Phase 3, hard gate).
- [ ] **Throughput parity check**: vLLM aggregate tok/s at N=4 ≥
      llama.cpp aggregate tok/s at N=4 OR documented justification why
      not (Phase 4, soft gate).
- [ ] `docker compose --profile qwen36-27b-vllm up` produces a working
      server reachable at port 8080.
- [ ] `make run-qwen36-27b-vllm` works.
- [ ] Tests in `rq-vllm` repo: `pytest tests/quantization/test_rotorquant.py`
      green.
- [ ] README documents the substrate-choice question explicitly: when
      llama.cpp path, when vLLM path.
- [ ] Closeout follow-ups document written if needed.

## Risks & Mitigations

Ranked by likelihood × impact.

| # | Risk | L×I | Mitigation |
|---|---|---|---|
| 1 | **vLLM internal API churn** during the sprint window | High × Med | Pin to a specific stable tag at kickoff; do not rebase mid-sprint. |
| 2 | **PPL parity miss** (kernel port bug producing silent quality regression) | Med × High | Hard Δppl ≤ 0.05% gate at Phase 3. If failed, debug before Phase 4. Compare per-token KV outputs against llama.cpp directly if needed. |
| 3 | **3-bpe block packing incompatible with vLLM PagedAttention block-size assumptions** | Med × High | Identify in Phase 2; if hit, use a smaller paged-block size or a custom KV layout. Document any deviation as a Sprint 005 follow-up. |
| 4 | **Throughput regression vs llama.cpp** at N=1-4 | Med × Med | Acceptable in Sprint 004 — soft gate. Document and design follow-up. vLLM's strengths are at N≥4 with continuous batching. |
| 5 | **vLLM build/CUDA compatibility issues** with current driver / CUDA version | Med × Med | Docker isolation; pin CUDA version in Dockerfile. Use matrix testing if needed. |
| 6 | **mitkox/vllm-turboquant patches outdated** vs vLLM API we target | Low × Low | We're not basing on those patches, just reading them. If outdated, just read upstream vLLM's quantization code directly. |
| 7 | **Sprint 004 scope creep** to include iso3 / iso4 / planar4 mid-sprint | Med × Low | Strict scope: planar3 only. Other variants are Sprint 005. Document any cross-cutting changes that benefit them as deferred items. |
| 8 | **HF transformers metadata for rotorquant not yet defined** | Low × Med | Define a minimal `quantization_config` schema in this sprint; document for use by future model conversions. |

## Security Considerations

- vLLM fork inherits upstream vLLM's request validation and OpenAI-API
  authentication patterns. No new auth surface added.
- RotorQuant kernel is a passive transform on KV state; doesn't open
  new privilege paths.
- New Docker image must avoid baking secrets (HF tokens, etc.) into
  layers — use build-args + runtime env, mirroring existing
  `docker/Dockerfile`.

## Dependencies

- **Upstream**: `vllm-project/vllm` latest stable tag. Pinned at sprint
  kickoff.
- **Reference**: `mitkox/vllm-turboquant` (read-only, not depended on).
- **Source kernels**: `johndpope/llama-cpp-turboquant` at commit
  `20efe75` for the planar3 reference implementation to port.
- **Hardware**: RTX 5090 32 GB for the bring-up + benchmark; eventual
  multi-GPU follow-up sprints.
- **Validation harness**: existing `scripts/eval_perplexity.py` + the
  newly migrated `scripts/bench_n_parallel.py` (commit `7560c88`).

## Open Questions

1. **vLLM tag pinning**: which exact tag? Confirm at kickoff. (Initial
   guess: latest `v0.7.x` stable.)
2. **PPL evaluation context**: keep ctx=4096 for parity with existing
   llama.cpp baselines, or expand to 8K/16K to test long-ctx kernel
   correctness? Recommend ctx=4096 for parity, follow-up for long-ctx.
3. **HF transformers `quantization_config` schema**: minimum viable for
   `quant_method = "rotorquant"`. Should include rotation type
   (planar/iso) and bit budget (3/4). Need exact JSON.
4. **Should the rq-vllm fork track upstream automatically** (e.g., a
   weekly-rebase CI job), or rebase manually per sprint? Recommend
   manual per-sprint to keep churn predictable.
5. **Multi-GPU scope in this sprint**: add to Sprint 004, or defer to
   Sprint 005? Recommend defer; bring-up + planar3 + parity is already
   a 2-3 week sprint.
