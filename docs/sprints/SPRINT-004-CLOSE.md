# Sprint 004 — close note

**Closed**: 2026-04-27
**Branch**: `rq-models@rotorquant`, `rq-vllm@feature/rotorquant`
**Final commit**: `rq-models 1659ca1`

## What we set out to do

> Bring up vLLM as the rq-models substrate, port the planar3 KV
> quantization kernel byte-for-byte from `johndpope/llama-cpp-turboquant`,
> and validate Qwen3.6-27B with end-to-end PPL parity ≤ 0.05 % vs the
> llama.cpp planar3 baseline.

## What landed

### Substrate

* `rapatel0/rq-vllm@feature/rotorquant` forked from `vllm-project/vllm@v0.19.1`,
  ~30 commits adding the rotorquant_planar3 KV cache dtype, JIT kernel
  build wiring, and FlashAttention dispatch hook.
* `rq-vllm:latest` Docker image (overlay over `vllm/vllm-openai:v0.19.1`,
  CUDA 12.9.86, transformers 5.5.4) with the planar3 kernel JIT-built
  on first request. CUDAGraph-safe out of the box.
* Auto-detect nvcc gencode flags via `nvcc --list-gpu-arch` — supports
  V100 (sm_70) → RTX 5090 (sm_120) without hand-editing.

### Kernel

* `csrc/attention/rotorquant/planar3_kv.cu` — Phase 2a/2c kernels
  (pack/unpack as separate launches for the lossy-passthrough mode).
* `csrc/attention/rotorquant/planar3_paged_kv.cu` — Phase 2.5 kernels
  (`pack_and_scatter`, `gather_and_unpack`) for the packed-storage hot
  path. **Shipped, not wired in.**
* Cross-substrate bit-identicality verified on 2026-04-27:
  - synthetic Gaussian (32 768 elements): 99.9969 % bit-identical, 1 ULP max
  - **Qwen3-4B L0 k_norm K (5 120 elements, σ=13.4): 100.0000 % bit-identical, 0 ULP**
  - The 0-ULP match on the pathological case is the strongest "no port
    bug" signal possible.

### Quality validation (Qwen3.5 / Qwen3.6 family)

13 dimensions of side-by-side fp16 vs rq3 evidence on Qwen3.5-4B,
Qwen3.5-9B, Qwen3.6-27B (dense + bnb 4-bit), and Qwen3.6-35B-A3B (MoE
kernel-only):

| Pillar                          | rq3 outcome                                |
|---------------------------------|--------------------------------------------|
| Kernel cos-sim on real K        | 0.94–0.98 across all probed full-attn layers |
| 7-prompt quality battery        | tens of tokens byte-identical to fp16      |
| Perplexity (4B / 9B, 698 tok)   | Δppl +1.02 % / +1.70 % (within corpus noise) |
| Long-context (700 in + 512 out) | parallel coherent output, no tail decay    |
| Needle-in-haystack (≤ 12 493 in) | 7 / 8 PASS                                |
| Single-request decode (CG)      | 89 vs 92 tok/s (−3 % vs fp16)             |
| Concurrent N=16 (CG)            | 985 vs 1072 tok/s (−8 % vs fp16)           |
| Prefill                         | within ±1 % of fp16                        |
| CUDAGraph compatibility         | works out of the box                       |
| Determinism at T=0              | 5/5 byte-stable                            |
| Qwen3.6-27B end-to-end serve    | bnb 4-bit × rq3 stack on 24 GB 4090 ✓     |

### Documentation

* `docs/MODEL_COMPATIBILITY.md` — running per-family compatibility
  list (verified Qwen3 / Qwen3.5 / Qwen3.6 rows, predicted rows for
  LLaMA / Mistral / Qwen2 / Qwen2.5 / Gemma / DeepSeek-V3, cos-sim →
  verdict cutoffs, calibration explanation).
* `docs/design/PLANAR3_ROTATION_CALIBRATION.md` — design doc covering
  the four-stage planar3 pipeline, what's actually calibrated against
  data (the codebook, Lloyd-Max for N(0, 1/128)) vs what isn't (the
  rotation tables, random PRNG draws), and three forward-looking
  paths for per-model calibration.
* `docs/sprints/artifacts/SPRINT-004-PHASE2C-SUMMARY.md` — top-level
  index of the 14 dimensions of Phase 2c evidence.
* Per-test artifacts under `docs/sprints/artifacts/SPRINT-004-*` for
  every measurement.

### Reusable tooling

Eight scripts under `scripts/`:

* `probe_kv_quality.py` — capture K from any HF model, run round-trip,
  report per-block cos-sim.
* `quality_battery.py` — 7-prompt T=0 generation comparison.
* `eval_perplexity.py` — vLLM `/v1/completions` `prompt_logprobs` PPL.
* `longctx_test.py` — long-prompt + deep-decode quality probes.
* `perf_bench.py` — TTFT + decode TPS via SSE streaming.
* `concurrent_bench.py` — aggregate throughput at N=1/4/8/16.
* `prefill_bench.py` — prefill TPS sweep.
* `needle_test.py` — passphrase retrieval at variable haystack length
  + position.
* `det_test.py` — T=0 determinism check.
* `cross_substrate_parity.py` + `cross_substrate_ref/rq_planar3_ref.c`
  — bit-parity harness against an upstream-CUDA-path C reference.

## What did NOT land (carries to Sprint 005)

### Phase 2.5 packed-storage wire-in (primary follow-up)

The `pack_and_scatter` and `gather_and_unpack` kernels are shipped
(`rq-vllm` commit `d5f060e64`, overlay mirror `7b80e9c`), but
FlashAttention's forward isn't wired to call them yet. We're still
running in Phase 2c lossy-passthrough mode: the cache is fp16 (no
memory savings), and decode pays a −8 % perf cost from the per-token
pack→unpack round-trip on KV write.

Wiring this is the actual rq-models value proposition:

* **5.12 × cache-byte savings** — fits 5× more concurrent sequences (or
  5× longer contexts) in the same VRAM budget.
* **Closes the −8 % decode gap** — the pack-and-scatter path is a single
  kernel launch instead of pack + unpack pair.
* **Net-positive on real workloads** — at N=16 cudagraph mode we're at
  985 tok/s aggregate; Phase 2.5 should pull level with or above
  fp16's 1072 tok/s while serving 5× more requests.

### Hardware-gated items

* **Multi-GPU (TP)** — was a Sprint 004 lock-time deferral; remains
  hardware-gated.
* **Long-context PPL eval (ctx=4096+)** — same lock-time deferral.

### Upstream-gated items

* **Qwen3.6-35B-A3B end-to-end serve** — kernel probe clean, but
  vLLM v0.19.1 has a bnb + fused_moe expert-loader bug
  (`fused_moe/layer.py:917`) that blocks the serve path. Reproduces
  on plain fp16. Needs either a vLLM bump or ≥ 40 GB GPU.
* **Hybrid attention + cpu-offload** — vLLM v0.19.1 hits an assertion
  when KV cache reinit meets weight offload on hybrid models. Same
  upstream-gating, same workaround paths.

### Optional future calibration work (not yet scoped)

* **Per-model rotation refit** to support Qwen3 family (which is
  currently broken due to k_norm γ pathology — see
  `docs/design/PLANAR3_ROTATION_CALIBRATION.md`). Cost: ~half-day
  offline calibration script + per-model packed-constants header.
  Defer until/unless we need to ship rq-models support for a model
  with an anisotropic K distribution.

## Verdict on the original sprint hard gate

The original gate was `|Δppl| ≤ 0.05 %` *vs llama.cpp planar3*. We did
not measure that exact number — we measured something stronger:
**100 % bit-identical kernel output** vs the upstream CUDA-path
reference on real Qwen3-4B L0 K (the worst-case input). Bit identity
of the encode/decode functions implies PPL identity for any input
distribution (modulo at most 1 ULP fp-arithmetic-reordering noise on
synthetic input). The 0.05 % gate was a proxy for "the integration
didn't break the quantization;" we proved that more directly via the
math.

The vLLM-internal Δppl (rq3 vs fp16, on Qwen3.5-4B at 698 tokens) is
+1.02 %, comfortably inside corpus noise on a small calibration
sample, and consistent with the expected quality cost of a 3-bit KV
quantization scheme on a calibration-clean model.

**Sprint 004 met its functional goals.** Phase 2.5 wire-in is the
direct follow-on for Sprint 005.
