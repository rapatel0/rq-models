# Sprint 003 — Deferred Items

Items discussed, proposed in drafts, or raised in critiques but explicitly
excluded from Sprint 003 scope. Each item has a target sprint and prerequisites
so future planners can orient quickly.

---

## D-001: SPECTRAL_3BIT Variant

**What**: A `SPECTRAL_3BIT` quantization preset (3-bit signal / 1-bit noise),
analogous to `planar3`. Would enable a head-to-head comparison of spectral3 vs
planar3 at the same ~0.875 bpe budget.

**Why deferred**: Doubles the benchmark matrix (two SpectralQuant variants × all
baselines × two models). Critiques recommended proving 4-bit signal / 2-bit noise
works before adding variants. Explicitly dropped during sprint planning.

**Target sprint**: Sprint 004

**Prerequisites**: Sprint 003 spectral4 validated; `SPECTRAL_4BIT` preset and
`SpectralQuantizer` implemented. Adding 3-bit is then a config + codebook change.

**Files**: `turboquant/config.py` (add `SPECTRAL_3BIT`), `turboquant/spectral/quantizer.py`
(add 3-bit Lloyd-Max codebook fitting path)

---

## D-002: Hybrid SpectralQuant + Clifford Rotor

**What**: Apply a Clifford isoclinic rotor within the d_eff signal subspace after
PCA projection. At d_eff=4 (K channels), the rotor is nearly free (4×4 rotation).
May improve Lloyd-Max quantization accuracy in the signal subspace by aligning
principal axes with codebook structure.

**Why deferred**: Speculative research; IsoQuant benchmarks already showed isoclinic
rotors don't consistently improve quality. Per critique: "Hybrid Clifford+SpectralQuant
as a use case inflates expectations." Deferred as an open question per user decision.

**Target sprint**: Sprint 004 or later (experimental)

**Prerequisites**: Sprint 003 spectral4 validated; baseline SpectralQuant PPL
measured. Not worth implementing until PCA calibration is proven.

**Files**: `turboquant/spectral/quantizer.py` (encode_k/encode_v would optionally
apply rotor after projection), `turboquant/spectral/calibrator.py` (rotor parameters
could be fitted alongside Lloyd-Max codebooks)

---

## D-003: GGUF Bridge Script (spectral_ggml.py)

**What**: A Python script to export SpectralQuant calibration sidecars into a
GGUF-compatible format or embed them in the GGUF metadata section. Enables the
C/CUDA Sprint 004 implementation to consume sidecar data without maintaining two
separate format parsers.

**Why deferred**: Codex draft included this as Sprint 003 scope. Rejected: the
Python prototype validation should be complete before designing the GGUF format.
The safetensors sidecar is sufficient for the Python prototype. GGUF bridge is
Sprint 004 work.

**Target sprint**: Sprint 004

**Prerequisites**: Sprint 003 calibration sidecar format finalized and validated;
`docs/SPECTRAL-CUDA-DESIGN.md` (Phase 5) written — the bridge format depends on
the C/CUDA design decisions.

**Files**: New file `scripts/spectral_ggml.py`; may require changes to GGUF
format handling in `ggml/` (llama.cpp fork)

---

## D-004: C/CUDA llama.cpp Kernel Implementation

**What**: Full implementation of `GGML_TYPE_SPECTRAL4` as a first-class KV cache
type in the TurboQuant llama.cpp fork. Includes:
- `GGML_TYPE_SPECTRAL4` enum + type registration in `ggml/src/ggml.c`
- CUDA encode kernel: K/V f16 → PCA project → split → Lloyd-Max lookup → pack
- CUDA decode kernel: unpack → Lloyd-Max → back-project → f16
- `llama-kv-cache.cpp` hooks for spectral4 cache type
- Docker entrypoint: `KV_CACHE_TYPE=spectral4` support

**Why deferred**: "C/CUDA llama.cpp integration is designed (15-20% of effort)
but implemented in Sprint 004." The Python prototype is the validation gate —
implementing C/CUDA before PPL is validated is high-risk scope.

**Target sprint**: Sprint 004

**Prerequisites**: Sprint 003 kill gate passed (spectral4 PPL < 8.20 on 27B);
`docs/SPECTRAL-CUDA-DESIGN.md` written with full audit and kernel design.

**Files**: `ggml/src/ggml.c`, `ggml/src/ggml.h`, `ggml/src/ggml-cuda/`
(new kernel file), `src/llama-kv-cache.cpp`, `docker/entrypoint.sh`,
`docker-compose.yml`

---

## D-005: Qwen3.6-35B-A3B SpectralQuant Support

**What**: Calibration and validation of SpectralQuant on Qwen3.6-35B-A3B (MoE
architecture). The MoE model has a different attention layer distribution and
potentially different KV activation spectral structure than Qwen3.5-27B.

**Why deferred**: MoE architecture complexity is out of scope for Sprint 003.
Recommendation: validate on Qwen3.5-27B (dense) first before handling MoE-specific
edge cases (router layers, expert K/V cache separation).

**Target sprint**: Sprint 004 (alongside C/CUDA if spectral4 is validated)

**Prerequisites**: Sprint 003 spectral4 validated on Qwen3.5-27B; `SpectralCalibrator`
handles SSM skip generically (should already work, but MoE expert layers need
review).

**Files**: `scripts/calibrate_spectral.py` (add `--model qwen3.6-35b` support),
`calibration/calibration-qwen3.6-35b.safetensors` (generated artifact)

---

## D-006: D-011 Open WebUI Integration (Sprint 002 Deferral)

**What**: Open WebUI frontend integration for the turbo server. Users want a
chat UI without running their own Open WebUI instance.

**Why deferred**: Carried forward from Sprint 002. Not related to SpectralQuant.
Deferred again to Sprint 004 (or later) pending core quantization work stabilization.

**Target sprint**: Sprint 004 or Sprint 005 (low priority relative to model quality)

**Prerequisites**: None technical; scheduling constraint only.

**Files**: `docker-compose.yml` (add open-webui service), `docker/` (nginx proxy
config if needed)

---

## D-007: D-013 Benchmark CI (Sprint 002 Deferral)

**What**: Automated CI pipeline that runs perplexity and throughput benchmarks
on every PR, preventing regression. Would catch future changes that degrade PPL
or tok/s before merge.

**Why deferred**: Carried forward from Sprint 002. Requires GPU runner setup.
Not practical until benchmark suite is stable (which Sprint 003 finalizes).

**Target sprint**: Sprint 004 or Sprint 005

**Prerequisites**: Sprint 003 `scripts/benchmark_spectral.py` stable; GitHub
Actions self-hosted runner with GPU access configured.

**Files**: `.github/workflows/benchmark.yml` (new), `scripts/` (benchmark
scripts must be CI-compatible with fixed seeds)

---

## D-008: Multi-GPU / Tensor Parallel Support

**What**: SpectralQuant calibration and inference across multiple GPUs with
tensor parallelism. The current design assumes a single-GPU KV cache layout.

**Why deferred**: Explicitly noted as a non-goal in the architecture section.
Sprint 003 intent is single-GPU validation. TP would require KV layout changes
that break the current sidecar format.

**Target sprint**: Future (post Sprint 004)

**Prerequisites**: C/CUDA implementation (D-004) stable on single GPU; TP
design for llama.cpp KV cache types.

**Files**: `turboquant/spectral/kv_cache.py`, `docs/SPECTRAL-CUDA-DESIGN.md`
(note TP non-goal explicitly)

---

## D-009: Non-Contiguous KV Cache (After Eviction)

**What**: Handle KV cache eviction (sliding window, streaming LLM, etc.) with
spectral quantization. Current design assumes append-only cache; eviction creates
non-contiguous entries that require re-indexing.

**Why deferred**: Identified as a missing edge case in critique. Out of scope
for prototype validation. Qwen3.5-27B's trained context (262K) is large enough
that eviction is not needed for Sprint 003 use cases.

**Target sprint**: Future (when eviction support is needed)

**Prerequisites**: Sprint 003 basic SpectralKVCache stable.

**Files**: `turboquant/spectral/kv_cache.py`

---

## Summary Table

| Item | Description | Target Sprint | Blocker |
|------|-------------|---------------|---------|
| D-001 | SPECTRAL_3BIT variant | Sprint 004 | spectral4 validated |
| D-002 | Hybrid SpectralQuant + Clifford rotor | Sprint 004+ (experimental) | spectral4 validated, PPL baseline measured |
| D-003 | GGUF bridge script (spectral_ggml.py) | Sprint 004 | SPECTRAL-CUDA-DESIGN.md finalized |
| D-004 | C/CUDA llama.cpp kernel implementation | Sprint 004 | Kill gate passed + design doc |
| D-005 | Qwen3.6-35B-A3B SpectralQuant support | Sprint 004 | spectral4 validated on 27B |
| D-006 | Open WebUI integration (D-011) | Sprint 004–005 | None technical |
| D-007 | Benchmark CI (D-013) | Sprint 004–005 | Sprint 003 benchmark scripts stable |
| D-008 | Multi-GPU / tensor parallel | Post Sprint 004 | C/CUDA single-GPU stable |
| D-009 | Non-contiguous KV cache (post-eviction) | Future | Basic SpectralKVCache stable |
