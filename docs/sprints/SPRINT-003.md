# Sprint 003: SpectralQuant KV Cache Integration

**Date**: 2026-04-20
**Hardware**: NVIDIA RTX 5090 (32 GB), GPU may be co-occupied by training
**Model targets**: Qwen3.5-27B Q4_K_M (primary), Qwen3.5-9B Q4_K_M (dev iteration)
**Current best baseline**: planar3 PPL = 8.20 (wikitext-2, ctx=4096)

---

## Overview

The existing KV cache quantization backends (planar4/planar3/iso4/iso3) apply
data-oblivious rotations — every dimension is treated equally regardless of
information content. SpectralQuant (Dynamis Labs) exploits the natural spectral
asymmetry of transformer KV caches: key vectors concentrate ~96-97% of their
signal in just 3-4% of head dimensions, while value vectors require 40-55 signal
dimensions. By running a 15-30s PCA calibration pass, SpectralQuant identifies
these signal dimensions per layer and applies high-precision Lloyd-Max quantization
only where it matters, while aggressively discarding the noise subspace.

This sprint delivers a validated Python prototype of SpectralQuant in the
`turboquant/` library with **asymmetric K/V quantization** (d_eff ≈ 4-5 for K,
d_eff ≈ 40-55 for V), calibrated offline on wikitext-2 and committed as sidecar
files. The primary deliverable is a measured answer: does SpectralQuant beat
planar3's 8.20 PPL on Qwen3.5-27B? If yes, it becomes the new default and Sprint
004 implements the C/CUDA llama.cpp kernel. If no, the sprint produces a research
report explaining why and closes the question.

C/CUDA llama.cpp integration is designed (15-20% of effort) but implemented in
Sprint 004. The Python prototype is the validation gate.

## Use Cases

1. **Higher quality at current context**: Users at 112K-245K context on RTX 4090/5090
   want lower perplexity degradation from KV compression. SpectralQuant targets
   PPL < 8.20 (beating planar3) at ≥ 4.9x compression.

2. **More context at current quality**: SpectralQuant's higher compression ratio
   frees VRAM for longer context. If spectral4 achieves 5.95x (SpectralQuant paper
   claim), that extends RTX 5090 max context from 300K to ~370K tokens at
   iso-quality vs planar3.

3. **Research baseline**: Unified Python benchmark harness comparing SpectralQuant
   vs planar4/planar3/iso4/iso3/f16 on the same hardware and methodology as prior
   sprints. Reproducible and extensible.

4. **Sprint 004 unblocking**: A validated design document for `GGML_TYPE_SPECTRAL4`
   and the CUDA kernel, ready for implementation in the llama.cpp fork.

## Architecture

### K/V Asymmetry

SpectralQuant exploits that K and V have fundamentally different spectral structure:

| Channel | Signal dims (d_eff) | Noise dims | Strategy |
|---------|-------------------|------------|----------|
| K | ~4–5 of 256 | ~251–252 | 4-bit Lloyd-Max on signal, 2-bit on noise |
| V | ~40–55 of 256 | ~201–216 | 4-bit Lloyd-Max on signal, 2-bit on noise |

Both K and V use the same algorithm; only the calibrated d_eff differs. Asymmetric
treatment requires separate `eigenvectors_k`, `codebook_k`, `eigenvectors_v`,
`codebook_v` per attention layer.

### Data Flow

```
Calibration (offline, ~15-30s):
  wikitext-2 sample (512 tokens × N prompts)
      → forward pass through model (no_grad, f16)
      → collect KV activations per attention layer [prompts, heads, tokens, head_dim]
      → PCA at f32 per head → eigenvalues, eigenvectors
      → d_eff_k: dims explaining 99% K variance (~4-5)
      → d_eff_v: dims explaining 99% V variance (~40-55)
      → if no clear gap: fallback to full-dim uniform quantization for that layer
      → Lloyd-Max codebook fit (4-bit signal, 2-bit noise) for K and V separately
      → validate PCA at f32 vs f16 (cosine similarity of eigenvectors > 0.999)
      → save: calibration-{model}.safetensors

Inference (online, per token):
  raw K/V (f16) [heads, head_dim]
      → [project onto eigenvectors (f32)] → K_signal [heads, d_eff_k]
      → [split K_noise = K - K_signal_reconstructed]
      → [Lloyd-Max 4-bit encode K_signal] + [Lloyd-Max 2-bit encode K_noise]
      → [pack] → compact KV entry

  compact KV entry
      → [unpack, Lloyd-Max decode signal + noise]
      → [inverse project] → K_hat (f16)
      → feed to attention

SSM layers: no KV cache; SpectralKVCache passes through unchanged.
```

### Module Structure

```
turboquant/
  spectral/
    __init__.py          public API: SpectralKVCache, SpectralCalibrator, load_calibration
    calibrator.py        PCA calibration, d_eff selection, Lloyd-Max fitting
    quantizer.py         online encode/decode (asymmetric K/V)
    store.py             safetensors sidecar serialization/deserialization
    kv_cache.py          DynamicCache subclass
tests/
  test_spectral.py       unit + integration tests
scripts/
  calibrate_spectral.py  CLI: produces calibration-{model}.safetensors
  benchmark_spectral.py  PPL, cosine sim, NIAH, throughput comparison
```

### Calibration Sidecar Format

```
calibration-qwen3.5-27b.safetensors:
  layer_{n}_eigvec_k:    float32 [n_kv_heads, head_dim, d_eff_k]
  layer_{n}_eigvec_v:    float32 [n_kv_heads, head_dim, d_eff_v]
  layer_{n}_codebook_k:  float32 [n_kv_heads, 2^4, d_eff_k]  # 4-bit signal
  layer_{n}_codebook_v:  float32 [n_kv_heads, 2^4, d_eff_v]
  layer_{n}_codebook_k_noise: float32 [n_kv_heads, 2^2, head_dim-d_eff_k]  # 2-bit noise
  layer_{n}_codebook_v_noise: float32 [n_kv_heads, 2^2, head_dim-d_eff_v]
  layer_{n}_d_eff_k:     int32 scalar
  layer_{n}_d_eff_v:     int32 scalar
  ... for each of the 16 attention layers
```

Estimated size: ~2–5 MB per model. Stored in `calibration/` directory; generated
by `scripts/calibrate_spectral.py` and committed via git-lfs (or generated locally
with a documented one-liner if git-lfs is unavailable).

## Implementation

### Phase 1: SpectralQuant Core (~30% of effort)

**Files:**
- `turboquant/spectral/__init__.py`
- `turboquant/spectral/calibrator.py`
- `turboquant/spectral/quantizer.py`
- `turboquant/spectral/store.py`
- `turboquant/spectral/kv_cache.py`
- `turboquant/config.py` — add `SPECTRAL_4BIT` preset

**Tasks:**
- [ ] `SpectralCalibrator.fit(model, calibration_prompts)`:
  - Collect KV activations for all 16 attention layers (skip SSM)
  - Run PCA at f32 precision per head for K and V separately
  - Select d_eff_k and d_eff_v via 99% variance threshold
  - Eigenvalue gap fallback: if variance is flat (no clear gap), set d_eff = head_dim
    (fall back to full-dim Lloyd-Max for that layer)
  - Fit Lloyd-Max codebooks (4-bit signal, 2-bit noise) for K and V
  - Validate: cosine similarity of f32 vs f16 PCA eigenvectors > 0.999
- [ ] `SpectralQuantizer(calibration)`: holds K/V codebooks + eigenvectors
  - `encode_k(k)` → packed indices (signal 4-bit + noise 2-bit)
  - `encode_v(v)` → packed indices
  - `decode_k(indices)` → f16 K approximation
  - `decode_v(indices)` → f16 V approximation
  - Handle edge cases: empty cache (first token), context < d_eff, layer without calibration
- [ ] `CalibrationStore.save/load`: safetensors round-trip for all layer data
- [ ] `SpectralKVCache(DynamicCache)`:
  - `update()`: quantize new K/V tokens via `SpectralQuantizer`
  - SSM layer detection: inspect `layer_type` from model config; skip if not attention
  - GQA handling: 4 KV heads, head_dim=256 for 27B
  - Fallback: if sidecar absent → raise `CalibrationNotFoundError` with clear message
- [ ] `turboquant/config.py`: add `SPECTRAL_4BIT = BitConfig(signal_bits=4, noise_bits=2)`

### Phase 2: Calibration Pipeline (~15% of effort)

**Files:**
- `scripts/calibrate_spectral.py`

**Tasks:**
- [ ] CLI: `python scripts/calibrate_spectral.py --model Qwen3.5-27B --output calibration/`
- [ ] Collect activations from N=32 wikitext-2 prompts (512 tokens each)
- [ ] GPU memory guard: run with `torch.no_grad()`, f16, batch_size=1; target < 8 GB
- [ ] Sweep d_eff variants for validation: run with d_eff fixed at {2, 4, 6} for K and
  {20, 40, 55} for V on Qwen3.5-9B to validate default variance-threshold heuristic
- [ ] Calibrate Qwen3.5-9B (for dev): `calibration/calibration-qwen3.5-9b.safetensors`
- [ ] Calibrate Qwen3.5-27B: `calibration/calibration-qwen3.5-27b.safetensors`
- [ ] Verify calibration time < 30s on RTX 5090 for 27B
- [ ] Document: `calibration/README.md` with regeneration one-liner

### Phase 3: Tests (~10% of effort)

**Files:**
- `tests/test_spectral.py`

**Tasks:**
- [ ] Unit test: `SpectralCalibrator` produces correct d_eff (verify against reference
  SpectralQuant library on same calibration data; cosine similarity of eigenvectors > 0.999)
- [ ] Unit test: `SpectralQuantizer` encode → decode roundtrip on synthetic KV
  (shape [4, 256]) achieves cosine similarity > 0.94 for K and V
- [ ] Unit test: `SpectralQuantizer` encode → decode on real Qwen3.5-9B KV activations
  (collect 10 prompts; check per-layer cosine similarity > 0.94)
- [ ] Unit test: SSM layer skip (pass a non-attention layer; verify no-op)
- [ ] Unit test: first-token edge case (empty cache, d_eff fallback)
- [ ] Integration test: generate 100 tokens with `SpectralKVCache` on Qwen3.5-9B;
  verify output is coherent (no NaN, no repetition loop, top-1 token matches f16 ≥ 80%)
- [ ] Calibration precision test: compare f32 vs f16 PCA eigenvectors;
  eigenvector cosine similarity > 0.999 (validates f32 calibration path)
- [ ] All existing `pytest tests/` pass without regression

### Phase 4: Benchmarks (~25% of effort)

**Files:**
- `scripts/benchmark_spectral.py`
- `docs/BENCHMARK-REPORT.md`
- `docs/QUANTIZATION-GUIDE.md`

**Tasks:**

**Early kill gate** (run on Qwen3.5-9B, ctx=256, before investing in Phase 5):
- [ ] Perplexity: spectral4 vs planar4/planar3/f16 on Qwen3.5-9B
- [ ] If spectral4 PPL > planar3 equivalent on 9B: halt, document findings, treat as
  negative result sprint. Document K/V d_eff distributions and hypothesize why.

**Full benchmarks on Qwen3.5-27B** (proceed only if kill gate passes):
- [ ] Perplexity: spectral4 vs f16/planar4/planar3/iso4/iso3 (wikitext-2, ctx=4096)
- [ ] Long-context perplexity: spectral4 vs planar3 at ctx=16384 (catch codebook drift)
- [ ] Per-layer cosine similarity: K and V separately for all 16 attention layers
- [ ] NIAH recall: spectral4 at ctx=4K and ctx=16K (use `scripts/eval_niah.py`)
- [ ] Compression ratio: bytes stored (K+V together) vs f16 baseline
- [ ] d_eff distribution: report actual d_eff_k and d_eff_v per layer (validate 4-5 / 40-55 claim)
- [ ] Decode throughput: tok/s with spectral4 vs planar4 baseline (target: ≥ 95% of planar4)
- [ ] GPU memory during inference: validate < 8 GB co-occupancy constraint at ctx=4096
- [ ] Update `docs/BENCHMARK-REPORT.md` and `docs/QUANTIZATION-GUIDE.md`

### Phase 5: C/CUDA Integration Design (~20% of effort)

**Files:**
- `docs/SPECTRAL-CUDA-DESIGN.md`

**Tasks:**
- [ ] Audit `ggml/src/ggml.h`: identify `ggml_type` enum extension points and
  `ggml_type_size` / `ggml_blck_size` requirements for a composite spectral type
- [ ] Audit `src/llama-kv-cache.cpp`: trace `cache_type_k` / `cache_type_v` paths
  from CLI arg through `ggml_new_tensor` to understand where quantization hooks live
- [ ] Audit `ggml/src/ggml-cuda/`: identify existing dequantization kernel patterns
  (especially how existing quant types dispatch to CUDA ops)
- [ ] Design `GGML_TYPE_SPECTRAL4` tensor layout:
  - Signal indices (4-bit) + noise indices (2-bit) packed per element
  - Per-layer metadata pointer (eigenvectors + codebooks) stored outside the tensor
  - Memory layout compatible with CUDA coalesced access
- [ ] Design calibration sidecar loading: how `llama_model_load_from_file` discovers
  and loads a `.spectral.safetensors` sidecar alongside the `.gguf` model file
- [ ] Design CUDA kernels:
  - `spectral_encode_kernel`: K/V f16 → project → split → Lloyd-Max lookup → pack
  - `spectral_decode_kernel`: unpack → Lloyd-Max decode → back-project → f16
  - Target: fused single-kernel per-token decode (avoid 16× separate kernel launches)
- [ ] Document how `--cache-type-k spectral4` flows from `docker/entrypoint.sh`
  through llama-server CLI arg parsing to the ggml KV cache allocation
- [ ] Estimate CUDA register pressure and shared memory usage for RTX 5090 (SM 120)
- [ ] Document Sprint 004 scope: what's needed to ship `spectral4` as a llama.cpp
  KV cache type behind `KV_CACHE_TYPE=spectral4`

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `turboquant/spectral/__init__.py` | Create | Package init, public API |
| `turboquant/spectral/calibrator.py` | Create | PCA calibration, d_eff selection, Lloyd-Max fitting (K/V asymmetric) |
| `turboquant/spectral/quantizer.py` | Create | Online encode/decode with asymmetric K/V |
| `turboquant/spectral/store.py` | Create | Safetensors sidecar serialization |
| `turboquant/spectral/kv_cache.py` | Create | `SpectralKVCache` DynamicCache subclass |
| `turboquant/config.py` | Modify | Add `SPECTRAL_4BIT` preset |
| `scripts/calibrate_spectral.py` | Create | Offline calibration CLI |
| `scripts/benchmark_spectral.py` | Create | Full benchmark suite |
| `tests/test_spectral.py` | Create | Unit + integration tests |
| `calibration/calibration-qwen3.5-9b.safetensors` | Create | Qwen3.5-9B calibration (dev) |
| `calibration/calibration-qwen3.5-27b.safetensors` | Create | Qwen3.5-27B calibration (production) |
| `calibration/README.md` | Create | Regeneration instructions |
| `docs/SPECTRAL-CUDA-DESIGN.md` | Create | C/CUDA design doc for Sprint 004 |
| `docs/BENCHMARK-REPORT.md` | Modify | Add spectral4 benchmark column |
| `docs/QUANTIZATION-GUIDE.md` | Modify | Add SpectralQuant calibration section |

## Definition of Done

### Always Required
- [ ] `SpectralCalibrator` produces PCA bases + Lloyd-Max codebooks for all 16
  attention layers of Qwen3.5-27B in < 30s on RTX 5090
- [ ] PCA eigenvectors match reference SpectralQuant library (cosine similarity > 0.999)
  on same calibration data
- [ ] `SpectralKVCache` encode→decode roundtrip cosine similarity > 0.94 per layer
  (validated on real Qwen3.5-9B and Qwen3.5-27B activations)
- [ ] SSM layers correctly skipped (validated by unit test)
- [ ] GQA correctly handled (4 KV heads, head_dim=256 for 27B)
- [ ] All existing `pytest tests/` pass without regression
- [ ] C/CUDA design document written with all audit, layout, and kernel design tasks complete
- [ ] GPU memory during benchmark < 8 GB (co-occupancy safe)

### If Kill Gate Passes (spectral4 PPL ≤ planar3 on 9B early check)
- [ ] Perplexity spectral4 (wikitext-2, ctx=4096, Qwen3.5-27B) < 8.20 (beats planar3)
- [ ] Compression ratio ≥ 4.9x vs f16 KV cache
- [ ] Decode throughput ≥ 95% of planar4 tok/s baseline
- [ ] NIAH recall at ctx=4K and ctx=16K: spectral4 ≥ planar3
- [ ] Long-context PPL at ctx=16K measured (catch codebook drift)
- [ ] d_eff distribution documented per layer
- [ ] `docs/BENCHMARK-REPORT.md` updated with full comparison table

### If Kill Gate Fails (spectral4 PPL > planar3 on 9B)
- [ ] Research report written: per-layer cosine similarity, d_eff distributions,
  hypothesis for why SpectralQuant doesn't win on Qwen3.5 architecture
- [ ] `planar4` confirmed as default; no change to entrypoint or compose
- [ ] C/CUDA design document still written (may still be worth implementing for completeness)

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SpectralQuant PPL doesn't beat planar3 | Medium | High | Kill gate after Phase 3 on Qwen3.5-9B; document and stop per user decision |
| V channel d_eff too high for meaningful compression | Medium | High | Sweep d_eff_v ∈ {20, 40, 55} in Phase 2; if d_eff_v > 100 the approach has no advantage |
| PCA eigenvalue gap is ambiguous for some layers | Medium | Medium | Implement fallback: if flat eigenspectrum, use full-dim Lloyd-Max for that layer |
| GPU OOM during calibration (co-occupied) | Medium | Medium | batch_size=1, ctx=256, torch.no_grad(); use 9B for dev; 27B only when GPU free |
| Decode throughput < 95% of planar4 | Medium | High | Profile PCA projection cost early (Phase 3); if too slow, consider cached projection |
| Calibration corpus distribution shift | Low | Medium | Test cosine similarity on code + multilingual prompts; document sensitivity |
| PCA numerical instability at f32 (near-degenerate covariance) | Low | Medium | Use `torch.linalg.eigh` (symmetric eigendecomposition, more stable than SVD); validate vs f64 |
| C/CUDA design doc too thin for Sprint 004 | Medium | High | 20% effort allocation; critique specifically called this out — produce actionable doc |

## Security Considerations

- Calibration sidecar files are loaded at inference startup from disk. Use safetensors
  format (no pickle, no arbitrary code execution, built-in integrity validation).
- Validate tensor shapes against expected model architecture before using (check
  `[n_kv_heads, head_dim, d_eff]` matches model config); reject malformed sidecars.
- Do not use private prompts for calibration corpus — use public wikitext-2 only.
  KV activation statistics reveal information about calibration data.
- Sidecar file path is constructed from model name (user-controlled via `MODEL_NAME`
  env var); sanitize the path to prevent directory traversal.

## Dependencies

- **Sprint 001**: `turboquant/` library structure (`DynamicCache` subclass pattern,
  `BitConfig`, `tests/` structure)
- **Sprint 002**: Docker benchmark infrastructure, `llama-perplexity` binary,
  `scripts/eval_niah.py`, `scripts/eval_perplexity.py`
- **External Python**: `safetensors`, `transformers`, `torch` (all installed),
  `scipy` (Lloyd-Max via `scipy.cluster.vq.kmeans`)
- **Reference**: Dynamis-Labs/spectralquant (algorithm reference only, not a
  runtime dependency)

## Open Questions

1. **d_eff selection threshold**: 99% variance explained is a starting point.
   The Phase 2 d_eff sweep will determine if this is appropriate for Qwen3.5.

2. **K vs V bit allocation**: 4-bit signal + 2-bit noise is the default.
   If V needs d_eff=50 of 256 dims with 4-bit, the compression ratio drops significantly.
   Consider 3-bit noise for V to recover compression headroom.

3. **Hybrid SpectralQuant + Clifford rotor** (deferred): Apply a Clifford rotor
   within the d_eff signal subspace after PCA. Near-free at d_eff=4 (K), potentially
   useful. Sprint 004 experiment.

4. **SPECTRAL_3BIT variant** (deferred to Sprint 004): 3-bit signal / 1-bit noise.
   Potentially matches or beats planar3 at lower compression, as planar benchmarks showed.

5. **Qwen3.6-35B-A3B support** (deferred to Sprint 004): MoE architecture has different
   attention layer distribution. Validate on 27B first.
