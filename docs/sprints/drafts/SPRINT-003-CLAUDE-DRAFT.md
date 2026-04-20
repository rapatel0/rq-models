# Sprint 003: SpectralQuant KV Cache Integration

## Overview

The current KV cache quantization backends (planar3/planar4/iso3/iso4) use data-oblivious random rotations — every dimension is treated equally regardless of how much signal it carries. SpectralQuant takes a fundamentally different approach: PCA calibration identifies the ~4% of head dimensions that carry meaningful signal (d_eff ≈ 4-5 per layer), then applies Lloyd-Max nonuniform quantization at higher precision to those signal dimensions while aggressively compressing the noise dimensions. This data-aware strategy should yield better perplexity at equal or higher compression ratios.

This sprint delivers a validated Python prototype of SpectralQuant integrated into the `turboquant/` library, with perplexity benchmarks proving it beats planar3's 8.20 PPL on Qwen3.5-27B. C/CUDA llama.cpp integration is designed but deferred to Sprint 004 — the scope and architectural uncertainty of a new ggml tensor type is too high for one sprint.

The guiding decision is **Python-first**: fast iteration on quality validation, with a clear handoff artifact (design doc + serialization format) for the C/CUDA sprint.

## Use Cases

1. **Quality-sensitive long-context inference**: Users running Qwen3.5-27B at 100K+ tokens on RTX 5090 want the lowest possible perplexity degradation from KV compression. SpectralQuant targets <8.20 PPL (beating planar3) at ≥4.9x compression.

2. **Research validation pipeline**: Researchers comparing KV cache quantization techniques need an apples-to-apples benchmark harness. This sprint extends the existing `turboquant/` library with a SpectralQuant backend and adds a unified comparison script.

3. **Calibration-guided deployment**: Operators deploying to specific models want per-model calibration data that optimally tunes quantization parameters, rather than one-size-fits-all codebooks. SpectralQuant's PCA bases and Lloyd-Max codebooks are fitted per model per layer.

4. **Hybrid SpectralQuant + PlanarQuant exploration**: Researchers want to test applying Clifford rotors within the d_eff signal subspace after PCA projection, potentially combining the best of both approaches.

## Architecture

### Data Flow

```
Calibration (offline, ~15-30s per model):
  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
  │ Calibration  │───▶│ PCA per      │───▶│ Lloyd-Max fit   │
  │ prompts      │    │ layer/head   │    │ per layer       │
  └─────────────┘    └──────────────┘    └─────────────────┘
                           │                      │
                           ▼                      ▼
                     eigenvectors            codebooks
                     + eigenvalues          (signal/noise)
                           │                      │
                           └──────┬───────────────┘
                                  ▼
                          calibration.safetensors
                          (per-model sidecar file)

Inference (online):
  ┌─────────┐    ┌──────────┐    ┌───────────────┐    ┌──────────┐
  │ KV token │───▶│ PCA      │───▶│ Quantize:     │───▶│ Store    │
  │ (f16)    │    │ project  │    │ signal→LM-4b  │    │ compact  │
  └─────────┘    └──────────┘    │ noise →LM-2b  │    └──────────┘
                                  └───────────────┘
  ┌──────────┐    ┌──────────┐    ┌───────────────┐    ┌──────────┐
  │ Retrieve │───▶│ Dequant  │───▶│ PCA inverse   │───▶│ f16 KV   │
  │ compact  │    │ signal+  │    │ project       │    │ approx   │
  └──────────┘    │ noise    │    └───────────────┘    └──────────┘
                  └──────────┘
```

### Key Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `SpectralCalibrator` | `turboquant/spectral/calibrator.py` | PCA eigendecomposition per layer; determines d_eff; fits Lloyd-Max codebooks for signal and noise subspaces |
| `SpectralQuantizer` | `turboquant/spectral/quantizer.py` | Online encode/decode: PCA projection → split signal/noise → nonuniform quantize → pack |
| `SpectralKVCache` | `turboquant/spectral/kv_cache.py` | `DynamicCache` subclass (like `TurboKVCache`) using `SpectralQuantizer` for compress/decompress |
| `CalibrationStore` | `turboquant/spectral/store.py` | Serialize/deserialize PCA bases + codebooks to safetensors sidecar |
| Benchmark scripts | `scripts/benchmark_spectral.py` | PPL, cosine similarity, compression ratio, latency comparisons |

### SSM Layer Handling

Qwen3.5-27B has 48/64 SSM layers with no KV cache. The calibrator must detect attention-vs-SSM layers. Strategy: introspect the model config for `attention_layers` list (Qwen3.5 exposes this), or detect by presence of `k_proj`/`v_proj` parameters. SSM layers are skipped during calibration and at runtime — `SpectralKVCache.update()` passes them through without quantization.

### Calibration Persistence

PCA bases and Lloyd-Max codebooks are stored as a **safetensors sidecar file** (one per model). This avoids modifying GGUF format (which would require llama.cpp changes) and is compatible with both the Python prototype and future C/CUDA integration. Format:

```
calibration.safetensors:
  layer_0_eigenvectors: [head_dim, head_dim]  # PCA basis (float16)
  layer_0_eigenvalues:  [head_dim]            # for d_eff selection
  layer_0_d_eff:        scalar                # effective dimensions
  layer_0_signal_codebook: [2^b_signal]       # Lloyd-Max centroids
  layer_0_noise_codebook:  [2^b_noise]        # Lloyd-Max centroids
  ...per attention layer...
```

## Implementation

### Phase 1: SpectralQuant Core (~35% of effort)

**Files:**
- `turboquant/spectral/__init__.py` — Package init
- `turboquant/spectral/calibrator.py` — PCA calibration engine
- `turboquant/spectral/quantizer.py` — Encode/decode with signal/noise split
- `turboquant/spectral/store.py` — Safetensors serialization

**Tasks:**
- [ ] Implement `SpectralCalibrator`: run calibration prompts through model, collect KV activations per attention layer, compute PCA, determine d_eff via eigenvalue gap/ratio heuristic, fit Lloyd-Max codebooks for signal dims (4-bit) and noise dims (2-bit)
- [ ] Implement `SpectralQuantizer.encode()`: project KV vector via PCA basis, split into signal (top d_eff dims) and noise (remaining dims), quantize each with respective Lloyd-Max codebook, pack into compact representation
- [ ] Implement `SpectralQuantizer.decode()`: unpack, dequantize signal and noise, recombine, inverse PCA projection
- [ ] Implement `CalibrationStore`: save/load calibration artifacts as safetensors with layer-indexed keys
- [ ] Validate PCA eigenvectors match SpectralQuant reference implementation on same calibration data
- [ ] Unit tests for encode→decode roundtrip: verify cosine similarity >0.94 on synthetic and real KV activations

### Phase 2: KV Cache Integration (~25% of effort)

**Files:**
- `turboquant/spectral/kv_cache.py` — `DynamicCache` subclass
- `turboquant/config.py` — Add SpectralQuant presets

**Tasks:**
- [ ] Implement `SpectralKVCache(DynamicCache)` following the same interface as `TurboKVCache`: override `update()` to quantize, override cache properties to dequantize on retrieval
- [ ] Handle SSM layer detection: skip quantization for non-attention layers
- [ ] Handle GQA: quantize at `num_key_value_heads` granularity (4 KV heads, head_dim=256 for 27B)
- [ ] Handle edge cases: first token (empty cache), very short context (fewer tokens than d_eff)
- [ ] Add `SPECTRAL_4BIT` and `SPECTRAL_3BIT` presets to `BitConfig` (4-bit signal + 2-bit noise = ~2.5 effective bits; 3-bit signal + 1-bit noise = ~1.5 effective bits)
- [ ] Integration test: generate 100 tokens with `SpectralKVCache` on Qwen3.5-9B, verify output is coherent

### Phase 3: Calibration Pipeline (~15% of effort)

**Files:**
- `scripts/calibrate_spectral.py` — CLI calibration script
- `turboquant/spectral/calibrator.py` — Calibration data collection

**Tasks:**
- [ ] Build calibration script: load model, run N calibration prompts (wikitext-2 subset), collect KV activations, run `SpectralCalibrator`, save sidecar
- [ ] Calibrate Qwen3.5-27B: produce `calibration-qwen3.5-27b.safetensors`
- [ ] Calibrate Qwen3.5-9B: produce calibration for fast iteration
- [ ] Verify calibration completes in <30s on RTX 5090 for Qwen3.5-27B
- [ ] Verify d_eff values are reasonable (expect 4-5 per layer based on SpectralQuant paper)

### Phase 4: Benchmarking & Comparison (~20% of effort)

**Files:**
- `scripts/benchmark_spectral.py` — Comprehensive benchmark script
- `docs/BENCHMARK-REPORT.md` — Updated with SpectralQuant results

**Tasks:**
- [ ] Measure perplexity on wikitext-2 (ctx=4096) for SpectralQuant vs planar3/planar4/iso3/iso4/f16 on Qwen3.5-27B
- [ ] Measure per-layer cosine similarity (SpectralQuant KV vs f16 KV) on 100-token prompts
- [ ] Measure compression ratio (bytes stored vs f16 baseline)
- [ ] Measure encode/decode latency per token (target: <1ms per token for SpectralQuant)
- [ ] Run benchmarks on Qwen3.5-9B as fast validation (GPU-memory conscious: ctx=256 for iterative runs)
- [ ] Update `docs/BENCHMARK-REPORT.md` with results table

### Phase 5: C/CUDA Integration Design (~5% of effort)

**Files:**
- `docs/SPECTRAL-CUDA-DESIGN.md` — Design document for Sprint 004

**Tasks:**
- [ ] Document the ggml tensor type design: composite layout for signal + noise subspaces with per-layer PCA bases
- [ ] Document calibration sidecar loading in llama.cpp (GGUF metadata pointer to sidecar, or embed in GGUF)
- [ ] Document CUDA kernel design: fused PCA project → split → dequant kernel
- [ ] Identify required llama.cpp fork changes: `ggml.c` type registration, `llama-kv-cache.cpp` hooks, `CMakeLists.txt`

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `turboquant/spectral/__init__.py` | Create | Package init, public API exports |
| `turboquant/spectral/calibrator.py` | Create | PCA calibration + Lloyd-Max codebook fitting |
| `turboquant/spectral/quantizer.py` | Create | Online encode/decode with signal/noise split |
| `turboquant/spectral/kv_cache.py` | Create | `SpectralKVCache` DynamicCache subclass |
| `turboquant/spectral/store.py` | Create | Safetensors sidecar serialization |
| `turboquant/config.py` | Modify | Add `SPECTRAL_4BIT` / `SPECTRAL_3BIT` presets |
| `scripts/calibrate_spectral.py` | Create | CLI calibration pipeline |
| `scripts/benchmark_spectral.py` | Create | PPL / cosine sim / compression benchmarks |
| `tests/test_spectral.py` | Create | Unit + integration tests for spectral module |
| `docs/BENCHMARK-REPORT.md` | Modify | Add SpectralQuant benchmark results |
| `docs/SPECTRAL-CUDA-DESIGN.md` | Create | C/CUDA integration design for Sprint 004 |

## Definition of Done

- [ ] `SpectralCalibrator` produces PCA bases and Lloyd-Max codebooks for all 16 attention layers of Qwen3.5-27B in <30s
- [ ] PCA eigenvectors match SpectralQuant reference implementation (cosine similarity >0.999 on same calibration data)
- [ ] `SpectralKVCache` encode→decode roundtrip achieves cosine similarity >0.94 per layer on Qwen3.5-27B
- [ ] Perplexity (wikitext-2, ctx=4096, Qwen3.5-27B) is lower than planar3's 8.20
- [ ] Compression ratio ≥4.9x vs f16 KV cache
- [ ] Encode + decode latency <1ms per token on RTX 5090
- [ ] SSM layers correctly skipped (no quantization applied to non-attention layers)
- [ ] GQA handled correctly (4 KV heads, head_dim=256)
- [ ] All existing tests pass (`pytest tests/`)
- [ ] No regressions to existing KV cache types (planar4/planar3/iso4/iso3/f16)
- [ ] Calibration sidecar files committed for Qwen3.5-27B and Qwen3.5-9B
- [ ] C/CUDA design document written and reviewed
- [ ] Benchmark results added to `docs/BENCHMARK-REPORT.md`

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SpectralQuant PPL doesn't beat planar3 on Qwen3.5 architecture | Medium | High | Run Qwen3.5-9B benchmarks early (Phase 1) as a fast signal; if PPL is worse, pivot to hybrid SpectralQuant+PlanarQuant approach (PCA projection + Clifford rotors in signal subspace) |
| PCA calibration is too slow (>30s) at head_dim=256 | Low | Medium | PCA on 256-dim vectors is cheap; if slow, subsample calibration tokens or use randomized SVD |
| SpectralQuant reference library has bugs or undocumented assumptions | Medium | Medium | Validate against first principles (PCA eigendecomposition is well-understood); don't blindly copy — reimplement from the math |
| Lloyd-Max codebook fitting diverges for noise dimensions | Low | Medium | Use robust initialization (uniform quantile init); fall back to uniform quantization for noise dims if needed |
| GPU OOM during calibration (training job co-occupying GPU) | Medium | Low | Use Qwen3.5-9B for iterative work; run Qwen3.5-27B calibration with `torch.no_grad()` and f16 precision; limit calibration batch to 4-8 sequences |
| Scope creep into C/CUDA integration | Medium | High | Hard scope boundary: Python prototype + design doc only. C/CUDA is Sprint 004. |
| Calibration data overfits to wikitext-2 | Low | Medium | Use diverse calibration corpus (mix of wikitext, code, conversation); validate PPL on held-out data |

## Security Considerations

- Calibration sidecar files (safetensors) are model-specific artifacts loaded at inference time. Safetensors format has built-in integrity validation (no pickle, no arbitrary code execution).
- No new network-facing code. The `SpectralKVCache` runs entirely within the inference pipeline.

## Dependencies

- **Prior work**: Sprint 001 (`turboquant/` library structure, `TurboKVCache`, `BitConfig`), Sprint 002 (Docker deployment, benchmark methodology)
- **External**: `safetensors` Python library (for calibration persistence), `transformers` (for model loading and `DynamicCache`), `torch` (PCA via `torch.linalg.eigh`)
- **Reference**: SpectralQuant paper/library from Dynamis Labs (algorithm reference, not a runtime dependency)
- **Hardware**: RTX 5090 (32 GB) for Qwen3.5-27B calibration and benchmarks; Qwen3.5-9B for iterative development

## Open Questions

1. **d_eff selection heuristic**: SpectralQuant uses an eigenvalue gap/ratio to determine d_eff per layer. What threshold? The paper suggests d_eff ≈ 4-5 for Qwen3.5, but this should be validated empirically. Options: fixed percentile of variance explained (e.g., 90%), eigenvalue ratio threshold, or the reference library's heuristic.

2. **Signal bit depth vs noise bit depth**: The default split is 4-bit signal / 2-bit noise, but this is tunable. Should we sweep bit allocations (e.g., 3/1, 4/2, 5/2) to find the Pareto frontier for Qwen3.5?

3. **K vs V asymmetric quantization**: TurboQuant uses different algorithms for K (inner-product preserving) and V (MSE). Should SpectralQuant also apply different quantization strategies to K and V, or does PCA-based projection make this unnecessary?

4. **Hybrid SpectralQuant + PlanarQuant**: Applying a Clifford rotor within the d_eff signal subspace after PCA projection is computationally cheap (d_eff=4 → trivial rotor). This could improve Lloyd-Max accuracy in the signal subspace. Worth testing but not blocking — treat as an optional Phase 4 experiment.

5. **Qwen3.6-35B-A3B support**: The MoE model has a different attention architecture. Should calibration target both models, or is Qwen3.5-27B sufficient for validation? Recommendation: Qwen3.5-27B only for Sprint 003; Qwen3.6 in Sprint 004 alongside C/CUDA integration.

6. **Calibration corpus sensitivity**: How sensitive is SpectralQuant to the choice of calibration data? If PCA bases shift significantly with different corpora, the approach may be fragile. Mitigate by testing with 2-3 different calibration sets.
