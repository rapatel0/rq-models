# Sprint 003: SpectralQuant KV Cache Integration

> **Author**: Codex-substitute draft (gpt-5.4 unavailable — authored by orchestrator)

## Overview

SpectralQuant (Dynamis Labs) introduces a fundamentally different approach to KV
cache compression: rather than applying a uniform rotation before quantization
(TurboQuant/RotorQuant/PlanarQuant), it exploits the natural spectral asymmetry
of transformer KV caches — keys concentrate ~96-97% of their signal in just 3-4%
of head dimensions, while values require 40-55 signal dimensions. By using PCA
calibration to identify these signal dimensions per layer, SpectralQuant selectively
applies high-precision (Lloyd-Max) quantization to the signal subspace and
aggressively discards the noise subspace.

This sprint delivers SpectralQuant integration in two phases: (1) a validated Python
prototype in `turboquant/` that proves quality gains on Qwen3.5-27B, with measured
perplexity, cosine similarity, and compression ratio; (2) a C/CUDA integration design
document for the llama.cpp fork that maps the Python algorithm to the existing
`ggml_type` / KV cache quantization hook architecture. Full C/CUDA integration is a
stretch goal, deferred to Sprint 004 if the prototype phase reveals unforeseen
complexity.

The sprint is motivated by benchmarked evidence that PlanarQuant already beats
IsoQuant significantly (planar3 PPL 8.20 vs iso4 PPL 8.37, 4K context, Qwen3.5-27B).
SpectralQuant claims a further improvement: cosine similarity 0.9485 vs TurboQuant's
0.9226, and 5.95x compression vs PlanarQuant's ~4.9x. If validated on our hardware
and model, it warrants becoming the new default.

## Use Cases

1. **Higher quality at current context**: Users running Qwen3.5-27B at 112K tokens
   (RTX 4090) or 245K tokens (RTX 5090) get measurably better output quality at the
   same VRAM — SpectralQuant's targeted compression preserves signal dims that
   planar3 discards.

2. **More context at current quality**: SpectralQuant's higher compression ratio
   (~5.95x vs planar4's 3.8x) frees VRAM headroom for longer context windows without
   quality regression — e.g., 330K → 400K context on RTX 5090 at iso-quality.

3. **Calibration-free variants**: Users serving multiple models can pre-compute
   SpectralQuant calibration (15s) at startup from a small wikitext sample, storing
   per-layer PCA bases as a sidecar file. Subsequent restarts load the sidecar
   instantly.

4. **Research baseline**: Having both PlanarQuant and SpectralQuant in the same
   benchmark harness establishes a reproducible comparison baseline for future
   KV cache compression research on this hardware.

## Architecture

```
Calibration path (offline, ~15s):
  calibration_corpus → [tokenize] → [forward pass, collect KV activations]
      → [PCA per layer per head] → d_eff, V_signal (eigenvectors)
      → [Lloyd-Max codebook fit on projected signal dims]
      → serialize → spectral_calibration_{model}.bin

Inference path (per token):
  raw KV (f16) [L, H, d]
      → [project onto V_signal] → KV_signal [L, H, d_eff]   (d_eff ≈ 4-5)
      → [Lloyd-Max encode] → quantized signal  (4-bit, ~d_eff * 4 bits)
      → [1-bit or zero fill] → noise dims      (negligible)
      → store: [signal codebook indices | noise placeholder]

Dequantization path (attention compute):
  → [Lloyd-Max decode signal] → KV_signal_hat
  → [back-project via V_signal^T] → KV_hat (approximate f16)
  → feed into standard attention
```

**Key structures**:
- `SpectralCalibration`: per-layer dataclass holding `V_signal` (eigenvectors,
  shape `[n_heads, d, d_eff]`), `codebooks` (Lloyd-Max, shape `[n_heads, 2^b, d_eff]`),
  `d_eff` (int per head)
- `SpectralKVCache`: `DynamicCache` subclass; stores compressed KV per layer

**SSM layer handling**: Qwen3.5-27B has 48/64 SSM (Mamba) layers with no KV cache.
Calibration skips layers where `layer.is_attention == False`. The cache falls back
to f16 for SSM recurrent state (unchanged from current behaviour).

**GQA handling**: 4 KV heads, head_dim=256. PCA is computed per KV head. d_eff
per head is determined by explaining 99% of variance in calibration data.

## Implementation

### Phase 1: Python Prototype (~45% of effort)

**Files:**
- `turboquant/spectral.py` — New module: `SpectralCalibration`, `SpectralEncoder`,
  `SpectralDecoder`
- `turboquant/spectral_kv_cache.py` — `SpectralKVCache(DynamicCache)` subclass
- `turboquant/config.py` — Add `SPECTRAL_4BIT` preset
- `scripts/calibrate_spectral.py` — CLI: runs forward pass on wikitext-2 sample,
  fits PCA + Lloyd-Max per layer, saves `.bin` sidecar
- `scripts/eval_spectral_ppl.py` — Perplexity eval using `SpectralKVCache`;
  differential cosine-similarity logging per layer

**Tasks:**
- [ ] Implement `SpectralCalibration.from_activations(kv_activations, variance_threshold=0.99)`
- [ ] Implement Lloyd-Max codebook fitting (`n_iter=100`, `n_centroids=16` for 4-bit)
- [ ] Implement `SpectralEncoder.encode(kv)` → quantized indices + residual flag
- [ ] Implement `SpectralDecoder.decode(indices)` → reconstructed KV (f16)
- [ ] Implement `SpectralKVCache.update()` to call encode on new tokens
- [ ] Implement `SpectralKVCache.get_seq_length()` and standard DynamicCache API
- [ ] Calibration script: collect activations from 512-token wikitext-2 sample
- [ ] Validate: cosine similarity vs f16 KV on Qwen3.5-27B, target > 0.94 per layer
- [ ] Perplexity eval: wikitext-2, ctx=4096, compare against planar3 baseline (8.20)

### Phase 2: llama.cpp Integration Design (~30% of effort)

**Files:**
- `docs/spectralquant-integration.md` — Full design document for C/CUDA integration
- `turboquant/spectral_ggml.py` — Python script to export calibration as GGUF
  metadata (key-value pairs in the model file or sidecar `.gguf`)

**Tasks:**
- [ ] Audit `ggml/src/ggml.h` and `ggml/src/ggml-cuda/` for existing KV quant hooks
- [ ] Design new `GGML_TYPE_SPECTRAL4` tensor type: layout spec (signal indices +
  d_eff metadata per layer)
- [ ] Design calibration sidecar: how to embed PCA bases and codebooks in a
  `.gguf` sidecar loaded by `llama_model_load`
- [ ] Design CUDA kernel: `spectral_encode_kernel` (project → Lloyd-Max lookup) and
  `spectral_decode_kernel` (reverse lookup → back-project)
- [ ] Document: how `--cache-type-k spectral4` flows from entrypoint.sh through
  llama-server to the GGML ops

### Phase 3: Benchmarks & Documentation (~25% of effort)

**Files:**
- `docs/BENCHMARK-REPORT.md` — Add SpectralQuant column to PPL table
- `docs/QUANTIZATION-GUIDE.md` — Add SpectralQuant section with calibration instructions
- `README.md` — Update default recommendation if SpectralQuant wins on PPL

**Tasks:**
- [ ] Run full perplexity sweep: f16, planar4, planar3, spectral4 on Qwen3.5-27B ctx=4096
- [ ] Measure calibration time on RTX 5090 (target: <30s)
- [ ] Measure VRAM usage at 4096 context: compare spectral4 vs planar3
- [ ] Measure cosine similarity per attention layer (16 layers for Qwen3.5-27B)
- [ ] Update benchmark docs with all measured values

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `turboquant/spectral.py` | Create | Core SpectralQuant calibration, encode, decode |
| `turboquant/spectral_kv_cache.py` | Create | HuggingFace DynamicCache subclass |
| `turboquant/config.py` | Modify | Add SPECTRAL_4BIT preset |
| `scripts/calibrate_spectral.py` | Create | Offline calibration CLI |
| `scripts/eval_spectral_ppl.py` | Create | Perplexity eval with SpectralKVCache |
| `turboquant/spectral_ggml.py` | Create | Export calibration to GGUF sidecar |
| `docs/spectralquant-integration.md` | Create | C/CUDA integration design doc |
| `docs/BENCHMARK-REPORT.md` | Modify | Add SpectralQuant benchmark column |
| `docs/QUANTIZATION-GUIDE.md` | Modify | Add calibration instructions |
| `README.md` | Modify | Update default if SpectralQuant wins |

## Definition of Done

- [ ] `SpectralKVCache` passes unit tests: encode → decode round-trip cosine similarity
  > 0.94 on random KV tensors (shape matching Qwen3.5-27B heads)
- [ ] Calibration script completes in < 30s on RTX 5090 with Qwen3.5-27B
- [ ] Perplexity measured: spectral4 PPL on wikitext-2 ctx=4096 < 8.20 (beats planar3)
- [ ] Compression ratio ≥ 4.9x vs f16 KV (matches or beats planar3)
- [ ] SSM layers are correctly skipped (no KV cache corruption on Qwen3.5-27B)
- [ ] GQA (4 KV heads, head_dim=256) handled correctly
- [ ] C/CUDA integration design document written and reviewed
- [ ] Benchmark docs updated with measured values
- [ ] No regression: planar4 / planar3 / iso4 / iso3 / f16 still work unchanged

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SpectralQuant PPL doesn't beat planar3 on our models | Medium | High | Still document results; investigate why (model-specific spectral structure?) |
| PCA variance threshold d_eff too small → attention degradation | Medium | High | Sweep variance threshold 0.95–0.999; validate with NIAH recall test |
| Calibration GPU memory exceeds budget during training | Medium | Medium | Run calibration with batch_size=1, ctx=256; use Qwen3.5-9B for dev |
| GGUF sidecar format incompatible with llama.cpp model loading | Low | High | Fall back to runtime calibration at server startup |
| Qwen3.6-35B-A3B MoE attention architecture incompatible | Medium | Medium | Defer Qwen3.6 support to Sprint 004; validate on 27B first |
| Lloyd-Max codebook training diverges at low d_eff | Low | Medium | Use k-means initialisation; fallback to uniform quantization if diverges |

## Security Considerations

- Calibration sidecar files are loaded at server startup; validate file format before
  parsing to prevent malformed sidecar attacks (path traversal, buffer overflow).
- SpectralQuant calibration exposes KV activation statistics; do not use private
  data for calibration — use public wikitext-2 corpus.

## Dependencies

- Sprint 001: `turboquant/` Python library (TurboQuantMSE, DynamicCache subclass) —
  SpectralKVCache extends the same pattern
- Sprint 002: Docker build infrastructure, `llama-perplexity` binary in container
- External: `numpy`, `scipy` (already available), `transformers` (already installed)
- External reference: Dynamis-Labs/spectralquant for algorithm validation

## Open Questions

1. Should calibration be run once at Docker build time (baked into the image) or
   at container startup (more flexible, adds ~30s cold start)?
2. What variance threshold determines d_eff? 99% is a reasonable starting point
   but may be too aggressive for some layers.
3. Does SpectralQuant's key insight (keys concentrate in 4 dims) hold for Qwen3.5's
   hybrid SSM+attention architecture, which has only 16 attention layers at
   different depths than a pure transformer?
4. Should Sprint 003 scope in a `spectral3` (3-bit signal dims) variant to compare
   against planar3, or keep scope to 4-bit only?
