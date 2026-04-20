# Sprint 003 Intent: SpectralQuant KV Cache Integration

## Seed

Implement SpectralQuant (https://github.com/Dynamis-Labs/spectralquant) as a new
KV cache quantization backend in the RotorQuant llama.cpp fork, benchmark it against
the existing planar3/planar4/iso3/iso4 backends, and determine whether it should
become the new default.

## Context

- **Current state**: Production Docker image serving Qwen3.5-27B and Qwen3.6-35B-A3B
  via llama.cpp RotorQuant fork (`johndpope/llama-cpp-turboquant`, branch
  `feature/planarquant-kv-cache`, commit `20efe75`). Default KV cache is `planar4`
  (switched from `iso4` on 2026-04-20 after measured PPL benchmarks).
- **Perplexity results (measured 2026-04-20)**: planar3=8.20, planar4=8.25, iso4=8.37,
  iso3=8.43 on Qwen3.5-27B Q4_K_M, wikitext-2, ctx=4096. PlanarQuant consistently
  beats IsoQuant; planar3 beats iso4 at lower bit depth.
- **SpectralQuant**: Python research library (Dynamis Labs). Uses PCA eigenspectral
  calibration (~15s) to identify ~4% of head dims that carry signal (d_eff ≈ 4-5
  per layer), then applies Lloyd-Max nonuniform quantization to signal dims and
  aggressively compresses noise dims. Claims 5.95x compression and cosine similarity
  0.9485 vs TurboQuant's 0.9226.
- **Architecture**: Hybrid SSM+attention model (Qwen3.5 has 16/64 full-attention
  layers; Qwen3.6-35B-A3B is MoE). SpectralQuant must handle GQA (4 KV heads,
  head_dim=256 for 27B).
- **GPU**: RTX 5090 (32 GB VRAM). Training job may co-occupy GPU; benchmarks must
  be GPU-memory conscious.

## Recent Sprint Context

- **Sprint 001**: Implemented TurboQuant in Python (turboquant/ library), validated
  on Qwen3.5-27B. Follow-up items F-001–F-005 remain (throughput >1ms, d=64
  missing from codebook script, K bias at 2.5-bit).
- **Sprint 002**: Containerised RotorQuant llama.cpp fork into production Docker image.
  Added 16GB/24GB/32GB/40GB GPU profiles, throughput mode (parallel slots), Gemma 4,
  perplexity benchmarks.
- **2026-04-20 (this session)**: Benchmarked planar3/planar4/iso3/iso4 on Qwen3.5-9B
  and Qwen3.5-27B. Switched default from iso4→planar4. Discussed SpectralQuant and
  combining it with Clifford rotors (hybrid approach).
- **Deferred from Sprint 002**: D-011 (Open WebUI), D-013 (automated benchmark CI),
  D-010 (multi-GPU) — these are Sprint 003 candidates but lower priority than
  SpectralQuant.

## Vision Context

No vision document — planning from scratch. The implied north star is: best
possible KV cache quality-per-bit on consumer GPUs (RTX 5090 / 4090 / 5060),
enabling large context (112K–260K tokens) for Qwen3.5-27B and Qwen3.6-35B-A3B.

## Relevant Codebase Areas

**Python research library (turboquant/):**
- `turboquant/kv_cache.py` — HuggingFace DynamicCache subclass; where SpectralQuant
  calibration and encode/decode would live for Python validation
- `turboquant/core.py` — TurboQuantMSE, TurboQuantProd; SpectralQuant's PCA +
  Lloyd-Max replaces these
- `turboquant/codebook.py` — codebook storage; SpectralQuant uses per-layer
  Lloyd-Max codebooks fitted on calibration data

**llama.cpp fork (C/CUDA):**
- KV cache quantization hooks: `src/llama-kv-cache.cpp`, `ggml/src/ggml-cuda/`
- Existing quant types: iso3/iso4/planar3/planar4 registered in `ggml/src/ggml.c`
- Build system: `CMakeLists.txt` — adding new CUDA kernel requires registering here

**Docker / deployment:**
- `docker/entrypoint.sh` — `KV_CACHE_TYPE` env var; add `spectral4`
- `docker-compose.yml` — profile-based configuration
- `docker/Dockerfile` — CUDA build; any new kernel compiles here

**SpectralQuant reference:**
- https://github.com/Dynamis-Labs/spectralquant
- Key modules: PCA calibration, spectral rotation, Lloyd-Max codebooks,
  selective QJL error correction, SpectralQuantEngine

## Constraints

- GPU memory: Must stay below ~8 GB during benchmarks (training co-occupies GPU)
- Use smaller model (Qwen3.5-9B) or ctx=256 for iterative benchmarks
- llama.cpp fork: all C/CUDA changes must compile against CUDA 13.1 on SM 120
  (RTX 5090 Blackwell)
- No HTTPS in llama.cpp build (SSL not compiled in) — model downloads must use
  `hf` CLI, not `--hf-repo` flag
- Existing KV cache types (planar4, planar3, iso4, iso3, f16) must continue to work
- Docker multi-stage build pattern must be preserved

## Success Criteria

1. SpectralQuant calibration runs in <30s on Qwen3.5-27B (Python prototype)
2. Measured perplexity (wikitext-2, ctx=4096, Qwen3.5-27B) lower than planar3's
   8.20 — i.e., SpectralQuant beats the current best
3. Compression ratio ≥ planar3 (≥4.9x vs f16 KV cache)
4. Decode speed ≥ 95% of f16 baseline (~65.8 tok/s)
5. Either (a) integrated into llama.cpp fork as a new `spectral4` KV cache type,
   OR (b) if C/CUDA integration is too large for one sprint, a validated Python
   prototype with a clear integration design document

## Verification Strategy

- **Reference implementation**: Dynamis-Labs/spectralquant Python library — verify
  our calibration produces matching PCA eigenvalues/eigenvectors
- **Differential testing**: Compare SpectralQuant-compressed KV vs f16 KV on
  identical prompts; measure cosine similarity per layer (target: >0.94)
- **Perplexity**: wikitext-2-raw via `llama-perplexity` binary (same methodology
  as prior benchmarks) — apples-to-apples comparison
- **Edge cases**: GQA (n_kv_heads=4 for 27B), SSM layers (must be skipped),
  very short context (d_eff selection when context < calibration data), first
  token (empty KV cache)

## Uncertainty Assessment

- **Correctness**: High — SpectralQuant is a research library with no llama.cpp
  integration. PCA calibration format, codebook serialization, and CUDA kernel
  design are all open questions.
- **Scope**: High — C/CUDA integration may be a full sprint on its own. Python
  prototype is achievable; production integration may need Sprint 004.
- **Architecture**: High — SpectralQuant's selective compression (signal dims vs
  noise dims) requires a fundamentally different KV cache layout than iso/planar
  (which use uniform element-wise quantization). This means a new ggml tensor type
  or a composite layout.

## Open Questions

1. **Python-first or C-first?** Should Sprint 003 deliver a validated Python
   prototype in `turboquant/` (fast, validates quality) or aim for C/CUDA
   llama.cpp integration (production-ready but higher risk)?
2. **Calibration persistence**: How do we serialize PCA bases and Lloyd-Max
   codebooks per layer for use at inference time? GGUF metadata? Sidecar file?
3. **SSM layer handling**: Qwen3.5-27B has 48/64 SSM layers with no KV cache.
   SpectralQuant must skip them. How does the calibration distinguish attention
   from SSM layers?
4. **Combining with PlanarQuant**: Should we apply a Clifford rotor within
   the d_eff signal subspace after PCA projection? (Near-free at d_eff=4, may
   improve Lloyd-Max accuracy marginally.)
5. **Benchmark threshold**: If SpectralQuant is better in PPL but slower in
   decode, what is the acceptable decode speed floor?
6. **Qwen3.6 support**: Should the sprint target only Qwen3.5-27B or also
   Qwen3.6-35B-A3B (MoE, different attention architecture)?
