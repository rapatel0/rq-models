# Sprint 003 Merge Notes

## Base Draft

**Claude draft** used as base. Stronger package structure, better DoD, dual
signal/noise quantization, safetensors sidecar, and test file all validated.

## Interview Decisions Applied

| Question | Decision | Impact on Sprint |
|----------|----------|-----------------|
| Scope | Python prototype + C/CUDA design doc | C/CUDA deferred to Sprint 004 |
| Calibration location | Offline, committed to repo | Sidecars pre-computed and stored; generation script in DoD |
| K/V asymmetry | Yes — asymmetric d_eff for K vs V | K: d_eff≈4–5, V: d_eff≈40–55; Phase 1 design must accommodate |
| Fallback if PPL > 8.20 | Document and stop | Kill criterion added: validate on 9B early; halt if losing |

## Valid Critiques Accepted

From Claude critique:
- **Drop SPECTRAL_3BIT from Sprint 003**: Focus on 4-bit signal / 2-bit noise first. 3-bit variant deferred.
- **Upgrade C/CUDA design to 15-20% effort**: Codex draft's specific audit/design tasks incorporated.
- **Add decode throughput benchmark**: "95% of planar4 tok/s" added to DoD.
- **Kill criterion added**: If spectral4 PPL > 8.20 on Qwen3.5-9B after Phase 1, halt and document.
- **Remove Use Case #4** (hybrid Clifford): Deferred to open question.
- **Add long-context validation** (ctx ≥ 16K): Added to Phase 4 benchmarks.
- **Eigenvalue gap fallback**: If no clear signal/noise split, fall back to full-dim quantization for that layer.
- **Don't commit calibration binaries raw**: Use git-lfs OR document regeneration command; DoD gates on generation script, not committed binary.

From Codex critique:
- **NIAH test added to DoD**: `scripts/eval_niah.py` already exists; add 4K and 16K NIAH to benchmark suite.
- **d_eff sweep added**: Sweep d_eff ∈ {2, 4, 6, 8} for K; {20, 40, 55, 70} for V during Phase 3.
- **Sidecar absent fallback**: Entrypoint falls back to planar4 if no sidecar found; clear error message.
- **PCA at f32 vs f16**: Calibration precision validation task added.

## Critiques Rejected

| Critique | Rejection Reason |
|----------|-----------------|
| "Hybrid Clifford+SpectralQuant as a use case" (my Codex sub draft) | Speculative; deferred to open question per Claude critique |
| "spectral_ggml.py in Sprint 003" (Codex draft) | GGUF bridge is Sprint 004 work; not needed for Python prototype validation |
| "Calibration only at startup" | User explicitly chose offline/committed |

## K/V Asymmetry Design Impact

The user's choice of asymmetric K/V is architecturally significant:
- K channels: d_eff ≈ 4–5 (4.25 bpe signal, ~0.1 bpe noise → ~4.5 bpe total)
- V channels: d_eff ≈ 40–55 of 256 dims (~16–21% of dims), higher-bit noise
- Separate `SpectralCalibrator` codebooks and eigenvectors for K and V per layer
- `SpectralQuantizer` must handle K and V with different d_eff configs
- Overall compression ratio changes: K is still highly compressed; V is less so
- This is the correct choice per SpectralQuant paper and likely needed to beat planar3 on V channels

## Scope Summary

**In scope (Sprint 003)**:
- Python prototype: SpectralCalibrator, SpectralQuantizer (asymmetric K/V), SpectralKVCache
- Calibration scripts for Qwen3.5-27B and Qwen3.5-9B
- Perplexity, cosine similarity, NIAH, throughput benchmarks
- C/CUDA integration design document (15-20% effort)
- Updated benchmark docs

**Deferred to Sprint 004**:
- llama.cpp C/CUDA kernel implementation
- GGUF sidecar format and bridge script
- Qwen3.6-35B-A3B support
- SPECTRAL_3BIT variant
- Hybrid SpectralQuant + PlanarQuant (Clifford in signal subspace)
- D-011 Open WebUI, D-013 Benchmark CI (Sprint 002 deferrals)
