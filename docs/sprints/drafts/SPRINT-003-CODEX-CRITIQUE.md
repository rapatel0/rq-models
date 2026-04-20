# Sprint 003 Codex-Substitute Critique

> Critiques of SPRINT-003-CLAUDE-DRAFT.md and SPRINT-003-CODEX-DRAFT.md

---

## Critique of SPRINT-003-CLAUDE-DRAFT.md

### Strengths

1. **Clean subpackage structure** (`turboquant/spectral/`): Separating calibrator,
   quantizer, store, and kv_cache into distinct modules is correct. Avoids the
   monolith mistake from Sprint 001's `kv_cache.py`.

2. **Safetensors sidecar format is the right call**: Avoids GGUF format changes
   (would require llama.cpp fork surgery) while remaining readable by both Python
   prototype and future C/CUDA integration.

3. **Clear phase separation**: Python prototype first, C/CUDA design doc deferred.
   This is the right risk management given architecture uncertainty.

4. **Calibration overfitting risk identified**: The note about using diverse
   calibration data (not just wikitext-2) is important and correct — wikitext-2
   is a narrow register corpus.

### Weaknesses

1. **`d_eff` selection is underspecified**: The draft says "eigenvalue gap/ratio
   heuristic" but doesn't specify a default. This is a critical hyperparameter —
   wrong d_eff silently degrades quality. The DoD should require a sweep: run
   calibration at d_eff ∈ {2, 4, 6, 8} and measure PPL + cosine sim per layer.
   Without this, the implementation has an unvalidated magic number.

2. **Latency target may be unreachable**: The <1ms/token DoD for encode+decode is
   the same target that Sprint 001 F-001 found unachievable for TurboQuant
   (measured 44ms on A100). SpectralQuant's PCA projection adds a matrix multiply
   at inference time (KV × V_signal, shape [4, 256, d_eff]) per layer per token.
   At 16 attention layers: 16 matrix multiplies per token decode step. This is
   likely significantly slower than PlanarQuant's in-register ops. The DoD should
   either relax this target to "within 5% of planar4 decode tok/s" or add a
   profiling task.

3. **No NIAH (needle-in-a-haystack) test**: The DoD measures PPL and cosine
   similarity but not recall fidelity at long contexts. NIAH at 4K and 16K would
   catch catastrophic attention failures that PPL misses. Prior sprints already
   have NIAH infrastructure (`scripts/eval_niah.py`).

4. **Calibration sidecar size not estimated**: For Qwen3.5-27B with 16 attention
   layers, head_dim=256, 4 KV heads: each layer needs `[4, 256, d_eff]` float16
   eigenvectors ≈ 4 × 256 × 5 × 2 bytes = 10 KB per layer × 16 = 160 KB. Plus
   codebooks: 4 heads × 16 centroids × 5 dims × 2 bytes = 640 bytes per layer ×
   16 = ~10 KB. Total: ~170 KB. This should be documented — it's small enough to
   commit to the repo.

5. **Missing: what happens if calibration sidecar is absent?** The draft assumes
   the sidecar always exists. There should be a fallback: if no calibration sidecar
   is found, fall back to `planar4` (or raise a clear error). This affects the
   Docker entrypoint design.

### Risk Analysis Gaps

- **Residual attention error accumulation**: At long contexts (100K+ tokens), small
  per-token reconstruction errors in the KV cache accumulate. The cosine similarity
  metric at single-token level doesn't capture this. The draft should add a "long
  context coherence" test: generate 1000 tokens at 16K context and check NIAH recall.

- **Layer depth sensitivity**: SpectralQuant's paper shows d_eff varies significantly
  by layer depth (early layers vs late layers). The implementation should validate
  that d_eff is not uniformly 4-5 across all layers — some may need d_eff=8-10.

### DoD Completeness: 7/10
Missing: NIAH test, decode latency vs planar4 comparison, d_eff sweep, fallback
behavior when sidecar absent.

---

## Critique of SPRINT-003-CODEX-DRAFT.md

### Strengths

1. **Calibration-free variants use case is important**: Noting that operators can
   pre-compute at startup (vs bake into image) is a real deployment decision the
   implementation needs to accommodate. The draft raises it; the Claude draft misses it.

2. **Qwen3.6-35B-A3B explicitly addressed**: The MoE architecture deferral to Sprint 004
   is clearly stated with rationale. Good scope management.

3. **Sidecar attack surface noted in Security**: The risk of malformed sidecar files
   is real — the sidecar is loaded from disk at startup and contains tensor data.
   Safetensors mitigates this (no pickle), but validating tensor shapes against
   model architecture before use is worth adding to the DoD.

4. **`spectral3` variant question raised**: Whether to include a 3-bit signal variant
   alongside 4-bit is a good open question. PlanarQuant results show 3-bit can beat
   4-bit; the same may be true for SpectralQuant.

### Weaknesses

1. **Flat file layout vs subpackage**: All spectral code in one file
   (`turboquant/spectral.py`) is wrong at this complexity level. The Claude draft's
   subpackage approach is clearly superior and should be adopted.

2. **No test file**: The DoD doesn't mention unit tests or `tests/test_spectral.py`.
   The Sprint 001 experience (F-005, missing d=64 in codebook tests) shows that
   missing test coverage directly causes bugs. This is a significant gap.

3. **Lloyd-Max codebook fitting details are thin**: "n_iter=100, n_centroids=16"
   is stated without justification. 16 centroids = 4-bit. But the noise dims use
   what? The draft says "1-bit or zero fill" in the architecture diagram, which
   contradicts "aggressive compression" — 1-bit and 0-bit are very different. This
   needs to be resolved.

4. **`spectral_ggml.py` scope creep risk**: Including a GGUF export script in Sprint
   003 scope is premature — this is Sprint 004 work. The Python prototype validation
   should be complete before designing the GGUF format. Remove from Sprint 003 scope.

5. **No integration test**: The DoD has no test that actually runs end-to-end inference
   with `SpectralKVCache` on a real model. "Unit test cosine similarity > 0.94" is
   necessary but not sufficient — a 100-token generation test on Qwen3.5-9B is needed.

### Risk Analysis Gaps

- **Missing risk: key vs value asymmetry**: The paper shows values require 40-55 signal
  dimensions vs keys' 4-5. This means the same `d_eff` cannot be used for both K and V.
  Neither draft addresses this explicitly — the Codex draft is silent, which is worse
  than the Claude draft which at least raises it as an open question.

- **Missing risk: calibration corpus sensitivity**: Not addressed. If d_eff and codebooks
  vary significantly with different calibration data, production deployments will have
  unpredictable quality.

### DoD Completeness: 5/10
Missing: unit tests file, integration test, d_eff sweep, K/V asymmetry handling,
NIAH test, sidecar shape validation.

---

## Summary

| Criterion | Claude Draft | Codex Draft |
|-----------|:------------:|:-----------:|
| Module structure | ✅ subpackage | ❌ flat file |
| Unit tests | ✅ mentioned | ❌ absent |
| Integration tests | ✅ mentioned | ❌ absent |
| Latency target realism | ⚠️ may be too aggressive | ✅ uses decode tok/s |
| NIAH test | ❌ missing | ❌ missing |
| K/V asymmetry | ✅ open question | ❌ silent |
| d_eff sweep | ❌ missing | ❌ missing |
| Sidecar fallback | ❌ missing | ❌ missing |
| Security | ✅ adequate | ✅ good |
| Scope discipline | ✅ GGUF deferred | ⚠️ GGUF in scope |

**Recommendation**: Use the Claude draft as the base. Incorporate from Codex draft:
(1) calibration-at-startup deployment flexibility, (2) explicit Qwen3.6 deferral
statement, (3) `spectral3` variant as an open question. Add from this critique:
(4) NIAH test in DoD, (5) d_eff sweep task, (6) sidecar fallback behavior, (7)
realistic latency target framed as "within 5% of planar4 decode tok/s".
