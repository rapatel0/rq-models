# Sprint 003 Draft Critique: Codex vs Claude

> **Date**: 2026-04-20
> **Reviewer**: Claude (automated critique)
> **Inputs**: SPRINT-003-CODEX-DRAFT.md, SPRINT-003-CLAUDE-DRAFT.md, SPRINT-003-INTENT.md

---

## Executive Summary

Both drafts converge on the same core strategy — Python-first SpectralQuant prototype with C/CUDA deferred — and cover the key technical challenges (PCA calibration, Lloyd-Max codebooks, SSM skip, GQA handling). The Claude draft is more thorough in implementation structure, edge case coverage, and Definition of Done. The Codex draft is more concise and includes a stronger integration design phase. Neither draft adequately addresses the Intent's decode speed criterion or the GPU memory constraint during benchmarking.

---

## 1. Codex Draft — Strengths

1. **Conciseness**: The draft is tighter and easier to scan. Architecture section uses a single clear ASCII diagram that maps calibration, inference, and dequant paths in one view.

2. **Stronger Phase 2 (llama.cpp design)**: Allocates 30% of effort to the integration design, with specific tasks (audit `ggml.h`, design `GGML_TYPE_SPECTRAL4`, CUDA kernel design, entrypoint.sh flow). This produces a more actionable handoff artifact for Sprint 004.

3. **Security section**: Explicitly calls out path traversal and buffer overflow risks in sidecar parsing, and warns against using private data for calibration. Practical and specific.

4. **Open Question #4 (3-bit variant)**: Raises whether `spectral3` should be in scope — a useful scoping question that the Claude draft addresses by just including it without discussion.

5. **Explicit GGUF sidecar export script** (`spectral_ggml.py`): Plans for the Python-to-GGUF bridge, which is a concrete Sprint 004 unblocking artifact.

## 2. Codex Draft — Weaknesses

1. **Flat file structure**: All core code in `turboquant/spectral.py` (one file) and `turboquant/spectral_kv_cache.py`. At the expected complexity (calibrator + quantizer + store + cache), a single `spectral.py` will become unwieldy. The Claude draft's package structure (`turboquant/spectral/`) is more maintainable.

2. **No test file**: No `tests/test_spectral.py` or any test infrastructure mentioned in the files summary. The DoD references unit tests but doesn't plan where they live or what they cover beyond round-trip cosine similarity.

3. **No latency target**: The Intent specifies "decode speed >= 95% of f16 baseline (~65.8 tok/s)" as a success criterion. The Codex draft omits any encode/decode latency measurement from both the tasks and the DoD. This is a gap — SpectralQuant adds PCA projection and Lloyd-Max lookup per token, which could regress throughput.

4. **No noise dimension quantization**: The architecture section mentions "1-bit or zero fill → noise dims (negligible)" but the implementation tasks only cover signal dimension encoding. The Claude draft explicitly plans 2-bit noise quantization with separate codebooks, which better preserves reconstruction quality.

5. **Calibration only on wikitext-2**: Single calibration corpus. No plan to validate sensitivity to calibration data distribution. If PCA bases are corpus-sensitive, the approach may be fragile in production with diverse prompts.

6. **Missing edge cases**: No mention of handling very short contexts (fewer tokens than d_eff), first-token empty cache, or what happens when PCA eigenvalue gap is ambiguous (no clear signal/noise split).

7. **Effort percentages don't add up to actionable time**: 45% + 30% + 25% = 100%, but no indication of elapsed time or sprint duration. The Claude draft has the same issue but at least has 5 phases with more granular task breakdowns.

## 3. Claude Draft — Strengths

1. **Package structure**: `turboquant/spectral/` with separate files for calibrator, quantizer, store, and kv_cache. Clean separation of concerns, testable in isolation.

2. **Richer Definition of Done (13 items)**: Covers PCA validation against reference, latency (<1ms per token), calibration sidecar committed, existing tests passing, and benchmark docs updated. More comprehensive gate for sprint completion.

3. **Latency target included**: "Encode + decode latency <1ms per token on RTX 5090" — directly addresses the Intent's decode speed concern, even if the specific threshold differs from the Intent's "95% of f16 baseline" framing.

4. **Signal + noise dual quantization**: Explicitly plans 4-bit signal / 2-bit noise split with separate Lloyd-Max codebooks for each. This is closer to the SpectralQuant paper's approach and should yield better reconstruction than the Codex draft's "zero fill noise" strategy.

5. **Edge case handling**: Phase 2 tasks explicitly list first-token, short-context, and GQA edge cases. Phase 1 includes validation against the reference implementation.

6. **Calibration corpus sensitivity**: Risk table and Open Question #6 both flag corpus overfitting. Plans to test with 2-3 different calibration sets.

7. **safetensors for sidecar**: Concrete format choice with integrity validation (no pickle). Better than the Codex draft's generic `.bin` format, which has no built-in safety guarantees.

8. **Test file planned**: `tests/test_spectral.py` in the files summary with unit + integration tests.

9. **Both 4-bit and 3-bit presets**: `SPECTRAL_4BIT` and `SPECTRAL_3BIT` in scope, enabling the planar3 vs spectral3 comparison that the Codex draft raised as an open question.

## 4. Claude Draft — Weaknesses

1. **Weak C/CUDA integration design phase**: Only 5% effort allocated. Four bullet-point tasks for a design doc. The Codex draft's 30% allocation with specific audit and design tasks will produce a much more useful Sprint 004 handoff. This is the Claude draft's biggest gap — if Sprint 004 starts with a thin design doc, the C/CUDA integration will face the same "open question" uncertainty that Sprint 003 is trying to resolve.

2. **No GGUF bridge script**: No equivalent of the Codex draft's `spectral_ggml.py` for exporting calibration to GGUF-compatible format. The safetensors sidecar is good for Python but doesn't address how the C/CUDA sprint will consume it.

3. **Hybrid SpectralQuant+PlanarQuant as a use case**: Use Case #4 includes "applying Clifford rotors within the d_eff signal subspace." This is speculative research that shouldn't be a primary use case — it's an experiment at best. Including it as a use case inflates expectations.

4. **Scope creep risk from 3-bit preset**: Adding `SPECTRAL_3BIT` doubles the benchmark matrix (two SpectralQuant variants x multiple baselines x two models). The Codex draft wisely questions whether this belongs in Sprint 003. Including it without discussion risks scope bloat.

5. **Calibration for two models**: Plans sidecar files for both Qwen3.5-27B and Qwen3.5-9B. The 9B is fine for iterative dev, but committing its calibration as a DoD item adds work without clear production value. The Intent focuses on 27B.

6. **Open Question #3 (K vs V asymmetric quantization)**: Important question but no proposed answer or decision framework. If this needs resolution during the sprint, it could block Phase 1 implementation.

---

## 5. Gaps in Risk Analysis

### Both Drafts Miss

| Gap | Why It Matters |
|-----|----------------|
| **Decode speed regression** | The Intent requires "decode speed >= 95% of f16 baseline (~65.8 tok/s)." PCA projection adds a matrix multiply per token per layer. Neither draft models this cost or sets a kill criterion if throughput drops below the floor. |
| **GPU memory during benchmarks** | The Intent warns "Must stay below ~8 GB during benchmarks (training co-occupies GPU)." Neither draft sizes the calibration or benchmark memory footprint against this constraint. The Codex draft mentions batch_size=1/ctx=256 for calibration but not for benchmarks. |
| **Numerical stability of PCA at f16** | Both drafts assume PCA via `torch.linalg.eigh` at f16 precision. Eigendecomposition of near-degenerate covariance matrices at f16 can produce noisy eigenvectors. Neither draft tests f16 vs f32 calibration or sets a tolerance. |
| **Calibration-inference distribution shift** | Calibration on wikitext-2 (natural language) may not capture the KV activation distribution for code, structured data, or multilingual prompts. Neither draft proposes a held-out evaluation on a different domain. The Claude draft mentions it in risks but doesn't add a task. |
| **Model update invalidates calibration** | If the Qwen model weights change (fine-tuning, quantization variant), the PCA bases become stale. Neither draft addresses calibration versioning or staleness detection. |

### Codex-Specific Gaps

| Gap | Why It Matters |
|-----|----------------|
| **No rollback plan if SpectralQuant loses** | The risk table says "still document results" but doesn't define what the sprint delivers if SpectralQuant doesn't beat planar3. Is it a research report? A library with no default? The Claude draft has the same issue but at least proposes a hybrid pivot. |
| **No integration test** | DoD has unit tests but no end-to-end generation test. The Claude draft includes "generate 100 tokens with SpectralKVCache on Qwen3.5-9B, verify output is coherent." |

### Claude-Specific Gaps

| Gap | Why It Matters |
|-----|----------------|
| **No security consideration for safetensors loading** | The Codex draft warns about sidecar parsing attacks. The Claude draft trusts safetensors' built-in validation but doesn't consider path traversal in the sidecar file path resolution. |
| **Phase 5 is too thin to be a real phase** | At 5% effort, the C/CUDA design phase is a checkbox, not a deliverable. If it produces an insufficiently detailed design doc, Sprint 004 inherits all the architectural uncertainty. |

---

## 6. Missing Edge Cases

| Edge Case | Codex | Claude | Notes |
|-----------|-------|--------|-------|
| First token (empty KV cache) | Not mentioned | Covered (Phase 2 task) | Quantizer must handle zero-length cache gracefully |
| Context shorter than d_eff | Not mentioned | Covered (Phase 2 task) | PCA projection with fewer tokens than dimensions is degenerate |
| Layer with ambiguous eigenvalue gap | Not mentioned | Not mentioned | If no clear signal/noise split exists, d_eff selection is arbitrary. Need a fallback (e.g., skip quantization for that layer, use full-dim Lloyd-Max) |
| Non-contiguous KV cache (after eviction) | Not mentioned | Not mentioned | If KV cache eviction is used (common at long context), quantized entries may be non-contiguous. Both caches assume append-only. |
| Mixed precision inference (some layers f16, some spectral) | Not mentioned | Implied by SSM skip | Should be explicit: attention layers get SpectralQuant, SSM layers get f16, and the cache must handle heterogeneous layer types. |
| Codebook drift over long sequences | Not mentioned | Not mentioned | Lloyd-Max codebooks fitted on short calibration data may be suboptimal for KV activations at token position 100K+. Neither draft plans a long-context validation beyond ctx=4096. |
| Multi-GPU / tensor parallel | Not mentioned | Not mentioned | The Intent mentions multi-GPU as deferred (D-010), but if calibration assumes single-GPU KV layout, TP would break it. Worth noting as a non-goal. |

---

## 7. Definition of Done Completeness

### Codex DoD (9 items) — Assessment

| Item | Verdict |
|------|---------|
| Round-trip cosine >0.94 | Good but only on "random KV tensors" — should also validate on real model activations |
| Calibration <30s | Good |
| PPL < 8.20 | Good — matches Intent success criterion |
| Compression >= 4.9x | Good |
| SSM layers skipped | Good |
| GQA handled | Good |
| C/CUDA design doc | Good |
| Benchmark docs updated | Good |
| No regression | Good |
| **Missing**: Latency target | Decode speed is a success criterion in the Intent |
| **Missing**: Existing tests pass | Should gate on `pytest tests/` passing |
| **Missing**: Integration test | End-to-end generation coherence |

### Claude DoD (13 items) — Assessment

| Item | Verdict |
|------|---------|
| Calibrator produces artifacts <30s | Good |
| PCA matches reference (cosine >0.999) | Good — higher bar than Codex, validates correctness |
| Round-trip cosine >0.94 | Good |
| PPL < 8.20 | Good |
| Compression >= 4.9x | Good |
| Latency <1ms per token | Good — addresses Intent gap, but note the Intent frames it as "95% of f16 throughput," not absolute latency |
| SSM layers skipped | Good |
| GQA handled | Good |
| All existing tests pass | Good |
| No regressions | Good |
| Calibration sidecars committed | Questionable — binary artifacts in git? Should be generated, not committed |
| C/CUDA design doc | Good but weak given 5% effort allocation |
| Benchmark results in docs | Good |
| **Missing**: Decode speed vs f16 baseline | The <1ms latency target is per-token encode/decode, not end-to-end generation throughput. The Intent's "95% of 65.8 tok/s" is a different metric. |
| **Missing**: Kill criterion | What happens if PPL > 8.20? Neither DoD defines a "sprint failed" outcome. |

---

## 8. Recommendation

**Use the Claude draft as the base**, with these modifications drawn from the Codex draft and this critique:

1. **Upgrade Phase 5 (C/CUDA design) to 15-20% effort**, incorporating the Codex draft's specific audit/design tasks and the GGUF export bridge script.
2. **Drop `SPECTRAL_3BIT` from Sprint 003 scope** — keep it as an open question for Sprint 004. Focus on proving 4-bit signal / 2-bit noise works.
3. **Add decode throughput benchmark** (tok/s vs f16 baseline) to both tasks and DoD, matching the Intent's "95% of 65.8 tok/s" criterion.
4. **Add GPU memory measurement** during calibration and benchmarks to validate the <8 GB co-occupancy constraint.
5. **Add a kill criterion**: If spectral4 PPL > 8.20 after Phase 1 validation on Qwen3.5-9B, escalate before investing in Phases 3-5. Define what the sprint delivers in the "doesn't beat planar3" scenario.
6. **Test PCA at f32 vs f16** — add a calibration precision validation task.
7. **Remove Use Case #4** (hybrid Clifford+SpectralQuant) — it's speculative and inflates scope expectations. Keep it as an open question only.
8. **Don't commit calibration binaries to git** — generate them as build artifacts and document the generation command.
9. **Add long-context validation** — at least one PPL or cosine-similarity check at ctx >= 16K to catch codebook drift.
10. **Add eigenvalue gap fallback** — document what happens when a layer has no clear signal/noise split.
