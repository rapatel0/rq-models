# Sprint 001 Critique: TurboQuant KV Cache Quantization

**Reviewer**: Claude Sonnet 4.6 (1M context)
**Date**: 2026-03-24
**Drafts Reviewed**:
- `SPRINT-001-CODEX-DRAFT.md` (referred to as "Codex draft")
- `SPRINT-001-GEMINI-DRAFT.md` (referred to as "Gemini draft")
- `SPRINT-001-INTENT.md` (ground truth / reference)

---

## Executive Summary

Both drafts are competent structural documents but carry materially different failure modes. The Codex draft is verbose and implementation-rich, making it easier to spot precise algorithmic errors. The Gemini draft is compact and organized, but its brevity hides several silent omissions. Neither draft correctly handles the QJL projection matrix scaling, the HuggingFace cache subclassing interface, GQA head layout, or batch_size > 1. Neither meaningfully engages with the bfloat16 question despite it being the standard dtype for Qwen3.5-27B inference.

---

## Section 1: Codex Draft (`SPRINT-001-CODEX-DRAFT.md`)

### 1.1 Strengths

- **Implementation depth**: The Codex draft is the only one that provides concrete, runnable-looking pseudocode for all four main classes (`TurboQuantMSE`, `TurboQuantProd`, `QuantizedDynamicCache`, `detect_outlier_channels`). This is the right level of specificity for a sprint planning document — it forces latent assumptions to the surface.
- **Distortion bound propagation**: Lines 200–270 correctly cite the MSE bound `D_mse ≤ (√3π/2)·(1/4^b)` and the inner product bound. The per-bit numerical evaluations (b=2: ≤ 0.117, b=3: ≤ 0.03) match the intent document exactly.
- **Memory layout arithmetic**: Lines 296–312 correctly compute the per-token bytes for outlier (12 bytes) and regular (24 bytes) channels independently.
- **Phase timeline**: The 15-day breakdown with hourly estimates (Phase 1: ~20h, Phase 2: ~25h, Phase 3: ~20h) is realistic and actionable. The Gemini draft provides no timeline at all.
- **Risk table**: Lines 934–945 enumerate nine specific risks with likelihood/impact ratings. This is the most complete risk treatment of either draft.
- **Reproducibility gate**: Lines 921–928 correctly call out the need to seed both `torch.manual_seed()` and `np.random.seed()` and lock dependency versions. This is missing from the Gemini draft.
- **Open questions section**: Lines 992–1021 correctly identify the seven open questions from the intent document, including the bfloat16 question (line 1006) and the uint8 packing question (line 1010).

### 1.2 Weaknesses

#### 1.2.1 Critical: QJL Projection Matrix Scaling is Wrong

**Lines 221–222 (TurboQuantProd offline setup)**:
```
S = random_normal(128, 128, std=1/√128)  # normalized for stability
```

The intent document (line 59) specifies `S_{i,j} ~ N(0, 1)` — i.e., standard normal, **not** `N(0, 1/d)`. The scaling factor `1/√d` is already baked into the dequantization formula: `x̃_qjl = (√(π/2) / d) · γ · S^T · qjl`. Pre-scaling `S` by `1/√d` at initialization and then dividing by `d` again in dequantization would produce a reconstruction that is `1/√d` times too small. The QJL correction term would then contribute negligibly, degrading inner product estimation toward the pure MSE result and breaking the unbiasedness guarantee.

**Line 260 (dequant_prod)**:
```python
c = np.sqrt(np.pi / 2) / self.d  # ≈ 0.0627 for d=128
x_hat_qjl = c * gamma * (qjl @ self.projection_matrix)
```

The matrix multiplication here is `qjl @ S`, which gives shape `[*batch_dims, d]` when `qjl` is `[*batch_dims, d]` and `S` is `[d, d]`. This is correct only if `S` is treated as the transpose. The intent document specifies `S^T · qjl` (a matrix-vector product with `S^T` on the left), which is equivalent to `qjl @ S` in batched notation — so the direction is fine. However, if `S` was stored as `S` (not `S^T`), then `qjl @ self.projection_matrix` uses `S`, not `S^T`. The draft never clarifies whether `projection_matrix` holds `S` or `S^T`. This ambiguity will produce a silent bug in 50% of implementations.

#### 1.2.2 Critical: Effective Bit-Width Math is Self-Contradictory

**Lines 304–305**:
```
(32*3 + 96*2) / 128 = (96 + 192) / 128 = 288 / 128 = 2.25 bits
(plus ~1% overhead for outlier channel indices, gamma, and metadata)
≈ 2.5 effective bits
```

The draft correctly computes 2.25 bits for the quantized indices alone, then hand-waves "+~1% overhead ≈ 2.5 bits." But the overhead from metadata is **not 1%**. Each token per head also stores:
- Outlier channel index list: 32 indices into [0, 128), requiring 7 bits each = 224 bits = 28 bytes. This is stored once per layer (not per token), so amortizes over sequence length, but the draft does not clarify this.
- Residual norm `γ` (float16): 2 bytes per token per head.
- QJL bits: 128 bits = 16 bytes per token per head (if `TurboQuantProd` is used for V).

The draft uses `TurboQuantMSE` (not `TurboQuantProd`) for both K and V throughout (lines 385–396), yet section 3.5 of the intent requires `TurboQuantProd` for inner product (attention score) fidelity. If only `TurboQuantMSE` is used, the QJL component is entirely absent and the compression ratio is exactly 2.25 bits without any metadata correction needed — but then the unbiased inner product guarantee is also absent. The draft conflates the two modes without acknowledging the choice.

#### 1.2.3 Important: The Beta Distribution Parameters are Wrong for the Codebook

**Lines 148 and 453**:
```
Beta(shape_a=(d-1)/2, shape_b=(d-1)/2)
```

This is correct for the Beta distribution parameterization, but `scipy.stats.beta` takes shape parameters `a` and `b` as positional or keyword args, **not** `shape_a`/`shape_b`. The actual call should be `scipy.stats.beta((d-1)/2, (d-1)/2)`. This is a minor API error but will throw a `TypeError` if copied verbatim.

More importantly, the codebook operates on the **rotated** vector components `y_j = (Π x)_j`. After rotation of a unit vector from the sphere, each component `y_j` follows a distribution proportional to `(1-t²)^{(d-3)/2}` on `[-1, 1]` (a scaled Beta). The intent document (line 42) writes the density explicitly as `f_X(x) = [Γ(d/2) / (√π · Γ((d-1)/2))] · (1-x²)^{(d-3)/2}`. This is the arcsine-like distribution for large `d`, concentrated near ±1 only for very small `d`. The Codex draft's example centroids at line 162:
```
For b=2: c ≈ [-0.82, -0.27, +0.27, +0.82]
```
are plausible for small `d` but will be incorrect at `d=128` where the Beta(63.5, 63.5) distribution is very tightly concentrated near zero (mean=0, variance ≈ 1/(2d-1) ≈ 0.004). The centroids for `d=128, b=2` should be approximately `[-0.054, -0.018, +0.018, +0.054]`, not ±0.27 and ±0.82. Copying the example values as validation targets in `test_codebook.py` (line 467: "hand-verify against paper Table 1") will produce a test that passes with grossly wrong values if not checked carefully.

#### 1.2.4 Important: `__setitem__` / `__getitem__` is Not the Correct HuggingFace DynamicCache Interface

**Lines 364–436**: The draft subclasses `DynamicCache` and overrides `__setitem__` and `__getitem__`. The actual `DynamicCache` API (as of `transformers >= 4.36`) does not use `__setitem__` and `__getitem__` for cache updates. The correct interface requires overriding:
- `update(key_states, value_states, layer_idx, cache_kwargs)` — called by each attention layer to append new KV.
- `get_seq_length(layer_idx)` — returns current cache length for a given layer.
- Optionally `to_legacy_cache()` and `from_legacy_cache()` for compatibility.

The `__setitem__` / `__getitem__` pattern is a dict-like interface that `DynamicCache` does not define. A subclass overriding these methods will never have them called by the model's attention forward pass, resulting in the parent `DynamicCache.update()` being called instead (storing full fp16 tensors), with zero quantization applied. This is a silent correctness failure.

#### 1.2.5 Important: Outlier Detection is Not Data-Oblivious

**Lines 280–292**: The draft detects outlier channels by computing `K.abs().mean(dim=[0, 1, 2])` over the prefill batch. This is a data-dependent operation — different inputs produce different outlier sets. The intent document (line 93) explicitly states: "Data-oblivious / online: no calibration data, no offline tuning on model activations." The paper's TurboQuant algorithm is designed to be data-oblivious, meaning outlier channel indices should be fixed at initialization time (not computed from the input). The correct approach is either: (a) use fixed indices (e.g., the first 32 dimensions), or (b) determine indices from a separate offline calibration pass that is not counted as part of the online inference budget. The draft's approach computes indices from the prefill pass, which makes the method data-dependent and inconsistent with the paper's guarantees.

#### 1.2.6 Important: 3.5-bit Split is Described but Not Implemented

**Lines 56–57 (Gemini draft)** and lines 275–312 (Codex): The Codex draft describes 2.5-bit splitting but the implementation code only shows 2.5-bit. The 3.5-bit variant (64 channels @ 4-bit + 64 channels @ 3-bit, per intent line 79) is mentioned in the overview but no concrete implementation code, test, or file is listed for it. The Definition of Done (line 851) only checks 2.5-bit. An evaluator testing 3.5-bit will find it unimplemented.

#### 1.2.7 Important: `__getitem__` Uses Undefined Variables

**Lines 417–421**:
```python
K_full = torch.zeros(
    batch_size, seq_len, 128, device=self.device
)
```

`batch_size` and `seq_len` are not defined in the `__getitem__` scope. They would need to be inferred from the cached tensors (e.g., `self.k_cache_quant[layer_idx]['outlier_idx'][0].shape`). This code will throw a `NameError` at runtime.

Additionally, the reconstruction concatenates along `dim=2` (line 420: `torch.cat(k_outlier_list, dim=2)`), but the stored tensors are indexed by token (each entry in the list is a single-token slice), so the sequence dimension is the list length, not a tensor dimension. The `torch.cat` should be along `dim=0` or `dim=1` depending on the storage layout, and the result shape needs a `num_heads` dimension that is entirely absent from the `K_full` allocation (which has shape `[batch, seq, 128]` but should be `[batch, num_heads, seq, 128]` or `[batch, seq, num_heads, 128]`).

#### 1.2.8 Minor: The `gamma_norm` Buffer Registered in `TurboQuantMSE` is Unused

**Line 168**:
```
self.register_buffer('gamma_norm', torch.tensor(np.sqrt(np.pi/2 * d), dtype=torch.float32))
```

`TurboQuantMSE` does not use a norm buffer — that is only needed for `TurboQuantProd`. Registering it in the MSE class is dead code that will confuse implementers.

#### 1.2.9 Minor: Compression Ratio Calculation is Overstated

**Lines 309–312**: The draft claims `256 / 36 ≈ 7.1×` compression. The 36-byte figure counts only quantized index bytes (12 + 24). It omits:
- Per-head, per-token metadata for `TurboQuantProd` (if used): γ norm (fp16, 2 bytes) + QJL bits (128 bits = 16 bytes) = 18 bytes additional.
- With `TurboQuantProd` for K (for inner product fidelity), total per-head bytes = 36 + 18 = 54 bytes, giving `256/54 ≈ 4.7×`, not 7.1×.

The intent document (line 7) targets "4.5x–6x" which is consistent with the corrected figure, not the 7.1× stated here.

### 1.3 Algorithm Correctness Summary (Codex)

| Component | Correct? | Notes |
|---|---|---|
| Rotation matrix Π generation (QR) | Yes | Lines 143–146 |
| Beta distribution parameters for codebook | Partially | API arg name wrong; centroid examples wrong for d=128 |
| QUANT_mse rotate-then-quantize order | Yes | Lines 176–184 |
| DEQUANT_mse lookup-then-rotate-back order | Yes | Lines 193–197 |
| QJL projection matrix scaling | **No** | Lines 221–222: uses N(0,1/d) instead of N(0,1) |
| QJL dequantization formula | Partially | Formula is correct; transpose ambiguity is a latent bug |
| Residual norm stored as scalar | Yes | Line 244 |
| 2.5-bit split arithmetic | Partially | 2.25 bits correct; metadata overhead hand-waved |
| HF DynamicCache subclassing API | **No** | `__setitem__`/`__getitem__` not the correct interface |
| Data-oblivious constraint | **No** | Outlier detection uses live prefill data |

### 1.4 HF Integration Gaps (Codex)

The fundamental integration error is the `__setitem__`/`__getitem__` API assumption (see 1.2.4). Beyond that:

- The draft patches the model by overriding `layer.self_attn._past_key_values` (line 660), which is an internal attribute that may not exist on Qwen3.5-27B's attention module. The correct approach is to pass a custom cache to `model.generate(past_key_values=cache)` or use the `cache_implementation` argument added in transformers 4.41.
- The draft does not account for Qwen3.5-27B's GQA layout. With 8 KV heads and 28 query heads (typical for 27B models), the K and V tensors passed to the cache have shape `[batch, num_kv_heads, seq, head_dim]` = `[batch, 8, seq, 128]`. The draft's reconstruction code ignores the `num_heads` dimension entirely in the `K_full` allocation, which will produce shape mismatches.
- There is no mention of the `sliding_window` or `attention_sink` patterns that newer Qwen models may use for long-context generation.

### 1.5 Outlier Splitting Correctness (Codex)

The split arithmetic (32 × 3-bit + 96 × 2-bit = 2.25 bits) is correct. The 3.5-bit variant (64 × 4-bit + 64 × 3-bit = 3.5 bits) is mentioned in one line but not implemented. The outlier selection is data-dependent (violation of the data-oblivious constraint). The split-and-merge channel reconstruction code has the undefined variable bugs noted in 1.2.7.

### 1.6 Definition of Done (Codex)

The Codex DoD is the stronger of the two. It has specific numerical thresholds (±10% of bounds, ≥ 0.99 NIAH recall, ≤ 0.1 nats perplexity regression, < 1ms latency, ≥ 80% test coverage). It adds the code quality gates (PEP 8, type hints, docstrings) and reproducibility gates (seeded RNG, locked deps) that the Gemini draft omits.

Gaps:
- No DoD gate for 3.5-bit mode (only 2.5-bit is verified).
- No DoD gate confirming batch_size > 1 correctness.
- No DoD gate for bfloat16 input handling.
- The "coherent text" success criterion from the Gemini draft is absent, which is arguably more user-visible than perplexity.
- The NIAH DoD (line 881) says "at least 10 trials per length" — this is too few for a 1% failure rate to be statistically detectable; 100 trials are needed.

### 1.7 Risk Analysis Gaps (Codex)

The nine-row risk table is the best of either draft. Still missing:
- **GQA head layout mismatch**: If the cache receives `[batch, 8, seq, 128]` but the code assumes `[batch, seq, 128]`, every call will silently produce wrong reconstructions.
- **bfloat16 / float32 precision loss**: Qwen3.5-27B loads in bfloat16. Rotating a bfloat16 vector through a float32 rotation matrix involves a dtype promotion. The draft mentions float32 internally (line 958) but does not track where the cast happens or whether it is applied before `quant_mse`.
- **torch.compile interaction**: The codebook lookup via `self.codebook[idx]` (integer indexing) is not always compatible with `torch.compile`'s graph capture. Not listed as a risk.
- **Streaming generation cache growth**: As the cache grows to 32K entries, `torch.cat` over the accumulated list in `__getitem__` becomes O(N²) in time. Not listed as a risk.

### 1.8 Missing Edge Cases (Codex)

| Edge Case | Addressed? | Notes |
|---|---|---|
| batch_size > 1 | Partially | Shape mentioned but `K_full` allocation is wrong |
| GQA head layout (8 KV, 28 Q heads) | **No** | `num_heads` dimension absent from reconstruction |
| bfloat16 inputs | Mentioned (open question 4) but not resolved | No implementation decision |
| Streaming generation (per-token append) | Yes | Correctly described in decoding phase |
| Empty/zero-length sequences | No |  |
| Sequence length = 1 (single-token prefill) | No | Edge case for outlier detection over 1 token |
| Grouped Query Attention key replication | **No** | Not mentioned; GQA does not replicate KV to match Q heads at the cache level |

---

## Section 2: Gemini Draft (`SPRINT-001-GEMINI-DRAFT.md`)

### 2.1 Strengths

- **Conciseness**: The Gemini draft is well-organized and readable. It conveys the essential structure without the noise of half-formed pseudocode. For a manager or reviewer who wants to understand scope, it is the better read.
- **Correct identification of `TurboQuantProd` for K**: Section 3.1 (Architecture, Mathematical Primitives) correctly notes that `TurboQuantProd` should be used specifically for inner product preservation (i.e., for K cache, since K participates in attention score computation `Q @ K^T`). The Codex draft conflates both K and V under `TurboQuantMSE`.
- **Beta distribution density stated correctly**: Line 15 gives the density formula `f_X(x) = [Γ(d/2) / (√π · Γ((d-1)/2))] · (1-x²)^{(d-3)/2}` verbatim from the paper, which matches the intent document exactly.
- **Correct call out of `update()` method**: Line 22 states `TurboKVCache` should subclass `transformers.Cache` and specifies "override `update()` method" — which is the correct HuggingFace cache interface, unlike the Codex draft's `__setitem__`/`__getitem__` approach.
- **Open question on outlier selection**: The Gemini draft (line 97) raises the same open question as the intent document about whether outliers are fixed or per-calibration, which is the right question to flag.
- **`TurboQuantProd` return type**: Line 48-50 correctly shows the return type as `(mse_indices, qjl_bits, residual_norm)` with three components matching Algorithm 2 exactly.

### 2.2 Weaknesses

#### 2.2.1 Critical: QJL Projection Matrix Scaling is Wrong (Same Error as Codex)

**Line 16**: "A $d \times d$ matrix with $S_{i,j} \sim \mathcal{N}(0, 1/d)$"

Same error as the Codex draft. The intent document (line 59) specifies `S_{i,j} ~ N(0, 1)`. Scaling S by `1/d` instead of using standard normal entries is inconsistent with the dequantization formula `(√(π/2) / d) · γ · S^T · qjl`, which already incorporates the `1/d` normalization factor. Pre-scaling S by `1/d` makes the QJL correction `1/d` times too small, destroying unbiasedness. Both drafts share this error, suggesting it may come from a common misreading of the paper.

#### 2.2.2 Critical: MSE Distortion Bound Formula is Slightly Wrong

**Line 37**: `D_{mse} \le \frac{\sqrt{3\pi}}{2} \frac{1}{4^b}`

The intent document (line 84) gives `D_mse ≤ (√3π/2)·(1/4^b)`. Note that `(√3π)/2` and `√(3π)/2` are different: `(√3·π)/2 ≈ 2.72` while `√(3π)/2 ≈ 1.37`. The correct form from the paper (and confirmed in the intent document's numerical evaluations: b=1 → 0.36) requires `√(3π)/2 ≈ 1.37`, since `1.37 / 4^1 ≈ 0.34`, close to 0.36. The Gemini draft writes `\frac{\sqrt{3\pi}}{2}` which in LaTeX renders as `√(3π)/2` — this is actually correct notation. But the Codex draft at line 202 writes `(√3·π/2)` which is ambiguous and could be read as `(√3)·(π/2) ≈ 2.72` per term. If the Gemini draft's LaTeX is interpreted as `√(3π)/2`, it is correct. If it is read as `√3 · π/2` (as written literally), it is wrong. This ambiguity should be resolved by writing `\sqrt{3\pi}/2` explicitly. Given the brevity of the Gemini draft, the ambiguity is more likely to cause errors downstream.

#### 2.2.3 Important: Method Signatures for `TurboQuantProd.dequantize` are Incomplete

**Lines 48–51**:
```python
class TurboQuantProd:
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # returns (mse_indices, qjl_bits, residual_norm)
    def dequantize(self, mse_indices, qjl_bits, residual_norm) -> torch.Tensor: ...
```

The `dequantize` signature takes `residual_norm` but the intent document names it `γ = ‖r‖₂`. This is a naming inconsistency, not a bug. However, the draft provides no pseudocode for the dequantization step, which is the harder part to implement correctly (it requires `(√(π/2) / d) · γ · S^T · qjl`). The Codex draft at least provides the formula, even if it has the scaling bug.

#### 2.2.4 Important: 2.5-bit Effective Precision Arithmetic is Never Verified in the Gemini Draft

**Lines 25–27 (Phase 3)**:
> 2.5-bit: 32 channels (3-bit) + 96 channels (2-bit)

The Gemini draft states the split without verifying the arithmetic: `(32×3 + 96×2)/128 = 2.25 bits`. The paper calls this "2.5-bit" because metadata overhead (γ norm, QJL bits if using `TurboQuantProd`) brings the effective bit count up toward 2.5. The draft does not acknowledge this or compute it. A developer reading the Gemini draft would implement a 2.25-bit system and call it 2.5-bit, which fails on the compression ratio DoD.

#### 2.2.5 Important: 3.5-bit Split Not Defined in the Gemini Draft

**Line 27**: "3.5-bit: Similar split at higher precision"

No channel counts are given. For 3.5-bit, the intent document (line 79) states the split is `(64 channels × 3-bit + 64 channels × 4-bit) / 128 = 3.5 bits`. The Gemini draft leaves this as "similar split at higher precision" which is not implementable. A developer must guess.

#### 2.2.6 Important: HF Integration — `transformers.Cache` vs `DynamicCache`

**Line 22**: "A subclass of `transformers.Cache`"

Subclassing `Cache` (the abstract base class) rather than `DynamicCache` (the concrete implementation) is actually the more correct approach — it avoids inheriting `DynamicCache`'s internal storage and forces all required methods to be implemented explicitly. However, the Gemini draft then does not specify which abstract methods of `Cache` must be overridden. The `Cache` base class (as of transformers 4.41) requires at minimum:
- `update(key_states, value_states, layer_idx, cache_kwargs)`
- `get_seq_length(layer_idx)`
- `get_max_cache_shape()`
- `get_usable_length(new_seq_length, layer_idx)`

The Gemini draft mentions overriding `update()` correctly but omits `get_seq_length()`, which is called in `model.generate()` to check when to stop generating. An unimplemented `get_seq_length()` will raise `NotImplementedError` during generation.

#### 2.2.7 Minor: No Per-Task Effort Estimates

The Gemini draft has no time estimates for any task. This makes it impossible to assess sprint feasibility. The Codex draft's estimates are arguably too optimistic (4 hours for Lloyd-Max from scratch is tight) but having them is better than having none.

#### 2.2.8 Minor: Dependencies Section is Sparse

The Gemini draft (lines 90–93) lists `torch >= 2.2.0` and `transformers >= 4.38.0`. However:
- `transformers 4.38.0` predates the stabilized `Cache` ABC API, which changed significantly in 4.41 and again in 4.45. The minimum required version to get a stable `Cache.update()` interface is `>= 4.41.0`.
- No `scipy` version is pinned, yet the Lloyd-Max implementation depends on `scipy.optimize` behavior that changed between 1.9 and 1.11.

### 2.3 Algorithm Correctness Summary (Gemini)

| Component | Correct? | Notes |
|---|---|---|
| Rotation matrix Π generation (QR) | Yes | Line 14 |
| Beta distribution parameters for codebook | Yes | Correctly cited on line 15 |
| QUANT_mse rotate-then-quantize order | Yes | Phase 1 description |
| DEQUANT_mse lookup-then-rotate-back order | Yes | Phase 1 description |
| QJL projection matrix scaling | **No** | Line 16: N(0,1/d) instead of N(0,1) |
| QJL dequantization formula | Not shown | Only method signatures provided |
| Residual norm stored as scalar | Yes | Line 50 |
| 2.5-bit split arithmetic | Partially | Split stated but arithmetic not verified |
| 3.5-bit split | **No** | "Similar split" — not defined |
| HF Cache subclassing API | Mostly correct | `update()` mentioned; missing 3 abstract methods |
| Data-oblivious constraint | Flagged as open question | Better than Codex's silent violation |

### 2.4 HF Integration Gaps (Gemini)

- Correctly identifies `update()` as the override point, which is better than the Codex draft.
- Does not specify the full set of abstract methods that must be overridden.
- Does not address GQA head layout: with Qwen3.5-27B having `num_key_value_heads=8` and `num_attention_heads` (presumably 28 or 32), the `update()` method receives K/V of shape `[batch, 8, seq_len, 128]`. The draft's `OutlierSplitter` operates on the last dimension (128), which is correct, but does not acknowledge that the head dimension is axis 1, not axis 2.
- Does not address the model patching strategy: how does `TurboKVCache` get passed to `model.generate()`? The Gemini draft mentions `turboquant/model.py` for "Qwen3.5-27B monkey-patching logic" but provides no detail at all about what patching means.
- Does not address `cache_position` kwarg that HuggingFace passes to `update()` in newer versions.

### 2.5 Outlier Splitting Correctness (Gemini)

- 2.5-bit split stated (32 outlier + 96 non-outlier) but not verified arithmetically.
- 3.5-bit split is not specified.
- The open question about how outlier channels are selected is correctly preserved (line 97). This is the most important detail the Gemini draft gets right that could easily have been silently resolved incorrectly.

### 2.6 Definition of Done (Gemini)

The Gemini DoD (lines 74–80) has six items. Problems:
- **"Distortion within 5% of theoretical bounds"** (line 75): The intent document specifies 10%, not 5%. Using 5% as the threshold may cause spurious test failures due to Monte Carlo variance — 10k samples are needed to estimate a 5% relative bound reliably.
- **"Zero mean bias"** (line 76): This is technically correct for `TurboQuantProd` but is not a measurable threshold. What p-value cutoff is used? What sample size? An unbiased estimator will still show nonzero empirical bias on any finite sample. A concrete threshold (e.g., `|E[error]| < 0.001` at n=10k) is needed.
- **"Coherent text"** (line 77): Not measurable. Any test that passes/fails on this criterion is subjective.
- **"NIAH recall ≥ 0.99 for 32k context"** (line 78): Correct threshold from the intent document (line 104). But only 32k is listed; the intent requires testing at 4k, 8k, 16k, and 32k.
- **"Memory usage reduced by ≥ 4.5x"** (line 79): Correct per intent document (line 7 says "4.5x–6x"). The Codex draft's 7.1× figure is wrong; the Gemini draft's 4.5× is the correct lower bound.
- **"Inference latency overhead < 2ms per token"** (line 80): The intent document (line 105) specifies < 1ms on A100. The Gemini draft doubles this threshold to 2ms without justification.

The Gemini DoD lacks:
- Code quality gates (linting, type hints, docstrings)
- Reproducibility gates (seeded RNG, locked versions)
- Any gate for 3.5-bit mode
- Any gate for batch_size > 1
- Perplexity regression gate (present in intent and Codex, missing here)

### 2.7 Risk Analysis Gaps (Gemini)

The Gemini risks section (lines 82–86) has three items:
1. Python quantization overhead
2. Qwen integration with custom attention kernels
3. QR decomposition numerical stability

Missing from both drafts but critical for this sprint:
- **GQA head layout assumption**: If `num_key_value_heads` is not 8, or if the head layout is different than expected, the outlier splitting operates on the wrong dimension.
- **bfloat16 to float32 precision regression**: Casting bfloat16 activations to float32 for rotation, then back to bfloat16, introduces rounding that compounds over 64 layers.
- **Outlier index storage memory**: If outlier indices are stored per-layer per-token (not per-layer globally), the storage overhead dominates at long sequence lengths.
- **`torch.cat` O(N²) accumulation**: Growing the cache list and re-concatenating on every `get()` call is quadratic in sequence length. At 32K tokens, this is a serious latency bottleneck not listed anywhere.
- **Model weight format**: Qwen3.5-27B may be distributed in safetensors or GGUF format; standard `from_pretrained` may not handle it without additional dependencies.

### 2.8 Missing Edge Cases (Gemini)

| Edge Case | Addressed? | Notes |
|---|---|---|
| batch_size > 1 | **No** | Not mentioned anywhere |
| GQA head layout (8 KV heads) | **No** | Head dimension acknowledged but GQA structure not discussed |
| bfloat16 inputs | Partially (open question) | Not resolved |
| Streaming generation | Partially | Online/streaming mentioned in use cases |
| Empty sequences | **No** |  |
| `cache_position` argument in HF update() | **No** |  |
| flash_attention_2 / sdpa backend interaction | **No** |  |
| Multi-turn conversation (cache reuse) | **No** |  |

---

## Section 3: Cross-Draft Comparison

### 3.1 Where They Agree (and Both are Wrong)

Both drafts share these errors:

1. **QJL projection matrix scaling `N(0, 1/d)` instead of `N(0, 1)`**: This is the most consequential shared error. It will produce a QJL correction term that is `1/√d ≈ 0.088` times the correct magnitude, making the residual correction essentially negligible and degrading `TurboQuantProd` to approximately `TurboQuantMSE` with extra storage overhead. Inner product bias will be nonzero, and the paper's distortion bounds will not hold.

2. **Outlier channel counts stated without the 3.5-bit split arithmetic**: Both state the 2.5-bit split correctly but leave the 3.5-bit split either absent (Gemini) or unimplemented (Codex).

3. **No treatment of `cache_position` or incremental position IDs**: HuggingFace's attention forward pass passes `cache_position` to `update()` in recent versions. Neither draft mentions this argument.

4. **No treatment of the GQA expansion step**: In GQA, the model internally expands K and V from 8 heads to match Q's head count before computing attention scores. This expansion happens inside the attention module, after the cache `update()` returns. The K/V stored in the cache have `num_kv_heads=8` not `num_attention_heads`. Both drafts' memory layout discussions reference 128-dim vectors correctly, but neither explicitly acknowledges that the cache stores 8-head tensors, not query-head-count tensors.

### 3.2 Where the Codex Draft is Strictly Better

- Time estimates and phasing
- Concrete pseudocode that exposes implementation bugs
- Risk table with likelihood/impact
- Reproducibility and code quality gates in DoD
- More complete open questions

### 3.3 Where the Gemini Draft is Strictly Better

- HF cache override method (`update()` vs `__setitem__`)
- Correct identification of `TurboQuantProd` for K (vs MSE for both K and V)
- Preserving the data-oblivious open question rather than silently violating it
- Correct 4.5× compression lower bound (vs Codex's wrong 7.1× without metadata)
- Cleaner, less error-prone document structure

### 3.4 What Neither Draft Gets Right

1. **Rotation convention**: The intent specifies `y = Π · x` (matrix-vector product, Π on the left). Both drafts implement `y = x @ Π.T`, which is equivalent if Π is applied left-to-right in row-vector convention. This is algebraically identical but the drafts should be explicit about convention to avoid transposition bugs in the dequantization step (`x̃ = Π^T · ỹ` vs `x̃ = ỹ @ Π`).

2. **Codebook operates on normalized vectors**: The Lloyd-Max codebook is computed for the distribution of `(Πx)_j` where `x` is a **unit sphere** vector. If the input KV vectors are not normalized (and they won't be in general), the codebook bins will be wrong. The correct procedure is: (1) store `‖x‖₂` as a scalar norm, (2) normalize `x` to the unit sphere, (3) rotate and quantize, (4) on dequantize, look up codebook and scale by `‖x‖₂`. Neither draft implements this normalization step. This is likely the most important correctness gap in both documents.

3. **`TurboQuantProd` is for K; `TurboQuantMSE` (not Prod) is correct for V**: The paper uses `TurboQuantProd` for vectors that will participate in inner products (K, since attention computes `Q @ K^T`) and `TurboQuantMSE` for V (since V contributes via weighted sum, not dot product). Only the Gemini draft partially acknowledges this distinction. The Codex draft uses `TurboQuantMSE` for both, which is incorrect for K.

4. **Per-head quantizer vs per-layer quantizer**: The intent document says "Apply TurboQuant per attention head, per token." For GQA with 8 KV heads, this means 8 independent quantizer instances per layer (one rotation matrix Π and one codebook per head), not one per layer. Both drafts instantiate one quantizer per layer, which is an implementation shortcut not grounded in the paper.

---

## Section 4: Actionable Corrections

Regardless of which draft is taken forward, the following must be fixed before implementation begins:

1. **Fix QJL matrix scaling**: Change `S_{i,j} ~ N(0, 1/d)` to `S_{i,j} ~ N(0, 1)` in both the codebook generation script and the `TurboQuantProd` initialization. Do not pre-scale S; rely on the `1/d` in the dequantization formula.

2. **Add vector normalization to TurboQuantMSE**: Before rotating, compute `norm = ‖x‖₂`, normalize `x̂ = x / norm`, rotate and quantize `x̂`, store `(idx, norm)`. On dequantize, recover `x̃ = norm · Π^T · ŷ`. This step is implied by the paper's unit-sphere assumption and is missing from both drafts.

3. **Fix HF cache interface**: Override `update(key_states, value_states, layer_idx, cache_kwargs)` and `get_seq_length(layer_idx)`. Remove any `__setitem__`/`__getitem__` override logic.

4. **Define 3.5-bit split explicitly**: `64 channels × 4-bit + 64 channels × 3-bit = (256 + 192)/128 = 3.5 bits`. Add a `TurboQuantMSE(b=4)` instance, generate a `codebook_d128_b4.npy`, and add a 3.5-bit code path to `QuantizedKVCache.update()`.

5. **Fix outlier selection to be data-oblivious**: Use fixed indices (e.g., indices 0–31, or a fixed random permutation seeded at init time) rather than live magnitude computation from the prefill batch.

6. **Add batch_size > 1 test**: The smoke test and distortion tests must include `batch_size=4` to catch shape broadcasting bugs.

7. **Add GQA head dimension test**: Explicitly test that cache `update()` correctly handles input shape `[batch, 8, seq, 128]` and returns reconstructed K/V of the same shape.

8. **Resolve bfloat16 strategy**: Decide whether to cast bfloat16 inputs to float32 before rotation (recommended) and where to cast back. Document this decision explicitly rather than leaving it as an open question.

9. **Fix the latency DoD threshold**: Use `< 1ms` per the intent document, not `< 2ms` (Gemini draft).

10. **Fix the Monte Carlo sample count in DoD**: Use 10k samples (per intent line 100), not 1000 (Codex draft line 504) and not unspecified (Gemini draft).
