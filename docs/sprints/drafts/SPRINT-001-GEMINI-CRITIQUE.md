# Sprint 001 Critique: TurboQuant KV Cache Quantization for Qwen3.5-27B

This critique evaluates the implementation plans for Sprint 001, comparing the architectural rigor and execution feasibility of the proposed drafts.

## 1. Executive Summary
Both drafts provide a solid foundation for implementing TurboQuant. The **Claude Draft** excels in architectural abstractions and mathematical fidelity to the TurboQuant paper (specifically regarding Beta distribution parameters and parameter sharing). The **Codex Draft** provides a more realistic execution timeline and a superior "Definition of Done" regarding software engineering standards (reproducibility, linting, and numerical stability).

## 2. Strengths of Drafts

### Claude Draft
- **Mathematical Accuracy:** Correctly identifies the Beta distribution parameters as $(d-1)/2$.
- **Architectural Efficiency:** Correctly proposes sharing the rotation matrix $\Pi$ and projection matrix $S$ across all 64 layers. Since the algorithm is data-oblivious, per-layer unique rotations are unnecessary and would waste ~131k parameters (128x128 float32) per layer.
- **Configurability:** Uses a `BitConfig` dataclass to manage the 2.5-bit vs. 3.5-bit split, which is cleaner for experimentation.

### Codex Draft
- **Realistic Scheduling:** The 15-day phased approach (Foundation $\rightarrow$ Integration $\rightarrow$ Validation) is well-calibrated for a solo implementation.
- **DoD Rigor:** Includes critical "Must Pass" criteria for reproducibility (fixed seeds) and numerical stability ($||\Pi^T \Pi - I||_F < 1e-6$).
- **Integrated Smoke Testing:** Explicitly schedules a Day 10 "Smoke Test" to catch HuggingFace integration issues early.

## 3. Algorithm Correctness & Outlier Logic
- **Data-Obliviousness:** The paper mandates a data-oblivious approach. **Claude** adheres to this by suggesting a fixed first-N channel split for outliers. **Codex** suggests a magnitude-based detection during prefill. While magnitude-based detection is common in other quantizers (like KIVI), it violates the strict "data-oblivious" requirement of TurboQuant. Sprint 001 should prioritize the fixed-N approach to validate the paper's core claims.
- **TurboQuantProd Residuals:** Both drafts correctly implement the QJL-on-residuals logic. However, neither explicitly defines the scale factor $c = \sqrt{\pi/2}/d$ as a constant that should be pre-computed to avoid $O(D)$ redundant operations.

## 4. HuggingFace Integration Strategy
Both drafts rely on subclassing `DynamicCache`.
- **The $O(N^2)$ Risk:** Overriding `__getitem__` to dequantize the *entire* sequence for every new token generated will lead to quadratic slowdown as the context grows.
- **Refinement:** The implementation should ensure that `__getitem__` only returns the dequantized tensors, but the cache itself should support an optimized `update` that only quantizes the *new* token's KV pairs.

## 5. Missing Edge Cases & Risk Gaps

### GQA Layout Awareness
Qwen3.5-27B uses Grouped Query Attention (32 Query heads, 8 KV heads).
- **Risk:** The implementation must ensure that `__getitem__` returns a tensor with 8 KV heads. If the dequantizer accidentally broadcasts to 32 heads *before* returning, it will inflate VRAM usage by 4x.
- **Batch > 1:** Both drafts mention batching but don't specify how to handle variable-length sequences in a batch (padding/masking). `QuantizedDynamicCache` must handle `K_new` shapes with batch dimensions correctly.

### bfloat16 Numerical Stability
Qwen3.5-27B uses `bfloat16`.
- **Gap:** Rotating a `bfloat16` vector by a random $\Pi$ matrix is prone to significant error due to the limited 7-bit mantissa.
- **Requirement:** Implementation **MUST** upcast to `float32` for the rotation and quantization steps, casting back to the model's native dtype only after dequantization.

### Streaming Memory Fragmentation
- **Gap:** Storing quantized indices as a list of small tensors (one per token) will cause severe CUDA memory fragmentation.
- **Requirement:** The implementation should pre-allocate a `uint8` buffer for a maximum sequence length (e.g., 32k) and use a pointer to fill it, similar to `StaticCache`.

## 6. Definition of Done (DoD) Enhancements
To be considered "Done," the following must be added to the existing criteria:
1. **Numerical Stability:** Verify that `Dequant(Quant(x))` in `fp32` has an MSE within 1% of the theoretical Beta-distribution bound.
2. **GQA Compatibility:** Verify the model generates coherent text with `num_key_value_heads=8`.
3. **Latency Amortization:** Quantization overhead must be measured *separately* for prefill (large batch) and decode (single token).

## 7. Final Recommendation
Proceed with a hybrid plan: Use **Claude's** mathematical formulation and parameter-sharing architecture, but follow **Codex's** 15-day execution timeline and rigorous DoD. Prioritize `float32` upcasting for all rotation operations to ensure the 0.997 NIAH recall is achievable.
