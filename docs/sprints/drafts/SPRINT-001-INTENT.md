# Sprint 001 Intent: TurboQuant KV Cache Quantization for Qwen3.5-27B

## Seed

Implement TurboQuant (arXiv:2504.19874) — an online, data-oblivious vector
quantization method — as a drop-in KV cache compressor for Qwen3.5-27B
(HuggingFace transformers). Target: 2.5-bit and 3.5-bit effective KV cache
compression with quality-neutral generation.

## Context

- **Fresh repository** — only `turboquant_paper.pdf` exists; no prior code, tests, or conventions
- **No prior sprints** — this is greenfield implementation
- **Source of truth is the paper** (arXiv:2504.19874, ICLR 2026); no reference implementation exists publicly
- **Target model**: Qwen3.5-27B on HuggingFace; architecture is GQA decoder-only transformer, head_dim=128, ~64 layers, 8 KV heads
- **Implementation stack**: Python 3.11+, PyTorch 2.x, HuggingFace `transformers`, `scipy` for offline codebook computation

## Recent Sprint Context

No prior sprints. This is Sprint 001.

## Relevant Codebase Areas

Since this is greenfield, the following modules need to be created:

- `turboquant/codebook.py` — offline Lloyd-Max codebook generation for Beta distribution
- `turboquant/core.py` — TurboQuantMSE and TurboQuantProd classes (Algorithms 1 & 2)
- `turboquant/kv_cache.py` — KV cache wrapper integrating TurboQuant per attention layer
- `turboquant/outlier.py` — outlier channel detection and splitting logic
- `turboquant/model.py` — Qwen3.5-27B model patching via HuggingFace hooks
- `tests/` — unit tests for distortion bounds and integration tests for inference
- `scripts/generate_codebooks.py` — offline codebook precomputation script
- `scripts/eval_niah.py` — Needle-In-A-Haystack evaluation
- `scripts/eval_longbench.py` — LongBench-E evaluation

## Algorithm Specification (from paper)

### TurboQuant_mse (Algorithm 1)
**Setup** (global, per head dimension d and bit-width b):
1. Generate random rotation matrix Π ∈ R^(d×d) via QR decomposition of random N(0,1) matrix
2. Compute codebook centroids c_1…c_{2^b} by solving Lloyd-Max on Beta((d-1)/2, (d-1)/2) distribution
   - Codebook is symmetric: c_i ∈ [-1, 1], sorted ascending
   - For b=1: centroids ≈ ±√(2/πd); for b=2: ±0.453/√d, ±1.51/√d (scaled)
   - Solved once offline via scipy continuous k-means; stored as precomputed tables

**QUANT_mse(x):**
- y = Π · x
- idx_j = argmin_{k∈[2^b]} |y_j - c_k| for each j ∈ [d]
- output: idx (array of b-bit integers, shape [d])

**DEQUANT_mse(idx):**
- ỹ_j = c_{idx_j} for each j
- x̃ = Π^T · ỹ
- output: x̃ ∈ R^d

### TurboQuant_prod (Algorithm 2)
**Setup** (global, per head dimension d and bit-width b):
- Instantiate TurboQuant_mse with bit-width b-1
- Generate random projection matrix S ∈ R^(d×d) with S_{i,j} ~ N(0,1)

**QUANT_prod(x):**
- idx = QUANT_mse(x) using bit-width b-1
- r = x - DEQUANT_mse(idx)   {residual vector}
- qjl = sign(S · r)           {1-bit QJL on residual, qjl ∈ {-1,+1}^d}
- γ = ‖r‖₂                   {scalar norm}
- output: (idx, qjl, γ)

**DEQUANT_prod(idx, qjl, γ):**
- x̃_mse = DEQUANT_mse(idx)
- x̃_qjl = (√(π/2) / d) · γ · S^T · qjl
- output: x̃_mse + x̃_qjl

### KV Cache Application (paper §4.3)
- Apply TurboQuant per attention head, per token (online/streaming)
- **Outlier treatment for 2.5-bit**: Split 128 channels into 32 outlier + 96 non-outlier
  - Outlier channels quantized at 3-bit with one TurboQuant instance
  - Non-outlier channels quantized at 2-bit with a separate TurboQuant instance
  - Effective bit-precision: (32×3 + 96×2)/128 = 2.5 bits
- **3.5-bit**: Similar split at higher precision
- Applied during streaming generation (not just prefill)
- Store quantized repr in a custom `QuantizedKVCache` class replacing HuggingFace's `DynamicCache`

### Distortion Guarantees
- MSE: D_mse ≤ (√3π/2)·(1/4^b); b=1,2,3,4 → 0.36, 0.117, 0.03, 0.009
- Inner product: D_prod ≤ (√3π²·‖y‖²/d)·(1/4^b); b=1,2,3,4 → 1.57/d, 0.56/d, 0.18/d, 0.047/d
- Lower bounds: D_mse ≥ 1/4^b; D_prod ≥ (‖y‖²/d)·(1/4^b)

## Constraints

- Data-oblivious / online: no calibration data, no offline tuning on model activations
- The rotation matrix Π and projection matrix S must be fixed at initialization (not learned)
- Codebooks must be precomputed offline (not recomputed at inference time)
- Must support batch inference (batch_size > 1)
- Must integrate with HuggingFace `generate()` without modifying model weights
- Pure PyTorch v1 (Triton kernels optional for future sprint)
- Applied to both K and V caches (separately)

## Success Criteria

1. `TurboQuantMSE` and `TurboQuantProd` unit tests match paper's distortion bounds within 10% at d=128, b=2,3,4 (Monte Carlo over 10k random unit vectors)
2. End-to-end Qwen3.5-27B inference at 2.5-bit and 3.5-bit KV cache runs without error on a single A100
3. Perplexity on wikitext-2 at 3.5-bit ≤ full-precision baseline + 0.1 nats
4. NIAH recall score ≥ 0.99 at 3.5-bit (paper reports 0.997)
5. KV quantization overhead < 1ms per token on A100 (quantize + store + dequantize path)

## Verification Strategy

- **Distortion unit tests**: Generate random unit vectors, quantize/dequantize, measure MSE and inner product error; compare to paper's closed-form bounds per bit-width
- **Codebook validation**: Verify codebook centroids sum to ~0 (symmetry), verify quantization regions cover [-1,1]
- **Integration smoke test**: Run Qwen3.5-27B with 256-token prompt, verify output shapes and no NaN/Inf
- **Perplexity regression**: wikitext-2 sliding window perplexity (stride=512, max_length=2048) at 3.5-bit vs baseline
- **NIAH**: Needle-In-A-Haystack via gkamradt benchmark at 4k–32k context lengths
- **Throughput benchmark**: `torch.cuda.Event` timing of quantize/dequantize over 1000 calls

## Uncertainty Assessment

- **Correctness uncertainty: Medium** — The algorithms are fully specified in the paper; main risk is subtle implementation bugs in the QJL residual path and outlier channel splitting
- **Scope uncertainty: Low** — Well-defined: KV cache only, Qwen3.5-27B only, no weight quantization
- **Architecture uncertainty: Medium** — HuggingFace KV cache hooks can be fragile; need to verify Qwen3.5-27B uses standard `DynamicCache` interface and that GQA head layout is as expected

## Open Questions

1. Does Qwen3.5-27B use HuggingFace's standard `DynamicCache` or a custom cache class? (affects integration approach)
2. What is the exact Qwen3.5-27B architecture: confirmed head_dim, num_key_value_heads, num_layers?
3. For outlier channel detection: does the paper use a fixed set of outlier indices (e.g., always the top-32 by magnitude) or a per-layer calibration? (The paper implies fixed/data-oblivious — top channels by coordinate index? or magnitude on first batch?)
4. Should we support bfloat16 inputs (standard for Qwen3.5) and quantize in float32 or stay in bfloat16 throughout?
5. For KV cache memory layout: store quantized as packed uint8 tensors or keep as float16 indices?
