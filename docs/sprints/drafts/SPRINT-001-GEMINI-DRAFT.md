# Sprint 001: TurboQuant KV Cache Quantization for Qwen3.5-27B

## Overview
Implement **TurboQuant** (arXiv:2504.19874), a data-oblivious vector quantization algorithm, to compress the KV cache of Qwen3.5-27B. The goal is to achieve 2.5-bit and 3.5-bit effective precision with negligible quality loss. This involves implementing two core primitives: `TurboQuantMSE` (for L2 distance preservation) and `TurboQuantProd` (for unbiased inner product estimation), combined with an outlier channel splitting strategy.

## Use Cases
- **Long-context Inference**: Reduce VRAM footprint of the KV cache by ~4.5x-6x, enabling 128k+ context on a single A100.
- **Throughput Optimization**: Reduce HBM bandwidth bottleneck during the decoding phase by loading compressed KV tensors.
- **Quality-Neutral Compression**: Maintain needle-in-a-haystack (NIAH) performance above 99% accuracy at 3.5-bit precision.

## Architecture

### 1. Mathematical Primitives
- **Random Rotation ($\Pi$)**: A fixed $d \times d$ orthogonal matrix generated via QR decomposition of a random Gaussian matrix. Used to spread information across dimensions and induce a Beta distribution.
- **Lloyd-Max Codebooks**: Precomputed optimal scalar quantizers for the Beta distribution $f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi}\Gamma((d-1)/2)} (1-x^2)^{(d-3)/2}$.
- **QJL Projection ($S$)**: A $d \times d$ matrix with $S_{i,j} \sim \mathcal{N}(0, 1/d)$ used for 1-bit quantization of the residual vector in `TurboQuantProd`.

### 2. Implementation Components
- `TurboQuantMSE`: Implements Algorithm 1. Preserves L2 norm.
- `TurboQuantProd`: Implements Algorithm 2. Uses `TurboQuantMSE` for $b-1$ bits and a 1-bit QJL for the residual to ensure unbiasedness.
- `OutlierSplitter`: Logic to partition the 128-dim head into "Outlier" and "Non-Outlier" channels.
- `TurboKVCache`: A subclass of `transformers.Cache` that stores `uint8` packed indices and metadata (norms, QJL bits).

### 3. Memory Layout
For a single head ($d=128$) at 2.5-bit effective precision:
- **Outlier (32 channels)**: 3-bit indices (stored in packed format).
- **Non-Outlier (96 channels)**: 2-bit indices.
- **Metadata**: Scalar norm $\gamma$ (float16) and QJL signs (packed 128 bits = 16 bytes).

## Implementation Phases

### Phase 1: Offline Precomputation & Math Validation
- **Goal**: Generate codebooks and verify distortion bounds against paper's Theorem 1 & 2.
- **Tasks**:
    - Implement `scripts/generate_codebooks.py` using `scipy.optimize` to solve Lloyd-Max for Beta distribution.
    - Precompute $\Pi$ and $S$ for $d=128$.
    - Verify MSE bounds: $D_{mse} \le \frac{\sqrt{3\pi}}{2} \frac{1}{4^b}$.

### Phase 2: Core Algorithms (`turboquant/core.py`)
- **Goal**: Functional `TurboQuantMSE` and `TurboQuantProd` classes.
- **Method Signatures**:
    ```python
    class TurboQuantMSE:
        def quantize(self, x: torch.Tensor) -> torch.Tensor: ... # returns indices
        def dequantize(self, indices: torch.Tensor) -> torch.Tensor: ...

    class TurboQuantProd:
        def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ... 
        # returns (mse_indices, qjl_bits, residual_norm)
        def dequantize(self, mse_indices, qjl_bits, residual_norm) -> torch.Tensor: ...
    ```

### Phase 3: Outlier Splitting & Cache Integration
- **Goal**: Implement 2.5-bit and 3.5-bit splitting logic.
- **Logic**: 
    - 2.5-bit: 32 channels (3-bit) + 96 channels (2-bit).
    - 3.5-bit: 64 channels (4-bit) + 64 channels (3-bit).
- **Files**: `turboquant/outlier.py`, `turboquant/kv_cache.py`.

### Phase 4: Model Patching & Evaluation
- **Goal**: Integration with Qwen3.5-27B.
- **Tasks**:
    - Implement `turboquant/model.py` to replace `DynamicCache` with `TurboKVCache`.
    - Run `scripts/eval_niah.py` and `scripts/eval_longbench.py`.

## Files Summary
- `turboquant/codebook.py`: Lloyd-Max solver and codebook storage.
- `turboquant/core.py`: `TurboQuantMSE`, `TurboQuantProd`.
- `turboquant/outlier.py`: Channel splitting and bit allocation.
- `turboquant/kv_cache.py`: `TurboKVCache(transformers.Cache)` implementation.
- `turboquant/model.py`: Qwen3.5-27B monkey-patching logic.
- `tests/test_distortion.py`: Unit tests for mathematical correctness.

## Definition of Done
- [ ] `TurboQuantMSE` distortion at $b=2, 3$ is within 5% of theoretical $D_{mse}$ bounds.
- [ ] `TurboQuantProd` demonstrates zero mean bias in inner product estimation ($E[\langle x, \tilde{y} \rangle] \approx \langle x, y \rangle$).
- [ ] Qwen3.5-27B generates coherent text with `TurboKVCache` at 2.5-bit and 3.5-bit.
- [ ] 3.5-bit NIAH recall score is $\ge 0.99$ for 32k context.
- [ ] Memory usage of KV cache is reduced by $\ge 4.5x$ compared to float16.
- [ ] Inference latency overhead is $< 2ms$ per token (unoptimized Python).

## Risks
- **Overhead**: Python-based quantization/dequantize might be slow without Triton kernels (deferred to Sprint 002).
- **Qwen Integration**: Qwen3.5 might use custom attention kernels that bypass `transformers.Cache`.
- **Numerical Stability**: QR decomposition for large $\Pi$ must be stable to ensure orthogonality.

## Security
- No sensitive data or credentials are involved. All weights and codebooks are public or generated locally.

## Dependencies
- `torch >= 2.2.0`
- `transformers >= 4.38.0`
- `scipy` (for offline codebook generation)
- `numpy`

## Open Questions
- Should we use the first 32 channels as outliers, or the 32 channels with the largest average magnitude across a calibration set? (Paper suggests data-oblivious; will assume fixed first-N indices unless calibration is trivial).
- Will `torch.compile` sufficiently optimize the rotation $\Pi \cdot x$ and QJL $S \cdot r$ operations?
