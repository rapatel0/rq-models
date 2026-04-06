# Sprint 001 Draft: TurboQuant KV Cache Quantization for Qwen3.5-27B

**Sprint Duration**: 3 weeks (15 days active development)
**Team Size**: 1 (solo implementation with async code review)
**Baseline**: Greenfield; only `turboquant_paper.pdf` and intent document exist
**Target Model**: Qwen3.5-27B (GQA decoder-only, 128 dims, 64 layers, 8 KV heads)
**Success Path**: 2.5-bit and 3.5-bit KV cache compression, <1% perplexity regression on wikitext-2

---

## Overview

This sprint implements TurboQuant (arXiv:2504.19874) as a drop-in KV cache compressor for Qwen3.5-27B transformers. TurboQuant is an **online, data-oblivious vector quantization method** that exploits the Beta distribution of random projections to achieve 2.5–3.5 bit effective compression with minimal quality loss.

**Why now**: Qwen3.5-27B on HuggingFace is a fresh 27B parameter model with 64 transformer layers. At 8 KV attention heads and d=128 hidden dimensions per head, the per-token KV cache footprint is ~2MB (fp16). For long-context workloads (32K tokens), this becomes 64GB—a critical bottleneck for consumer hardware and expensive cloud inference. TurboQuant targets this exact pain point with theoretically grounded distortion bounds and proven empirical quality (paper reports 0.997 NIAH recall at 3.5-bit).

**High-level approach**:
1. **Offline codebook generation**: Precompute Lloyd-Max centroids for Beta((d-1)/2, (d-1)/2) via scipy at d=128 for b∈{2,3} bits.
2. **Quantization primitives**: Implement `TurboQuantMSE` (Algorithm 1: rotation + nearest-neighbor) and `TurboQuantProd` (Algorithm 2: MSE + residual QJL) as PyTorch modules.
3. **Outlier handling**: For 2.5-bit target, split 128 channels into 32 outlier (3-bit) + 96 regular (2-bit) channels using magnitude-based heuristics during prefill.
4. **KV cache integration**: Subclass HuggingFace's `DynamicCache` as `QuantizedDynamicCache` to transparently quantize/dequantize K and V at append time.
5. **Model patching**: Hook into Qwen3.5-27B's `_prepare_decoder_attention_mask()` to install quantized cache at generation start.
6. **Validation**: Unit tests on distortion bounds, integration smoke tests, perplexity on wikitext-2, NIAH at 4k–32k tokens, and throughput benching.

---

## Use Cases

1. **Long-context inference on consumer GPU** (RTX 4090, A100): Compress KV cache 4–5× (2.5-bit) to fit 32K–64K token contexts entirely in VRAM, enabling multi-hour chat sessions on a single device.

2. **Cost-optimized cloud batch inference** (Bedrock, SageMaker): Reduce per-token KV memory footprint from ~2KB to ~500B (2.5-bit), allowing tighter batching and lower instance costs.

3. **Real-time retrieval-augmented generation**: Quantized KV enables caching external context (knowledge base snippets) with minimal overhead, making expensive retrieval calls amortizable across many queries.

4. **Privacy-preserving local LLM services**: Deploy Qwen3.5-27B locally on moderate hardware (24GB RTX 4090) with full 32K context support, avoiding cloud API dependencies.

5. **Baseline for future kernel optimization**: This pure-PyTorch sprint provides correctness reference for future Triton/CUDA kernel implementations targeting sub-millisecond quantization latency.

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ Qwen3.5-27B Model (HuggingFace transformers)                        │
│                                                                     │
│  ┌──────────────────────┐                  ┌─────────────────────┐ │
│  │ attention.self       │                  │ attention.self      │ │
│  │ (Layer 0)            │  ... (62 more)  │ (Layer 63)          │ │
│  │                      │                  │                     │ │
│  │ ┌──────────────────┐ │                  │ ┌─────────────────┐ │ │
│  │ │ DynamicCache     │ │                  │ │ DynamicCache    │ │ │
│  │ │ (HF default)     │ │                  │ │ (HF default)    │ │ │
│  │ └──────────────────┘ │                  │ └─────────────────┘ │ │
│  └──────────────────────┘                  └─────────────────────┘ │
│                                                                     │
│  [Model patching at init_]                                          │
│       │                                                             │
│       ├─> Replace cache: DynamicCache → QuantizedDynamicCache      │
│       ├─> Inject TurboQuantMSE(d=128, b=2) and (b=3) instances    │
│       ├─> Load outlier channel indices (from prefill pass)         │
│       └─> Register per-layer quantizers                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
          │
          └─> generate() with max_new_tokens, top_p, etc.
                (standard HF API, no changes to user code)
```

### Data Flow: Token Generation (Streaming)

**Prefill phase** (prompt processing):
```
Input: prompt_ids ∈ Z^{seq_len}
  │
  ├─> Embed & forward through 64 transformer layers (standard)
  │
  ├─> Layer i attention:
  │     K_new = attention.self.k_proj(hidden_state)   [shape: (batch, 1, seq_len, 128)]
  │     V_new = attention.self.v_proj(hidden_state)   [shape: (batch, 1, seq_len, 128)]
  │     │
  │     ├─> [DURING PREFILL] Detect outlier channels (only once per layer):
  │     │      outlier_idx = top-32-by-mean-magnitude(K_new.abs().mean(dim=[0,1,2]))
  │     │      (stored for reuse across all generate tokens)
  │     │
  │     ├─> append_to_cache(K_new, V_new) via QuantizedDynamicCache
  │     │     │
  │     │     ├─> Quantize outlier channels (32 dims @ 3-bit):
  │     │     │      K_out = TurboQuantMSE(K_new[..., outlier_idx], b=3)
  │     │     │      V_out = TurboQuantMSE(V_new[..., outlier_idx], b=3)
  │     │     │
  │     │     ├─> Quantize regular channels (96 dims @ 2-bit):
  │     │     │      K_reg = TurboQuantMSE(K_new[..., regular_idx], b=2)
  │     │     │      V_reg = TurboQuantMSE(V_new[..., regular_idx], b=2)
  │     │     │
  │     │     └─> Store in QuantizedKVState:
  │     │          {
  │     │            'k_outlier_idx': idx_3bit,      # uint8, shape [seq, 32*log2(8)=96 bits]
  │     │            'k_regular_idx': idx_2bit,      # uint8, shape [seq, 96*log2(4)=192 bits]
  │     │            'v_outlier_idx': idx_3bit,
  │     │            'v_regular_idx': idx_2bit,
  │     │            'outlier_indices': [0,5,12,...] # cached, reused for all tokens
  │     │          }
  │     │
  │     └─> Attention score computation:
  │          Q = attention.self.q_proj(hidden_state)
  │          K_full = get_cache(K)[...] → Dequantize + concatenate outlier+regular
  │          scores = Q @ K_full.T / sqrt(d)
  │          attn_out = softmax(scores) @ get_cache(V)[...]
  │
  └─> logits = lm_head(final_hidden_state)
Output: next_token ∈ Z
```

**Decoding phase** (per-token generation, repeated 256+ times):
```
Input: single token_id ∈ Z
  │
  ├─> Embed & forward through 64 layers
  │
  └─> Layer i attention:
       K_new, V_new = project (shapes: [batch, 1, 1, 128])
       │
       ├─> append_to_cache(K_new, V_new) [SAME PATH AS PREFILL]
       │     (outlier_indices already cached from prefill)
       │     (quantization is purely online, no calibration)
       │
       └─> Attend over all cached K,V (now partially quantized)
           (dequantize on-the-fly in attention, minimal overhead)
```

### Quantization Algorithm Details

#### **TurboQuantMSE (Algorithm 1)**

**Offline Setup** (run once at module init):
```
Input: d = 128 (head dimension), b ∈ {2, 3} (bits)
Output: (Π, codebook_c)

1. Generate rotation matrix Π ∈ R^{128×128}:
   A = random_normal(128, 128)
   Q, R = qr(A)  # PyTorch qr() or torch.linalg.qr()
   Π = Q  # Q has orthonormal columns

2. Compute codebook centroids c ∈ R^{2^b} for Beta((128-1)/2, 63.5) = Beta(63.5, 63.5):
   Use scipy.stats.rv_continuous for Beta, scipy.optimize.minimize for Lloyd-Max:
     def lloyd_max_beta_codebook(b, d=128):
       dist = scipy.stats.beta(shape_a=(d-1)/2, shape_b=(d-1)/2)
       k = 2**b  # num centroids
       # Initialize: k quantiles of Beta
       centroids = dist.ppf(np.linspace(0, 1, k+1)[1:-1])
       # Iterate (typically 20 iterations converges):
       for _ in range(20):
         # Assign points to nearest centroid (numerical integration)
         # Update centroids as conditional means
         ...
       return np.sort(centroids)  # sorted ≈ [-1, ..., +1]

   For b=2: c ≈ [-0.82, -0.27, +0.27, +0.82]
   For b=3: c ≈ [-0.91, -0.56, -0.19, ..., +0.91]

3. Store as learnable parameters (fixed, non-trainable):
   self.register_buffer('rotation_matrix', torch.tensor(Π, dtype=torch.float32))
   self.register_buffer('codebook', torch.tensor(c, dtype=torch.float32))
   self.register_buffer('gamma_norm', torch.tensor(np.sqrt(np.pi/2 * d), dtype=torch.float32))
```

**Forward Quantization** (during cache append):
```
Input: x ∈ R^{batch × seq × head × d}
Output: idx ∈ Z^{batch × seq × head × d} (indices, packed as uint8)

def quant_mse(x: Tensor) -> Tensor:
  # x shape: [*batch_dims, d]
  y = x @ self.rotation_matrix.T  # rotate

  # Nearest-neighbor search over 2^b centroids
  # Broadcasting trick: compute distance to all centroids at once
  dist = torch.abs(y.unsqueeze(-1) - self.codebook)  # [*batch_dims, d, 2^b]
  idx = torch.argmin(dist, dim=-1)  # [*batch_dims, d]

  return idx  # dtype: int64 (will pack to uint8 when storing)
```

**Forward Dequantization** (during attention):
```
Input: idx ∈ Z^{batch × seq × head × d}
Output: x̃ ∈ R^{batch × seq × head × d}

def dequant_mse(idx: Tensor) -> Tensor:
  # idx shape: [*batch_dims, d]
  y_hat = self.codebook[idx]  # [*batch_dims, d]
  x_hat = y_hat @ self.rotation_matrix  # rotate back
  return x_hat
```

**Distortion guarantee**:
```
E[||x - x̃||²] ≤ (√3·π/2) · (1/4^b)
For b=2: ≤ 0.117
For b=3: ≤ 0.03
(Empirically validated via Monte Carlo on 10k random unit vectors ∼ N(0, I))
```

---

#### **TurboQuantProd (Algorithm 2)**

**Offline Setup**:
```
Input: d = 128, b ∈ {2, 3}
Output: (mse_quantizer, S, codebook_c for b-1 bits)

1. Create TurboQuantMSE instance with bit-width (b-1):
   self.mse_quantizer = TurboQuantMSE(d=128, b=b-1)

2. Generate projection matrix S ∈ R^{128×128} for QJL:
   S = random_normal(128, 128, std=1/√128)  # normalized for stability
   self.register_buffer('projection_matrix', torch.tensor(S, dtype=torch.float32))
```

**Forward Quantization**:
```
Input: x ∈ R^{batch × seq × head × d}
Output: (idx, qjl, gamma) representing (b-1) + 1-bit quantization of residual

def quant_prod(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
  # Step 1: MSE quantization with b-1 bits
  idx = self.mse_quantizer.quant_mse(x)  # [*batch_dims, d]
  x_hat_mse = self.mse_quantizer.dequant_mse(idx)

  # Step 2: Residual vector
  r = x - x_hat_mse  # [*batch_dims, d]

  # Step 3: QJL (Quantized Johnson-Lindenstrauss) on residual
  # Project residual, then quantize sign
  proj_r = r @ self.projection_matrix.T  # [*batch_dims, d]
  qjl = torch.sign(proj_r)  # [*batch_dims, d], values in {-1, +1}

  # Step 4: Residual norm
  gamma = torch.norm(r, p=2, dim=-1, keepdim=True)  # [*batch_dims, 1]

  return (idx, qjl, gamma)
```

**Forward Dequantization**:
```
Input: (idx, qjl, gamma)
Output: x̃ ∈ R^{batch × seq × head × d}

def dequant_prod(idx: Tensor, qjl: Tensor, gamma: Tensor) -> Tensor:
  # Reconstruct MSE part
  x_hat_mse = self.mse_quantizer.dequant_mse(idx)  # [*batch_dims, d]

  # Reconstruct QJL part (residual approx)
  c = np.sqrt(np.pi / 2) / self.d  # ≈ 0.0627 for d=128
  x_hat_qjl = c * gamma * (qjl @ self.projection_matrix)  # [*batch_dims, d]

  return x_hat_mse + x_hat_qjl
```

**Distortion guarantee** (inner product):
```
E[|(x·y) - (x̃·y)|] ≤ (√3·π²·||y||²/d) · (1/4^b)
For b=2: ≤ 0.56 / d  (≤ 0.0044 at d=128)
For b=3: ≤ 0.18 / d  (≤ 0.0014 at d=128)
(Tight bounds empirically verified in paper)
```

---

### Outlier Channel Splitting (2.5-bit Target)

**Problem**: 2-bit quantization (0.117 MSE) is too lossy for all 128 channels uniformly. Solution: allocate 32 channels 3-bit precision, 96 channels 2-bit.

**Algorithm**:
```
def detect_outlier_channels(K: Tensor, num_outlier: int = 32) -> List[int]:
  """
  K: shape [batch, seq_len, num_kv_heads, d]
  Detect magnitude-based outliers once per layer during prefill.
  """
  # Mean absolute value per channel across batch, seq, and head
  channel_magnitude = K.abs().mean(dim=[0, 1, 2])  # [d]

  # Top-32 by magnitude become outliers
  outlier_indices = torch.topk(channel_magnitude, k=num_outlier).indices.tolist()

  return sorted(outlier_indices)  # e.g., [5, 12, 18, ..., 127]
```

**Storage**:
```
Outlier channels (32 dims, 3-bit):
  Each token's 32 channels: 32 * 3 bits = 96 bits = 12 bytes per token per head

Regular channels (96 dims, 2-bit):
  Each token's 96 channels: 96 * 2 bits = 192 bits = 24 bytes per token per head

Effective bit-width:
  (32*3 + 96*2) / 128 = (96 + 192) / 128 = 288 / 128 = 2.25 bits
  (plus ~1% overhead for outlier channel indices, gamma, and metadata)
  ≈ 2.5 effective bits

Compression ratio vs fp16:
  fp16: 128 dims * 2 bytes = 256 bytes per token per head
  TurboQuant 2.5-bit: (12 + 24) bytes ≈ 36 bytes per token per head
  Ratio: 256 / 36 ≈ 7.1× compression
```

---

### QuantizedDynamicCache Subclass

HuggingFace's `DynamicCache` maintains K and V as lists of tensors (one per layer). We subclass it to intercept `__setitem__` and quantize on append.

```python
class QuantizedDynamicCache(DynamicCache):
  """
  Drop-in replacement for HuggingFace DynamicCache that transparently
  quantizes K and V tensors on append and dequantizes on retrieval.
  """

  def __init__(
    self,
    config: PretrainedConfig,
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    device: str = "cuda",
  ):
    super().__init__()
    self.config = config
    self.num_layers = num_layers
    self.device = device

    # Per-layer quantizers
    self.quantizers_mse_2bit = nn.ModuleList([
      TurboQuantMSE(d=head_dim, b=2).to(device)
      for _ in range(num_layers)
    ])
    self.quantizers_mse_3bit = nn.ModuleList([
      TurboQuantMSE(d=head_dim, b=3).to(device)
      for _ in range(num_layers)
    ])

    # Outlier indices (per-layer, computed once during prefill)
    self.outlier_indices = [None] * num_layers
    self.prefill_done = False

    # Store quantized representations
    self.k_cache_quant = [
      {'outlier_idx': [], 'regular_idx': []}
      for _ in range(num_layers)
    ]
    self.v_cache_quant = [
      {'outlier_idx': [], 'regular_idx': []}
      for _ in range(num_layers)
    ]

  def __setitem__(self, layer_idx: int, cache_entry: Tuple[Tensor, Tensor]):
    """
    Intercept cache append. cache_entry = (K_new, V_new).
    K_new shape: [batch_size, 1, seq_len, head_dim] (only new token)
    V_new shape: [batch_size, 1, seq_len, head_dim]
    """
    K_new, V_new = cache_entry

    # Detect outliers on first prefill token
    if not self.prefill_done and self.outlier_indices[layer_idx] is None:
      self.outlier_indices[layer_idx] = detect_outlier_channels(
        K_new, num_outlier=32
      )

    outlier_idx_list = self.outlier_indices[layer_idx]
    regular_idx_list = [i for i in range(128) if i not in outlier_idx_list]

    # Quantize K
    K_outlier = K_new[..., outlier_idx_list]  # [batch, 1, seq, 32]
    K_regular = K_new[..., regular_idx_list]  # [batch, 1, seq, 96]

    k_out_idx = self.quantizers_mse_3bit[layer_idx].quant_mse(K_outlier)
    k_reg_idx = self.quantizers_mse_2bit[layer_idx].quant_mse(K_regular)

    self.k_cache_quant[layer_idx]['outlier_idx'].append(k_out_idx)
    self.k_cache_quant[layer_idx]['regular_idx'].append(k_reg_idx)

    # Quantize V (same split)
    V_outlier = V_new[..., outlier_idx_list]
    V_regular = V_new[..., regular_idx_list]

    v_out_idx = self.quantizers_mse_3bit[layer_idx].quant_mse(V_outlier)
    v_reg_idx = self.quantizers_mse_2bit[layer_idx].quant_mse(V_regular)

    self.v_cache_quant[layer_idx]['outlier_idx'].append(v_out_idx)
    self.v_cache_quant[layer_idx]['regular_idx'].append(v_reg_idx)

  def __getitem__(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
    """
    Retrieve and dequantize full K and V for the given layer.
    Called during attention computation.
    """
    # Dequantize K
    k_outlier_list = [
      self.quantizers_mse_3bit[layer_idx].dequant_mse(idx)
      for idx in self.k_cache_quant[layer_idx]['outlier_idx']
    ]
    k_regular_list = [
      self.quantizers_mse_2bit[layer_idx].dequant_mse(idx)
      for idx in self.k_cache_quant[layer_idx]['regular_idx']
    ]

    # Reconstruct full K by interleaving outlier and regular channels
    K_full = torch.zeros(
      batch_size, seq_len, 128, device=self.device
    )
    K_full[..., self.outlier_indices[layer_idx]] = torch.cat(k_outlier_list, dim=2)
    K_full[..., [i for i in range(128) if i not in self.outlier_indices[layer_idx]]] = torch.cat(k_regular_list, dim=2)

    # Same for V
    v_outlier_list = [
      self.quantizers_mse_3bit[layer_idx].dequant_mse(idx)
      for idx in self.v_cache_quant[layer_idx]['outlier_idx']
    ]
    v_regular_list = [
      self.quantizers_mse_2bit[layer_idx].dequant_mse(idx)
      for idx in self.v_cache_quant[layer_idx]['regular_idx']
    ]
    V_full = torch.zeros(batch_size, seq_len, 128, device=self.device)
    V_full[..., self.outlier_indices[layer_idx]] = torch.cat(v_outlier_list, dim=2)
    V_full[..., [i for i in range(128) if i not in self.outlier_indices[layer_idx]]] = torch.cat(v_regular_list, dim=2)

    return (K_full, V_full)
```

---

## Implementation

### Phase 1: Foundations (Days 1–4, ~20h estimated)

**Objective**: Core quantization primitives with unit test validation.

#### Phase 1a: Codebook Generation (Days 1–2)

- [ ] **File**: `/home/ravi/repos/turbo/turboquant/codebook.py`
  - [ ] Function: `generate_lloyd_max_codebook(d: int, b: int, num_iterations: int = 20) -> np.ndarray`
    - Input: d (dimension, e.g., 128), b (bits, e.g., 2 or 3)
    - Output: sorted codebook centroids c ∈ R^{2^b}, shape (2^b,)
    - Use `scipy.stats.beta` with Alpha = Beta = (d-1)/2
    - Implement iterative Lloyd-Max: initialize at quantiles, alternate assignment and update steps
    - Verify symmetry: c[-1] ≈ -c[0], sum(c) ≈ 0
    - Estimated effort: 3h

  - [ ] Function: `precompute_and_save_codebooks(output_dir: str) -> Dict[Tuple[int,int], np.ndarray]`
    - Generate codebooks for all (d, b) ∈ {(128, 2), (128, 3)}
    - Save to `output_dir/codebook_d128_b2.npy`, etc.
    - Return dict for in-memory access during testing
    - Estimated effort: 1h

- [ ] **File**: `/home/ravi/repos/turbo/tests/test_codebook.py`
  - [ ] Unit tests: `test_lloyd_max_symmetry()` (verify c sums to ~0 and symmetric)
  - [ ] Unit tests: `test_lloyd_max_coverage()` (verify bins partition [-1,1])
  - [ ] Unit tests: `test_codebook_values()` (hand-verify against paper Table 1 at d=128, b=2,3)
  - [ ] Estimated effort: 2h

**Deliverables**:
- `turboquant/codebook.py` with Lloyd-Max implementation
- Precomputed `.npy` files for (d=128, b=2) and (d=128, b=3)
- Test suite validating codebook properties

---

#### Phase 1b: TurboQuantMSE Module (Days 2–3)

- [ ] **File**: `/home/ravi/repos/turbo/turboquant/core.py`
  - [ ] Class: `TurboQuantMSE(nn.Module)`
    ```python
    __init__(self, d: int, b: int, codebook_dir: str = "./codebooks"):
      - Load rotation matrix Π via QR of random(d, d)
      - Load codebook c from precomputed .npy file
      - Register as non-trainable buffers
      - Estimated effort: 2h

    def quant_mse(self, x: Tensor) -> Tensor:
      - Rotate: y = x @ Π.T
      - Nearest-neighbor over codebook (broadcasting)
      - Return indices as int64 (will pack to uint8 later)
      - Estimated effort: 1h

    def dequant_mse(self, idx: Tensor) -> Tensor:
      - Look up codebook[idx]
      - Inverse rotate: x̃ = ŷ @ Π
      - Return reconstructed x̃
      - Estimated effort: 1h
    ```

  - [ ] Estimated effort for `TurboQuantMSE`: 4h

- [ ] **File**: `/home/ravi/repos/turbo/tests/test_core_mse.py`
  - [ ] `test_mse_distortion_bounds()`: Monte Carlo on 1000 random unit vectors ∼ N(0,I)/√d
    - For each vector x: compute x̃ = dequant(quant(x))
    - Measure MSE = ||x - x̃||²
    - Compare empirical E[MSE] against paper bound (√3π/2) * (1/4^b)
    - Verify empirical ≤ theoretical within 10%
    - Estimated effort: 2h

  - [ ] `test_mse_orthogonality()`: Verify Π ∈ SO(d) (Π^T Π = I)
    - Estimated effort: 0.5h

  - [ ] `test_mse_codebook_loading()`: Verify codebooks load correctly
    - Estimated effort: 0.5h

**Deliverables**:
- `turboquant/core.py` with `TurboQuantMSE` class
- Unit tests validating distortion bounds and matrix properties

---

#### Phase 1c: TurboQuantProd Module (Day 4)

- [ ] **File**: `/home/ravi/repos/turbo/turboquant/core.py` (extend)
  - [ ] Class: `TurboQuantProd(nn.Module)`
    ```python
    __init__(self, d: int, b: int, codebook_dir: str):
      - Instantiate internal TurboQuantMSE with b-1
      - Generate projection matrix S ~ N(0, 1/√d)
      - Register as buffer
      - Estimated effort: 1h

    def quant_prod(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
      - MSE quant: idx = mse.quant_mse(x)
      - Residual: r = x - mse.dequant_mse(idx)
      - QJL: qjl = sign(r @ S.T)
      - Norm: gamma = ||r||_2
      - Return (idx, qjl, gamma)
      - Estimated effort: 1h

    def dequant_prod(self, idx, qjl, gamma) -> Tensor:
      - Reconstruct: x̃_mse = mse.dequant_mse(idx)
      - Reconstruct: x̃_qjl = c * gamma * (qjl @ S)  where c = √(π/2) / d
      - Return x̃_mse + x̃_qjl
      - Estimated effort: 1h
    ```

- [ ] **File**: `/home/ravi/repos/turbo/tests/test_core_prod.py`
  - [ ] `test_prod_inner_product_distortion()`: Monte Carlo on 1000 pairs (x, y)
    - Compute x̃, ỹ via quant/dequant
    - Measure inner product error = |(x·y) - (x̃·ỹ)|
    - Compare against (√3π²·||y||²/d)·(1/4^b)
    - Estimated effort: 2h

**Deliverables**:
- `TurboQuantProd` class in `turboquant/core.py`
- Unit tests for inner product distortion bounds

---

### Phase 2: Outlier Handling & KV Cache Integration (Days 5–9, ~25h estimated)

**Objective**: Implement outlier detection and HuggingFace cache hook.

#### Phase 2a: Outlier Channel Detection (Days 5–6)

- [ ] **File**: `/home/ravi/repos/turbo/turboquant/outlier.py`
  - [ ] Function: `detect_outlier_channels(K: Tensor, num_outlier: int = 32) -> List[int]`
    - Input: K of shape (batch, seq, num_heads, d) or similar
    - Mean absolute value per channel
    - Top-32 by magnitude
    - Return sorted list of indices
    - Estimated effort: 1h

  - [ ] Function: `split_channels(x: Tensor, outlier_indices: List[int]) -> Tuple[Tensor, Tensor]`
    - Split a tensor into (outlier_subset, regular_subset) based on indices
    - Return both subsets
    - Estimated effort: 0.5h

  - [ ] Function: `merge_channels_2_5bit(x_outlier: Tensor, x_regular: Tensor, outlier_indices: List[int], d: int) -> Tensor`
    - Inverse of split: reconstruct full x with correct channel ordering
    - Estimated effort: 0.5h

- [ ] **File**: `/home/ravi/repos/turbo/tests/test_outlier.py`
  - [ ] `test_split_merge_invertibility()`: Random tensor, split, merge, verify equal
    - Estimated effort: 1h

  - [ ] `test_outlier_detection_determinism()`: Same input → same outlier set
    - Estimated effort: 0.5h

**Deliverables**:
- `turboquant/outlier.py` with detection and splitting functions
- Unit tests for invertibility

---

#### Phase 2b: QuantizedDynamicCache Subclass (Days 6–8)

- [ ] **File**: `/home/ravi/repos/turbo/turboquant/kv_cache.py`
  - [ ] Class: `QuantizedDynamicCache(DynamicCache)`
    ```python
    __init__(self, config, num_layers, num_kv_heads, head_dim, device):
      - Initialize parent DynamicCache
      - Create per-layer TurboQuantMSE(b=2) and (b=3) quantizers
      - Initialize outlier_indices list (None for each layer)
      - Initialize quantized storage dicts
      - Estimated effort: 2h

    def __setitem__(self, layer_idx: int, cache_entry):
      - First call: detect_outlier_channels() and cache indices
      - Split K and V into outlier (32-dim) and regular (96-dim)
      - Quantize with appropriate quantizer (b=3 for outlier, b=2 for regular)
      - Append to cache storage
      - Estimated effort: 2h

    def __getitem__(self, layer_idx: int) -> (K_full, V_full):
      - Dequantize all stored indices for outlier/regular separately
      - Merge channels back to full 128-dim K and V
      - Return as (K, V) tuple matching HF shape contract
      - Estimated effort: 2h

    def get_seq_length(self) -> int:
      - Return current sequence length
      - Estimated effort: 0.5h
    ```

- [ ] **File**: `/home/ravi/repos/turbo/tests/test_kv_cache.py`
  - [ ] `test_cache_append_retrieve()`: Append synthetic K,V, retrieve, check shapes and no NaN
    - Estimated effort: 2h

  - [ ] `test_cache_outlier_detection_once()`: Verify outliers detected on first append only
    - Estimated effort: 1h

  - [ ] `test_cache_quantization_loss()`: Append known vectors, retrieve, measure reconstruction MSE
    - Estimated effort: 2h

**Deliverables**:
- `turboquant/kv_cache.py` with `QuantizedDynamicCache`
- Unit tests validating cache operations

---

#### Phase 2c: Model Patching & HuggingFace Integration (Days 8–9)

- [ ] **File**: `/home/ravi/repos/turbo/turboquant/model.py`
  - [ ] Function: `patch_qwen_for_quantized_cache(model: PreTrainedModel, quantize_config: Dict) -> None`
    ```python
    Purpose: Drop-in hook to replace DynamicCache with QuantizedDynamicCache

    Steps:
    1. Extract architecture from config:
       - num_layers = config.num_hidden_layers (expect 64)
       - num_kv_heads = config.num_key_value_heads (expect 8)
       - head_dim = config.hidden_size // config.num_attention_heads (expect 128)

    2. Instantiate QuantizedDynamicCache

    3. Override model's attention layers to use this cache:
       - For each layer in model.model.layers:
         - Replace layer.self_attn._past_key_values with cache instance
         - Modify forward() to pass cache via past_key_values=cache

    4. Register forward hook or patch generate() method

    Estimated effort: 3h
    ```

  - [ ] Function: `generate_with_quantized_cache(model, input_ids, quantize_config, **generate_kwargs) -> Tensor`
    - Wrapper around model.generate() that installs cache before forward pass
    - Estimated effort: 1h

- [ ] **File**: `/home/ravi/repos/turbo/tests/test_model_integration.py`
  - [ ] `test_qwen_architecture_detection()`: Verify architecture extraction from Qwen config
    - Estimated effort: 1h

  - [ ] `test_patch_installation()`: After patching, verify cache is QuantizedDynamicCache
    - Estimated effort: 1h

**Deliverables**:
- `turboquant/model.py` with patching functions
- Integration tests for Qwen3.5-27B architecture

---

### Phase 3: End-to-End Validation & Optimization (Days 10–15, ~20h estimated)

**Objective**: Smoke tests, perplexity eval, NIAH, and throughput benchmarking.

#### Phase 3a: Integration Smoke Test (Days 10–11)

- [ ] **File**: `/home/ravi/repos/turbo/scripts/smoke_test.py`
  - [ ] Load Qwen3.5-27B (via HuggingFace model hub, may require auth token)
  - [ ] Patch with quantized cache
  - [ ] Feed 256-token prompt (e.g., wikitext-2 sample)
  - [ ] Generate 64 tokens
  - [ ] Verify output shapes, no NaN/Inf, no exceptions
  - [ ] Print example generation
  - [ ] Estimated effort: 3h (including debugging potential HF integration issues)

- [ ] **File**: `/home/ravi/repos/turbo/tests/test_integration_smoke.py`
  - [ ] Unit test wrapper around smoke_test.py
  - [ ] Estimated effort: 1h

**Deliverables**:
- `scripts/smoke_test.py` running end-to-end inference
- Integration test suite

---

#### Phase 3b: Perplexity Evaluation (Days 11–12)

- [ ] **File**: `/home/ravi/repos/turbo/scripts/eval_perplexity.py`
  - [ ] Load wikitext-2 validation set
  - [ ] Compute sliding window perplexity (stride=512, max_length=2048):
    - For each window: forward pass with prompt, measure log-likelihood of target tokens
    - Accumulate loss, compute perplexity = exp(mean_loss)
  - [ ] Run twice: (1) baseline (no quantization), (2) with 3.5-bit TurboQuant
  - [ ] Report: perplexity_baseline, perplexity_quant, delta = perplexity_quant - perplexity_baseline
  - [ ] Assert: delta ≤ 0.1 nats (per sprint success criterion)
  - [ ] Estimated effort: 4h (including wikitext-2 download, evaluation loop, troubleshooting)

**Deliverables**:
- `scripts/eval_perplexity.py` with wikitext-2 sliding window evaluation
- Results CSV/JSON with perplexity deltas

---

#### Phase 3c: NIAH Evaluation (Days 13–14)

- [ ] **File**: `/home/ravi/repos/turbo/scripts/eval_niah.py`
  - [ ] Implement Needle-In-A-Haystack benchmark (or use gkamradt public version):
    - Insert a needle (factoid, e.g., "The answer is 12345") at random position in long context (4k, 8k, 16k, 32k tokens)
    - Generate query asking for the needle
    - Check if model retrieves it correctly
    - Repeat 10 times per context length, compute recall
  - [ ] Run baseline (no quantization) and 3.5-bit quantized
  - [ ] Report recall vs context length for both
  - [ ] Assert: recall_quant ≥ 0.99 at all tested lengths (paper reports 0.997)
  - [ ] Estimated effort: 4h (implementing benchmark, running full sweep)

**Deliverables**:
- `scripts/eval_niah.py` with Needle-In-A-Haystack benchmark
- Results showing recall by context length

---

#### Phase 3d: Throughput & Memory Benchmarking (Days 14–15)

- [ ] **File**: `/home/ravi/repos/turbo/scripts/benchmark_throughput.py`
  - [ ] Measure quantize + dequantize latency:
    - Time 1000 iterations of quant_mse + dequant_mse for d=128, b=2,3
    - Report: mean latency (ms), std, per-token amortized cost
  - [ ] Measure cache append latency:
    - Time 1000 appends of (K_new, V_new) with outlier detection
    - Report: mean latency (ms), breakdown by quantize/dequantize/split/merge
  - [ ] Measure memory usage:
    - fp16 baseline: 128 dims * 2 bytes = 256 bytes per (K,V) per layer per token
    - TurboQuant 2.5-bit: ~36 bytes per (K,V) per layer per token
    - Verify achieved savings on actual model (64 layers, 8 heads)
  - [ ] Estimated effort: 3h

- [ ] **File**: `/home/ravi/repos/turbo/scripts/benchmark_memory.py`
  - [ ] Profile peak memory usage during generation:
    - Baseline (fp16 cache): torch.cuda.max_memory_allocated()
    - Quantized (2.5-bit): torch.cuda.max_memory_allocated()
    - Report reduction percentage
  - [ ] Estimated effort: 2h

**Deliverables**:
- `scripts/benchmark_throughput.py` measuring latency
- `scripts/benchmark_memory.py` measuring memory savings
- Benchmark results (JSON/CSV) with latency, memory, and per-token costs

---

### Phase 4: Documentation & Final Integration (Days 15, ~5h estimated)

- [ ] **File**: `/home/ravi/repos/turbo/README.md`
  - [ ] Project overview, usage example, API documentation
  - [ ] Estimated effort: 2h

- [ ] **File**: `/home/ravi/repos/turbo/IMPLEMENTATION_NOTES.md`
  - [ ] Design decisions, known limitations, future work
  - [ ] Estimated effort: 1h

- [ ] **File**: `/home/ravi/repos/turbo/RESULTS.md`
  - [ ] Empirical results: perplexity delta, NIAH recall, throughput, memory savings
  - [ ] Comparison to paper baseline
  - [ ] Estimated effort: 2h

**Deliverables**:
- Complete documentation suite
- Comprehensive results report

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `turboquant/codebook.py` | Create | Lloyd-Max codebook generation for Beta distribution |
| `turboquant/core.py` | Create | `TurboQuantMSE` and `TurboQuantProd` classes |
| `turboquant/outlier.py` | Create | Outlier channel detection and splitting |
| `turboquant/kv_cache.py` | Create | `QuantizedDynamicCache` HuggingFace subclass |
| `turboquant/model.py` | Create | Qwen3.5-27B model patching utilities |
| `turboquant/__init__.py` | Create | Package initialization, public API exports |
| `tests/test_codebook.py` | Create | Unit tests for Lloyd-Max implementation |
| `tests/test_core_mse.py` | Create | Distortion bound tests for TurboQuantMSE |
| `tests/test_core_prod.py` | Create | Distortion bound tests for TurboQuantProd |
| `tests/test_outlier.py` | Create | Outlier detection unit tests |
| `tests/test_kv_cache.py` | Create | QuantizedDynamicCache unit tests |
| `tests/test_model_integration.py` | Create | Qwen3.5-27B integration tests |
| `tests/test_integration_smoke.py` | Create | End-to-end smoke test suite |
| `scripts/generate_codebooks.py` | Create | Precompute and save codebooks to disk |
| `scripts/smoke_test.py` | Create | End-to-end inference smoke test |
| `scripts/eval_perplexity.py` | Create | Perplexity evaluation on wikitext-2 |
| `scripts/eval_niah.py` | Create | Needle-In-A-Haystack benchmark |
| `scripts/benchmark_throughput.py` | Create | Latency and throughput benchmarking |
| `scripts/benchmark_memory.py` | Create | Memory usage profiling |
| `README.md` | Create | Project overview and usage |
| `IMPLEMENTATION_NOTES.md` | Create | Design rationale and future work |
| `RESULTS.md` | Create | Empirical results and comparisons |
| `requirements.txt` | Create | Python dependencies (pytorch, transformers, scipy, etc.) |
| `setup.py` | Create | Package setup for pip install |
| `codebooks/codebook_d128_b2.npy` | Create | Precomputed Lloyd-Max centroids for b=2 |
| `codebooks/codebook_d128_b3.npy` | Create | Precomputed Lloyd-Max centroids for b=3 |

---

## Definition of Done

### Algorithmic Correctness (Must Pass)

- [ ] **Codebook validation**: Lloyd-Max centroids c ∈ R^{2^b} for (d=128, b=2,3) computed and verified
  - Symmetry: max(|c[i] + c[-i-1]|) < 1e-6
  - Coverage: min(c) ≤ -0.9, max(c) ≥ 0.9
  - Centroids saved to `.npy` files

- [ ] **TurboQuantMSE distortion**: Monte Carlo validation on 1000 unit vectors
  - Empirical MSE ≤ theoretical bound (√3π/2)·(1/4^b) within ±10%
  - For b=2: empirical ≤ 0.117 ± 0.012
  - For b=3: empirical ≤ 0.03 ± 0.003
  - Rotation matrix Π verified orthonormal (||Π^T Π - I||_F < 1e-6)

- [ ] **TurboQuantProd distortion**: Monte Carlo validation on 1000 pairs (x, y)
  - Empirical inner product error ≤ (√3π²·||y||²/d)·(1/4^b) within ±10%
  - For b=2: empirical ≤ 0.56/d ± 5% at d=128
  - For b=3: empirical ≤ 0.18/d ± 5% at d=128

- [ ] **Outlier splitting**: 32 channels @ 3-bit + 96 channels @ 2-bit reconstructs to 2.5 effective bits
  - (32×3 + 96×2) / 128 = 2.25 bits (verified)
  - Reconstruction MSE for split channels ≤ blended bound

### Integration Testing (Must Pass)

- [ ] **Qwen3.5-27B architecture**: Correctly detected from HuggingFace config
  - num_layers = 64, num_kv_heads = 8, head_dim = 128 confirmed
  - DynamicCache interface confirmed compatible

- [ ] **Cache integration smoke test**: End-to-end inference without errors
  - Load model (4-5 min on A100)
  - Patch with QuantizedDynamicCache
  - Feed 256-token prompt
  - Generate 64 tokens
  - Output has correct shape, no NaN/Inf
  - Completed in < 2 minutes per iteration

- [ ] **Prefill + decode path**: Both prefill and per-token decoding quantize/dequantize correctly
  - Outlier indices detected once during prefill
  - Per-token quantization appends to cache correctly
  - Attention computation retrieves and dequantizes without errors

### Quality Metrics (Must Pass)

- [ ] **Perplexity on wikitext-2**: 3.5-bit quantization ≤ baseline + 0.1 nats
  - Baseline (fp16): measured from full-precision run
  - Quantized (3.5-bit): measured from same eval set
  - Delta reported to 0.01 nats precision

- [ ] **NIAH recall**: ≥ 0.99 at context lengths 4k, 8k, 16k, 32k
  - At least 10 trials per length
  - All lengths must meet threshold
  - Expected: match or exceed paper's reported 0.997

### Performance Metrics (Must Pass)

- [ ] **Quantization latency**: < 1ms per token on A100
  - Measured: (quantize + dequantize + cache append) per token
  - Amortized over full generation (prefill + 256 decoding tokens)
  - Expected: ~0.3–0.5ms per token

- [ ] **Memory savings**: 4–7× compression on KV cache
  - Baseline (fp16): 256 bytes per (K,V) per layer per token
  - Quantized (2.5-bit): ~36 bytes per (K,V) per layer per token
  - Effective ratio: 256 / 36 ≈ 7.1×

- [ ] **Peak GPU memory reduction**: ≥ 25% for 32K context on A100
  - Baseline peak memory: measure with fp16 cache
  - Quantized peak memory: measure with 2.5-bit cache
  - Reduction: (baseline - quant) / baseline ≥ 0.25

### Code Quality (Must Pass)

- [ ] **Test coverage**: ≥ 80% of core functions (`codebook.py`, `core.py`, `outlier.py`, `kv_cache.py`)
  - Run `pytest --cov` and generate report
  - Critical paths (quantize, dequantize, outlier detection) 100% covered

- [ ] **No undefined behavior**: All functions handle edge cases
  - Empty tensors: graceful error or empty output
  - NaN/Inf inputs: logged/clamped, not propagated
  - dtype mismatches: explicit casting or error message

- [ ] **Code style**: Adherence to PEP 8 (checked via `flake8` or `pylint`)
  - Max line length: 100 characters
  - Type hints on all public functions
  - Docstrings on all classes and functions (Google-style)

### Reproducibility (Must Pass)

- [ ] **Seeded RNG**: All random initialization (Π, S, outlier detection) reproducible
  - Set `torch.manual_seed()` and `np.random.seed()` to fixed value
  - Same seed → same results across runs
  - Documented in README

- [ ] **Dependency versions locked**: `requirements.txt` pins exact versions
  - pytorch==2.x.x, transformers==4.x.x, scipy==1.x.x, numpy==1.x.x
  - Reproducible on any machine with same env

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| **Qwen3.5-27B uses non-standard KV cache** | Low | High | Query HF docs/model card; inspect `model.model.layers[0].self_attn.forward()` early (Day 5 smoke test) |
| **Lloyd-Max codebook convergence issues** | Low | High | Validate convergence with 1e-6 tolerance, use SciPy's robust solvers, log iteration history |
| **Rotation matrix Π ill-conditioned** | Very Low | Medium | QR decomposition is numerically stable; verify ||Π^T Π - I||_F on every init |
| **Outlier channel detection overfits to prefill** | Medium | Medium | Use magnitude-based static splitting (top-32 by channel magnitude), validate on train/val sets separately |
| **Memory layout bugs in __getitem__** | Medium | High | Exhaustive unit tests for split/merge invertibility, add assertions for shape/dtype in cache |
| **Perplexity regression > 0.1 nats** | Medium | High | Increase bit-width to 3.5-bit (more expensive), validate on larger wikitext-2 sample, tune outlier thresholds |
| **NIAH recall < 0.99** | Medium | High | Same as perplexity; may indicate attention distortion; increase b or revisit QJL formulation |
| **Throughput overhead > 1ms/token** | Low | Medium | Profile with `torch.profiler`, optimize with Triton (Phase 2 sprint), batch quantization ops |
| **Model patching breaks model.generate()** | Low | High | Use HF's documented `past_key_values` interface, avoid monkey-patching internals, test with multi-GPU (if available) |
| **CUDA out-of-memory on A100** | Very Low | High | A100 has 40GB; Qwen3.5-27B @ fp16 needs ~55GB for 32K tokens; quantization brings to ~10GB; unlikely to OOM |

---

## Security Considerations

1. **Model Access**: Qwen3.5-27B model download from HuggingFace requires valid auth token (if gated model).
   - Mitigation: Document token setup in README, use environment variables (HF_TOKEN).

2. **Memory Safety**: Indexing outlier channels must not exceed bounds (0–127).
   - Mitigation: Assert len(outlier_indices) ≤ 128 and all indices ∈ [0, 128).

3. **Numerical Stability**: Quantization introduces rounding; outer product computations (S^T · qjl) must not accumulate large errors.
   - Mitigation: Use float32 internally, dequantize to float32, then cast back to fp16 if needed.

4. **Reproducibility**: Fixed random seeds ensure no info leakage across runs.
   - Mitigation: Document seed usage; all RNG seeded before any quantizer init.

---

## Dependencies

### Python Packages
- `torch>=2.0.0` (PyTorch with CUDA support)
- `transformers>=4.35.0` (HuggingFace transformers, Qwen3.5 support)
- `scipy>=1.11.0` (Lloyd-Max codebook computation)
- `numpy>=1.24.0` (numerical operations)
- `pandas>=2.0.0` (optional, for results logging)
- `pytest>=7.4.0` (unit testing)
- `pytest-cov>=4.1.0` (test coverage)

### Hardware
- **Minimum**: 1 GPU with ≥ 40GB VRAM (A100, RTX 6000 Ada, or equivalent)
- **Recommended**: A100-40GB, tested throughout sprint

### Model Weights
- **Qwen3.5-27B**: Requires HuggingFace model hub access (may be gated)
  - Download via `transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-27B")`
  - Estimated size: ~55GB (fp16 weights only)
  - Download time: 10–20 min on 1Gbps internet

### Data
- **wikitext-2**: Auto-downloaded by HuggingFace datasets library
- **Needle-In-A-Haystack**: Generated synthetically in script (no external download)

---

## Open Questions

1. **Does Qwen3.5-27B use standard HuggingFace `DynamicCache`?**
   - Action: Inspect model.py on Day 5 (smoke test phase)
   - Impact: If custom cache, subclassing strategy may differ

2. **What is the exact head dimension and layer count?**
   - Expected: 128 dims, 64 layers (per intent)
   - Action: Verify from model config loaded on Day 5

3. **For outlier detection: fixed set (e.g., always top-32 by magnitude) or per-batch calibration?**
   - Paper implies: data-oblivious, so fixed set once at prefill start
   - Action: Implement fixed-set approach; if perplexity/NIAH fails, revisit per-batch variant

4. **Should we support bfloat16 inputs natively, or quantize in float32?**
   - Expected: Quantize in float32 for numerical stability, dequantize to model dtype
   - Action: Implement float32 path; benchmark overhead

5. **Should quantized cache store as packed uint8 tensors or sparse indices?**
   - Expected: Packed uint8 for memory efficiency (36 bytes vs 256 bytes per token per head)
   - Action: Implement packed uint8; verify no alignment/indexing bugs in __getitem__

6. **Can we batch quantization across multiple (K, V) pairs to amortize overhead?**
   - Expected: Yes; use vectorized operations in `__setitem__` loop
   - Action: Profile batched vs per-token quantization; optimize accordingly

7. **Do we need to support multi-GPU inference (e.g., model parallelism)?**
   - Expected: Single-GPU only for this sprint (A100-40GB sufficient)
   - Action: Document limitation; defer multi-GPU to Phase 2 sprint

---

## Summary

This sprint implements TurboQuant, a theoretically grounded 2.5–3.5 bit KV cache compressor, as a drop-in module for Qwen3.5-27B. The implementation is organized in four phases: (1) core quantization primitives with strict distortion bound validation, (2) HuggingFace integration via `QuantizedDynamicCache` subclassing, (3) empirical evaluation (perplexity, NIAH, throughput), and (4) documentation.

**Key success criteria**: distortion bounds ±10% of theory, 0.997+ NIAH recall at 3.5-bit, <0.1 nats perplexity regression, <1ms quantization overhead per token, and 4–7× KV cache compression.

The implementation is conservative (pure PyTorch, no custom kernels), focusing on correctness. Future sprints will optimize with Triton kernels, support multi-GPU, and extend to other quantization schemes.

